"""
Corebench Evaluator - Green agent with MCP tool integration using SimpleMCPClient

This agent:
1. Sets up Corebench environments
2. Initializes MCP client to connect to tool servers via JSON-RPC
3. Sends task prompts to the purple agent (the agent being tested)
4. Parses the purple agent's tool-call responses
5. Routes tool calls to MCP servers
6. Steps through the environment and collects metrics
"""
import argparse
import asyncio
import json
import logging
import subprocess
import time
from typing import Any, Optional

import gymnasium as gym
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    DataPart,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest
from agentbeats.tool_provider import ToolProvider

from tau2.data_model.simulation import RewardInfo
from tau2.environment.tool import Tool
from tau2.gym import TAU_BENCH_ENV_ID, register_gym_agent
from tau2.run import get_tasks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tau2_evaluator")

RESPOND_ACTION_NAME = "respond"

# Register tau-bench gym environments
register_gym_agent()


class SimpleMCPClient:
    """Simple MCP client using direct JSON-RPC over stdio"""
    
    def __init__(self, server_command: list[str]):
        self.server_command = server_command
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.tools = []
        self.server_info = {}
    
    async def connect(self):
        """Start the MCP server and initialize"""
        logger.info(f"Starting MCP server: {' '.join(self.server_command)}")
        
        # Start server process
        self.process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0  # Unbuffered
        )
        
        # Give server a moment to start
        await asyncio.sleep(2)
        
        # Check if process is alive
        if self.process.poll() is not None:
            raise RuntimeError("MCP server process died immediately")
        
        # Send initialize request
        init_response = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "corebench-evaluator",
                "version": "1.0.0"
            }
        })
        
        if "error" in init_response:
            raise RuntimeError(f"Initialize failed: {init_response['error']}")
        
        self.server_info = init_response["result"].get("serverInfo", {})
        
        # List tools
        tools_response = await self._send_request("tools/list", {})
        if "result" in tools_response:
            self.tools = tools_response["result"].get("tools", [])
        
        logger.info(f"MCP client connected with {len(self.tools)} tools")
        return self
    
    async def _send_request(self, method: str, params: dict, timeout: float = 30.0) -> dict:
        """Send a JSON-RPC request and get response"""
        self.request_id += 1
        
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        
        # Send request
        request_line = json.dumps(request) + "\n"
        self.process.stdin.write(request_line)
        self.process.stdin.flush()
        
        # Read response with timeout
        try:
            response_line = await asyncio.wait_for(
                asyncio.to_thread(self.process.stdout.readline),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request '{method}' timed out after {timeout}s")
        
        if not response_line:
            raise RuntimeError("Server closed connection")
        
        return json.loads(response_line)
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return its result as text"""
        response = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        if "error" in response:
            return f"Error: {response['error']}"
        
        # Extract text content from result
        result = response.get("result", {})
        content = result.get("content", [])
        
        if content and len(content) > 0:
            # Return first text content item
            first_item = content[0]
            if isinstance(first_item, dict) and "text" in first_item:
                return first_item["text"]
            return str(first_item)
        
        return "Tool executed but returned no content"
    
    async def disconnect(self):
        """Clean up server process"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        logger.info("MCP client disconnected")


def tools_to_str(tools: list[Tool]) -> str:
    """Convert tau-bench tools to JSON schema format."""
    return json.dumps([tool.openai_schema for tool in tools], indent=2)


def mcp_tools_to_str(mcp_tools: list) -> str:
    """Convert MCP tools to a readable format for the agent."""
    tool_list = []
    for tool in mcp_tools:
        # Handle both dict and object formats
        if isinstance(tool, dict):
            tool_dict = {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("inputSchema", {})
            }
        else:
            tool_dict = {
                "name": getattr(tool, "name", "unknown"),
                "description": getattr(tool, "description", ""),
                "parameters": getattr(tool, "inputSchema", {})
            }
        tool_list.append(tool_dict)
    return json.dumps(tool_list, indent=2)


def get_task_ids(domain: str, task_ids: Optional[list[str]], num_tasks: Optional[int] = None) -> list[str]:
    """Get task IDs for the domain, optionally limited to num_tasks."""
    task_set_name = domain
    task_split_name = "base"
    if task_ids is None:
        tasks = get_tasks(task_set_name=task_set_name, task_split_name=task_split_name)
    else:
        tasks = get_tasks(
            task_set_name=task_set_name,
            task_split_name=task_split_name,
            task_ids=task_ids,
        )

    result = [task.id for task in tasks]
    if num_tasks is not None:
        result = result[:num_tasks]
    return result


class CoreBenchEvaluator(GreenAgent):
    """Green agent that evaluates purple agents using tau-bench with MCP tools via SimpleMCPClient."""

    def __init__(self):
        self._required_roles = ["agent"]  # The purple agent being tested
        self._required_config_keys = ["domain"]
        self._tool_provider = ToolProvider()
        self._mcp_client: Optional[SimpleMCPClient] = None
        self._mcp_tools = []

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        return True, "ok"

    async def _init_mcp_client(self, mcp_server_command: list[str]) -> list:
        """
        Initialize MCP client connection to a tool server.
        
        Args:
            mcp_server_command: Command to start the MCP server, e.g. ["uv", "run", "mcp", "run", "mcp_server.py"]
        
        Returns:
            List of available MCP tools
        """
        try:
            logger.info(f"Initializing MCP client with command: {' '.join(mcp_server_command)}")
            
            self._mcp_client = SimpleMCPClient(mcp_server_command)
            await self._mcp_client.connect()
            
            self._mcp_tools = self._mcp_client.tools
            
            logger.info(f"MCP tools available: {[t.get('name') if isinstance(t, dict) else getattr(t, 'name', 'unknown') for t in self._mcp_tools]}")
            return self._mcp_tools
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            raise

    async def _call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Call an MCP tool and return its result.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of arguments for the tool
        
        Returns:
            Tool execution result as string
        """
        try:
            if self._mcp_client is None:
                return "Error: MCP client not initialized"
            
            logger.debug(f"Calling MCP tool: {tool_name} with args: {arguments}")
            
            result = await self._mcp_client.call_tool(tool_name, arguments)
            return result
                
        except Exception as e:
            error_msg = f"Error calling MCP tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Starting corebench evaluation: {req}")
        start_time = time.time()

        domain = req.config["domain"]
        task_ids = req.config.get("task_ids", None)
        num_tasks = req.config.get("num_tasks", None)
        max_steps = req.config.get("max_steps", 200)
        user_llm = req.config.get("user_llm", "openai/gpt-5-mini")
        user_llm_args = req.config.get("user_llm_args", {})
        
        # MCP server configuration
        use_mcp = req.config.get("use_mcp", False)
        mcp_server_command = req.config.get("mcp_server_command", ["uv", "run", "mcp", "run", "mcp_server.py"])

        # Get the purple agent URL
        agent_url = str(req.participants["agent"])

        # Initialize MCP client if enabled
        if use_mcp:
            try:
                await self._init_mcp_client(mcp_server_command)
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"MCP client initialized with {len(self._mcp_tools)} tools")
                )
            except Exception as e:
                await updater.update_status(
                    TaskState.error,
                    new_agent_text_message(f"Failed to initialize MCP: {str(e)}")
                )
                return

        # Get task IDs
        resolved_task_ids = get_task_ids(domain, task_ids, num_tasks)
        logger.info(f"Running {len(resolved_task_ids)} tasks for domain {domain}")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation of {len(resolved_task_ids)} tasks in {domain} domain")
        )

        metrics: dict[str, Any] = {"tasks": {}}

        try:
            for task_id in resolved_task_ids:
                logger.info(f"Running task {task_id}...")
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Running task {task_id}...")
                )

                try:
                    reward = await self._run_single_task(
                        agent_url=agent_url,
                        domain=domain,
                        task_id=task_id,
                        max_steps=max_steps,
                        user_llm=user_llm,
                        user_llm_args=user_llm_args,
                        use_mcp=use_mcp,
                    )
                    metrics["tasks"][task_id] = reward
                    logger.info(f"Task {task_id} completed with reward: {reward}")
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")
                    metrics["tasks"][task_id] = 0.0

            time_used = time.time() - start_time
            total_reward = sum(metrics["tasks"].values())
            num_completed = len(metrics["tasks"])
            pass_rate = (total_reward / num_completed * 100) if num_completed > 0 else 0

            result_data = {
                "domain": domain,
                "score": total_reward,
                "max_score": num_completed,
                "pass_rate": pass_rate,
                "task_rewards": metrics["tasks"],
                "time_used": time_used,
                "used_mcp": use_mcp,
            }

            # Format task results for display
            task_results_str = "\n".join(
                f"  {task_id}: {'✓' if reward == 1.0 else '✗'} ({reward})"
                for task_id, reward in metrics["tasks"].items()
            )

            summary = f"""Tau2 Benchmark Results
Domain: {domain}
Tasks: {num_completed}
Pass Rate: {pass_rate:.1f}% ({int(total_reward)}/{num_completed})
Time: {time_used:.1f}s
MCP Tools: {'Enabled' if use_mcp else 'Disabled'}

Task Results:
{task_results_str}"""

            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=summary)),
                    Part(root=DataPart(data=result_data)),
                ],
                name="Result",
            )

        finally:
            self._tool_provider.reset()
            # Clean up MCP client
            if self._mcp_client:
                try:
                    await self._mcp_client.disconnect()
                    logger.info("MCP client cleanup complete")
                except Exception as e:
                    logger.error(f"Error cleaning up MCP client: {e}")

    async def _run_single_task(
        self,
        agent_url: str,
        domain: str,
        task_id: str,
        max_steps: int,
        user_llm: str,
        user_llm_args: dict,
        use_mcp: bool = False,
    ) -> float:
        """Run a single tau-bench task and return the reward."""

        env = gym.make(
            TAU_BENCH_ENV_ID,
            domain=domain,
            task_id=task_id,
            max_steps=max_steps,
            user_llm=user_llm,
            user_llm_args=user_llm_args,
            all_messages_as_observation=False,
        )

        terminated = False
        observation, info = env.reset()

        # Build the initial task description for the purple agent
        task_description = self._build_task_prompt(info, observation, use_mcp)

        # Start a new conversation with the purple agent
        next_message = task_description
        is_first_message = True

        while not terminated:
            logger.debug(f"Sending to purple agent: {next_message[:200]}...")

            # Send message to purple agent
            response = await self._tool_provider.talk_to_agent(
                message=next_message,
                url=agent_url,
                new_conversation=is_first_message,
            )
            is_first_message = False

            logger.debug(f"Purple agent response: {response[:200]}...")

            # Parse the purple agent's action and execute tools if needed
            try:
                action, tool_result = await self._parse_and_execute_tools(response, use_mcp)
                
                # If we executed a tool via MCP, send the result back to the agent
                # and continue the loop without stepping the environment yet
                if tool_result is not None:
                    logger.info(f"Tool executed via MCP, result: {tool_result[:200]}...")
                    next_message = f"Tool execution result:\n{tool_result}\n\nPlease continue with your task."
                    continue
                
            except Exception as e:
                logger.error(f"Failed to parse agent response: {e}")
                # When parsing fails, respond with error as plain text (not a tool call)
                action = "I encountered an error processing the request."

            # Step the environment with either a JSON string (tool call) or plain text (user response)
            observation, reward, terminated, truncated, info = env.step(action)
            logger.debug(f"Environment step: reward={reward}, terminated={terminated}")

            if terminated:
                break

            next_message = observation

        # Extract final reward
        if info.get("reward_info"):
            reward_info = RewardInfo.model_validate_json(info["reward_info"])
            return reward_info.reward
        return float(reward)

    def _build_task_prompt(self, info: dict, observation: str, use_mcp: bool = False) -> str:
        """Build the initial task prompt for the purple agent."""
        
        # Get tools based on whether MCP is enabled
        if use_mcp and self._mcp_tools:
            tools_str = mcp_tools_to_str(self._mcp_tools)
            tools_section = f"""Here's a list of MCP tools you can use (you can use at most one tool at a time):
{tools_str}"""
        else:
            tools_str = tools_to_str(info["tools"])
            tools_section = f"""Here's a list of tools you can use (you can use at most one tool at a time):
{tools_str}"""
        
        return f"""
{info["policy"]}

{tools_section}

Please respond in JSON format. Wrap the JSON with <json>...</json> tags.
The JSON should contain:
- "name": the tool call function name, or "{RESPOND_ACTION_NAME}" if you want to respond directly.
- "arguments": the arguments for the tool call, or {{"content": "your message here"}} if you want to respond directly.

You should only use one tool at a time!
You cannot respond to user and use a tool at the same time!

Examples of responses:
<json>
{json.dumps({"name": "find_user_id_by_name_zip", "arguments": {"first_name": "Yusuf", "last_name": "Rossi", "zip_code": "19122"}}, indent=2)}
</json>

<json>
{json.dumps({"name": RESPOND_ACTION_NAME, "arguments": {"content": "Hello, how can I help you today?"}}, indent=2)}
</json>

Now here is the user message:
{observation}
"""

    async def _parse_and_execute_tools(self, response: str, use_mcp: bool = False) -> tuple[str, Optional[str]]:
        """
        Parse the purple agent's response and execute MCP tools if needed.
        
        Returns:
            tuple: (action_for_env, tool_result)
                - action_for_env: Action to send to tau-bench environment (or None if tool was executed)
                - tool_result: Result from MCP tool execution (or None if no tool was executed)
        """
        import re

        json_str = None

        # Try to extract JSON from <json>...</json> tags
        match = re.search(r'<json>\s*(.*?)\s*</json>', response, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try to extract JSON from markdown code blocks ```json ... ```
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # Try to extract from generic code blocks ``` ... ```
                match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                if match:
                    json_str = match.group(1)

        if json_str:
            action_dict = json.loads(json_str)
        else:
            # Try to parse the entire response as JSON
            action_dict = json.loads(response)

        tool_name = action_dict["name"]
        arguments = action_dict["arguments"]
        
        # Check if it's a respond action (direct user response)
        is_tool_call = tool_name != RESPOND_ACTION_NAME
        
        if not is_tool_call:
            # Direct response to user - pass to environment
            return arguments["content"], None
        
        # If MCP is enabled, check if this is an MCP tool
        if use_mcp and self._mcp_tools:
            mcp_tool_names = [
                t.get('name') if isinstance(t, dict) else getattr(t, 'name', None)
                for t in self._mcp_tools
            ]
            if tool_name in mcp_tool_names:
                # Execute via MCP and return result to agent (don't step environment yet)
                logger.info(f"Executing MCP tool: {tool_name}")
                result = await self._call_mcp_tool(tool_name, arguments)
                return None, result  # None means don't step env, result goes back to agent
        
        # Otherwise, it's a tau-bench tool - return as JSON for environment
        return json.dumps(action_dict), None


def tau2_evaluator_agent_card(name: str, url: str) -> AgentCard:
    """Create the agent card for the tau2 evaluator."""
    skill = AgentSkill(
        id="corebench_evaluation",
        name="CoreBench Benchmark Evaluation",
        description="Evaluates agents on core-bench tasks (airline, retail domains) with optional MCP tool integration",
        tags=["benchmark", "evaluation", "corebench", "mcp"],
        examples=[
            '{"participants": {"agent": "http://localhost:9019"}, "config": {"domain": "airline", "num_tasks": 5}}',
            '{"participants": {"agent": "http://localhost:9019"}, "config": {"domain": "airline", "num_tasks": 5, "use_mcp": true, "mcp_server_command": ["uv", "run", "mcp", "run", "mcp_server.py"]}}'
        ],
    )
    return AgentCard(
        name=name,
        description="Corebench benchmark evaluator with MCP tool support (SimpleMCPClient) - tests agents on customer service tasks",
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


async def main():
    parser = argparse.ArgumentParser(description="Run the corebench evaluator agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    args = parser.parse_args()

    agent_url = args.card_url or f"http://{args.host}:{args.port}/"

    agent = CoreBenchEvaluator()
    executor = GreenExecutor(agent)
    agent_card = tau2_evaluator_agent_card("CoreBenchEvaluator", agent_url)

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
    uvicorn_server = uvicorn.Server(uvicorn_config)
    await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())