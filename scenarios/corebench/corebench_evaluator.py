"""
Corebench Evaluator - Green agent with MCP tool integration using SimpleMCPClient
Enhanced with comprehensive logging to file

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
import shutil
import sys
import traceback
from typing import Any, Optional, Dict
from pathlib import Path
import os

import numpy as np
from scipy.stats import t
import math
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

# Import shared logging
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_logging import setup_logging

# Setup logging - will be initialized in main()
logger = logging.getLogger("evaluator")

# Suppress verbose library logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("a2a").setLevel(logging.INFO)

RESPOND_ACTION_NAME = "FINAL_ANSWER"

# Define workspace directory
WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")


class SimpleMCPClient:
    """Simple MCP client using direct JSON-RPC over stdio"""
    
    def __init__(self, server_command: list[str], cwd: Optional[str] = None):
        self.server_command = server_command
        self.cwd = cwd
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.tools = []
        self.server_info = {}
    
    async def connect(self):
        """Start the MCP server and initialize"""
        logger.info(f"Starting MCP server: {' '.join(self.server_command)}")
        logger.debug(f"MCP server working directory: {self.cwd}")
        
        # Start server process
        self.process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.cwd,
            bufsize=0  # Unbuffered
        )
        
        # Give server a moment to start
        await asyncio.sleep(2)
        
        # Check if process is alive
        if self.process.poll() is not None:
            stderr_output = self.process.stderr.read() if self.process.stderr else ""
            logger.error(f"MCP server process died immediately. Stderr: {stderr_output}")
            raise RuntimeError("MCP server process died immediately")
        
        logger.debug("MCP server process started, sending initialize request")
        
        # Send initialize request
        init_response = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "corebench-evaluator",
                "version": "1.0.0"
            }
        })
        
        logger.debug(f"Initialize response: {json.dumps(init_response, indent=2)}")
        
        if "error" in init_response:
            logger.error(f"Initialize failed: {init_response['error']}")
            raise RuntimeError(f"Initialize failed: {init_response['error']}")
        
        self.server_info = init_response["result"].get("serverInfo", {})
        logger.info(f"MCP server info: {self.server_info}")
        
        # List tools
        tools_response = await self._send_request("tools/list", {})
        logger.debug(f"Tools list response: {json.dumps(tools_response, indent=2)}")
        
        if "result" in tools_response:
            self.tools = tools_response["result"].get("tools", [])
        
        logger.info(f"MCP client connected with {len(self.tools)} tools")
        for tool in self.tools:
            tool_name = tool.get('name') if isinstance(tool, dict) else str(tool)
            logger.debug(f"  - {tool_name}")
        
        return self
    
    async def _send_request(self, method: str, params: dict, timeout: float = 90.0) -> dict:
        """Send a JSON-RPC request and get response"""
        self.request_id += 1
        
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        
        logger.debug(f"Sending MCP request {self.request_id}: {method}")
        logger.debug(f"Request payload: {json.dumps(request, indent=2)}")
        
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
            logger.error(f"Request '{method}' (id={self.request_id}) timed out after {timeout}s")
            raise TimeoutError(f"Request '{method}' timed out after {timeout}s")
        
        if not response_line:
            logger.error("MCP server closed connection")
            raise RuntimeError("Server closed connection")
        
        logger.debug(f"Received response for request {self.request_id}: {response_line[:500]}")
        
        return json.loads(response_line)
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return its result as text"""
        logger.info(f"Calling MCP tool: {tool_name}")
        
        response = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        if "error" in response:
            error_msg = f"Error: {response['error']}"
            logger.error(f"MCP tool error: {error_msg}")
            return error_msg
        
        # Extract text content from result
        result = response.get("result", {})
        content = result.get("content", [])
        
        logger.debug(f"Tool result content: {content}")
        
        if content and len(content) > 0:
            # Return first text content item
            first_item = content[0]
            if isinstance(first_item, dict) and "text" in first_item:
                result_text = first_item["text"]
                logger.info(f"Tool {tool_name} returned {len(result_text)} chars")
                return result_text
            return str(first_item)
        
        logger.warning(f"Tool {tool_name} executed but returned no content")
        return "Tool executed but returned no content"
    
    async def disconnect(self):
        """Clean up server process"""
        logger.info("Disconnecting MCP client")
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
                logger.info("MCP server terminated cleanly")
            except subprocess.TimeoutExpired:
                logger.warning("MCP server didn't terminate, killing")
                self.process.kill()
                self.process.wait()
        logger.info("MCP client disconnected")


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


def download_corebench_capsule(
    capsule_id: str,
    target_dir: str = "./scenarios/corebench/capsules"
) -> str:
    """Download and extract a CoreBench capsule from the repository."""
    import urllib.request
    import tarfile
    import time
    import socket
    from pathlib import Path
    
    logger.info(f"Downloading capsule {capsule_id}")
    
    try:
        # Create target directory if it doesn't exist
        capsules_dir = Path(target_dir)
        capsules_dir.mkdir(parents=True, exist_ok=True)
        
        capsule_dir = capsules_dir / capsule_id
        
        # Check if already downloaded
        if capsule_dir.exists():
            logger.info(f"Capsule {capsule_id} already exists at {capsule_dir}")
            return f"Capsule {capsule_id} already exists at {capsule_dir}"
        
        # Download URL and paths
        capsule_url = f"https://corebench.cs.princeton.edu/capsules/{capsule_id}.tar.gz"
        tar_path = capsules_dir / f"{capsule_id}.tar.gz"
        
        logger.debug(f"Download URL: {capsule_url}")
        logger.debug(f"Tar path: {tar_path}")
        
        # Download with retry logic
        max_retries = 5
        backoff_factor = 1
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Downloading capsule {capsule_id} (attempt {attempt}/{max_retries})...")
                socket.setdefaulttimeout(300)  # 5 minutes timeout
                urllib.request.urlretrieve(capsule_url, tar_path)
                logger.info(f"Download complete: {tar_path.stat().st_size} bytes")
                break
            except Exception as e:
                logger.warning(f"Download attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    logger.error(f"Failed to download capsule {capsule_id} after {max_retries} attempts")
                    return f"Failed to download capsule {capsule_id} after {max_retries} attempts: {str(e)}"
                
                sleep_time = backoff_factor * (2 ** (attempt - 1))
                logger.info(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
        
        # Extract the archive
        try:
            logger.info(f"Extracting capsule {capsule_id}...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=capsules_dir)
            
            # Remove tar file after successful extraction
            tar_path.unlink()
            logger.info(f"Extraction complete, tar file removed")
            
            return f"Successfully downloaded and extracted capsule {capsule_id} to {capsule_dir}"
            
        except Exception as e:
            logger.error(f"Failed to extract capsule {capsule_id}: {e}")
            if tar_path.exists():
                tar_path.unlink()
            return f"Failed to extract capsule {capsule_id}: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error downloading capsule {capsule_id}: {e}")
        logger.debug(traceback.format_exc())
        return f"Error downloading capsule {capsule_id}: {str(e)}"


def get_tasks(task_set_name):
    logger.info(f"Loading tasks for: {task_set_name}")
    
    core_test_path = os.path.join(os.path.dirname(__file__), "core_test.json")
        
    # Check if core_test.json exists
    if not os.path.exists(core_test_path):
        encrypted_file = os.path.join(os.path.dirname(__file__), "core_test.json.gpg")
        decrypt_command = f"gpg --output {core_test_path} --decrypt {encrypted_file}"
        error_msg = f"Have you decrypted core_test.json.gpg? Use the following command:\n{decrypt_command}. The password is \"reproducibility\"."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    with open(core_test_path, 'r') as f:
        dataset = json.load(f)
    
    logger.info(f"Loaded {len(dataset)} tasks")
    return dataset


def get_task_ids(domain: str, task_ids: Optional[list[str]], num_tasks: Optional[int] = None) -> list[str]:
    """Get task IDs for the domain, optionally limited to num_tasks."""
    task_set_name = domain
    task_split_name = "base"
    
    logger.info(f"Getting task IDs for domain: {domain}, num_tasks: {num_tasks}")
    
    if task_ids is None:
        tasks = get_tasks(task_set_name=task_set_name)
    else:
        tasks = get_tasks(task_set_name=task_set_name)

    result = tasks
    if num_tasks is not None:
        result = result[:num_tasks]
    
    logger.info(f"Selected {len(result)} tasks")
    return result


class CoreBenchEvaluator(GreenAgent):
    """Green agent that evaluates purple agents using corebench with MCP tools via SimpleMCPClient."""

    def __init__(self):
        self._required_roles = ["agent"]  # The purple agent being tested
        self._required_config_keys = ["domain"]
        self._tool_provider = ToolProvider()
        self._mcp_client: Optional[SimpleMCPClient] = None
        self._mcp_tools = []
        self._workspace_dir = WORKSPACE_DIR
    
    # Reset workspace directory
    def _reset_workspace(self) -> None:
        logger.info(f"Resetting workspace: {self._workspace_dir}")
        if os.path.exists(self._workspace_dir):
            shutil.rmtree(self._workspace_dir)
        os.makedirs(self._workspace_dir, exist_ok=True)
        logger.debug("Workspace reset complete")

    # Apply difficulty-specific filters to the folder where capsules are staged(copied)
    def _apply_difficulty_filters(self, domain: str) -> None:
        """Apply difficulty-specific filters to the capsule in the workspace."""
        logger.info(f"Applying difficulty filters for domain: {domain}")
        
        env_dir = os.path.join(self._workspace_dir, "environment")
        results_dir = os.path.join(env_dir, "results")

        # remove results directory for medium and hard difficulties
        if domain in ("corebench_medium", "corebench_hard"):
            if os.path.isdir(results_dir):
                logger.info(f"Removing results directory for {domain}")
                shutil.rmtree(results_dir)

        if domain == "corebench_hard":
            reproducing_path = os.path.join(env_dir, "REPRODUCING.md")
            nested_env_dir = os.path.join(env_dir, "environment")
            run_sh = os.path.join(env_dir, "code", "run.sh")
            run_plain = os.path.join(env_dir, "code", "run")

            files_to_remove = [reproducing_path, nested_env_dir, run_sh, run_plain]
            for file_path in files_to_remove:
                if os.path.isfile(file_path):
                    logger.debug(f"Removing file: {file_path}")
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    logger.debug(f"Removing directory: {file_path}")
                    shutil.rmtree(file_path)
        
        logger.debug("Difficulty filters applied")
    
    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            logger.error(f"Missing roles: {missing_roles}")
            return False, f"Missing roles: {missing_roles}"
        
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            logger.error(f"Missing config keys: {missing_config_keys}")
            return False, f"Missing config keys: {missing_config_keys}"
        
        logger.info("Request validation passed")
        return True, "ok"

    async def _init_mcp_client(self, mcp_server_command: list[str]) -> list:
        """Initialize MCP client connection to a tool server."""
        try:
            logger.info(f"Initializing MCP client with command: {' '.join(mcp_server_command)}")
            
            self._mcp_client = SimpleMCPClient(mcp_server_command, cwd=self._workspace_dir)
            await self._mcp_client.connect()
            
            self._mcp_tools = self._mcp_client.tools
            
            tool_names = [t.get('name') if isinstance(t, dict) else getattr(t, 'name', 'unknown') for t in self._mcp_tools]
            logger.info(f"MCP tools available: {tool_names}")
            return self._mcp_tools
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            logger.debug(traceback.format_exc())
            raise

    async def _call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return its result."""
        try:
            if self._mcp_client is None:
                logger.error("MCP client not initialized")
                return "Error: MCP client not initialized"
            
            logger.debug(f"Calling MCP tool: {tool_name} with args: {json.dumps(arguments, indent=2)}")
            
            result = await self._mcp_client.call_tool(tool_name, arguments)
            logger.debug(f"MCP tool result length: {len(result)} chars")
            return result
                
        except Exception as e:
            error_msg = f"Error calling MCP tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return error_msg

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"=" * 80)
        logger.info(f"STARTING COREBENCH EVALUATION")
        logger.info(f"=" * 80)
        logger.info(f"Request: {req.model_dump_json(indent=2)}")
        
        start_time = time.time()

        domain = req.config["domain"]
        task_ids = req.config.get("task_ids", None)
        num_tasks = req.config.get("num_tasks", None)
        max_steps = req.config.get("max_steps", 200)
        user_llm = req.config.get("user_llm", "openai/gpt-5-mini")
        user_llm_args = req.config.get("user_llm_args", {})
        
        logger.info(f"Domain: {domain}")
        logger.info(f"Num tasks: {num_tasks}")
        logger.info(f"Max steps: {max_steps}")
        
        # MCP server configuration
        use_mcp = req.config.get("use_mcp", False)
        mcp_server_command = req.config.get("mcp_server_command", ["uv", "run", "mcp", "run", "mcp_server.py"])
        resolved_mcp_command = []
        for part in mcp_server_command:
            if isinstance(part, str) and part.endswith(".py") and not os.path.isabs(part):
                resolved_mcp_command.append(os.path.abspath(part))
            else:
                resolved_mcp_command.append(part)

        logger.info(f"Use MCP: {use_mcp}")
        logger.info(f"MCP command: {' '.join(resolved_mcp_command)}")

        # Get the purple agent URL
        agent_url = str(req.participants["agent"])
        logger.info(f"Purple agent URL: {agent_url}")

        # Initialize MCP client if enabled
        if use_mcp:
            # Ensure workspace exists before starting MCP server
            self._reset_workspace()
            try:
                await self._init_mcp_client(resolved_mcp_command)
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"MCP client initialized with {len(self._mcp_tools)} tools")
                )
            except Exception as e:
                logger.error(f"Failed to initialize MCP: {e}")
                await updater.update_status(
                    TaskState.failed,
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
            for idx, task in enumerate(resolved_task_ids, 1):
                task_id = task["capsule_id"]
                logger.info(f"=" * 80)
                logger.info(f"TASK {idx}/{len(resolved_task_ids)}: {task_id}")
                logger.info(f"=" * 80)
                
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Running task {task_id}...")
                )
                try:
                    reward = await self._run_single_task(
                        agent_url=agent_url,
                        domain=domain,
                        task=task,
                        task_id=task_id,
                        max_steps=max_steps,
                        user_llm=user_llm,
                        user_llm_args=user_llm_args,
                        use_mcp=use_mcp,
                    )
                    eval_results = {
                        task_id: reward
                    }
                    metric_json = self._get_metrics(eval_results)
                    metrics["tasks"][task_id] = metric_json["accuracy"]
                    logger.info(f"Task {task_id} completed with reward: {metric_json}")
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")
                    logger.debug(traceback.format_exc())
                    metrics["tasks"][task_id] = 0.0

            time_used = time.time() - start_time
            total_reward = sum(metrics["tasks"].values())
            num_completed = len(metrics["tasks"])
            pass_rate = (total_reward / num_completed * 100) if num_completed > 0 else 0

            logger.info(f"=" * 80)
            logger.info(f"EVALUATION COMPLETE")
            logger.info(f"Total reward: {total_reward}/{num_completed}")
            logger.info(f"Pass rate: {pass_rate:.1f}%")
            logger.info(f"Time: {time_used:.1f}s")
            logger.info(f"=" * 80)

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

            summary = f"""Core-bench Benchmark Results
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
        task: dict,
        task_id: str,
        max_steps: int,
        user_llm: str,
        user_llm_args: dict,
        use_mcp: bool = True,
    ) -> float:
        """Run a single task and return the reward."""
        logger.info(f"Starting single task: {task_id}")

        terminated = False
        
        # Build the initial task description for the purple agent
        logger.debug("Building task prompt")
        task_description = self._build_task_prompt(task, domain, use_mcp)
        logger.debug(f"Task description length: {len(task_description)} chars")
        
        env_dir = os.path.join(self._workspace_dir, "environment")
        
        # Download capsule to workspace
        logger.info(f"Downloading capsule {task_id}")
        download_result = download_corebench_capsule(task_id, target_dir=self._workspace_dir)
        logger.info(f"Download result: {download_result}")

        # Rename to environment directory
        capsule_path = os.path.join(self._workspace_dir, task_id)
        env_dir = os.path.join(self._workspace_dir, "environment")
        
        logger.debug(f"Renaming {capsule_path} to {env_dir}")
        os.rename(capsule_path, env_dir)

        # Apply difficulty filters
        self._apply_difficulty_filters(domain)

        # Start a new conversation with the purple agent
        next_message = task_description
        is_first_message = True

        logger.info("Sending initial task to purple agent")
        logger.debug(f"Message preview: {next_message[:500]}...")

        while not terminated:
            logger.debug(f"Sending to purple agent (first_message={is_first_message})")

            # Send message to purple agent
            try:
                response = await self._tool_provider.talk_to_agent(
                    message=next_message,
                    url=agent_url,
                    new_conversation=is_first_message,
                )
                logger.info(f"Purple agent response length: {len(response)} chars")
                logger.debug(f"Response preview: {response[:500]}...")
            except Exception as e:
                logger.error(f"Failed to communicate with purple agent: {e}")
                logger.debug(traceback.format_exc())
                raise
            
            is_first_message = False

            # Parse the purple agent's action and execute tools if needed
            try:
                logger.debug("Parsing and executing tools")
                action, tool_result = await self._parse_and_execute_tools(response, use_mcp)
                logger.debug(f"Parse result - action: {str(action)[:200]}, tool_result: {str(tool_result)[:200] if tool_result else None}")
                
                # If we executed a tool via MCP, send the result back to the agent
                # and continue the loop without stepping the environment yet
                if tool_result is not None:
                    logger.info(f"Tool executed via MCP, result length: {len(tool_result)} chars")
                    next_message = f"Tool execution result:\n{tool_result}\n\nPlease continue with your task."
                    continue
                
            except Exception as e:
                logger.error(f"Failed to parse agent response: {e}")
                logger.debug(traceback.format_exc())
                # When parsing fails, respond with error as plain text
                action = "I encountered an error processing the request."

            # Step the environment with either a JSON string (tool call) or plain text (user response)
            logger.debug("Parsing final answer")
            answer = await self._parse_and_execute_tools(response)
            answer = answer[0]
            logger.info(f"FINAL ANSWER type: {type(answer)}")
            logger.info(f"FINAL ANSWER: {answer}")

            if terminated:
                break

            break

        gt_result = task["results"]
        logger.info(f"Ground truth result keys: {list(gt_result[0].keys())}")

        # Calculate total questions from ground truth
        numeric_keys = [key for key in gt_result[0].keys() if isinstance(gt_result[0][key], (int, float))]
        list_keys = [key for key in gt_result[0].keys() if isinstance(gt_result[0][key], list)]
        string_keys = [key for key in gt_result[0].keys() if isinstance(gt_result[0][key], str)]
        
        total_written_questions = len([key for key in string_keys if 'fig' not in key]) + len([key for key in numeric_keys if 'fig' not in key]) + len([key for key in list_keys if 'fig' not in key])
        total_vision_questions = len([key for key in string_keys if 'fig' in key]) + len([key for key in numeric_keys if 'fig' in key]) + len([key for key in list_keys if 'fig' in key])
        
        logger.info(f"Total questions - written: {total_written_questions}, vision: {total_vision_questions}")
        
        try:
            # Parse the agent's answer as a dictionary
            if type(answer) is str:
                logger.debug("Parsing answer as JSON string")
                reported_result = json.loads(answer)
            elif type(answer) is dict:
                logger.debug("Answer is already a dict")
                reported_result = answer
            else:
                error_msg = f"Invalid solution format for task {task_id}: {answer}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Evaluate the result using the prediction interval logic
            logger.debug("Evaluating reported result against ground truth")
            logger.debug(f"Reported result: {json.dumps(reported_result, indent=2)}")
            logger.debug(f"Ground truth: {json.dumps(gt_result, indent=2)}")
            
            evaluation = self.__eval_result_json(gt_result, reported_result)

        except Exception as e:
            logger.error(f"Failed to evaluate result: {e}")
            logger.debug(traceback.format_exc())
            evaluation = {
                "correct_written_answers": 0,
                "correct_vision_answers": 0,
                "total_written_questions": total_written_questions,
                "total_vision_questions": total_vision_questions,
                "error": str(e)
            }

        logger.info(f"Evaluation result: {json.dumps(evaluation, indent=2)}")
        
        logger.info("Cleaning up environment")
        env_dir = os.path.join(self._workspace_dir, "environment")
        if os.path.exists(env_dir):
            shutil.rmtree(env_dir)
        logger.debug("Environment cleanup complete")
        
        return evaluation

    def __eval_result_json(self, gt_result: list, reported_result: Dict):
        """Evaluates the reported result against the ground truth using prediction intervals."""
        logger.debug("Starting result evaluation")

        # Returns the number of correctly answered questions in the result json
        correct_written_answers = 0
        correct_vision_answers = 0
        question_breakdown = []

        # Separate keys into numeric, string, and list types
        numeric_keys = [key for key in gt_result[0].keys() if isinstance(gt_result[0][key], (int, float))]
        list_keys = [key for key in gt_result[0].keys() if isinstance(gt_result[0][key], list)]
        string_keys = [key for key in gt_result[0].keys() if isinstance(gt_result[0][key], str)]

        total_written_questions = len([key for key in string_keys if 'fig' not in key]) + len([key for key in numeric_keys if 'fig' not in key]) + len([key for key in list_keys if 'fig' not in key])
        total_vision_questions = len([key for key in string_keys if 'fig' in key]) + len([key for key in numeric_keys if 'fig' in key]) + len([key for key in list_keys if 'fig' in key])

        try:
            # For each value, convert to float if possible and remove the percentage sign
            for key in reported_result.keys():
                try:
                    if '%' in reported_result[key]:
                        reported_result[key] = reported_result[key].replace('%', '')
                    reported_result[key] = float(reported_result[key])
                except:
                    pass

            # Calculate mean and standard error for numeric keys
            mean_result = {key: np.mean([result[key] for result in gt_result]) for key in numeric_keys}
            std_dev_result = {key: np.std([result[key] for result in gt_result], ddof=1) for key in numeric_keys}
            sample_size = len(gt_result)

            # Calculate the 95% prediction interval bounds for numeric keys
            t_value = t.ppf(0.975, sample_size - 1)
            prediction_interval_bounds = {
                key: (
                    mean_result[key] - t_value * std_dev_result[key] * math.sqrt(1 + 1/sample_size),
                    mean_result[key] + t_value * std_dev_result[key] * math.sqrt(1 + 1/sample_size)
                )
                for key in numeric_keys
            }

            try:
                for key in reported_result.keys():
                    if key in numeric_keys:
                        value = reported_result[key]
                        lower_bound, upper_bound = prediction_interval_bounds[key]
                        if not isinstance(value, (int, float)):
                            # Wrong answer
                            question_breakdown.append({
                                "question": key,
                                "type": "numeric",
                                "is_vision": 'fig' in key,
                                "correct": False,
                                "submitted": value,
                                "prediction_interval": {
                                    "lower": round(lower_bound, 3),
                                    "upper": round(upper_bound, 3)
                                }
                            })
                            continue

                        is_correct = bool(lower_bound <= value <= upper_bound)

                        if is_correct:
                            if 'fig' in key:
                                correct_vision_answers += 1
                            else:
                                correct_written_answers += 1

                        question_breakdown.append({
                            "question": key,
                            "type": "numeric",
                            "is_vision": 'fig' in key,
                            "correct": is_correct,
                            "submitted": value,
                            "prediction_interval": {
                                "lower": round(lower_bound, 3),
                                "upper": round(upper_bound, 3)
                            }
                        })
                        
                    elif key in list_keys:
                        is_correct = reported_result[key] == gt_result[0][key]
                        if is_correct:
                            if 'fig' in key: correct_vision_answers += 1
                            else: correct_written_answers += 1
                        
                        question_breakdown.append({
                            "question": key,
                            "type": "list",
                            "is_vision": 'fig' in key,
                            "correct": is_correct,
                            "submitted": reported_result[key],
                            "expected": gt_result[0][key]
                        })
                        
                    elif key in string_keys:
                        is_correct = str(reported_result[key]).lower() == str(gt_result[0][key]).lower()
                        if is_correct:
                            if 'fig' in key: correct_vision_answers += 1
                            else: correct_written_answers += 1
                        
                        question_breakdown.append({
                            "question": key,
                            "type": "string",
                            "is_vision": 'fig' in key,
                            "correct": is_correct,
                            "submitted": reported_result[key],
                            "expected": gt_result[0][key]
                        })
            except Exception as e:
                logger.error(f"Error evaluating individual questions: {e}")
                logger.debug(traceback.format_exc())
        
        except Exception as e:
            logger.error(f"Error calculating prediction intervals: {e}")
            logger.debug(traceback.format_exc())

        return {
            "correct_written_answers": correct_written_answers, 
            "correct_vision_answers": correct_vision_answers, 
            "total_written_questions": total_written_questions, 
            "total_vision_questions": total_vision_questions,
            "question_breakdown": question_breakdown
        }

    def _get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate accuracy, successful tasks, and failed tasks IDs"""
        # Initialize counters
        correct_written_tasks = 0
        correct_vision_tasks = 0
        correct_tasks = 0
        total_written_tasks = 0
        total_vision_tasks = 0
        total_tasks = len(eval_results)
        
        # For tracking successful and failed task IDs
        successful_tasks = []
        failed_tasks = []
        
        # Calculate task-based metrics
        for task_id, result in eval_results.items():
            written_correct = result.get("correct_written_answers", 0)
            vision_correct = result.get("correct_vision_answers", 0)
            written_total = result.get("total_written_questions", 0)
            vision_total = result.get("total_vision_questions", 0)
            
            # Check if task has written questions
            if written_total > 0:
                total_written_tasks += 1
                # Check if all written questions are correct
                if written_correct == written_total:
                    correct_written_tasks += 1
            
            # Check if task has vision questions
            if vision_total > 0:
                total_vision_tasks += 1
                # Check if all vision questions are correct
                if vision_correct == vision_total:
                    correct_vision_tasks += 1
            
            # Check if all questions in the task are correct
            if (written_correct == written_total and vision_correct == vision_total and 
                (written_total > 0 or vision_total > 0)):
                correct_tasks += 1
                successful_tasks.append(task_id)
            else:
                failed_tasks.append(task_id)
        
        # Calculate accuracies
        accuracy = correct_tasks / total_tasks if total_tasks > 0 else 0
        written_accuracy = correct_written_tasks / total_written_tasks if total_written_tasks > 0 else 0
        vision_accuracy = correct_vision_tasks / total_vision_tasks if total_vision_tasks > 0 else 0
        
        return {
            "accuracy": accuracy,
            "written_accuracy": written_accuracy,
            "vision_accuracy": vision_accuracy,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks
        }

    def _build_task_prompt(self, task: dict, domain: str, use_mcp: bool = True) -> str:
        """Build the initial task prompt for the purple agent."""

        observation = str(task["results"][0].keys())
        logger.debug(f"Observation keys: {observation}")

        # Get tools based on whether MCP is enabled
        if use_mcp and self._mcp_tools:
            tools_str = mcp_tools_to_str(self._mcp_tools)
            tools_section = f"""Here's a list of MCP tools you can use (you can use at most one tool at a time):
{tools_str}"""
            logger.debug(f"Tools section created with {len(self._mcp_tools)} tools")
        
        task_prompt = task["task_prompt"]
        logger.debug(f"Task prompt: {task_prompt}")

        if domain == "corebench_easy":    
            instruction_prompt = f"Task: codeocean_easy\n\nYour goal is to answer questions about the output of scientific code. You should read through the files in the `environment/results` directory to answer the following questions: {observation}. Your submitted answer should be a Python dictionary whose keys are exactly those questions and values are the answers. **You should not actually run or execute any code.** All answers can be obtained by reading through the results directory."
        elif domain == "corebench_medium":
            instruction_prompt = f"Task: codeocean_medium\n\nYour goal is to test the computational reproducibility of the repository cloned to your current directory, which is code from a scientific paper. Specifically, you need to {task_prompt} to answer the following questions: {observation}. Your submitted answer should be a Python dictionary whose keys are exactly those questions and values are the answers. You should read the instructions on how to reproduce the capsule in REPRODUCING.md."
        elif domain == "corebench_hard":            
            instruction_prompt = f"Task: codeocean_hard\n\nYour goal is to test the computational reproducibility of the repository cloned to your current directory, which is code from a scientific paper. Specifically, you need to {task_prompt} to answer the following questions: {observation}. Your submitted answer should be a Python dictionary whose keys are exactly those questions and values are the answers. You should install all of the requirements found in the Readme file and then run the commands necessary to answer the questions."
        else:
            raise ValueError(f"Unknown domain: {domain}")

        logger.debug(f"Instruction prompt created for domain: {domain}")

        full_prompt = f"""

{instruction_prompt}
\n\n
{tools_section}

Please respond in JSON format. Wrap the JSON with <json>...</json> tags. When including multi-line strings inside JSON, you MUST escape newlines as \\n.
The JSON should contain:
- "name": the tool call function name, or "{RESPOND_ACTION_NAME}" if you are ready to provide the final answer.
- "arguments": the arguments for the tool call, or {{"content": {{"question1": "answer1", "question2": "answer2"}}}} to provide the final answer.

You should only use one tool at a time!
You cannot provide the final answer and use a tool at the same time!

Examples of responses:
<json>
{json.dumps({"name": "find_user_id_by_name_zip", "arguments": {"first_name": "Yusuf", "last_name": "Rossi", "zip_code": "19122"}}, indent=2)}
</json>

<json>
{json.dumps({"name": RESPOND_ACTION_NAME, "arguments": {"content":{
    "Report the error of the LSTM.": 0.4142,
    "From Table 3.2: Heart-attack risk for different categories, report the risk-level of category 2.": 0.348,
    "fig Report the x-axis label of the figure for the phi-2 experiment.": "Solubility"
}}}, indent=2)}
</json>

"""
        logger.debug(f"Full prompt length: {len(full_prompt)} chars")
        return full_prompt

    async def _parse_and_execute_tools(self, response: str, use_mcp: bool = False) -> tuple[str, Optional[str]]:
        """
        Parse the purple agent's response and execute MCP tools if needed.
        
        Returns:
            tuple: (action_for_env, tool_result)
                - action_for_env: Action to send to tau-bench environment (or None if tool was executed)
                - tool_result: Result from MCP tool execution (or None if no tool was executed)
        """
        import re

        logger.debug("Parsing tool call from response")
        json_str = None

        # Try to extract JSON from <json>...</json> tags
        match = re.search(r'<json>\s*(.*?)\s*</json>', response, re.DOTALL)
        if match:
            json_str = match.group(1)
            logger.debug("Found JSON in <json> tags")
        else:
            # Try to extract JSON from markdown code blocks ```json ... ```
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                json_str = match.group(1)
                logger.debug("Found JSON in ```json``` code block")
            else:
                # Try to extract from generic code blocks ``` ... ```
                match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    logger.debug("Found JSON in ``` code block")

        action_dict = None
        if json_str:
            try:
                action_dict = json.loads(json_str)
                logger.debug(f"Parsed JSON successfully")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON in tags: {e}")
                logger.debug(f"JSON string was: {json_str[:500]}")
        else:
            # Try to parse the entire response as JSON
            try:
                action_dict = json.loads(response)
                logger.debug("Parsed entire response as JSON")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse response as JSON: {e}")

        if action_dict is None:
            # Fallback: try to parse the first JSON object in the response
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    action_dict = json.loads(match.group(0))
                    logger.debug("Parsed fallback JSON object")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse fallback JSON object: {e}")

        if action_dict is None:
            logger.warning("No JSON tool call found; returning raw response")
            return response, None

        try:
            tool_name = action_dict["name"]
            arguments = action_dict["arguments"]
            logger.info(f"Parsed tool call: {tool_name}")
            logger.debug(f"Arguments: {json.dumps(arguments, indent=2)}")
        except Exception as e:
            logger.warning(f"Tool call JSON is missing required fields: {e}")
            return response, None
        
        # Check if it's a respond action (direct user response)
        is_tool_call = tool_name != RESPOND_ACTION_NAME
        
        if not is_tool_call:
            logger.info(f"Tool is FINAL_ANSWER, returning content")
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
        logger.info(f"Returning tool call as JSON for environment")
        return json.dumps(action_dict), None


def tau2_evaluator_agent_card(name: str, url: str) -> AgentCard:
    """Create the agent card for the tau2 evaluator."""
    skill = AgentSkill(
        id="corebench_evaluation",
        name="CoreBench Benchmark Evaluation",
        description="Evaluates agents on core-bench tasks with optional MCP tool integration",
        tags=["benchmark", "evaluation", "corebench", "mcp"],
        examples=[
            '{"participants": {"agent": "http://localhost:9019"}, "config": {"domain": "airline", "num_tasks": 5}}',
            '{"participants": {"agent": "http://localhost:9019"}, "config": {"domain": "airline", "num_tasks": 5, "use_mcp": true, "mcp_server_command": ["uv", "run", "mcp", "run", "mcp_server.py"]}}'
        ],
    )
    return AgentCard(
        name=name,
        description="Corebench benchmark evaluator with MCP tool support (SimpleMCPClient)",
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

    # Setup shared logging
    log_file = setup_logging("evaluator")
    
    logger.info(f"Starting CoreBench Evaluator")
    logger.info(f"Host: {args.host}, Port: {args.port}")
    logger.info(f"Log file: {log_file}")

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

    uvicorn_config = uvicorn.Config(
        server.build(), 
        host=args.host, 
        port=args.port,
        log_level="info"
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)
    
    logger.info("Server starting...")
    await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())