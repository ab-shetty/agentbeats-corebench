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
import shutil
from typing import Any, Optional, Dict
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("corebench_evaluator")

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


# def tools_to_str(tools: list[Tool]) -> str:
#     """Convert tau-bench tools to JSON schema format."""
#     return json.dumps([tool.openai_schema for tool in tools], indent=2)


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
    """
    Download and extract a CoreBench capsule from the repository.
    Use this tool to prepare a capsule's code and data for evaluation.
    
    Args:
        capsule_id: The ID of the capsule to download (e.g., "1234567")
        target_dir: Directory where capsules should be stored (default: "./capsules")
    
    Returns:
        Path to the extracted capsule directory or error message
    """
    import urllib.request
    import tarfile
    import time
    import socket
    from pathlib import Path
    
    try:
        # Create target directory if it doesn't exist
        capsules_dir = Path(target_dir)
        capsules_dir.mkdir(parents=True, exist_ok=True)
        
        capsule_dir = capsules_dir / capsule_id
        
        # Check if already downloaded
        if capsule_dir.exists():
            return f"Capsule {capsule_id} already exists at {capsule_dir}"
        
        # Download URL and paths
        capsule_url = f"https://corebench.cs.princeton.edu/capsules/{capsule_id}.tar.gz"
        tar_path = capsules_dir / f"{capsule_id}.tar.gz"
        
        # Download with retry logic
        max_retries = 5
        backoff_factor = 1
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Downloading capsule {capsule_id} (attempt {attempt}/{max_retries})...")
                socket.setdefaulttimeout(300)  # 5 minutes timeout
                urllib.request.urlretrieve(capsule_url, tar_path)
                break
            except Exception as e:
                if attempt == max_retries:
                    return f"Failed to download capsule {capsule_id} after {max_retries} attempts: {str(e)}"
                
                sleep_time = backoff_factor * (2 ** (attempt - 1))
                print(f"Download failed, retrying in {sleep_time}s...")
                time.sleep(sleep_time)
        
        # Extract the archive
        try:
            print(f"Extracting capsule {capsule_id}...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=capsules_dir)
            
            # Remove tar file after successful extraction
            tar_path.unlink()
            
            return f"Successfully downloaded and extracted capsule {capsule_id} to {capsule_dir}"
            
        except Exception as e:
            if tar_path.exists():
                tar_path.unlink()
            return f"Failed to extract capsule {capsule_id}: {str(e)}"
            
    except Exception as e:
        return f"Error downloading capsule {capsule_id}: {str(e)}"
        
def delete_corebench_capsule(
    capsule_id: str,
    capsules_dir: str = "./scenarios/corebench/capsules"
) -> str:
    """
    Delete a CoreBench capsule directory and clean up any associated files.
    Use this tool to remove a capsule that's no longer needed.
    
    Args:
        capsule_id: The ID of the capsule to delete (e.g., "1234567")
        capsules_dir: Directory where capsules are stored (default: "./capsules")
    
    Returns:
        Success message or error message
    """
    import shutil
    from pathlib import Path
    
    try:
        capsules_base = Path(capsules_dir)
        capsule_dir = capsules_base / capsule_id
        tar_path = capsules_base / f"{capsule_id}.tar.gz"
        
        # Check if capsule directory exists
        if not capsule_dir.exists():
            # Check if tar file exists
            if tar_path.exists():
                tar_path.unlink()
                return f"Deleted incomplete download: {tar_path}"
            return f"Capsule {capsule_id} not found in {capsules_dir}"
        
        # Confirm it's actually a directory
        if not capsule_dir.is_dir():
            return f"Error: {capsule_dir} exists but is not a directory"
        
        # Delete the capsule directory
        print(f"Deleting capsule {capsule_id}...")
        shutil.rmtree(capsule_dir)
        
        # Also delete tar file if it exists
        if tar_path.exists():
            tar_path.unlink()
            return f"Successfully deleted capsule {capsule_id} and its archive from {capsules_dir}"
        
        return f"Successfully deleted capsule {capsule_id} from {capsules_dir}"
        
    except PermissionError as e:
        return f"Permission denied when deleting capsule {capsule_id}: {str(e)}"
    except Exception as e:
        return f"Error deleting capsule {capsule_id}: {str(e)}"

def get_tasks(task_set_name):

    core_test_path = os.path.join(os.path.dirname(__file__), "core_test.json")
        
    # Check if core_test.json exists, if not, throw an error with instructions to decrypt
    if not os.path.exists(core_test_path):
        encrypted_file = os.path.join(os.path.dirname(__file__), "core_test.json.gpg")
        decrypt_command = f"gpg --output {core_test_path} --decrypt {encrypted_file}"
        raise FileNotFoundError(f"Have you decrypted core_test.json.gpg? Use the following command:\n{decrypt_command}. The password is \"reproducibility\".")
        
    with open(core_test_path, 'r') as f:
        dataset = json.load(f)

    # os.remove(core_test_path)

    return dataset


def get_task_ids(domain: str, task_ids: Optional[list[str]], num_tasks: Optional[int] = None) -> list[str]:
    """Get task IDs for the domain, optionally limited to num_tasks."""
    task_set_name = domain
    task_split_name = "base"
    if task_ids is None:
        tasks = get_tasks(task_set_name=task_set_name)
    else:
        tasks = get_tasks(
            task_set_name=task_set_name,
            #task_ids=task_ids,
        )

    result = tasks
    if num_tasks is not None:
        result = result[:num_tasks]
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
        if os.path.exists(self._workspace_dir):
            shutil.rmtree(self._workspace_dir)
        os.makedirs(self._workspace_dir, exist_ok=True)

    # Apply difficulty-specific filters to the folder where capsules are staged(copied)
    def _apply_difficulty_filters(self, domain: str) -> None:
        """
        
        """
        env_dir = os.path.join(self._workspace_dir, "environment")
        results_dir = os.path.join(env_dir, "results")

        if domain in ("corebench_medium", "corebench_hard"):
            if os.path.isdir(results_dir):
                shutil.rmtree(results_dir)

        if domain == "corebench_hard":
            reproducing_path = os.path.join(env_dir, "REPRODUCING.md")
            nested_env_dir = os.path.join(env_dir, "environment")
            run_sh = os.path.join(env_dir, "code", "run.sh")
            run_plain = os.path.join(env_dir, "code", "run")

            if os.path.isfile(reproducing_path):
                os.remove(reproducing_path)
            if os.path.isdir(nested_env_dir):
                shutil.rmtree(nested_env_dir)
            if os.path.isfile(run_sh):
                os.remove(run_sh)
            if os.path.isfile(run_plain):
                os.remove(run_plain)
    
    # Stage (copy) the capsule to the workspace directory
    def _stage_capsule_to_workspace(self, capsule_id: str, domain: str) -> None:
        capsule_dir = os.path.join(os.path.dirname(__file__), "capsules", capsule_id)
        if not os.path.isdir(capsule_dir):
            raise FileNotFoundError(f"Capsule directory not found: {capsule_dir}")

        self._reset_workspace()
        shutil.copytree(capsule_dir, self._workspace_dir, dirs_exist_ok=True)
        self._apply_difficulty_filters(domain) # Apply filters based on difficulty level

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
            
            self._mcp_client = SimpleMCPClient(mcp_server_command, cwd=self._workspace_dir)
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

        domain = req.config["domain"] # corebench_easy, corebench_medium or corebench_hard
        task_ids = req.config.get("task_ids", None)
        num_tasks = req.config.get("num_tasks", None)
        max_steps = req.config.get("max_steps", 200)
        user_llm = req.config.get("user_llm", "openai/gpt-5-mini")
        user_llm_args = req.config.get("user_llm_args", {})
        
        # MCP server configuration
        use_mcp = req.config.get("use_mcp", False)
        mcp_server_command = req.config.get("mcp_server_command", ["uv", "run", "mcp", "run", "mcp_server.py"])
        resolved_mcp_command = []
        for part in mcp_server_command:
            if isinstance(part, str) and part.endswith(".py") and not os.path.isabs(part):
                resolved_mcp_command.append(os.path.abspath(part))
            else:
                resolved_mcp_command.append(part)

        # Get the purple agent URL
        agent_url = str(req.participants["agent"])

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
            for task in resolved_task_ids:
                task_id = task["capsule_id"]
                logger.info(f"Running task {task_id}...")
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

        terminated = False
        
        # Build the initial task description for the purple agent
        print("Building task...")
        task_description = self._build_task_prompt(task, domain, use_mcp)
        download_corebench_capsule(task_id)
        self._stage_capsule_to_workspace(task_id, domain)

        # Once capsule is downloaded, stage(copy) it to the workspace
        # (After copying capsules to workspace, difficulty level filters are applied)
        self._stage_capsule_to_workspace(task_id, domain)

        # Start a new conversation with the purple agent
        next_message = task_description
        is_first_message = True

        while not terminated:
            logger.debug(f"Sending to purple agent: {next_message[:300]}...")

            # Send message to purple agent
            response = await self._tool_provider.talk_to_agent(
                message=next_message,
                url=agent_url,
                new_conversation=is_first_message,
            )
            is_first_message = False

            logger.debug(f"Purple agent response: {response[:300]}...")

            # Parse the purple agent's action and execute tools if needed
            try:
                action, tool_result = await self._parse_and_execute_tools(response, use_mcp)
                
                # If we executed a tool via MCP, send the result back to the agent
                # and continue the loop without stepping the environment yet
                if tool_result is not None:
                    logger.info(f"Tool executed via MCP, result: {tool_result[:300]}...")
                    next_message = f"Tool execution result:\n{tool_result}\n\nPlease continue with your task."
                    continue
                
            except Exception as e:
                logger.error(f"Failed to parse agent response: {e}")
                # When parsing fails, respond with error as plain text (not a tool call)
                action = "I encountered an error processing the request."

            # Step the environment with either a JSON string (tool call) or plain text (user response)
            answer = await self._parse_and_execute_tools(response)
            answer = answer[0]
            print("RESPONSE:", answer)

            if terminated:
                break

            break

        gt_result = task["results"]

        # Calculate total questions from ground truth (regardless of parsing success)
        numeric_keys = [key for key in gt_result[0].keys() if isinstance(gt_result[0][key], (int, float))]
        list_keys = [key for key in gt_result[0].keys() if isinstance(gt_result[0][key], list)]
        string_keys = [key for key in gt_result[0].keys() if isinstance(gt_result[0][key], str)]
        
        total_written_questions = len([key for key in string_keys if 'fig' not in key]) + len([key for key in numeric_keys if 'fig' not in key]) + len([key for key in list_keys if 'fig' not in key])
        total_vision_questions = len([key for key in string_keys if 'fig' in key]) + len([key for key in numeric_keys if 'fig' in key]) + len([key for key in list_keys if 'fig' in key])
        
        try:
            # Parse the agent's answer as a dictionary
            if type(answer) is str:
                reported_result = json.loads(answer)
            elif type(answer) is dict:
                reported_result = answer
            else:
                raise ValueError(f"Invalid solution format for task {task_id}: {answer}")
            
            # Evaluate the result using the prediction interval logic
            print ("Reported Result:", reported_result)
            print ("Ground truth Result:", gt_result)
            evaluation = self.__eval_result_json(gt_result, reported_result)

        except Exception as e:
            evaluation = {
                "correct_written_answers": 0,
                "correct_vision_answers": 0,
                "total_written_questions": total_written_questions,
                "total_vision_questions": total_vision_questions,
                "error": str(e)
            }

        print ("Evaluation:", evaluation)
        print("Deleting capsule")
        delete_corebench_capsule(task_id)
        print("capsule deletion run")
        return evaluation

    def __eval_result_json(self, gt_result: list, reported_result: Dict):
        """Evaluates the reported result against the ground truth using prediction intervals."""

        # Returns the number of correctly answered questions in the result json
        correct_written_answers = 0
        correct_vision_answers = 0
        question_breakdown = []  # Track details

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
                        lower_bound, upper_bound = prediction_interval_bounds[key]
                        is_correct = (lower_bound <= reported_result[key] <= upper_bound)
                        if is_correct:
                            if 'fig' in key: correct_vision_answers += 1
                            else: correct_written_answers += 1
                        
                        # Add to breakdown
                        question_breakdown.append({
                            "question": key,
                            "type": "numeric",
                            "is_vision": 'fig' in key,
                            "correct": is_correct,
                            "submitted": reported_result[key],
                            "prediction_interval": {
                                "lower": round(lower_bound, 3),
                                "upper": round(upper_bound, 3)
                            }
                        })
                        
                    elif key in list_keys:
                        # Direct list comparison
                        is_correct = reported_result[key] == gt_result[0][key]
                        if is_correct:
                            if 'fig' in key: correct_vision_answers += 1
                            else: correct_written_answers += 1
                        
                        # Add to breakdown
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
                        
                        # Add to breakdown
                        question_breakdown.append({
                            "question": key,
                            "type": "string",
                            "is_vision": 'fig' in key,
                            "correct": is_correct,
                            "submitted": reported_result[key],
                            "expected": gt_result[0][key]
                        })
            except Exception:
                pass
        except Exception as e:
            print(f"Error evaluating result: {e}")

        return {"correct_written_answers": correct_written_answers, 
                "correct_vision_answers": correct_vision_answers, 
                "total_written_questions": total_written_questions, 
                "total_vision_questions": total_vision_questions,
                "question_breakdown": question_breakdown}

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

        print("Observation:", observation)
        # Get tools based on whether MCP is enabled
        if use_mcp and self._mcp_tools:
            tools_str = mcp_tools_to_str(self._mcp_tools)
            tools_section = f"""Here's a list of MCP tools you can use (you can use at most one tool at a time):
{tools_str}"""
        
        # Add instruction prompt by difficulty level
        logger.info(f"observation: {observation}")
        task_prompt = task["task_prompt"]
        logger.info(f"task_prompt: {task_prompt}")

        if domain == "corebench_easy":    
            instruction_prompt = f"Task: codeocean_easy\n\nYour goal is to answer questions about the output of scientific code. You should read through the files in the `environment/results` directory to answer the following questions: {observation}. Your submitted answer should be a Python dictionary whose keys are exactly those questions and values are the answers. **You should not actually run or execute any code.** All answers can be obtained by reading through the results directory."
            logger.info(f"instruction_prompt: {instruction_prompt}")
        elif domain == "corebench_medium":
            instruction_prompt = f"Task: codeocean_medium\n\nYour goal is to test the computational reproducibility of the repository cloned to your current directory, which is code from a scientific paper. Specifically, you need to {task_prompt} to answer the following questions: {observation}. Your submitted answer should be a Python dictionary whose keys are exactly those questions and values are the answers. You should read the instructions on how to reproduce the capsule in REPRODUCING.md."
            logger.info(f"instruction_prompt: {instruction_prompt}")    
        elif domain == "corebench_hard":            
            instruction_prompt = f"Task: codeocean_hard\n\nYour goal is to test the computational reproducibility of the repository cloned to your current directory, which is code from a scientific paper. Specifically, you need to {task_prompt} to answer the following questions: {observation}. Your submitted answer should be a Python dictionary whose keys are exactly those questions and values are the answers. You should install all of the requirements found in the Readme file and then run the commands necessary to answer the questions."
            logger.info(f"instruction_prompt: {instruction_prompt}")
        else:
            raise ValueError(f"Unknown domain: {domain}")

        return f"""

{instruction_prompt}
\n\n
{tools_section}

Please respond in JSON format. Wrap the JSON with <json>...</json> tags.
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
