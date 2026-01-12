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
import re
import subprocess
import time
import shutil
import sys
import traceback
import uuid
from typing import Any, Optional, Dict
from pathlib import Path
from datetime import datetime, timezone
from pathlib import Path
import os

import numpy as np
from scipy.stats import t
import math
import gymnasium as gym
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

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

# Import metrics module
from metrics import (
    evaluate_accuracy,
    evaluate_reproducibility,
    evaluate_faithfulness,
    evaluate_task_adherence,
    compute_efficiency,
    aggregate_results,
    AccuracyMetrics,
    ReproducibilityMetrics,
    FaithfulnessMetrics,
    TaskAdherenceMetrics,
    EfficiencyMetrics,
    TaskEvaluation,
    AggregateMetrics,
    _empty_accuracy_metrics,
)

# Setup logging - will be initialized in main()
logger = logging.getLogger("evaluator")

# Suppress verbose library logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("a2a").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


RESPOND_ACTION_NAME = "FINAL_ANSWER"

# Define workspace directory
WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")

# =============================================================================
# AGENT PROMPT CONSTRAINTS
# These structured constraints help guide the purple agent's behavior and
# prevent common failure modes observed during benchmark runs.
# =============================================================================

# General behavioral constraints applicable to all tasks
COMMON_CONSTRAINTS = [
    "Role: You are a seasoned research assistant with extensive scientific computing and R&D skills.",
    "Uncertainty: If you are unsure of what to do, make your best guess using available context.",
    "Verification: Before using resources like scripts, verify their presence using 'ls' or 'find'.",
    "Images: When reproducing figures, check the results directory for image files (.png, .pdf, .jpg) before querying the vision model.",
    "Formatting: Ensure final JSON keys match the task questions EXACTLY. Values must be precise (numeric or exact text) without commentary.",
    "Precision: For numeric answers, report the exact value from results. Do not round unless the question asks for it.",
]

# Constraints specific to MCP tool usage
TOOL_CONSTRAINTS = [
    "File Reading: Use 'inspect_file_as_text' to read code, documentation, or text-based results.",
    "Vision: To analyze images (plots, charts, figures), use 'query_vision_language_model' with the image path.",
    "Search: If you need external documentation, use 'web_search' sparingly.",
    "Shell State: The 'execute_bash' tool is STATELESS. 'cd' commands do NOT persist between calls. Use absolute paths (e.g. 'ls environment/results') or chain commands (e.g. 'cd environment && python run.py').",
    "Timeouts: Long-running commands may timeout. Break complex operations into smaller steps.",
]

# Easy mode: just read results, don't execute code
EASY_CONSTRAINTS = [
    "No Execution: You should NOT run or execute any code. All answers are in the results directory.",
    "Explore First: Start by listing 'environment/results' to see available output files.",
    "Read Outputs: Use 'inspect_file_as_text' to read .txt, .csv, .json, or log files in results.",
]

# Medium mode: follow REPRODUCING.md instructions
MEDIUM_CONSTRAINTS = [
    "Instructions: Read 'environment/REPRODUCING.md' FIRST to understand how to run the capsule.",
    "Existing Results: If results already exist in 'environment/results', read them before re-running code.",
    "Docker Preferred: If REPRODUCING.md mentions Docker, use the Docker command rather than installing dependencies manually.",
    "Output Location: After running code, check 'environment/results' or the working directory for output files.",
]

# Hard mode: no instructions, must infer from Dockerfile/README
HARD_CONSTRAINTS = [
    "No Instructions: In Hard Mode, REPRODUCING.md is deleted. You must infer how to run the code.",
    "Discovery: Check 'environment/code/README.md', 'environment/Dockerfile', or 'environment/code/run.sh' for clues.",
    "Dependencies: Check for 'requirements.txt' in 'environment/' OR 'environment/code/' and install dependencies.",
    "Docker Strategy: If a Dockerfile exists, your primary goal should be to build/run that container (with --platform linux/amd64 if needed).",
    "Fallback: If Docker fails, try to manually replicate the Dockerfile's RUN commands.",
]


class AgentAction(BaseModel):
    """Validated JSON envelope for purple-agent outputs."""
    name: str
    arguments: dict[str, Any]


class ExecutionTraceWriter:
    """
    Write a per-task execution trace as JSONL for analysis and LLM-as-judge evaluation.
    
    This captures the core evaluation loop from the evaluator's perspective:
    - task_start: metadata about the task
    - agent_response: raw purple agent response (includes reasoning if present)
    - action: parsed tool call from agent
    - tool_result: result of tool execution (full content stored for faithfulness eval)
    - protocol_error: parsing/validation failures
    - final_answer: agent's submitted answer
    - evaluation: computed metrics
    
    Usage:
        with ExecutionTraceWriter(jsonl_path, run_id) as trace:
            trace.add({"type": "task_start", ...})
            ...
    
    The trace serves multiple purposes:
    1. Debugging and analysis (human-readable JSONL)
    2. LLM-as-judge input (action_trace, tool evidence)
    3. Performance benchmarking
    """

    def __init__(self, jsonl_path: Path, run_id: str):
        """
        Initialize the trace writer.
        
        Args:
            jsonl_path: Path to write JSONL trace file
            run_id: Unique identifier for this evaluation run (correlates multiple tasks)
        """
        self.jsonl_path = jsonl_path
        self.run_id = run_id
        self._events: list[dict[str, Any]] = []
        self._fp: Optional[Any] = None
        self._closed = False

    def __enter__(self) -> "ExecutionTraceWriter":
        """Context manager entry - opens the file."""
        self._fp = open(self.jsonl_path, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures file is closed."""
        self.close()
        return None  # Don't suppress exceptions

    @staticmethod
    def _truncate(value: Any, *, limit: int = 4000) -> Any:
        """Recursively truncate long strings in nested structures."""
        if isinstance(value, str) and len(value) > limit:
            return value[:limit] + f"... (truncated, original_len={len(value)})"
        if isinstance(value, list):
            return [ExecutionTraceWriter._truncate(v, limit=limit) for v in value]
        if isinstance(value, dict):
            return {k: ExecutionTraceWriter._truncate(v, limit=limit) for k, v in value.items()}
        return value

    def add(self, event: dict[str, Any], *, truncate: bool = True) -> None:
        """
        Add an event to the trace.
        
        Args:
            event: Event dict with at least a "type" key
            truncate: Whether to truncate long string values (default True)
        """
        if self._closed:
            logger.warning("Attempted to add event to closed trace")
            return
            
        # Add run_id to every event for correlation
        event = {"run_id": self.run_id, **event}
        
        if truncate:
            event = self._truncate(event)
        
        self._events.append(event)
        
        if self._fp and not self._fp.closed:
            self._fp.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self._fp.flush()

    def get_events(self, event_type: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Get trace events, optionally filtered by type.
        
        Args:
            event_type: If provided, filter to only events of this type
            
        Returns:
            List of matching events
        """
        if event_type is None:
            return self._events.copy()
        return [e for e in self._events if e.get("type") == event_type]

    def get_tool_calls(self) -> list[dict[str, Any]]:
        """Extract tool calls for faithfulness evaluation."""
        return [
            {"tool": e.get("name"), "arguments": e.get("arguments", {})}
            for e in self._events
            if e.get("type") == "action" and e.get("name") != RESPOND_ACTION_NAME
        ]

    def get_tool_results(self) -> list[dict[str, str]]:
        """Extract tool results for faithfulness evaluation."""
        return [
            {"tool": e.get("tool"), "result": e.get("full_result", e.get("summary", ""))}
            for e in self._events
            if e.get("type") == "tool_result"
        ]

    def close(self) -> None:
        """Close the trace file."""
        if not self._closed and self._fp and not self._fp.closed:
            self._fp.close()
        self._closed = True


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
    
    async def _send_request(self, method: str, params: dict, timeout: float = 510.0) -> dict:
        """Send a JSON-RPC request and get response.
        
        Args:
            method: JSON-RPC method name
            params: Request parameters
            timeout: Response timeout in seconds (default 600s = 10min to allow for Docker 
                     runs and long-running ML evaluation - note: mismatched MCP Server & Evaluator MCP client timeouts can cause hangs)
        """
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


def get_tasks(task_set_name: str) -> list[dict]:
    """
    Load the full CoreBench task list from `core_test.json`.

    Returns: one dict per task (includes `capsule_id`, `task_prompt`, `results`, etc.).
    """
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


def get_task_ids(domain: str, task_ids: Optional[list[str]], num_tasks: Optional[int] = None) -> list[dict]:
    """
    Select which tasks to run for this evaluation session.

    - loads *all* tasks via `get_tasks()`
    - if `task_ids` is provided, filters to only those specific task IDs
    - if `num_tasks` is provided, takes the first `num_tasks` entries

    Returns: the selected task dicts from `core_test.json`.
    """
    task_set_name = domain
    task_split_name = "base"
    
    logger.info(f"Getting task IDs for domain: {domain}, task_ids: {task_ids}, num_tasks: {num_tasks}")
    
    tasks = get_tasks(task_set_name=task_set_name)
    
    # Filter by specific task_ids if provided
    if task_ids is not None:
        logger.info(f"Filtering tasks to specific IDs: {task_ids}")
        selected_tasks = [task for task in tasks if task.get("capsule_id") in task_ids]
        logger.info(f"Found {len(selected_tasks)} matching tasks")
    else:
        selected_tasks = tasks
    
    # Further limit by num_tasks if provided
    if num_tasks is not None:
        selected_tasks = selected_tasks[:num_tasks]
    
    logger.info(f"Selected {len(selected_tasks)} tasks")
    return selected_tasks


class CoreBenchEvaluator(GreenAgent):
    """
    Green agent that evaluates a purple agent on CoreBench tasks with MCP tools via SimpleMCPClient.

    Responsibilities:
    - Download and stage each capsule into a local workspace directory.
    - Build the instruction + tool prompt for the purple agent.
    - If MCP is enabled, start an MCP server and execute requested tools via JSON-RPC.
    - Collect the purple agent's `FINAL_ANSWER` and score it against ground truth.
    """

    def __init__(self):
        self._required_roles = ["agent"]  # The purple agent being tested
        self._required_config_keys = ["domain"]
        self._tool_provider = ToolProvider()
        self._mcp_client: Optional[SimpleMCPClient] = None
        self._mcp_tools = []
        self._workspace_dir = WORKSPACE_DIR
    
    def _reset_workspace(self) -> None:
        """
        Reset the capsule workspace directory.

        This removes any previous staged capsule/environment and recreates the workspace
        directory so tools (especially MCP tools that run with `cwd=self._workspace_dir`)
        have a clean, predictable place to operate.
        """
        logger.info(f"Resetting workspace: {self._workspace_dir}")
        if os.path.exists(self._workspace_dir):
            shutil.rmtree(self._workspace_dir)
        os.makedirs(self._workspace_dir, exist_ok=True)
        logger.debug("Workspace reset complete")

    def _apply_difficulty_filters(self, domain: str) -> list[str]:
        """
        Apply difficulty filters to the staged capsule and return removed files/paths.
        """
        logger.info(f"Applying difficulty filters for domain: {domain}")

        removed_paths: list[str] = []
        
        env_dir = os.path.join(self._workspace_dir, "environment")
        results_dir = os.path.join(env_dir, "results")

        # remove results directory for medium and hard difficulties
        if domain in ("corebench_medium", "corebench_hard"):
            if os.path.isdir(results_dir):
                logger.info(f"Removing results directory for {domain}")
                shutil.rmtree(results_dir)
                removed_paths.append("environment/results")

        # additional instructions/scripts removals for hard difficulties
        if domain == "corebench_hard":
            reproducing_path = os.path.join(env_dir, "REPRODUCING.md")
            nested_env_dir = os.path.join(env_dir, "environment")
            run_sh = os.path.join(env_dir, "code", "run.sh")
            run_plain = os.path.join(env_dir, "code", "run")

            files_to_remove: list[tuple[str, str]] = [
                (reproducing_path, "environment/REPRODUCING.md"),
                (nested_env_dir, "environment/environment"),
                (run_sh, "environment/code/run.sh"),
                (run_plain, "environment/code/run"),
            ]
            for abs_path, rel_path in files_to_remove:
                file_path = abs_path
                if os.path.isfile(file_path):
                    logger.debug(f"Removing file: {file_path}")
                    os.remove(file_path)
                    removed_paths.append(rel_path)
                elif os.path.isdir(file_path):
                    logger.debug(f"Removing directory: {file_path}")
                    shutil.rmtree(file_path)
                    removed_paths.append(rel_path)
        
        logger.debug(f"Difficulty filters applied: {removed_paths}")
        return removed_paths
    
    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """
        Validate the incoming `EvalRequest` before starting evaluation.

        Required:
        - participants["agent"]: the purple agent endpoint URL.
        - config["domain"]: difficulty selector (e.g. corebench_medium).
        """
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


    def _hint_for_tool_result(self, tool_name: str, tool_result: str) -> Optional[str]:
        """Return a short, actionable hint to purple agent based on common tool failure patterns."""
        text = tool_result.lower()

        # Architecture / TensorFlow Specifics
        if "no matching distribution found for tensorflow" in text:
            return (
                "CRITICAL: ARM64 host cannot pip install TensorFlow. "
                "Run the capsule via Docker with x86 emulation:\n"
                "`docker run --platform linux/amd64 --rm -v $PWD/environment:/workspace <image> bash run`\n"
                "Find the Docker image in REPRODUCING.md or environment/Dockerfile."
            )

        if "exec format error" in text:
            return (
                    "Architecture Mismatch detected. You are trying to run an x86 binary on an ARM host. "
                    "Use the Docker command with '--platform linux/amd64' to run this."
            )
        
        # Python Environment Management
        if "externally-managed-environment" in text:
            return "System Python is locked. Create a venv: `python3 -m venv myenv && source myenv/bin/activate` before installing."
       
        # File System Guidance
        if "error reading file:" in text and "not a regular file" in text:
            # Check if the agent was looking for the specific missing file
            if "REPRODUCING.md" in tool_name or "REPRODUCING.md" in text:
                 return (
                    "The file 'REPRODUCING.md' appears to be missing. "
                    "If `find` cannot locate it, you are likely in a Hard Mode task where instructions were deleted. "
                    "You must instead inspect 'environment/Dockerfile' or 'environment/code/README.md' "
                    "to figure out the correct run commands."
                )
            
            # Generic file not found hint
            return (
                "The path is likely a directory or does not exist. "
                "CoreBench stages files under 'environment/'. "
                "Use `find environment -name filename` to locate it first."
            )
        
        # Shell Limitations
        if "source: not found" in text:
            return "The shell doesn't support `source`. Use `. venv/bin/activate` instead."

        if "sudo: a password is required" in text:
            return "Root access is not available. Do not use sudo."
        
        if "tools/call" in text and "timed out" in text:
            return "The command likely ran longer than the MCP timeout; try splitting into smaller steps or increasing timeouts."

        return None

    def _write_tool_output(self, *, tool_name: str, tool_result: str, index: int) -> str:
        """Persist full tool output to a workspace file and return its workspace-relative path instead of overloading model context."""
        import re

        out_dir = os.path.join(self._workspace_dir, "tool_outputs")
        os.makedirs(out_dir, exist_ok=True)

        safe_tool = re.sub(r"[^a-zA-Z0-9_-]+", "_", tool_name).strip("_") or "tool"
        filename = f"{index:04d}_{safe_tool}.txt"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(tool_result)

        # MCP tools run with cwd=self._workspace_dir, so this relative path is resolvable by the agent.
        return f"tool_outputs/{filename}"

    def _summarize_tool_result(self, tool_result: str, *, head_lines: int = 60, tail_lines: int = 60) -> str:
        """Summarize long tool output by keeping the head and tail."""
        lines = tool_result.splitlines()
        if len(lines) <= head_lines + tail_lines + 5:
            return tool_result

        head = "\n".join(lines[:head_lines])
        tail = "\n".join(lines[-tail_lines:])
        omitted = max(0, len(lines) - head_lines - tail_lines)
        return f"{head}\n\n... ({omitted} lines omitted) ...\n\n{tail}"

    def _format_tool_result_for_agent(self, *, tool_name: str, tool_result: str, index: int) -> str:
        """Standardize what we send back after tool execution."""
        max_inline_chars = 6000
        hint = self._hint_for_tool_result(tool_name, tool_result)

        saved_path: Optional[str] = None
        inline = tool_result
        # save output to file if it's too long
        if len(tool_result) > max_inline_chars:
            saved_path = self._write_tool_output(tool_name=tool_name, tool_result=tool_result, index=index)
            inline = self._summarize_tool_result(tool_result)

        msg = f"Tool execution result ({tool_name}):\n{inline}\n\n"
        if saved_path:
            msg += (
                f"(Full output saved to {saved_path}. "
                f"You can read it with inspect_file_as_text.)\n\n"
            )
        if hint:
            msg += f"Hint: {hint}\n\n"
        msg += "Please continue with your task."
        return msg


    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        """
        Run a full CoreBench evaluation session.

        Flow:
        1) Start an MCP server and cache its tool schemas.
        2) Loads tasks from `core_test.json`.
        3) For each task: download capsule, stage it, interact with the purple agent
           (tool calls + tool results), then score the final answer.
        4) Report final metrics. 
        """
        logger.info(f"=" * 80)
        logger.info(f"STARTING COREBENCH EVALUATION")
        logger.info(f"=" * 80)
        logger.info(f"Request: {req.model_dump_json(indent=2)}")
        
        start_time = time.time()
        
        # Generate unique run ID for correlating all task traces in this evaluation
        run_id = str(uuid.uuid4())[:8]
        logger.info(f"Run ID: {run_id}")

        domain = req.config["domain"]
        task_ids = req.config.get("task_ids", None)
        num_tasks = req.config.get("num_tasks", None)
        max_steps = req.config.get("max_steps", 200)
        
        # LLM-as-judge model configuration
        # By default, use the same model as the purple agent (from env vars)
        # Can be overridden in scenario.toml config
        default_judge_model = os.getenv("COREBENCH_TEXT_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
        judge_llm = req.config.get("judge_llm", default_judge_model)
        logger.info(f"LLM-as-judge model: {judge_llm}")
        
        user_llm_args = req.config.get("user_llm_args", {})
        keep_traces = req.config.get("keep_traces", False)  # Whether to keep trace files after run
        
        logger.info(f"Domain: {domain}")
        logger.info(f"Num tasks: {num_tasks}")
        logger.info(f"Max steps: {max_steps}")
        logger.info(f"Keep traces: {keep_traces}")
        
        # MCP server configuration
        use_mcp = req.config.get("use_mcp", False)
        mcp_server_command = req.config.get("mcp_server_command", ["uv", "run", "mcp", "run", "mcp_server.py"])
        resolved_mcp_command = []
        for part in mcp_server_command:
            if isinstance(part, str) and part.endswith(".py") and not os.path.isabs(part):
                # The MCP process is started with `cwd=self._workspace_dir`, so resolve any
                # local `.py` paths to absolute paths to avoid "file not found" errors.
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

        # Collect all task evaluations
        task_evaluations: list[TaskEvaluation] = []
        track_restoration = domain in ("corebench_medium", "corebench_hard")

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
                    task_evaluation = await self._run_single_task(
                        agent_url=agent_url,
                        domain=domain,
                        task=task,
                        task_id=task_id,
                        max_steps=max_steps,
                        judge_llm=judge_llm,
                        use_mcp=use_mcp,
                        run_id=run_id,
                        keep_traces=keep_traces,
                    )
                    task_evaluations.append(task_evaluation)
                    logger.info(f"Task {task_id} completed: "
                               f"accuracy={task_evaluation.accuracy.accuracy:.1%}, "
                               f"faithfulness={task_evaluation.faithfulness.score:.2f}")
                except Exception as e:
                    logger.error(f"Task {task_id} failed with exception: {e}")
                    logger.debug(traceback.format_exc())
                    # Create a failed evaluation
                    from metrics import (
                        AccuracyMetrics, FaithfulnessMetrics, 
                        TaskAdherenceMetrics, EfficiencyMetrics, _empty_accuracy_metrics
                    )
                    failed_eval = TaskEvaluation(
                        task_id=task_id,
                        domain=domain,
                        success=False,
                        accuracy=_empty_accuracy_metrics(),
                        reproducibility=None,
                        faithfulness=FaithfulnessMetrics(
                            score=0.0, is_grounded=False, suspected_guessing=True,
                            reasoning=f"Task failed: {e}", evidence_summary="", flagged_answers=[]
                        ),
                        task_adherence=TaskAdherenceMetrics(
                            score=0.0, followed_instructions=False, navigation_quality="poor",
                            reasoning=f"Task failed: {e}", strengths=[], weaknesses=["Task execution failed"]
                        ),
                        efficiency=EfficiencyMetrics(
                            steps_used=0, max_steps=max_steps, tool_calls=0,
                            time_seconds=0.0, protocol_errors=0
                        ),
                        submitted_answer={},
                        ground_truth=task.get("results", [{}]),
                    )
                    task_evaluations.append(failed_eval)

            # =====================================================================
            # AGGREGATE RESULTS
            # =====================================================================
            time_used = time.time() - start_time
            
            aggregate = aggregate_results(task_evaluations)
            
            logger.info(f"=" * 80)
            logger.info(f"⭐ EVALUATION COMPLETE ⭐")
            logger.info(f"Tasks: {aggregate.num_successful}/{aggregate.num_tasks} passed ({aggregate.pass_rate:.1%})")
            logger.info(f"Mean accuracy: {aggregate.mean_accuracy:.1%}")
            logger.info(f"Mean faithfulness: {aggregate.mean_faithfulness:.2f}")
            logger.info(f"Mean task adherence: {aggregate.mean_adherence:.2f}")
            if aggregate.mean_restoration_rate is not None:
                logger.info(f"Mean reproducibility: {aggregate.mean_restoration_rate:.1%}")
            logger.info(f"Suspected guessing: {aggregate.num_suspected_guessing}/{aggregate.num_tasks}")
            logger.info(f"Mean steps: {aggregate.mean_steps:.1f}, Mean tools: {aggregate.mean_tool_calls:.1f}")
            logger.info(f"Total time: {time_used:.1f}s")
            logger.info(f"=" * 80)

            # Build result data for leaderboard
            result_data = {
                "domain": domain,
                "num_tasks": aggregate.num_tasks,
                "num_successful": aggregate.num_successful,
                "pass_rate": aggregate.pass_rate,
                
                # Accuracy metrics
                "mean_accuracy": aggregate.mean_accuracy,
                "mean_written_accuracy": aggregate.mean_written_accuracy,
                "mean_vision_accuracy": aggregate.mean_vision_accuracy,
                
                # Reproducibility (if applicable)
                "mean_restoration_rate": aggregate.mean_restoration_rate,
                
                # Faithfulness metrics
                "mean_faithfulness": aggregate.mean_faithfulness,
                "num_suspected_guessing": aggregate.num_suspected_guessing,
                
                # Task adherence
                "mean_adherence": aggregate.mean_adherence,
                
                # Efficiency
                "mean_steps": aggregate.mean_steps,
                "mean_tool_calls": aggregate.mean_tool_calls,
                "mean_time": aggregate.mean_time,
                
                # Meta
                "total_time": time_used,
                "used_mcp": use_mcp,
                
                # Per-task breakdown
                "task_results": aggregate.task_results,
            }

            # Format task results for display
            task_results_str = "\n".join(
                f"  {tid}: {'✅' if info['success'] else '❌'} "
                f"(acc={info['accuracy']:.1%}, faith={info['faithfulness']:.2f})"
                for tid, info in aggregate.task_results.items()
            )

            restoration_line = ""
            if aggregate.mean_restoration_rate is not None:
                restoration_line = f"Reproducibility: {aggregate.mean_restoration_rate:.1%}\n"

            summary = f"""\n⭐ CoreBench Benchmark Results ⭐
Domain: {domain}
Tasks: {aggregate.num_successful}/{aggregate.num_tasks} passed ({aggregate.pass_rate:.1%})

📊 Metrics:
  Accuracy: {aggregate.mean_accuracy:.1%} (written: {aggregate.mean_written_accuracy:.1%}, vision: {aggregate.mean_vision_accuracy:.1%})
  Faithfulness: {aggregate.mean_faithfulness:.2f}
  Task Adherence: {aggregate.mean_adherence:.2f}
  {restoration_line}Suspected Guessing: {aggregate.num_suspected_guessing}/{aggregate.num_tasks}

⚡ Efficiency:
  Avg Steps: {aggregate.mean_steps:.1f}
  Avg Tool Calls: {aggregate.mean_tool_calls:.1f}
  Total Time: {time_used:.1f}s

📋 Task Results:
{task_results_str}

MCP Tools: {'Enabled' if use_mcp else 'Disabled'}"""

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
        judge_llm: str,
        use_mcp: bool = True,
        run_id: str = "",
        keep_traces: bool = False,
    ) -> TaskEvaluation:
        """
        Run a single task and return a complete TaskEvaluation.
        
        This method orchestrates the full task lifecycle:
        1. Setup workspace and download capsule
        2. Interact with purple agent (tool calls loop)
        3. Evaluate all metrics (accuracy, reproducibility, faithfulness, adherence, efficiency)
        4. Cleanup and return structured evaluation
        
        Args:
            judge_llm: LLM model name for LLM-as-judge evaluations (faithfulness, adherence)
        Args:
            keep_traces: If True, trace files are kept after run. If False (default), deleted.
        """
        import platform
        
        logger.info(f"Starting single task: {task_id}")
        task_start_time = time.time()

        terminated = False

        # Initialize execution trace for this task
        trace: Optional[ExecutionTraceWriter] = None
        trace_dir = Path(os.getenv("COREBENCH_TRACE_DIR") or os.getenv("COREBENCH_LOG_DIR") or "logs") / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        trace_jsonl = trace_dir / f"corebench_trace_{run_id}_{trace_stamp}_{task_id}.jsonl"
        
        try:
            trace = ExecutionTraceWriter(trace_jsonl, run_id=run_id or str(uuid.uuid4())[:8])
            trace.__enter__()  # Open the file
            trace.add({
                "type": "task_start",
                "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "task_id": task_id,
                "domain": domain,
                "host": f"{platform.system()} {platform.release()} ({platform.machine()})",
                "questions": list(task["results"][0].keys()),
                "agent_url": agent_url,
            })
        except Exception as e:
            logger.warning(f"Failed to initialize execution trace writer: {e}")
            trace = None
        
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

        # Apply difficulty filters (and remember what we removed for restoration metrics).
        removed_by_difficulty_filters = self._apply_difficulty_filters(domain)

        # Start a new conversation with the purple agent
        next_message = task_description
        is_first_message = True

        logger.info("Sending initial task to purple agent")
        logger.debug(f"Message preview: {next_message[:500]}...")

        answer: Any = None
        protocol_errors = 0
        max_protocol_errors = 5
        steps_used = 0
        tool_exec_index = 0
        # NOTE: tool_calls and tool_results are now extracted from trace via
        # trace.get_tool_calls() and trace.get_tool_results() for LLM-as-judge

        while not terminated and steps_used < max_steps: # Prevent infinite loops
            logger.debug(f"Sending to purple agent (first_message={is_first_message})")

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
            steps_used += 1
            
            # Capture raw agent response for debugging/analysis
            # This may include reasoning if the agent model exposes it
            if trace:
                trace.add({
                    "type": "agent_response",
                    "turn": steps_used,
                    "raw_response": response,
                })

            try:
                action, tool_result = await self._parse_and_execute_tools(response, use_mcp)
                protocol_errors = 0
                if trace:
                    trace.add({
                        "type": "action",
                        "turn": steps_used,
                        "name": action.name,
                        "arguments": action.arguments,
                    })
            except Exception as e:
                protocol_errors += 1
                logger.warning(f"Invalid agent response (protocol_errors={protocol_errors}/{max_protocol_errors}): {e}")
                logger.debug(traceback.format_exc())
                if trace:
                    trace.add({
                        "type": "protocol_error",
                            "turn": steps_used,
                            "error": str(e),
                        }
                    )

                if protocol_errors >= max_protocol_errors:
                    logger.error("Too many protocol errors; failing task with empty answer")
                    answer = {}
                    break

                next_message = (
                    "Your last message was NOT a valid JSON action for this benchmark.\n\n"
                    "Reply with EXACTLY ONE JSON object wrapped in <json>...</json>.\n"
                    "No extra text.\n\n"
                    f"Error: {str(e)}\n"
                )
                continue

            if tool_result is not None:
                logger.info(f"Tool executed via MCP, result length: {len(tool_result)} chars")
                tool_exec_index += 1
                
                # Capture tool result in trace (includes full_result for faithfulness eval)
                if trace:
                    exit_code: Optional[int] = None
                    if action.name == "execute_bash":
                        match = re.search(r"Exit Code:\s*(\d+)", tool_result)
                        if match:
                            exit_code = int(match.group(1))
                    trace.add({
                        "type": "tool_result",
                        "turn": steps_used,
                        "tool": action.name,
                        "exit_code": exit_code,
                        "hint": self._hint_for_tool_result(action.name, tool_result),
                        "summary": self._summarize_tool_result(tool_result, head_lines=25, tail_lines=25),
                        "full_result": tool_result,  # Stored for faithfulness evaluation
                    })
                next_message = self._format_tool_result_for_agent(tool_name=action.name, tool_result=tool_result, index=tool_exec_index)
                continue

            if action.name == RESPOND_ACTION_NAME:
                answer = action.arguments.get("content")
                if answer is None:
                    next_message = (
                        f"Invalid {RESPOND_ACTION_NAME}: missing `arguments.content`.\n\n"
                        f"Reply again with <json>{{\"name\": \"{RESPOND_ACTION_NAME}\", \"arguments\": {{\"content\": {{...}}}}}}</json>.\n"
                    )
                    continue
                logger.info(f"FINAL ANSWER type: {type(answer)}")
                logger.info(f"FINAL ANSWER: {answer}")
                if trace:
                    trace.add(
                        {
                            "type": "final_answer",
                            "turn": steps_used,
                            "content": answer,
                        }
                    )
                break

            # Agent requested a tool that isn't available (or MCP is disabled).
            mcp_tool_names = []
            if self._mcp_tools:
                mcp_tool_names = [
                    t.get("name") if isinstance(t, dict) else getattr(t, "name", "unknown")
                    for t in self._mcp_tools
                ]
            next_message = (
                f"Unsupported tool call: {action.name}\n\n"
                f"Use one of the available MCP tools: {mcp_tool_names}\n"
                f"Or respond with {RESPOND_ACTION_NAME}.\n"
            )

        if steps_used >= max_steps and answer is None:
            logger.error(f"Reached max_steps={max_steps} without a FINAL_ANSWER; failing with empty answer")
            answer = {}

        # =====================================================================
        # EVALUATION PHASE - Using Clean Metrics Module
        # =====================================================================
        task_end_time = time.time()
        task_time_seconds = task_end_time - task_start_time
        
        gt_result = task["results"]
        task_prompt = task.get("task_prompt", "")
        logger.info(f"Ground truth result keys: {list(gt_result[0].keys())}")
        
        # Parse the agent's submitted answer
        reported_result: dict[str, Any] = {}
        try:
            if isinstance(answer, str):
                logger.debug("Parsing answer as JSON string")
                reported_result = json.loads(answer)
            elif isinstance(answer, dict):
                logger.debug("Answer is already a dict")
                reported_result = answer
            else:
                logger.warning(f"Invalid answer type: {type(answer)}, using empty dict")
                reported_result = {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse answer as JSON: {e}")
            reported_result = {}
        
        logger.debug(f"Reported result: {json.dumps(reported_result, indent=2)}")
        
        # Extract data from trace for LLM-as-judge evaluations
        action_trace = trace.get_events("action") if trace else []
        tool_calls = trace.get_tool_calls() if trace else []
        tool_results_for_faithfulness = trace.get_tool_results() if trace else []
        
        # 1. ACCURACY: Evaluate answer correctness
        logger.info("Computing accuracy metrics...")
        accuracy_metrics = evaluate_accuracy(gt_result, reported_result)
        logger.info(f"Accuracy: {accuracy_metrics.correct_answers}/{accuracy_metrics.total_questions} "
                   f"({accuracy_metrics.accuracy:.1%})")
        
        # 2. REPRODUCIBILITY: Check restored files (medium/hard only)
        reproducibility_metrics: Optional[ReproducibilityMetrics] = None
        if removed_by_difficulty_filters:
            logger.info("Computing reproducibility metrics...")
            reproducibility_metrics = evaluate_reproducibility(
                self._workspace_dir, 
                removed_by_difficulty_filters
            )
            logger.info(f"Reproducibility: {reproducibility_metrics.restored_count}/{reproducibility_metrics.targets_count} "
                       f"({reproducibility_metrics.restoration_rate:.1%})")
        
        # 3. FAITHFULNESS: LLM-as-judge for answer grounding
        logger.info(f"Computing faithfulness metrics (LLM-as-judge with {judge_llm})...")
        required_questions = list(gt_result[0].keys())
        faithfulness_metrics = await evaluate_faithfulness(
            questions=required_questions,
            submitted=reported_result,
            tool_calls=tool_calls,
            tool_results=tool_results_for_faithfulness,
            judge_model=judge_llm,
        )
        logger.info(f"Faithfulness: {faithfulness_metrics.score:.2f} "
                   f"(grounded={faithfulness_metrics.is_grounded}, "
                   f"guessing={faithfulness_metrics.suspected_guessing})")
        
        # Count command timeouts for task adherence context
        command_timeouts = sum(
            1 for r in tool_results_for_faithfulness
            if "timed out" in str(r.get("result", "")).lower() 
            or "timeout" in str(r.get("result", "")).lower()
        )
        
        # 4. TASK ADHERENCE: LLM-as-judge for execution quality
        logger.info(f"Computing task adherence metrics (LLM-as-judge with {judge_llm})...")
        adherence_metrics = await evaluate_task_adherence(
            domain=domain,
            task_prompt=task_prompt,
            steps_used=steps_used,
            tool_calls=tool_calls,
            protocol_errors=protocol_errors,
            submitted=reported_result,
            accuracy_result=accuracy_metrics,
            action_trace=action_trace,
            judge_model=judge_llm,
            command_timeouts=command_timeouts,
        )
        logger.info(f"Task adherence: {adherence_metrics.score:.2f} "
                   f"(navigation={adherence_metrics.navigation_quality})")
        
        # 5. EFFICIENCY: Resource usage metrics
        efficiency_metrics = compute_efficiency(
            steps_used=steps_used,
            max_steps=max_steps,
            tool_calls=tool_calls,
            time_seconds=task_time_seconds,
            protocol_errors=protocol_errors,
            tool_results=tool_results_for_faithfulness,  # For counting timeouts
        )
        timeout_info = f", {efficiency_metrics.command_timeouts} timeouts" if efficiency_metrics.command_timeouts else ""
        logger.info(f"Efficiency: {steps_used}/{max_steps} steps, "
                   f"{len(tool_calls)} tool calls, {task_time_seconds:.1f}s{timeout_info}")
        
        # Build complete evaluation result
        task_success = accuracy_metrics.accuracy == 1.0
        
        evaluation = TaskEvaluation(
            task_id=task_id,
            domain=domain,
            success=task_success,
            accuracy=accuracy_metrics,
            reproducibility=reproducibility_metrics,
            faithfulness=faithfulness_metrics,
            task_adherence=adherence_metrics,
            efficiency=efficiency_metrics,
            submitted_answer=reported_result,
            ground_truth=gt_result,
        )
        
        # Log evaluation summary
        eval_dict = evaluation.to_dict()
        logger.info(f"Evaluation summary: {json.dumps(eval_dict, indent=2, default=str)}")
        
        if trace:
            trace.add({"type": "evaluation", "evaluation": eval_dict})
        
        # =====================================================================
        # CLEANUP
        # =====================================================================
        logger.info("Cleaning up environment")
        env_dir = os.path.join(self._workspace_dir, "environment")
        if os.path.exists(env_dir):
            shutil.rmtree(env_dir)
        logger.debug("Environment cleanup complete")

        # Close trace file and optionally delete it
        if trace:
            trace_path = trace.jsonl_path
            try:
                trace.close()
                if keep_traces:
                    logger.info(f"Wrote execution trace: {trace_path}")
                else:
                    # Delete trace file to save disk space
                    if trace_path.exists():
                        trace_path.unlink()
                        logger.debug(f"Deleted trace file: {trace_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup execution trace: {e}")
        
        return evaluation

    # =========================================================================
    # DEPRECATED: Old metric methods removed - now using metrics.py module
    # The following functionality has been moved to metrics.py:
    # - __eval_result_json -> evaluate_accuracy()
    # - _compute_task_completion_metrics -> (integrated into AccuracyMetrics)
    # - _score_reproducability -> evaluate_reproducibility()
    # - _compute_faithfulness_metrics -> evaluate_faithfulness() (LLM-as-judge)
    # - _get_metrics -> aggregate_results()
    # =========================================================================

    def _build_task_prompt(self, task: dict, domain: str, use_mcp: bool = True) -> str:
        """Build the initial task prompt for the purple agent.
        
        The prompt is structured in sections:
        1. Task objective and questions
        2. Constraints (common + domain-specific + tool-specific)
        3. Available tools
        4. Response format instructions
        """
        observation = str(task["results"][0].keys())
        task_prompt_text = task.get("task_prompt", "")
        logger.debug(f"Observation keys: {observation}")
        logger.debug(f"Task prompt: {task_prompt_text}")

        # Build tools section
        tools_section = "No MCP tools available."
        if use_mcp and self._mcp_tools:
            tools_str = mcp_tools_to_str(self._mcp_tools)
            tools_section = f"""## Available Tools
You have access to the following MCP tools (use ONE tool at a time):
{tools_str}"""
            logger.debug(f"Tools section created with {len(self._mcp_tools)} tools")
        
        # Build domain-specific instruction and constraints
        if domain == "corebench_easy":
            objective = (
                f"Your goal is to answer questions about the output of scientific code. "
                f"Read through the files in `environment/results` to answer these questions:\n{observation}\n\n"
                f"**You should NOT run or execute any code.** All answers are available in the results directory."
            )
            domain_constraints = EASY_CONSTRAINTS
        elif domain == "corebench_medium":
            objective = (
                f"Your goal is to test the computational reproducibility of a scientific code capsule. "
                f"Specifically, you need to {task_prompt_text} to answer these questions:\n{observation}\n\n"
                f"Read `environment/REPRODUCING.md` for instructions on how to run the capsule."
            )
            domain_constraints = MEDIUM_CONSTRAINTS
        elif domain == "corebench_hard":
            objective = (
                f"Your goal is to test the computational reproducibility of a scientific code capsule. "
                f"Specifically, you need to {task_prompt_text} to answer these questions:\n{observation}\n\n"
                f"No reproduction instructions are provided. You must figure out how to run the code yourself."
            )
            domain_constraints = HARD_CONSTRAINTS
        else:
            raise ValueError(f"Unknown domain: {domain}")

        # Combine all constraints
        all_constraints = COMMON_CONSTRAINTS.copy()
        if use_mcp:
            all_constraints.extend(TOOL_CONSTRAINTS)
        all_constraints.extend(domain_constraints)
        
        constraints_text = "\n".join(f"- {c}" for c in all_constraints)
        
        logger.debug(f"Built prompt with {len(all_constraints)} constraints for domain: {domain}")

        # Build the full prompt
        full_prompt = f"""# Task: {domain.replace('_', ' ').title()}

## Objective
{objective}

Your submitted answer must be a Python dictionary where:
- Keys are EXACTLY the questions listed above
- Values are the precise answers (numeric values, exact text, etc.)

## Important Constraints
{constraints_text}

{tools_section}

## Response Format
Respond with a single JSON object wrapped in `<json>...</json>` tags.

The JSON must contain:
- `"name"`: The tool to call, OR `"{RESPOND_ACTION_NAME}"` when ready to submit your final answer
- `"arguments"`: The tool arguments, OR `{{"content": {{...}}}}` with your answer dictionary

**Rules:**
- Use only ONE tool per response
- Do NOT combine a tool call with a final answer
- When including multi-line strings in JSON, escape newlines as `\\n`

## Examples

Calling a tool:
<json>
{json.dumps({"name": "inspect_file_as_text", "arguments": {"file_path": "environment/results/output.txt", "start_line": 1, "end_line": 50}}, indent=2)}
</json>

Submitting final answer:
<json>
{json.dumps({"name": RESPOND_ACTION_NAME, "arguments": {"content": {"Report the error of the LSTM.": 0.4142, "Report the x-axis label of the figure.": "Solubility"}}}, indent=2)}
</json>

Begin by exploring the environment to understand the task.
"""
        logger.debug(f"Full prompt length: {len(full_prompt)} chars")
        return full_prompt

    async def _parse_and_execute_tools(self, response: str, use_mcp: bool = False) -> tuple[AgentAction, Optional[str]]:
        """
        Parse the purple agent's response as a single JSON action and optionally execute an MCP tool.

        The benchmark protocol expects the purple agent to ALWAYS return exactly one JSON object
        wrapped in `<json>...</json>` tags (or equivalent code block).

        Returns:
            (action, tool_result)
            - action: validated JSON envelope (`AgentAction`)
            - tool_result: text from MCP tool execution, if a tool was executed

        Raises:
            ValueError: if the response cannot be parsed/validated as an `AgentAction`.
        """
        import re

        logger.debug("Parsing tool call from response")

        json_str: Optional[str] = None

        # Prefer explicit <json>...</json> tags.
        match = re.search(r"<json>\s*(.*?)\s*</json>", response, re.DOTALL)
        if match:
            json_str = match.group(1)
            logger.debug("Found JSON in <json> tags")
        else:
            # Try markdown code blocks.
            match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                json_str = match.group(1)
                logger.debug("Found JSON in ```json``` code block")
            else:
                match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    logger.debug("Found JSON in ``` code block")

        action_dict: Optional[dict[str, Any]] = None
        if json_str is not None:
            try:
                action_dict = json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in <json> tags: {e}") from e
        else:
            # Allow raw JSON as a fallback.
            try:
                action_dict = json.loads(response)
            except json.JSONDecodeError as e:
                raise ValueError(
                    "Purple agent response did not contain a JSON action. "
                    "Reply with <json>{\"name\": ..., \"arguments\": {...}}</json> only."
                ) from e

        try:
            # must return valid schema with name: str and arguments: dict
            action = AgentAction.model_validate(action_dict)
        except ValidationError as e:
            raise ValueError(f"Purple agent JSON missing required fields: {e}") from e

        tool_name = action.name
        arguments = action.arguments
        logger.info(f"Parsed tool call: {tool_name}")
        logger.debug(f"Arguments: {json.dumps(arguments, indent=2)}")

        if tool_name == RESPOND_ACTION_NAME:
            return action, None

        if use_mcp and self._mcp_tools:
            mcp_tool_names = {
                t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
                for t in self._mcp_tools
            }
            if tool_name in mcp_tool_names:
                logger.info(f"Executing MCP tool: {tool_name}")
                result = await self._call_mcp_tool(tool_name, arguments)
                return action, result

        return action, None


def tau2_evaluator_agent_card(name: str, url: str) -> AgentCard:
    """Create the agent card for the evaluator."""
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
