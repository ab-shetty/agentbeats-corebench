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
from dataclasses import asdict
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
import os

import numpy as np
from scipy.stats import t
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

import urllib.request
import tarfile
import zipfile
import socket
import gdown

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
from scenarios.corebench.metrics.metrics import (
    evaluate_accuracy,
    evaluate_task_adherence,
    aggregate_results,
    extract_methodology_metrics,
    AccuracyMetrics,
    TaskAdherenceMetrics,
    _empty_accuracy_metrics,
)
from scenarios.corebench.metrics.models import (
    TaskEvaluation,
    AggregateMetrics,
    MethodologyMetrics,
)

from model_prices import MODEL_PRICES_DICT

# Optional Phoenix Cloud exporter (set PHOENIX_API_KEY to enable)
try:
    from scenarios.corebench.phoenix_exporter import export_trace as phoenix_export_trace
except ImportError:
    phoenix_export_trace = None  # Phoenix exporter not available

# Setup logging - will be initialized in main()
logger = logging.getLogger("evaluator")

# Suppress verbose library logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("a2a").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


RESPOND_ACTION_NAME = "FINAL_ANSWER"

# Define workspace directory
WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")

# =============================================================================
# AGENT PROMPT CONSTRAINTS
# These structured constraints help guide the purple agent's behavior and
# prevent common failure modes observed during benchmark runs.
# =============================================================================

# Constraints specific to MCP tool usage
TOOL_CONSTRAINTS = [
    "Shell State: The 'execute_bash' tool is STATELESS. 'cd' commands do NOT persist between calls. Use relative paths from the capsule root or chain commands (e.g. 'cd code && python run.py').",
]
    # "File Reading: Use 'inspect_file_as_text' to read code, documentation, or text-based results.",
    # "Vision: To analyze images (plots, charts, figures), use 'query_vision_language_model' with the image path.",
    # "Vision Questions: Questions starting with 'fig' or mentioning 'figure/plot/chart' typically REQUIRE using 'query_vision_language_model' on the relevant image file.",
    # "Search: If you need external documentation, use 'web_search' sparingly.",
    # "Working Directory: Tools run in the capsule root (the folder that contains code/, results/, data/). Use paths like 'results/' or 'code/', not 'environment/results'.",
    # "Shell State: The 'execute_bash' tool is STATELESS. 'cd' commands do NOT persist between calls. Use relative paths from the capsule root or chain commands (e.g. 'cd code && python run.py').",
    # "Large Outputs: Avoid `cat` on long logs (tool output may be truncated). Prefer `grep`/`tail` or `inspect_file_as_text` to target the exact lines you need.",
    # "Timeouts: Long-running commands may timeout. Break complex operations into smaller steps.",

class AgentAction(BaseModel):
    """Validated JSON envelope for purple-agent outputs."""
    name: str
    arguments: dict[str, Any]


class ExecutionTraceWriter:
    """
    Write a per-task execution trace as JSONL for analysis and LLM-as-judge evaluation.
    
    Usage:
        with ExecutionTraceWriter(jsonl_path, run_id) as trace:
            trace.add({"type": "task_start", ...})
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

    # Fields that should NOT be truncated (needed for checking llm as a judge reproducibility)
    _NO_TRUNCATE_KEYS = {"prompt"}

    @staticmethod
    def _truncate(value: Any, *, limit: int = 4000, key: str = "") -> Any:
        """Recursively truncate long strings in nested structures."""
        if isinstance(value, str) and len(value) > limit:
            return value[:limit] + f"... (truncated, original_len={len(value)})"
        if isinstance(value, list):
            return [ExecutionTraceWriter._truncate(v, limit=limit) for v in value]
        if isinstance(value, dict):
            return {
                k: v if k in ExecutionTraceWriter._NO_TRUNCATE_KEYS
                else ExecutionTraceWriter._truncate(v, limit=limit, key=k)
                for k, v in value.items()
            }
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
            
        # Write event to log file with run_id for correlation
        event = {**event, "run_id": self.run_id}
        
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
        """
        if event_type is None:
            return self._events.copy()
        return [e for e in self._events if e.get("type") == event_type]

    def get_tool_calls(self) -> list[dict[str, Any]]:
        """
        Extract tool calls.

        Prefers the evaluator-originated `tool_call` events ("green -> mcp") when present.
        Falls back to the legacy `action` events (purple requests) for older traces.
        """
        tool_call_events = [e for e in self._events if e.get("type") == "tool_call"]
        if tool_call_events:
            return [
                {"tool": e.get("tool"), "arguments": e.get("arguments", {})}
                for e in tool_call_events
            ]

        return [
            {"tool": e.get("name"), "arguments": e.get("arguments", {})}
            for e in self._events
            if e.get("type") == "action" and e.get("name") != RESPOND_ACTION_NAME
        ]

    def get_tool_results(self) -> list[dict[str, str]]:
        """Extract tool results"""
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
        
        # List tools
        tools_response = await self._send_request("tools/list", {})
        logger.debug(f"Tools list response: {json.dumps(tools_response, indent=2)}")
        
        if "result" in tools_response:
            self.tools = tools_response["result"].get("tools", [])
        
        for tool in self.tools:
            tool_name = tool.get('name') if isinstance(tool, dict) else str(tool)
            logger.debug(f"  - {tool_name}")
        
        return self
    
    async def _send_request(self, method: str, params: dict, timeout: float = 1000.0) -> dict:
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
        
        # logger.debug(f"Received response for request {self.request_id}: {response_line[:500]}")
        
        return json.loads(response_line)
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return its result as text"""
        logger.debug(f"Calling MCP tool: {tool_name}")
        
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
                logger.debug(f"Tool {tool_name} returned {len(result_text)} chars")
                return result_text
            return str(first_item)
        
        logger.warning(f"Tool {tool_name} executed but returned no content")
        return "Tool executed but returned no content"
    
    async def disconnect(self):
        """Clean up server process"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
                # logger.info("MCP server terminated cleanly")
            except subprocess.TimeoutExpired:
                logger.warning("MCP server didn't terminate, killing")
                self.process.kill()
                self.process.wait()

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

# ----------------------
# Load manifest once
# ----------------------
# Manifest JSON example:
# [
#  {"capsule_id": "capsule-3560168", "gdrive_file_id": "1vZK3IitSvqu_Ic3zrcviJ1aojYX6lpUY"},
# ]

HERE = Path(__file__).resolve().parent
CAPSULE_EXTENSION_PATH = HERE / "capsule_extension.json"
if not CAPSULE_EXTENSION_PATH.exists():
    logger.warning(f"{CAPSULE_EXTENSION_PATH} does not exist — GDrive fallback will fail")

with open(CAPSULE_EXTENSION_PATH) as f:
    CAPSULE_LOOKUP = {c["capsule_id"]: c.get("gdrive_file_id") for c in json.load(f)}
logger.info(f"Loaded {len(CAPSULE_LOOKUP)} capsules from manifest")

# with open("./scenarios/corebench/capsule_extension.json") as f:
#     CAPSULE_LOOKUP = {c["capsule_id"]: c.get("gdrive_file_id") for c in json.load(f)}


# ----------------------
# Helper: Princeton download
# ----------------------
def _download_from_princeton(capsule_id: str, capsules_dir: Path):
    """
    Try to download a capsule from Princeton CoreBench mirror.
    Returns (success_flag: bool, message: str)
    """
    capsule_dir = capsules_dir / capsule_id
    tar_path = capsules_dir / f"{capsule_id}.tar.gz"
    capsule_url = f"https://corebench.cs.princeton.edu/capsules/{capsule_id}.tar.gz"

    try:
        capsules_dir.mkdir(parents=True, exist_ok=True)
        socket.setdefaulttimeout(300)
        logger.info(f"Downloading {capsule_id} from Princeton mirror...")
        urllib.request.urlretrieve(capsule_url, tar_path)
    except Exception as e:
        # Likely 404 if capsule not in Princeton
        return False, f"Capsule {capsule_id} not found in Princeton: {e}"

    try:
        logger.info(f"Extracting {capsule_id}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(capsules_dir)
        tar_path.unlink()
        return True, f"Downloaded capsule {capsule_id} from Princeton"
    except Exception as e:
        if tar_path.exists():
            tar_path.unlink()
        return False, f"Failed to extract Princeton capsule {capsule_id}: {e}"


# ----------------------
# Helper: Google Drive download
# ----------------------
def download_capsule_from_gdrive(capsule_id: str, gdrive_file_id: str, target_dir: Path):
    """
    Download a capsule ZIP from Google Drive and extract it
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    capsule_dir = target_dir / capsule_id
    if capsule_dir.exists():
        logger.info(f"{capsule_id} already exists at {capsule_dir}")
        return capsule_dir

    zip_path = target_dir / f"{capsule_id}.zip"
    url = f"https://drive.google.com/uc?id={gdrive_file_id}"

    logger.info(f"Downloading {capsule_id} from Google Drive...")
    gdown.download(url, str(zip_path), quiet=False)

    # logger.info(f"Extracting {capsule_id}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(capsule_dir)

    zip_path.unlink()
    logger.info(f"Downloaded and extracted {capsule_id} to {capsule_dir}")
    return capsule_dir


# ----------------------
# Main: download single capsule (targeted)
# ----------------------
def download_corebench_capsule(capsule_id: str, target_dir: str = "./scenarios/corebench/capsules"):
    """
    Download a single CoreBench capsule (targeted):
    1) Try Princeton mirror first
    2) Fallback to Google Drive if listed in manifest
    """
    target_dir = Path(target_dir)
    capsule_dir = target_dir / capsule_id
    
    logger.info(f"Downloading capsule {capsule_id}")

    if capsule_dir.exists():
        logger.info(f"{capsule_id} already exists locally")
        return capsule_dir  # Return path, not message

    # Try Princeton
    ok, msg = _download_from_princeton(capsule_id, target_dir)
    if ok:
        logger.info(msg)
        return capsule_dir  # Return path, not message
    
    logger.warning(f"Princeton failed: {msg}")

    # Fallback: Google Drive
    gdrive_file_id = CAPSULE_LOOKUP.get(capsule_id)
    if gdrive_file_id:
        return download_capsule_from_gdrive(capsule_id, gdrive_file_id, target_dir)

    raise FileNotFoundError(f"No download source available for {capsule_id}")



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
    
    return dataset


def get_task_ids(domain: str, task_ids: Optional[list[str]], num_tasks: Optional[int] = None, task_index: Optional[int] = None) -> list[dict]:
    """
    Select which tasks to run for this evaluation session.

    - loads *all* tasks via `get_tasks()`
    - if `task_ids` is provided, filters to only those specific task IDs
    - if `num_tasks` is provided, takes the first `num_tasks` entries

    Returns: the selected task dicts from `core_test.json`.
    """
    task_set_name = domain
    task_split_name = "base"
    
    logger.info(f"Getting task IDs for domain: {domain}, task_ids: {task_ids}, num_tasks: {num_tasks}, task_index: {task_index}")
    
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
    
    # Select specific task by index if provided
    if task_index is not None and 0 <= task_index < len(selected_tasks):
        selected_tasks = [selected_tasks[task_index]]
    
    return selected_tasks


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """
    Calculate cost in dollars based on model name and token counts.

    Args:
        model_name: Model name with provider prefix (e.g., "nebius/Qwen/Qwen3-Coder-30B-A3B-Instruct")
        input_tokens: Total input/prompt tokens
        output_tokens: Total output/completion tokens

    Returns:
        Cost in dollars, or None if model not found in price dictionary
    """
    if not model_name or model_name not in MODEL_PRICES_DICT:
        logger.warning(f"Model '{model_name}' not found in MODEL_PRICES_DICT")
        return None

    prices = MODEL_PRICES_DICT[model_name]
    prompt_price = prices.get("prompt_tokens", 0)
    completion_price = prices.get("completion_tokens", 0)

    cost = (input_tokens * prompt_price) + (output_tokens * completion_price)
    logger.debug(f"Cost calculation: model={model_name}, input={input_tokens}, output={output_tokens}, "
                f"prompt_price={prompt_price}, completion_price={completion_price}, cost=${cost:.6f}")
    return cost


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
        self._keep_environment = False
    
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
        Remove files based on difficulty level.

        Returns:
            List of deleted file/directory paths (relative to workspace)
        """
        deleted_files: list[str] = []
        env_dir = os.path.join(self._workspace_dir, "environment")
        results_dir = os.path.join(env_dir, "results")

        def _rel(abs_path: str) -> str:
            return os.path.relpath(abs_path, self._workspace_dir)

        # MEDIUM/HARD: Remove results
        if domain in ("corebench_medium", "corebench_hard"):
            if os.path.isdir(results_dir):
                logger.info(f"Removing results/ for {domain}")
                shutil.rmtree(results_dir)
                deleted_files.append("environment/results/")

        # HARD: Remove REPRODUCING.md, environment/ and run scripts 
        if domain == "corebench_hard":
            paths_to_remove = [
                os.path.join(env_dir, "REPRODUCING.md"),
                os.path.join(env_dir, "environment"),
                os.path.join(env_dir, "code", "run.sh"),
                os.path.join(env_dir, "code", "run"),
            ]

            for abs_path in paths_to_remove:
                if os.path.exists(abs_path):
                    rel_path = _rel(abs_path)
                    if os.path.isfile(abs_path):
                        logger.info(f"Removing file for hard mode: {rel_path}")
                        os.remove(abs_path)
                        deleted_files.append(rel_path)
                    else:
                        logger.info(f"Removing directory for hard mode: {rel_path}")
                        shutil.rmtree(abs_path)
                        deleted_files.append(f"{rel_path}/")

        # Log summary of deleted files for debugging
        if deleted_files:
            logger.info(f"[{domain}] Deleted {len(deleted_files)} files/dirs: {deleted_files}")
        else:
            logger.info(f"No files matched for deletion")

        return deleted_files
    
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

    async def _init_mcp_client(self, mcp_server_command: list[str], cwd: Optional[str] = None) -> list:
        """Initialize MCP client connection to a tool server."""
        try:
            logger.info(f"Initializing MCP client with command: {' '.join(mcp_server_command)}")
            effective_cwd = cwd or self._workspace_dir
            self._mcp_client = SimpleMCPClient(mcp_server_command, cwd=effective_cwd)
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
            
            result = await self._mcp_client.call_tool(tool_name, arguments)
            # logger.debug(f"MCP tool result length: {len(result)} chars")
            return result
                
        except Exception as e:
            error_msg = f"Error calling MCP tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return error_msg


    def _hint_for_tool_result(self, tool_name: str, tool_result: str) -> Optional[str]:
        """Return a short context clue for the Purple Agent based on select tool failure patterns.

        Only hints that provide value beyond the original error message are included.
        Hints should clarify cryptic errors, not repeat information or give away solutions.
        """
        # Safety check for empty/None results
        if not tool_result:
            return None

        text = tool_result.lower()

        # Filter successful results
        # Bash exit code 0 = success, no hint needed
        if tool_name == "execute_bash" and "exit code: 0" in text:
            return None
        # Non-error results without tracebacks are successful
        # Bash errors (Exit Code 1) contain "traceback" so they pass through
        if not tool_result.strip().startswith("Error") and "traceback" not in text:
            return None

        # File read errors
        # MCP returns "Error reading file: Not a regular file" which is cryptic
        # This hint clarifies that the path might be a directory, not a file
        if tool_name == "inspect_file_as_text" and ("error reading file" in text or "not a regular file" in text):
            return "File read error. Ensure the path is correct and points to a file, not a directory."

        return None

    def _write_tool_output(self, *, tool_name: str, tool_result: str, index: int) -> str:
        """Persist full tool output to a workspace file and return its workspace-relative path instead of overloading model context."""
        import re

        # MCP tools are sandboxed to the capsule root: workspace/environment.
        env_dir = os.path.join(self._workspace_dir, "environment")
        out_dir = os.path.join(env_dir, "tool_outputs")
        os.makedirs(out_dir, exist_ok=True)

        safe_tool = re.sub(r"[^a-zA-Z0-9_-]+", "_", tool_name).strip("_") or "tool"
        filename = f"{index:04d}_{safe_tool}.txt"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(tool_result)

        # Return a path relative to the capsule root so the agent can read it with inspect_file_as_text.
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

    def _format_methodology_score(self, score: Optional[float]) -> str:
        """Format a per-task methodology score for display, handling missing values."""
        if score is None:
            return "N/A"
        try:
            return f"{score:.2f}"
        except (TypeError, ValueError):
            logger.warning("Unexpected methodology_score value %r; treating as N/A", score)
            return "N/A"


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
        start_time = time.time()
        run_id = str(uuid.uuid4())[:8]

        # Extract config
        domain = req.config["domain"]
        task_ids = req.config.get("task_ids", None)
        num_tasks = req.config.get("num_tasks", None)
        task_index = req.config.get("task_index", None)
        max_steps = req.config.get("max_steps", 200)
        use_cache = req.config.get("use_cache", False)  # Whether to cache capsules for reuse
        user_llm_args = req.config.get("user_llm_args", {})
        keep_traces = req.config.get("keep_traces", False)  # Whether to keep trace files after run

        # LLM-as-judge model
        default_judge_model = os.getenv("COREBENCH_TEXT_MODEL") or "gpt-5-mini"
        judge_llm = req.config.get("judge_llm", default_judge_model) # if override in scenario.toml exists

        logger.info(f"=" * 80)
        logger.info(f"🚀 STARTING COREBENCH EVALUATION | Run ID: {run_id}")
        logger.info(f"📊 Domain: {domain} | Tasks: {num_tasks or 'all'}{f' (index {task_index})' if task_index else ''} | Max Steps: {max_steps}")
        logger.info(f"🤖 Judge Model: {judge_llm}")
        logger.info(f"Keep traces: {keep_traces} | Use cache: {use_cache}")

        # Create a per-run trace folder: logs/traces/<YYYYMMDD>_<run_id>_<domain>/
        runday = datetime.now(timezone.utc).strftime('%Y%m%d')
        base_trace_dir = Path(os.getenv("COREBENCH_TRACE_DIR") or os.getenv("COREBENCH_LOG_DIR") or "logs") / "traces"
        run_trace_dir = base_trace_dir / f"{runday}_{run_id}_{domain}"
        run_trace_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"🗂️  Run trace dir: {run_trace_dir}")
        
        # MCP server configuration
        use_mcp = req.config.get("use_mcp", False)
        mcp_server_command = req.config.get("mcp_server_command", ["uv", "run", "mcp", "run", "mcp_server.py"])
        self._keep_environment = req.config.get("keep_environment", False)
        resolved_mcp_command = []
        for part in mcp_server_command:
            if isinstance(part, str) and part.endswith(".py") and not os.path.isabs(part):
                # The MCP process is started with `cwd=self._workspace_dir`, so resolve any
                # local `.py` paths to absolute paths to avoid "file not found" errors.
                resolved_mcp_command.append(os.path.abspath(part))
            else:
                resolved_mcp_command.append(part)

        # Get the purple agent URL
        agent_url = str(req.participants["agent"])
        logger.info(f"🔗 Purple Agent: {agent_url}")

        # Initialize MCP client if enabled
        if use_mcp:
            # Ensure workspace exists before starting MCP server
            self._reset_workspace()

        # Get task IDs - is this the same as num tasks and ID's above?
        resolved_task_ids = get_task_ids(domain, task_ids, num_tasks, task_index)
        logger.debug(f"📝 Running {len(resolved_task_ids)} tasks")
        logger.debug("")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation of {len(resolved_task_ids)} tasks in {domain} domain")
        )

        # Collect all task evaluations and cost metadata
        task_evaluations: list[TaskEvaluation] = []
        task_cost_metadata: list[Optional[Dict[str, Any]]] = []

        try:
            for idx, task in enumerate(resolved_task_ids, 1):
                task_id = task["capsule_id"]
                logger.info(f"📦 TASK [{idx}/{len(resolved_task_ids)}] Starting {task_id}")
                
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Running task {task_id}...")
                )
                try:
                    task_evaluation, cost_meta = await self._run_single_task(
                        agent_url=agent_url,
                        domain=domain,
                        task=task,
                        task_id=task_id,
                        max_steps=max_steps,
                        judge_llm=judge_llm,
                        use_mcp=use_mcp,
                        run_id=run_id,
                        run_trace_dir=run_trace_dir,
                        keep_traces=keep_traces,
                        use_cache=use_cache,
                        mcp_server_command=resolved_mcp_command,
                    )
                    task_evaluations.append(task_evaluation)
                    task_cost_metadata.append(cost_meta)
                    
                    # Show task summary
                    logger.info(f"✅ Task {task_id} Complete:")
                    logger.info(f"   Accuracy: {task_evaluation.accuracy.accuracy:.1%} ({task_evaluation.accuracy.correct_answers}/{task_evaluation.accuracy.total_questions})")
                    logger.info(f"   Task Adherence: {task_evaluation.task_adherence.score:.2f}/1.0")
                    # Show violations prominently if any
                    if task_evaluation.methodology_metrics and task_evaluation.methodology_metrics.violations:
                        violations_str = ', '.join(task_evaluation.methodology_metrics.violations)
                        logger.info(f"   ⚠️ Violations: {violations_str}")

                except Exception as e:
                    logger.error(f"❌ Task {task_id} failed: {e}")
                    logger.debug(traceback.format_exc())
                    # Clean up environment directory to prevent "Directory not empty" errors
                    env_dir = os.path.join(self._workspace_dir, "environment")
                    if os.path.exists(env_dir):
                        try:
                            shutil.rmtree(env_dir)
                            logger.debug(f"Cleaned up environment directory after failure")
                        except Exception as cleanup_err:
                            logger.warning(f"Failed to clean up environment: {cleanup_err}")
                    # Create a failed evaluation
                    from scenarios.corebench.metrics.metrics import (
                        AccuracyMetrics, TaskAdherenceMetrics, _empty_accuracy_metrics
                    )
                    failed_eval = TaskEvaluation(
                        task_id=task_id,
                        domain=domain,
                        success=False,
                        accuracy=_empty_accuracy_metrics(),
                        task_adherence=TaskAdherenceMetrics(
                            score=0.0,
                            reasoning=f"Task failed: {e}", strengths=[], weaknesses=["Task execution failed"]
                        ),
                        submitted_answer={},
                        ground_truth=task.get("results", [{}]),
                        task_cost=None,
                    )
                    task_evaluations.append(failed_eval)
                    task_cost_metadata.append(None)  # No cost data for failed tasks

            # =====================================================================
            # AGGREGATE RESULTS
            # =====================================================================
            time_used = time.time() - start_time

            aggregate = aggregate_results(task_evaluations)

            # Calculate cost efficiency (cost per task)
            total_cost: Optional[float] = None
            cost_efficiency: Optional[float] = None
            total_input_tokens = 0
            total_output_tokens = 0
            model_used: Optional[str] = None

            for cost_meta in task_cost_metadata:
                if cost_meta:
                    total_input_tokens += cost_meta.get("input_tokens", 0)
                    total_output_tokens += cost_meta.get("output_tokens", 0)
                    if cost_meta.get("cost") is not None:
                        if total_cost is None:
                            total_cost = 0.0
                        total_cost += cost_meta["cost"]
                    if model_used is None and cost_meta.get("model"):
                        model_used = cost_meta["model"]

            cost_efficiency = None
            if total_cost is not None and aggregate.num_tasks > 0:
                cost_efficiency = total_cost / aggregate.num_tasks
                logger.info(f"Cost efficiency: ${cost_efficiency:.6f}/task (total: ${total_cost:.6f})")
            else:
                logger.warning("Could not calculate cost efficiency (no cost data available)")

            logger.info(f"\n{'=' * 80}")
            logger.info(f"🏆 EVALUATION COMPLETE")
            logger.info(f"{'=' * 80}")
            logger.info(f"✅ Success Rate: {aggregate.num_successful}/{aggregate.num_tasks} ({aggregate.pass_rate:.1%})")
            logger.info(f"📊 Mean Accuracy: {aggregate.mean_accuracy:.1%}")
            logger.info(f"📋 Mean Task Adherence: {aggregate.mean_adherence:.2f}/1.0")
            logger.info(f"🔧 Mean Methodology Score: {aggregate.mean_methodology_score:.2f}/1.0")
            logger.info(f"   - Doc Read Rate: {aggregate.doc_read_rate:.1%}")
            logger.info(f"   - Execution Attempt Rate: {aggregate.execution_attempt_rate:.1%}")
            logger.info(f"   - Successful Execution Rate: {aggregate.successful_execution_rate:.1%}")
            logger.info(f"   - Mean Error Recovery Rate: {aggregate.mean_error_recovery_rate:.1%}")

            # Build result data for leaderboard
            result_data = {
                "domain": domain,
                "num_tasks": aggregate.num_tasks,
                "num_successful": aggregate.num_successful,
                "pass_rate": round(aggregate.pass_rate, 4),
                "cost_efficiency": round(cost_efficiency, 4) if cost_efficiency is not None else None,  # Dollar cost per task

                # Accuracy metrics
                "mean_accuracy": round(aggregate.mean_accuracy, 4),
                "mean_written_accuracy": round(aggregate.mean_written_accuracy, 4),
                "mean_vision_accuracy": round(aggregate.mean_vision_accuracy, 4),
                "mean_adherence": round(aggregate.mean_adherence, 4),

                # Methodology metrics (deterministic)
                "mean_methodology_score": round(aggregate.mean_methodology_score, 4),
                "doc_read_rate": round(aggregate.doc_read_rate, 4),
                "execution_attempt_rate": round(aggregate.execution_attempt_rate, 4),
                "successful_execution_rate": round(aggregate.successful_execution_rate, 4),
                "mean_error_recovery_rate": round(aggregate.mean_error_recovery_rate, 4),

                "total_time": round(time_used, 2),
                "used_mcp": use_mcp,

                # Cost tracking
                "total_cost": round(total_cost, 4),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "model_used": model_used,

                # Per-task breakdown
                "task_results": aggregate.task_results,
            }

            # Format task results for display
            task_results_str = "\n".join(
                f"  {tid}: {'✅' if info['success'] else '❌'} "
                f"(acc={info['accuracy']:.1%}, "
                f"process={self._format_methodology_score(info.get('methodology_score'))})"
                for tid, info in aggregate.task_results.items()
            )

            cost_line = ""
            if cost_efficiency is not None:
                cost_line = f"  Cost Efficiency: ${cost_efficiency:.6f}/task (total: ${total_cost:.4f})\n"
                cost_line += f"  Tokens: {total_input_tokens:,} input, {total_output_tokens:,} output\n"

            summary = f"""\n⭐ CoreBench Benchmark Results ⭐
Domain: {domain}
Tasks: {aggregate.num_successful}/{aggregate.num_tasks} passed ({aggregate.pass_rate:.1%})

📊 Accuracy Metrics:
  Accuracy: {aggregate.mean_accuracy:.1%}
  Written: {aggregate.mean_written_accuracy:.1%}, Vision: {aggregate.mean_vision_accuracy:.1%}

🔧 Methodology Metrics (Deterministic):
  Methodology Score: {aggregate.mean_methodology_score:.2f}/1.0
  Doc Read Rate: {aggregate.doc_read_rate:.1%}
  Execution Attempt Rate: {aggregate.execution_attempt_rate:.1%}
  Successful Execution Rate: {aggregate.successful_execution_rate:.1%}
  Error Recovery Rate: {aggregate.mean_error_recovery_rate:.1%}

📋 Task Adherence (LLM Judge): {aggregate.mean_adherence:.2f}/1.0

⚡ Total Time: {time_used:.1f}s
{cost_line}
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

            # Write run summary to trace folder
            if run_trace_dir:
                summary_path = run_trace_dir / f"run_summary_{run_id}.json"
                with open(summary_path, "w") as f:
                    json.dump({
                        "run_id": run_id,
                        "domain": domain,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "num_tasks": len(resolved_task_ids),
                        "task_ids": [t["capsule_id"] for t in resolved_task_ids],
                        "model_used": model_used,
                        "aggregate_metrics": asdict(aggregate) if aggregate else None,
                    }, f, indent=2, default=str)
                logger.info(f"📄 Run summary written to: {summary_path}")

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
        mcp_server_command: Optional[list[str]] = None,
        run_id: str = "",
        run_trace_dir: Optional[Path] = None,
        keep_traces: bool = False,
        use_cache: bool = False,
    ) -> tuple[TaskEvaluation, Optional[Dict[str, Any]]]:
        """
        Run a single task and return a complete TaskEvaluation and cost metadata.

        This method orchestrates the full task lifecycle:
        1. Setup workspace and download capsule
        2. Interact with purple agent (tool calls loop)
        3. Evaluate all metrics (accuracy, methodology, adherence)
        4. Cleanup and return structured evaluation

        Args:
            judge_llm: LLM model name for LLM-as-judge evaluations (faithfulness, adherence)
            keep_traces: If True, trace files are kept after run. If False (default), deleted.
            use_cache: If True, capsules are downloaded to a cache directory and copied to workspace.
                      Subsequent runs reuse cached capsules. If False (default), capsules are
                      downloaded directly to workspace each time.

        Returns:
            Tuple of (TaskEvaluation, cost_metadata) where cost_metadata contains:
            - model: Model name used by purple agent
            - input_tokens: Total input tokens used
            - output_tokens: Total output tokens used
            - cost: Total cost in dollars (None if model not in price dict)
        """
        import platform
        
        logger.info(f"Starting single task: {task_id}")

        terminated = False

        # Initialize execution trace for this task
        trace: Optional[ExecutionTraceWriter] = None
        if run_trace_dir is not None:
            trace_dir = run_trace_dir
        else:
            base_trace_dir = Path(os.getenv("COREBENCH_TRACE_DIR") or os.getenv("COREBENCH_LOG_DIR") or "logs") / "traces"
            run_day = datetime.now(timezone.utc).strftime("%Y%m%d")
            # Use domain if available, else 'unknown'
            trace_dir = base_trace_dir / f"{run_day}_{run_id or 'unknown'}_{domain if domain else 'unknown'}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        trace_jsonl = trace_dir / f"{domain}_{trace_stamp}_{task_id}.jsonl"

        logger.info(f"📝 Writing trace to: {trace_jsonl}")

        try:
            trace = ExecutionTraceWriter(trace_jsonl, run_id=run_id or str(uuid.uuid4())[:8])
            trace.__enter__()  # Open the file
            trace.add(
                {
                    "type": "task_start",
                    "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "task_id": task_id,
                    "domain": domain,
                    "host": f"{platform.system()} {platform.release()} ({platform.machine()})",
                    "questions": list(task["results"][0].keys()),
                }
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize trace: {e}")
            trace = None
        
        os.makedirs(self._workspace_dir, exist_ok=True)
        env_dir = os.path.join(self._workspace_dir, "environment")

        # Ensure env_dir doesn't exist before setup
        if os.path.exists(env_dir):
            logger.warning(f"Environment directory already exists, removing it first")
            shutil.rmtree(env_dir)

        if use_cache:
            # Use caching: download to cache directory, then copy to workspace
            capsules_dir = os.path.join(os.path.dirname(__file__), "capsules")
            os.makedirs(capsules_dir, exist_ok=True)
            cached_capsule_path = os.path.join(capsules_dir, task_id)

            if os.path.exists(cached_capsule_path):
                logger.info(f"Found cached capsule {task_id}")
            else:
                logger.info(f"Downloading capsule {task_id} to cache")
                download_result = download_corebench_capsule(task_id, target_dir=capsules_dir)
                logger.info(f"Download result: {download_result}")

            # Copy from cache to environment directory
            logger.info(f"Copying cached capsule to workspace")
            shutil.copytree(cached_capsule_path, env_dir)
        else:
            # Default behavior: download directly to workspace
            logger.info(f"Downloading capsule {task_id} to workspace")
            download_result = download_corebench_capsule(task_id, target_dir=self._workspace_dir)
            logger.info(f"Download result: {download_result}")

            # Rename to environment directory
            capsule_path = os.path.join(self._workspace_dir, task_id)
            logger.debug(f"Renaming {capsule_path} to {env_dir}")
            os.rename(capsule_path, env_dir)

        # Apply difficulty filters
        deleted_files = self._apply_difficulty_filters(domain)

        # Add trace event for deleted files
        if trace and deleted_files:
            trace.add({
                "type": "difficulty_filter",
                "deleted_files": deleted_files,
            })

        # Initialize MCP client after staging so PWD is workspace/environment
        if use_mcp:
            try:
                await self._init_mcp_client(mcp_server_command or [], cwd=env_dir)
                # Check docker availability via MCP. This is to distinguish agent error from docker error.
                try:
                    docker_check = await self._call_mcp_tool(
                        "execute_bash",
                        {"command": "docker --version || which docker || echo 'docker not found'"}
                    )
                    logger.info(f"Docker availability (MCP): {docker_check.strip()}")
                except Exception as e:
                    logger.warning(f"Docker availability check failed: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize MCP: {e}")
                raise

        # Build the initial task description for the purple agent
        task_description = self._build_task_prompt(task, domain, use_mcp)

        # Start a new conversation with the purple agent
        next_message = task_description
        is_first_message = True

        logger.info(f"💬 Starting conversation with purple agent\n")

        answer: Any = None
        answer_metadata: Dict[str, Any] = {}  # Token/cost metadata from purple agent
        protocol_errors = 0
        max_protocol_errors = 5
        steps_used = 0
        tool_exec_index = 0

        while not terminated and steps_used < max_steps: # Prevent infinite loops
            turn = steps_used + 1
            logger.debug(f"Sending to purple agent (first_message={is_first_message})")
            
            try:
                response = await self._tool_provider.talk_to_agent(
                    message=next_message,
                    url=agent_url,
                    new_conversation=is_first_message,
                )
                response_preview = response[:300] + "..." if len(response) > 300 else response
                logger.debug(f"Purple agent response ({len(response)} chars): {response_preview}")
            except Exception as e:
                logger.error(f"Failed to communicate with purple agent: {e}")
                logger.debug(traceback.format_exc())
                raise

            is_first_message = False
            steps_used += 1

            # Extract and trace "plan" if present
            plan_match = re.search(r'<plan>\n?(.*?)\n?</plan>', response, re.DOTALL)
            if plan_match:
                if trace:
                    trace.add({
                        "type": "plan",
                        "flow": "purple -> trace",
                        "turn": turn,
                        "content": plan_match.group(1).strip(),
                    })
                # Remove plan block so _parse_and_execute_tools only sees JSON
                response = response.replace(plan_match.group(0), "").strip()

            try:
                action, tool_result = await self._parse_and_execute_tools(response, use_mcp)
                protocol_errors = 0
                # Note: action is now traced as tool_call (lines 1488-1496) instead of here
            except Exception as e:
                protocol_errors += 1
                logger.warning(f"Invalid agent response (protocol_errors={protocol_errors}/{max_protocol_errors}): {e}")
                logger.debug(traceback.format_exc())
                if trace:
                    # Classify the error type for better analysis
                    error_str = str(e).lower()
                    if "json" in error_str or "decode" in error_str or "parse" in error_str:
                        error_type = "json_decode"
                    elif "missing" in error_str or "required" in error_str:
                        error_type = "missing_field"
                    elif "valid" in error_str or "type" in error_str or "expected" in error_str:
                        error_type = "validation"
                    else:
                        error_type = "unknown"

                    trace.add(
                        {
                            "type": "protocol_error",
                            "flow": "purple -> green",
                            "turn": steps_used,
                            "error_type": error_type,
                            "error_message": str(e),
                            "raw_response_preview": response[:500] if response else "",
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

            # Log what the agent requested except for FINAL_ANSWER
            if action.name != RESPOND_ACTION_NAME:
                logger.info(f"📤 Agent requested: {action.name}")
                args_preview = json.dumps(action.arguments, indent=2, default=str)
                if len(args_preview) > 500:
                    args_preview = args_preview[:500] + "\n   ... (truncated)"
                logger.debug(f"   Arguments: {args_preview}")

            if tool_result is not None:
                result_preview = tool_result[:300] + "..." if len(tool_result) > 300 else tool_result
                logger.debug(f"Tool result ({len(tool_result)} chars): {result_preview}")
                #logger.debug(f"Full tool result: {tool_result}")
                tool_exec_index += 1
                
                # Extract key info for logging
                exit_code: Optional[int] = None
                if action.name == "execute_bash":
                    # Get command from arguments
                    cmd = action.arguments.get("command", "")
                    # Shorten display since we already saw it in the "Agent requested" log
                    cmd_display = cmd[:50] + "..." if len(cmd) > 50 else cmd
                    
                    # Extract exit code
                    match = re.search(r"Exit Code:\s*(\d+)", tool_result)
                    if match:
                        exit_code = int(match.group(1))
                    
                    # Show command and exit status
                    status = f"✓ Exit {exit_code}" if exit_code == 0 else f"❌ Exit {exit_code}"
                    logger.debug(f"🔧 bash finished: {cmd_display} → {status}")
                    
                    # Show truncated output (first 3 lines)
                    output_lines = tool_result.split('\n')
                    if len(output_lines) > 5:
                        preview = '\n'.join(output_lines[:5])
                        logger.debug(f"   Output: {preview}\n   ... ({len(output_lines)} lines total)")
                    elif tool_result.strip():
                        logger.debug(f"   Output: {tool_result[:200]}")
                        
                elif action.name == "inspect_file_as_text":
                    file_path = action.arguments.get("file_path", "")
                    file_size = len(tool_result)
                    num_lines = tool_result.count('\n')
                    logger.debug(f"🔧 inspect_file_as_text: {file_path} → {file_size} bytes, {num_lines} lines")
                    
                elif action.name == "query_vision_language_model":
                    image_path = action.arguments.get("image_path", "")
                    question = action.arguments.get("question", "")
                    question_display = question[:60] + "..." if len(question) > 60 else question
                    logger.debug(f"🔧 query_vision_language_model: {image_path}")
                    logger.debug(f"   Question: {question_display}")
                    logger.debug(f"   Answer: {tool_result[:250]}")
                    
                elif action.name == "web_search":
                    query = action.arguments.get("query", "")
                    logger.debug(f"🔧 web_search: {query}")
                    logger.debug(f"   Result: {tool_result[:250]}")
                    
                else:
                    # Generic catch all for all other tools
                    logger.info(f"🔧 {action.name}")
                    if tool_result and len(tool_result) > 100:
                        logger.debug(f"   Result: {tool_result[:200]}...")
                    elif tool_result:
                        logger.debug(f"   Result: {tool_result}")
                
                # Capture tool result in trace
                if trace:
                    trace.add(
                        {
                            "type": "tool_call",
                            "flow": "green -> mcp",
                            "turn": steps_used,
                            "tool": action.name,
                            "arguments": action.arguments,
                        }
                    )
                    trace.add(
                        {
                            "type": "tool_result",
                            "flow": "mcp -> green",
                            "turn": steps_used,
                            "tool": action.name,
                            "exit_code": exit_code,
                            "timed_out": "timed out" in tool_result.lower() or "timeout" in tool_result.lower(),
                            "hint": self._hint_for_tool_result(action.name, tool_result),
                            "summary": self._summarize_tool_result(tool_result, head_lines=15, tail_lines=15),
                        }
                    )
                next_message = self._format_tool_result_for_agent(tool_name=action.name, tool_result=tool_result, index=tool_exec_index)
                continue
            
            if action.name == RESPOND_ACTION_NAME:
                answer = action.arguments.get("content")
                # Extract token/cost metadata from purple agent
                answer_metadata = action.arguments.get("_metadata", {})

                if answer is None:
                    logger.warning(f"FINAL_ANSWER missing content; proceeding with empty answer")
                    answer = {}
                
                # Single consolidated FINAL_ANSWER log
                logger.info(f"📤 FINAL_ANSWER received (type={type(answer)})")
                logger.debug(f"   Content: {answer}")
                if answer_metadata:
                    logger.info(f"   Metadata: {answer_metadata}")
                
                if trace:
                    trace.add({
                        "type": "final_answer",
                        "flow": "purple -> green",
                        "turn": steps_used,
                        "content": answer,
                        "metadata": answer_metadata,
                    })
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
            logger.error(f"❌ Reached max_steps without answer")
            answer = {}

        # =====================================================================
        # EVALUATION PHASE
        # =====================================================================
        logger.info(f"\n{'\u2500' * 80}")
        logger.info(f"📊 EVALUATING TASK")
        logger.info(f"{'\u2500' * 80}\n")
        
        gt_result = task["results"]
        task_prompt = task.get("task_prompt", "")
        
        # Parse the agent's submitted answer
        reported_result: dict[str, Any] = {}
        try:
            if isinstance(answer, str):
                reported_result = json.loads(answer)
            elif isinstance(answer, dict):
                reported_result = answer
            else:
                logger.warning(f"⚠️  Invalid answer type: {type(answer)}")
                reported_result = {}
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse answer: {e}")
            reported_result = {}
        
        logger.debug(f"Reported result: {json.dumps(reported_result, indent=2)}")

        # Calculate cost from token metadata
        cost_metadata: Optional[Dict[str, Any]] = None
        if answer_metadata:
            model_name = answer_metadata.get("model", "")
            input_tokens = answer_metadata.get("input_tokens", 0)
            output_tokens = answer_metadata.get("output_tokens", 0)
            task_cost = calculate_cost(model_name, input_tokens, output_tokens)
            cost_metadata = {
                "model": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": task_cost,
            }
            logger.info(f"Task cost: model={model_name}, input_tokens={input_tokens}, "
                       f"output_tokens={output_tokens}, cost=${task_cost:.6f}" if task_cost else
                       f"Task cost: model={model_name}, input_tokens={input_tokens}, "
                       f"output_tokens={output_tokens}, cost=N/A (model not in price dict)")

        # Extract data from trace for LLM-as-judge evaluations
        tool_call_events = trace.get_events("tool_call") if trace else []
        tool_result_events = trace.get_events("tool_result") if trace else []
        tool_calls_count = len(tool_call_events)
        
        # 1. ACCURACY
        logger.info(f"1️⃣  Computing accuracy...")
        accuracy_metrics = evaluate_accuracy(gt_result, reported_result)
        logger.info(f"   ✓ Accuracy: {accuracy_metrics.correct_answers}/{accuracy_metrics.total_questions} ({accuracy_metrics.accuracy:.1%})")
        
        # Log expected vs submitted for debugging (with question numbers)
        logger.info(f"\n   📋 ANSWER COMPARISON:")
        expected_keys = list(gt_result[0].keys()) if gt_result else []
        submitted_keys = set(reported_result.keys()) if reported_result else set()
        # Debug: show key mismatch if any
        missing_keys = set(expected_keys) - submitted_keys
        extra_keys = submitted_keys - set(expected_keys)
        if missing_keys:
            logger.debug(f"   ⚠️ Missing keys in submission: {list(missing_keys)[:3]}")
        if extra_keys:
            logger.debug(f"   ⚠️ Extra keys in submission: {list(extra_keys)[:3]}")
        for i, key in enumerate(expected_keys, 1):
            expected_val = gt_result[0].get(key, "<missing>")
            # Retrieve submitted question string
            submitted_val = reported_result.get(key, "<not submitted>")
            match = "✓" if key in [r.question for r in accuracy_metrics.question_results if r.correct] else "✗"
            # Truncate long values for display
            exp_str = str(expected_val)[:50] + "..." if len(str(expected_val)) > 50 else str(expected_val)
            sub_str = str(submitted_val)[:50] + "..." if len(str(submitted_val)) > 50 else str(submitted_val)
            key_display = f"Q{i}: {key[:35]}..." if len(key) > 35 else f"Q{i}: {key}"
            logger.debug(f"   {match} {key_display}")
            logger.debug(f"      Expected:  {exp_str}")
            logger.debug(f"      Submitted: {sub_str}")

        # METHODOLOGY METRICS
        logger.info(f"2️⃣  Extracting methodology metrics...")
        methodology_metrics = extract_methodology_metrics(
            tool_calls=tool_call_events,
            tool_results=tool_result_events,
            domain=domain,
            task_prompt=task_prompt,
            deleted_files=deleted_files,
            capsule_id=task_id,
        )
        # Show script execution info
        executed_list = methodology_metrics.executed_scripts or []
        failed_list = methodology_metrics.attempted_failed_scripts or []
        executed_basenames = {os.path.basename(s) for s in executed_list}
        failed_basenames = {os.path.basename(s) for s in failed_list}

        if methodology_metrics.expected_scripts:
            expected_basenames = {os.path.basename(s) for s in methodology_metrics.expected_scripts}
            logger.info(f"   Expected: {list(expected_basenames)}")

            # What matched vs what didn't
            matched_success = expected_basenames & executed_basenames
            matched_failed = expected_basenames & failed_basenames
            other_success = executed_basenames - expected_basenames
            other_failed = failed_basenames - expected_basenames

            if matched_success:
                logger.info(f"   Executed: {list(matched_success)}")
            elif matched_failed:
                logger.info(f"   Executed: (none - expected script failed)")
            else:
                logger.info(f"   Executed: (none of expected)")

            # Show what else ran
            if other_success:
                logger.info(f"   Also ran: {list(other_success)} (succeeded, not expected)")
            if matched_failed:
                logger.info(f"   Failed: {list(matched_failed)} (expected script)")
            if other_failed:
                logger.info(f"   Failed: {list(other_failed)} (not expected)")
        else:
            # Generic task - no specific scripts in prompt
            if executed_list:
                logger.info(f"   Executed: {list(executed_basenames)} (generic task)")
            elif failed_list:
                logger.info(f"   Attempted: {list(failed_basenames)} (failed - generic task)")
            elif methodology_metrics.attempted_execution:
                logger.info(f"   Executed: (attempted but failed - generic task)")
        logger.info(f"   ✓ Methodology Score: {methodology_metrics.methodology_score:.2f}/1.0")
        # Show score breakdown with max possible for each component
        breakdown = methodology_metrics.score_breakdown
        if breakdown:
            # Define max weights per domain for display
            if breakdown.domain == "corebench_hard":
                max_doc, max_script, max_expected, max_success, max_recovery = 0.15, 0.20, 0.45, 0.20, 0.0
            elif breakdown.domain == "corebench_medium":
                max_doc, max_script, max_expected, max_success, max_recovery = 0.25, 0.0, 0.35, 0.25, 0.15
            else:  # easy
                max_doc, max_script, max_expected, max_success, max_recovery = 1.0, 0.0, 0.0, 0.0, 0.0

            logger.info(f"   Score Breakdown:")
            # Doc read
            docs_list = ', '.join(methodology_metrics.docs_read) if methodology_metrics.docs_read else 'none read'
            logger.info(f"     Doc Read:        {breakdown.doc_read_score:.2f}/{max_doc:.2f}  ({docs_list})")
            # Script read (only for hard mode)
            if breakdown.domain == "corebench_hard":
                scripts_list = ', '.join(methodology_metrics.scripts_read) if methodology_metrics.scripts_read else 'none read'
                logger.info(f"     Script Read:     {breakdown.script_read_score:.2f}/{max_script:.2f}  ({scripts_list})")
            # Script identification - show which expected scripts were actually run
            expected_basenames = {os.path.basename(s) for s in methodology_metrics.expected_scripts} if methodology_metrics.expected_scripts else set()
            executed_basenames = {os.path.basename(s) for s in (methodology_metrics.executed_scripts or [])}
            failed_basenames = {os.path.basename(s) for s in (methodology_metrics.attempted_failed_scripts or [])}

            # Filter out __ALL_R_SCRIPTS__ marker (used for for-loop glob scoring in metrics.py)
            executed_basenames.discard("__ALL_R_SCRIPTS__")
            failed_basenames.discard("__ALL_R_SCRIPTS__")

            all_attempted = executed_basenames | failed_basenames

            # Build detailed log showing expected vs attempted
            if expected_basenames:
                expected_str = ', '.join(sorted(expected_basenames))
            else:
                expected_str = "(generic task)"

            # Calculate what matched expected scripts (both successful and failed)
            matched_succeeded = expected_basenames & executed_basenames
            matched_failed = expected_basenames & failed_basenames
            unmatched_succeeded = executed_basenames - expected_basenames
            unmatched_failed = failed_basenames - expected_basenames

            # Build detailed coverage message
            coverage_parts = []
            if matched_succeeded:
                coverage_parts.append(f"✓ ran correct: {', '.join(sorted(matched_succeeded))}")
            if matched_failed:
                coverage_parts.append(f"✓ tried correct (failed): {', '.join(sorted(matched_failed))}")
            if unmatched_succeeded:
                coverage_parts.append(f"ran other: {', '.join(sorted(unmatched_succeeded))}")
            if unmatched_failed:
                coverage_parts.append(f"tried other (failed): {', '.join(sorted(unmatched_failed))}")

            if not coverage_parts:
                if methodology_metrics.attempted_execution:
                    coverage_parts.append("attempted execution (script names not captured)")
                else:
                    coverage_parts.append("not attempted")

            logger.info(f"     Script Attempt:  {breakdown.execution_coverage_score:.2f}/{max_expected:.2f}  (expected: {expected_str})")
            for part in coverage_parts:
                logger.info(f"                                     ({part})")
            # Execution success bonus - any script completing successfully
            if methodology_metrics.successful_execution:
                executed_list = methodology_metrics.executed_scripts or []
                if executed_list:
                    # Check if succeeded scripts are expected or other
                    success_basenames = {os.path.basename(s) for s in executed_list}
                    expected_success = success_basenames & expected_basenames
                    other_success = success_basenames - expected_basenames
                    success_parts = []
                    if expected_success:
                        success_parts.append(f"✓ {', '.join(sorted(expected_success))}")
                    if other_success:
                        success_parts.append(f"other: {', '.join(sorted(other_success))}")
                    success_detail = ', '.join(success_parts) if success_parts else "✓ code executed successfully"
                else:
                    success_detail = "✓ code executed successfully"
            else:
                success_detail = "✗ no successful run"
            logger.info(f"     Run Success:     {breakdown.successful_execution_score:.2f}/{max_success:.2f}  ({success_detail})")
            # Error info
            err = methodology_metrics.error_recovery
            if err.total_errors > 0:
                error_types_str = ', '.join(f"{k}:{v}" for k, v in err.error_types.items()) if err.error_types else 'unclassified'
                logger.info(f"     Error Info:      {err.errors_recovered}/{err.total_errors} recovered - {error_types_str}")
            else:
                logger.info(f"     Error Info:      no errors")
            # Penalties
            if breakdown.penalty != 0:
                logger.info(f"     Penalty:        {breakdown.penalty:.2f}       (no deps install on failure)")
            logger.info(f"     ─────────────────────")
        # Show additional details
        if methodology_metrics.stdout_captured:
            logger.info(f"   - Stdout Captured: {methodology_metrics.stdout_total_bytes:,} bytes")
        if methodology_metrics.violations:
            logger.info(f"   ⚠️ Violations: {', '.join(methodology_metrics.violations)}")

        # Add methodology metrics to trace
        if trace:
            trace.add({
                "type": "methodology_metrics",
                "flow": "green -> trace",
                "methodology_score": round(methodology_metrics.methodology_score, 4),
                "read_documentation": methodology_metrics.read_documentation,
                "docs_read": methodology_metrics.docs_read,
                "read_target_script": methodology_metrics.read_target_script,
                "scripts_read": methodology_metrics.scripts_read,
                "attempted_execution": methodology_metrics.attempted_execution,
                "execution_attempts": methodology_metrics.execution_attempts,
                "successful_execution": methodology_metrics.successful_execution,
                "installed_dependencies": methodology_metrics.installed_dependencies,
                "expected_scripts": methodology_metrics.expected_scripts,
                "executed_scripts": methodology_metrics.executed_scripts,
                "execution_coverage": round(methodology_metrics.execution_coverage, 4),
                "stdout_captured": methodology_metrics.stdout_captured,
                "stdout_total_bytes": methodology_metrics.stdout_total_bytes,
                "error_recovery": {
                    "total_errors": methodology_metrics.error_recovery.total_errors,
                    "errors_recovered": methodology_metrics.error_recovery.errors_recovered,
                    "recovery_rate": round(methodology_metrics.error_recovery.recovery_rate, 4),
                    "consecutive_failures": methodology_metrics.error_recovery.consecutive_failures,
                    "persistence_score": round(methodology_metrics.error_recovery.persistence_score, 4),
                },
                "violations": methodology_metrics.violations,
                "score_breakdown": {
                    "domain": methodology_metrics.score_breakdown.domain,
                    "doc_read_score": round(methodology_metrics.score_breakdown.doc_read_score, 4),
                    "script_read_score": round(methodology_metrics.score_breakdown.script_read_score, 4),
                    "execution_coverage_score": round(methodology_metrics.score_breakdown.execution_coverage_score, 4),
                    "successful_execution_score": round(methodology_metrics.score_breakdown.successful_execution_score, 4),
                    "error_recovery_score": round(methodology_metrics.score_breakdown.error_recovery_score, 4),
                    "penalty": round(methodology_metrics.score_breakdown.penalty, 4),
                    "total": round(methodology_metrics.score_breakdown.total, 4),
                } if methodology_metrics.score_breakdown else None,
            })

        # TASK ADHERENCE
        logger.info(f"3️⃣  Computing task adherence (LLM judge: {judge_llm})...")
        trace_event_callback = trace.add if trace else None
        adherence_metrics = await evaluate_task_adherence(
            domain=domain,
            task_prompt=task_prompt,
            questions=expected_keys,  # The actual questions the agent needs to answer
            tool_calls_count=tool_calls_count,
            submitted=reported_result,
            tool_calls=tool_call_events,
            tool_results=tool_result_events,
            workspace_dir=self._workspace_dir,
            trace_event_callback=trace_event_callback,
            judge_model=judge_llm,
        )
        logger.info(f"   ✓ Adherence Score: {adherence_metrics.score:.2f}/1.0")
        if adherence_metrics.reasoning:
            logger.info(f"\n   💭 Judge Reasoning (Task Adherence):")
            for line in adherence_metrics.reasoning.split('\n'):
                if line.strip():
                    logger.info(f"      {line}")
            logger.info("")
        
        # Build complete evaluation result
        task_success = accuracy_metrics.accuracy == 1.0

        evaluation = TaskEvaluation(
            task_id=task_id,
            domain=domain,
            success=task_success,
            accuracy=accuracy_metrics,
            task_adherence=adherence_metrics,
            submitted_answer=reported_result,
            ground_truth=gt_result,
            task_cost=cost_metadata.get("cost") if cost_metadata else None,
            methodology_metrics=methodology_metrics,
        )
        
        # Add evaluation to trace as dict
        eval_dict = evaluation.to_dict()

        # add evaluation to trace
        if trace:
            trace.add({"type": "evaluation", "flow": "green -> trace", "evaluation": eval_dict})
        
        logger.info("=" * 80)
        logger.info(f"✅ Completed task evaluation for capsule {task_id}")
        logger.info("=" * 80)

        # =====================================================================
        # CLEANUP
        # =====================================================================
        if not self._keep_environment:
            logger.info("Cleaning up environment")
            env_dir = os.path.join(self._workspace_dir, "environment")
            if os.path.exists(env_dir):
                shutil.rmtree(env_dir)
            logger.debug("Environment cleanup complete")
        else:
            logger.info(f"Keeping environment directory: {self._workspace_dir}/environment")
        
        # If MCP was used, disconnect the MCP server/client for this task
        if use_mcp and self._mcp_client:
            try:
                await self._mcp_client.disconnect()
            except Exception as e:
                logger.error(f"Error cleaning up MCP client: {e}")
            self._mcp_client = None
        
        # Export trace to Phoenix Cloud (if enabled) before closing
        if trace and phoenix_export_trace:
            try:
                phoenix_export_trace(
                    events=trace.get_events(),
                    task_id=task_id,
                    run_id=run_id,
                    domain=domain,
                )
            except Exception as e:
                logger.warning(f"Failed to export trace to Phoenix: {e}")

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

        return evaluation, cost_metadata
      
    def _build_task_prompt(self, task: dict, domain: str, use_mcp: bool = True) -> str:
        """Build the initial task prompt for the purple agent.
        
        The prompt is structured in sections:
        1. Task objective and questions
        2. Constraints (common + domain-specific + tool-specific)
        3. Available tools
        4. Response format instructions
        """
        observation = str(task["results"][0].keys())
        task_prompt = task["task_prompt"]
        logger.debug(f"Observation keys: {observation}")
        logger.debug(f"Task prompt: {task_prompt}")

        # Build tools section
        tools_section = "No MCP tools available."
        if use_mcp and self._mcp_tools:
            tools_str = mcp_tools_to_str(self._mcp_tools)
            tools_section = f"""## Available Tools
You have access to the following MCP tools (use ONE tool at a time):
{tools_str}"""
            logger.debug(f"Tools section created with {len(self._mcp_tools)} tools")
        
        # Format questions with numbers for readability (keys should be the question text itself).
        questions = list(task["results"][0].keys())
        json_fields = str(task["results"][0].keys())
        required_keys = questions
        required_keys_json = json.dumps(required_keys, indent=2, ensure_ascii=False)
        final_answer_example_content = {q: f"<ANSWER_{i+1}>" for i, q in enumerate(questions)}
        
        # Build domain-specific instruction and constraints
        if domain == "corebench_easy":
            objective = (
                f"Task: codeocean_easy\n\nYou should read through the files in the `environment/results` directory to answer the following questions: {json_fields}. Your submitted answer should be a Python dictionary whose keys are exactly those questions and values are the answers. **You should not actually run or execute any code.** All answers can be obtained by reading through the results directory."
            )

        elif domain == "corebench_medium":
            objective = (
                f"codeocean_medium\n\nYour goal is to test the computational reproducibility of the repository cloned to your current directory, which is code from a scientific paper. Specifically, you need to {task_prompt} to answer the following questions: {json_fields}. Your submitted answer should be a Python dictionary whose keys are exactly those questions and values are the answers. You should read the instructions on how to reproduce the capsule in REPRODUCING.md."
            )
        elif domain == "corebench_hard":
            objective = (
                f"Task: codeocean_hard\n\nYour goal is to test the computational reproducibility of the repository cloned to your current directory, which is code from a scientific paper. Specifically, you need to {task_prompt} to answer the following questions: {json_fields}. Your submitted answer should be a Python dictionary whose keys are exactly those questions and values are the answers. You should install all of the requirements found in the Readme file and then run the commands necessary to answer the questions."
            )
        else:
            raise ValueError(f"Unknown domain: {domain}")

        all_constraints: list[str] = []
        if use_mcp:
            all_constraints.extend(TOOL_CONSTRAINTS)
        constraints_text = "\n".join(f"- {c}" for c in all_constraints)
        logger.debug(f"Built prompt with {len(all_constraints)} constraints for domain: {domain}")

        # Build the full prompt
        full_prompt = f"""# Task: {domain.replace('_', ' ').title()}
## Objective
{objective}

## Your Goal
Your submitted answer must be a JSON object where:
- Keys are the EXACT QUESTION TEXT strings from the prompt (copy/paste exactly)
- Values are the EXACT answers you extracted from files.

⚠️ CRITICAL: You MUST extract real values. Do not summarize. Do not return "status" keys.
⚠️ REQUIRED: Return EXACTLY these keys (no more, no less):\n{required_keys_json}

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
- When including multi-line strings in a JSON tool call's content, escape newlines as `\\n`

## Examples

Calling a tool:
<json>
{json.dumps({"name": "execute_bash", "arguments": {"command": "grep -r 'accuracy' results/"}}, indent=2)}
</json>

Submitting final answer:
<json>
{json.dumps({"name": RESPOND_ACTION_NAME, "arguments": {"content": final_answer_example_content}}, indent=2, ensure_ascii=False)}
</json>

Begin by exploring the environment to understand the task.
"""
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
        logger.debug(f"Parsed tool call: {tool_name}, arguments: {json.dumps(arguments, indent=2)}")

        if tool_name == RESPOND_ACTION_NAME:
            return action, None

        if use_mcp and self._mcp_tools:
            mcp_tool_names = {
                t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
                for t in self._mcp_tools
            }
            if tool_name in mcp_tool_names:
                # logger.info(f"Executing MCP tool: {tool_name}")
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
    
    # logger.info("Server starting...")
    await uvicorn_server.serve()

def test_download_capsules():
    workspace_dir = "./test_corebench_download"
    os.makedirs(workspace_dir, exist_ok=True)
    
    capsule_ids = ["capsule-5507257", "capsule-3560168"]
    
    for capsule_id in capsule_ids:
        logger.info(f"Downloading capsule {capsule_id}...")
        try:
            result_path = download_corebench_capsule(capsule_id, target_dir=workspace_dir)
            if result_path.exists():  # Now works correctly
                logger.info(f"✅ Capsule {capsule_id} downloaded at: {result_path}")
        except Exception as e:
            logger.error(f"❌ Failed to download capsule {capsule_id}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
    # test_download_capsules()
