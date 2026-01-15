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
    evaluate_reproducibility,
    evaluate_task_adherence,
    compute_efficiency,
    aggregate_results,
    AccuracyMetrics,
    ReproducibilityMetrics,
    TaskAdherenceMetrics,
    EfficiencyMetrics,
    TaskEvaluation,
    AggregateMetrics,
    _empty_accuracy_metrics,
)

from model_prices import MODEL_PRICES_DICT

# Setup logging - will be initialized in main()
logger = logging.getLogger("evaluator")

# Suppress verbose library logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("a2a").setLevel(logging.WARNING)
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
    "Role: You are a seasoned research assistant with scientific computing experience.",
    "No guessing: If unsure, re-open the relevant files or outputs and extract evidence; do not invent values.",
    "Precision: Copy numeric values exactly as written; do not round unless the question asks for it.",
    "Answer format: For each question, return ONLY the value (number/label). Do not add explanations or sentences.",
    "Verification: Prefer targeted searches (grep/find) over repeatedly listing directories.",
    "Images: If a question references a figure/plot/chart, locate the image under results/ before using the vision tool.",
]

# Constraints specific to MCP tool usage
TOOL_CONSTRAINTS = [
    "File Reading: Use 'inspect_file_as_text' to read code, documentation, or text-based results.",
    "Vision: To analyze images (plots, charts, figures), use 'query_vision_language_model' with the image path.",
    "Vision Questions: Questions starting with 'fig' or mentioning 'figure/plot/chart' typically REQUIRE using 'query_vision_language_model' on the relevant image file.",
    "Search: If you need external documentation, use 'web_search' sparingly.",
    "Working Directory: Tools run in the capsule root (the folder that contains code/, results/, data/). Use paths like 'results/' or 'code/', not 'environment/results'.",
    "Shell State: The 'execute_bash' tool is STATELESS. 'cd' commands do NOT persist between calls. Use relative paths from the capsule root or chain commands (e.g. 'cd code && python run.py').",
    "Large Outputs: Avoid `cat` on long logs (tool output may be truncated). Prefer `grep`/`tail` or `inspect_file_as_text` to target the exact lines you need.",
    "Timeouts: Long-running commands may timeout. Break complex operations into smaller steps.",
]

# Easy mode: just read results, don't execute code
EASY_CONSTRAINTS = [
    "⚠️ EASY MODE - The experiments have ALREADY RUN.",
    "⛔️ NEGATIVE CONSTRAINT: Do NOT read .py files. The answer is NOT in the code logic.",
    "⛔️ NEGATIVE CONSTRAINT: Do NOT execute python code.",
    "Step 1: Run 'ls -R' to find result files (e.g. results/output, logs/run.log).",
    "Step 2: If a file is large or contains errors at the top, DO NOT GIVE UP.",
    "Step 3: Use 'grep' (via execute_bash) to find specific keywords like 'accuracy', 'score', 'test', or 'result' inside the file.",
    "Example: execute_bash(command='grep -i \"accuracy\" results/output')",
    "Tip: Many outputs contain multiple occurrences of a metric; the requested value is often near the END. Use `tail -n 200 results/output` and `grep -n -i <keyword> results/output | tail -n 20` to find the final value.",
    "Step 4: Extract the EXACT numeric value. If the file has 'libcudart' errors, ignore them and look for the final metrics at the bottom."
]

# Medium mode: follow REPRODUCING.md instructions
MEDIUM_CONSTRAINTS = [
    "Instructions: Read 'REPRODUCING.md' FIRST to understand how to run the capsule.",
    "Existing Results: If results already exist in 'results/', read them before re-running code.",
    "Docker Preferred: If REPRODUCING.md mentions Docker, use the Docker command rather than installing dependencies manually.",
    "Output Location: After running code, check 'results/' or the working directory for output files.",
    "Fallback: If local Python execution fails repeatedly, try using the Docker container described in REPRODUCING.md instead of giving up.",
]

# Hard mode: no instructions, must infer from Dockerfile/README
HARD_CONSTRAINTS = [
    "No Instructions: In Hard Mode, REPRODUCING.md is deleted. You must infer how to run the code.",
    "Discovery: Check 'code/README.md', 'Dockerfile', or 'code/run.sh' for clues.",
    "Dependencies: Check for 'requirements.txt' in './' or 'code/' and install dependencies.",
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
        # logger.info(f"MCP server info: {self.server_info}")
        
        # List tools
        tools_response = await self._send_request("tools/list", {})
        logger.debug(f"Tools list response: {json.dumps(tools_response, indent=2)}")
        
        if "result" in tools_response:
            self.tools = tools_response["result"].get("tools", [])
        
        # logger.info(f"MCP client connected with {len(self.tools)} tools")
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
        # logger.info("Disconnecting MCP client")
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
                # logger.info("MCP server terminated cleanly")
            except subprocess.TimeoutExpired:
                logger.warning("MCP server didn't terminate, killing")
                self.process.kill()
                self.process.wait()
        # logger.info("MCP client disconnected")


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
    
    # logger.info(f"Loaded {len(dataset)} tasks")
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
    
    # logger.info(f"Selected {len(selected_tasks)} tasks")
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
        # logger.info(f"Resetting workspace: {self._workspace_dir}")
        if os.path.exists(self._workspace_dir):
            shutil.rmtree(self._workspace_dir)
        os.makedirs(self._workspace_dir, exist_ok=True)
        logger.debug("Workspace reset complete")

    def _apply_difficulty_filters(self, domain: str) -> list[str]:
        """Apply difficulty filters to the staged capsule and return removed files/paths."""
        logger.info(f"Applying difficulty filters for domain: {domain}")

        removed_paths: list[str] = []

        env_dir = os.path.join(self._workspace_dir, "environment") # capsule root folder
        results_dir = os.path.join(env_dir, "results")  # capsule results folder

        print(f"[DEBUG] Calculated path for env_dir:     {env_dir}")
        print(f"[DEBUG] Calculated path for results_dir: {results_dir}")

        def _rel_to_workspace(abs_path: str) -> str:
            return os.path.relpath(abs_path, self._workspace_dir)

        # MEDIUM MODE: ONLY GIVEN DOCKERFILE + README INSTRUCTIONS
        if domain in ("corebench_medium", "corebench_hard"):
            if os.path.isdir(results_dir):
                logger.info(f"Removing results directory for {domain}")
                shutil.rmtree(results_dir)
                removed_paths.append(_rel_to_workspace(results_dir))

        # HARDMODE ONLY GIVEN README INSTRUCTIONS 
        if domain == "corebench_hard":
            paths_to_remove = [
                os.path.join(env_dir, "REPRODUCING.md"),
                os.path.join(env_dir, "environment"),
                os.path.join(env_dir, "code", "run.sh"),
                os.path.join(env_dir, "code", "run"),
            ]

            for abs_path in paths_to_remove:
                rel_path = _rel_to_workspace(abs_path)
                if os.path.exists(abs_path): # Check exists first to cover both file and dir
                    if os.path.isfile(abs_path):
                        print(f"[DEBUG] REMOVING FILE: {abs_path}")
                        logger.debug(f"Removing file: {abs_path}")
                        os.remove(abs_path)
                        removed_paths.append(rel_path)
                    elif os.path.isdir(abs_path):
                        print(f"[DEBUG] REMOVING DIR:  {abs_path}")
                        logger.debug(f"Removing directory: {abs_path}")
                        shutil.rmtree(abs_path)
                        removed_paths.append(rel_path)
                else:
                    print(f"[DEBUG] SKIPPING: {rel_path} (Not found)")
        
        print(f"[DEBUG] Final list of removed paths: {removed_paths}")
        logger.debug(f"Difficulty filters applied: {removed_paths}")
        return removed_paths

    @staticmethod
    def _snapshot_tree_paths(
        root_dir: str,
        *,
        skip_dir_names: Optional[set[str]] = None,
    ) -> tuple[set[str], set[str]]:
        """
        Snapshot file + directory paths under a root directory.

        Returns:
            (files, dirs) as sets of paths relative to root_dir.
        """
        files: set[str] = set()
        dirs: set[str] = set()

        if not root_dir or not os.path.exists(root_dir):
            return files, dirs

        skip = skip_dir_names or set()

        for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
            dirnames[:] = [d for d in dirnames if d not in skip]

            rel_dir = os.path.relpath(dirpath, root_dir)
            if rel_dir != ".":
                if os.sep != "/":
                    rel_dir = rel_dir.replace(os.sep, "/")
                dirs.add(rel_dir)

            for filename in filenames:
                rel_path = os.path.relpath(os.path.join(dirpath, filename), root_dir)
                if os.sep != "/":
                    rel_path = rel_path.replace(os.sep, "/")
                files.add(rel_path)

        return files, dirs
    
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
            # logger.info(f"MCP tools available: {tool_names}")
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
            # logger.debug(f"MCP tool result length: {len(result)} chars")
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
                "TensorFlow wheel not available for this Python/platform. "
                "Prefer running the capsule using its provided Docker image (see `REPRODUCING.md` or `Dockerfile`). "
                "If you're on an ARM host and the capsule's image/binaries are x86_64, add `--platform linux/amd64` to `docker run`."
            )

        if "exec format error" in text:
            return (
                "Architecture mismatch (e.g., x86_64 binary on an ARM host). "
                "Prefer using the capsule's Docker image. If you are on ARM and need to run x86_64, add `--platform linux/amd64` to `docker run`."
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
                    "You must instead inspect 'Dockerfile' or 'code/README.md' "
                    "to figure out the correct run commands."
                )
            
            # Generic file not found hint
            return (
                "The path is likely a directory or does not exist. "
                "CoreBench stages files under the current working directory (capsule root). "
                "Use `find . -name filename` to locate it first."
            )

        # Code Ocean container path accessed outside container
        if "/data" in text and ("cannot access" in text or "no such file or directory" in text):
            return (
                "The path `/data/` is a Code Ocean container path. In CoreBench, the capsule root is the current directory; "
                "use `data/` (and `results/`) as relative paths, or run inside the provided Docker container if required."
            )
        
        # Shell Limitations
        if "source: not found" in text:
            return "The shell doesn't support `source`. Use `. venv/bin/activate` instead."

        if "sudo: a password is required" in text:
            return "Root access is not available. Do not use sudo."
        
        if "tools/call" in text and "timed out" in text:
            return "The command likely ran longer than the MCP timeout; try splitting into smaller steps or increasing timeouts."

        # Common results parsing patterns
        if tool_name == "execute_bash":
            # HTML visualizations are common for figure questions.
            if "visualize_results.html" in text or (".html" in text and "results/" in text):
                return (
                    "This capsule includes an HTML visualization. If a question references a figure, "
                    "read the HTML with `inspect_file_as_text(file_path='results/visualize_results.html')` "
                    "and search within it for the needed labels/values. Some plots are embedded as base64 images "
                    "(`data:image/...;base64,`), which you can extract and decode via shell if needed."
                )

            # sklearn-style classification report (often includes the metric as a standalone number nearby)
            if (
                "precision" in text
                and "recall" in text
                and "f1-score" in text
                and "accuracy" in text
                and ("macro avg" in text or "weighted avg" in text)
            ):
                return (
                    "This output looks like a classification report. If asked for accuracy/error, "
                    "do NOT summarize—extract the exact numeric value from the results file. "
                    "Try `grep -n -i \"accuracy\" results/output | tail -n 20` and/or `tail -n 120 results/output`."
                )

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
        logger.info(f"🚀 STARTING COREBENCH EVALUATION")
        logger.info(f"=" * 80)
        
        start_time = time.time()
        
        # Generate unique run ID for correlating all task traces in this evaluation
        run_id = str(uuid.uuid4())[:8]
        logger.info(f"📋 Run ID: {run_id}")

        # Create a per-run trace folder: logs/traces/<YYYYMMDD>_<run_id>_<domain>/
        runday = datetime.now(timezone.utc).strftime('%Y%m%d')
        domain = req.config["domain"]
        base_trace_dir = Path(os.getenv("COREBENCH_TRACE_DIR") or os.getenv("COREBENCH_LOG_DIR") or "logs") / "traces"
        run_trace_dir = base_trace_dir / f"{runday}_{run_id}_{domain}"
        run_trace_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"🗂️  Run trace dir: {run_trace_dir}")

        domain = req.config["domain"]
        task_ids = req.config.get("task_ids", None)
        num_tasks = req.config.get("num_tasks", None)
        task_index = req.config.get("task_index", None)
        max_steps = req.config.get("max_steps", 200)
        
        # LLM-as-judge model configuration
        # By default, use the same model as the purple agent (from env vars)
        # Can be overridden in scenario.toml config
        default_judge_model = os.getenv("COREBENCH_TEXT_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
        judge_llm = req.config.get("judge_llm", default_judge_model)
        
        user_llm_args = req.config.get("user_llm_args", {})
        keep_traces = req.config.get("keep_traces", False)  # Whether to keep trace files after run
        use_cache = req.config.get("use_cache", False)  # Whether to cache capsules for reuse

        logger.info(f"Domain: {domain}")
        logger.info(f"Num tasks: {num_tasks}")
        if task_index is not None:
            logger.info(f"Task index: {task_index}")
        #logger.info(f"Max steps: {max_steps}")
        #logger.info(f"Keep traces: {keep_traces}")
        logger.info(f"Use cache: {use_cache}")
        use_cache = req.config.get("use_cache", False)  # Whether to cache capsules for reuse

        logger.info(f"📊 Domain: {domain} | Tasks: {num_tasks or 'all'} | Max Steps: {max_steps}")
        logger.info(f"🤖 Judge Model: {judge_llm}")
        
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
        # logger.info(f"🔗 Purple Agent: {agent_url}")

        # Initialize MCP client if enabled
        if use_mcp:
            # Ensure workspace exists before starting MCP server
            self._reset_workspace()

        # Get task IDs
        resolved_task_ids = get_task_ids(domain, task_ids, num_tasks, task_index)
        logger.info(f"📝 Running {len(resolved_task_ids)} tasks")
        logger.info("")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation of {len(resolved_task_ids)} tasks in {domain} domain")
        )

        # Collect all task evaluations and cost metadata
        task_evaluations: list[TaskEvaluation] = []
        task_cost_metadata: list[Optional[Dict[str, Any]]] = []
        track_restoration = domain in ("corebench_medium", "corebench_hard")

        try:
            for idx, task in enumerate(resolved_task_ids, 1):
                task_id = task["capsule_id"]
                logger.info(f"\n{'=' * 80}")
                logger.info(f"📦 TASK {idx}/{len(resolved_task_ids)}: {task_id}")
                logger.info(f"{'=' * 80}\n")
                
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
                    logger.info(f"\n{'─' * 80}")
                    logger.info(f"✅ Task {task_id} Complete:")
                    logger.info(f"   Accuracy: {task_evaluation.accuracy.accuracy:.1%} ({task_evaluation.accuracy.correct_answers}/{task_evaluation.accuracy.total_questions})")
                    # logger.info(f"   Faithfulness: {task_evaluation.faithfulness.score:.2f}/5")
                    logger.info(f"   Task Adherence: {task_evaluation.task_adherence.score:.2f}/1.0")
                    logger.info(f"   Steps: {task_evaluation.efficiency.steps_used}/{max_steps}")
                    logger.info(f"{'─' * 80}\n")
                    
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
                        AccuracyMetrics, TaskAdherenceMetrics, EfficiencyMetrics, _empty_accuracy_metrics
                    )
                    failed_eval = TaskEvaluation(
                        task_id=task_id,
                        domain=domain,
                        success=False,
                        accuracy=_empty_accuracy_metrics(),
                        reproducibility=None,
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
            if aggregate.mean_restoration_rate is not None:
                logger.info(f"♻️  Mean Reproducibility: {aggregate.mean_restoration_rate:.1%}")
            logger.info(f"⚡ Avg Steps: {aggregate.mean_steps:.1f} | Avg Tools: {aggregate.mean_tool_calls:.1f}")
            logger.info(f"⏱️  Total Time: {time_used:.1f}s")
            logger.info(f"{'=' * 80}\n")

            # Build result data for leaderboard
            result_data = {
                "domain": domain,
                "num_tasks": aggregate.num_tasks,
                "num_successful": aggregate.num_successful,
                "pass_rate": aggregate.pass_rate,
                "cost_efficiency": round(cost_efficiency, 4) if cost_efficiency is not None else None,  # Dollar cost per task

                # Accuracy metrics
                "mean_accuracy": aggregate.mean_accuracy,
                "mean_written_accuracy": aggregate.mean_written_accuracy,
                "mean_vision_accuracy": aggregate.mean_vision_accuracy,
                "mean_restoration_rate": aggregate.mean_restoration_rate,
                "mean_adherence": aggregate.mean_adherence,
                "mean_steps": aggregate.mean_steps,
                "mean_tool_calls": aggregate.mean_tool_calls,
                "mean_time": aggregate.mean_time,
                "total_time": time_used,
                "used_mcp": use_mcp,

                # Cost tracking
                "total_cost": total_cost,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "model_used": model_used,

                # Per-task breakdown
                "task_results": aggregate.task_results,
            }

            # Format task results for display
            task_results_str = "\n".join(
                f"  {tid}: {'✅' if info['success'] else '❌'} "
                f"(acc={info['accuracy']:.1%})"
                for tid, info in aggregate.task_results.items()
            )

            restoration_line = ""
            if aggregate.mean_restoration_rate is not None:
                restoration_line = f"Reproducibility: {aggregate.mean_restoration_rate:.1%}\n"

            cost_line = ""
            if cost_efficiency is not None:
                cost_line = f"  Cost Efficiency: ${cost_efficiency:.6f}/task (total: ${total_cost:.4f})\n"
                cost_line += f"  Tokens: {total_input_tokens:,} input, {total_output_tokens:,} output\n"

            summary = f"""\n⭐ CoreBench Benchmark Results ⭐
Domain: {domain}
Tasks: {aggregate.num_successful}/{aggregate.num_tasks} passed ({aggregate.pass_rate:.1%})

📊 Metrics:
  Accuracy: {aggregate.mean_accuracy:.1%} (written: {aggregate.mean_written_accuracy:.1%}, vision: {aggregate.mean_vision_accuracy:.1%})
  Task Adherence: {aggregate.mean_adherence:.2f}
  {restoration_line}Suspected Guessing: {aggregate.num_suspected_guessing}/{aggregate.num_tasks}

⚡ Efficiency:
  Avg Steps: {aggregate.mean_steps:.1f}
  Avg Tool Calls: {aggregate.mean_tool_calls:.1f}
  Total Time: {time_used:.1f}s
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
        3. Evaluate all metrics (accuracy, reproducibility, adherence, efficiency)
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
        task_start_time = time.time()

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
        trace_jsonl = trace_dir / f"trace_{run_id}_{trace_stamp}_{task_id}.jsonl"

        logger.info(f"📝 Writing trace to: {trace_jsonl}")

        try:
            trace = ExecutionTraceWriter(trace_jsonl, run_id=run_id or str(uuid.uuid4())[:8])
            trace.__enter__()  # Open the file
            trace.add(
                {
                    "type": "task_start",
                    "flow": "green -> trace",
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

        # Apply difficulty filters (and remember what we removed for restoration metrics).
        removed_by_difficulty_filters = self._apply_difficulty_filters(domain)

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

        # Snapshot capsule state (after difficulty filters) for reproducibility debugging.
        workspace_snapshot_skip_dirs = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "tool_outputs",  # written by evaluator for long tool outputs
        }
        baseline_files: set[str] = set()
        baseline_dirs: set[str] = set()
        if removed_by_difficulty_filters:
            baseline_files, baseline_dirs = self._snapshot_tree_paths(
                env_dir, skip_dir_names=workspace_snapshot_skip_dirs
            )

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
        final_answer_invalid_count = 0

        while not terminated and steps_used < max_steps: # Prevent infinite loops
            turn = steps_used + 1
            if trace:
                trace.add(
                    {
                        "type": "agent_prompt",
                        "flow": "green -> purple",
                        "turn": turn,
                        "message": next_message,
                        "new_conversation": is_first_message,
                    }
                )
            logger.debug(f"Sending to purple agent (first_message={is_first_message})")

            try:
                response = await self._tool_provider.talk_to_agent(
                    message=next_message,
                    url=agent_url,
                    new_conversation=is_first_message,
                )
                response_preview = response[:300] + "..." if len(response) > 300 else response
                logger.info(f"Purple agent response ({len(response)} chars): {response_preview}")
                logger.debug(f"Full response: {response}")
            except Exception as e:
                logger.error(f"Failed to communicate with purple agent: {e}")
                logger.debug(traceback.format_exc())
                raise

            is_first_message = False
            steps_used += 1
            if trace:
                trace.add(
                    {
                        "type": "agent_response",
                        "flow": "purple -> green",
                        "turn": steps_used,
                        "raw_response": response,
                    }
                )

            try:
                action, tool_result = await self._parse_and_execute_tools(response, use_mcp)
                protocol_errors = 0
                if trace:
                    trace.add(
                        {
                            "type": "action",
                            "flow": "purple -> green",
                            "turn": steps_used,
                            "name": action.name,
                            "arguments": action.arguments,
                        }
                    )
            except Exception as e:
                protocol_errors += 1
                logger.warning(f"Invalid agent response (protocol_errors={protocol_errors}/{max_protocol_errors}): {e}")
                logger.debug(traceback.format_exc())
                if trace:
                    trace.add(
                        {
                            "type": "protocol_error",
                            "flow": "purple -> green",
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

            # Log what the agent requested (tool call details)
            logger.info(f"📤 Agent requested: {action.name}")
            args_preview = json.dumps(action.arguments, indent=2, default=str)
            if len(args_preview) > 500:
                args_preview = args_preview[:500] + "\n   ... (truncated)"
            logger.info(f"   Arguments: {args_preview}")

            if tool_result is not None:
                result_preview = tool_result[:300] + "..." if len(tool_result) > 300 else tool_result
                logger.info(f"Tool result ({len(tool_result)} chars): {result_preview}")
                logger.debug(f"Full tool result: {tool_result}")
                tool_exec_index += 1
                
                # Extract key info for logging
                exit_code: Optional[int] = None
                if action.name == "execute_bash":
                    # Get command from arguments
                    cmd = action.arguments.get("command", "")
                    cmd_display = cmd[:80] + "..." if len(cmd) > 80 else cmd
                    
                    # Extract exit code
                    match = re.search(r"Exit Code:\s*(\d+)", tool_result)
                    if match:
                        exit_code = int(match.group(1))
                    
                    # Show command and exit status
                    status = f"✓ Exit {exit_code}" if exit_code == 0 else f"❌ Exit {exit_code}"
                    logger.info(f"🔧 execute_bash: {cmd_display} → {status}")
                    
                    # Show truncated output (first 3 lines)
                    output_lines = tool_result.split('\n')
                    if len(output_lines) > 5:
                        preview = '\n'.join(output_lines[:3])
                        #logger.info(f"   Output: {preview}\n   ... ({len(output_lines)} lines total)")
                    elif tool_result.strip():
                        logger.info(f"   Output: {tool_result[:200]}")
                        
                elif action.name == "inspect_file_as_text":
                    file_path = action.arguments.get("file_path", "")
                    file_size = len(tool_result)
                    num_lines = tool_result.count('\n')
                    logger.info(f"🔧 inspect_file_as_text: {file_path} → {file_size} bytes, {num_lines} lines")
                    
                elif action.name == "query_vision_language_model":
                    image_path = action.arguments.get("image_path", "")
                    question = action.arguments.get("question", "")
                    question_display = question[:60] + "..." if len(question) > 60 else question
                    logger.info(f"🔧 query_vision_language_model: {image_path}")
                    logger.info(f"   Question: {question_display}")
                    logger.info(f"   Answer: {tool_result[:250]}")
                    
                elif action.name == "web_search":
                    query = action.arguments.get("query", "")
                    logger.info(f"🔧 web_search: {query}")
                    logger.info(f"   Result: {tool_result[:250]}")
                    
                else:
                    # Generic tool logging
                    logger.info(f"🔧 {action.name}")
                    if tool_result and len(tool_result) > 100:
                        logger.info(f"   Result: {tool_result[:200]}...")
                    elif tool_result:
                        logger.info(f"   Result: {tool_result}")
                
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
                            "summary": self._summarize_tool_result(tool_result, head_lines=25, tail_lines=25),
                        }
                    )
                next_message = self._format_tool_result_for_agent(tool_name=action.name, tool_result=tool_result, index=tool_exec_index)
                continue

            if action.name == RESPOND_ACTION_NAME:
                # Enforce that the agent answers ALL questions before ending the task.
                reference = task["results"][0] if task.get("results") else {}
                required_questions = list(reference.keys())
                required_question_set = set(required_questions)

                answer = action.arguments.get("content")
                # Extract token/cost metadata from purple agent
                answer_metadata = action.arguments.get("_metadata", {})

                if answer is None:
                    # Be forgiving: some agents place answers directly under arguments instead of arguments.content.
                    legacy_content = {
                        k: v for k, v in action.arguments.items()
                        if k != "_metadata"
                    }
                    if legacy_content:
                        logger.warning(
                            "FINAL_ANSWER missing arguments.content; treating remaining arguments as answer content"
                        )
                        answer = legacy_content
                    else:
                        example_key = required_questions[0] if required_questions else "<question>"
                        final_answer_invalid_count += 1
                        if final_answer_invalid_count <= 3:
                            next_message = (
                                f"Invalid {RESPOND_ACTION_NAME}: missing `arguments.content`.\n\n"
                                f"Reply again with <json>{{\"name\": \"{RESPOND_ACTION_NAME}\", \"arguments\": {{\"content\": {{\"{example_key}\": <value>}}}}}}</json>.\n"
                            )
                            continue
                        logger.warning(
                            "Too many invalid FINAL_ANSWER submissions (missing content); proceeding with empty answer"
                        )
                        answer = {}
                        break

                if not isinstance(answer, dict):
                    example_key = required_questions[0] if required_questions else "<question>"
                    final_answer_invalid_count += 1
                    if final_answer_invalid_count <= 3:
                        next_message = (
                            f"Invalid {RESPOND_ACTION_NAME}: `arguments.content` must be a JSON object.\n\n"
                            f"Use the EXACT QUESTION TEXT as keys (copy from the prompt), not keys like \"status\" or \"output\".\n\n"
                            f"Reply again with <json>{{\"name\": \"{RESPOND_ACTION_NAME}\", \"arguments\": {{\"content\": {{\"{example_key}\": <value>}}}}}}</json>.\n"
                        )
                        continue
                    logger.warning(
                        "Too many invalid FINAL_ANSWER submissions (non-object content); proceeding with empty answer"
                    )
                    answer = {}
                    break

                # Normalize keys (strip whitespace) and keep only string/int keys.
                normalized_answer: dict[str, Any] = {}
                for key, value in answer.items():
                    if isinstance(key, str):
                        key_str = key.strip()
                    elif isinstance(key, int):
                        key_str = str(key)
                    else:
                        continue
                    normalized_answer[key_str] = value

                # Legacy compatibility: if the agent used numeric keys ("1", "2", ...), map them to question text by index.
                # (Only do this when no exact question-text keys were provided.)
                has_question_keys = any(k in required_question_set for k in normalized_answer.keys())
                numeric_keyed = {k: v for k, v in normalized_answer.items() if k.isdigit()}
                if not has_question_keys and numeric_keyed:
                    mapped: dict[str, Any] = {}
                    for k, v in numeric_keyed.items():
                        idx = int(k) - 1
                        if 0 <= idx < len(required_questions):
                            mapped[required_questions[idx]] = v
                    logger.warning(
                        "FINAL_ANSWER used numeric keys; mapping by index to question-text keys"
                    )
                    normalized_answer = mapped

                # Keep only required question keys.
                filtered_answer = {k: v for k, v in normalized_answer.items() if k in required_question_set}
                if filtered_answer != normalized_answer:
                    dropped = sorted(set(normalized_answer.keys()) - set(filtered_answer.keys()))
                    if dropped:
                        logger.warning(f"Dropping unexpected answer keys: {dropped}")

                missing_questions = [q for q in required_questions if q not in filtered_answer]
                if missing_questions:
                    final_answer_invalid_count += 1
                    if final_answer_invalid_count <= 3:
                        example_content = {q: "<value>" for q in required_questions}
                        next_message = (
                            f"Invalid {RESPOND_ACTION_NAME}: missing answers for {len(missing_questions)} required question(s).\n\n"
                            f"Missing question keys:\n- " + "\n- ".join(missing_questions) + "\n\n"
                            f"Reply again with {RESPOND_ACTION_NAME} including ALL required questions as keys.\n\n"
                            f"<json>\n"
                            f"{json.dumps({'name': RESPOND_ACTION_NAME, 'arguments': {'content': example_content}}, indent=2)}\n"
                            f"</json>\n"
                        )
                        continue
                    logger.warning(
                        "Too many invalid FINAL_ANSWER submissions (missing keys); proceeding with partial answer"
                    )
                    answer = filtered_answer
                    break

                # Type-check numeric questions so the agent cannot submit summaries like "output from the script".
                # This does NOT reveal the correct value; it only enforces the expected type.
                numeric_type_errors: list[str] = []
                for i, question_text in enumerate(required_questions, start=1):
                    expected_value = reference.get(question_text)
                    submitted_value = filtered_answer.get(question_text)

                    expects_numeric = isinstance(expected_value, (int, float, np.integer, np.floating))
                    if not expects_numeric:
                        continue

                    if isinstance(submitted_value, (int, float, np.integer, np.floating)):
                        continue
                    if isinstance(submitted_value, str):
                        cleaned = submitted_value.strip().replace("%", "")
                        try:
                            float(cleaned)
                            continue
                        except ValueError:
                            pass

                    preview = str(submitted_value)
                    if len(preview) > 80:
                        preview = preview[:80] + "..."
                    question_snippet = question_text if len(question_text) <= 80 else question_text[:77] + "..."
                    numeric_type_errors.append(f'Q{i} ("{question_snippet}") expects a number; got "{preview}"')

                if numeric_type_errors:
                    # Don't block task completion on type errors; log and continue.
                    logger.warning(
                        "FINAL_ANSWER contains non-numeric values for numeric questions:\n- %s",
                        "\n- ".join(numeric_type_errors),
                    )

                if filtered_answer != answer:
                    logger.warning(
                        "FINAL_ANSWER contained unexpected keys; keeping only required question-text keys"
                    )
                    answer = filtered_answer
                logger.info(f"FINAL ANSWER type: {type(answer)}")
                logger.info(f"FINAL ANSWER: {answer}")
                if answer_metadata:
                    logger.info(f"FINAL ANSWER metadata: {answer_metadata}")
                if trace:
                    trace.add(
                        {
                            "type": "final_answer",
                            "flow": "purple -> green",
                            "turn": steps_used,
                            "content": answer,
                            "metadata": answer_metadata,
                        }
                    )

                if removed_by_difficulty_filters:
                    try:
                        final_files, final_dirs = self._snapshot_tree_paths(
                            env_dir, skip_dir_names=workspace_snapshot_skip_dirs
                        )
                        created_files = sorted(final_files - baseline_files - {f for f in final_files if 'venv' in f.split(os.sep)} )
                        # ignore venv files in reproducability debug
                        created_dirs = sorted(final_dirs - baseline_dirs - {d for d in final_dirs if 'venv' in d.split(os.sep)})

                        logger.info(
                            "📁 Reproducibility debug: %d new files, %d new dirs under capsule root",
                            len(created_files),
                            len(created_dirs),
                        )

                        max_list = 60
                        if created_files:
                            logger.info("   New files:")
                            for rel_path in created_files[:max_list]:
                                logger.info("   + %s", rel_path)
                            if len(created_files) > max_list:
                                logger.info("   ... (%d more files)", len(created_files) - max_list)

                        if created_dirs:
                            logger.info("   New dirs:")
                            for rel_path in created_dirs[:max_list]:
                                logger.info("   + %s/", rel_path)
                            if len(created_dirs) > max_list:
                                logger.info("   ... (%d more dirs)", len(created_dirs) - max_list)

                        if trace:
                            max_trace = 200
                            trace.add(
                                {
                                    "type": "workspace_diff",
                                    "flow": "green -> trace",
                                    "turn": steps_used,
                                    "root": env_dir,
                                    "skip_dir_names": sorted(workspace_snapshot_skip_dirs),
                                    "created_files_count": len(created_files),
                                    "created_dirs_count": len(created_dirs),
                                    "created_files": created_files[:max_trace],
                                    "created_dirs": created_dirs[:max_trace],
                                    "truncated": (
                                        len(created_files) > max_trace or len(created_dirs) > max_trace
                                    ),
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Failed to snapshot capsule contents for reproducibility debug: {e}")
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
        task_end_time = time.time()
        task_time_seconds = task_end_time - task_start_time
        
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
        action_trace = trace.get_events("action") if trace else []
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
        for i, key in enumerate(expected_keys, 1):
            expected_val = gt_result[0].get(key, "<missing>")
            # Check if agent submitted by number or by key
            submitted_val = reported_result.get(str(i)) or reported_result.get(key, "<not submitted>")
            match = "✓" if key in [r.question for r in accuracy_metrics.question_results if r.correct] else "✗"
            # Truncate long values for display
            exp_str = str(expected_val)[:50] + "..." if len(str(expected_val)) > 50 else str(expected_val)
            sub_str = str(submitted_val)[:50] + "..." if len(str(submitted_val)) > 50 else str(submitted_val)
            key_display = f"Q{i}: {key[:35]}..." if len(key) > 35 else f"Q{i}: {key}"
            logger.info(f"   {match} {key_display}")
            logger.info(f"      Expected:  {exp_str}")
            logger.info(f"      Submitted: {sub_str}")

        
        # 2. REPRODUCIBILITY
        reproducibility_metrics: Optional[ReproducibilityMetrics] = None
        if removed_by_difficulty_filters:
            logger.info(f"2️⃣  Computing reproducibility...")
            reproducibility_metrics = evaluate_reproducibility(
                self._workspace_dir, 
                removed_by_difficulty_filters
            )
            logger.info(f"   ✓ Restored: {reproducibility_metrics.restored_count}/{reproducibility_metrics.targets_count} ({reproducibility_metrics.restoration_rate:.1%})")
        
        # Count command timeouts for task adherence context
        command_timeouts = sum(
            1 for r in tool_result_events
            if r.get("timed_out", False)
            or "timed out" in str(r.get("summary", "")).lower()
            or "timeout" in str(r.get("summary", "")).lower()
        )
        
        # 4. TASK ADHERENCE
        logger.info(f"4️⃣  Computing task adherence (LLM judge: {judge_llm})...")
        trace_event_callback = trace.add if trace else None
        adherence_metrics = await evaluate_task_adherence(
            domain=domain,
            task_prompt=task_prompt,
            steps_used=steps_used,
            tool_calls_count=tool_calls_count,
            protocol_errors=protocol_errors,
            submitted=reported_result,
            accuracy_result=accuracy_metrics,
            action_trace=action_trace,
            tool_calls=tool_call_events,
            tool_results=tool_result_events,
            workspace_dir=self._workspace_dir,
            trace_event_callback=trace_event_callback,
            judge_model=judge_llm,
            command_timeouts=command_timeouts,
        )
        logger.info(f"   ✓ Score: {adherence_metrics.score:.2f}/1.0")
        logger.info(f"   ✓ Navigation Quality: {adherence_metrics.navigation_quality}")
        if adherence_metrics.reasoning:
            logger.info(f"\n   💭 Judge Reasoning (Task Adherence):")
            for line in adherence_metrics.reasoning.split('\n'):
                if line.strip():
                    logger.info(f"      {line}")
            logger.info("")
        
        # 5. EFFICIENCY
        efficiency_metrics = compute_efficiency(
            steps_used=steps_used,
            max_steps=max_steps,
            tool_calls_count=tool_calls_count,
            time_seconds=task_time_seconds,
            protocol_errors=protocol_errors,
            command_timeouts=command_timeouts,  # For counting timeouts
        )
        timeout_info = f", {efficiency_metrics.command_timeouts} timeouts" if efficiency_metrics.command_timeouts else ""
        logger.info(f"5️⃣  Efficiency: {steps_used}/{max_steps} steps, {tool_calls_count} tools, {task_time_seconds:.1f}s{timeout_info}")
        
        # Build complete evaluation result
        task_success = accuracy_metrics.accuracy == 1.0
        
        evaluation = TaskEvaluation(
            task_id=task_id,
            domain=domain,
            success=task_success,
            accuracy=accuracy_metrics,
            reproducibility=reproducibility_metrics,
            task_adherence=adherence_metrics,
            efficiency=efficiency_metrics,
            submitted_answer=reported_result,
            ground_truth=gt_result,
            task_cost=cost_metadata.get("cost") if cost_metadata else None,
        )
        
        # Log evaluation summary
        eval_dict = evaluation.to_dict()
        logger.info(f"Evaluation summary: {json.dumps(eval_dict, indent=2, default=str)}")
        
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
        
        # Format questions with numbers for readability (keys should be the question text itself).
        questions = list(task["results"][0].keys())
        numbered_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        required_keys = questions
        required_keys_json = json.dumps(required_keys, indent=2, ensure_ascii=False)
        final_answer_example_content = {q: f"<ANSWER_{i+1}>" for i, q in enumerate(questions)}
        
        # Build domain-specific instruction and constraints
        if domain == "corebench_easy":
            objective = (
                f"Your goal is to answer questions about the output of scientific code. "
                f"Read through the files in `results/` to answer these questions:\n\n{numbered_questions}\n\n"
                f"**You should NOT run or execute any code.** All answers are available in the results directory."
            )
            domain_constraints = EASY_CONSTRAINTS
        elif domain == "corebench_medium":
            objective = (
                f"Your goal is to test the computational reproducibility of a scientific code capsule. "
                f"Specifically, you need to {task_prompt_text} to answer these questions:\n\n{numbered_questions}\n\n"
                f"Read `REPRODUCING.md` for instructions on how to run the capsule."
            )
            domain_constraints = MEDIUM_CONSTRAINTS
        elif domain == "corebench_hard":
            objective = (
                f"Your goal is to test the computational reproducibility of a scientific code capsule. "
                f"Specifically, you need to {task_prompt_text} to answer these questions:\n\n{numbered_questions}\n\n"
                f"No reproduction instructions are provided. You must figure out how to run the code yourself."
            )
            domain_constraints = HARD_CONSTRAINTS
        else:
            raise ValueError(f"Unknown domain: {domain}")

        all_constraints: list[str] = []
        if use_mcp:
            all_constraints.extend(TOOL_CONSTRAINTS)
        all_constraints.extend(COMMON_CONSTRAINTS)
        all_constraints.extend(domain_constraints)
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
    
    #logger.info(f"Starting CoreBench Evaluator")
    #logger.info(f"Host: {args.host}, Port: {args.port}")
    #logger.info(f"Log file: {log_file}")

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
