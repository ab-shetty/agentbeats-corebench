"""
CoreBench Purple Agent (Competition-Safe)
Enhanced with comprehensive logging to file

This agent:
1. Receives tasks from the green evaluator
2. Reasons about the task using an LLM
3. Emits structured JSON tool *intent*
4. NEVER executes tools
5. NEVER spawns MCP servers
"""

import argparse
import json
import re
import sys
import os
import logging
import traceback
import time
from typing import Any, Optional
from pathlib import Path
from datetime import datetime

import uvicorn
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

# Import shared logging
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_logging import setup_logging

# Logger will be initialized in main()
logger = logging.getLogger("purple_agent")

# Suppress verbose LiteLLM debug logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("a2a").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# Configuration:
# - Model specified via COREBENCH_TEXT_MODEL env var (uses litellm format: "provider/model")
# - Examples: "openai/gpt-4", "anthropic/claude-3-opus", "nebius/Qwen/Qwen3-Coder-30B-A3B-Instruct"
# - If COREBENCH_TEXT_API_BASE is set → self-hosted vLLM (prepends "openai/" if needed)
DEFAULT_MODEL = "openai/gpt-5-nano"
TEXT_API_BASE = (os.getenv("COREBENCH_TEXT_API_BASE") or "").strip()
TEXT_API_KEY = (os.getenv("COREBENCH_TEXT_API_KEY") or "").strip()

# Will be set in main() after CLI parsing
TEXT_MODEL: str = DEFAULT_MODEL


# =========================
# SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = """You are a helpful agent being evaluated for computational reproducibility.

You will receive:
1. A task description
2. A list of available tools (schemas only)
3. Instructions on what must be verified

IMPORTANT RULES:
- You must NEVER execute tools yourself
- You must NEVER assume tool results
- You may only REQUEST tool usage using JSON
- Respond with EXACTLY ONE action per turn

CRITICAL - ANSWER FORMAT:
- When reporting numeric values, copy them EXACTLY as they appear in the output
- Do NOT convert between decimal (0.96) and percentage (96%) formats
- If output shows "96.12", submit 96.12 (not 0.9612)
- Always preserve the exact precision and scale from the source

RESPONSE FORMAT:
You must output ONLY a JSON object wrapped in <json>...</json> tags. No extra text.

To use a tool, respond with:
<json>
{
    "name": "tool_name", 
    "arguments": {...}
}
</json>

To provide the final answer, respond with:
<json>
{
    "name": "FINAL_ANSWER", 
    "arguments": {"content": {"question text here": answer}}
}
</json>
"""


# =========================
# PURPLE AGENT EXECUTOR
# =========================

class CoreBenchPurpleAgent(AgentExecutor):
    """
    Purple agent that emits tool intent only.
    Tool execution is handled entirely by the green evaluator.
    """

    def __init__(self):
        self.ctx_id_to_messages: dict[str, list[dict]] = {}
        # Token tracking per context
        self.ctx_id_to_tokens: dict[str, dict[str, int]] = {}
        # Track task prompts and TTL for context GC (evaluator may reject FINAL_ANSWER and reprompt).
        self.ctx_id_to_task_prompt: dict[str, str] = {}
        # Track how often we've rejected FINAL_ANSWER for formatting/keys to avoid loopiness.
        self.ctx_id_to_final_answer_rejections: dict[str, int] = {}
        self.ctx_id_last_seen: dict[str, float] = {}
        self.ctx_id_done: set[str] = set()

    def _gc_contexts(self, *, now: float, keep_ctx: str, ttl_seconds: float = 900.0, max_contexts: int = 64) -> None:
        """Garbage-collect old conversation state to avoid unbounded growth."""
        # TTL cleanup
        for ctx_id, last_seen in list(self.ctx_id_last_seen.items()):
            if ctx_id == keep_ctx:
                continue
            if now - last_seen <= ttl_seconds:
                continue
            self.ctx_id_to_messages.pop(ctx_id, None)
            self.ctx_id_to_tokens.pop(ctx_id, None)
            self.ctx_id_to_task_prompt.pop(ctx_id, None)
            self.ctx_id_to_final_answer_rejections.pop(ctx_id, None)
            self.ctx_id_last_seen.pop(ctx_id, None)
            self.ctx_id_done.discard(ctx_id)

        # Max-contexts cleanup (drop oldest, excluding keep_ctx)
        if len(self.ctx_id_last_seen) <= max_contexts:
            return
        for ctx_id, _last_seen in sorted(self.ctx_id_last_seen.items(), key=lambda kv: kv[1]):
            if len(self.ctx_id_last_seen) <= max_contexts:
                break
            if ctx_id == keep_ctx:
                continue
            self.ctx_id_to_messages.pop(ctx_id, None)
            self.ctx_id_to_tokens.pop(ctx_id, None)
            self.ctx_id_to_task_prompt.pop(ctx_id, None)
            self.ctx_id_to_final_answer_rejections.pop(ctx_id, None)
            self.ctx_id_last_seen.pop(ctx_id, None)
            self.ctx_id_done.discard(ctx_id)


    @staticmethod
    def _try_parse_float(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().replace("%", "")
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_required_question_keys(task_prompt: str) -> list[str]:
        """
        Extract required question-text keys from the evaluator prompt.

        The evaluator includes a JSON list like:
        'REQUIRED: Return EXACTLY these keys ...:\n[ "question1", ... ]'
        """
        marker = "REQUIRED: Return EXACTLY these keys"
        idx = task_prompt.find(marker)
        if idx == -1:
            return []

        start = task_prompt.find("[", idx)
        if start == -1:
            return []

        try:
            parsed, _end = json.JSONDecoder().raw_decode(task_prompt[start:])
        except json.JSONDecodeError:
            return []

        if not isinstance(parsed, list):
            return []

        required: list[str] = []
        for item in parsed:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if s:
                required.append(s)
        return required

    def _completion_kwargs(self, messages: list[dict]) -> dict:
        """
        Build litellm.completion kwargs.

        If COREBENCH_TEXT_API_BASE is set → self-hosted vLLM with api_base/api_key
        Otherwise → pass model to litellm as-is (provider prefix already included)
        """
        if TEXT_API_BASE:
            # Self-hosted vLLM: OpenAI-compatible endpoint
            model = TEXT_MODEL
            if not model.startswith("openai/"):
                model = f"openai/{model}"
            return {
                "model": model,
                "messages": messages,
                "api_base": TEXT_API_BASE,
                "api_key": TEXT_API_KEY or "dummy",
            }
        else:
            # Pass model directly to litellm (provider prefix already in model name)
            return {"model": TEXT_MODEL, "messages": messages}

    def _track_tokens(self, context_id: str, response) -> None:
        """Track tokens from a completion response."""
        if context_id not in self.ctx_id_to_tokens:
            self.ctx_id_to_tokens[context_id] = {"input_tokens": 0, "output_tokens": 0}

        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
            completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
            self.ctx_id_to_tokens[context_id]["input_tokens"] += prompt_tokens
            self.ctx_id_to_tokens[context_id]["output_tokens"] += completion_tokens
            logger.debug(f"Token usage: input={prompt_tokens}, output={completion_tokens}, "
                        f"total_input={self.ctx_id_to_tokens[context_id]['input_tokens']}, "
                        f"total_output={self.ctx_id_to_tokens[context_id]['output_tokens']}")

    def _get_effective_model_name(self) -> str:
        """Get the effective model name being used."""
        if TEXT_API_BASE:
            # Local vLLM uses openai/ prefix
            model = TEXT_MODEL
            if not model.startswith("openai/"):
                model = f"openai/{model}"
            return model
        else:
            return TEXT_MODEL

    # -------------------------
    # Utility: parse tool call
    # -------------------------
    def _parse_tool_call(self, text: str) -> Optional[dict]:
        # Log what we're trying to parse (truncated for readability)
        logger.debug(f"Parsing response: {text[:500]}...")
        
        # Helper to fix common JSON escape issues (e.g., \| in grep patterns)
        def _fix_json_escapes(s: str) -> str:
            # Fix invalid escapes like \| \' etc. by double-escaping or removing
            # This handles patterns like: "grep -i 'error\|result'" which should be "grep -i 'error|result'"
            # or properly escaped as "grep -i 'error\\|result'"
            import re as re_inner
            # Replace \| with | (pipe doesn't need escaping in JSON strings)
            s = s.replace('\\|', '|')  # literal \| -> |
            # Replace other invalid single-char escapes
            s = re_inner.sub(r'\\([^"\\nrtbfu/])', r'\1', s)
            return s
        
        # Helper to normalize loosely formatted tool calls into the required envelope
        def _normalize_tool_call(obj: dict) -> Optional[dict]:
            if not isinstance(obj, dict):
                return None

            # Already wrapped as a tool call envelope.
            if "name" in obj and "arguments" in obj and isinstance(obj.get("arguments"), dict):
                tool_name = obj.get("name")
                arguments = obj.get("arguments") or {}

                # Special-case FINAL_ANSWER: evaluator expects arguments.content to be the answer dict.
                if tool_name == "FINAL_ANSWER":
                    content = arguments.get("content")
                    if isinstance(content, dict):
                        return obj

                    # Common model mistake: put answers directly under arguments instead of arguments.content.
                    extracted_content: dict[str, Any] = {}
                    for k, v in arguments.items():
                        if k in ("_metadata", "content"):
                            continue
                        if isinstance(k, int):
                            key_str = str(k)
                        elif isinstance(k, str):
                            key_str = k.strip()
                        else:
                            continue
                        if key_str:
                            extracted_content[key_str] = v

                    rebuilt_args: dict[str, Any] = {"content": extracted_content}
                    if "_metadata" in arguments:
                        rebuilt_args["_metadata"] = arguments["_metadata"]
                    return {"name": "FINAL_ANSWER", "arguments": rebuilt_args}

                return obj
            
            # Handle {"FINAL_ANSWER": value} format (model using FINAL_ANSWER as key instead of name)
            if "FINAL_ANSWER" in obj:
                val = obj["FINAL_ANSWER"]
                if isinstance(val, dict):
                    return {"name": "FINAL_ANSWER", "arguments": {"content": val}}
                # If it's just a status like true/"complete", return empty content
                return {"name": "FINAL_ANSWER", "arguments": {"content": {}}}
                
            # Map common "final answer" shorthands to the required envelope
            for key in ("final_answer", "final_result", "result", "answer", "answers", "content"):
                if isinstance(obj.get(key), dict):
                    return {"name": "FINAL_ANSWER", "arguments": {"content": obj[key]}}
                
            # If the JSON looks like a bare answers dict, assume it's a final answer payload.
            keys = list(obj.keys())
            if keys and all(isinstance(k, str) for k in keys):
                stripped = [k.strip() for k in keys]
                looks_numeric = all(k.isdigit() for k in stripped)
                looks_question_text = any(" " in k for k in stripped)
                if looks_numeric or looks_question_text:
                    return {"name": "FINAL_ANSWER", "arguments": {"content": obj}}
                
            return None
        
        # Try <json>...</json> tags first
        match = re.search(r"<json>\s*(.*?)\s*</json>", text, re.DOTALL)
        if match:
            json_str = match.group(1)
            logger.debug(f"Found <json> tags, content: {json_str[:200]}")
            try:
                parsed = json.loads(json_str)
                logger.debug(f"Parsed JSON successfully: {type(parsed)}, keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'N/A'}")

                normalized = _normalize_tool_call(parsed) if isinstance(parsed, dict) else None
                if normalized:
                    logger.debug(f"Normalized to: {normalized.get('name')}")
                    return normalized
                    
                logger.warning(f"JSON parsed but not recognized as tool call: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")
            except json.JSONDecodeError as e:
                # Try fixing missing closing brace (common Llama4 issue)
                try:
                    fixed = json_str.rstrip()
                    # Count braces to see if we're missing closing ones
                    open_braces = fixed.count('{') - fixed.count('}')
                    if open_braces > 0:
                        fixed = fixed + ('}' * open_braces)
                        parsed = json.loads(fixed)
                        normalized = _normalize_tool_call(parsed) if isinstance(parsed, dict) else None
                        if normalized:
                            return normalized
                except json.JSONDecodeError:
                    pass
                    
                # Try fixing common escape issues
                try:
                    fixed = _fix_json_escapes(json_str)
                    parsed = json.loads(fixed)
                    return _normalize_tool_call(parsed) if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    pass
                logger.warning(f"Failed to parse JSON from <json> tags: {e}")
                logger.debug(f"JSON string was: {json_str[:500]}")
                return None

        # Try markdown code blocks
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                parsed = json.loads(json_str)
                return _normalize_tool_call(parsed) if isinstance(parsed, dict) else None
            except json.JSONDecodeError as e:
                try:
                    parsed = json.loads(_fix_json_escapes(json_str))
                    return _normalize_tool_call(parsed) if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    pass
                logger.warning(f"Failed to parse JSON from code block: {e}")
                return None
        
        # Try plain ``` code blocks (no language specified)
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                parsed = json.loads(json_str)
                return _normalize_tool_call(parsed) if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                try:
                    parsed = json.loads(_fix_json_escapes(json_str))
                    if isinstance(parsed, dict) and "name" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
        
        # Try <tool_call>...</tool_call> (Llama-style)
        match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
            try:
                parsed = json.loads(json_str)
                return _normalize_tool_call(parsed) if isinstance(parsed, dict) else None
            except json.JSONDecodeError as e:
                try:
                    return json.loads(_fix_json_escapes(json_str))
                except json.JSONDecodeError:
                    pass
                logger.warning(f"Failed to parse JSON from <tool_call> tags: {e}")
        
        # Try <function_call>...</function_call>
        match = re.search(r"<function_call>\s*(.*?)\s*</function_call>", text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try parsing entire response as JSON
        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict):
                return _normalize_tool_call(parsed) or parsed
        except json.JSONDecodeError:
            pass
        
        # Try to find a JSON object that looks like a tool call {"name": ..., "arguments": ...}
        # Use a more precise pattern to find valid JSON objects
        json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Last resort: find any balanced JSON object
        # Start from each { and try to find matching }
        for i, char in enumerate(text):
            if char == '{':
                depth = 0
                for j, c in enumerate(text[i:], i):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                candidate = text[i:j+1]
                                parsed = json.loads(candidate)
                                if isinstance(parsed, dict) and "name" in parsed:
                                    logger.info(f"Found tool call via balanced braces extraction")
                                    return parsed
                            except json.JSONDecodeError:
                                pass
                            break

        logger.warning(f"No JSON found in response. First 200 chars: {text[:200]}")
        return None

    # -------------------------
    # Main execution loop
    # -------------------------
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        now = time.time()
        self.ctx_id_last_seen[context.context_id] = now
        self._gc_contexts(now=now, keep_ctx=context.context_id)

        logger.info(f"=" * 80)
        logger.info(f"PURPLE AGENT - NEW REQUEST")
        logger.info(f"=" * 80)
        # logger.info(f"Context ID: {context.context_id}")
        # logger.info(f"Input length: {len(user_input)} chars")
        logger.debug(f"Full input: {user_input}")

        # Initialize purple agent conversation
        if context.context_id not in self.ctx_id_to_messages:
            logger.info("Initializing new conversation with system prompt")
            self.ctx_id_to_messages[context.context_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
        # Capture the initial task prompt once per context_id (used for answer validation).
        if context.context_id not in self.ctx_id_to_task_prompt:
            self.ctx_id_to_task_prompt[context.context_id] = user_input
            self.ctx_id_to_final_answer_rejections[context.context_id] = 0

        messages = self.ctx_id_to_messages[context.context_id]
        messages.append({"role": "user", "content": user_input})
        logger.debug(f"Conversation has {len(messages)} messages")

        max_turns = 10

        for turn in range(max_turns):
            logger.info(f"--- Turn {turn + 1}/{max_turns} ---")
            try:
                # logger.debug(f"Sending {len(messages)} messages to LLM")
                logger.debug(f"Messages: {json.dumps(messages, indent=2)}")
                
                # model can take values such as:
                # "gemini/gemini-3-pro-preview"
                # "openai/gpt-5-mini"
                # "nebius/Qwen/Qwen3-Coder-30B-A3B-Instruct"
                logger.info("Calling LLM:")
                response = completion(**self._completion_kwargs(messages))
                self._track_tokens(context.context_id, response)

                # Log the response details
                logger.debug(f"LLM Response object: {response}")
                logger.debug(f"Response choices: {response.choices if hasattr(response, 'choices') else 'No choices'}")
                
                # Check if response has the expected JSON structure
                if not hasattr(response, 'choices') or not response.choices:
                    logger.error("LLM response has no choices!")
                    logger.error(f"Response type: {type(response)}")
                    logger.error(f"Response dict: {response.__dict__ if hasattr(response, '__dict__') else 'No __dict__'}")
                    raise ValueError("LLM response missing choices")
                
                if not response.choices[0]:
                    logger.error("First choice is None or empty!")
                    raise ValueError("First choice is empty")
                
                if not hasattr(response.choices[0], 'message'):
                    logger.error(f"First choice has no message! Choice: {response.choices[0]}")
                    raise ValueError("Choice has no message attribute")
                
                if not hasattr(response.choices[0].message, 'content'):
                    logger.error(f"Message has no content! Message: {response.choices[0].message}")
                    raise ValueError("Message has no content attribute")

                message = response.choices[0].message
                assistant_content = message.content

                # Handle reasoning-only responses (common with some models)
                if assistant_content is None:
                    reasoning_content = getattr(message, 'reasoning_content', None)
                    if reasoning_content:
                        logger.warning("Got reasoning but no content, prompting model to provide actual tool call")
                        logger.debug(f"Reasoning: {reasoning_content}")
                        
                        # Add reasoning to history for context
                        messages.append({
                            "role": "assistant",
                            "content": f"[Internal reasoning: {reasoning_content}]"
                        })
                        
                        # Prompt for actual action
                        messages.append({
                            "role": "user",
                            "content": (
                                "You've analyzed what needs to be done. Now provide the actual tool call "
                                "in the required JSON format:\n\n"
                                "<json>\n"
                                "{\n"
                                '  "name": "tool_name",\n'
                                '  "arguments": {...}\n'
                                "}\n"
                                "</json>"
                            )
                        })
                        
                        # Call LLM again
                        logger.info("Calling LLM again for actual tool call")
                        response = completion(**self._completion_kwargs(messages))
                        self._track_tokens(context.context_id, response)

                        assistant_content = response.choices[0].message.content
                        if assistant_content is None:
                            logger.error("Follow-up request also returned no content!")
                            raise ValueError("Both initial and follow-up responses had no content")
                    else:
                        logger.error("Assistant content is None and no reasoning_content!")
                        logger.error(f"Message: {message}")
                        raise ValueError("Assistant content is None")
                
                response_preview = assistant_content[:300] + "..." if len(assistant_content) > 300 else assistant_content
                logger.debug(f"LLM response ({len(assistant_content)} chars): {response_preview}")

                messages.append(
                    {"role": "assistant", "content": assistant_content}
                )

                # Check for structured JSON tool intent
                tool_call = self._parse_tool_call(assistant_content)

                if (
                    tool_call is None                       # parsing failed
                    or not isinstance(tool_call, dict)      # parsed but wrong type
                    or "name" not in tool_call              # missing required field
                    or "arguments" not in tool_call         # missing required field
                    or not isinstance(tool_call.get("arguments"), dict)  # arguments not a dict
                ):
                    # Log what we received to help debug format issues
                    logger.warning("No valid structured tool call found; reprompting for JSON-only output")
                    logger.warning(f"   LLM said: {assistant_content[:200]}{'...' if len(assistant_content) > 200 else ''}")
                    messages.append({
                        "role": "user",
                        "content": """Your previous response was NOT a valid tool call.

Reply with ONLY a JSON object wrapped in <json>...</json> tags. No extra text.

Example:
<json>
{"name": "execute_bash", "arguments": {"command": "ls"}}
</json>"""
                    })
                    continue

                # Extra enforcement for FINAL_ANSWER: ensure arguments.content exists and is non-empty.
                if tool_call.get("name") == "FINAL_ANSWER":
                    arguments = tool_call.get("arguments") or {}

                    # If the model put answers directly under arguments, coerce into arguments.content.
                    if "content" not in arguments and isinstance(arguments, dict):
                        coerced_content: dict[str, Any] = {}
                        for k, v in arguments.items():
                            if k == "_metadata":
                                continue
                            if isinstance(k, int):
                                key_str = str(k)
                            elif isinstance(k, str):
                                key_str = k.strip()
                            else:
                                continue
                            if key_str:
                                coerced_content[key_str] = v
                        tool_call["arguments"] = {
                            "content": coerced_content,
                            **({"_metadata": arguments["_metadata"]} if "_metadata" in arguments else {}),
                        }
                        arguments = tool_call["arguments"]

                    task_prompt = self.ctx_id_to_task_prompt.get(context.context_id, "")
                    required_questions = self._extract_required_question_keys(task_prompt)
                    required_question_set = set(required_questions)

                    raw_content = arguments.get("content")
                    if not isinstance(raw_content, dict):
                        raw_content = {}

                    # Normalize content keys.
                    content: dict[str, Any] = {}
                    for k, v in raw_content.items():
                        if isinstance(k, int):
                            key_str = str(k)
                        elif isinstance(k, str):
                            key_str = k.strip()
                        else:
                            continue
                        if key_str:
                            content[key_str] = v

                    # Legacy compatibility: map numeric keys ("1") or numbered keys ("1. ...") by index.
                    if required_questions and content:
                        mapped: dict[str, Any] = {}
                        for k, v in content.items():
                            if k in required_question_set:
                                mapped[k] = v
                                continue

                            if k.isdigit():
                                idx = int(k) - 1
                                if 0 <= idx < len(required_questions):
                                    mapped[required_questions[idx]] = v
                                continue

                            m = re.match(r"^(?:q)?(\d+)\s*[\.\):]\s*(.*)$", k, flags=re.IGNORECASE)
                            if m:
                                idx = int(m.group(1)) - 1
                                if 0 <= idx < len(required_questions):
                                    mapped[required_questions[idx]] = v
                                continue

                            mapped[k] = v
                        content = mapped

                    # Filter out unexpected keys to reduce evaluator warnings.
                    filtered_content = (
                        {k: v for k, v in content.items() if k in required_question_set}
                        if required_questions
                        else content
                    )

                    max_rejections = 2
                    rejection_count = self.ctx_id_to_final_answer_rejections.get(context.context_id, 0)

                    def _reject_final_answer(*, reason: str, prompt: str) -> bool:
                        nonlocal rejection_count
                        if rejection_count >= max_rejections:
                            logger.warning(
                                "FINAL_ANSWER enforcement: too many rejections (%d); allowing submission (%s)",
                                rejection_count,
                                reason,
                            )
                            return False
                        rejection_count += 1
                        self.ctx_id_to_final_answer_rejections[context.context_id] = rejection_count
                        messages.append({"role": "user", "content": prompt})
                        return True

                    # Require at least one tool result before answering (prevents blind guessing).
                    saw_tool_result = any(
                        m.get("role") == "user"
                        and isinstance(m.get("content"), str)
                        and "Tool execution result" in m.get("content")
                        for m in messages
                    )

                    # Reject obvious placeholder answers.
                    def _looks_like_placeholder(v: Any) -> bool:
                        if not isinstance(v, str):
                            return False
                        s = v.strip()
                        if not s:
                            return True
                        if re.fullmatch(r"value\\d+", s, flags=re.IGNORECASE):
                            return True
                        if "<" in s and ">" in s:
                            return True
                        return False

                    if not saw_tool_result:
                        if _reject_final_answer(
                            reason="no_tool_result",
                            prompt=(
                                "Do NOT guess. You must request tools and read outputs before FINAL_ANSWER.\n\n"
                                "Start with a tool call like:\n"
                                "<json>\n"
                                "{\"name\":\"execute_bash\",\"arguments\":{\"command\":\"ls -R\"}}\n"
                                "</json>"
                            ),
                        ):
                            continue

                    if not filtered_content:
                        example_key = required_questions[0] if required_questions else "question text here"
                        if _reject_final_answer(
                            reason="empty_content",
                            prompt=(
                                "Your FINAL_ANSWER is invalid for this benchmark.\n\n"
                                "Reply with:\n"
                                "<json>\n"
                                f"{{\"name\":\"FINAL_ANSWER\",\"arguments\":{{\"content\":{{\"{example_key}\":<ACTUAL_VALUE>}}}}}}\n"
                                "</json>\n\n"
                                "Rules:\n"
                                "- `arguments.content` must be a NON-EMPTY JSON object\n"
                                "- Keys must be the EXACT question text strings from the prompt\n"
                                "- Values must be the extracted answers (no placeholders like \"value1\")"
                            ),
                        ):
                            continue

                    if required_questions:
                        missing_questions = [q for q in required_questions if q not in filtered_content]
                        if missing_questions:
                            missing_preview = missing_questions[:12]
                            more = "" if len(missing_questions) <= 12 else f"\n... ({len(missing_questions) - 12} more)"
                            if _reject_final_answer(
                                reason="missing_keys",
                                prompt=(
                                    "Your FINAL_ANSWER is missing required question keys.\n\n"
                                    "Missing:\n- " + "\n- ".join(missing_preview) + more + "\n\n"
                                    "Do NOT submit FINAL_ANSWER until you have answers for ALL questions.\n"
                                    "Use tools to extract the missing values, then reply with FINAL_ANSWER using the EXACT question text as keys."
                                ),
                            ):
                                continue

                    if all(_looks_like_placeholder(v) for v in filtered_content.values()):
                        if _reject_final_answer(
                            reason="placeholders",
                            prompt=(
                                "Your FINAL_ANSWER contains placeholder values.\n\n"
                                "Go back, use tools to locate the exact values in files, then reply with FINAL_ANSWER "
                                "containing the real extracted numbers/labels only."
                            ),
                        ):
                            continue

                    tool_call["arguments"]["content"] = filtered_content

                # Add token metadata for FINAL_ANSWER
                if tool_call.get("name") == "FINAL_ANSWER":
                    tokens = self.ctx_id_to_tokens.get(context.context_id, {"input_tokens": 0, "output_tokens": 0})
                    tool_call["arguments"]["_metadata"] = {
                        "model": self._get_effective_model_name(),
                        "input_tokens": tokens["input_tokens"],
                        "output_tokens": tokens["output_tokens"],
                    }
                    logger.info(f"Adding token metadata to FINAL_ANSWER: {tool_call['arguments']['_metadata']}")

                # Always forward tool intent verbatim
                formatted = "<json>\n" + json.dumps(tool_call, indent=2) + "\n</json>"
                logger.info(f"Sending tool call: {tool_call.get('name', 'unknown')}")

                await event_queue.enqueue_event(
                    new_agent_text_message(
                        formatted,
                        context_id=context.context_id,
                    )
                )
                # logger.info("Response sent successfully")
                
                # Mark completion but keep state briefly (evaluator may reject and reprompt).
                if tool_call.get("name") == "FINAL_ANSWER":
                    self.ctx_id_done.add(context.context_id)
                return

            except Exception as e:
                logger.error(f"Purple agent error on turn {turn}: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.debug(traceback.format_exc())
                
                await event_queue.enqueue_event(
                    new_agent_text_message(
                        "<json>\n"
                        '{"name": "FINAL_ANSWER", "arguments": {"content": {}}}\n'
                        "</json>",
                        context_id=context.context_id,
                    )
                )
                self.ctx_id_done.add(context.context_id)
                return

        # If internal retries are exhausted without producing valid JSON, send a complaint fallback.
        logger.error("Exhausted max turns without producing a valid JSON tool call so returning empty FINAL_ANSWER")
        await event_queue.enqueue_event(
            new_agent_text_message(
                "<json>\n"
                '{"name": "FINAL_ANSWER", "arguments": {"content": {}}}\n'
                "</json>",
                context_id=context.context_id,
            )
        )
        self.ctx_id_done.add(context.context_id)
        return

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


# =========================
# AGENT CARD
# =========================

def prepare_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="corebench_reasoning",
        name="CoreBench Task Reasoning",
        description="Reasoning-only agent that emits structured tool intent",
        tags=["corebench", "benchmark", "reasoning"],
        examples=[],
    )

    return AgentCard(
        name="corebench_purple_agent",
        description="Purple agent for CoreBench evaluation (no tool execution)",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


# =========================
# SERVER ENTRYPOINT
# =========================

def main():
    global TEXT_MODEL

    parser = argparse.ArgumentParser("Run CoreBench Purple Agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9019)
    parser.add_argument("--card-url")
    args = parser.parse_args()

    # Resolve model: ENV > default
    TEXT_MODEL = os.getenv("COREBENCH_TEXT_MODEL") or DEFAULT_MODEL

    # Setup shared logging
    log_file = setup_logging("purple_agent")

    logger.info(f"Starting CoreBench Purple Agent")
    logger.info(f"Host: {args.host}, Port: {args.port}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Model: {TEXT_MODEL}")
    if TEXT_API_BASE:
        logger.info(f"Using self-hosted vLLM at: {TEXT_API_BASE}")

    card_url = args.card_url or f"http://{args.host}:{args.port}/"
    card = prepare_agent_card(card_url)

    request_handler = DefaultRequestHandler(
        agent_executor=CoreBenchPurpleAgent(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    # logger.info("Server starting...")
    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
        log_level="warning"
    )


if __name__ == "__main__":
    main()
