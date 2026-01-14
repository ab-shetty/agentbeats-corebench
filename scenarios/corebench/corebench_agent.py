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
from typing import Optional
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
        # Try <json>...</json> tags first
        match = re.search(r"<json>\s*(.*?)\s*</json>", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from <json> tags: {e}")
                logger.debug(f"JSON string was: {match.group(1)[:500]}")
                return None

        # Try markdown code blocks
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from code block: {e}")
                return None

        # Try parsing entire response as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Last resort: search for JSON object in text
        for pattern in [r"\{.*?\}", r"\{.*\}"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue

        logger.warning("No JSON found in response")
        return None

    # -------------------------
    # Main execution loop
    # -------------------------
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        logger.info(f"=" * 80)
        logger.info(f"PURPLE AGENT - NEW REQUEST")
        logger.info(f"=" * 80)
        logger.info(f"Context ID: {context.context_id}")
        logger.info(f"Input length: {len(user_input)} chars")
        logger.debug(f"Full input: {user_input}")

        # Initialize purple agent conversation
        if context.context_id not in self.ctx_id_to_messages:
            logger.info("Initializing new conversation with system prompt")
            self.ctx_id_to_messages[context.context_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

        messages = self.ctx_id_to_messages[context.context_id]
        messages.append({"role": "user", "content": user_input})
        logger.debug(f"Conversation has {len(messages)} messages")

        max_turns = 10

        for turn in range(max_turns):
            logger.info(f"--- Turn {turn + 1}/{max_turns} ---")
            try:
                logger.debug(f"Sending {len(messages)} messages to LLM")
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
                    logger.warning("No valid structured tool call found; reprompting for JSON-only output")
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
                logger.info("Response sent successfully")
                
                # Cleanup: Free memory after conversation ends (FINAL_ANSWER means task complete).
                # Without this, ctx_id_to_messages grows indefinitely across capsules.
                if tool_call.get("name") == "FINAL_ANSWER":
                    self.ctx_id_to_messages.pop(context.context_id, None)
                    self.ctx_id_to_tokens.pop(context.context_id, None)
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
                # Cleanup: Free memory after error fallback
                self.ctx_id_to_messages.pop(context.context_id, None)
                self.ctx_id_to_tokens.pop(context.context_id, None)
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
        # Cleanup: Free memory after max_turns fallback
        self.ctx_id_to_messages.pop(context.context_id, None)
        self.ctx_id_to_tokens.pop(context.context_id, None)
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

    logger.info("Server starting...")
    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
        log_level="info"
    )


if __name__ == "__main__":
    main()
