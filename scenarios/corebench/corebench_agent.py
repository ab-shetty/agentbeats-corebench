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
logging.getLogger("a2a").setLevel(logging.INFO)


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

To use a tool, respond with:
<json>
{"name": "tool_name", "arguments": {...}}
</json>

To respond directly:
<json>
{"name": "respond", "arguments": {"content": "..."}}
</json>

You may only do ONE action per turn.
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

    # -------------------------
    # Utility: parse tool call
    # -------------------------
    def _parse_tool_call(self, text: str) -> Optional[dict]:
        
        match = re.search(r"<json>\s*(.*?)\s*</json>", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from <json> tags: {e}")
                logger.debug(f"JSON string was: {match.group(1)[:500]}")
                return None

        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from code block: {e}")
                return None

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

        # Initialize conversation
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
                response = completion(
                    model="openai/gpt-5-mini", #"nebius/openai/gpt-oss-120b",
                    messages=messages,
                )

                assistant_content = response.choices[0].message.content
                logger.info(f"LLM turn {turn}: {assistant_content[:200]}")

                messages.append(
                    {"role": "assistant", "content": assistant_content}
                )

                # Check for structured JSON tool intent
                tool_call = self._parse_tool_call(assistant_content)

                if tool_call is None:
                    # Plain text response (allowed but discouraged)
                    logger.warning("No structured tool call found, sending plain text response")
                    await event_queue.enqueue_event(
                        new_agent_text_message(
                            assistant_content,
                            context_id=context.context_id,
                        )
                    )
                    return

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
                return

            except Exception as e:
                logger.error(f"Purple agent error on turn {turn}: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.debug(traceback.format_exc())
                
                await event_queue.enqueue_event(
                    new_agent_text_message(
                        "<json>\n"
                        '{"name": "respond", "arguments": {"content": "I encountered an error."}}\n'
                        "</json>",
                        context_id=context.context_id,
                    )
                )
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
    parser = argparse.ArgumentParser("Run CoreBench Purple Agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9019)
    parser.add_argument("--card-url")
    args = parser.parse_args()

    # Setup shared logging
    log_file = setup_logging("purple_agent")
    
    logger.info(f"Starting CoreBench Purple Agent")
    logger.info(f"Host: {args.host}, Port: {args.port}")
    logger.info(f"Log file: {log_file}")

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