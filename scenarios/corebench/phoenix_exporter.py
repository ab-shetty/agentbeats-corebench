"""
Optional Arize AX exporter for CoreBench traces.

To enable: Set both environment variables:
  - ARIZE_API_KEY: Your Arize API key (from Settings → API keys)
  - ARIZE_SPACE_ID: Your Arize space/workspace ID (from Settings → API keys)

To remove: Delete this file - the evaluator will gracefully handle its absence.

Converts ExecutionTraceWriter events (JSONL format) to OpenTelemetry spans
and exports them to Arize AX for visualization.

Docs: https://arize.com/docs/ax/observe/tracing/setup/manual-instrumentation
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("phoenix_exporter")

# Check if Arize export is enabled - requires both API key and Space ID
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY", "").strip()
ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID", "").strip()
ARIZE_PROJECT = os.getenv("ARIZE_PROJECT", "corebench").strip()

# Max attribute size (Arize/OTel typically limits to ~32KB, we use 16KB to be safe)
MAX_ATTR_SIZE = 16000

_ENABLED = bool(ARIZE_API_KEY and ARIZE_SPACE_ID)
_tracer = None
_tracer_provider = None
_initialized = False


def _truncate(text: str, max_len: int = MAX_ATTR_SIZE) -> str:
    """Truncate text to max length, adding indicator if truncated."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 50] + f"\n\n... [truncated, {len(text)} total chars]"


def _initialize():
    """Lazy initialization of OpenTelemetry components using Arize's helper."""
    global _tracer, _tracer_provider, _initialized

    if _initialized:
        return
    _initialized = True

    if not _ENABLED:
        if ARIZE_API_KEY and not ARIZE_SPACE_ID:
            logger.warning("Arize export disabled: ARIZE_SPACE_ID not set")
        elif ARIZE_SPACE_ID and not ARIZE_API_KEY:
            logger.warning("Arize export disabled: ARIZE_API_KEY not set")
        else:
            logger.debug("Arize export disabled (ARIZE_API_KEY and ARIZE_SPACE_ID not set)")
        return

    try:
        from arize.otel import register

        # Use Arize's official registration helper
        _tracer_provider = register(
            space_id=ARIZE_SPACE_ID,
            api_key=ARIZE_API_KEY,
            project_name=ARIZE_PROJECT,
        )

        _tracer = _tracer_provider.get_tracer("corebench.evaluator", "1.0.0")
        logger.info(f"Arize AX exporter initialized (project: {ARIZE_PROJECT})")

    except ImportError as e:
        logger.warning(f"Arize export disabled - missing dependencies: {e}")
        logger.warning("Run: uv add arize-otel")
    except Exception as e:
        logger.error(f"Failed to initialize Arize exporter: {e}")


def export_trace(
    events: list[dict[str, Any]],
    task_id: str,
    run_id: str,
    domain: str = "",
) -> bool:
    """
    Export trace events to Arize AX.

    Args:
        events: List of trace events from ExecutionTraceWriter.get_events()
        task_id: The capsule/task ID (e.g., "capsule-2804717")
        run_id: The evaluation run ID
        domain: The benchmark domain (e.g., "corebench_hard")

    Returns:
        True if export was attempted, False if disabled/unavailable
    """
    _initialize()

    if not _ENABLED or _tracer is None:
        return False

    try:
        from opentelemetry.trace import Status, StatusCode
        from opentelemetry import context as otel_context
        from opentelemetry.trace import set_span_in_context

        # Extract metadata from events
        task_start = next((e for e in events if e.get("type") == "task_start"), {})
        evaluation = next((e for e in events if e.get("type") == "evaluation"), {})
        final_answer = next((e for e in events if e.get("type") == "final_answer"), {})
        llm_judge_output = next((e for e in events if e.get("type") == "llm_judge_output"), {})
        llm_judge_input = next((e for e in events if e.get("type") == "llm_judge_input"), {})
        eval_data = evaluation.get("evaluation", {})

        # Create parent span for the entire task
        with _tracer.start_as_current_span(
            f"task:{task_id}",
            attributes={
                "corebench.run_id": run_id,
                "corebench.task_id": task_id,
                "corebench.domain": domain or task_start.get("domain", ""),
                "corebench.host": task_start.get("host", ""),
                "corebench.questions": json.dumps(task_start.get("questions", [])),
            }
        ) as parent_span:

            # Track tool calls for pairing with results
            # Key: turn number, Value: (span, arguments_dict)
            turn_to_tool: dict[int, tuple[Any, dict]] = {}

            for event in events:
                etype = event.get("type")
                turn = event.get("turn", 0)

                # Skip agent_prompt and agent_response - tool spans already capture
                # the conversation flow via input.value (args) and output.value (results)

                if etype == "tool_call":
                    tool_name = event.get("tool", "unknown")
                    arguments = event.get("arguments", {})
                    arguments_str = json.dumps(arguments, indent=2)

                    # Create a child span for the tool call, linked to parent
                    parent_ctx = set_span_in_context(parent_span)
                    tool_span = _tracer.start_span(
                        f"tool:{tool_name}",
                        context=parent_ctx,
                        attributes={
                            "tool.name": tool_name,
                            "turn": turn,
                            # OpenInference semantic convention for input
                            "input.value": _truncate(arguments_str),
                        }
                    )
                    turn_to_tool[turn] = (tool_span, arguments)

                elif etype == "tool_result":
                    tool_name = event.get("tool", "unknown")
                    # Find and close the matching tool call span
                    tool_data = turn_to_tool.pop(turn, None)
                    if tool_data:
                        tool_span, _ = tool_data
                        exit_code = event.get("exit_code")
                        timed_out = event.get("timed_out", False)
                        summary = event.get("summary", "")

                        tool_span.set_attribute("tool.exit_code", exit_code if exit_code is not None else -1)
                        tool_span.set_attribute("tool.timed_out", timed_out)
                        # OpenInference semantic convention for output
                        tool_span.set_attribute("output.value", _truncate(summary))

                        if event.get("hint"):
                            tool_span.set_attribute("tool.hint", event["hint"])

                        # Set status based on result
                        if timed_out:
                            tool_span.set_status(Status(StatusCode.ERROR, "Command timed out"))
                        elif exit_code is not None and exit_code != 0:
                            tool_span.set_status(Status(StatusCode.ERROR, f"Exit code: {exit_code}"))
                        else:
                            tool_span.set_status(Status(StatusCode.OK))

                        tool_span.end()

                elif etype == "protocol_error":
                    parent_span.add_event(
                        "protocol_error",
                        attributes={
                            "turn": turn,
                            "error_type": event.get("error_type", "unknown"),
                            "error_message": _truncate(event.get("error_message", "")),
                        }
                    )

            # Close any unclosed tool spans
            for turn, (span, _) in turn_to_tool.items():
                span.set_status(Status(StatusCode.ERROR, "Tool call not completed"))
                span.end()

            # Add final answer to parent span
            if final_answer:
                content = final_answer.get("content", {})
                metadata = final_answer.get("metadata", {})
                content_str = json.dumps(content, indent=2) if isinstance(content, dict) else str(content)

                parent_span.set_attribute("final_answer.content", _truncate(content_str))
                parent_span.set_attribute("final_answer.turn", final_answer.get("turn", 0))

                if metadata:
                    parent_span.set_attribute("final_answer.model", metadata.get("model", ""))
                    parent_span.set_attribute("final_answer.input_tokens", metadata.get("input_tokens", 0))
                    parent_span.set_attribute("final_answer.output_tokens", metadata.get("output_tokens", 0))

            # === LLM JUDGE OUTPUT (task_adherence evaluation) ===
            if llm_judge_output:
                parsed = llm_judge_output.get("parsed", {})

                # Judge metadata
                parent_span.set_attribute("judge.name", llm_judge_output.get("judge", ""))
                parent_span.set_attribute("judge.model", llm_judge_output.get("model", ""))
                parent_span.set_attribute("judge.domain", llm_judge_output.get("domain", ""))

                # Parsed scores and assessment
                parent_span.set_attribute("judge.score", parsed.get("score", 0))
                parent_span.set_attribute("judge.followed_instructions", parsed.get("followed_instructions", False))
                parent_span.set_attribute("judge.navigation_quality", parsed.get("navigation_quality", ""))
                parent_span.set_attribute("judge.reasoning", _truncate(parsed.get("reasoning", "")))

                # Component scores
                component_scores = parsed.get("component_scores", {})
                if component_scores:
                    parent_span.set_attribute("judge.component_scores", json.dumps(component_scores))

                # Strengths and weaknesses
                strengths = parsed.get("strengths", [])
                if strengths:
                    parent_span.set_attribute("judge.strengths", json.dumps(strengths))

                weaknesses = parsed.get("weaknesses", [])
                if weaknesses:
                    parent_span.set_attribute("judge.weaknesses", json.dumps(weaknesses))

                penalties = parsed.get("penalties_applied", [])
                if penalties:
                    parent_span.set_attribute("judge.penalties_applied", json.dumps(penalties))

                # Raw judge output for reference
                parent_span.set_attribute("judge.raw_output", _truncate(llm_judge_output.get("raw", "")))

            # === LLM JUDGE INPUT (context for the judge) ===
            if llm_judge_input:
                parent_span.set_attribute("judge_input.task_prompt", _truncate(llm_judge_input.get("task_prompt", "")))
                parent_span.set_attribute("judge_input.steps_used", llm_judge_input.get("steps_used", 0))
                parent_span.set_attribute("judge_input.tool_calls_count", llm_judge_input.get("tool_calls_count", 0))
                parent_span.set_attribute("judge_input.protocol_errors", llm_judge_input.get("protocol_errors", 0))
                parent_span.set_attribute("judge_input.command_timeouts", llm_judge_input.get("command_timeouts", 0))
                parent_span.set_attribute("judge_input.has_answer", llm_judge_input.get("has_answer", False))
                parent_span.set_attribute("judge_input.answer_summary", llm_judge_input.get("answer_summary", ""))
                parent_span.set_attribute("judge_input.action_summary", _truncate(llm_judge_input.get("action_summary", "")))

            # === FULL EVALUATION METRICS ===
            if eval_data:
                # Top-level
                parent_span.set_attribute("eval.task_id", eval_data.get("task_id", ""))
                parent_span.set_attribute("eval.domain", eval_data.get("domain", ""))
                parent_span.set_attribute("eval.success", eval_data.get("success", False))

                # Accuracy metrics
                accuracy = eval_data.get("accuracy", {})
                parent_span.set_attribute("eval.accuracy.total_questions", accuracy.get("total_questions", 0))
                parent_span.set_attribute("eval.accuracy.correct_answers", accuracy.get("correct_answers", 0))
                parent_span.set_attribute("eval.accuracy.accuracy", accuracy.get("accuracy", 0))
                parent_span.set_attribute("eval.accuracy.total_written", accuracy.get("total_written", 0))
                parent_span.set_attribute("eval.accuracy.correct_written", accuracy.get("correct_written", 0))
                parent_span.set_attribute("eval.accuracy.written_accuracy", accuracy.get("written_accuracy", 0))
                parent_span.set_attribute("eval.accuracy.total_vision", accuracy.get("total_vision", 0))
                parent_span.set_attribute("eval.accuracy.correct_vision", accuracy.get("correct_vision", 0))
                parent_span.set_attribute("eval.accuracy.vision_accuracy", accuracy.get("vision_accuracy", 0))

                # Question results - detailed breakdown
                question_results = accuracy.get("question_results", [])
                if question_results:
                    parent_span.set_attribute("eval.accuracy.question_results", _truncate(json.dumps(question_results, indent=2)))

                missing_questions = accuracy.get("missing_questions", [])
                if missing_questions:
                    parent_span.set_attribute("eval.accuracy.missing_questions", json.dumps(missing_questions))

                # Reproducibility metrics
                repro = eval_data.get("reproducibility", {})
                parent_span.set_attribute("eval.reproducibility.success", repro.get("success", False))
                parent_span.set_attribute("eval.reproducibility.results_dir_exists", repro.get("results_dir_exists", False))
                parent_span.set_attribute("eval.reproducibility.num_output_files", repro.get("num_output_files", 0))
                parent_span.set_attribute("eval.reproducibility.total_output_bytes", repro.get("total_output_bytes", 0))
                parent_span.set_attribute("eval.reproducibility.reason", repro.get("reason", ""))

                output_files = repro.get("output_files", [])
                if output_files:
                    parent_span.set_attribute("eval.reproducibility.output_files", json.dumps(output_files))

                # Task adherence metrics (from judge)
                adherence = eval_data.get("task_adherence", {})
                parent_span.set_attribute("eval.adherence.score", adherence.get("score", 0))
                parent_span.set_attribute("eval.adherence.followed_instructions", adherence.get("followed_instructions", False))
                parent_span.set_attribute("eval.adherence.navigation_quality", adherence.get("navigation_quality", ""))
                parent_span.set_attribute("eval.adherence.reasoning", _truncate(adherence.get("reasoning", "")))
                parent_span.set_attribute("eval.adherence.status", adherence.get("status", ""))

                adherence_strengths = adherence.get("strengths", [])
                if adherence_strengths:
                    parent_span.set_attribute("eval.adherence.strengths", json.dumps(adherence_strengths))

                adherence_weaknesses = adherence.get("weaknesses", [])
                if adherence_weaknesses:
                    parent_span.set_attribute("eval.adherence.weaknesses", json.dumps(adherence_weaknesses))

                if adherence.get("error_message"):
                    parent_span.set_attribute("eval.adherence.error_message", adherence["error_message"])

                # Efficiency metrics
                efficiency = eval_data.get("efficiency", {})
                parent_span.set_attribute("eval.efficiency.steps_used", efficiency.get("steps_used", 0))
                parent_span.set_attribute("eval.efficiency.max_steps", efficiency.get("max_steps", 0))
                parent_span.set_attribute("eval.efficiency.tool_calls", efficiency.get("tool_calls", 0))
                parent_span.set_attribute("eval.efficiency.time_seconds", efficiency.get("time_seconds", 0))
                parent_span.set_attribute("eval.efficiency.protocol_errors", efficiency.get("protocol_errors", 0))
                parent_span.set_attribute("eval.efficiency.command_timeouts", efficiency.get("command_timeouts", 0))

                # Cost
                if eval_data.get("task_cost") is not None:
                    parent_span.set_attribute("eval.task_cost", eval_data["task_cost"])

                # Set overall status
                if eval_data.get("success"):
                    parent_span.set_status(Status(StatusCode.OK))
                else:
                    parent_span.set_status(Status(StatusCode.ERROR, "Task failed"))

        # Force flush to ensure spans are sent before returning
        if _tracer_provider:
            _tracer_provider.force_flush(timeout_millis=10000)

        logger.info(f"Exported trace for {task_id} to Arize AX")
        return True

    except Exception as e:
        logger.error(f"Failed to export trace to Arize: {e}")
        return False


def is_enabled() -> bool:
    """Check if Arize export is enabled."""
    return _ENABLED
