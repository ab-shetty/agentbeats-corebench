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
from typing import Any

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
        # set_global_tracer_provider=False prevents a2a-sdk's internal traces
        _tracer_provider = register(
            space_id=ARIZE_SPACE_ID,
            api_key=ARIZE_API_KEY,
            project_name=ARIZE_PROJECT,
            set_global_tracer_provider=False,
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

    Creates multiple traces grouped into one session:
    - One trace per tool call (with input/output)
    - One summary trace with evaluation metrics

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
        from opentelemetry.trace import Status, StatusCode, get_current_span
        from openinference.instrumentation import using_session
        from openinference.semconv.trace import SpanAttributes

        # Extract metadata from events
        task_start = next((e for e in events if e.get("type") == "task_start"), {})
        evaluation = next((e for e in events if e.get("type") == "evaluation"), {})
        final_answer = next((e for e in events if e.get("type") == "final_answer"), {})
        llm_judge_output = next((e for e in events if e.get("type") == "llm_judge_output"), {})
        llm_judge_input = next((e for e in events if e.get("type") == "llm_judge_input"), {})
        eval_data = evaluation.get("evaluation", {})

        # Session ID groups all traces for this task together
        session_id = f"{task_id}_{run_id}"

        # All traces within this context share the same session
        with using_session(session_id):

            # Pair tool_call and tool_result events by turn
            tool_calls = {e["turn"]: e for e in events if e.get("type") == "tool_call"}
            tool_results = {e["turn"]: e for e in events if e.get("type") == "tool_result"}

            # Track the first span_id for evaluation logging
            first_span_id = None

            # Create one trace per tool call
            for turn in sorted(tool_calls.keys()):
                call = tool_calls[turn]
                result = tool_results.get(turn, {})

                tool_name = call.get("tool", "unknown")
                arguments = call.get("arguments", {})
                arguments_str = json.dumps(arguments, indent=2)

                exit_code = result.get("exit_code")
                timed_out = result.get("timed_out", False)
                summary = result.get("summary", "")

                # Each tool call is a ROOT span = its own trace
                with _tracer.start_as_current_span(
                    f"[{turn}] {tool_name}",
                    attributes={
                        "tool.name": tool_name,
                        "turn": turn,
                        "corebench.task_id": task_id,
                        "corebench.run_id": run_id,
                        "input.value": _truncate(arguments_str),
                        "output.value": _truncate(summary),
                        "tool.exit_code": exit_code if exit_code is not None else -1,
                        "tool.timed_out": timed_out,
                    }
                ) as span:
                    # Explicitly set session_id on span
                    span.set_attribute(SpanAttributes.SESSION_ID, session_id)

                    # Capture first span_id for evaluation logging
                    if first_span_id is None:
                        span_context = span.get_span_context()
                        first_span_id = format(span_context.span_id, '016x')

                    if result.get("hint"):
                        span.set_attribute("tool.hint", result["hint"])

                    if timed_out:
                        span.set_status(Status(StatusCode.ERROR, "Command timed out"))
                    elif exit_code is not None and exit_code != 0:
                        span.set_status(Status(StatusCode.ERROR, f"Exit code: {exit_code}"))
                    else:
                        span.set_status(Status(StatusCode.OK))

            # Create a summary trace with task metadata and evaluation
            with _tracer.start_as_current_span(
                f"[summary] {task_id}",
                attributes={
                    "corebench.run_id": run_id,
                    "corebench.task_id": task_id,
                    "corebench.domain": domain or task_start.get("domain", ""),
                    "corebench.host": task_start.get("host", ""),
                    "corebench.questions": json.dumps(task_start.get("questions", [])),
                }
            ) as summary_span:
                # Explicitly set session_id on span
                summary_span.set_attribute(SpanAttributes.SESSION_ID, session_id)

                # Add plan events (agent reasoning steps)
                for event in events:
                    if event.get("type") == "plan":
                        summary_span.add_event("plan", attributes={
                            "turn": event.get("turn", 0),
                            "flow": event.get("flow", ""),
                            "content": _truncate(event.get("content", "")),
                        })

                # Final answer
                if final_answer:
                    content = final_answer.get("content", {})
                    metadata = final_answer.get("metadata", {})
                    content_str = json.dumps(content, indent=2) if isinstance(content, dict) else str(content)

                    summary_span.set_attribute("final_answer.content", _truncate(content_str))
                    summary_span.set_attribute("final_answer.turn", final_answer.get("turn", 0))

                    if metadata:
                        summary_span.set_attribute("final_answer.model", metadata.get("model", ""))
                        summary_span.set_attribute("final_answer.input_tokens", metadata.get("input_tokens", 0))
                        summary_span.set_attribute("final_answer.output_tokens", metadata.get("output_tokens", 0))

                # LLM Judge Output
                if llm_judge_output:
                    parsed = llm_judge_output.get("parsed", {})

                    summary_span.set_attribute("judge.model", llm_judge_output.get("model", ""))
                    summary_span.set_attribute("judge.domain", llm_judge_output.get("domain", ""))
                    summary_span.set_attribute("judge.score", parsed.get("score", 0))
                    summary_span.set_attribute("judge.reasoning", _truncate(parsed.get("reasoning", "")))
                    if parsed.get("strengths"):
                        summary_span.set_attribute("judge.strengths", json.dumps(parsed["strengths"]))
                    if parsed.get("weaknesses"):
                        summary_span.set_attribute("judge.weaknesses", json.dumps(parsed["weaknesses"]))

                # LLM Judge Input
                if llm_judge_input:
                    summary_span.set_attribute("judge_input.task_prompt", _truncate(llm_judge_input.get("task_prompt", "")))
                    summary_span.set_attribute("judge_input.tool_calls_count", llm_judge_input.get("tool_calls_count", 0))
                    summary_span.set_attribute("judge_input.command_timeouts", llm_judge_input.get("command_timeouts", 0))
                    summary_span.set_attribute("judge_input.has_answer", llm_judge_input.get("has_answer", False))
                    summary_span.set_attribute("judge_input.tool_interactions", _truncate(llm_judge_input.get("tool_interactions", "")))

                # Full Evaluation Metrics
                if eval_data:
                    summary_span.set_attribute("eval.task_id", eval_data.get("task_id", ""))
                    summary_span.set_attribute("eval.domain", eval_data.get("domain", ""))
                    summary_span.set_attribute("eval.success", eval_data.get("success", False))

                    # Accuracy
                    accuracy = eval_data.get("accuracy", {})
                    summary_span.set_attribute("eval.accuracy.total_questions", accuracy.get("total_questions", 0))
                    summary_span.set_attribute("eval.accuracy.correct_answers", accuracy.get("correct_answers", 0))
                    summary_span.set_attribute("eval.accuracy.accuracy", accuracy.get("accuracy", 0))
                    summary_span.set_attribute("eval.accuracy.total_written", accuracy.get("total_written", 0))
                    summary_span.set_attribute("eval.accuracy.correct_written", accuracy.get("correct_written", 0))
                    summary_span.set_attribute("eval.accuracy.written_accuracy", accuracy.get("written_accuracy", 0))
                    summary_span.set_attribute("eval.accuracy.total_vision", accuracy.get("total_vision", 0))
                    summary_span.set_attribute("eval.accuracy.correct_vision", accuracy.get("correct_vision", 0))
                    summary_span.set_attribute("eval.accuracy.vision_accuracy", accuracy.get("vision_accuracy", 0))

                    if accuracy.get("question_results"):
                        summary_span.set_attribute("eval.accuracy.question_results", _truncate(json.dumps(accuracy["question_results"], indent=2)))
                    if accuracy.get("extra_questions"):
                        summary_span.set_attribute("eval.accuracy.extra_questions", json.dumps(accuracy["extra_questions"]))

                    # Adherence
                    adherence = eval_data.get("task_adherence", {})
                    summary_span.set_attribute("eval.adherence.score", adherence.get("score", 0))
                    summary_span.set_attribute("eval.adherence.reasoning", _truncate(adherence.get("reasoning", "")))
                    summary_span.set_attribute("eval.adherence.status", adherence.get("status", ""))

                    if adherence.get("strengths"):
                        summary_span.set_attribute("eval.adherence.strengths", json.dumps(adherence["strengths"]))
                    if adherence.get("weaknesses"):
                        summary_span.set_attribute("eval.adherence.weaknesses", json.dumps(adherence["weaknesses"]))
                    if adherence.get("error_message"):
                        summary_span.set_attribute("eval.adherence.error_message", adherence["error_message"])

                    # Efficiency
                    efficiency = eval_data.get("efficiency", {})
                    summary_span.set_attribute("eval.efficiency.steps_used", efficiency.get("steps_used", 0))
                    summary_span.set_attribute("eval.efficiency.max_steps", efficiency.get("max_steps", 0))
                    summary_span.set_attribute("eval.efficiency.tool_calls", efficiency.get("tool_calls", 0))
                    summary_span.set_attribute("eval.efficiency.time_seconds", efficiency.get("time_seconds", 0))
                    summary_span.set_attribute("eval.efficiency.protocol_errors", efficiency.get("protocol_errors", 0))
                    summary_span.set_attribute("eval.efficiency.command_timeouts", efficiency.get("command_timeouts", 0))

                    # Methodology
                    methodology = eval_data.get("methodology_metrics", {})
                    if methodology:
                        summary_span.set_attribute("eval.methodology.score", methodology.get("methodology_score", 0))
                        summary_span.set_attribute("eval.methodology.read_documentation", methodology.get("read_documentation", False))
                        summary_span.set_attribute("eval.methodology.read_target_script", methodology.get("read_target_script", False))
                        summary_span.set_attribute("eval.methodology.attempted_execution", methodology.get("attempted_execution", False))
                        summary_span.set_attribute("eval.methodology.successful_execution", methodology.get("successful_execution", False))
                        summary_span.set_attribute("eval.methodology.execution_coverage", methodology.get("execution_coverage", 0))
                        summary_span.set_attribute("eval.methodology.installed_dependencies", methodology.get("installed_dependencies", False))
                        summary_span.set_attribute("eval.methodology.read_preexisting_results", methodology.get("read_preexisting_results", False))

                        if methodology.get("docs_read"):
                            summary_span.set_attribute("eval.methodology.docs_read", json.dumps(methodology["docs_read"]))
                        if methodology.get("scripts_read"):
                            summary_span.set_attribute("eval.methodology.scripts_read", json.dumps(methodology["scripts_read"]))
                        if methodology.get("executed_scripts"):
                            summary_span.set_attribute("eval.methodology.executed_scripts", json.dumps(methodology["executed_scripts"]))
                        if methodology.get("violations"):
                            summary_span.set_attribute("eval.methodology.violations", json.dumps(methodology["violations"]))

                        # Error recovery
                        error_recovery = methodology.get("error_recovery", {})
                        if error_recovery:
                            summary_span.set_attribute("eval.methodology.error_recovery.total_errors", error_recovery.get("total_errors", 0))
                            summary_span.set_attribute("eval.methodology.error_recovery.recovery_rate", error_recovery.get("recovery_rate", 0))
                            summary_span.set_attribute("eval.methodology.error_recovery.persistence_score", error_recovery.get("persistence_score", 0))

                    if eval_data.get("task_cost") is not None:
                        summary_span.set_attribute("eval.task_cost", eval_data["task_cost"])

                    # Set overall status
                    if eval_data.get("success"):
                        summary_span.set_status(Status(StatusCode.OK))
                    else:
                        summary_span.set_status(Status(StatusCode.ERROR, "Task failed"))

        # Force flush to ensure spans are sent before returning
        if _tracer_provider:
            _tracer_provider.force_flush(timeout_millis=10000)

        # Log session-level evaluations if we have eval data and a span_id
        if eval_data and first_span_id:
            try:
                import pandas as pd
                from arize.pandas.logger import Client

                adherence = eval_data.get("task_adherence", {})
                accuracy = eval_data.get("accuracy", {})
                methodology = eval_data.get("methodology_metrics", {})

                # Build methodology explanation from key behaviors
                methodology_parts = []
                if methodology.get("read_documentation"):
                    methodology_parts.append("read docs")
                if methodology.get("read_target_script"):
                    methodology_parts.append("read script")
                if methodology.get("attempted_execution"):
                    methodology_parts.append("attempted exec")
                if methodology.get("successful_execution"):
                    methodology_parts.append("exec succeeded")
                if methodology.get("read_preexisting_results"):
                    methodology_parts.append("READ RESULTS (anti-pattern)")
                coverage = methodology.get("execution_coverage", 0)
                methodology_explanation = f"{', '.join(methodology_parts) or 'no actions'}; coverage={coverage:.0%}"

                methodology_pass = (
                    methodology.get("attempted_execution", False) and
                    not methodology.get("read_preexisting_results", False)
                )

                # Create evaluation DataFrame
                eval_df = pd.DataFrame([{
                    "context.span_id": first_span_id,
                    # Adherence score
                    "eval.adherence.score": adherence.get("score", 0),
                    "eval.adherence.label": "pass" if adherence.get("score", 0) >= 0.5 else "fail",
                    "eval.adherence.explanation": adherence.get("reasoning", "")[:500],
                    # Accuracy score
                    "eval.accuracy.score": accuracy.get("accuracy", 0),
                    "eval.accuracy.label": "pass" if eval_data.get("success", False) else "fail",
                    "eval.accuracy.explanation": f"{accuracy.get('correct_answers', 0)}/{accuracy.get('total_questions', 0)} correct",
                    "eval.methodology.score": methodology.get("methodology_score", 0),
                    "eval.methodology.label": "pass" if methodology_pass else "fail",
                    "eval.methodology.explanation": methodology_explanation,
                }])

                # Log evaluations to Arize
                arize_client = Client(space_id=ARIZE_SPACE_ID, api_key=ARIZE_API_KEY)
                arize_client.log_evaluations_sync(eval_df, ARIZE_PROJECT)
                logger.debug(f"Logged evaluations for session {session_id}")

            except Exception as e:
                logger.warning(f"Failed to log evaluations: {e}")

        num_tool_traces = len(tool_calls)
        logger.info(f"Exported {num_tool_traces + 1} traces for {task_id} to Arize AX (session: {session_id})")
        return True

    except Exception as e:
        logger.error(f"Failed to export trace to Arize: {e}")
        return False


def is_enabled() -> bool:
    """Check if Arize export is enabled."""
    return _ENABLED
