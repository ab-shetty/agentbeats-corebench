"""
CoreBench Evaluation Metrics Module
===================================
"""

from dataclasses import asdict
from typing import Any, Callable, Optional
import json
import logging
import math
import os
import sys

import re

import numpy as np
from scipy.stats import t
import litellm

from scenarios.corebench.metrics.models import (
    QuestionResult,
    AccuracyMetrics,
    TaskAdherenceMetrics,
    ErrorRecoveryMetrics,
    MethodologyScoreBreakdown,
    MethodologyMetrics,
    TaskEvaluation,
    AggregateMetrics,
)

logger = logging.getLogger("evaluator.metrics")

_VISION_KEY_PATTERN = re.compile(r"^\s*fig(?:ure)?s?\b", re.IGNORECASE)

# Float tolerance for interval comparisons (scaled from machine epsilon)
# N = 1e5 → rel_tol ≈ 2e-11, abs_tol ≈ 2e-11
_REL_TOL = 1e5 * sys.float_info.epsilon
_ABS_TOL = 1e5 * sys.float_info.epsilon
# Python utility modules that should not be counted as script executions
# Used by _extract_scripts_from_command() and _is_target_execution()
_NON_SCRIPT_MODULES = frozenset({
    'venv', 'pip', 'pytest', 'unittest', 'http', 'json', 'ensurepip',
    'compileall', 'pdb', 'cprofile', 'profile', 'timeit', 'trace'
})

def _is_vision_question(key: str) -> bool:
    """Check if question requires vision tool based on regex pattern in CORE-Bench paper"""
    return bool(_VISION_KEY_PATTERN.search(key))


# =============================================================================
# ACCURACY EVALUATION
# =============================================================================

def evaluate_accuracy(
    ground_truth: list[dict[str, Any]],
    submitted: dict[str, Any],
) -> AccuracyMetrics:
    """
    Evaluate answer accuracy against ground truth using prediction intervals.
    
    For numeric values: Uses 95% prediction intervals to account for run-to-run variance
    in stochastic experiments (e.g., ML training with random seeds).
    
    For strings: Case-insensitive exact match.
    For lists: Element-wise exact comparison.
    """
    if not ground_truth or not ground_truth[0]:
            logger.error("Empty or invalid ground truth provided")
            return _empty_accuracy_metrics()
    
    # Ensure submitted is a dict (handle malformed answers like int, str, etc.)
    if not isinstance(submitted, dict):
        logger.warning(f"Submitted answer is not a dict (got {type(submitted).__name__}), treating as empty")
        submitted = {}

    # Categorize ground truth keys by type
    reference = ground_truth[0]
    numeric_keys = [k for k, v in reference.items() if isinstance(v, (int, float))]
    string_keys = [k for k, v in reference.items() if isinstance(v, str)]
    list_keys = [k for k, v in reference.items() if isinstance(v, list)]
    
    required_questions = list(reference.keys())

    # Calculate prediction intervals for numeric values
    prediction_intervals = _compute_prediction_intervals(ground_truth, numeric_keys)
    
    # Evaluate each question
    question_results: list[QuestionResult] = []
    correct_written = 0
    correct_vision = 0
    total_written = 0
    total_vision = 0
    
    for key in required_questions:
        is_vision = _is_vision_question(key)
        
        if is_vision:
            total_vision += 1
        else:
            total_written += 1
        
        if key not in submitted:
            # Missing answer - rare in practice since agents follow the prompt format
            result = QuestionResult(
                question=key,
                question_type="unknown",
                is_vision=is_vision,
                correct=False,
                submitted=None,
                expected=reference.get(key),
                prediction_interval=prediction_intervals.get(key),
            )
            question_results.append(result)
            continue
        
        submitted_value = submitted[key]
        
        if key in numeric_keys:
            correct, result = _evaluate_numeric(
                key, submitted_value, prediction_intervals.get(key), is_vision
            )
        elif key in string_keys:
            correct, result = _evaluate_string(
                key, submitted_value, reference[key], is_vision
            )
        elif key in list_keys:
            correct, result = _evaluate_list(
                key, submitted_value, reference[key], is_vision
            )
        else:
            # Unknown type - shouldn't happen
            correct = False
            result = QuestionResult(
                question=key,
                question_type="unknown",
                is_vision=is_vision,
                correct=False,
                submitted=submitted_value,
                expected=reference.get(key),
            )
        
        question_results.append(result)
        
        if correct:
            if is_vision:
                correct_vision += 1
            else:
                correct_written += 1
    
    # Identify extra questions submitted but not required (agent invented/reworded keys)
    extra_questions = [k for k in submitted.keys() if k not in required_questions]

    total = total_written + total_vision
    correct = correct_written + correct_vision

    return AccuracyMetrics(
        total_questions=total,
        correct_answers=correct,
        accuracy=correct / total if total > 0 else 0.0,
        total_written=total_written,
        correct_written=correct_written,
        written_accuracy=correct_written / total_written if total_written > 0 else 0.0,
        total_vision=total_vision,
        correct_vision=correct_vision,
        vision_accuracy=correct_vision / total_vision if total_vision > 0 else 0.0,
        question_results=question_results,
        extra_questions=extra_questions,
    )


def _check_scale_mismatch(submitted: float, interval: dict) -> tuple[bool, Optional[str]]:
    """
    Check if submitted value is a scale mismatch (100x off from expected).

    Ground truth values are mixed: some are decimals (0-1), some percentages (1-100).
    This detects when the agent converted between formats incorrectly.

    Returns:
        (is_mismatch, direction):
            - (True, "submitted_is_decimal") if submitted ~= expected/100
            - (True, "submitted_is_percent") if submitted ~= expected*100
            - (False, None) if no scale mismatch detected
    """
    mean = interval.get("mean", 0)

    if mean == 0 or submitted == 0:
        return False, None

    # Check if submitted is ~100x smaller (agent converted percentage to decimal)
    # e.g., submitted 0.96 when expected 96.12
    if 0 < abs(submitted) < 1.5 and abs(mean) > 1.5:
        scaled_up = submitted * 100
        if interval["lower"] <= scaled_up <= interval["upper"]:
            return True, "submitted_is_decimal"

    # Check if submitted is ~100x larger (agent converted decimal to percentage)
    # e.g., submitted 96 when expected 0.96
    if abs(submitted) > 1.5 and 0 < abs(mean) < 1.5:
        scaled_down = submitted / 100
        if interval["lower"] <= scaled_down <= interval["upper"]:
            return True, "submitted_is_percent"

    return False, None


def _compute_prediction_intervals(
    ground_truth: list[dict], 
    numeric_keys: list[str]
) -> dict[str, dict]:
    """Compute 95% prediction intervals for numeric values."""
    intervals = {}
    sample_size = len(ground_truth)
    
    if sample_size < 2:
        # Can't compute intervals with < 2 samples, use exact match
        for key in numeric_keys:
            value = ground_truth[0].get(key, 0)
            intervals[key] = {"lower": value, "upper": value, "mean": value}
        return intervals
    
    t_value = t.ppf(0.975, sample_size - 1)
    
    for key in numeric_keys:
        values = [gt.get(key, 0) for gt in ground_truth]
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        margin = t_value * std * math.sqrt(1 + 1/sample_size)
        intervals[key] = {
            "lower": mean - margin,
            "upper": mean + margin,
            "mean": mean,
        }
    
    return intervals


def _evaluate_numeric(
    key: str,
    submitted: Any,
    interval: Optional[dict],
    is_vision: bool,
) -> tuple[bool, QuestionResult]:
    """Evaluate a numeric answer against prediction interval."""
    # Normalize string submissions to float.
    # Agents sometimes output numeric answers as strings (e.g., "96.5%" or "1,200,000").
    if isinstance(submitted, str):
        cleaned = submitted.strip().replace('%', '').replace(',', '')
        try:
            submitted = float(cleaned)
        except ValueError:
            pass  # Keep as string, will fail the type check below

    if not isinstance(submitted, (int, float)) or (isinstance(submitted, float) and math.isnan(submitted)):
        return False, QuestionResult(
            question=key,
            question_type="numeric",
            is_vision=is_vision,
            correct=False,
            submitted=submitted,
            prediction_interval=interval,
        )

    if interval is None:
        return False, QuestionResult(
            question=key,
            question_type="numeric",
            is_vision=is_vision,
            correct=False,
            submitted=submitted,
        )

    # Explicit bool() to ensure Python bool, not numpy.bool_
    lower = interval["lower"]
    upper = interval["upper"]
    correct = bool(
        (lower <= submitted <= upper)
        or math.isclose(submitted, lower, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
        or math.isclose(submitted, upper, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
    )

    # Log scale mismatches for debugging (agent got correct value but wrong scale)
    if not correct:
        is_mismatch, direction = _check_scale_mismatch(submitted, interval)
        if is_mismatch:
            if direction == "submitted_is_decimal":
                logger.warning(
                    f"SCALE MISMATCH for '{key}': Agent submitted {submitted} but expected ~{interval['mean']:.4f}. "
                    f"Agent appears to have converted a percentage to decimal (submitted*100={submitted*100:.4f} would match)."
                )
            else:
                logger.warning(
                    f"SCALE MISMATCH for '{key}': Agent submitted {submitted} but expected ~{interval['mean']:.4f}. "
                    f"Agent appears to have converted a decimal to percentage (submitted/100={submitted/100:.4f} would match)."
                )

    return correct, QuestionResult(
        question=key,
        question_type="numeric",
        is_vision=is_vision,
        correct=correct,
        submitted=submitted,
        prediction_interval=interval,
    )


def _evaluate_string(
    key: str,
    submitted: Any,
    expected: str,
    is_vision: bool,
) -> tuple[bool, QuestionResult]:
    """Evaluate a string answer (case-insensitive)."""
    # Explicit bool() to ensure Python bool
    correct = bool(str(submitted).lower().strip() == expected.lower().strip())

    return correct, QuestionResult(
        question=key,
        question_type="string",
        is_vision=is_vision,
        correct=correct,
        submitted=submitted,
        expected=expected,
    )


def _evaluate_list(
    key: str,
    submitted: Any,
    expected: list,
    is_vision: bool,
) -> tuple[bool, QuestionResult]:
    """Evaluate a list answer (exact element-wise match).

    Simple direct comparison:
    - Order matters: [1,2] != [2,1]
    - Type matters: [1, 2] != ["1", "2"]
    """
    # Handle None as empty list
    if submitted is None:
        submitted = []

    # Direct comparison (order matters, types matter)
    correct = bool(submitted == expected)

    return correct, QuestionResult(
        question=key,
        question_type="list",
        is_vision=is_vision,
        correct=correct,
        submitted=submitted,
        expected=expected,
    )


def _empty_accuracy_metrics() -> AccuracyMetrics:
    return AccuracyMetrics(
        total_questions=0,
        correct_answers=0,
        accuracy=0.0,
        total_written=0,
        correct_written=0,
        written_accuracy=0.0,
        total_vision=0,
        correct_vision=0,
        vision_accuracy=0.0,
        question_results=[],
        extra_questions=[],
    )


# =============================================================================
# TASK ADHERENCE EVALUATION (LLM-as-Judge)
# =============================================================================
# Provides qualitative assessment of agent execution for debugging and improvement.
# Unlike methodology_score (deterministic), this is subjective LLM judgment.
#
# WHAT THE JUDGE RECEIVES:
# 1. Domain (corebench_easy/medium/hard)
# 2. Task instruction (e.g., "Run main.py")
# 3. Questions to answer (e.g., ["What is the accuracy?", "What is the F1 score?"])
# 4. Capsule docs: README files from workspace/environment/
#    - Shows what documentation was available to the agent and gives context
# 5. Execution stats: tool_calls_count, command_timeouts
# 6. Tool interactions: all 40 tool call/result pairs (truncated)
# =============================================================================

# Domain-specific criteria for LLM judge - only the relevant one is injected
DOMAIN_CRITERIA = {
    "corebench_hard": """
## Hard Mode

The agent must figure out how to run the code and reproduce results. It could do this by:
- Reading README.md or other documentation
- Inspecting scripts to understand dependencies
- Exploring the directory structure
- Trial and error with debugging

## Gold Standard Workflow
Understand the codebase → Execute the code → Debug errors if needed → Extract results

## Core Process Scoring Guide (50%)
Score based on highest milestone achieved:
- **45-50**: Executed correct script AND extracted results 
- **35-44**: Correct approach + strong debugging, even if things didn't work out
- **25-34**: Executed code but wrong script or incomplete results 
- **15-24**: Understood codebase but never executed 
- **0-14**: No meaningful attempt

## Red Flags
- Never attempted to run code
- Gave up after first error without trying to debug
""",

    "corebench_medium": """
## Medium Mode

The agent has access to REPRODUCING.md with step-by-step instructions.

## Gold Standard Workflow
Read REPRODUCING.md instructions → Execute documented command → Handle errors → Generate output

## Core Process Scoring Guide (50%)
- **45-50**: Followed documented instructions and executed successfully
- **30-44**: Found instructions but execution incomplete
- **15-29**: Attempted execution without reading instructions
- **0-14**: Never attempted to run code

## Red Flags
- Never attempted documented command
- Ignored REPRODUCING.md instructions
""",

    "corebench_easy": """
## Easy Mode

The task is to READ existing results from the results/ directory, NOT re-run code.

## Gold Standard Workflow
List results/ directory → Read output files → Extract values

## Core Process Scoring Guide (50%)
- **45-50**: Located and read results files, extracted values
- **30-44**: Found results but incomplete extraction
- **15-29**: Struggled to locate results files
- **0-14**: Wasted time running scripts instead of reading results

## Red Flags
- Attempted to run scripts or install dependencies (unnecessary)
""",
}

TASK_ADHERENCE_PROMPT = """You are an expert evaluator assessing how well an AI agent executed a computational reproducibility benchmark task.
{domain_criteria}

## Other Components
- **Problem Solving (25%)**: How well did agent handle errors? (20-25: debugged effectively OR smooth execution with no errors, 10-19: hit errors, tried to debug but gave up, 0-9: hit errors and made no attempt to resolve them)
- **Discovery (15%)**: How efficiently did agent find information? (12-15: direct path, 6-11: some wandering, 0-5: lost)
- **Technical (10%)**: Command correctness, avoiding redundant operations

## Task Context
Task instruction: {task_prompt}

Questions to answer:
{questions}

{capsule_docs}

## Execution Trace
Tool calls: {tool_calls_count}

Tool calls + results:
{tool_interactions}

Final answer provided: {has_answer}

## Output
Assign scores: Core _/50, Problem _/25, Discovery _/15, Technical _/10

```json
{{
    "score": <float 0.0-1.0>,
    "reasoning": "<component breakdown + observations>",
    "component_scores": {{
        "core_process": "<X/50>",
        "problem_solving": "<X/25>",
        "discovery": "<X/15>",
        "technical": "<X/10>"
    }},
    "strengths": ["<specific good behaviors>"],
    "weaknesses": ["<specific gaps and any red flags>"]
}}
```
"""

def _read_text_file_head_bytes(path: str, max_bytes: int) -> tuple[str, bool]:
    """Read up to max_bytes from a file, returning (text, was_truncated)."""
    with open(path, "rb") as f:
        data = f.read(max_bytes + 1)
    truncated = len(data) > max_bytes
    text = data[:max_bytes].decode("utf-8", errors="replace")
    return text, truncated


def _build_capsule_docs(
    *,
    workspace_dir: str,
    max_bytes: int = 10_000,
) -> str:
    """
    Build documentation context for the LLM judge.

    This function finds and formats capsule documentation (README files) for
    insertion into the TASK_ADHERENCE_PROMPT at {capsule_docs}.

    Returns:
        Formatted markdown string, or empty string if no docs found.
    """
    env_dir = os.path.join(workspace_dir, "environment")

    # start search in root and then code directory
    for subdir in ("", "code"):
        directory = os.path.join(env_dir, subdir) if subdir else env_dir
        if not os.path.isdir(directory):
            continue

        for filename in os.listdir(directory):
            lower = filename.lower()
            # Match to inconsitent naming standards, ex README.md, readme.txt, README_CAPSULE.md, etc.
            if lower.startswith("readme") and lower.endswith((".md", ".txt")):
                abs_path = os.path.join(directory, filename)
                rel_path = f"{subdir}/{filename}" if subdir else filename
                # read file content
                try:
                    text, truncated = _read_text_file_head_bytes(abs_path, max_bytes)
                except OSError:
                    continue
                
                # format and return file content
                suffix = "\n...[truncated]..." if truncated else ""
                return f"## Capsule Docs (available to agent)\n\n### {rel_path}\n```\n{text}{suffix}\n```\n"

    return ""


def _build_tool_interactions(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    *,
    max_args_chars: int = 1200,
) -> str:
    """Format tool calls and results as a string of all tool interactions for the LLM judge."""
    if not tool_calls:
        return "(No tool calls recorded)"

    # Match results to calls by (turn, tool) key
    results_by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
    for event in tool_results:
        key = (event.get("turn"), event.get("tool"))
        if key not in results_by_key:
            results_by_key[key] = event

    pairs: list[tuple[dict[str, Any], Optional[dict[str, Any]]]] = []
    for call in tool_calls:
        key = (call.get("turn"), call.get("tool"))
        pairs.append((call, results_by_key.get(key)))

    def _format_pair(call: dict[str, Any], result: Optional[dict[str, Any]]) -> str:
        turn = call.get("turn", "?")
        tool = call.get("tool", call.get("name", "unknown"))
        arguments = call.get("arguments", {})
        args_json = json.dumps(arguments, ensure_ascii=False, default=str, indent=2)
        if len(args_json) > max_args_chars:
            args_json = args_json[:max_args_chars] + "\n... (truncated)"

        lines = [f"[Turn {turn}] {tool}", f"Arguments:\n{args_json}"]

        if result:
            timed_out = result.get("timed_out", False)
            hint = result.get("hint")
            summary = str(result.get("summary", "") or "")

            if timed_out:
                lines.append("⚠️ Command timed out")
            if summary.strip():
                lines.append(f"Output:\n{summary}")
            if hint:
                lines.append(f"Hint: {hint}")
        else:
            lines.append("Output: (missing tool output)")

        return "\n".join(lines)

    formatted = [_format_pair(call, result) for call, result in pairs]
    return "\n\n".join(formatted)


def _calculate_score_from_components(component_scores: dict[str, str], fallback_score: float) -> float:
    """Calculate task adherence score from component scores.

    Parses component scores like {"core_process": "30/50", "problem_solving": "12/25", ...}
    and computes the final score as sum of numerators / 100.

    Falls back to LLM-provided score if parsing fails.
    """
    if not component_scores:
        return fallback_score

    total = 0
    try:
        for key, value in component_scores.items():
            # Expected format: "30/50", "12/25", etc.
            if "/" not in str(value):
                logger.warning(f"Invalid component score format for {key}: {value}")
                return fallback_score
            parts = str(value).split("/")
            if len(parts) != 2:
                logger.warning(f"Invalid component score format for {key}: {value}")
                return fallback_score
            numerator = float(parts[0].strip())
            total += numerator
        return total / 100.0
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse component scores, using LLM score: {e}")
        return fallback_score


async def evaluate_task_adherence(
    domain: str,
    task_prompt: str,
    questions: list[str],
    tool_calls_count: int,
    submitted: dict[str, Any],
    tool_calls: Optional[list[dict[str, Any]]] = None,
    tool_results: Optional[list[dict[str, Any]]] = None,
    *,
    workspace_dir: str,
    trace_event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    judge_model: str = "gpt-5-mini",
) -> TaskAdherenceMetrics:
    """Use LLM-as-judge to evaluate task execution quality."""
    capsule_docs = _build_capsule_docs(workspace_dir=workspace_dir)
    tool_interactions = _build_tool_interactions(tool_calls or [], tool_results or [])

    # Format questions as numbered list for clarity
    questions_formatted = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(questions)) if questions else "(No questions specified)"

    # Inject only the relevant domain criteria (defaults to hard mode)
    domain_criteria = DOMAIN_CRITERIA.get(domain, DOMAIN_CRITERIA["corebench_hard"])

    prompt = TASK_ADHERENCE_PROMPT.format(
        task_prompt=task_prompt,
        questions=questions_formatted,
        capsule_docs=capsule_docs,
        domain_criteria=domain_criteria,
        tool_calls_count=tool_calls_count,
        tool_interactions=tool_interactions,
        has_answer="Yes" if submitted else "No",
    )
    
    # API configuration - match mcp_server.py pattern for OpenAI models
    openai_api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()

    model_name = judge_model
    completion_kwargs = {
        "model": model_name,
        "api_base": "https://api.openai.com/v1",
        "api_key": openai_api_key,
    }
    logger.debug(f"Task adherence judge using model={model_name}")
    
    if trace_event_callback is not None:
        try:
            trace_event_callback(
                {
                    "type": "llm_judge_input",
                    "flow": "green -> judge",
                    "domain": domain,
                    "model": model_name,
                    "task_prompt": task_prompt,
                    "questions": questions,
                    "tool_calls_count": tool_calls_count,
                    "has_answer": bool(submitted),
                    "capsule_docs": capsule_docs,
                    "tool_interactions": tool_interactions,
                    "prompt": prompt,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to write task adherence judge input trace: {e}")

    try:
        response = await litellm.acompletion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            **completion_kwargs,    # contains model, api_base, api_key
        )
        
        raw_content = response.choices[0].message.content
        result = json.loads(raw_content)

        if trace_event_callback is not None:
            try:
                trace_event_callback(
                    {
                        "type": "llm_judge_output",
                        "model": model_name,
                        "parsed": result,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to write task adherence judge output trace: {e}")
        
        # Calculate scores manually from LLM judge
        component_scores = result.get("component_scores", {})
        llm_score = float(result.get("score", 0.0))
        calculated_score = _calculate_score_from_components(component_scores, llm_score)

        return TaskAdherenceMetrics(
            score=calculated_score,
            reasoning=str(result.get("reasoning", "")),
            strengths=list(result.get("strengths", [])),
            weaknesses=list(result.get("weaknesses", [])),
        )
        
    except Exception as e:
        logger.error(f"Task adherence evaluation failed: {e}")
        if trace_event_callback is not None:
            try:
                trace_event_callback(
                    {
                        "type": "llm_judge_error",
                        "model": model_name,
                        "error": str(e),
                    }
                )
            except Exception as trace_e:
                logger.warning(f"Failed to write task adherence judge error trace: {trace_e}")
        return TaskAdherenceMetrics(
            score=0.0,
            reasoning=f"Evaluation failed: {e}",
            strengths=[],
            weaknesses=[],
            status="error",
            error_message=str(e),
        )


# ================================================================================================
# METHODOLOGY METRICS (Deterministic Extraction from Traces)
# ================================================================================================
#
# This section extracts methodology metrics by analyzing tool_call/tool_result traces.
# Unlike LLM-based evaluation, these metrics are fully deterministic and reproducible.
#
# Key behaviors tracked:
# - Documentation reading
# - Script inspection (understanding what to run)
# - Execution attempts and success
# - Dependency installation
# - Errors encountered
# - Anti-patterns (e.g., reading pre-existing results before executing)

def _is_documentation(path: str) -> bool:
    """Check if file path represents a document like README.md"""
    if not path:
        return False
    path_lower = path.lower()
    filename = os.path.basename(path_lower)

    # Direct matches for documentation files
    doc_patterns = [
        "readme", "readme.md", "readme.txt", "readme.rst",
        "reproducing.md", "readme.pdf",
    ]

    if filename in doc_patterns or filename.startswith("readme"):
        return True

    return False


def _is_target_execution(cmd: str) -> bool:
    """Check if a command attempts to execute a target script.

    Detects common execution patterns for Python, R, Docker, Jupyter, and
    shell scripts. Excludes debugging and installation commands.

    Excludes:
    - One-liner debugging commands (python -c "...")
    - Dependency installation (python -m pip ...)
    - Package management commands (install.packages, etc.)
    - Debug/info commands (cat, print, etc.)
    """
    if not cmd:
        return False
    cmd_lower = cmd.lower()

    # Patterns that indicate real execution:
    # - rmarkdown::render, knitr::knit, bookdown::render - rendering .Rmd/.Rnw files
    r_execution_patterns = ["rmarkdown::render", "knitr::knit", "bookdown::render", "source("]
    if any(pattern in cmd_lower for pattern in r_execution_patterns):
        return True

    # Exclude one-liner debugging commands and specific patterns
    exclusion_patterns = [
        "python -c ",
        "python3 -c ",
        "/python -m pip",  # Full path python
        "/python3 -m pip",
        "rscript -e ",     # Rscript one-liners for debugging
        "rscript -e\"",    # Without space after -e
        " --version",      # Version checks (python --version, Rscript --version, etc.)
    ]
    if any(pattern in cmd_lower for pattern in exclusion_patterns):
        return False

    # Exclude python -m with utility modules in _NON_SCRIPT_MODULES
    py_m_match = re.search(r"python[0-9.]*\s+-m\s+(\S+)", cmd_lower)
    if py_m_match:
        module = py_m_match.group(1).split(".")[0]
        if module in _NON_SCRIPT_MODULES:
            return False

    # Exclude Rscript package installs and debug commands
    if "rscript" in cmd_lower:
        # These are NOT execution attempts:
        rscript_exclusions = [
            "install.packages",
            ".libpaths()",
            "cat(",
            "print(",
            "str(",           # Structure inspection
            "head(",          # Preview data
            "class(",         # Type checking
            "names(",         # Column/attribute names
            "dim(",           # Dimensions
            "summary(",       # Summary statistics
            "packageversion(",
            "installed.packages(",
        ]
        if any(excl in cmd_lower for excl in rscript_exclusions):
            return False
        # Only count Rscript if it's actually running/rendering something
        # (not just querying the environment)

    # Common execution patterns (all lowercase for matching against cmd_lower)
    execution_patterns = [
        "python ", "python3 ",
        "docker run", "docker build", "docker-compose",
        "cmake ", "jupyter", "papermill",
        "rscript",
    ]
    if any(pattern in cmd_lower for pattern in execution_patterns):
        return True

    # Shell script execution patterns
    # Check if bash/sh is used with a .sh file anywhere in the command
    if ("bash " in cmd_lower or "sh " in cmd_lower) and ".sh" in cmd_lower:
        return True

    # Direct script execution (./script.sh, ./run.py, etc.)
    if cmd_lower.startswith("./") and any(ext in cmd_lower for ext in [".sh", ".py", ".r"]):
        return True

    return False


def _is_dependency_install(cmd: str) -> bool:
    """Check if a command installs dependencies."""
    if not cmd:
        return False
    cmd_lower = cmd.lower()

    install_patterns = [
        "pip install", "pip3 install",
        "conda install", "mamba install",
        "apt-get install", "apt install", "yum install",
        "npm install", "yarn add",
        "cargo install", "go get",
        "requirements.txt",  # pip install -r requirements.txt
    ]
    return any(pattern in cmd_lower for pattern in install_patterns)


# Script file extensions for read detection
_SCRIPT_EXTENSIONS = (".py", ".sh", ".r", ".R", ".Rmd", ".jl", ".m", ".ipynb")


def _extract_script_reads_from_bash(cmd: str) -> list[str]:
    """Extract script file paths from bash commands that read file contents.

    Detects common file reading patterns:
    - cat file.py
    - head/tail file.R
    - sed -n '...' file.Rmd
    - less/more file.sh
    - grep pattern file.py (when targeting specific file)

    Returns:
        List of script file paths that were read by the command.
    """
    if not cmd:
        return []

    scripts_found: list[str] = []

    # Split command by common separators (pipes, semicolons, &&, ||)
    # We want to analyze each subcommand
    import shlex

    # Simple tokenization - split on whitespace but respect quotes
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        # If shlex fails (unbalanced quotes), fall back to simple split
        tokens = cmd.split()

    if not tokens:
        return []

    # Commands that read file contents
    read_commands = {"cat", "head", "tail", "less", "more", "sed", "awk", "grep", "view", "bat"}

    # Check if first token (or token after env vars) is a read command
    cmd_name = tokens[0].split("/")[-1]  # Handle full paths like /usr/bin/cat

    # Also check for common patterns like: sed -n '1,200p' file.R
    for i, token in enumerate(tokens):
        base_cmd = token.split("/")[-1]
        if base_cmd in read_commands:
            # Look for file arguments after the command
            for j in range(i + 1, len(tokens)):
                arg = tokens[j]
                # Skip flags (start with -)
                if arg.startswith("-"):
                    continue
                # Skip quoted patterns (for sed/awk/grep)
                if arg.startswith("'") or arg.startswith('"'):
                    continue
                # Skip common grep/sed patterns
                if "=" in arg or arg.startswith("s/") or arg.startswith("/"):
                    continue
                # Check if this looks like a script file
                if arg.endswith(_SCRIPT_EXTENSIONS):
                    # Normalize path (remove leading ./ if present)
                    normalized = arg.lstrip("./")
                    if normalized not in scripts_found:
                        scripts_found.append(normalized)

    return scripts_found


def _classify_error(summary: str) -> str:
    """Classify an error type from tool result summary for diagnostics.

    Returns error category (e.g., 'import_error', 'file_not_found', 'timeout')
    based on patterns observed in CoreBench traces. Used in error recovery
    metrics to understand what types of errors agents encounter.
    """
    if not summary:
        return "unknown"
    summary_lower = summary.lower()

    # Import/Module errors
    if "modulenotfounderror" in summary_lower or "no module named" in summary_lower:
        return "import_error"

    # File system errors
    if "filenotfounderror" in summary_lower or "no such file" in summary_lower:
        return "file_not_found"
    if "permission denied" in summary_lower:
        return "permission_error"

    # Timeout errors
    if "timeout" in summary_lower or "timed out" in summary_lower:
        return "timeout"

    # Container/Docker errors
    if "docker" in summary_lower:
        return "docker_error"

    # Memory errors
    if "memory" in summary_lower or "oom" in summary_lower or "killed" in summary_lower:
        return "memory_error"

    # Syntax errors
    if "syntaxerror" in summary_lower:
        return "syntax_error"

    # Type/Attribute/Value errors (common in Python)
    if "typeerror" in summary_lower:
        return "type_error"
    if "attributeerror" in summary_lower:
        return "attribute_error"
    if "valueerror" in summary_lower:
        return "value_error"
    if "keyerror" in summary_lower:
        return "key_error"
    if "indexerror" in summary_lower:
        return "index_error"

    # Dependency/Build errors
    if "could not find a version" in summary_lower or "no matching distribution" in summary_lower:
        return "version_not_found"
    if "build" in summary_lower and ("failed" in summary_lower or "error" in summary_lower):
        return "build_error"
    if "pip" in summary_lower or "install" in summary_lower:
        return "install_error"

    # Jupyter/Kernel errors
    if "kernel" in summary_lower or "nbconvert" in summary_lower:
        return "kernel_error"

    # Runtime errors
    if "runtimeerror" in summary_lower:
        return "runtime_error"
    
    # R-specific errors
    if "there is no package called" in summary_lower:
        return "r_package_missing"
    if "could not find function" in summary_lower:
        return "r_function_not_found"
    if "cannot open file" in summary_lower:
        return "file_not_found"

    # Path errors
    if "no such file or directory" in summary_lower:
        return "file_not_found"
    if "setwd" in summary_lower and "error" in summary_lower:
        return "working_directory_error"

    # CLI/argument errors
    if "unrecognized arguments" in summary_lower or "invalid argument" in summary_lower:
        return "cli_argument_error"
    if "usage:" in summary_lower and "error:" in summary_lower:
        return "cli_argument_error"

    # Tool/API validation errors (MCP tool parameter issues)
    if "field required" in summary_lower or "validation error" in summary_lower:
        return "tool_validation_error"
    if "unknown command" in summary_lower or "valid commands:" in summary_lower:
        return "tool_validation_error"
    if "old_str and new_str are required" in summary_lower:
        return "tool_validation_error"
    if "not a regular file" in summary_lower:
        return "tool_validation_error"

    # System command not found (missing system dependencies like pandoc, latex, etc.)
    if "command not found" in summary_lower or "not found in path" in summary_lower:
        return "command_not_found"

    return "other"


def _compute_error_recovery(
    tool_calls: list[dict],
    tool_results: list[dict],
) -> ErrorRecoveryMetrics:
    """Analyze error recovery patterns from execution attempts."""
    # Build a timeline of success/failure
    results_by_turn: dict[int, dict] = {}
    for result in tool_results:
        turn = result.get("turn", 0)
        results_by_turn[turn] = result

    # Only track execution attempts (filter out ls, cat, pip, etc.)
    execution_events = []
    error_type_counts: dict[str, int] = {}

    for call in tool_calls:
        tool_name = call.get("tool", "")
        if tool_name != "execute_bash":
            continue

        args = call.get("arguments", {})
        cmd = args.get("command", "")

        # Only track actual script executions, not ls/cat/pip/etc.
        if not _is_target_execution(cmd):
            continue

        turn = call.get("turn", 0)
        result = results_by_turn.get(turn, {})
        exit_code = result.get("exit_code")
        timed_out = result.get("timed_out", False)
        summary = str(result.get("summary", "")).lower()

        is_error = (
            exit_code is not None and exit_code != 0
        ) or timed_out or "error" in summary or "traceback" in summary

        error_type = _classify_error(summary) if is_error else None

        # Count error types for aggregate statistics
        if error_type:
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

        execution_events.append({
            "turn": turn,
            "is_error": is_error,
            "error_type": error_type,
        })

    total_errors = sum(1 for e in execution_events if e["is_error"])

    if total_errors == 0:
        return ErrorRecoveryMetrics(
            total_errors=0,
            errors_recovered=0,
            recovery_rate=1.0,  # Perfect - no errors to recover from
            consecutive_failures=0,
            persistence_score=1.0,
            error_types={},
        )

    # Count recoveries: execution error followed by execution success
    errors_recovered = 0
    max_consecutive = 0
    current_streak = 0

    for i, e in enumerate(execution_events):
        if e["is_error"]:
            current_streak += 1
            max_consecutive = max(max_consecutive, current_streak)

            # Check if recovered (next execution succeeds)
            if i + 1 < len(execution_events) and not execution_events[i + 1]["is_error"]:
                errors_recovered += 1
        else:
            current_streak = 0

    recovery_rate = errors_recovered / total_errors if total_errors > 0 else 0.0

    # Persistence score: combination of recovery rate and not giving up
    # Penalize for long consecutive failure streaks
    streak_penalty = min(1.0, max_consecutive / 5.0)  # 5+ consecutive failures = max penalty
    persistence_score = max(0.0, recovery_rate * (1.0 - streak_penalty * 0.5))

    return ErrorRecoveryMetrics(
        total_errors=total_errors,
        errors_recovered=errors_recovered,
        recovery_rate=recovery_rate,
        consecutive_failures=max_consecutive,
        persistence_score=persistence_score,
        error_types=error_type_counts,
    )


# Maximum bytes to keep in stdout sample (last N bytes preserved)
_MAX_STDOUT_SAMPLE_BYTES = 4096


def _parse_expected_scripts(task_prompt: str, capsule_id: str = "") -> list[str]:
    """Extract expected script filenames from task_prompt.

    Parses script filenames mentioned in the task prompt that the agent should execute.
    This enables computing execution_coverage: what fraction of expected scripts were run.

    Supported patterns (covers 62/66 = 94% of CORE-Bench tasks):
        - Python scripts: .py
        - Jupyter notebooks: .ipynb
        - Shell scripts: .sh
        - R scripts & markdown: .R, .Rmd

    For 4 capsules with vague "run all" prompts, we hardcode the intended scripts
    based on the task_prompt. 
    """
    if not task_prompt:
        return []

    # special cases 
    # "Run the python files..."
    if capsule_id == "capsule-8536428":
        return ["n_gram__combined.py", "empath_train.py"]

    # "Run 'physalia_automators.reports' as a python module"
    if capsule_id == "capsule-3593259":
        return ["reports.py"]  # Matches both "reports.py" and module extraction

    # "Run all the .Rmd files using Rscript and render them as html"
    if capsule_id == "capsule-2345790":
        return [
            "Study1-encoding_analyses.Rmd",
            "Study1-recall_analyses.Rmd",
            "Study2-encoding_analyses.Rmd",
            "Study2-recall_analyses.Rmd",
            "Study3-encoding_analyses.Rmd",
            "Study3-recall_analyses.Rmd",
            "Study4-encoding_analyses.Rmd",
            "Study4-recall_analyses.Rmd",
            "Study1-2-4-combined_analyses.Rmd",
            "Publication_figures.Rmd",
        ]

    # "Run all the .R scripts in the ../code folder"
    if capsule_id == "capsule-5136217":
        return [
            "2_classify_political.R",
            "3_merge_survey.R",
            "4_descriptive_analysis.R",
            "5_custom_panels.R",
            "6_event_study.R",
            "7_event_study_party_binsR.R",
            "8_usage_tables.R",
            "9_weighted_event_study.R",
            "10_plot_Gtrends.R",
            "11_prepare_for_publication.R",
            "ISSUES_2_classify_political copy.R",
            "ISSUES_3_merge_survey.R",
            "ISSUES_4_descriptive_analysis.R",
        ]

    # general cases
    scripts = []

    # Python scripts (.py), ex: "Run step_0_vit_encode.py, then step_1_train.py"
    py_matches = re.findall(r"['\"]?(\S+\.py)['\"]?", task_prompt)
    scripts.extend(py_matches)

    # Jupyter notebooks (.ipynb), ex: "Run the jupyter notebook visualize_results.ipynb"
    ipynb_matches = re.findall(r"['\"]?(\S+\.ipynb)['\"]?", task_prompt)
    scripts.extend(ipynb_matches)

    # Shell scripts (.sh), ex: "Run the bash script demo.sh"
    sh_matches = re.findall(r"['\"]?(\S+\.sh)['\"]?", task_prompt)
    scripts.extend(sh_matches)

    # R scripts (.R), ex: "Run pancancer_calculation.R using Rscript"
    r_matches = re.findall(r"['\"]?(\S+\.R)['\"]?(?!\w)", task_prompt)
    scripts.extend(r_matches)

    # R Markdown (.Rmd), ex: "Run manuscript.Rmd using Rscript and render it as a pdf"
    rmd_matches = re.findall(r"['\"]?(\S+\.Rmd)['\"]?", task_prompt)
    scripts.extend(rmd_matches)

    # Deduplicate while preserving order, normalize to basenames
    seen = set()
    unique_scripts = []
    for s in scripts:
        s_clean = os.path.basename(s.strip("'\""))
        if s_clean not in seen:
            seen.add(s_clean)
            unique_scripts.append(s_clean)

    return unique_scripts


def _extract_executed_scripts(
    tool_calls: list[dict],
    tool_results: list[dict],
) -> tuple[list[str], list[str], int, str]:
    """Extract executed script filenames and stdout from tool calls.

    Tracks both successful executions (exit_code=0) and attempted but failed scripts.
    Also captures stdout from successful script executions for diagnostics.
    Uses turn-based matching (not array indices) for robustness.

    Args:
        tool_calls: List of tool_call events from trace
        tool_results: List of tool_result events from trace

    Returns:
        Tuple of (executed_scripts, attempted_failed_scripts, stdout_total_bytes, stdout_sample)
    """
    if not tool_calls:
        return [], [], 0, ""

    # Build results lookup by turn (robust matching)
    results_by_turn: dict[int, dict] = {}
    for result in tool_results:
        turn = result.get("turn", 0)
        results_by_turn[turn] = result

    executed = []
    attempted_failed = []
    all_stdout_parts = []
    total_stdout_bytes = 0

    for call in tool_calls:
        tool_name = call.get("tool", "")
        if tool_name != "execute_bash":
            continue

        args = call.get("arguments", {})
        cmd = args.get("command", "")
        if not cmd:
            continue

        turn = call.get("turn", 0)
        result = results_by_turn.get(turn, {})

        # Check if execution succeeded
        exit_code = result.get("exit_code")
        is_success = exit_code == 0

        # Extract script names from the command
        scripts_in_cmd = []

        # Python scripts (.py), ex: "python script.py", "python3 -u main.py"
        py_exec = re.findall(r"python[0-9.]*\s+(?:-[^\s]+\s+)*(\S+\.py)", cmd)
        scripts_in_cmd.extend(py_exec)

        # Python modules via -m flag, ex: "python -m package.module", "python3 -m mymodule"
        # Extracts the last component as "module.py" for matching against expected scripts
        py_module = re.findall(r"python[0-9.]*\s+(?:-[^\s]+\s+)*-m\s+(\S+)", cmd)
        for mod in py_module:
            # Skip utility modules (lowercase for case-insensitive matching)
            base_module = mod.split(".")[0].lower()
            if base_module in _NON_SCRIPT_MODULES:
                continue
            # Convert "package.subpackage.module" to "module.py"
            module_name = mod.split(".")[-1] + ".py"
            scripts_in_cmd.append(module_name)

        # Shell scripts (.sh), ex: "bash demo.sh", "./run.sh"
        sh_exec = re.findall(r"(?:bash\s+|\./)(\S+\.sh)", cmd)
        scripts_in_cmd.extend(sh_exec)

        # Jupyter notebooks (.ipynb), ex: "jupyter nbconvert --to html --execute notebook.ipynb"
        ipynb_exec = re.findall(r"(?:jupyter|nbconvert)\s+.*?(\S+\.ipynb)", cmd)
        scripts_in_cmd.extend(ipynb_exec)

        # Code Ocean entry point ./run or bash run (no extension) (for medium difficulty tasks)
        run_exec = re.findall(r"(?:bash\s+|\./)(run)(?:\s|$)", cmd)
        scripts_in_cmd.extend(run_exec)

        # R scripts (.R) via Rscript, ex: "Rscript analysis.R"
        r_exec = re.findall(r"Rscript\s+(?!-e)(?:-[^\s]+\s+)*['\"]?(\S+\.R)\b['\"]?", cmd)
        scripts_in_cmd.extend(r_exec)

        # R scripts via source() in Rscript -e, ex: "Rscript -e \"source('script.R', echo=TRUE)\""
        source_exec = re.findall(r"source\s*\(\s*['\"]([^'\"]+\.R)['\"]", cmd, re.IGNORECASE)
        scripts_in_cmd.extend(source_exec)

        # Shell for loops with R script globs, ex: "for f in code/*.R; do ..."
        # Marks all expected .R scripts as attempted when a glob pattern is used
        r_glob_match = re.search(r"for\s+\w+\s+in\s+([^\s;]+/\*\.R)", cmd)
        if r_glob_match:
            scripts_in_cmd.append("__ALL_R_SCRIPTS__")

        # R Markdown (.Rmd) via rmarkdown::render(), ex: "Rscript -e \"rmarkdown::render('report.Rmd')\""
        rmd_exec = re.findall(r"render\(['\"]([^'\"]+\.Rmd)['\"]", cmd)
        scripts_in_cmd.extend(rmd_exec)

        # If we found scripts in this command, track them
        if scripts_in_cmd:
            if is_success:
                executed.extend(scripts_in_cmd)
                stdout_text = str(result.get("summary", ""))
                if stdout_text.strip():
                    all_stdout_parts.append(stdout_text)
                    total_stdout_bytes += len(stdout_text)
            else:
                # Track attempted but failed scripts
                attempted_failed.extend(scripts_in_cmd)

    # Deduplicate and get basenames for successful executions
    seen = set()
    unique = []
    for script in executed:
        basename = os.path.basename(script)
        if basename not in seen:
            seen.add(basename)
            unique.append(basename)

    # Deduplicate attempted_failed (exclude those that eventually succeeded)
    failed_seen = set()
    unique_failed = []
    for script in attempted_failed:
        basename = os.path.basename(script)
        # Only include if not already in successful executions
        if basename not in seen and basename not in failed_seen:
            failed_seen.add(basename)
            unique_failed.append(basename)

    # Build stdout sample (truncate to last N bytes if too large)
    # decide whether to keep these or not, are not used for anything right now
    combined_stdout = "\n---\n".join(all_stdout_parts)
    if len(combined_stdout) > _MAX_STDOUT_SAMPLE_BYTES:
        stdout_sample = (
            f"... (truncated, {total_stdout_bytes:,} bytes total) ...\n"
            + combined_stdout[-_MAX_STDOUT_SAMPLE_BYTES:]
        )
    else:
        stdout_sample = combined_stdout

    return unique, unique_failed, total_stdout_bytes, stdout_sample


def _compute_execution_coverage(
    expected_scripts: list[str],
    executed_scripts: list[str],
    successful_execution: bool,
    attempted_failed_scripts: list[str] | None = None,
    include_failed_attempts: bool = False,
) -> float:
    """Compute what fraction of expected scripts were executed (or attempted).

    Args:
        expected_scripts: Scripts parsed from task_prompt
        executed_scripts: Scripts successfully executed
        successful_execution: Whether execution completed successfully (exit code 0)
        attempted_failed_scripts: Scripts that were attempted but failed
        include_failed_attempts: If True, count failed attempts toward coverage (for HARD mode)

    Returns:
        Coverage from 0.0 to 1.0
    """
    # No expected scripts parsed from task_prompt
    if not expected_scripts:
        return 1.0 if successful_execution else 0.0

    expected_basenames = {os.path.basename(s) for s in expected_scripts}
    executed_basenames = {os.path.basename(s) for s in executed_scripts}

    if include_failed_attempts and attempted_failed_scripts:
        failed_basenames = {os.path.basename(s) for s in attempted_failed_scripts}
        all_attempted = executed_basenames | failed_basenames
    else:
        all_attempted = executed_basenames

    # Handle __ALL_R_SCRIPTS__ marker: when a for-loop glob was used (e.g., "for f in code/*.R"),
    if "__ALL_R_SCRIPTS__" in all_attempted:
        all_attempted.discard("__ALL_R_SCRIPTS__")
        for expected in expected_basenames:
            if expected.endswith(".R"):
                all_attempted.add(expected)

    matched = expected_basenames & all_attempted

    # Special case: "run" matches "run.sh" (for medium difficulty tasks)
    if "run" in expected_basenames and "run" not in matched:
        if "run.sh" in all_attempted:
            matched.add("run")

    # Jupyter notebooks with slightly different output names i.e. "Notebook.ipynb" matches "exec_Notebook.ipynb"
    for expected in expected_basenames - matched:
        if not expected.endswith(".ipynb"):
            continue
        expected_stem = expected[:-6]  # Remove ".ipynb"
        for attempted in all_attempted:
            if expected_stem in attempted and attempted.endswith(".ipynb"):
                matched.add(expected)
                break

    return len(matched) / len(expected_basenames)


def extract_methodology_metrics(tool_calls: list[dict], tool_results: list[dict],
    domain: str, task_prompt: str = "", deleted_files: list[str] | None = None,
    capsule_id: str = "",
) -> MethodologyMetrics:
    """Extract deterministic process metrics from trace events.

    This function analyzes tool_call and tool_result events to measure:
    - Whether the agent read documentation
    - Whether the agent attempted execution
    - Whether execution succeeded
    - Whether dependencies were installed
    - Error recovery patterns
    - Anti-patterns (like reading pre-existing results in hard mode)
    """
    # Initialize tracking variables
    docs_read: list[str] = []
    scripts_read: list[str] = []
    read_documentation = False
    read_target_script = False
    attempted_execution = False
    execution_attempts = 0
    successful_execution = False
    installed_dependencies = False
    read_preexisting_results = False
    violations: list[str] = []

    # Track when successful execution first happened (for violation detection)
    first_successful_execution_turn: int | None = None
    results_reads_before_execution: list[tuple[int, str]] = []  # (turn, file_path)

    # Build results lookup for exit codes
    results_by_turn: dict[int, dict] = {}
    for result in tool_results:
        turn = result.get("turn", 0)
        results_by_turn[turn] = result

    # Analyze each tool call
    for call in tool_calls:
        tool = call.get("tool", "")
        args = call.get("arguments", {})
        turn = call.get("turn", 0)
        result = results_by_turn.get(turn, {})

        if tool == "inspect_file_as_text":
            file_path = args.get("file_path", "")
            summary = str(result.get("summary", ""))
            read_failed = summary.startswith("Error") or "not found" in summary.lower() or "not a regular file" in summary.lower()
            
            # Check for documentation reads
            if _is_documentation(file_path) and not read_failed:
                if file_path not in docs_read:  # Deduplicate
                    docs_read.append(file_path)
                read_documentation = True

            # Check for reading target script (code files)
            # Includes Python, Shell, R, R Markdown, Julia, MATLAB, Jupyter notebooks
            if file_path.endswith((".py", ".sh", ".r", ".R", ".Rmd", ".jl", ".m", ".ipynb")) and not read_failed:
                if file_path not in scripts_read:  # Deduplicate
                    scripts_read.append(file_path)
                read_target_script = True

            # Track results/ reads for later violation analysis
            if domain == "corebench_hard":
                if "results/" in file_path or "/results" in file_path or "result_" in file_path:
                    results_reads_before_execution.append((turn, file_path))

        elif tool == "query_vision_language_model":
            # Track vision model queries on results/output files (images in results folders)
            image_path = args.get("image_path", "")
            if domain == "corebench_hard":
                # Check if querying images in results-like directories
                if ("results/" in image_path or "/results" in image_path or
                    "result_" in image_path or "output" in image_path.lower()):
                    results_reads_before_execution.append((turn, f"[vision] {image_path}"))

        elif tool == "execute_bash":
            cmd = args.get("command", "")
            exit_code = result.get("exit_code")

            # Check for execution attempts
            if _is_target_execution(cmd):
                attempted_execution = True
                execution_attempts += 1

                if exit_code == 0:
                    successful_execution = True
                    # Record the first successful execution turn
                    if first_successful_execution_turn is None:
                        first_successful_execution_turn = turn

            # Check for dependency installation
            if _is_dependency_install(cmd):
                installed_dependencies = True

            # Check for script reads via bash commands (sed, cat, head, etc.)
            # This fixes the gap where agents reading scripts via bash weren't credited
            if exit_code == 0:  # Only credit successful reads
                bash_script_reads = _extract_script_reads_from_bash(cmd)
                for script_path in bash_script_reads:
                    if script_path not in scripts_read:
                        scripts_read.append(script_path)
                    read_target_script = True

    # Now determine violations: only flag results reads that happened BEFORE successful execution
    if domain == "corebench_hard":
        for read_turn, file_path in results_reads_before_execution:
            # If no successful execution, any results read is suspicious
            if first_successful_execution_turn is None or read_turn < first_successful_execution_turn:
                read_preexisting_results = True
                # happened for capsule-9670283 where it used vision model of pre-existing results image
                if file_path.startswith("[vision]"):
                    violations.append(f"Queried pre-existing results image: {file_path[9:]}")
                else:
                    violations.append(f"Read pre-existing results: {file_path}")

    # Compute error recovery metrics
    error_recovery = _compute_error_recovery(tool_calls, tool_results)

    # Parse expected scripts from task_prompt and extract executed scripts
    expected_scripts = _parse_expected_scripts(task_prompt, capsule_id)

    # Filter out scripts that were deleted by difficulty filter (e.g., run.sh in hard mode)
    # These are impossible to execute, so we shouldn't penalize for not running them
    if deleted_files:
        deleted_basenames = set()
        for df in deleted_files:
            # Extract basename from paths like "environment/code/run.sh" or "environment/code/run"
            basename = os.path.basename(df.rstrip("/"))
            if basename:
                deleted_basenames.add(basename)

        original_count = len(expected_scripts)
        expected_scripts = [s for s in expected_scripts if os.path.basename(s) not in deleted_basenames]

        if len(expected_scripts) < original_count:
            # Log that we filtered out as impossible-to-execute scripts
            filtered_out = original_count - len(expected_scripts)
            logger.info("Filtered out %d expected scripts that were deleted by difficulty filter.", filtered_out)
            pass

    executed_scripts, attempted_failed_scripts, stdout_total_bytes, stdout_sample = _extract_executed_scripts(
        tool_calls, tool_results
    )

    # Compute coverage metrics:
    success_coverage = _compute_execution_coverage(expected_scripts, executed_scripts, successful_execution)
    attempt_coverage = _compute_execution_coverage(
        expected_scripts, executed_scripts, successful_execution,
        attempted_failed_scripts=attempted_failed_scripts,
        include_failed_attempts=True,
    )
    stdout_captured = stdout_total_bytes > 0

    # Compute methodology score based on domain (violations are tracked separately for diagnostics)
    methodology_score, score_breakdown = _compute_methodology_score(
        domain=domain,
        read_documentation=read_documentation,
        read_target_script=read_target_script,
        attempted_execution=attempted_execution,
        successful_execution=successful_execution,
        installed_dependencies=installed_dependencies,
        error_recovery=error_recovery,
        attempt_coverage=attempt_coverage,
    )

    return MethodologyMetrics(
        read_documentation=read_documentation,
        docs_read=docs_read,
        read_target_script=read_target_script,
        scripts_read=scripts_read,
        attempted_execution=attempted_execution,
        execution_attempts=execution_attempts,
        successful_execution=successful_execution,
        expected_scripts=expected_scripts,
        executed_scripts=executed_scripts,
        attempted_failed_scripts=attempted_failed_scripts,
        execution_coverage=success_coverage,
        stdout_captured=stdout_captured,
        stdout_total_bytes=stdout_total_bytes,
        stdout_sample=stdout_sample,
        installed_dependencies=installed_dependencies,
        error_recovery=error_recovery,
        read_preexisting_results=read_preexisting_results,
        violations=violations,
        methodology_score=methodology_score,
        score_breakdown=score_breakdown,
    )


def _compute_methodology_score(
    domain: str,
    read_documentation: bool,
    read_target_script: bool,
    attempted_execution: bool,
    successful_execution: bool,
    installed_dependencies: bool,
    error_recovery: ErrorRecoveryMetrics,
    attempt_coverage: float = 0.0,
) -> tuple[float, MethodologyScoreBreakdown]:
    """Compute a deterministic methodology score from 0-1 based on observed events.

    Scoring weights by domain:
                        EASY    MEDIUM  HARD
    Doc reading:        100%    25%     15%
    Script reading:     -       15%     20%
    Exec components:    -       60%     65%
    Error recovery:     -       -       - (logged only)

    Args:
        attempt_coverage: Fraction of expected scripts attempted, including failures (0.0-1.0)

    HARD mode execution scoring (0.65 total):
    - attempt_coverage (0.45): Credit for attempting expected scripts
      Full proportional credit even if scripts fail - running the correct script is the challenge in hardmode. 
    - successful_execution (0.20): Bonus for any script completing successfully, for creative solutions. 

    Tier ordering ensures "tried right thing" beats "random luck":
    1. Expected succeeds: 0.45 + 0.20 = 0.65
    2. Expected fails: 0.45 + 0.00 = 0.45
    3. Random succeeds: 0.15 + 0.20 = 0.35
    4. Random fails: 0.15 + 0.00 = 0.15

    Returns:
        Tuple of (final_score, breakdown) where breakdown shows each component's contribution.
    """
    # Initialize component scores
    doc_read_score = 0.0
    script_read_score = 0.0
    execution_coverage_score = 0.0
    successful_execution_score = 0.0
    error_recovery_score = 0.0
    penalty = 0.0

    if domain == "corebench_easy":
        # Full credit for reading files, penalty for execution attempts
        if not attempted_execution:
            doc_read_score = 1.0
        else:
            penalty = -0.7 

    elif domain == "corebench_medium":
        if read_documentation:
            doc_read_score = 0.25
        if read_target_script:
            script_read_score = 0.15
        if attempt_coverage > 0:
            execution_coverage_score = 0.40 * attempt_coverage
        elif attempted_execution:
            execution_coverage_score = 0.10  # Partial credit for random/creative script
        if successful_execution:
            successful_execution_score = 0.20
        error_recovery_score = error_recovery.persistence_score * 0.15

    elif domain == "corebench_hard":
        if read_documentation:
            doc_read_score = 0.15
        if read_target_script:
            script_read_score = 0.20
        if attempt_coverage > 0:
            execution_coverage_score = 0.45 * attempt_coverage  # attempted expected scripts
        elif attempted_execution:
            execution_coverage_score = 0.15  # Partial credit for random/creative script
        if successful_execution:
            successful_execution_score = 0.20  # Bonus for any script completing successfully
        # compute for logging errors encountered but doesn't affect scoring
        error_recovery_score = error_recovery.persistence_score
        # Penalty for not attempting dependency installation when execution failed (MIGHT ZERO OUT)
        if not successful_execution and not installed_dependencies and attempted_execution:
            penalty = -0.05

    raw_total = (doc_read_score + script_read_score + execution_coverage_score +
                     successful_execution_score + penalty)
    final_score = min(1.0, max(0.0, raw_total))

    breakdown = MethodologyScoreBreakdown(
        domain=domain,
        doc_read_score=doc_read_score,
        script_read_score=script_read_score,
        execution_coverage_score=execution_coverage_score,
        successful_execution_score=successful_execution_score,
        error_recovery_score=error_recovery_score,
        penalty=penalty,
        total=final_score,
    )

    return final_score, breakdown


# =============================================================================
# AGGREGATE METRICS
# =============================================================================

def aggregate_results(evaluations: list[TaskEvaluation]) -> AggregateMetrics:
    """Aggregate metrics across multiple task evaluations."""
    if not evaluations:
        return AggregateMetrics(
            num_tasks=0,
            num_successful=0,
            pass_rate=0.0,
            mean_accuracy=0.0,
            mean_written_accuracy=0.0,
            mean_vision_accuracy=0.0,
            mean_adherence=0.0,
            mean_methodology_score=0.0,
            doc_read_rate=0.0,
            execution_attempt_rate=0.0,
            successful_execution_rate=0.0,
            mean_error_recovery_rate=0.0,
            task_results={},
            error_type_distribution={},
        )

    num_tasks = len(evaluations)
    num_successful = sum(1 for e in evaluations if e.success)

    # Accuracy means
    mean_accuracy = np.mean([e.accuracy.accuracy for e in evaluations])
    mean_written = np.mean([e.accuracy.written_accuracy for e in evaluations])
    mean_vision = np.mean([e.accuracy.vision_accuracy for e in evaluations])

    # Adherence
    mean_adherence = np.mean([e.task_adherence.score for e in evaluations])

    # Methodology metrics (deterministic)
    methodology_evals = [e for e in evaluations if e.methodology_metrics is not None]
    if methodology_evals:
        mean_methodology_score = np.mean([e.methodology_metrics.methodology_score for e in methodology_evals])
        doc_read_rate = sum(1 for e in methodology_evals if e.methodology_metrics.read_documentation) / len(methodology_evals)
        execution_attempt_rate = sum(1 for e in methodology_evals if e.methodology_metrics.attempted_execution) / len(methodology_evals)
        successful_execution_rate = sum(1 for e in methodology_evals if e.methodology_metrics.successful_execution) / len(methodology_evals)
        mean_error_recovery_rate = np.mean([e.methodology_metrics.error_recovery.recovery_rate for e in methodology_evals])
    else:
        mean_methodology_score = 0.0
        doc_read_rate = 0.0
        execution_attempt_rate = 0.0
        successful_execution_rate = 0.0
        mean_error_recovery_rate = 0.0

    # Per-task summary
    task_results = {
        e.task_id: {
            "success": bool(e.success),
            "accuracy": round(float(e.accuracy.accuracy), 4),
            "adherence": round(float(e.task_adherence.score), 4),
            "methodology_score": round(float(e.methodology_metrics.methodology_score), 4) if e.methodology_metrics else None,
        }
        for e in evaluations
    }

    # Aggregate error type distribution across all tasks
    error_type_distribution: dict[str, int] = {}
    for e in methodology_evals:
        if e.methodology_metrics and e.methodology_metrics.error_recovery:
            for error_type, count in e.methodology_metrics.error_recovery.error_types.items():
                error_type_distribution[error_type] = error_type_distribution.get(error_type, 0) + count

    return AggregateMetrics(
        num_tasks=num_tasks,
        num_successful=num_successful,
        pass_rate=num_successful / num_tasks,
        mean_accuracy=float(mean_accuracy),
        mean_written_accuracy=float(mean_written),
        mean_vision_accuracy=float(mean_vision),
        mean_adherence=float(mean_adherence),
        mean_methodology_score=float(mean_methodology_score),
        doc_read_rate=float(doc_read_rate),
        execution_attempt_rate=float(execution_attempt_rate),
        successful_execution_rate=float(successful_execution_rate),
        mean_error_recovery_rate=float(mean_error_recovery_rate),
        task_results=task_results,
        error_type_distribution=error_type_distribution,
    )
