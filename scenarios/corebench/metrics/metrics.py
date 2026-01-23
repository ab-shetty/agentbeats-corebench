"""
CoreBench Evaluation Metrics Module
====================================

This module provides metrics for evaluating the purple agent's performance
on the CoreBench reproducibility benchmark. The benchmark (agentified for AgentBeats) tests
whether agents can reproduce both the environment and results of published research papers.

ACCURACY: 
- Measures answer correctness
- Are the submitted answers correct compared to ground truth? (does not care about how agent reached them)
   - Numeric values: Uses 95% prediction intervals to handle run-to-run variance
     in stochastic experiments (e.g., ML training with different random seeds).
     The interval is computed using t-distribution: mean ± t(0.975, n-1) * std * sqrt(1 + 1/n)
   - String values: Case-insensitive exact match after stripping whitespace
   - List values: Element-wise exact comparison (order matters, types matter)

    Ground Truth For Accuracy Evaluation (from core_test.json):
    [
        {"question1": 0.95, "question2": "label", ...},
    ]

METHODOLOGY SCORE:
- Measures reproduction process fidelity (what makes this a reproducibility benchmark)
- Measures whether the agent followed proper scientific reproduction methodology, reading documentation, 
  executing code, handling errors, regardless of whether they got the correct final answer.
    - Deterministically catches cases where the agent got the correct answer through shortcuts (non-reproduction) means
      ex: extracting x-axis label from code instead of reproducing the graph).
    - Differentiates "honest success" vs "shortcut success"

TASK ADHERENCE:
- LLM-as-Judge serves as a qualitative judgement of process. 
    - Evaluates rule compliance, and problem-solving approach
    - Assesses how well the agent understood and executed the task
    - Returns qualitative assessment [excellent/good/fair/poor] + [strengths] & [weaknesses]
    - Can be used as validation of the methodology score

EFFICIENCY: 
- How resource-efficient was the agent's approach?
   - Steps used vs maximum allowed
   - Tool call count and execution time
   - Protocol/format errors encountered
"""

from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional
import json
import logging
import math
import os

import re

import ast
import numpy as np
from scipy.stats import t
import litellm

logger = logging.getLogger("evaluator.metrics")

# =============================================================================
# VISION QUESTION DETECTION
# =============================================================================

# Regex pattern for detecting vision-related questions based on CORE-Bench paper:
_VISION_KEY_PATTERN = re.compile(r"^\s*fig(?:ure)?s?\b", re.IGNORECASE)

def _is_vision_question(key: str) -> bool:
    """
    Check if a question key indicates vision/figure-related content.
    
    Based on CORE-Bench paper definition: questions requiring extraction of
    results from figures, graphs, plots, charts, or images.
    """
    return bool(_VISION_KEY_PATTERN.search(key))


# =============================================================================
# JSON SERIALIZATION HELPERS
# =============================================================================

def _make_json_safe(obj: Any) -> Any:
    """Recursively convert an object to be JSON serializable.
    
    Handles numpy types, dataclasses, and nested structures.
    """
    if obj is None:
        return None
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    elif hasattr(obj, '__dataclass_fields__'):
        return _make_json_safe(asdict(obj))
    else:
        return obj


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QuestionResult:
    """Result for a single evaluation question."""
    question: str
    question_type: str  # "numeric", "string", "list"
    is_vision: bool  # True if question requires figure analysis
    correct: bool
    submitted: Any
    expected: Any = None
    prediction_interval: Optional[dict] = None  # For numeric: {"lower": x, "upper": y}


@dataclass 
class AccuracyMetrics:
    """Metrics for answer correctness."""
    # Overall
    total_questions: int
    correct_answers: int
    accuracy: float

    # By modality
    total_written: int
    correct_written: int
    written_accuracy: float

    total_vision: int
    correct_vision: int
    vision_accuracy: float

    # Detailed breakdown
    question_results: list[QuestionResult]
    missing_questions: list[str]
    extra_questions: list[str]


@dataclass
class TaskAdherenceMetrics:
    """LLM-as-Judge metric for task execution quality, providing qualitative assessment and feedback.

    Evaluates how well the agent followed task instructions, navigated the codebase,
    and solved problems.
    """
    score: float                # Overall quality score from 0.0 (poor) to 1.0 (excellent)
    followed_instructions: bool # Whether the agent followed task prompt and documentation
    reasoning: str              # LLM judge's explanation of assessment score
    strengths: list[str]        # List of things the agent did well
    weaknesses: list[str]       # List of areas for improvement
    status: str = "success"     # "success" if LLM judge evaluation completed, "error" if judge API call failed
    error_message: Optional[str] = None


@dataclass
class ErrorRecoveryMetrics:
    """Metrics for tracking error recovery behavior.

    Measures how well the agent handles and recovers from errors during execution.
    A key predictor of success that captures agent persistence.

    NOTE the _classify_error() function was added for
    computing error classifications but they were being discarded. Error type
    distribution is valuable for diagnostics (e.g., identifying if agents struggle
    with import_error vs timeout vs file_not_found).
    """
    total_errors: int           # Total number of errors encountered
    errors_recovered: int       # Number of errors where agent successfully continued
    recovery_rate: float        # Fraction of (errors_recovered / total_errors)
    consecutive_failures: int   # Maximum consecutive failed attempts
    persistence_score: float    # Combined score 0.0-1.0 reflecting recovery behavior
    error_types: dict[str, int] # Count of each error type (from _classify_error)


@dataclass
class MethodologyMetrics:
    """Deterministic methodology metrics extracted from execution traces.

    These metrics capture observable behaviors without relying on LLM judgment,
    providing objective measurement of whether the agent followed correct methodology.
    """
    # Discovery phase
    read_documentation: bool        # Whether agent read README/REPRODUCING.md
    docs_read: list[str]            # List of documentation files that were read
    read_target_script: bool        # Whether agent inspected the script to run

    # Execution phase
    attempted_execution: bool       # Whether agent tried to run target script
    execution_attempts: int         # Number of execution attempts
    successful_execution: bool      # Whether at least one execution succeeded (exit_code=0)

    # Script coverage               - from task_prompt analysis
    expected_scripts: list[str]         # Scripts parsed from task_prompt that should be executed
    executed_scripts: list[str]         # Scripts the agent actually ran successfully
    execution_coverage: float           #  Fraction of expected scripts that were run (0.0-1.0)

    # Stdout diagnostics            - for "ran code but output to stdout" cases)
    # NOTE (2026-01-22): Added stdout_sample field - _extract_executed_scripts() was
    # computing stdout sample but it was being discarded with `_` prefix. Stdout sample
    # is valuable for diagnosing "ran code but got wrong answer" cases where the agent
    # may have output results to stdout instead of writing to files.
    stdout_captured: bool                   # Whether any stdout was captured from successful script runs
    stdout_total_bytes: int                 # Total bytes of stdout captured
    stdout_sample: str                      # Last N bytes of stdout (truncated if too large)

    # Dependencies 
    installed_dependencies: bool            # Whether agent ran pip/apt install commands

    # Recovery
    error_recovery: ErrorRecoveryMetrics    # Detailed error recovery metrics

    # Anti-patterns
    read_preexisting_results: bool          # Whether agent read results/ before executing
    violations: list[str]                   # List of detected anti-pattern descriptions

    # Final score
    methodology_score: float = 0.0          # Combined deterministic methodology score (0.0-1.0)


@dataclass
class EfficiencyMetrics:
    """Resource efficiency metrics.
    
    Tracks resource usage including steps, tool calls, time, and errors.
    The command_timeouts field is useful for identifying when agents hit
    infrastructure limits (e.g., slow Docker/emulation on ARM64).
    """
    steps_used: int
    max_steps: int
    tool_calls: int
    time_seconds: float
    protocol_errors: int
    command_timeouts: int = 0  # Commands that hit timeout limit

    @property
    def step_efficiency(self) -> float:
        """Fraction of max steps NOT used (higher = more efficient)."""
        if self.max_steps == 0:
            return 0.0
        return 1.0 - (self.steps_used / self.max_steps)


@dataclass
class TaskEvaluation:
    """Complete evaluation result for a single task."""
    task_id: str
    domain: str
    success: bool  # True if all questions answered correctly

    accuracy: AccuracyMetrics
    task_adherence: TaskAdherenceMetrics
    efficiency: EfficiencyMetrics

    # Raw data for debugging
    submitted_answer: Any
    ground_truth: list[dict]

    # Cost tracking (None if not available)
    task_cost: Optional[float] = None
    
    # Process metrics (deterministic extraction from traces)
    methodology_metrics: Optional[MethodologyMetrics] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Uses _make_json_safe to handle numpy types and other non-serializable objects.
        """
        return _make_json_safe({
            "task_id": self.task_id,
            "domain": self.domain,
            "success": self.success,
            "accuracy": asdict(self.accuracy),
            "task_adherence": asdict(self.task_adherence),
            "efficiency": asdict(self.efficiency),
            "task_cost": self.task_cost,
            "methodology_metrics": asdict(self.methodology_metrics) if self.methodology_metrics else None,
        })


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
    
    Args:
        ground_truth: List of result dicts from multiple runs (from core_test.json)
        submitted: Agent's submitted answer dict
    """
    if not ground_truth or not ground_truth[0]:
            logger.error("Empty or invalid ground truth provided")
            return _empty_accuracy_metrics()
    
    # Ensure submitted is a dict (handle malformed answers like int, str, etc.)
    if not isinstance(submitted, dict):
        logger.warning(f"Submitted answer is not a dict (got {type(submitted).__name__}), treating as empty")
        submitted = {}
    
    # Normalize submitted answers (handle % signs, convert strings to numbers where possible)
    submitted = _normalize_submitted(submitted.copy() if submitted else {})
    
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
            # Missing answer
            result = QuestionResult(
                question=key,
                question_type=_get_type(key, numeric_keys, string_keys, list_keys),
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
    
    # Identify extra questions submitted but not required
    extra_questions = [k for k in submitted.keys() if k not in required_questions]
    missing_questions = [k for k in required_questions if k not in submitted]

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
        missing_questions=missing_questions,
        extra_questions=extra_questions,
    )


def _extract_numeric_value(text: str) -> Optional[float]:
    """
    Extract a numeric value from text, handling various formats.

    Handles:
    - Commas in numbers: "1,200,000" -> 1200000.0
    - Dollar signs: "$6,003.03" -> 6003.03
    - Percentage signs: "96.5%" -> 96.5
    - Extra text: "$6,003.03 per person" -> 6003.03
    - Word multipliers (when no currency symbol): "~1.2 Million" -> 1200000.0

    Note on word multipliers:
    - When there's a currency symbol ($), we DON'T apply word multipliers because
      the multiplier is likely a unit label, not a scaling factor.
      E.g., "$1.80 Trillion USD" -> 1.80 (question already asks in "Trillion USD" units)
    - When there's NO currency symbol, we DO apply word multipliers.
      E.g., "~1.2 Million" -> 1200000.0 (question asks for absolute count)

    Returns:
        Extracted float value, or None if no valid number found.
    """
    text_lower = text.lower().strip()
    has_currency = '$' in text or '€' in text or '£' in text

    # Remove common prefixes/symbols
    cleaned = text.replace('$', '').replace('€', '').replace('£', '').replace('~', '').replace('%', '').strip()

    # Try to find a number pattern (with optional commas and decimals)
    # Pattern matches: 1,234,567.89 or 1234567.89 or 1234567 or .89
    number_pattern = r'[\d,]+\.?\d*|\.\d+'
    match = re.search(number_pattern, cleaned)

    if not match:
        return None

    try:
        # Remove commas and convert to float
        num_str = match.group().replace(',', '')
        value = float(num_str)

        # Only apply word multipliers if NO currency symbol present
        # Currency values with "Trillion/Million" are usually labeled units, not multipliers
        # E.g., "$1.80 Trillion USD" means 1.80 in trillion-USD units (expected: 1.8)
        # But "~1.2 Million lives" means 1,200,000 lives (expected: 1200000)
        if not has_currency:
            text_after_number = text_lower[text_lower.find(match.group()) + len(match.group()):]
            text_before_number = text_lower[:text_lower.find(match.group())]
            context = text_before_number + " " + text_after_number

            if 'trillion' in context:
                value *= 1_000_000_000_000
            elif 'billion' in context:
                value *= 1_000_000_000
            elif 'million' in context:
                value *= 1_000_000
            elif 'thousand' in context or ' k' in context:
                value *= 1_000

        return value
    except ValueError:
        return None


def _normalize_submitted(submitted: dict) -> dict:
    """
    Normalize submitted answers for comparison against expected values.

    BUG FIX (2026-01-23): The original implementation only stripped whitespace and '%',
    which caused agents to be marked WRONG for correct answers with different formatting.

    Discovered issues (8 out of 157 numeric answers affected across all runs):

    1. Commas in numbers - trace 58b9c99c, capsule-4977619:
       - Submitted: "6,003.03" -> Expected: 6003.03 -> Marked WRONG (should be CORRECT)
       - Submitted: "1,200,000" -> Expected: 1200000.0 -> Marked WRONG (should be CORRECT)

    2. Dollar signs + extra text - trace 511f05a3, capsule-4977619:
       - Submitted: "$6,003.03 per person" -> Expected: 6003.03 -> Marked WRONG
       - Submitted: "$1.80 Trillion USD" -> Expected: 1.8 -> Marked WRONG

    3. Word multipliers - trace a9093eb1, capsule-4977619:
       - Submitted: "~1.2 Million" -> Expected: 1200000.0 -> Marked WRONG

    4. Prefix text - trace ed7c1d42, capsule-7716865:
       - Submitted: "ID 109" -> Expected: 109.0 -> Marked WRONG

    This bug also prevented scale_mismatch detection from ever triggering, since
    values remained as strings instead of being converted to floats for comparison.
    """
    for key, value in submitted.items():
        if isinstance(value, str):
            # First try simple float conversion (fastest path)
            cleaned = value.strip().replace("%", "")
            try:
                submitted[key] = float(cleaned)
                continue
            except ValueError:
                pass

            # Try extracting numeric value from formatted text
            extracted = _extract_numeric_value(value)
            if extracted is not None:
                submitted[key] = extracted
            # Otherwise keep as string (for actual string answers)

    return submitted


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
    if not isinstance(submitted, (int, float)) or (isinstance(submitted, float) and math.isnan(submitted)):
        return False, QuestionResult(
            question=key,
            question_type="numeric",
            is_vision=is_vision,
            correct=False,
            submitted=submitted,
            prediction_interval=_round_interval(interval) if interval else None,
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
    correct = bool(interval["lower"] <= submitted <= interval["upper"])

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
        prediction_interval=_round_interval(interval),
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

    Handles edge cases:
    - None submitted → treated as empty list
    - String representation of list → parsed with ast.literal_eval
    - Order matters: [1,2] != [2,1]
    - Type matters: [1, 2] != ["1", "2"]

    Args:
        key: Question identifier
        submitted: Agent's submitted answer (may be None, list, string, or other)
        expected: Ground truth list from reference
        is_vision: Whether this question involves figure analysis

    Returns:
        Tuple of (is_correct, QuestionResult)

    BUG FIX (2026-01-22):
    ---------------------
    CRITICAL BUG DISCOVERED: When agents submitted list answers as strings like
    "[0.1, 0.05, 0.01]", the old code did `list(submitted)` which converted the
    STRING into a list of CHARACTERS:

        list("[0.1, 0.05, 0.01]")
        → ['[', '0', '.', '1', ',', ' ', '0', '.', '0', '5', ',', ' ', '0', '.', '0', '1', ']']

    This caused FALSE NEGATIVES where correct answers were marked wrong!

    EVIDENCE (from trace analysis of run 9f352cee, capsule-0921079):
        - Agent submitted: "[0.1, 0.05, 0.01]" (string representation)
        - Got parsed as: ["[", "0", ".", "1", ",", " ", ...] (17-element char array!)
        - Expected was: [0.1, 0.05, 0.01] (3-element float list)
        - Result: INCORRECT (but should have been CORRECT!)

    The fix uses ast.literal_eval() to properly parse string representations of
    Python lists, tuples, dicts, etc. This safely evaluates literal structures
    without executing arbitrary code.
    """
    # Handle None as empty list
    if submitted is None:
        submitted = []

    # Handle string representations of lists (e.g., "[0.1, 0.05, 0.01]")
    # This fixes the critical bug where list(string) creates a char array
    if isinstance(submitted, str):
        submitted_stripped = submitted.strip()
        if submitted_stripped.startswith('[') and submitted_stripped.endswith(']'):
            try:
                # ast.literal_eval safely parses Python literals (lists, dicts, etc.)
                # without executing arbitrary code
                parsed = ast.literal_eval(submitted_stripped)
                if isinstance(parsed, (list, tuple)):
                    submitted = list(parsed)
                    logger.debug(f"Parsed string list '{submitted_stripped}' → {submitted}")
            except (ValueError, SyntaxError) as e:
                # If parsing fails, keep the original string for error reporting
                logger.warning(f"Failed to parse list string '{submitted_stripped}': {e}")
                pass

    # Ensure submitted is a list for comparison
    if not isinstance(submitted, list):
        # Try to convert if it's a tuple or similar (but NOT a string anymore!)
        if isinstance(submitted, tuple):
            submitted = list(submitted)
        else:
            # Can't convert, definitely incorrect
            return False, QuestionResult(
                question=key,
                question_type="list",
                is_vision=is_vision,
                correct=False,
                submitted=submitted,
                expected=expected,
            )

    # Exact element-wise comparison (order matters, types matter)
    correct = bool(submitted == expected)

    return correct, QuestionResult(
        question=key,
        question_type="list",
        is_vision=is_vision,
        correct=correct,
        submitted=submitted,
        expected=expected,
    )


def _get_type(key: str, numeric: list, strings: list, lists: list) -> str:
    if key in numeric:
        return "numeric"
    if key in strings:
        return "string"
    if key in lists:
        return "list"
    return "unknown"


def _round_interval(interval: dict) -> dict:
    return {
        "lower": round(interval["lower"], 6),
        "upper": round(interval["upper"], 6),
    }


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
        missing_questions=[],
        extra_questions=[],
    )


# =============================================================================
# TASK ADHERENCE EVALUATION (LLM-as-Judge)
# judge gets passed tool calls + results
# =============================================================================
TASK_ADHERENCE_PROMPT = """You are an expert evaluator assessing how well an AI agent executed a computational reproducibility benchmark task.

## CORE PRINCIPLE
CoreBench measures COMPUTATIONAL REPRODUCIBILITY - the ability to re-run scientific code and reproduce results.
- Reading pre-existing results is NOT reproduction (even if answer is correct)
- Following the correct process is more important than getting the right answer
- task_adherence measures PROCESS FIDELITY, not outcome correctness

## Task Context
Domain: {domain}
Task prompt: {task_prompt}
{capsule_context}

## Domain-Specific Success Criteria

### MEDIUM MODE (corebench_medium)
**Gold Standard:** Read markdown instructions → Execute documented command (usually Docker) → Handle errors → Generate output

**Automatic Disqualifiers (cap score at 0.3):**
- Never read markdown instructions
- Read pre-existing results instead of executing
- Never attempted documented command

### HARD MODE (corebench_hard)
**Context:** In hard mode, the following are DELETED:
- REPRODUCING.md (step-by-step reproduction instructions)
- code/run.sh and code/run (entry point scripts)
- environment/ nested directory
- results/ directory

**NOT deleted (available to agent):**
- README.md files (project documentation - agent can use these!)
- Dockerfile, requirements.txt, and other dependency files
- All source code files

The agent must figure out how to run the code by:
- Reading README.md for project context and setup instructions
- Reading the target script to understand dependencies
- Checking for Dockerfile, requirements.txt
- Inferring execution method from the code itself

**Gold Standard:** Analyze code/scripts → Identify dependencies → Set up environment → Execute target script → Debug errors → Generate output

**Automatic Disqualifiers (cap score at 0.3):**
- Read pre-existing results instead of executing
- Never attempted to run the target script
- Made no effort to resolve dependency errors
- Gave up without trying alternative approaches

**NOT a disqualifier in hard mode:**
- Not reading README.md IF the capsule doesn't have one (check capsule_context_debug.docs)
- Using code analysis as a discovery method (reading script source to understand dependencies)
- Not finding Dockerfile IF one doesn't exist for this capsule

### EASY MODE (corebench_easy)
**Gold Standard:** List results/ directory → Read output files → Extract exact values → DO NOT execute code

**Disqualifiers:**
- Attempted to run scripts or install dependencies

## Scoring Rubric (0.0 - 1.0)

### Component Weights
- **Core Process (50%)**: Did agent follow the primary reproduction workflow?
- **Problem Solving (25%)**: How well did agent handle obstacles?
- **Discovery (15%)**: How efficiently did agent find needed information?
- **Technical Execution (10%)**: Proper tool usage and commands


## Execution Trace
Steps taken: {steps_used}
Tool calls: {tool_calls_count}
Protocol errors: {protocol_errors}
Command timeouts: {command_timeouts}

{timeout_note}

Tool calls + results:
{tool_interactions}

Final answer: {has_answer}

## Process Metrics (Deterministic)
{methodology_metrics_summary}

## Your Task

1. Assign component scores to the trace: Core _/50, Problem _/25, Discovery _/15, Technical _/10
2. Check for automatic penalties
3. Calculate final score and write reasoning

## Output Format
```json
{{
    "score": <float 0.0-1.0>,
    "followed_instructions": <boolean>,
    "reasoning": "<Decision Tree path + component breakdown + penalties>",
    "component_scores": {{
        "core_process": "<X/50>",
        "problem_solving": "<X/25>",
        "discovery": "<X/15>",
        "technical": "<X/10>"
    }},
    "penalties_applied": ["<list any penalties>"],
    "strengths": ["<specific behaviors>"],
    "weaknesses": ["<specific gaps>"]
}}
```
"""

def _read_text_file_head_bytes(path: str, max_bytes: int) -> tuple[str, bool, int]:
    with open(path, "rb") as f:
        data = f.read(max_bytes + 1)
    truncated = len(data) > max_bytes
    consumed = min(len(data), max_bytes)
    text = data[:max_bytes].decode("utf-8", errors="replace")
    return text, truncated, consumed


_README_FILENAME_RE = re.compile(r"^readme(?:\\.[a-z0-9]+)?$", re.IGNORECASE)
_INCLUDE_DOC_EXTENSIONS = {"", ".md", ".markdown", ".txt", ".rst"}
_EXCLUDE_DOC_EXTENSIONS = {".pdf"}


def _capsule_default_workspace_dir() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "workspace"))


def _discover_capsule_docs(env_dir: str, *, max_depth: int = 4) -> tuple[list[str], list[dict[str, str]]]:
    """
    Find README-like docs inside a staged capsule environment.

    Returns:
        (included_candidates, excluded_docs)
    """
    included: list[str] = []
    excluded: list[dict[str, str]] = []

    if not os.path.isdir(env_dir):
        return [], []

    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

    for root, dirs, files in os.walk(env_dir):
        rel_root = os.path.relpath(root, env_dir)
        depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
        if depth >= max_depth:
            dirs[:] = []
        else:
            dirs[:] = [d for d in dirs if d not in skip_dirs]

        for fname in files:
            fname_lower = fname.lower()
            is_reproducing = fname_lower == "reproducing.md"
            is_readme = bool(_README_FILENAME_RE.match(fname))
            if not (is_reproducing or is_readme):
                continue

            rel_path = fname if rel_root == "." else os.path.normpath(os.path.join(rel_root, fname))
            ext = os.path.splitext(fname)[1].lower()

            if ext in _EXCLUDE_DOC_EXTENSIONS:
                excluded.append({"path": rel_path, "reason": f"excluded_extension:{ext}"})
                continue
            if ext not in _INCLUDE_DOC_EXTENSIONS:
                excluded.append({"path": rel_path, "reason": f"unsupported_extension:{ext}"})
                continue

            included.append(rel_path)

    def _sort_key(path: str) -> tuple[int, int, str]:
        norm = path.replace("\\", "/")
        lower = norm.lower()
        name = os.path.basename(lower)
        depth = 0 if "/" not in lower else lower.count("/")

        if name == "reproducing.md":
            group = 0
        elif "/" not in lower and name.startswith("readme"):
            group = 1
        elif lower.startswith("code/") and name.startswith("readme"):
            group = 2
        elif lower.startswith("environment/") and name.startswith("readme"):
            group = 3
        else:
            group = 4

        return group, depth, lower

    return sorted(set(included), key=_sort_key), excluded


def _build_capsule_context(
    *,
    workspace_dir: Optional[str] = None,
    max_total_bytes: int = 12_000,
    per_file_bytes: int = 6_000,
    max_files: int = 4,
) -> tuple[str, dict[str, Any]]:
    workspace_dir = (workspace_dir or "").strip()
    if not workspace_dir:
        workspace_dir = (os.environ.get("COREBENCH_WORKSPACE_DIR") or "").strip() or _capsule_default_workspace_dir()

    env_dir = os.path.join(workspace_dir, "environment")

    docs, excluded = _discover_capsule_docs(env_dir)
    if not docs and not excluded:
        return "", {"workspace_dir": workspace_dir, "env_dir": env_dir, "docs": [], "excluded": excluded, "included": []}

    remaining = max_total_bytes
    included: list[dict[str, Any]] = []
    skipped: list[str] = []
    chunks: list[str] = []

    for rel_path in docs:
        if len(included) >= max_files or remaining <= 0:
            skipped.append(rel_path)
            continue

        abs_path = os.path.join(env_dir, rel_path)
        if not os.path.isfile(abs_path):
            continue

        read_bytes = min(per_file_bytes, remaining)
        try:
            text, truncated, consumed = _read_text_file_head_bytes(abs_path, read_bytes)
        except OSError:
            continue

        suffix = "\n...[truncated]..." if truncated else ""
        chunks.append(f"### {rel_path}\n```\n{text}{suffix}\n```")
        included.append(
            {
                "path": rel_path,
                "bytes_consumed": consumed,
                "truncated": truncated,
                "included_bytes": read_bytes,
            }
        )
        remaining -= consumed

    overview_lines: list[str] = []
    overview_lines.append("## Capsule Docs (available to agent)")
    if docs:
        overview_lines.append("Docs discovered (README*/REPRODUCING.md):")
        for rel_path in docs[:30]:
            overview_lines.append(f"- {rel_path}")
        if len(docs) > 30:
            overview_lines.append(f"- ... ({len(docs) - 30} more)")
    else:
        overview_lines.append("**No documentation files found in this capsule.**")
        overview_lines.append("Note: In hard mode, REPRODUCING.md is deleted but README.md files are preserved.")
    if excluded:
        overview_lines.append("Excluded docs:")
        for item in excluded[:20]:
            overview_lines.append(f"- {item.get('path')}: {item.get('reason')}")
        if len(excluded) > 20:
            overview_lines.append(f"- ... ({len(excluded) - 20} more)")
    if skipped:
        overview_lines.append("Not excerpted (budget):")
        for rel_path in skipped[:20]:
            overview_lines.append(f"- {rel_path}")
        if len(skipped) > 20:
            overview_lines.append(f"- ... ({len(skipped) - 20} more)")

    overview = "\n".join(overview_lines) + "\n"
    body = "\n\n".join(chunks)
    context = overview if not body else overview + "\n" + body + "\n"

    debug = {
        "workspace_dir": workspace_dir,
        "env_dir": env_dir,
        "docs": docs,
        "excluded": excluded,
        "included": included,
        "skipped": skipped,
        "remaining_bytes": remaining,
    }
    return context, debug


def _build_tool_interactions(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    *,
    max_interactions: int = 24,
    max_result_chars: int = 1200,
    max_args_chars: int = 700,
) -> str:
    if not tool_calls:
        return "(No tool calls recorded)"

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
            exit_code = result.get("exit_code", None)
            timed_out = result.get("timed_out", False)
            hint = result.get("hint")
            summary = str(result.get("summary", "") or "")
            if len(summary) > max_result_chars:
                summary = summary[:max_result_chars] + "\n... (truncated)"

            lines.append(f"Result: exit_code={exit_code}, timed_out={timed_out}")
            if summary.strip():
                lines.append(f"Summary:\n{summary}")
            if hint:
                lines.append(f"Evaluator hint shown to agent:\n{hint}")
        else:
            lines.append("Result: (missing tool_result event)")

        return "\n".join(lines)

    if len(pairs) <= max_interactions:
        formatted = [_format_pair(call, result) for call, result in pairs]
        return "\n\n".join(formatted)

    head = 8
    tail = max_interactions - head
    formatted_head = [_format_pair(call, result) for call, result in pairs[:head]]
    formatted_tail = [_format_pair(call, result) for call, result in pairs[-tail:]]
    skipped = max(0, len(pairs) - max_interactions)
    return "\n\n".join(formatted_head + [f"... ({skipped} tool interactions omitted) ..."] + formatted_tail)


async def evaluate_task_adherence(
    domain: str,
    task_prompt: str,
    steps_used: int,
    tool_calls_count: int,
    protocol_errors: int,
    submitted: dict[str, Any],
    accuracy_result: AccuracyMetrics,
    action_trace: list[dict],
    tool_calls: Optional[list[dict[str, Any]]] = None,
    tool_results: Optional[list[dict[str, Any]]] = None,
    workspace_dir: Optional[str] = None,
    trace_event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    judge_model: str = "nebius/openai/gpt-oss-120b",  # match DEFAULT_MODEL
    command_timeouts: int = 0,
    methodology_metrics: Optional[MethodologyMetrics] = None,
) -> TaskAdherenceMetrics:
    """Use LLM-as-judge to evaluate task execution quality.

    NOTE: Accuracy is intentionally NOT passed to the judge to avoid anchoring bias.
    The judge should evaluate process fidelity independently of outcome correctness.
    """
    action_summary = _build_action_summary(action_trace)

    # Build process metrics summary for the judge (replaces answer_summary to avoid bias)
    if methodology_metrics:
        methodology_metrics_summary = f"""Documentation Read: {'Yes' if methodology_metrics.read_documentation else 'No'} ({', '.join(methodology_metrics.docs_read[:3]) if methodology_metrics.docs_read else 'none'})
Target Script Read: {'Yes' if methodology_metrics.read_target_script else 'No'}
Execution Attempted: {'Yes' if methodology_metrics.attempted_execution else 'No'} ({methodology_metrics.execution_attempts} attempts)
Execution Succeeded: {'Yes' if methodology_metrics.successful_execution else 'No'}
Dependencies Installed: {'Yes' if methodology_metrics.installed_dependencies else 'No'}
Error Recovery Rate: {methodology_metrics.error_recovery.recovery_rate:.1%} ({methodology_metrics.error_recovery.errors_recovered}/{methodology_metrics.error_recovery.total_errors} errors recovered)
Violations: {', '.join(methodology_metrics.violations) if methodology_metrics.violations else 'None'}
Deterministic Methodology Score: {methodology_metrics.methodology_score:.2f}"""
    else:
        methodology_metrics_summary = "(Process metrics not available - evaluate from trace)"

    # Build timeout context note for fair judging
    if command_timeouts > 0:
        timeout_note = (
            f"NOTE: {command_timeouts} command(s) hit the timeout limit. This is an infrastructure "
            "constraint (e.g., slow Docker emulation on ARM64), not necessarily an agent error. "
            "Consider this when evaluating - the agent may have used the correct approach but was "
            "blocked by execution time limits."
        )
    else:
        timeout_note = ""

    capsule_context, capsule_debug = _build_capsule_context(workspace_dir=workspace_dir)
    tool_interactions = _build_tool_interactions(tool_calls or [], tool_results or [])
    prompt = TASK_ADHERENCE_PROMPT.format(
        domain=domain,
        task_prompt=task_prompt,
        capsule_context=capsule_context,
        steps_used=steps_used,
        tool_calls_count=tool_calls_count,
        protocol_errors=protocol_errors,
        command_timeouts=command_timeouts,
        timeout_note=timeout_note,
        tool_interactions=tool_interactions,
        has_answer="Yes" if submitted else "No",
        methodology_metrics_summary=methodology_metrics_summary,
    )
    
    # Same API configuration logic as corebench_agent.py
    api_base = (os.environ.get("COREBENCH_TEXT_API_BASE") or "").strip()
    api_key = (os.environ.get("COREBENCH_TEXT_API_KEY") or "").strip()
    
    if api_base:
        # Self-hosted mode: prepend openai/ for custom endpoint
        model_name = judge_model
        if not model_name.startswith("openai/"):
            model_name = f"openai/{model_name}"
        completion_kwargs = {
            "model": model_name,
            "api_base": api_base,
            "api_key": api_key or "dummy",
        }
    else:
        # Cloud mode: pass model to litellm as-is (provider prefix handles routing)
        model_name = judge_model
        completion_kwargs = {
            "model": model_name,
        }
    logger.debug(f"Task adherence judge using model={model_name}")
    
    if trace_event_callback is not None:
        try:
            trace_event_callback(
                {
                    "type": "llm_judge_input",
                    "judge": "task_adherence",
                    "flow": "green -> judge",
                    "domain": domain,
                    "model": model_name,
                    "task_prompt": task_prompt,
                    "steps_used": steps_used,
                    "tool_calls_count": tool_calls_count,
                    "protocol_errors": protocol_errors,
                    "command_timeouts": command_timeouts,
                    "timeout_note": timeout_note,
                    "has_answer": bool(submitted),
                    "methodology_metrics_summary": methodology_metrics_summary,
                    "action_summary": action_summary,
                    "capsule_context_debug": capsule_debug,
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
                        "judge": "task_adherence",
                        "flow": "judge -> green",
                        "domain": domain,
                        "model": model_name,
                        "raw": raw_content,
                        "parsed": result,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to write task adherence judge output trace: {e}")
        
        return TaskAdherenceMetrics(
            score=float(result.get("score", 0.0)),
            followed_instructions=bool(result.get("followed_instructions", False)),
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
                        "judge": "task_adherence",
                        "flow": "judge -> green",
                        "domain": domain,
                        "model": model_name,
                        "error": str(e),
                    }
                )
            except Exception as trace_e:
                logger.warning(f"Failed to write task adherence judge error trace: {trace_e}")
        return TaskAdherenceMetrics(
            score=0.0,
            followed_instructions=False,
            reasoning=f"Evaluation failed: {e}",
            strengths=[],
            weaknesses=[],
            status="error",
            error_message=str(e),
        )


def _build_action_summary(action_trace: list[dict], max_actions: int = 30) -> str:
    """
    Build a summary of purple agent tool calls and the MCP server results. 
    """

    # Helper to format a single event string
    def _format_event(event: dict) -> str:
        if event.get("type") != "tool_call":
            return ""

        turn = event.get("turn", "?")
        name = event.get("tool", "unknown")  # tool_call events use "tool" field
        args = event.get("arguments", {})
        
        if name == "execute_bash":
            cmd = args.get("command", "")[:100]  # Truncate long commands
            return f"[{turn}] bash: {cmd}"
        elif name == "inspect_file_as_text":
            path = args.get("file_path", "")
            return f"[{turn}] read: {path}"
        elif name == "FINAL_ANSWER":
            return f"[{turn}] FINAL_ANSWER submitted"
        else:
            return f"[{turn}] {name}"

    lines = []
    
    # Short trace, show everything
    if len(action_trace) <= max_actions:
        for event in action_trace:
            line = _format_event(event)
            if line: lines.append(line)
            
    # Long trace, show Head + Tail
    else:
        head_count = 10
        for event in action_trace[:head_count]:
            line = _format_event(event)
            if line: lines.append(line)
            
        # SKIPPED SECTION
        skipped_count = len(action_trace) - max_actions
        lines.append(f"\n... [{skipped_count} intermediate steps skipped] ...\n")
        
        tail_count = max_actions - head_count
        for event in action_trace[-tail_count:]:
            line = _format_event(event)
            if line: lines.append(line)

    return "\n".join(lines) if lines else "(No actions recorded)"


# =============================================================================
# EFFICIENCY METRICS
# =============================================================================

def compute_efficiency(
    steps_used: int,
    max_steps: int,
    tool_calls_count: int,
    time_seconds: float,
    protocol_errors: int,
    command_timeouts: int = 0,
) -> EfficiencyMetrics:
    """
    Compute efficiency metrics for the task execution.
    
    Args:
        steps_used: Number of interaction steps taken
        max_steps: Maximum allowed steps
        tool_calls_count: Number of tool calls made (excludes FINAL_ANSWER)
        time_seconds: Total time taken in seconds
        protocol_errors: Number of protocol/format errors
    """
    return EfficiencyMetrics(
        steps_used=steps_used,
        max_steps=max_steps,
        tool_calls=tool_calls_count,
        time_seconds=time_seconds,
        protocol_errors=protocol_errors,
        command_timeouts=command_timeouts,
    )


# =============================================================================
# PROCESS METRICS (Deterministic Extraction from Traces)
# =============================================================================

def _is_documentation(path: str) -> bool:
    """Check if a file path is documentation (README, REPRODUCING.md, etc.)."""
    if not path:
        return False
    path_lower = path.lower()
    basename = os.path.basename(path_lower)

    # Direct matches for documentation files
    doc_patterns = [
        "readme", "readme.md", "readme.txt", "readme.rst",
        "reproducing.md", "readme.pdf",
    ]

    if basename in doc_patterns or basename.startswith("readme"):
        return True

    # Note: Correctly detects READMEs in any subdir (e.g., code/README.md) since we only check basename
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

    # EDGE CASE (2026-01-23): Certain R patterns via `Rscript -e "..."` are ACTUAL execution,
    # not debugging. We must check for these BEFORE the general `rscript -e` exclusion below.
    #
    # Patterns that indicate real execution:
    # - rmarkdown::render, knitr::knit, bookdown::render - rendering .Rmd/.Rnw files
    #   Discovered in capsule-7186268 where agent rendered an .Rmd file
    # - source( - running an R script via source('script.R')
    #   Discovered in capsule-5136217 where agent ran `Rscript -e "source('script.R', echo=TRUE)"`
    #   but attempted_execution was incorrectly set to False
    r_execution_patterns = ["rmarkdown::render", "knitr::knit", "bookdown::render", "source("]
    if any(pattern in cmd_lower for pattern in r_execution_patterns):
        return True

    # Exclude one-liner debugging commands and pip via python -m
    # BUG FIX (2026-01-22): Added "rscript -e" to exclusions. In capsule-4933686,
    # agent used debug commands like `Rscript -e "load(...); str(clinical)"` which
    # incorrectly set successful_execution=True even though Main.R was never run.
    # BUG FIX (2026-01-23): Added "--version" to exclusions. Version checks are not
    # actual execution attempts.
    exclusion_patterns = [
        "python -c ",
        "python3 -c ",
        "python -m pip",
        "python3 -m pip",
        "/python -m pip",  # Full path python
        "/python3 -m pip",
        "rscript -e ",     # Rscript one-liners for debugging
        "rscript -e\"",    # Without space after -e
        " --version",      # Version checks (python --version, Rscript --version, etc.)
    ]
    if any(pattern in cmd_lower for pattern in exclusion_patterns):
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
    # BUG FIX (2026-01-23): Previous patterns "bash run", "sh run" only matched when
    # bash/sh was immediately followed by "run". This missed commands like:
    #   "bash code/run_all.sh" (path between bash and script name)
    # Discovered in capsule-8412128 (trace 154d5dd9) where agent ran "bash code/run_all.sh"
    # but attempted_execution was incorrectly set to False.
    #
    # New logic: Check if bash/sh is used with a .sh file anywhere in the command
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


def _classify_error(summary: str) -> str:
    """Classify an error type from tool result summary.

    Categories based on actual errors observed in CoreBench traces.

    FOUND "OTHER" BUG: function isn't catching the actual error patterns in these traces. Check what the summaries actually contain
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

    return "other"


def _compute_error_recovery(
    tool_calls: list[dict],
    tool_results: list[dict],
) -> ErrorRecoveryMetrics:
    """Analyze error recovery patterns from tool interactions.

    An error is "recovered" if the agent continues with productive actions after it.
    """
    # Build a timeline of success/failure
    results_by_turn: dict[int, dict] = {}
    for result in tool_results:
        turn = result.get("turn", 0)
        results_by_turn[turn] = result

    errors = []
    error_type_counts: dict[str, int] = {}  # Collect error types for diagnostics

    for call in tool_calls:
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

        errors.append({
            "turn": turn,
            "is_error": is_error,
            "error_type": error_type,
        })

    total_errors = sum(1 for e in errors if e["is_error"])

    if total_errors == 0:
        return ErrorRecoveryMetrics(
            total_errors=0,
            errors_recovered=0,
            recovery_rate=1.0,  # Perfect - no errors to recover from
            consecutive_failures=0,
            persistence_score=1.0,
            error_types={},
        )

    # Count recoveries: error followed by successful action
    errors_recovered = 0
    consecutive_failures = 0
    max_consecutive = 0
    current_streak = 0

    for i, e in enumerate(errors):
        if e["is_error"]:
            current_streak += 1
            max_consecutive = max(max_consecutive, current_streak)

            # Check if recovered (next action succeeds or agent continues)
            if i + 1 < len(errors) and not errors[i + 1]["is_error"]:
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


def _parse_expected_scripts(task_prompt: str) -> list[str]:
    """Extract expected script filenames from task_prompt.

    Parses script filenames mentioned in the task prompt that the agent should execute.
    This enables computing execution_coverage: what fraction of expected scripts were run.

    Supported patterns (covers ~94% of CORE-Bench tasks):
        - Python scripts: .py
        - Jupyter notebooks: .ipynb
        - Shell scripts: .sh
        - R scripts: .R
        - R Markdown: .Rmd
        - Code Ocean entry point: "run the run file"

    NOT SUPPORTED (returns empty list):
        - Python modules without extension: "Run 'physalia_automators.reports' as a python module"
        - Vague "all files" patterns: "Run all the .R scripts in the folder"
    """
    if not task_prompt:
        return []

    scripts = []

    # these prompts are questionable in hard mode since it deletes run files
    # Code Ocean convention "Run the run file" -> entry point is code/run or code/run.sh
    if "run the run file" in task_prompt.lower():
        scripts.append("run")

    # Python scripts (.py)
    # Example: "Run step_0_vit_encode.py, then step_1_train.py"
    py_matches = re.findall(r"['\"]?(\S+\.py)['\"]?", task_prompt)
    scripts.extend(py_matches)

    # Jupyter notebooks (.ipynb)
    # Example: "Run the jupyter notebook visualize_results.ipynb"
    ipynb_matches = re.findall(r"['\"]?(\S+\.ipynb)['\"]?", task_prompt)
    scripts.extend(ipynb_matches)

    # Shell scripts (.sh)
    # Example: "Run the bash script 'demo.sh'"
    sh_matches = re.findall(r"['\"]?(\S+\.sh)['\"]?", task_prompt)
    scripts.extend(sh_matches)

    # R scripts (.R) - use word boundary to avoid matching .Rmd
    # Example: "Run 'pancancer_calculation.R' using Rscript"
    r_matches = re.findall(r"['\"]?(\S+\.R)['\"]?(?!\w)", task_prompt)
    scripts.extend(r_matches)

    # R Markdown (.Rmd)
    # Example: "Run 'manuscript.Rmd' using Rscript and render it as a pdf"
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
) -> tuple[list[str], int, str]:
    """Extract successfully executed script filenames and stdout from tool calls.

    Only counts executions that succeeded (exit_code=0).
    Also captures stdout from successful script executions for diagnostics
    (helps identify "ran code but output to stdout" cases).

    Uses turn-based matching (not array indices) for robustness.

    Args:
        tool_calls: List of tool_call events from trace
        tool_results: List of tool_result events from trace

    Returns:
        Tuple of (executed_scripts, stdout_total_bytes, stdout_sample)
    """
    if not tool_calls:
        return [], 0, ""

    # Build results lookup by turn (robust matching)
    results_by_turn: dict[int, dict] = {}
    for result in tool_results:
        turn = result.get("turn", 0)
        results_by_turn[turn] = result

    executed = []
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
        if exit_code is None or exit_code != 0:
            continue  # Only count successful executions

        # Extract script names from the command
        scripts_in_cmd = []

        # Python scripts (.py)
        # Example: "python script.py", "python3 -u main.py"
        py_exec = re.findall(r"python[0-9.]*\s+(?:-[^\s]+\s+)*(\S+\.py)", cmd)
        scripts_in_cmd.extend(py_exec)

        # Shell scripts (.sh)
        # Example: "bash demo.sh", "./run.sh"
        sh_exec = re.findall(r"(?:bash\s+|\./)(\S+\.sh)", cmd)
        scripts_in_cmd.extend(sh_exec)

        # Jupyter notebooks (.ipynb)
        # Example: "jupyter nbconvert --to html --execute notebook.ipynb"
        ipynb_exec = re.findall(r"(?:jupyter|nbconvert)\s+.*?(\S+\.ipynb)", cmd)
        scripts_in_cmd.extend(ipynb_exec)

        # Code Ocean entry point: ./run or bash run (no extension)
        run_exec = re.findall(r"(?:bash\s+|\./)(run)(?:\s|$)", cmd)
        scripts_in_cmd.extend(run_exec)

        # R scripts (.R) via Rscript
        # Example: "Rscript analysis.R"
        r_exec = re.findall(r"Rscript\s+(?!-e)(?:-[^\s]+\s+)*['\"]?(\S+\.R)\b['\"]?", cmd)
        scripts_in_cmd.extend(r_exec)

        # R Markdown (.Rmd) via rmarkdown::render()
        # Example: "Rscript -e \"rmarkdown::render('report.Rmd')\""
        rmd_exec = re.findall(r"render\(['\"]([^'\"]+\.Rmd)['\"]", cmd)
        scripts_in_cmd.extend(rmd_exec)

        # If we found scripts in this command, capture stdout
        if scripts_in_cmd:
            executed.extend(scripts_in_cmd)
            stdout_text = str(result.get("summary", ""))
            if stdout_text.strip():
                all_stdout_parts.append(stdout_text)
                total_stdout_bytes += len(stdout_text)

    # Deduplicate and get basenames
    seen = set()
    unique = []
    for script in executed:
        basename = os.path.basename(script)
        if basename not in seen:
            seen.add(basename)
            unique.append(basename)

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

    return unique, total_stdout_bytes, stdout_sample


def _compute_execution_coverage(expected_scripts: list[str], executed_scripts: list[str], successful_execution: bool) -> float:
    """Compute what fraction of expected scripts were executed.

    Handles special cases:
    - "run" matches "run.sh" (Code Ocean convention) (we only care about hardmode)
    - Fuzzy matching for Jupyter notebooks with modified output names

    Args:
        expected_scripts: Scripts parsed from task_prompt
        executed_scripts: Scripts successfully executed
        successful_execution: Whether execution completed successfully (exit code 0)

    Returns:
        Coverage from 0.0 to 1.0
    """
    # No expected scripts parsed from task_prompt
    # EDGE CASE (2026-01-23): In capsule-4977619, the agent ran verify_integrity.py and
    # run_sage_simulation.py instead of the main entry point run_all.sh. The task_prompt
    # only contained questions (no script names), so expected_scripts=[]. We can't verify
    # they ran the RIGHT scripts, so we trust successful_execution as the signal:
    # - If execution succeeded, give benefit of the doubt (1.0)
    # - If execution failed, give no credit (0.0)
    if not expected_scripts:
        return 1.0 if successful_execution else 0.0

    expected_basenames = {os.path.basename(s) for s in expected_scripts}
    executed_basenames = {os.path.basename(s) for s in executed_scripts}
    matched = expected_basenames & executed_basenames

    # Special case: "run" matches "run.sh"
    if "run" in expected_basenames and "run" not in matched:
        if "run.sh" in executed_basenames:
            matched.add("run")

    # Special case: Jupyter notebooks with modified output names
    # e.g., expected "Notebook.ipynb" matches executed "exec_Notebook.ipynb"
    for expected in expected_basenames - matched:
        if not expected.endswith(".ipynb"):
            continue
        expected_stem = expected[:-6]  # Remove ".ipynb"
        for executed in executed_basenames:
            if expected_stem in executed and executed.endswith(".ipynb"):
                matched.add(expected)
                break

    return len(matched) / len(expected_basenames)


def extract_methodology_metrics(tool_calls: list[dict], tool_results: list[dict],
    domain: str, task_prompt: str = "", deleted_files: list[str] | None = None,
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
                docs_read.append(file_path)
                read_documentation = True

            # Check for reading target script (code files)
            # Includes Python, Shell, R, R Markdown, Julia, MATLAB, Jupyter notebooks
            if file_path.endswith((".py", ".sh", ".r", ".R", ".Rmd", ".jl", ".m", ".ipynb")):
                read_target_script = True

            # Track results/ reads for later violation analysis - only applicable for easy mode
            # if domain == "corebench_hard":
            #     if "results/" in file_path or "/results" in file_path:
            #         results_reads_before_execution.append((turn, file_path))

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

    # Now determine violations: only flag results reads that happened BEFORE successful execution
    if domain == "corebench_hard":
        for read_turn, file_path in results_reads_before_execution:
            # If no successful execution, any results read is suspicious
            if first_successful_execution_turn is None or read_turn < first_successful_execution_turn:
                read_preexisting_results = True
                violations.append(f"Read pre-existing results: {file_path}")

    # Compute error recovery metrics
    error_recovery = _compute_error_recovery(tool_calls, tool_results)

    # Parse expected scripts from task_prompt and extract executed scripts
    expected_scripts = _parse_expected_scripts(task_prompt)

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

    executed_scripts, stdout_total_bytes, stdout_sample = _extract_executed_scripts(
        tool_calls, tool_results
    )
    execution_coverage = _compute_execution_coverage(expected_scripts, executed_scripts, successful_execution)
    stdout_captured = stdout_total_bytes > 0

    # Compute methodology score based on domain
    methodology_score = _compute_methodology_score(domain=domain,
        read_documentation=read_documentation,
        read_target_script=read_target_script,
        attempted_execution=attempted_execution,
        successful_execution=successful_execution,
        installed_dependencies=installed_dependencies,
        error_recovery=error_recovery,
        violations=violations,
        execution_coverage=execution_coverage,
    )

    return MethodologyMetrics(
        read_documentation=read_documentation,
        docs_read=docs_read,
        read_target_script=read_target_script,
        attempted_execution=attempted_execution,
        execution_attempts=execution_attempts,
        successful_execution=successful_execution,
        expected_scripts=expected_scripts,
        executed_scripts=executed_scripts,
        execution_coverage=execution_coverage,
        stdout_captured=stdout_captured,
        stdout_total_bytes=stdout_total_bytes,
        stdout_sample=stdout_sample,
        installed_dependencies=installed_dependencies,
        error_recovery=error_recovery,
        read_preexisting_results=read_preexisting_results,
        violations=violations,
        methodology_score=methodology_score,
    )


def _compute_methodology_score(
    domain: str,
    read_documentation: bool,
    read_target_script: bool,
    attempted_execution: bool,
    successful_execution: bool,
    installed_dependencies: bool,
    error_recovery: ErrorRecoveryMetrics,
    violations: list[str],
    execution_coverage: float = 0.0,
) -> float:
    """Compute a deterministic methodology score from 0-1 based on observed behaviors.

    The execution_coverage parameter enables partial credit for running some
    but not all expected scripts (parsed from task_prompt). This addresses
    the "stdout problem" where agents may run scripts correctly but not
    produce expected output files and just write results to stdout instead.
    """
    score = 0.0

    if domain == "corebench_easy":
        # Full credit for reading files
        if not attempted_execution:
            score = 1.0
        # Penalty for execution attempts
        else:
            score = 0.3

    elif domain == "corebench_medium":
        # - Read docs (25%)
        # - Execution coverage (35%) - partial credit based on expected vs executed scripts
        # - Successful execution (25%)
        # - Error recovery (15%)
        if read_documentation:
            score += 0.25
        # Use execution_coverage for partial credit
        if execution_coverage > 0:
            score += 0.35 * execution_coverage
        elif attempted_execution:
            # Partial credit if attempted but coverage couldn't be computed
            score += 0.35 * 0.5
        if successful_execution:
            score += 0.25
        score += error_recovery.persistence_score * 0.15

    elif domain == "corebench_hard":
        # - Read documentation/README (15%)
        # - Read code/scripts (15%)
        # - Execution coverage (30%) - partial credit based on expected vs executed scripts
        # - Successful execution (30%)
        # - Error recovery (10%)
        # - Penalty: no dep install on failure (-5%)
        if read_documentation:
            score += 0.15
        if read_target_script:
            score += 0.15
        # Use execution_coverage for partial credit, i.e. running some scripts
        if execution_coverage > 0:
            score += 0.30 * execution_coverage
        elif attempted_execution:
            # Partial credit if attempted but coverage couldn't be computed - i.e. "run file question"
            score += 0.30 * 0.5
        if successful_execution:
            score += 0.30
        score += error_recovery.persistence_score * 0.10

        # Penalty for not attempting dependency installation when execution failed
        # Installing deps is a means, not a goal - only penalize if execution was
        # attempted, it failed, AND agent didn't try to install deps to fix it
        if not successful_execution and not installed_dependencies and attempted_execution:
            score -= 0.05

        # Penalty for reading pre-existing results (anti-pattern)
        if violations:
            score = min(score, 0.3)  # Cap at 0.3 if violations

        # show log of calculated components to get to final score

    return min(1.0, max(0.0, score))


# =============================================================================
# AGGREGATE METRICS
# =============================================================================

@dataclass
class AggregateMetrics:
    """Aggregate metrics across all tasks in a benchmark run.

    Provides summary statistics for comparing agent performance across
    different models, configurations, or benchmark versions.
    """
    num_tasks: int
    num_successful: int
    pass_rate: float

    # Accuracy metrics
    mean_accuracy: float
    mean_written_accuracy: float
    mean_vision_accuracy: float

    mean_adherence: float

    # Methodology metrics (deterministic)
    mean_methodology_score: float
    doc_read_rate: float
    execution_attempt_rate: float
    successful_execution_rate: float
    mean_error_recovery_rate: float

    mean_steps: float
    mean_tool_calls: float
    mean_time: float

    task_results: dict[str, dict]


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
            mean_steps=0.0,
            mean_tool_calls=0.0,
            mean_time=0.0,
            task_results={},
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

    # Efficiency
    mean_steps = np.mean([e.efficiency.steps_used for e in evaluations])
    mean_tools = np.mean([e.efficiency.tool_calls for e in evaluations])
    mean_time = np.mean([e.efficiency.time_seconds for e in evaluations])

    # Per-task summary
    task_results = {
        e.task_id: {
            "success": bool(e.success),
            "accuracy": float(e.accuracy.accuracy),
            "adherence": float(e.task_adherence.score),
            "methodology_score": float(e.methodology_metrics.methodology_score) if e.methodology_metrics else None,
        }
        for e in evaluations
    }

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
        mean_steps=float(mean_steps),
        mean_tool_calls=float(mean_tools),
        mean_time=float(mean_time),
        task_results=task_results,
    )
