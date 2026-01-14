"""
CoreBench Evaluation Metrics Module
====================================

This module provides clean, well-defined metrics for evaluating the purple agent's performance
on the CoreBench reproducibility benchmark. The benchmark (agentified for AgentBeats) tests
whether agents can reproduce both the environment and results of published research papers.

Metrics Overview
----------------

1. **ACCURACY** - Are the submitted answers correct compared to ground truth?
   - Numeric values: Uses 95% prediction intervals to handle run-to-run variance
     in stochastic experiments (e.g., ML training with different random seeds).
     The interval is computed using t-distribution: mean ± t(0.975, n-1) * std * sqrt(1 + 1/n)
   - String values: Case-insensitive exact match after stripping whitespace
   - List values: Element-wise exact comparison (order matters, types matter)
   
2. **REPRODUCIBILITY** - Did the agent successfully restore removed files/folders?
   - Only applicable for medium/hard difficulty levels where files are removed
   - Quality checks: Files must have content (>= 10 bytes), directories must not be empty
   - Tracks executable permissions for scripts (informational)

3. **FAITHFULNESS** (LLM-as-Judge) - Are answers grounded in tool execution evidence?
   - Evaluates whether submitted answers can be traced back to actual tool outputs
   - Detects guessing, hallucination, or simulation behaviors
   - Returns score 0.0-1.0 plus boolean flags for grounding and suspected guessing

4. **TASK ADHERENCE** (LLM-as-Judge) - Did the agent properly follow task instructions?
   - Evaluates navigation strategy, rule compliance, and problem-solving approach
   - Assesses how well the agent understood and executed the task
   - Returns qualitative assessment (excellent/good/fair/poor) plus strengths/weaknesses

5. **EFFICIENCY** - How resource-efficient was the agent's approach?
   - Steps used vs maximum allowed
   - Tool call count and execution time
   - Protocol/format errors encountered

Usage
-----
The main entry points are:
- evaluate_accuracy(ground_truth, submitted) -> AccuracyMetrics
- evaluate_reproducibility(workspace_dir, removed_paths) -> ReproducibilityMetrics  
- evaluate_faithfulness(...) -> FaithfulnessMetrics (async, uses LLM)
- evaluate_task_adherence(...) -> TaskAdherenceMetrics (async, uses LLM)
- compute_efficiency(...) -> EfficiencyMetrics
- aggregate_results(evaluations) -> AggregateMetrics

Ground Truth Format
-------------------
Ground truth is expected as a list of dicts from multiple experiment runs:
[
    {"question1": 0.95, "question2": "label", ...},  # Run 1
    {"question1": 0.93, "question2": "label", ...},  # Run 2
    ...
]

For single-run experiments, provide a list with one dict.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import json
import logging
import math
import os

import re

import numpy as np
from scipy.stats import t
import litellm

logger = logging.getLogger("evaluator.metrics")

# =============================================================================
# VISION QUESTION DETECTION
# =============================================================================

# Regex pattern for detecting vision-related questions based on CORE-Bench paper:
# "extracting results from attributes of figures, graphs, plots, or PDF tables"
# Applied to question keys.
_VISION_KEY_PATTERN = re.compile(
    r"fig(ure)?s?"   # fig, figure, figures
    r"|plots?"       # plot, plots  
    r"|graphs?"      # graph, graphs
    r"|images?"      # image, images
    r"|charts?"      # chart, charts
    r"|heatmaps?"    # heatmap, heatmaps
    r"|screenshots?" # screenshot, screenshots
    r"|visuals?"     # visual, visuals
    ,
    re.IGNORECASE
)


def _is_vision_question(key: str) -> bool:
    """
    Check if a question key indicates vision/figure-related content.
    
    Based on CORE-Bench paper definition: questions requiring extraction of
    results from figures, graphs, plots, charts, or images.
    
    Args:
        key: Question identifier (e.g., "accuracy_fig1", "plot_error_rate")
        
    Returns:
        True if the key suggests a vision-based question
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
class ReproducibilityMetrics:
    """Metrics for file/environment restoration."""
    targets: list[str]  # Files/folders that were removed by difficulty filter
    restored: list[str]  # Successfully restored paths
    missing: list[str]   # Paths that were not restored
    restoration_rate: float
    # weighted_score: float  # MAYBE IMPLEMENT LATER weight by importance
    details: list[dict[str, Any]]  # Per-path metadata
    
    @property
    def targets_count(self) -> int:
        return len(self.targets)
    
    @property
    def restored_count(self) -> int:
        return len(self.restored)


@dataclass
class FaithfulnessMetrics:
    """LLM-as-Judge metrics for answer grounding."""
    score: float  # 0.0-1.0
    is_grounded: bool
    suspected_guessing: bool
    reasoning: str
    evidence_summary: str
    flagged_answers: list[str]  # Questions with suspicious answers
    status: str = "success"  # "success" or "error"
    error_message: Optional[str] = None


@dataclass
class TaskAdherenceMetrics:
    """LLM-as-Judge metrics for task execution quality.
    
    Evaluates how well the agent followed task instructions, navigated the codebase,
    and solved problems. Uses an LLM judge to assess execution quality.
    
    Attributes:
        score: Overall quality score from 0.0 (poor) to 1.0 (excellent)
        followed_instructions: Whether the agent followed task prompt and documentation
        navigation_quality: Qualitative assessment of codebase exploration strategy
        reasoning: LLM judge's explanation of the assessment
        strengths: List of things the agent did well
        weaknesses: List of areas for improvement
        status: "success" if evaluation completed, "error" if API call failed
        error_message: Error details if status is "error", None otherwise
    """
    score: float  # 0.0-1.0
    followed_instructions: bool
    navigation_quality: str  # "excellent", "good", "fair", "poor"
    reasoning: str
    strengths: list[str]
    weaknesses: list[str]
    status: str = "success"  # "success" or "error"
    error_message: Optional[str] = None


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
    reproducibility: Optional[ReproducibilityMetrics]
    faithfulness: FaithfulnessMetrics
    task_adherence: TaskAdherenceMetrics
    efficiency: EfficiencyMetrics

    # Raw data for debugging
    submitted_answer: Any
    ground_truth: list[dict]

    # Cost tracking (None if not available)
    task_cost: Optional[float] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Uses _make_json_safe to handle numpy types and other non-serializable objects.
        """
        return _make_json_safe({
            "task_id": self.task_id,
            "domain": self.domain,
            "success": self.success,
            "accuracy": asdict(self.accuracy),
            "reproducibility": asdict(self.reproducibility) if self.reproducibility else None,
            "faithfulness": asdict(self.faithfulness),
            "task_adherence": asdict(self.task_adherence),
            "efficiency": asdict(self.efficiency),
            "task_cost": self.task_cost,
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
        
    Returns:
        AccuracyMetrics with detailed breakdown
    """
    if not ground_truth or not ground_truth[0]:
            logger.error("Empty or invalid ground truth provided")
            return _empty_accuracy_metrics()
    
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


def _normalize_submitted(submitted: dict) -> dict:
    """Normalize submitted answers (strip %, convert numeric strings)."""
    for key, value in submitted.items():
        if isinstance(value, str):
            cleaned = value.strip().replace("%", "")
            try:
                submitted[key] = float(cleaned)
            except ValueError:
                pass  # Keep as string
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
    
    # If incorrect, check for scale mismatch (agent may have converted decimal<->percentage)
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
    - Order matters: [1,2] != [2,1]
    - Type matters: [1, 2] != ["1", "2"]
    
    Args:
        key: Question identifier
        submitted: Agent's submitted answer (may be None, list, or other)
        expected: Ground truth list from reference
        is_vision: Whether this question involves figure analysis
        
    Returns:
        Tuple of (is_correct, QuestionResult)
    """
    # Handle None as empty list
    if submitted is None:
        submitted = []
    
    # Ensure submitted is a list for comparison
    if not isinstance(submitted, list):
        # Try to convert if it's a tuple or similar
        try:
            submitted = list(submitted)
        except (TypeError, ValueError):
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
# REPRODUCIBILITY EVALUATION
# =============================================================================

# Minimum file size (bytes) for a file to count as "restored"
# Empty or near-empty files don't count as proper restoration
MIN_FILE_SIZE_BYTES = 10

# File extensions that should have executable permission
EXECUTABLE_EXTENSIONS = {".sh", ".py", ".pl", ".rb", ".bash"}


def evaluate_reproducibility(
    workspace_dir: str,
    removed_paths: list[str],
) -> ReproducibilityMetrics:
    """
    Evaluate how well the agent restored files removed by difficulty filters.
    
    For medium/hard difficulties, certain files (results/, REPRODUCING.md, run scripts)
    are removed before the task starts. This metric measures what percentage the
    agent successfully recreated with meaningful content.
    
    Restoration Criteria:
    - Files must exist AND have size >= MIN_FILE_SIZE_BYTES (10 bytes)
    - Directories must exist AND contain at least one entry
    - Scripts (.sh, .py, etc.) should ideally be executable (logged but not required)
    
    Args:
        workspace_dir: Path to the workspace directory
        removed_paths: List of relative paths that were removed
        
    Returns:
        ReproducibilityMetrics with restoration details including quality checks
    """
    from pathlib import Path
    import stat
    
    targets = sorted(set(removed_paths))
    workspace = Path(workspace_dir)
    
    restored = []
    missing = []
    details = []
    
    for rel_path in targets:
        abs_path = workspace / rel_path
        exists = abs_path.exists()
        
        meta = {
            "path": rel_path,
            "exists": exists,
            "kind": "missing",
            "valid": False,  # True only if restoration meets quality criteria
            "issues": [],    # List of quality issues found
        }
        
        if exists:
            if abs_path.is_file():
                meta["kind"] = "file"
                file_stat = abs_path.stat()
                meta["size_bytes"] = file_stat.st_size
                
                # Check file size - empty files don't count
                if file_stat.st_size >= MIN_FILE_SIZE_BYTES:
                    meta["valid"] = True
                else:
                    meta["issues"].append(f"file too small ({file_stat.st_size} bytes < {MIN_FILE_SIZE_BYTES})")
                
                # Check executable permission for scripts (informational)
                suffix = abs_path.suffix.lower()
                if suffix in EXECUTABLE_EXTENSIONS:
                    is_executable = bool(file_stat.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
                    meta["is_executable"] = is_executable
                    if not is_executable:
                        meta["issues"].append("script not executable (chmod +x recommended)")
                        # Note: we still count it as valid if size is OK
                        
            elif abs_path.is_dir():
                meta["kind"] = "dir"
                try:
                    num_entries = sum(1 for _ in abs_path.iterdir())
                    meta["num_entries"] = num_entries
                    
                    # Empty directories don't count as restored
                    if num_entries > 0:
                        meta["valid"] = True
                    else:
                        meta["issues"].append("directory is empty")
                except OSError as e:
                    meta["num_entries"] = -1
                    meta["issues"].append(f"cannot read directory: {e}")
            else:
                meta["kind"] = "other"
                meta["valid"] = True  # Symlinks, etc. - give benefit of doubt
            
            # Only count as restored if it passes quality checks
            if meta["valid"]:
                restored.append(rel_path)
            else:
                missing.append(rel_path)
                logger.warning(f"Restoration quality issue for '{rel_path}': {meta['issues']}")
        else:
            missing.append(rel_path)
        
        details.append(meta)
    
    rate = len(restored) / len(targets) if targets else 0.0
    
    return ReproducibilityMetrics(
        targets=targets,
        restored=restored,
        missing=missing,
        restoration_rate=rate,
        details=details,
    )


# =============================================================================
# FAITHFULNESS EVALUATION (LLM-as-Judge)
# =============================================================================

FAITHFULNESS_PROMPT = """You are an expert evaluator assessing whether an AI agent's answers are grounded in evidence.

## Task Context
The agent was given a reproducibility task to execute code from a scientific paper and answer questions about the results.

## Evaluation Criteria
A faithful answer is one that:
1. Can be directly traced to tool execution outputs (bash commands, file reads, etc.)
2. Uses actual values from experiment results, not guesses or simulations
3. Does not hallucinate or fabricate data

Red flags for unfaithful answers:
- Answers provided without corresponding tool evidence
- Using random/simulated values instead of actual execution results
- Phrases like "I'll simulate", "let's assume", "approximately" without evidence
- Answering questions about files/results that were never accessed

## Input
Questions asked: {questions}

Submitted answers:
{submitted_answers}

Tool calls and outputs (summarized):
{tool_summary}

## Output
Respond in JSON format:
{{
    "score": <float 0.0-1.0>,
    "is_grounded": <boolean>,
    "suspected_guessing": <boolean>,
    "reasoning": "<explanation of your assessment>",
    "evidence_summary": "<brief summary of what evidence supports the answers>",
    "flagged_answers": ["<list of question keys with suspicious answers>"]
}}
"""


async def evaluate_faithfulness(
    questions: list[str],
    submitted: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, str]],
    judge_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
) -> FaithfulnessMetrics:
    """
    Use LLM-as-judge to evaluate whether answers are grounded in evidence.
    
    Args:
        questions: List of required question keys
        submitted: Agent's submitted answers
        tool_calls: List of tool calls made by the agent
        tool_results: List of tool execution results
        judge_model: LLM model to use as judge (same format as purple agent)
        
    Returns:
        FaithfulnessMetrics from LLM evaluation
    """
    # Build tool summary (truncate long outputs)
    tool_summary = _build_tool_summary(tool_calls, tool_results)
    
    prompt = FAITHFULNESS_PROMPT.format(
        questions=json.dumps(questions, indent=2),
        submitted_answers=json.dumps(submitted, indent=2),
        tool_summary=tool_summary,
    )
    
    # Use same API configuration as purple agent
    # If COREBENCH_TEXT_API_BASE is set → self-hosted vLLM
    # Otherwise → Nebius API
    api_base = (os.environ.get("COREBENCH_TEXT_API_BASE") or "").strip()
    api_key = (os.environ.get("COREBENCH_TEXT_API_KEY") or os.environ.get("NEBIUS_API_KEY") or "").strip()
    
    if not api_base:
        # Use Nebius API
        api_base = "https://api.tokenfactory.nebius.com/v1/"
        api_key = os.environ.get("NEBIUS_API_KEY", "")
    
    if not api_key:
        logger.error("No API key found for LLM-as-judge (need COREBENCH_TEXT_API_KEY or NEBIUS_API_KEY)")
        return FaithfulnessMetrics(
            score=0.0,
            is_grounded=False,
            suspected_guessing=True,
            reasoning="Evaluation failed: No API key configured",
            evidence_summary="",
            flagged_answers=[],
        )
    
    # LiteLLM needs openai/ prefix for custom api_base with OpenAI-compatible APIs
    model_name = judge_model
    if not model_name.startswith("openai/"):
        model_name = f"openai/{judge_model}"
    
    logger.debug(f"LLM-as-judge: model={model_name}, api_base={api_base}")
    
    try:
        response = await litellm.acompletion(
            model=model_name,
            api_base=api_base,
            api_key=api_key,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return FaithfulnessMetrics(
            score=float(result.get("score", 0.0)),
            is_grounded=bool(result.get("is_grounded", False)),
            suspected_guessing=bool(result.get("suspected_guessing", True)),
            reasoning=str(result.get("reasoning", "")),
            evidence_summary=str(result.get("evidence_summary", "")),
            flagged_answers=list(result.get("flagged_answers", [])),
        )
        
    except Exception as e:
        logger.error(f"Faithfulness evaluation failed: {e}")
        return FaithfulnessMetrics(
            score=0.0, # Neutral score, but marked as error
            status="error",
            error_message=str(e),
            is_grounded=False,
            suspected_guessing=False,
            reasoning=f"Evaluation failed due to API error: {e}",
            evidence_summary="",
            flagged_answers=[],
        )


def _build_tool_summary(
    tool_calls: list[dict], 
    tool_results: list[dict],
    max_chars: int = 8000,
) -> str:
    """Build a summarized view of tool calls and results."""
    lines = []
    
    for i, (call, result) in enumerate(zip(tool_calls, tool_results), 1):
        tool_name = call.get("tool", "unknown")
        args = call.get("arguments", {})
        output = result.get("result", "")
        
        # Truncate long outputs
        if len(output) > 500:
            output = output[:250] + "\n...[truncated]...\n" + output[-250:]
        
        lines.append(f"[{i}] {tool_name}")
        if args:
            lines.append(f"    Args: {json.dumps(args)[:200]}")
        lines.append(f"    Output: {output}")
        lines.append("")
    
    summary = "\n".join(lines)
    
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n...[summary truncated]..."
    
    return summary if summary.strip() else "(No tool calls made)"


# =============================================================================
# TASK ADHERENCE EVALUATION (LLM-as-Judge)
# =============================================================================

TASK_ADHERENCE_PROMPT = """You are an expert evaluator assessing how well an AI agent executed a reproducibility benchmark task.

## Task Context
Domain: {domain}
Task prompt: {task_prompt}

The agent needed to:
1. Navigate the codebase and understand the project structure
2. Follow reproduction instructions (e.g., REPRODUCING.md, run scripts)
3. Execute the code to generate results
4. Extract answers to specific questions from the results

## Evaluation Criteria
Rate the agent on:
1. **Instruction Following**: Did it follow the task prompt and any documentation?
2. **Navigation Strategy**: How well did it explore and understand the codebase?
3. **Problem Solving**: How did it handle errors and obstacles?
4. **Rule Compliance**: Did it use tools appropriately without shortcuts or cheating?

## Execution Trace
Steps taken: {steps_used}
Tool calls: {tool_calls_count}
Protocol errors: {protocol_errors}
Command timeouts: {command_timeouts}

{timeout_note}

Key actions (summarized):
{action_summary}

Final answer provided: {has_answer}
Answer correctness: {answer_summary}

## Output
Respond in JSON format:
{{
    "score": <float 0.0-1.0>,
    "followed_instructions": <boolean>,
    "navigation_quality": "<excellent|good|fair|poor>",
    "reasoning": "<detailed explanation>",
    "strengths": ["<list of things done well>"],
    "weaknesses": ["<list of areas for improvement>"]
}}
"""


async def evaluate_task_adherence(
    domain: str,
    task_prompt: str,
    steps_used: int,
    tool_calls: list[dict],
    protocol_errors: int,
    submitted: dict[str, Any],
    accuracy_result: AccuracyMetrics,
    action_trace: list[dict],
    judge_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    command_timeouts: int = 0,
) -> TaskAdherenceMetrics:
    """
    Use LLM-as-judge to evaluate task execution quality.
    
    Args:
        domain: Task difficulty domain
        task_prompt: Original task instructions
        steps_used: Number of interaction steps
        tool_calls: List of tool calls made
        protocol_errors: Number of protocol/format errors
        submitted: Agent's submitted answers
        accuracy_result: Accuracy metrics
        action_trace: List of action events from trace
        judge_model: LLM model to use as judge
        command_timeouts: Number of commands that hit timeout limits
        
    Returns:
        TaskAdherenceMetrics from LLM evaluation
    """
    action_summary = _build_action_summary(action_trace)
    
    answer_summary = (
        f"{accuracy_result.correct_answers}/{accuracy_result.total_questions} correct"
        if accuracy_result.total_questions > 0
        else "No answers evaluated"
    )
    
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
    
    prompt = TASK_ADHERENCE_PROMPT.format(
        domain=domain,
        task_prompt=task_prompt,
        steps_used=steps_used,
        tool_calls_count=len(tool_calls),
        protocol_errors=protocol_errors,
        command_timeouts=command_timeouts,
        timeout_note=timeout_note,
        action_summary=action_summary,
        has_answer="Yes" if submitted else "No",
        answer_summary=answer_summary,
    )
    
    # Use same API configuration as purple agent
    api_base = (os.environ.get("COREBENCH_TEXT_API_BASE") or "").strip()
    api_key = (os.environ.get("COREBENCH_TEXT_API_KEY") or os.environ.get("NEBIUS_API_KEY") or "").strip()
    
    if not api_base:
        # Fall back to Nebius API
        api_base = "https://api.tokenfactory.nebius.com/v1/"
        api_key = os.environ.get("NEBIUS_API_KEY", "")
    
    if not api_key:
        logger.error("No API key set for LLM-as-judge evaluation (checked COREBENCH_TEXT_API_KEY and NEBIUS_API_KEY)")
        return TaskAdherenceMetrics(
            score=0.0,
            followed_instructions=False,
            navigation_quality="poor",
            reasoning="Evaluation failed: No API key available",
            strengths=[],
            weaknesses=[],
            status="error",
            error_message="No API key configured for LLM-as-judge",
        )
    
    # LiteLLM needs openai/ prefix when using custom api_base
    model_name = judge_model
    if not model_name.startswith("openai/"):
        model_name = f"openai/{judge_model}"
    
    logger.debug(f"Task adherence judge using model={model_name}, api_base={api_base}")
    
    try:
        response = await litellm.acompletion(
            model=model_name,
            api_base=api_base,
            api_key=api_key,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            # Note: temperature omitted as some models don't support it
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return TaskAdherenceMetrics(
            score=float(result.get("score", 0.0)),
            followed_instructions=bool(result.get("followed_instructions", False)),
            navigation_quality=str(result.get("navigation_quality", "poor")),
            reasoning=str(result.get("reasoning", "")),
            strengths=list(result.get("strengths", [])),
            weaknesses=list(result.get("weaknesses", [])),
        )
        
    except Exception as e:
        logger.error(f"Task adherence evaluation failed: {e}")
        return TaskAdherenceMetrics(
            score=0.0,
            followed_instructions=False,
            navigation_quality="poor",
            reasoning=f"Evaluation failed: {e}",
            strengths=[],
            weaknesses=[],
            status="error",
            error_message=str(e),
        )


def _build_action_summary(action_trace: list[dict], max_actions: int = 30) -> str:
    """
    Build a summary of agent actions. 
    If trace > max_actions, preserves the first 10 (setup) and last (max-10) actions (result).
    """
    
    # Helper to format a single event string
    def _format_event(event: dict) -> str:
        if event.get("type") != "action":
            return ""
            
        turn = event.get("turn", "?")
        name = event.get("name", "unknown")
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
    tool_calls: list[dict],
    time_seconds: float,
    protocol_errors: int,
    tool_results: Optional[list[dict]] = None,
) -> EfficiencyMetrics:
    """
    Compute efficiency metrics for the task execution.
    
    Args:
        steps_used: Number of interaction steps taken
        max_steps: Maximum allowed steps
        tool_calls: List of tool calls made
        time_seconds: Total time taken in seconds
        protocol_errors: Number of protocol/format errors
        tool_results: Optional list of tool result dicts to count timeouts
        
    Returns:
        EfficiencyMetrics
    """
    # Count commands that timed out
    command_timeouts = 0
    if tool_results:
        for result in tool_results:
            result_text = str(result.get("result", "")).lower()
            if "timed out" in result_text or "timeout" in result_text:
                command_timeouts += 1
    
    return EfficiencyMetrics(
        steps_used=steps_used,
        max_steps=max_steps,
        tool_calls=len(tool_calls),
        time_seconds=time_seconds,
        protocol_errors=protocol_errors,
        command_timeouts=command_timeouts,
    )


# =============================================================================
# AGGREGATE METRICS
# =============================================================================

@dataclass
class AggregateMetrics:
    """Aggregate metrics across all tasks in a benchmark run.
    
    Provides summary statistics for comparing agent performance across
    different models, configurations, or benchmark versions.
    
    Attributes:
        num_tasks: Total number of tasks evaluated
        num_successful: Tasks with 100% accuracy (all questions correct)
        pass_rate: Fraction of successful tasks (num_successful / num_tasks)
        
        mean_accuracy: Average accuracy across all tasks
        mean_written_accuracy: Average accuracy on text-based questions
        mean_vision_accuracy: Average accuracy on figure/chart questions
        accuracy_by_domain: Dict mapping domain (easy/medium/hard) to mean accuracy
        
        mean_restoration_rate: Average file restoration rate (medium/hard only)
        
        mean_faithfulness: Average LLM-judge faithfulness score
        num_suspected_guessing: Count of tasks flagged for potential guessing
        
        mean_adherence: Average LLM-judge task adherence score
        
        mean_steps: Average interaction steps per task
        mean_tool_calls: Average tool invocations per task
        mean_time: Average wall-clock time per task (seconds)
        
        task_results: Per-task summary dict for detailed analysis
    """
    num_tasks: int
    num_successful: int
    pass_rate: float
    
    # Accuracy
    mean_accuracy: float
    mean_written_accuracy: float
    mean_vision_accuracy: float
    
    # Reproducibility (if applicable)
    mean_restoration_rate: Optional[float]
    
    # Faithfulness
    mean_faithfulness: float
    num_suspected_guessing: int
    
    # Task adherence
    mean_adherence: float
    
    # Efficiency
    mean_steps: float
    mean_tool_calls: float
    mean_time: float
    
    # Per-task results
    task_results: dict[str, dict]


def aggregate_results(evaluations: list[TaskEvaluation]) -> AggregateMetrics:
    """
    Aggregate metrics across multiple task evaluations.
    
    Computes summary statistics including means, counts, and
    accuracy for comparing agent performance.
    
    Args:
        evaluations: List of TaskEvaluation results from individual tasks
        
    Returns:
        AggregateMetrics summary
    """
    if not evaluations:
        return AggregateMetrics(
            num_tasks=0,
            num_successful=0,
            pass_rate=0.0,
            mean_accuracy=0.0,
            mean_written_accuracy=0.0,
            mean_vision_accuracy=0.0,
            mean_restoration_rate=None,
            mean_faithfulness=0.0,
            num_suspected_guessing=0,
            mean_adherence=0.0,
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
    
    # Reproducibility (only count tasks with targets)
    repro_rates = [
        e.reproducibility.restoration_rate 
        for e in evaluations 
        if e.reproducibility and e.reproducibility.targets
    ]
    mean_restoration = np.mean(repro_rates) if repro_rates else None
    
    # Faithfulness
    mean_faithfulness = np.mean([e.faithfulness.score for e in evaluations])
    num_guessing = sum(1 for e in evaluations if e.faithfulness.suspected_guessing)
    
    # Adherence
    mean_adherence = np.mean([e.task_adherence.score for e in evaluations])
    
    # Efficiency
    mean_steps = np.mean([e.efficiency.steps_used for e in evaluations])
    mean_tools = np.mean([e.efficiency.tool_calls for e in evaluations])
    mean_time = np.mean([e.efficiency.time_seconds for e in evaluations])
    
    # Per-task summary
    task_results = {
        e.task_id: {
            "success": bool(e.success),
            "accuracy": float(e.accuracy.accuracy),
            "faithfulness": float(e.faithfulness.score),
            "adherence": float(e.task_adherence.score),
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
        mean_restoration_rate=float(mean_restoration) if mean_restoration is not None else None,
        mean_faithfulness=float(mean_faithfulness),
        num_suspected_guessing=num_guessing,
        mean_adherence=float(mean_adherence),
        mean_steps=float(mean_steps),
        mean_tool_calls=float(mean_tools),
        mean_time=float(mean_time),
        task_results=task_results,
    )
