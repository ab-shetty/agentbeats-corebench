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
from typing import Any, Callable, Optional
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

    if logger.isEnabledFor(logging.DEBUG):
        max_entries = 400
        max_depth = 3
        stat_budget = 80
        skip_names = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

        entries_logged = 0
        stats_used = 0
        lines: list[str] = []
        stack: list[tuple[str, int]] = [(os.fspath(workspace), 0)]

        while stack and entries_logged < max_entries:
            dir_path, depth = stack.pop()
            try:
                with os.scandir(dir_path) as it:
                    for entry in it:
                        if entries_logged >= max_entries:
                            break
                        if entry.name in skip_names:
                            continue

                        rel = os.path.relpath(entry.path, os.fspath(workspace))

                        if entry.is_dir(follow_symlinks=False):
                            lines.append(f"d {rel}/")
                            entries_logged += 1
                            if depth + 1 < max_depth:
                                stack.append((entry.path, depth + 1))
                            continue

                        if entry.is_file(follow_symlinks=False):
                            size_suffix = ""
                            if stats_used < stat_budget:
                                try:
                                    size_suffix = f" ({entry.stat(follow_symlinks=False).st_size}B)"
                                except OSError:
                                    size_suffix = ""
                                stats_used += 1
                            lines.append(f"f {rel}{size_suffix}")
                            entries_logged += 1
                            continue

                        lines.append(f"o {rel}")
                        entries_logged += 1
            except OSError as e:
                rel_dir = os.path.relpath(dir_path, os.fspath(workspace))
                lines.append(f"! {rel_dir}: {e}")
                entries_logged += 1

        if stack:
            lines.append(f"... [truncated after {max_entries} entries] ...")
        if lines:
            logger.debug(
                "Workspace contents under %s (depth<=%d):\n%s",
                workspace,
                max_depth,
                "\n".join(lines),
            )
    
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
# TASK ADHERENCE EVALUATION (LLM-as-Judge)
# judge gets passed tool calls + results
# , 
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
**Gold Standard:** Read README.md → Execute documented command (usually Docker) → Handle errors → Generate output

**Automatic Disqualifiers (cap score at 0.3):**
- Never read README.md
- Read pre-existing results instead of executing
- Never attempted documented command

### HARD MODE (corebench_hard)
**Gold Standard:** Check Dockerfile/README → Infer execution method → Attempt Docker/script execution → Debug errors → Generate output

**Automatic Disqualifiers (cap score at 0.3):**
- Never explored documentation (Dockerfile/README)  
- Read pre-existing results instead of executing
- Never attempted to run scripts
- Made no effort to resolve dependency errors

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

## Scoring Examples

**Example A: Medium - Perfect Execution (0.95)**
```
Trace shows: Read REPRODUCING.md → Executed Docker command from docs → Generated results
Scoring: Core 50/50 + Problem 24/25 + Discovery 15/15 + Technical 10/10 = 0.99 → 0.95
Reasoning: Perfect adherence to documented process. Minor deduction for not checking existing results first.
```

**Example B: Hard - Timeout (0.82)**
```
Trace shows: Found Dockerfile → Built Docker image → Ran container → TIMED OUT (infrastructure)
Scoring: Core 48/50 + Problem 23/25 + Discovery 15/15 + Technical 10/10 = 0.96 → 0.82
Reasoning: Correct approach throughout. Timeout is infrastructure issue, not agent error. Would be 0.95+ without timeout.
Penalty: -0.10 for incomplete (not agent's fault, but didn't finish)
```

**Example C: Medium - Read Existing Results (0.25)**
```
Trace shows: Listed files → Found results/ → Read output files → Never read REPRODUCING.md → Never executed
Scoring: Core 5/50 + Problem 5/25 + Discovery 10/15 + Technical 5/10 = 0.25
Reasoning: FAILED CORE OBJECTIVE. Found shortcut, didn't reproduce. Automatic cap at 0.3.
Penalty: -0.6 (read existing results)
```

**Example D: Hard - Gave Up Quickly (0.48)**
```
Trace shows: Ran script → ModuleNotFoundError → Tried pip install (failed) → Gave up → Never checked Dockerfile
Scoring: Core 20/50 + Problem 10/25 + Discovery 8/15 + Technical 10/10 = 0.48
Reasoning: Attempted execution (good) but gave up after one obstacle. Never explored Docker solution.
Penalty: -0.3 (gave up quickly)
```

**Example E: Medium - Wrong Method but Succeeded (0.68)**
```
Trace shows: Read REPRODUCING.md (said use Docker) → Docker unavailable → Manually installed deps → Ran script → Success
Scoring: Core 30/50 + Problem 23/25 + Discovery 14/15 + Technical 9/10 = 0.76 → 0.68
Reasoning: Reproduced successfully but didn't use documented method. Good problem-solving.
Penalty: -0.2 (deviated from instructions)
```

## Execution Trace
Steps taken: {steps_used}
Tool calls: {tool_calls_count}
Protocol errors: {protocol_errors}
Command timeouts: {command_timeouts}

{timeout_note}

Tool calls + results:
{tool_interactions}

Final answer: {has_answer}
Answer correctness: {answer_summary}

## Your Task

1. Assign component scores to the trace: Core _/50, Problem _/25, Discovery _/15, Technical _/10
2. Check for automatic penalties
3. Calculate final score and write reasoning

## Output Format
```json
{{
    "score": <float 0.0-1.0>,
    "followed_instructions": <boolean>,
    "navigation_quality": "<excellent|good|fair|poor>",
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
    judge_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    command_timeouts: int = 0,
) -> TaskAdherenceMetrics:
    """Use LLM-as-judge to evaluate task execution quality."""
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
        model_name = judge_model
        if not model_name.startswith("openai/"):
            model_name = f"openai/{judge_model}"
        if trace_event_callback is not None:
            try:
                trace_event_callback(
                    {
                        "type": "llm_judge_skipped",
                        "judge": "task_adherence",
                        "flow": "green -> judge",
                        "domain": domain,
                        "model": model_name,
                        "reason": "missing_api_key",
                        "api_base": api_base,
                        "has_api_key": False,
                        "prompt": prompt,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to write task adherence judge skipped trace: {e}")
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
                    "answer_summary": answer_summary,
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
            model=model_name,
            api_base=api_base,
            api_key=api_key,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
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
            navigation_quality=str(result.get("navigation_quality", "poor")),
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
            navigation_quality="poor",
            reasoning=f"Evaluation failed: {e}",
            strengths=[],
            weaknesses=[],
            status="error",
            error_message=str(e),
        )


def _build_action_summary(action_trace: list[dict], max_actions: int = 30) -> str:
    """
    Build a summary of purple agent actions and the MCP server results. 
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
    
    mean_accuracy: float
    mean_written_accuracy: float
    mean_vision_accuracy: float
    
    mean_restoration_rate: Optional[float]
    
    mean_adherence: float
    
    mean_steps: float
    mean_tool_calls: float
    mean_time: float
    
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
        mean_adherence=float(mean_adherence),
        mean_steps=float(mean_steps),
        mean_tool_calls=float(mean_tools),
        mean_time=float(mean_time),
        task_results=task_results,
    )
