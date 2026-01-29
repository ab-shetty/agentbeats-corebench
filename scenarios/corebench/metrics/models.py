"""
CoreBench Metrics Data Models
=============================

Dataclasses for all metric types used in CoreBench evaluation.
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Optional

import numpy as np


# =============================================================================
# JSON SERIALIZATION HELPERS
# =============================================================================

def _make_json_safe(obj: Any, decimal_places: int = 4) -> Any:
    """Recursively convert an object to be JSON serializable.

    Handles numpy types, dataclasses, and nested structures.
    Rounds floats to specified decimal places to avoid ugly artifacts
    like 0.3333333333333333 or 0.19999999999999998.
    """
    if obj is None:
        return None
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return round(float(obj), decimal_places)
    elif isinstance(obj, float):
        return round(obj, decimal_places)
    elif isinstance(obj, np.ndarray):
        return [_make_json_safe(item, decimal_places) for item in obj.tolist()]
    elif isinstance(obj, dict):
        return {str(k): _make_json_safe(v, decimal_places) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(item, decimal_places) for item in obj]
    elif hasattr(obj, '__dataclass_fields__'):
        return _make_json_safe(asdict(obj), decimal_places)
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
    extra_questions: list[str]  # Keys submitted by agent that weren't expected


@dataclass
class TaskAdherenceMetrics:
    """LLM-as-Judge metric for task execution quality, providing qualitative assessment and feedback.

    Evaluates how well the agent followed task instructions, navigated the codebase,
    and solved problems.
    """
    score: float                # Overall quality score from 0.0 (poor) to 1.0 (excellent)
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
class MethodologyScoreBreakdown:
    """Breakdown of methodology score components for transparency.

    Shows how each factor contributed to the final methodology score,
    making it easier to understand and debug scoring.
    """
    domain: str                          # Which scoring rubric was used
    doc_read_score: float                # Points from reading documentation
    script_read_score: float             # Points from reading target scripts
    execution_coverage_score: float      # Points from execution coverage
    successful_execution_score: float    # Points from successful execution
    error_recovery_score: float          # Points from error recovery
    penalty: float                       # Any penalties applied (negative)
    total: float                         # Final score after all components


@dataclass
class MethodologyMetrics:
    """Deterministic methodology metrics extracted from execution traces.

    These metrics capture observable behaviors without relying on LLM judgment,
    providing objective measurement of whether the agent followed correct methodology.
    """
    # Discovery phase
    read_documentation: bool        # Whether agent read README.md
    docs_read: list[str]            # List of documentation files that were read
    read_target_script: bool        # Whether agent inspected the script to run
    scripts_read: list[str]         # List of target script files that were read

    # Execution phase
    attempted_execution: bool       # Whether agent tried to run target script
    execution_attempts: int         # Number of execution attempts
    successful_execution: bool      # Whether at least one execution succeeded (exit_code=0)

    # Script coverage               - from task_prompt analysis
    expected_scripts: list[str]         # Scripts parsed from task_prompt that should be executed
    executed_scripts: list[str]         # Scripts the agent actually ran successfully
    attempted_failed_scripts: list[str] # Scripts the agent attempted but failed (exit_code != 0)
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
    score_breakdown: MethodologyScoreBreakdown | None = None  # Detailed breakdown of score components


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

    # Aggregate error diagnostics (for identifying systemic issues)
    error_type_distribution: dict[str, int] = field(default_factory=dict)  # Count of each error type across all tasks
