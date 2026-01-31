"""CoreBench Metrics Package - Evaluation metrics for reproducibility benchmark."""

# Re-export models for clean imports
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

# Re-export evaluation functions
from scenarios.corebench.metrics.metrics import (
    evaluate_accuracy,
    evaluate_task_adherence,
    aggregate_results,
    extract_methodology_metrics,
    _empty_accuracy_metrics,
)

__all__ = [
    # Models
    "QuestionResult",
    "AccuracyMetrics",
    "TaskAdherenceMetrics",
    "ErrorRecoveryMetrics",
    "MethodologyScoreBreakdown",
    "MethodologyMetrics",
    "TaskEvaluation",
    "AggregateMetrics",
    # Functions
    "evaluate_accuracy",
    "evaluate_task_adherence",
    "aggregate_results",
    "extract_methodology_metrics",
    "_empty_accuracy_metrics",
]
