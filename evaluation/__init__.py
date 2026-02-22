"""
Evaluation module for toxicity classification models.

Provides quantitative metrics, threshold tuning, and evaluation.
"""

from .metrics import (
    calculate_f1_scores,
    calculate_per_label_metrics,
    calculate_roc_auc,
    calculate_pr_auc,
    calculate_fpr_at_thresholds,
    generate_confusion_matrices,
    tune_thresholds,
    evaluate_model,
)

__all__ = [
    "calculate_f1_scores",
    "calculate_per_label_metrics",
    "calculate_roc_auc",
    "calculate_pr_auc",
    "calculate_fpr_at_thresholds",
    "generate_confusion_matrices",
    "tune_thresholds",
    "evaluate_model",
]

