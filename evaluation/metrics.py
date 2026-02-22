#!/usr/bin/env python3
"""
Quantitative metric calculations and threshold tuning for toxicity classification.

This module provides comprehensive evaluation metrics including:
- F1 scores (macro, micro, weighted)
- Per-label metrics (precision, recall, F1)
- ROC-AUC and PR-AUC scores
- False Positive Rate analysis at different thresholds
- Confusion matrices per label (text + matplotlib figure)
- Threshold tuning via cross-validation
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)
from sklearn.model_selection import KFold


def calculate_f1_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate macro, micro, and weighted F1 scores.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_samples, n_labels).
    y_pred : np.ndarray
        Predicted binary labels of shape (n_samples, n_labels).

    Returns
    -------
    dict
        Dictionary with keys 'macro', 'micro', 'weighted' and F1 scores as values.
    """
    return {
        "macro":    float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "micro":    float(f1_score(y_true, y_pred, average="micro",    zero_division=0)),
        "weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def calculate_per_label_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate precision, recall, and F1 score for each label.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_samples, n_labels).
    y_pred : np.ndarray
        Predicted binary labels of shape (n_samples, n_labels).
    labels : List[str]
        List of label names corresponding to columns in y_true/y_pred.

    Returns
    -------
    dict
        Dictionary mapping label names to dictionaries with 'precision', 'recall', 'f1'.
    """
    if y_true.shape[1] != len(labels):
        raise ValueError(
            f"Number of labels ({len(labels)}) doesn't match "
            f"y_true columns ({y_true.shape[1]})"
        )

    per_label_metrics = {}
    for i, label in enumerate(labels):
        per_label_metrics[label] = {
            "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "recall":    float(recall_score(   y_true[:, i], y_pred[:, i], zero_division=0)),
            "f1":        float(f1_score(       y_true[:, i], y_pred[:, i], zero_division=0)),
        }

    return per_label_metrics


def calculate_roc_auc(
    y_true: np.ndarray, y_pred_proba: np.ndarray, labels: List[str]
) -> Dict[str, float]:
    """
    Calculate ROC-AUC score for each label.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_samples, n_labels).
    y_pred_proba : np.ndarray
        Predicted probabilities of shape (n_samples, n_labels).
    labels : List[str]
        List of label names corresponding to columns in y_true/y_pred_proba.

    Returns
    -------
    dict
        Dictionary mapping label names to ROC-AUC scores.
    """
    if y_true.shape[1] != len(labels):
        raise ValueError(
            f"Number of labels ({len(labels)}) doesn't match "
            f"y_true columns ({y_true.shape[1]})"
        )
    if y_pred_proba.shape[1] != len(labels):
        raise ValueError(
            f"Number of labels ({len(labels)}) doesn't match "
            f"y_pred_proba columns ({y_pred_proba.shape[1]})"
        )

    roc_auc_scores = {}
    for i, label in enumerate(labels):
        try:
            if len(np.unique(y_true[:, i])) < 2:
                roc_auc_scores[label] = float("nan")
            else:
                roc_auc_scores[label] = float(
                    roc_auc_score(y_true[:, i], y_pred_proba[:, i])
                )
        except ValueError:
            roc_auc_scores[label] = float("nan")

    return roc_auc_scores


def calculate_pr_auc(
    y_true: np.ndarray, y_pred_proba: np.ndarray, labels: List[str]
) -> Dict[str, float]:
    """
    Calculate PR-AUC (Precision-Recall AUC) score for each label.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_samples, n_labels).
    y_pred_proba : np.ndarray
        Predicted probabilities of shape (n_samples, n_labels).
    labels : List[str]
        List of label names corresponding to columns in y_true/y_pred_proba.

    Returns
    -------
    dict
        Dictionary mapping label names to PR-AUC scores.
    """
    if y_true.shape[1] != len(labels):
        raise ValueError(
            f"Number of labels ({len(labels)}) doesn't match "
            f"y_true columns ({y_true.shape[1]})"
        )
    if y_pred_proba.shape[1] != len(labels):
        raise ValueError(
            f"Number of labels ({len(labels)}) doesn't match "
            f"y_pred_proba columns ({y_pred_proba.shape[1]})"
        )

    pr_auc_scores = {}
    for i, label in enumerate(labels):
        try:
            if np.sum(y_true[:, i]) == 0:
                pr_auc_scores[label] = float("nan")
            else:
                precision, recall, _ = precision_recall_curve(
                    y_true[:, i], y_pred_proba[:, i]
                )
                pr_auc_scores[label] = float(auc(recall, precision))
        except ValueError:
            pr_auc_scores[label] = float("nan")

    return pr_auc_scores


def calculate_fpr_at_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    labels: List[str],
    thresholds: List[float] = [0.3, 0.5, 0.7, 0.9],
) -> Dict[float, Dict[str, float]]:
    """
    Calculate False Positive Rate at different probability thresholds for each label.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_samples, n_labels).
    y_pred_proba : np.ndarray
        Predicted probabilities of shape (n_samples, n_labels).
    labels : List[str]
        List of label names corresponding to columns in y_true/y_pred_proba.
    thresholds : List[float]
        List of probability thresholds to evaluate (default: [0.3, 0.5, 0.7, 0.9]).

    Returns
    -------
    dict
        Dictionary mapping thresholds to dictionaries mapping label names to FPR values.
    """
    if y_true.shape[1] != len(labels):
        raise ValueError(
            f"Number of labels ({len(labels)}) doesn't match "
            f"y_true columns ({y_true.shape[1]})"
        )
    if y_pred_proba.shape[1] != len(labels):
        raise ValueError(
            f"Number of labels ({len(labels)}) doesn't match "
            f"y_pred_proba columns ({y_pred_proba.shape[1]})"
        )

    fpr_results = {}
    for threshold in thresholds:
        fpr_results[threshold] = {}
        for i, label in enumerate(labels):
            y_pred_thresh = (y_pred_proba[:, i] >= threshold).astype(int)
            tn = np.sum((y_true[:, i] == 0) & (y_pred_thresh == 0))
            fp = np.sum((y_true[:, i] == 0) & (y_pred_thresh == 1))
            fpr_results[threshold][label] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return fpr_results


def generate_confusion_matrices(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]
) -> Dict[str, np.ndarray]:
    """
    Generate confusion matrix for each label.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_samples, n_labels).
    y_pred : np.ndarray
        Predicted binary labels of shape (n_samples, n_labels).
    labels : List[str]
        List of label names corresponding to columns in y_true/y_pred.

    Returns
    -------
    dict
        Dictionary mapping label names to confusion matrices of shape (2, 2).
        Format: [[TN, FP], [FN, TP]]
    """
    if y_true.shape[1] != len(labels):
        raise ValueError(
            f"Number of labels ({len(labels)}) doesn't match "
            f"y_true columns ({y_true.shape[1]})"
        )

    cms = {}
    for i, label in enumerate(labels):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        if cm.shape == (1, 1):
            if y_true[:, i].sum() == 0:
                cm = np.array([[cm[0, 0], 0], [0, 0]])
            else:
                cm = np.array([[0, 0], [0, cm[0, 0]]])
        elif cm.shape in ((2, 1), (1, 2)):
            cm_full = np.zeros((2, 2), dtype=int)
            if y_true[:, i].sum() == 0:
                cm_full[0, :] = cm.flatten()[:2] if cm.shape[1] == 2 else [cm[0, 0], 0]
            else:
                cm_full[1, :] = cm.flatten()[-2:] if cm.shape[1] == 2 else [0, cm[0, 0]]
            cm = cm_full
        cms[label] = cm

    return cms


# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib confusion matrix plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    labels: List[str],
    output_path: Path,
    model_type: str = "model",
    ncols: int = 3,
) -> None:
    """
    Plot and save confusion matrices for all labels as a single matplotlib figure.

    Each subplot shows a heatmap with raw counts (TN/FP/FN/TP) and per-label
    FPR / TPR / Precision / F1 in the subtitle.

    Parameters
    ----------
    confusion_matrices : dict
        Mapping of label names to (2, 2) numpy arrays [[TN, FP], [FN, TP]].
    labels : List[str]
        Ordered list of label names.
    output_path : Path
        Directory where the PNG will be saved.
    model_type : str
        Model identifier used in the filename and figure title.
    ncols : int
        Number of columns in the subplot grid. Default: 3.
    """
    n = len(labels)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.8 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    cell_tags = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}

    for ax, label in zip(axes_flat, labels):
        cm = confusion_matrices[label]
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])

        # Derived rates
        total_neg = tn + fp
        total_pos = fn + tp
        fpr_val  = fp / total_neg if total_neg > 0 else float("nan")
        tpr_val  = tp / total_pos if total_pos > 0 else float("nan")
        prec_val = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        f1_val   = (
            2 * prec_val * tpr_val / (prec_val + tpr_val)
            if not (np.isnan(prec_val) or np.isnan(tpr_val) or (prec_val + tpr_val) == 0)
            else float("nan")
        )

        fmt = lambda v: f"{v:.3f}" if not np.isnan(v) else "N/A"

        # Heatmap
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Axis labels and ticks
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred NEG", "Pred POS"], fontsize=10)
        ax.set_yticklabels(["Actual NEG", "Actual POS"], fontsize=10)
        ax.tick_params(axis="x", labelrotation=15)

        # Cell annotations: "TN\n800"
        thresh = cm.max() / 2.0
        for row in range(2):
            for col in range(2):
                color = "white" if cm[row, col] > thresh else "black"
                ax.text(
                    col, row,
                    f"{cell_tags[(row, col)]}\n{cm[row, col]}",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color,
                )

        # Per-label title + metric subtitle
        ax.set_title(label, fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel(
            f"FPR={fmt(fpr_val)}  TPR={fmt(tpr_val)}  "
            f"Prec={fmt(prec_val)}  F1={fmt(f1_val)}",
            fontsize=8.5, labelpad=8,
        )

    # Hide any unused subplot slots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Confusion Matrices — {model_type}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    save_path = output_path / f"confusion_matrices_{model_type}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix plot saved to: {save_path}")


# Threshold tuning

def tune_thresholds(
    model,
    val_data: Tuple[List[str], np.ndarray],
    labels: List[str],
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    threshold_step: float = 0.05,
    cv_folds: int = 5,
) -> Dict[str, float]:
    """
    Tune probability thresholds per label using cross-validation on validation set.

    Tests different thresholds and selects the one that maximizes F1 score for each label.

    Parameters
    ----------
    model
        Model with predict_proba() method.
    val_data : Tuple[List[str], np.ndarray]
        Validation data: (X_val, y_val) where X_val is List[str] and y_val is np.ndarray.
    labels : List[str]
        List of label names.
    threshold_range : Tuple[float, float]
        Range of thresholds to test (min, max). Default: (0.1, 0.9).
    threshold_step : float
        Step size for threshold grid search. Default: 0.05.
    cv_folds : int
        Number of cross-validation folds. Default: 5.

    Returns
    -------
    dict
        Dictionary mapping label names to optimal threshold values.
    """
    X_val, y_val = val_data

    thresholds = np.arange(
        threshold_range[0], threshold_range[1] + threshold_step, threshold_step
    )
    y_proba = model.predict_proba(X_val)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    best_thresholds = {}
    for label_idx, label in enumerate(labels):
        best_f1 = -1
        best_threshold = 0.5

        for threshold in thresholds:
            f1_scores_cv = []
            for _, test_idx in kf.split(X_val):
                y_true_fold = y_val[test_idx, label_idx]
                y_proba_fold = y_proba[test_idx, label_idx]
                y_pred_fold = (y_proba_fold >= threshold).astype(int)
                f1_scores_cv.append(f1_score(y_true_fold, y_pred_fold, zero_division=0))

            avg_f1 = np.mean(f1_scores_cv)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_threshold = threshold

        best_thresholds[label] = float(best_threshold)

    return best_thresholds


# Evaluation orchestrator

def evaluate_model(
    model,
    test_data: Tuple[List[str], np.ndarray],
    labels: List[str],
    config: Dict,
    tuned_thresholds: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Comprehensive model evaluation orchestrator.

    Computes all metrics, generates confusion matrices (text + matplotlib figure),
    and saves results to the outputs directory.

    Parameters
    ----------
    model
        Model with predict() and predict_proba() methods.
    test_data : Tuple[List[str], np.ndarray]
        Test data: (X_test, y_test) where X_test is List[str] and y_test is np.ndarray.
    labels : List[str]
        List of label names.
    config : dict
        Configuration dictionary containing paths and evaluation settings.
    tuned_thresholds : Optional[Dict[str, float]]
        Dictionary mapping label names to tuned thresholds. If None, uses 0.5 for all labels.

    Returns
    -------
    dict
        Comprehensive results dictionary containing all metrics and evaluations.
    """
    X_test, y_test = test_data

    thresholds = tuned_thresholds or {label: 0.5 for label in labels}

    y_pred_proba = model.predict_proba(X_test)

    y_pred = np.zeros_like(y_pred_proba)
    for i, label in enumerate(labels):
        y_pred[:, i] = (y_pred_proba[:, i] >= thresholds[label]).astype(int)

    f1_scores       = calculate_f1_scores(y_test, y_pred)
    per_label_mets  = calculate_per_label_metrics(y_test, y_pred, labels)
    roc_auc_scores  = calculate_roc_auc(y_test, y_pred_proba, labels)
    pr_auc_scores   = calculate_pr_auc(y_test, y_pred_proba, labels)

    eval_config          = config.get("evaluation", {})
    confidence_thresholds = eval_config.get("confidence_thresholds", [0.3, 0.5, 0.7, 0.9])
    fpr_at_thresholds    = calculate_fpr_at_thresholds(
        y_test, y_pred_proba, labels, confidence_thresholds
    )

    cms = generate_confusion_matrices(y_test, y_pred, labels)

    results = {
        "f1_scores":          f1_scores,
        "per_label_metrics":  per_label_mets,
        "roc_auc_scores":     roc_auc_scores,
        "pr_auc_scores":      pr_auc_scores,
        "fpr_at_thresholds":  fpr_at_thresholds,
        "confusion_matrices": cms,
        "tuned_thresholds":   thresholds,
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    model_type  = config.get("model_type", "logistic_regression")
    outputs_dir = config.get("paths", {}).get("outputs_dir", "./outputs")
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)

    # Save text metrics report
    metrics_file = outputs_path / f"metrics_{model_type}.txt"
    save_metrics_to_file(results, labels, model_type, metrics_file)

    # Save matplotlib confusion matrix figure
    try:
        plot_confusion_matrices(cms, labels, outputs_path, model_type)
    except Exception as e:
        print(f"Warning: Could not save confusion matrix plot: {e}")
        print("Confusion matrices are still available in the text metrics file.")

    return results


# Text report

def save_metrics_to_file(
    results: Dict, labels: List[str], model_type: str, filepath: Path
) -> None:
    """
    Save evaluation metrics to a readable text file.

    Parameters
    ----------
    results : dict
        Results dictionary from evaluate_model().
    labels : List[str]
        List of label names.
    model_type : str
        Model type identifier (e.g., "logistic_regression").
    filepath : Path
        Path to save the metrics file.
    """
    col_w = 14

    with open(filepath, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"EVALUATION RESULTS - {model_type}\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write("=" * 60 + "\n\n")

        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write(f"F1 Macro:    {results['f1_scores']['macro']:.3f}\n")
        f.write(f"F1 Micro:    {results['f1_scores']['micro']:.3f}\n")
        f.write(f"F1 Weighted: {results['f1_scores']['weighted']:.3f}\n\n")

        # Per-label metrics
        f.write("PER-LABEL METRICS\n")
        f.write(
            f"{'Label':<20} {'Precision':<12} {'Recall':<12} "
            f"{'F1':<12} {'ROC-AUC':<12} {'PR-AUC':<12}\n"
        )
        f.write("-" * 80 + "\n")
        for label in labels:
            m       = results["per_label_metrics"][label]
            roc_auc = results["roc_auc_scores"][label]
            pr_auc  = results["pr_auc_scores"][label]
            roc_str = f"{roc_auc:.3f}" if not np.isnan(roc_auc) else "N/A"
            pr_str  = f"{pr_auc:.3f}"  if not np.isnan(pr_auc)  else "N/A"
            f.write(
                f"{label:<20} {m['precision']:<12.3f} {m['recall']:<12.3f} "
                f"{m['f1']:<12.3f} {roc_str:<12} {pr_str:<12}\n"
            )
        f.write("\n")

        # Tuned thresholds
        f.write("TUNED THRESHOLDS\n")
        for label in labels:
            f.write(f"  {label}: {results['tuned_thresholds'][label]:.2f}\n")
        f.write("\n")

        # FPR at thresholds
        f.write("FALSE POSITIVE RATE AT THRESHOLDS\n")
        for threshold in sorted(results["fpr_at_thresholds"].keys()):
            fpr_dict = results["fpr_at_thresholds"][threshold]
            fpr_str  = ", ".join(
                [f"{label}={fpr_dict[label]:.3f}" for label in labels]
            )
            f.write(f"  Threshold {threshold:.2f}: {fpr_str}\n")
        f.write("\n")



    print(f"Metrics saved to: {filepath}")
    
    # Print confusion matrix summary to console
    print("\nConfusion Matrix Summary:")
    print("-" * 80)
    for label in labels:
        cm = results["confusion_matrices"][label]
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])
        print(f"{label:20} TN={tn:6} FP={fp:6} FN={fn:6} TP={tp:6}")
    print("-" * 80)
