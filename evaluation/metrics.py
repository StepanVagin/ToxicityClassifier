#!/usr/bin/env python3
"""
Quantitative metric calculations and threshold tuning for toxicity classification.

This module provides comprehensive evaluation metrics including:
- F1 scores (macro, micro, weighted)
- Per-label metrics (precision, recall, F1)
- ROC-AUC and PR-AUC scores
- False Positive Rate analysis at different thresholds
- Confusion matrices per label (text + matplotlib figure)
- FPR vs. Recall curves at confidence thresholds
- Qualitative error analysis (false positives/negatives with strategic examples)
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


def calculate_gini_from_roc_auc(roc_auc_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate Gini coefficient from ROC-AUC scores.
    Gini = 2 * ROC_AUC - 1 (for binary classification).

    Parameters
    ----------
    roc_auc_scores : dict
        Dictionary mapping label names to ROC-AUC values.

    Returns
    -------
    dict
        Dictionary mapping label names to Gini values.
    """
    return {
        label: float(2 * auc - 1) if not np.isnan(auc) else float("nan")
        for label, auc in roc_auc_scores.items()
    }


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


def calculate_recall_at_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    labels: List[str],
    thresholds: List[float] = [0.3, 0.5, 0.7, 0.9],
) -> Dict[float, Dict[str, float]]:
    """
    Calculate Recall (True Positive Rate) at different probability thresholds for each label.

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
        Dictionary mapping thresholds to dictionaries mapping label names to recall values.
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

    recall_results = {}
    for threshold in thresholds:
        recall_results[threshold] = {}
        for i, label in enumerate(labels):
            y_pred_thresh = (y_pred_proba[:, i] >= threshold).astype(int)
            fn = np.sum((y_true[:, i] == 1) & (y_pred_thresh == 0))
            tp = np.sum((y_true[:, i] == 1) & (y_pred_thresh == 1))
            recall_results[threshold][label] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return recall_results


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


def plot_fpr_vs_recall_curves(
    fpr_at_thresholds: Dict[float, Dict[str, float]],
    recall_at_thresholds: Dict[float, Dict[str, float]],
    labels: List[str],
    output_path: Path,
    model_type: str = "model",
    ncols: int = 3,
) -> None:
    """
    Plot FPR vs. Recall curves at different confidence thresholds for all labels.

    Each subplot shows FPR (x-axis) vs. Recall (y-axis) with points for each threshold.
    Thresholds are color-coded and labeled.

    Parameters
    ----------
    fpr_at_thresholds : dict
        Dictionary mapping thresholds to dictionaries mapping label names to FPR values.
    recall_at_thresholds : dict
        Dictionary mapping thresholds to dictionaries mapping label names to recall values.
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

    # Color map for thresholds
    thresholds = sorted(fpr_at_thresholds.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
    threshold_colors = dict(zip(thresholds, colors))

    for ax, label in zip(axes_flat, labels):
        # Collect FPR and recall values for this label
        fpr_values = []
        recall_values = []
        threshold_labels = []
        
        for threshold in thresholds:
            fpr_val = fpr_at_thresholds[threshold].get(label, 0.0)
            recall_val = recall_at_thresholds[threshold].get(label, 0.0)
            fpr_values.append(fpr_val)
            recall_values.append(recall_val)
            threshold_labels.append(f"{threshold:.1f}")

        # Plot points with lines connecting them
        ax.plot(fpr_values, recall_values, 'o-', linewidth=2, markersize=8, alpha=0.7)
        
        # Annotate each point with threshold value
        for i, (fpr_val, recall_val, thresh) in enumerate(zip(fpr_values, recall_values, threshold_labels)):
            ax.annotate(
                thresh,
                (fpr_val, recall_val),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                alpha=0.8,
            )

        ax.set_xlabel('False Positive Rate (FPR)', fontsize=11)
        ax.set_ylabel('Recall', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=6)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, max(fpr_values) * 1.1 if fpr_values else 1.05)
        ax.set_ylim(-0.05, max(recall_values) * 1.1 if recall_values else 1.05)

    # Hide any unused subplot slots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"FPR vs. Recall Curves at Confidence Thresholds — {model_type}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    save_path = output_path / f"fpr_vs_recall_curves_{model_type}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"FPR vs. Recall curves plot saved to: {save_path}")


def plot_roc_auc_gini(
    roc_auc_scores: Dict[str, float],
    gini_scores: Dict[str, float],
    labels: List[str],
    output_path: Path,
    model_type: str = "model",
) -> None:
    """
    Plot ROC-AUC and Gini per label as a grouped bar chart.

    Parameters
    ----------
    roc_auc_scores : dict
        Dictionary mapping label names to ROC-AUC values.
    gini_scores : dict
        Dictionary mapping label names to Gini values.
    labels : List[str]
        Ordered list of label names.
    output_path : Path
        Directory where the PNG will be saved.
    model_type : str
        Model identifier used in the filename and figure title.
    """
    x = np.arange(len(labels))
    width = 0.35

    roc_vals = [roc_auc_scores.get(l, float("nan")) for l in labels]
    gini_vals = [gini_scores.get(l, float("nan")) for l in labels]

    # Replace nan with 0 for plotting (will show as empty)
    roc_plot = [v if not np.isnan(v) else 0 for v in roc_vals]
    gini_plot = [v if not np.isnan(v) else 0 for v in gini_vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, roc_plot, width, label="ROC-AUC", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, gini_plot, width, label="Gini", color="coral", alpha=0.85)

    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("Label", fontsize=11)
    ax.set_title(f"Test ROC-AUC and Gini by Label — {model_type}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars1, roc_vals):
        if not np.isnan(val):
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
    for bar, val in zip(bars2, gini_vals):
        if not np.isnan(val):
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    plt.tight_layout()
    save_path = output_path / f"roc_auc_gini_{model_type}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC-AUC and Gini plot saved to: {save_path}")


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    labels: List[str],
    thresholds: Dict[str, float],
    output_path: Path,
    model_type: str = "model",
    ncols: int = 3,
) -> None:
    """
    Plot per-label precision-recall curves with the chosen threshold marked.

    Each subplot shows precision vs recall as threshold varies. The tuned
    threshold point is marked (e.g., threat may use 0.3 for high recall,
    toxic may use 0.5 for balance).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_samples, n_labels).
    y_pred_proba : np.ndarray
        Predicted probabilities of shape (n_samples, n_labels).
    labels : List[str]
        Ordered list of label names.
    thresholds : Dict[str, float]
        Chosen threshold per label (marked on each curve).
    output_path : Path
        Directory where the PNG will be saved.
    model_type : str
        Model identifier.
    ncols : int
        Number of columns in subplot grid.
    """
    n = len(labels)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, (label_idx, label) in zip(axes_flat, enumerate(labels)):
        precision, recall, pr_thresholds = precision_recall_curve(
            y_true[:, label_idx], y_pred_proba[:, label_idx]
        )
        ax.plot(recall, precision, "b-", linewidth=2, label="PR curve")

        thr = thresholds.get(label, 0.5)
        prec_at_thr = precision_score(
            y_true[:, label_idx],
            (y_pred_proba[:, label_idx] >= thr).astype(int),
            zero_division=0,
        )
        rec_at_thr = recall_score(
            y_true[:, label_idx],
            (y_pred_proba[:, label_idx] >= thr).astype(int),
            zero_division=0,
        )
        ax.plot(
            rec_at_thr,
            prec_at_thr,
            "ro",
            markersize=10,
            label=f"threshold={thr:.2f}",
        )
        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=6)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Per-Label Precision-Recall Curves — {model_type}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    save_path = output_path / f"precision_recall_curves_{model_type}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Precision-recall curves plot saved to: {save_path}")


# Threshold tuning

def tune_thresholds(
    model,
    val_data: Tuple[List[str], np.ndarray],
    labels: List[str],
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    threshold_step: float = 0.05,
    cv_folds: int = 5,
    optimize_f1: str = "per_label",
) -> Dict[str, float]:
    """
    Tune probability thresholds per label using cross-validation on validation set.

    When optimize_f1 is "per_label", each label's threshold is optimized independently
    to maximize that label's F1. When "macro" or "weighted", thresholds are optimized
    jointly (iteratively) to maximize macro or weighted F1 across all labels.

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
    optimize_f1 : str
        "per_label" (maximize each label's F1), "macro", or "weighted".

    Returns
    -------
    dict
        Dictionary mapping label names to optimal threshold values.
    """
    X_val, y_val = val_data
    threshold_grid = np.arange(
        threshold_range[0], threshold_range[1] + threshold_step, threshold_step
    )
    y_proba = model.predict_proba(X_val)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    if optimize_f1 == "per_label":
        best_thresholds = {}
        for label_idx, label in enumerate(labels):
            best_f1 = -1
            best_threshold = 0.5
            for threshold in threshold_grid:
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

    # Macro or weighted: iterative per-label optimization
    average = "macro" if optimize_f1 == "macro" else "weighted"
    best_thresholds = {label: 0.5 for label in labels}
    max_iter = 3  # passes over all labels
    for _ in range(max_iter):
        for label_idx, label in enumerate(labels):
            best_score = -1
            best_threshold = best_thresholds[label]
            for threshold in threshold_grid:
                f1_scores_cv = []
                for _, test_idx in kf.split(X_val):
                    y_true_fold = y_val[test_idx]
                    y_pred_fold = np.zeros_like(y_true_fold)
                    for i, lbl in enumerate(labels):
                        thr = best_thresholds[lbl] if lbl != label else threshold
                        y_pred_fold[:, i] = (y_proba[test_idx, i] >= thr).astype(int)
                    f1_scores_cv.append(
                        f1_score(y_true_fold, y_pred_fold, average=average, zero_division=0)
                    )
                avg_f1 = np.mean(f1_scores_cv)
                if avg_f1 > best_score:
                    best_score = avg_f1
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
    gini_scores     = calculate_gini_from_roc_auc(roc_auc_scores)
    pr_auc_scores   = calculate_pr_auc(y_test, y_pred_proba, labels)

    eval_config          = config.get("evaluation", {})
    confidence_thresholds = eval_config.get("confidence_thresholds", [0.3, 0.5, 0.7, 0.9])
    fpr_at_thresholds    = calculate_fpr_at_thresholds(
        y_test, y_pred_proba, labels, confidence_thresholds
    )
    recall_at_thresholds = calculate_recall_at_thresholds(
        y_test, y_pred_proba, labels, confidence_thresholds
    )

    cms = generate_confusion_matrices(y_test, y_pred, labels)
    
    # Qualitative error analysis
    max_examples = eval_config.get("max_error_examples_per_label", 10)
    error_analysis = analyze_false_positives_negatives(
        X_test, y_test, y_pred, y_pred_proba, labels, thresholds, max_examples
    )

    results = {
        "f1_scores":          f1_scores,
        "per_label_metrics":  per_label_mets,
        "roc_auc_scores":     roc_auc_scores,
        "gini_scores":        gini_scores,
        "pr_auc_scores":      pr_auc_scores,
        "fpr_at_thresholds":  fpr_at_thresholds,
        "recall_at_thresholds": recall_at_thresholds,
        "confusion_matrices": cms,
        "error_analysis":     error_analysis,
        "tuned_thresholds":   thresholds,
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    model_type  = config.get("model_type", "logistic_regression")
    outputs_dir = config.get("paths", {}).get("outputs_dir", "./outputs")
    outputs_path = Path(outputs_dir)
    if not outputs_path.is_absolute():
        # Resolve relative to current working directory or project root
        outputs_path = Path.cwd() / outputs_path
    outputs_path.mkdir(parents=True, exist_ok=True)

    # Save text metrics report
    metrics_file = outputs_path / f"metrics_{model_type}.txt"
    save_metrics_to_file(results, labels, model_type, metrics_file)
    
    # Save qualitative error analysis
    analysis_file = outputs_path / f"error_analysis_{model_type}.txt"
    try:
        save_qualitative_analysis(error_analysis, labels, model_type, analysis_file)
    except Exception as e:
        print(f"Warning: Could not save qualitative analysis: {e}")

    # Save matplotlib confusion matrix figure
    try:
        plot_confusion_matrices(cms, labels, outputs_path, model_type)
    except Exception as e:
        print(f"Warning: Could not save confusion matrix plot: {e}")
        print("Confusion matrices are still available in the text metrics file.")

    # Save FPR vs. Recall curves plot
    try:
        plot_fpr_vs_recall_curves(
            fpr_at_thresholds, recall_at_thresholds, labels, outputs_path, model_type
        )
    except Exception as e:
        print(f"Warning: Could not save FPR vs. Recall curves plot: {e}")

    # Save ROC-AUC and Gini bar chart
    try:
        plot_roc_auc_gini(
            roc_auc_scores, gini_scores, labels, outputs_path, model_type
        )
    except Exception as e:
        print(f"Warning: Could not save ROC-AUC/Gini plot: {e}")

    # Save per-label precision-recall curves
    try:
        plot_precision_recall_curves(
            y_test, y_pred_proba, labels, thresholds, outputs_path, model_type
        )
    except Exception as e:
        print(f"Warning: Could not save precision-recall curves: {e}")

    return results


# Qualitative error analysis

def analyze_false_positives_negatives(
    X_test: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    labels: List[str],
    thresholds: Dict[str, float],
    max_examples_per_label: int = 10,
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Analyze false positives and false negatives for each label, selecting strategic examples.

    Strategic examples are prioritized by:
    1. High confidence errors (high prob for FP, low prob for FN)
    2. Multi-label errors (errors affecting multiple labels)
    3. High probability differences

    Parameters
    ----------
    X_test : List[str]
        List of test texts.
    y_true : np.ndarray
        True binary labels of shape (n_samples, n_labels).
    y_pred : np.ndarray
        Predicted binary labels of shape (n_samples, n_labels).
    y_pred_proba : np.ndarray
        Predicted probabilities of shape (n_samples, n_labels).
    labels : List[str]
        List of label names.
    thresholds : Dict[str, float]
        Dictionary mapping label names to thresholds used for predictions.
    max_examples_per_label : int
        Maximum number of examples to return per label per error type. Default: 10.

    Returns
    -------
    dict
        Dictionary with structure:
        {
            label: {
                'false_positives': [
                    {'text': str, 'probability': float, 'num_other_labels': int, 'other_labels': List[str]}
                ],
                'false_negatives': [
                    {'text': str, 'probability': float, 'num_other_labels': int, 'other_labels': List[str]}
                ]
            }
        }
    """
    analysis = {label: {'false_positives': [], 'false_negatives': []} for label in labels}

    for label_idx, label in enumerate(labels):
        threshold = thresholds[label]
        
        # Identify false positives and false negatives
        fp_mask = (y_true[:, label_idx] == 0) & (y_pred[:, label_idx] == 1)
        fn_mask = (y_true[:, label_idx] == 1) & (y_pred[:, label_idx] == 0)
        
        fp_indices = np.where(fp_mask)[0]
        fn_indices = np.where(fn_mask)[0]
        
        # Collect FP examples with metadata
        fp_examples = []
        for idx in fp_indices:
            prob = float(y_pred_proba[idx, label_idx])
            # Count how many other labels are also incorrectly predicted
            other_errors = sum(
                1 for i in range(len(labels))
                if i != label_idx and y_true[idx, i] != y_pred[idx, i]
            )
            other_labels = [
                labels[i] for i in range(len(labels))
                if i != label_idx and y_true[idx, i] != y_pred[idx, i]
            ]
            fp_examples.append({
                'index': int(idx),
                'text': X_test[idx],
                'probability': prob,
                'num_other_labels': other_errors,
                'other_labels': other_labels,
                'confidence_error': prob - threshold,  # How far above threshold
            })
        
        # Collect FN examples with metadata
        fn_examples = []
        for idx in fn_indices:
            prob = float(y_pred_proba[idx, label_idx])
            # Count how many other labels are also incorrectly predicted
            other_errors = sum(
                1 for i in range(len(labels))
                if i != label_idx and y_true[idx, i] != y_pred[idx, i]
            )
            other_labels = [
                labels[i] for i in range(len(labels))
                if i != label_idx and y_true[idx, i] != y_pred[idx, i]
            ]
            fn_examples.append({
                'index': int(idx),
                'text': X_test[idx],
                'probability': prob,
                'num_other_labels': other_errors,
                'other_labels': other_labels,
                'confidence_error': threshold - prob,  # How far below threshold
            })
        
        # Strategic selection: prioritize high confidence errors and multi-label errors
        # Sort FP by: (1) confidence_error (desc), (2) num_other_labels (desc)
        fp_examples.sort(key=lambda x: (x['confidence_error'], x['num_other_labels']), reverse=True)
        
        # Sort FN by: (1) confidence_error (desc), (2) num_other_labels (desc)
        fn_examples.sort(key=lambda x: (x['confidence_error'], x['num_other_labels']), reverse=True)
        
        # Select top examples
        analysis[label]['false_positives'] = fp_examples[:max_examples_per_label]
        analysis[label]['false_negatives'] = fn_examples[:max_examples_per_label]
    
    return analysis


def save_qualitative_analysis(
    analysis: Dict[str, Dict[str, List[Dict]]],
    labels: List[str],
    model_type: str,
    filepath: Path,
    max_text_length: int = 200,
) -> None:
    """
    Save qualitative error analysis to a readable text file.

    Parameters
    ----------
    analysis : dict
        Analysis dictionary from analyze_false_positives_negatives().
    labels : List[str]
        List of label names.
    model_type : str
        Model type identifier.
    filepath : Path
        Path to save the analysis file.
    max_text_length : int
        Maximum length of text to display (will be truncated). Default: 200.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"QUALITATIVE ERROR ANALYSIS - {model_type}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("STRATEGIC ERROR EXAMPLES\n")
        f.write("=" * 80 + "\n")
        f.write("Examples are prioritized by:\n")
        f.write("  1. High confidence errors (most certain mistakes)\n")
        f.write("  2. Multi-label errors (affecting multiple labels)\n")
        f.write("  3. High probability differences from threshold\n\n")
        
        for label in labels:
            fp_examples = analysis[label]['false_positives']
            fn_examples = analysis[label]['false_negatives']
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"LABEL: {label.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            # False Positives
            f.write(f"FALSE POSITIVES ({len(fp_examples)} examples shown)\n")
            f.write("-" * 80 + "\n")
            f.write("Predicted as toxic but actually NOT toxic\n")
            f.write("-" * 80 + "\n\n")
            
            if not fp_examples:
                f.write("  No false positives found.\n\n")
            else:
                for i, ex in enumerate(fp_examples, 1):
                    text_preview = ex['text'][:max_text_length]
                    if len(ex['text']) > max_text_length:
                        text_preview += "..."
                    
                    f.write(f"  Example {i}:\n")
                    f.write(f"    Text: {text_preview}\n")
                    f.write(f"    Predicted Probability: {ex['probability']:.4f}\n")
                    f.write(f"    Confidence Error: {ex['confidence_error']:.4f} (above threshold)\n")
                    if ex['num_other_labels'] > 0:
                        f.write(f"    Also misclassified: {', '.join(ex['other_labels'])}\n")
                    f.write("\n")
            
            # False Negatives
            f.write(f"\nFALSE NEGATIVES ({len(fn_examples)} examples shown)\n")
            f.write("-" * 80 + "\n")
            f.write("Predicted as NOT toxic but actually IS toxic\n")
            f.write("-" * 80 + "\n\n")
            
            if not fn_examples:
                f.write("  No false negatives found.\n\n")
            else:
                for i, ex in enumerate(fn_examples, 1):
                    text_preview = ex['text'][:max_text_length]
                    if len(ex['text']) > max_text_length:
                        text_preview += "..."
                    
                    f.write(f"  Example {i}:\n")
                    f.write(f"    Text: {text_preview}\n")
                    f.write(f"    Predicted Probability: {ex['probability']:.4f}\n")
                    f.write(f"    Confidence Error: {ex['confidence_error']:.4f} (below threshold)\n")
                    if ex['num_other_labels'] > 0:
                        f.write(f"    Also misclassified: {', '.join(ex['other_labels'])}\n")
                    f.write("\n")
    
    print(f"Qualitative analysis saved to: {filepath.resolve()}")


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
        roc_auc_vals = [v for v in results["roc_auc_scores"].values() if not np.isnan(v)]
        gini_vals = [v for v in results["gini_scores"].values() if not np.isnan(v)]
        roc_auc_macro = np.mean(roc_auc_vals) if roc_auc_vals else float("nan")
        gini_macro = np.mean(gini_vals) if gini_vals else float("nan")

        f.write("OVERALL METRICS\n")
        f.write(f"F1 Macro:      {results['f1_scores']['macro']:.3f}\n")
        f.write(f"F1 Micro:      {results['f1_scores']['micro']:.3f}\n")
        f.write(f"F1 Weighted:   {results['f1_scores']['weighted']:.3f}\n")
        f.write(f"ROC-AUC Macro: {roc_auc_macro:.3f}\n")
        f.write(f"Gini Macro:    {gini_macro:.3f}\n\n")

        # Per-label metrics
        f.write("PER-LABEL METRICS\n")
        f.write(
            f"{'Label':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} "
            f"{'ROC-AUC':<10} {'Gini':<10} {'PR-AUC':<10}\n"
        )
        f.write("-" * 90 + "\n")
        for label in labels:
            m       = results["per_label_metrics"][label]
            roc_auc = results["roc_auc_scores"][label]
            gini    = results["gini_scores"][label]
            pr_auc  = results["pr_auc_scores"][label]
            roc_str = f"{roc_auc:.3f}" if not np.isnan(roc_auc) else "N/A"
            gini_str = f"{gini:.3f}" if not np.isnan(gini) else "N/A"
            pr_str  = f"{pr_auc:.3f}"  if not np.isnan(pr_auc)  else "N/A"
            f.write(
                f"{label:<20} {m['precision']:<10.3f} {m['recall']:<10.3f} {m['f1']:<10.3f} "
                f"{roc_str:<10} {gini_str:<10} {pr_str:<10}\n"
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
