#!/usr/bin/env python3
"""
Extra matplotlib figures for training dynamics and test-set diagnostics.

Used by run_pipeline (training curves) and metrics.evaluate_model (evaluation dashboards).
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def plot_training_history(
    history: Optional[Dict],
    output_path: Path,
    model_type: str,
) -> None:
    """
    Multi-panel figure: train vs validation loss, validation F1 (macro/micro/weighted),
    and learning rate at end of each epoch.
    """
    if not history or not history.get("epoch"):
        return

    epochs = history["epoch"]
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_fm = history.get("val_f1_macro", [])
    val_fi = history.get("val_f1_micro", [])
    val_fw = history.get("val_f1_weighted", [])
    lrs = history.get("learning_rate", [])
    batch_steps = history.get("batch_step", [])
    batch_losses = history.get("batch_train_loss", [])
    epoch_boundaries = history.get("epoch_boundaries", [])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)

    ax = axes[0, 0]
    use_batch_loss = bool(batch_steps and batch_losses and len(batch_steps) == len(batch_losses))
    if use_batch_loss:
        ax.plot(batch_steps, batch_losses, color="#2c7fb8", linewidth=0.8, alpha=0.6, label="Train loss (batch)")
        # Overlay epoch-average train loss at epoch boundaries
        if train_loss and epoch_boundaries and len(train_loss) == len(epoch_boundaries):
            ax.plot(epoch_boundaries, train_loss, "o", color="#08519c", markersize=7,
                    zorder=5, label="Train loss (epoch avg)")
        # Vertical lines at epoch boundaries
        for i, xb in enumerate(epoch_boundaries):
            ax.axvline(xb, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
                       label="Epoch end" if i == 0 else None)
        # Val loss at epoch boundaries
        if val_loss and epoch_boundaries and len(val_loss) == len(epoch_boundaries):
            ax.plot(epoch_boundaries, val_loss, "s--", color="#e34a33", linewidth=2,
                    markersize=6, label="Val loss (epoch)")
        ax.set_xlabel("Batch step (global)")
        ax.set_ylabel("BCE loss")
    else:
        ax.plot(epochs, train_loss, "o-", color="#2c7fb8", linewidth=2, markersize=6, label="Train loss")
        if val_loss and len(val_loss) == len(epochs):
            ax.plot(epochs, val_loss, "s--", color="#e34a33", linewidth=2, markersize=6, label="Val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE loss (avg per batch)")
    ax.set_title("Loss")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if val_fm and len(val_fm) == len(epochs):
        ax.plot(epochs, val_fm, "o-", label="F1 macro", color="#2c7fb8", linewidth=2)
    if val_fi and len(val_fi) == len(epochs):
        ax.plot(epochs, val_fi, "s--", label="F1 micro", color="#7fbc41", linewidth=2)
    if val_fw and len(val_fw) == len(epochs):
        ax.plot(epochs, val_fw, "^:", label="F1 weighted", color="#f18f01", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 (val, threshold 0.5)")
    ax.set_title("Validation F1")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if lrs and len(lrs) == len(epochs):
        ax.plot(epochs, lrs, "D-", color="#6a51a3", linewidth=2, markersize=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("LR schedule (end-of-epoch)")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    ax = axes[1, 1]
    gap = None
    x_gap = epochs
    if train_loss and val_loss and len(train_loss) == len(val_loss) == len(epochs):
        t = np.asarray(train_loss, dtype=float)
        v = np.asarray(val_loss, dtype=float)
        if np.all(np.isfinite(t) & np.isfinite(v)):
            gap = t - v
            if use_batch_loss and epoch_boundaries and len(epoch_boundaries) == len(epochs):
                x_gap = epoch_boundaries
    if gap is not None:
        width = (x_gap[-1] - x_gap[0]) / max(len(x_gap) * 1.5, 1) if len(x_gap) > 1 else 0.6
        ax.bar(x_gap, gap, width=width, color="#35978f", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Batch step (global)" if use_batch_loss else "Epoch")
        ax.set_ylabel("Train loss − Val loss")
        ax.set_title("Generalization gap (↓ overfitting if stable)")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.set_visible(False)

    fig.suptitle(f"Training curves — {model_type}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_path / f"training_curves_{model_type}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to: {out}")


def plot_metrics_heatmap(
    results: Dict,
    labels: List[str],
    output_path: Path,
    model_type: str,
) -> None:
    """Heatmap: labels × (Precision, Recall, F1, ROC-AUC, PR-AUC, Brier); all in [0, 1]."""
    cols = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier"]
    mat = np.zeros((len(labels), len(cols)))
    for i, lab in enumerate(labels):
        m = results["per_label_metrics"][lab]
        mat[i, 0] = m["precision"]
        mat[i, 1] = m["recall"]
        mat[i, 2] = m["f1"]
        mat[i, 3] = results["roc_auc_scores"][lab]
        mat[i, 4] = results["pr_auc_scores"][lab]
        mat[i, 5] = results["brier_scores"]["per_label"][lab]

    mat_plot = np.nan_to_num(mat, nan=0.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mat_plot, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(cols)):
            v = mat[i, j]
            txt = "N/A" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Score")
    ax.set_title(f"Per-label metrics heatmap — {model_type}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = output_path / f"metrics_heatmap_{model_type}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Metrics heatmap saved to: {out}")


def plot_roc_curves_per_label(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    labels: List[str],
    output_path: Path,
    model_type: str,
) -> None:
    """Full ROC curves (one subplot per toxicity label)."""
    n = len(labels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    flat = axes.flatten()

    for idx, label in enumerate(labels):
        ax = flat[idx]
        y = y_true[:, idx]
        s = y_pred_proba[:, idx]
        if len(np.unique(y)) < 2:
            ax.text(0.5, 0.5, "single class", ha="center", va="center")
            ax.set_title(label)
            continue
        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#2c7fb8", lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(label, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(len(labels), len(flat)):
        flat[j].set_visible(False)

    fig.suptitle(f"ROC curves (test) — {model_type}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = output_path / f"roc_curves_{model_type}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curves saved to: {out}")


def plot_calibration_reliability(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    labels: List[str],
    output_path: Path,
    model_type: str,
    n_bins: int = 10,
) -> None:
    """Reliability diagrams: predicted probability vs empirical positive rate."""
    n = len(labels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    flat = axes.flatten()

    for idx, label in enumerate(labels):
        ax = flat[idx]
        y = y_true[:, idx].astype(float)
        p = y_pred_proba[:, idx]
        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(p, bins) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
        mean_pred = []
        frac_pos = []
        centers = []
        for b in range(n_bins):
            m = bin_ids == b
            if not np.any(m):
                continue
            mean_pred.append(float(np.mean(p[m])))
            frac_pos.append(float(np.mean(y[m])))
            centers.append((bins[b] + bins[b + 1]) / 2)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
        if centers:
            ax.plot(mean_pred, frac_pos, "o-", color="#e34a33", lw=2, markersize=6)
        ax.set_xlabel("Mean predicted prob.")
        ax.set_ylabel("Fraction positives")
        ax.set_title(label, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    for j in range(len(labels), len(flat)):
        flat[j].set_visible(False)

    fig.suptitle(
        f"Calibration (reliability, {n_bins} bins) — {model_type}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out = output_path / f"calibration_reliability_{model_type}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Calibration plot saved to: {out}")


def plot_test_summary_dashboard(
    results: Dict,
    labels: List[str],
    output_path: Path,
    model_type: str,
) -> None:
    """Single figure: global F1 variants, multilabel KPIs, mean AUCs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    names = ["F1 macro", "F1 micro", "F1 weighted"]
    vals = [
        results["f1_scores"]["macro"],
        results["f1_scores"]["micro"],
        results["f1_scores"]["weighted"],
    ]
    colors = ["#2c7fb8", "#7fbc41", "#f18f01"]
    ax.barh(names, vals, color=colors, alpha=0.9, edgecolor="black")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    ax.set_title("Overall F1 (test, tuned thresholds)", fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1]
    roc_vals = [v for v in results["roc_auc_scores"].values() if not np.isnan(v)]
    pr_vals = [v for v in results["pr_auc_scores"].values() if not np.isnan(v)]
    roc_m = float(np.mean(roc_vals)) if roc_vals else float("nan")
    pr_m = float(np.mean(pr_vals)) if pr_vals else float("nan")
    lrm = results["label_ranking_metrics"]
    lrap = lrm.get("label_ranking_avg_precision", float("nan"))

    kpi_names = [
        "Subset acc.",
        "Jaccard (macro)",
        "1 − Hamming",
        "Mean ROC-AUC",
        "Mean PR-AUC",
    ]
    kpi_vals = [
        results["subset_accuracy"],
        results["jaccard_scores"]["macro"],
        1.0 - results["hamming_loss"],
        roc_m,
        pr_m,
    ]
    if not np.isnan(lrap):
        kpi_names.append("LRAP")
        kpi_vals.append(lrap)

    x = np.arange(len(kpi_names))
    heights = [0.0 if (isinstance(v, float) and np.isnan(v)) else float(v) for v in kpi_vals]
    ax.bar(x, heights, color="#35978f", alpha=0.9, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(kpi_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Multilabel summary (test)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Test dashboard — {model_type}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_path / f"test_summary_dashboard_{model_type}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Test summary dashboard saved to: {out}")


def plot_mcc_kappa_per_label(
    results: Dict,
    labels: List[str],
    output_path: Path,
    model_type: str,
) -> None:
    """Bar charts for Matthews correlation and Cohen's κ (can be negative; N/A if undefined)."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(labels))
    mcc_vals = [results["matthews_corrcoef_per_label"][l] for l in labels]
    kap_vals = [results["cohen_kappa_per_label"][l] for l in labels]

    h0 = [np.nan_to_num(v, nan=0.0) for v in mcc_vals]
    b0 = ax0.bar(x, h0, color="#2c7fb8", edgecolor="black", linewidth=0.5)
    ax0.axhline(0, color="gray", linewidth=0.8)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=25, ha="right")
    ax0.set_ylabel("MCC")
    ax0.set_title("Matthews correlation", fontweight="bold")
    ax0.set_ylim(-1.05, 1.05)
    ax0.grid(True, axis="y", alpha=0.3)
    for i, (bar, v) in enumerate(zip(b0, mcc_vals)):
        if np.isnan(v):
            ax0.text(bar.get_x() + bar.get_width() / 2, 0.02, "N/A", ha="center", fontsize=7)

    h1 = [np.nan_to_num(v, nan=0.0) for v in kap_vals]
    b1 = ax1.bar(x, h1, color="#f18f01", edgecolor="black", linewidth=0.5)
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylabel("κ")
    ax1.set_title("Cohen's kappa", fontweight="bold")
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, axis="y", alpha=0.3)
    for i, (bar, v) in enumerate(zip(b1, kap_vals)):
        if np.isnan(v):
            ax1.text(bar.get_x() + bar.get_width() / 2, 0.02, "N/A", ha="center", fontsize=7)

    fig.suptitle(f"Agreement metrics (test) — {model_type}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_path / f"mcc_kappa_per_label_{model_type}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"MCC / kappa chart saved to: {out}")


def plot_per_label_grouped_scores(
    results: Dict,
    labels: List[str],
    output_path: Path,
    model_type: str,
) -> None:
    """Grouped bars: Precision, Recall, F1 per label."""
    x = np.arange(len(labels))
    w = 0.25
    prec = [results["per_label_metrics"][l]["precision"] for l in labels]
    rec = [results["per_label_metrics"][l]["recall"] for l in labels]
    f1 = [results["per_label_metrics"][l]["f1"] for l in labels]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, prec, w, label="Precision", color="#2c7fb8", edgecolor="black", linewidth=0.5)
    ax.bar(x, rec, w, label="Recall", color="#7fbc41", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, f1, w, label="F1", color="#f18f01", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.set_title(f"Precision / Recall / F1 by label (test) — {model_type}", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_path / f"per_label_precision_recall_f1_{model_type}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-label P/R/F1 chart saved to: {out}")


def plot_label_cooccurrence_correlation(
    y_true: np.ndarray,
    labels: List[str],
    output_path: Path,
    model_type: str,
) -> None:
    """Pearson correlation of label indicators (test set structure)."""
    C = np.corrcoef(y_true.astype(float).T)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Label correlation (test set) — {model_type}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = output_path / f"label_correlation_{model_type}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Label correlation heatmap saved to: {out}")


def plot_predicted_probability_distributions(
    y_pred_proba: np.ndarray,
    labels: List[str],
    output_path: Path,
    model_type: str,
) -> None:
    """Histogram of model scores per label (all test samples)."""
    n = len(labels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.8 * nrows), squeeze=False)
    flat = axes.flatten()
    for idx, label in enumerate(labels):
        ax = flat[idx]
        ax.hist(y_pred_proba[:, idx], bins=40, color="#6a51a3", alpha=0.85, edgecolor="white")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")
        ax.set_title(label, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
    for j in range(len(labels), len(flat)):
        flat[j].set_visible(False)
    fig.suptitle(f"Score distributions (test) — {model_type}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = output_path / f"probability_histograms_{model_type}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Probability histograms saved to: {out}")


def plot_labels_per_sample_distribution(
    y_true: np.ndarray,
    output_path: Path,
    model_type: str,
) -> None:
    """How many toxicity labels are active per comment (test set)."""
    counts = y_true.sum(axis=1).astype(int)
    max_k = int(counts.max()) + 1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(counts, bins=np.arange(-0.5, max_k + 0.5, 1), color="#35978f", edgecolor="black")
    ax.set_xlabel("Number of positive labels per sample")
    ax.set_ylabel("Count")
    ax.set_title(f"Multilabel count distribution (test, gold labels) — {model_type}", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_path / f"labels_per_sample_hist_{model_type}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Labels-per-sample histogram saved to: {out}")
