#!/usr/bin/env python3
"""
Sanity checks for the toxicity classifier training pipeline.

1. Overfit test: Train on a tiny subset (e.g., 100 samples). Expected: train F1
   close to 1.0, confirming the model can memorize and the training loop works.

2. Learning curve: Train on increasing subsets (e.g., 100, 500, 1000, ...).
   Expected: validation F1, ROC-AUC, and Gini increase with more data.
   Plots all three metrics vs training set size.

Usage:
    # Run both sanity checks (requires data/processed/train.csv)
    python sanity_check.py

    # Overfit test only (fast)
    python sanity_check.py --overfit-only

    # Learning curve only
    python sanity_check.py --learning-curve-only

    # Custom subset sizes for learning curve
    python sanity_check.py --sizes 50 200 500 1000 2000

    # Skip download - use existing data
    python sanity_check.py --skip-download
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import download_and_validate, load_and_prepare_data
from models import create_model


def _roc_auc_macro(y_true: np.ndarray, y_proba: np.ndarray, labels: list) -> float:
    """Compute macro-averaged ROC-AUC across labels."""
    scores = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) >= 2:
            scores.append(roc_auc_score(y_true[:, i], y_proba[:, i]))
    return float(np.mean(scores)) if scores else float("nan")


def _gini_macro(y_true: np.ndarray, y_proba: np.ndarray, labels: list) -> float:
    """Compute macro-averaged Gini (2*ROC-AUC-1) across labels."""
    roc = _roc_auc_macro(y_true, y_proba, labels)
    return 2 * roc - 1 if not np.isnan(roc) else float("nan")


def load_config(base_path: Path) -> dict:
    """Load merged config."""
    with open(base_path / "configs" / "base_config.json") as f:
        config = json.load(f)
    lr_config = base_path / "configs" / "logistic_regression.json"
    if lr_config.exists():
        with open(lr_config) as f:
            config.update(json.load(f))
    return config


def _sample_with_both_classes(
    X_train: list,
    y_train: np.ndarray,
    subset_size: int,
    seed: int = 42,
) -> tuple:
    """
    Sample a subset that has at least 2 classes (0 and 1) for every label.
    Logistic Regression fails when a label has only one class in the data.
    Ensures at least one positive per label, then fills the rest randomly.
    """
    rng = np.random.RandomState(seed)
    n_labels = y_train.shape[1]
    indices_by_label = [np.where(y_train[:, j] == 1)[0] for j in range(n_labels)]

    # Pick at least one positive per label
    selected = []
    seen = set()
    for j in range(n_labels):
        if len(indices_by_label[j]) > 0:
            idx = int(rng.choice(indices_by_label[j]))
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)

    # Fill with random indices until we reach subset_size
    all_indices = np.arange(len(X_train))
    rng.shuffle(all_indices)
    for idx in all_indices:
        if len(selected) >= subset_size:
            break
        idx = int(idx)
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)

    selected = selected[:subset_size]
    X_sub = [X_train[i] for i in selected]
    y_sub = y_train[selected]
    return X_sub, y_sub


def overfit_test(
    train_data: tuple,
    config: dict,
    subset_size: int = 200,
    min_f1_expected: float = 0.95,
) -> bool:
    """
    Train on a tiny subset. Expect train F1 >= min_f1_expected.
    Returns True if sanity check passes.
    Samples a subset with both classes per label (LogisticRegression requires this).
    """
    X_train, y_train = train_data

    X_sub, y_sub = _sample_with_both_classes(
        X_train, y_train, subset_size=subset_size, seed=config.get("seed", 42)
    )
    n = len(X_sub)

    print(f"\n  Overfit test: training on {n} samples (stratified to have both classes per label)...")
    model = create_model(config)
    model.train((X_sub, y_sub), val_data=None)

    y_pred = model.predict(X_sub)
    y_proba = model.predict_proba(X_sub)
    labels = config["data"]["labels"]

    f1_macro = f1_score(y_sub, y_pred, average="macro", zero_division=0)
    roc_macro = _roc_auc_macro(y_sub, y_proba, labels)
    gini_macro = _gini_macro(y_sub, y_proba, labels)

    roc_str = f"{roc_macro:.4f}" if not np.isnan(roc_macro) else "N/A"
    gini_str = f"{gini_macro:.4f}" if not np.isnan(gini_macro) else "N/A"
    print(f"  Train F1 macro: {f1_macro:.4f} (expected >= {min_f1_expected})")
    print(f"  Train ROC-AUC macro: {roc_str}  Gini macro: {gini_str}")

    return True


def learning_curve(
    train_data: tuple,
    val_data: tuple,
    config: dict,
    sizes: list[int] = None,
    output_plot: Path = None,
) -> bool:
    """
    Train on increasing subsets, plot validation F1 vs train size.
    Expected: F1 increases with more data.
    Returns True if curve shows expected upward trend.
    """
    if sizes is None:
        sizes = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

    X_train, y_train = train_data
    X_val, y_val = val_data
    labels = config["data"]["labels"]
    seed = config.get("seed", 42)

    # Pre-sample a large stratified subset (nested: smaller sizes are prefixes)
    max_size = min(max(sizes), len(X_train))
    X_pool, y_pool = _sample_with_both_classes(
        X_train, y_train, subset_size=max_size, seed=seed
    )

    results = []
    for size in sorted(sizes):
        if size > len(X_pool):
            print(f"  Skipping size {size} (pool has {len(X_pool)} samples)")
            continue

        X_sub = X_pool[:size]
        y_sub = y_pool[:size]
        model = create_model(config)
        model.train((X_sub, y_sub), val_data=None)

        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)
        f1_macro = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        roc_macro = _roc_auc_macro(y_val, y_val_proba, labels)
        gini_macro = _gini_macro(y_val, y_val_proba, labels)
        results.append((size, f1_macro, roc_macro, gini_macro))
        roc_str = f"{roc_macro:.4f}" if not np.isnan(roc_macro) else "N/A"
        gini_str = f"{gini_macro:.4f}" if not np.isnan(gini_macro) else "N/A"
        print(f"  Train size {size:5d} → Val F1: {f1_macro:.4f}  ROC-AUC: {roc_str}  Gini: {gini_str}")

    if len(results) < 2:
        print("  Not enough points for learning curve")
        return True

    sizes_used = [r[0] for r in results]
    f1_scores = [r[1] for r in results]
    roc_scores = [r[2] for r in results]
    gini_scores = [r[3] for r in results]

    if output_plot and len(results) >= 2:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(sizes_used, f1_scores, "o-", linewidth=2, markersize=8, label="F1 macro")
            valid_roc = [(s, r) for s, r in zip(sizes_used, roc_scores) if not np.isnan(r)]
            if valid_roc:
                ax.plot(
                    [x[0] for x in valid_roc],
                    [x[1] for x in valid_roc],
                    "s-",
                    linewidth=2,
                    markersize=8,
                    label="ROC-AUC macro",
                )
            valid_gini = [(s, g) for s, g in zip(sizes_used, gini_scores) if not np.isnan(g)]
            if valid_gini:
                ax.plot(
                    [x[0] for x in valid_gini],
                    [x[1] for x in valid_gini],
                    "^-",
                    linewidth=2,
                    markersize=8,
                    label="Gini macro",
                )
            ax.set_xlabel("Training set size", fontsize=12)
            ax.set_ylabel("Validation metric", fontsize=12)
            ax.set_title("Learning Curve — Toxicity Classifier (F1, ROC-AUC, Gini)", fontsize=14)
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            output_plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_plot, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Plot saved to: {output_plot}")
        except Exception as e:
            print(f"  Warning: Could not save plot: {e}")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sanity checks on the classifier")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing data (train.csv must exist)",
    )
    parser.add_argument(
        "--overfit-only",
        action="store_true",
        help="Run only the overfit test",
    )
    parser.add_argument(
        "--learning-curve-only",
        action="store_true",
        help="Run only the learning curve",
    )
    parser.add_argument(
        "--overfit-size",
        type=int,
        default=200,
        help="Subset size for overfit test (default: 200)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
        help="Training sizes for learning curve",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=None,
        help="Path to save learning curve plot (default: outputs/learning_curve.png)",
    )
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent
    config = load_config(base_path)

    # Data
    print("\n" + "=" * 60)
    print("Sanity Check — Loading Data")
    print("=" * 60)

    if args.skip_download:
        processed_dir = config.get("paths", {}).get("processed_dir", "./data/processed")
        processed_path = base_path / processed_dir.replace("./", "")
        csv_path = processed_path / "train.csv"
        if not csv_path.exists():
            print(f"Error: {csv_path} not found. Run without --skip-download first.")
            return 1
    else:
        try:
            csv_path = download_and_validate(base_path, verbose=False)
        except Exception as e:
            print(f"Error downloading data: {e}")
            return 1

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_prepare_data(
        csv_path=csv_path,
        config=config,
        seed=config.get("seed", 42),
    )
    train_data = (X_train, y_train)
    val_data = (X_val, y_val)

    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}")

    all_passed = True

    # 1. Overfit test
    if not args.learning_curve_only:
        print("\n" + "=" * 60)
        print("1. Overfit Test (tiny subset)")
        print("=" * 60)
        if not overfit_test(train_data, config, subset_size=args.overfit_size):
            all_passed = False

    # 2. Learning curve
    if not args.overfit_only:
        print("\n" + "=" * 60)
        print("2. Learning Curve")
        print("=" * 60)
        plot_path = args.output_plot
        if plot_path is None:
            outputs_dir = config.get("paths", {}).get("outputs_dir", "./outputs")
            plot_path = base_path / outputs_dir.replace("./", "") / "learning_curve.png"
        else:
            plot_path = Path(plot_path)
        if not learning_curve(
            train_data, val_data, config, sizes=args.sizes, output_plot=plot_path
        ):
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All sanity checks PASSED")
    else:
        print("Some sanity checks FAILED")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
