#!/usr/bin/env python3
"""
End-to-end pipeline: download data → prepare → train model → evaluate.

Usage:
    python run_pipeline.py [--config CONFIG] [--skip-download] [--no-tune]

    # Default: uses configs/base_config.json + configs/logistic_regression.json
    python run_pipeline.py

    # Custom config (merged with base):
    python run_pipeline.py --config configs/logistic_regression.json

    # Skip Kaggle download if data already exists:
    python run_pipeline.py --skip-download

    # Skip threshold tuning (use default 0.5 for all labels):
    python run_pipeline.py --no-tune
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import download_and_validate, load_and_prepare_data
from models import create_model
from training import train_model
from evaluation import tune_thresholds, evaluate_model


def load_config(base_path: Path, model_config_path: Path = None) -> dict:
    """Load and merge base config with model-specific config."""
    with open(base_path / "configs" / "base_config.json") as f:
        config = json.load(f)

    if model_config_path is None:
        model_config_path = base_path / "configs" / "logistic_regression.json"
    if model_config_path and model_config_path.exists():
        with open(model_config_path) as f:
            config.update(json.load(f))
    return config


def _resolve_outputs_dir(base_path: Path, config: dict) -> Path:  # ←
    """Resolve and create outputs directory from config."""
    raw = config.get("paths", {}).get("outputs_dir", "./outputs")
    path = Path(raw) if Path(raw).is_absolute() else base_path / raw.lstrip("./")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download data and train toxicity classification model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config JSON (merged with base_config)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download; assume data already exists",
    )
    parser.add_argument(                                            # ←
        "--no-tune",
        action="store_true",
        help="Skip threshold tuning; use default 0.5 for all labels",
    )
    args = parser.parse_args()

    base_path   = Path(__file__).resolve().parent
    config_path = Path(args.config) if args.config else None
    config      = load_config(base_path, config_path)
    model_type  = config.get("model_type", "logistic_regression")
    labels      = config["data"]["labels"]
    eval_config = config.get("evaluation", {})
    outputs_dir = _resolve_outputs_dir(base_path, config)          # ←

    # Dataset
    print("\n" + "=" * 60)
    print("Step 1: Dataset")
    print("=" * 60)
    if args.skip_download:
        processed_dir = config.get("paths", {}).get("processed_dir", "./data/processed")
        processed_path = base_path / processed_dir.replace("./", "")
        csv_path = processed_path / "train.csv"
        test_csv = processed_path / "test.csv"
        test_labels_csv = processed_path / "test_labels.csv"
        if not csv_path.exists():
            print(f"Error: {csv_path} not found. Run without --skip-download first.")
            return 1
        if not test_csv.exists() or not test_labels_csv.exists():
            print(f"Error: test.csv and test_labels.csv required in {processed_path}")
            print("Run without --skip-download to download the full dataset.")
            return 1
        print(f"Using existing data: {csv_path}")
        print(f"Test set: {test_csv}, {test_labels_csv}")
    else:
        csv_path = download_and_validate(base_path, verbose=True)
        print(f"Dataset ready: {csv_path}")

    # Prepare data
    print("\n" + "=" * 60)
    print("Step 2: Prepare data")
    print("=" * 60)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_prepare_data(
        csv_path=csv_path,
        config=config,
        seed=config.get("seed", 42),
    )
    train_data = (X_train, y_train)
    val_data   = (X_val,   y_val)
    test_data  = (X_test,  y_test)

    # Train model
    print("\n" + "=" * 60)
    print("Step 3: Train model")
    print("=" * 60)
    model = create_model(config)
    model, training_time, val_metrics = train_model(
        model, train_data, val_data, config
    )

    # Threshold tuning 
    print("\n" + "=" * 60)
    print("Step 4: Threshold tuning")
    print("=" * 60)

    if args.no_tune:                                                # ←
        tuned_thresholds = None
        print("Skipping tuning — using default threshold 0.5 for all labels.")
    else:
        optimize_f1 = eval_config.get("optimize_f1", "per_label")
        print(f"Tuning thresholds on validation set (optimize: {optimize_f1} F1)...")
        tuned_thresholds = tune_thresholds(
            model=model,
            val_data=val_data,
            labels=labels,
            threshold_range=tuple(eval_config.get("threshold_range", [0.1, 0.9])),
            threshold_step=eval_config.get("threshold_step", 0.05),
            cv_folds=eval_config.get("cv_folds", 5),
            optimize_f1=optimize_f1,
        )
        print("Tuned thresholds:")
        for label, thr in tuned_thresholds.items():                # ←
            print(f"  {label:<20} {thr:.2f}")
        
        # Re-evaluate validation set with tuned thresholds for fair comparison
        print("\nRe-evaluating validation set with tuned thresholds...")
        from evaluation.metrics import calculate_f1_scores
        import numpy as np
        X_val, y_val = val_data
        y_val_proba = model.predict_proba(X_val)
        y_val_pred = np.zeros_like(y_val_proba)
        for i, label in enumerate(labels):
            y_val_pred[:, i] = (y_val_proba[:, i] >= tuned_thresholds[label]).astype(int)
        val_f1_with_tuned = calculate_f1_scores(y_val, y_val_pred)
        val_metrics['f1_macro'] = val_f1_with_tuned['macro']
        val_metrics['f1_micro'] = val_f1_with_tuned['micro']
        val_metrics['f1_weighted'] = val_f1_with_tuned['weighted']
        print(f"  Val F1 macro (with tuned thresholds): {val_f1_with_tuned['macro']:.4f}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Step 5: Evaluation")
    print("=" * 60)
    print("Evaluating on test set...")
    test_results = evaluate_model(
        model=model,
        test_data=test_data,
        labels=labels,
        config=config,
        tuned_thresholds=tuned_thresholds,
    )

    # Summary
    metrics_file = outputs_dir / f"metrics_{model_type}.txt"       # ←
    cm_plot_file = outputs_dir / f"confusion_matrices_{model_type}.png"  # ←
    fpr_recall_plot = outputs_dir / f"fpr_vs_recall_curves_{model_type}.png"
    roc_gini_plot = outputs_dir / f"roc_auc_gini_{model_type}.png"
    pr_curves_plot = outputs_dir / f"precision_recall_curves_{model_type}.png"
    error_analysis_file = outputs_dir / f"error_analysis_{model_type}.txt"

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"  Model type:      {model_type}")
    print(f"  Training time:   {training_time:.2f}s")
    print(f"  Val  F1 macro:   {val_metrics.get('f1_macro', 0):.4f}")
    print(f"  Test F1 macro:   {test_results['f1_scores']['macro']:.4f}")
    print(f"  Test F1 micro:   {test_results['f1_scores']['micro']:.4f}")
    print(f"  Test F1 weighted:{test_results['f1_scores']['weighted']:.4f}")
    roc_macro = np.nanmean(list(test_results['roc_auc_scores'].values()))
    gini_macro = np.nanmean(list(test_results['gini_scores'].values()))
    print(f"  Test ROC-AUC macro: {roc_macro:.4f}")
    print(f"  Test Gini macro:   {gini_macro:.4f}")
    print()
    print("  Outputs:")                                             # ←
    print(f"    Metrics report      : {metrics_file.resolve()}")                  # ←
    print(f"    Confusion plots     : {cm_plot_file.resolve()}")                  # ←
    print(f"    FPR vs Recall curves: {fpr_recall_plot.resolve()}")
    print(f"    ROC-AUC & Gini plot : {roc_gini_plot.resolve()}")
    print(f"    PR curves           : {pr_curves_plot.resolve()}")
    print(f"    Error analysis      : {error_analysis_file.resolve()}")
    print()
    print("  Per-label F1 (test):")                                # ←
    for label in labels:                                           # ←
        f1 = test_results["per_label_metrics"][label]["f1"]        # ←
        thr = test_results["tuned_thresholds"][label]              # ←
        print(f"    {label:<20} F1={f1:.4f}  threshold={thr:.2f}") # ←

    return 0


if __name__ == "__main__":
    sys.exit(main())
