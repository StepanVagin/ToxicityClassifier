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
        csv_path = base_path / processed_dir.replace("./", "") / "train.csv"
        if not csv_path.exists():
            print(f"Error: {csv_path} not found. Run without --skip-download first.")
            return 1
        print(f"Using existing data: {csv_path}")
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
        print("Tuning thresholds on validation set...")
        tuned_thresholds = tune_thresholds(
            model=model,
            val_data=val_data,
            labels=labels,
            threshold_range=tuple(eval_config.get("threshold_range", [0.1, 0.9])),
            threshold_step=eval_config.get("threshold_step", 0.05),
            cv_folds=eval_config.get("cv_folds", 5),
        )
        print("Tuned thresholds:")
        for label, thr in tuned_thresholds.items():                # ←
            print(f"  {label:<20} {thr:.2f}")

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

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"  Model type:      {model_type}")
    print(f"  Training time:   {training_time:.2f}s")
    print(f"  Val  F1 macro:   {val_metrics.get('f1_macro', 0):.4f}")
    print(f"  Test F1 macro:   {test_results['f1_scores']['macro']:.4f}")
    print(f"  Test F1 micro:   {test_results['f1_scores']['micro']:.4f}")
    print(f"  Test F1 weighted:{test_results['f1_scores']['weighted']:.4f}")
    print()
    print("  Outputs:")                                             # ←
    print(f"    Metrics report : {metrics_file}")                  # ←
    print(f"    Confusion plots: {cm_plot_file}")                  # ←
    print()
    print("  Per-label F1 (test):")                                # ←
    for label in labels:                                           # ←
        f1 = test_results["per_label_metrics"][label]["f1"]        # ←
        thr = test_results["tuned_thresholds"][label]              # ←
        print(f"    {label:<20} F1={f1:.4f}  threshold={thr:.2f}") # ←

    return 0


if __name__ == "__main__":
    sys.exit(main())
