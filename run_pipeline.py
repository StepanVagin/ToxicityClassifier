#!/usr/bin/env python3
"""
End-to-end pipeline: download data → prepare → train model.

Usage:
    python run_pipeline.py [--config CONFIG]
    
    # Default: uses configs/base_config.json + configs/logistic_regression.json
    python run_pipeline.py
    
    # Custom config (merged with base):
    python run_pipeline.py --config configs/logistic_regression.json
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


def main():
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
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent
    config_path = Path(args.config) if args.config else None  # None = use default
    config = load_config(base_path, config_path)

    # Step 1: Ensure dataset is ready
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

    # Step 2: Load and prepare data
    print("\n" + "=" * 60)
    print("Step 2: Prepare data")
    print("=" * 60)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_prepare_data(
        csv_path=csv_path,
        config=config,
        seed=config.get("seed", 42),
    )
    train_data = (X_train, y_train)
    val_data = (X_val, y_val)

    # Step 3: Train model
    print("\n" + "=" * 60)
    print("Step 3: Train model")
    print("=" * 60)
    model = create_model(config)
    model, training_time, val_metrics = train_model(
        model, train_data, val_data, config
    )

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Val F1 macro:  {val_metrics.get('f1_macro', 0):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
