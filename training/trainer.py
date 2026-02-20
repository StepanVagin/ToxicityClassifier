#!/usr/bin/env python3
"""
Training module for toxicity classification models.

Trains any model implementing the ModelABC interface.
Orchestrates training, evaluation, and checkpoint saving.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score


def _resolve_models_dir(save_dir: str) -> Path:
    """Resolve models_dir relative to project root if it's a relative path."""
    if os.path.isabs(save_dir):
        return Path(save_dir)
    project_root = Path(__file__).resolve().parent.parent
    return (project_root / save_dir).resolve()


def train_model(
    model,
    train_data: Tuple[List[str], np.ndarray],
    val_data: Optional[Tuple[List[str], np.ndarray]],
    config: Dict[str, Any],
) -> Tuple[Any, float, Dict[str, Any]]:
    """
    Main training orchestrator.

    Sets random seed, measures training time, trains the model,
    evaluates on validation, saves checkpoint and metadata.

    Parameters
    ----------
    model : ModelABC
        Model instance with train(), predict_proba(), save(), get_model_size().
    train_data : (X_train, y_train)
        X_train: List[str] of cleaned texts
        y_train: np.ndarray of shape (n_samples, 6) binary labels
    val_data : (X_val, y_val) or None
        Same format as train_data. Used for evaluation if provided.
    config : dict
        Merged config with seed, data.labels, paths.models_dir, model_type.

    Returns
    -------
    Tuple[model, training_time_seconds, val_metrics]
    """
    seed = config.get("seed", 42)
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass

    start_time = time.perf_counter()
    model.train(train_data, val_data)
    training_time = time.perf_counter() - start_time

    print(f"Training completed in {training_time:.2f}s")

    val_metrics = {}
    if val_data is not None:
        labels = config.get("data", {}).get("labels", [])
        val_metrics = evaluate_on_validation(model, val_data, labels)
        print("\nValidation metrics:")
        for k, v in val_metrics.items():
            if isinstance(v, dict):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v:.4f}")

    save_dir = config.get("paths", {}).get("models_dir", "./models/saved")
    save_dir = str(_resolve_models_dir(save_dir))
    model_type = config.get("model_type", "logistic_regression")

    save_checkpoint(
        model=model,
        save_dir=save_dir,
        model_type=model_type,
        metrics=val_metrics,
        training_time=training_time,
    )

    return model, training_time, val_metrics


def evaluate_on_validation(
    model,
    val_data: Tuple[List[str], np.ndarray],
    labels: List[str],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute F1 metrics on validation data.

    Parameters
    ----------
    model : ModelABC
        Model with predict_proba() method.
    val_data : (X_val, y_val)
        X_val: List[str], y_val: np.ndarray (n_samples, n_labels)
    labels : List[str]
        Label names (e.g. ["toxic", "severe_toxic", ...])
    threshold : float
        Probability threshold for binary predictions (default 0.5).

    Returns
    -------
    dict
        Keys: f1_macro, f1_micro, f1_weighted, f1_per_label
    """
    X_val, y_true = val_data
    y_proba = model.predict_proba(X_val)
    y_pred = (y_proba >= threshold).astype(np.int32)

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    f1_per_label = {}
    for i, label in enumerate(labels):
        f1_per_label[label] = float(
            f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        )

    return {
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "f1_weighted": float(f1_weighted),
        "f1_per_label": f1_per_label,
    }


def save_checkpoint(
    model,
    save_dir: str,
    model_type: str,
    metrics: Dict[str, Any],
    training_time: float,
) -> None:
    """
    Save model and metadata to disk.

    Parameters
    ----------
    model : ModelABC
        Model with save() and get_model_size().
    save_dir : str
        Directory to save the model and metadata.
    model_type : str
        Identifier for the model (e.g. "logistic_regression").
    metrics : dict
        Validation metrics (e.g. from evaluate_on_validation).
    training_time : float
        Training time in seconds.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_path = save_path / f"{model_type}.pkl"
    model.save(str(model_path))
    print(f"Model saved to: {model_path}")

    model_size_mb = model.get_model_size()

    metadata = {
        "model_type": model_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "training_time_seconds": round(training_time, 2),
        "model_size_mb": model_size_mb,
        "val_metrics": metrics,
    }

    metadata_path = save_path / f"{model_type}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")
