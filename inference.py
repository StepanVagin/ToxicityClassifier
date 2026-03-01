#!/usr/bin/env python3
"""
Run toxicity classification inference with a trained model.

Loads a saved model, applies the same text preprocessing, and outputs
predictions (binary labels and probabilities) for input text.

Usage:
    # Single text from command line
    python inference.py --text "You are an idiot"

    # Multiple texts from command line
    python inference.py --text "Hello world" --text "You suck"

    # Read texts from file (one per line)
    python inference.py --input comments.txt

    # Custom model directory
    python inference.py --model-dir ./models/saved --text "Test comment"

    # Output to JSON
    python inference.py --text "Bad word" --format json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessor import clean_text
from models import create_model


def load_model_and_config(model_dir: Path, model_type: str = "logistic_regression") -> tuple:
    """
    Load saved model, config, and thresholds.

    Returns:
        Tuple of (model, labels, thresholds).
    """
    model_path = model_dir / f"{model_type}.pkl"
    metadata_path = model_dir / f"{model_type}_metadata.json"
    thresholds_path = model_dir / f"{model_type}_thresholds.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first: python run_pipeline.py"
        )

    # Load config (merge base + model)
    config_path = PROJECT_ROOT / "configs" / "base_config.json"
    model_config_path = PROJECT_ROOT / "configs" / "logistic_regression.json"
    with open(config_path) as f:
        config = json.load(f)
    if model_config_path.exists():
        with open(model_config_path) as f:
            config.update(json.load(f))
    config["model_type"] = model_type

    # Load model
    model = create_model(config)
    model.load(str(model_path))

    labels = config["data"]["labels"]

    # Load thresholds (tuned per-label, or default 0.5)
    thresholds = {label: 0.5 for label in labels}
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            thresholds.update(json.load(f))
    elif metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
            if "tuned_thresholds" in meta:
                thresholds.update(meta["tuned_thresholds"])

    return model, labels, thresholds


def predict(
    model,
    texts: List[str],
    labels: List[str],
    thresholds: Dict[str, float],
    clean: bool = True,
) -> List[dict]:
    """
    Run prediction on texts. Returns list of dicts with labels, probs, predicted.
    """
    if clean:
        texts = [clean_text(t) for t in texts]

    proba = model.predict_proba(texts)

    results = []
    for i, text in enumerate(texts):
        row = {
            "text": text[:200] + ("..." if len(text) > 200 else ""),
            "probabilities": {},
            "predicted": {},
        }
        for j, label in enumerate(labels):
            p = float(proba[i, j])
            thr = thresholds[label]
            row["probabilities"][label] = round(p, 4)
            row["predicted"][label] = 1 if p >= thr else 0
        results.append(row)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run toxicity classification inference on text",
    )
    parser.add_argument(
        "--text",
        "-t",
        action="append",
        default=[],
        help="Input text(s). Can be repeated.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Input file: one text per line.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "saved"),
        help="Directory containing saved model (default: models/saved)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="logistic_regression",
        help="Model type (default: logistic_regression)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip text cleaning (assume input is already preprocessed)",
    )
    args = parser.parse_args()

    texts = list(args.text)
    if args.input:
        path = Path(args.input)
        if not path.exists():
            print(f"Error: Input file not found: {path}", file=sys.stderr)
            return 1
        with open(path, encoding="utf-8") as f:
            texts.extend(line.strip() for line in f if line.strip())

    if not texts:
        print("Error: No input text. Use --text '...' or --input <file>", file=sys.stderr)
        return 1

    model_dir = Path(args.model_dir).resolve()
    try:
        model, labels, thresholds = load_model_and_config(model_dir, args.model_type)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    results = predict(model, texts, labels, thresholds, clean=not args.no_clean)

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        for i, r in enumerate(results):
            print(f"\n--- Example {i + 1} ---")
            print(f"Text: {r['text']}")
            print("Predictions (label: prob → 0/1):")
            for label in labels:
                p = r["probabilities"][label]
                pred = r["predicted"][label]
                print(f"  {label}: {p:.4f} → {pred}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
