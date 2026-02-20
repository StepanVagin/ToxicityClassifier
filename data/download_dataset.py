#!/usr/bin/env python3
"""
Download and validate the Jigsaw Toxic Comment Classification dataset from Kaggle.

This script is part of the data preparation pipeline. It downloads the dataset,
extracts CSV files, and validates their structure.

Usage:
    python -m data.download_dataset
    # or from project root:
    python data/download_dataset.py

Requirements:
    - pip install kaggle pandas
    - export KAGGLE_API_TOKEN="your_token"

Output:
    - data/raw/jigsaw-toxic-comment-classification-challenge.zip
    - data/processed/train.csv
    - data/processed/test.csv
    - data/processed/test_labels.csv
"""

import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .downloader import (
    download_jigsaw_dataset,
    check_dataset_exists,
    ensure_train_csv_ready,
)

EXPECTED_TRAIN_COLUMNS = [
    "id", "comment_text",
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
]

EXPECTED_TEST_COLUMNS = ["id", "comment_text"]

EXPECTED_TEST_LABELS_COLUMNS = [
    "id", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
]


def _get_paths(project_root: Optional[Path] = None) -> tuple:
    """Get raw and extract directories relative to project root."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    raw_dir = (project_root / "data" / "raw").resolve()
    extract_dir = (project_root / "data" / "processed").resolve()
    return raw_dir, extract_dir


def ensure_dataset_ready(
    project_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Download (if needed) and extract the dataset. Returns path to train.csv.

    Call this from pipelines or scripts that need the dataset ready.
    Skips download if data already exists.

    Parameters
    ----------
    project_root : str or Path, optional
        Project root directory. If None, inferred from this file's location.

    Returns
    -------
    Path
        Path to train.csv
    """
    if project_root is not None:
        project_root = Path(project_root).resolve()
    raw_dir, extract_dir = _get_paths(project_root)

    if check_dataset_exists(str(raw_dir), str(extract_dir)):
        csv_path = Path(ensure_train_csv_ready(str(raw_dir), str(extract_dir)))
    else:
        csv_path = Path(
            download_jigsaw_dataset(
                str(raw_dir), extract_to=str(extract_dir)
            )
        )
    return csv_path


def download_and_validate(
    project_root: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Path:
    """
    Download (if needed), extract, and validate the dataset.
    Same logic as main() but returns path for reuse in pipelines.

    Parameters
    ----------
    project_root : str or Path, optional
        Project root directory. If None, inferred from this file's location.
    verbose : bool
        If True, print directories, validation summary, label distribution.

    Returns
    -------
    Path
        Path to train.csv
    """
    if project_root is not None:
        project_root = Path(project_root).resolve()
    raw_dir, extract_dir = _get_paths(project_root)

    if verbose:
        print(f"\nDirectories:")
        print(f"  Raw (zips):     {raw_dir}")
        print(f"  Extracted CSV:  {extract_dir}")
        print("\n" + "-" * 70)

    csv_path = ensure_dataset_ready(project_root)

    # Validate train.csv
    train_df = pd.read_csv(csv_path, nrows=None)

    missing = [c for c in EXPECTED_TRAIN_COLUMNS if c not in train_df.columns]
    if missing:
        raise ValueError(
            f"Dataset validation failed: missing columns {missing}. "
            f"Found: {list(train_df.columns)}"
        )

    if verbose:
        print("\n" + "-" * 70)
        print("Validating train.csv...")
        print(f"Loaded train.csv successfully")
        print(f"All {len(EXPECTED_TRAIN_COLUMNS)} expected columns present")
        print(f"Dataset contains {len(train_df):,} rows")

        print(f"\nLabel Distribution:")
        label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        for label in label_cols:
            count = train_df[label].sum()
            pct = (count / len(train_df)) * 100
            print(f"  {label:15s}: {count:6,} ({pct:5.2f}%)")

        print(f"\nSample Comment:")
        sample_text = train_df["comment_text"].iloc[0]
        if len(sample_text) > 150:
            sample_text = sample_text[:150] + "..."
        print(f"  {sample_text}")

        # Validate test.csv if exists
        test_csv_path = extract_dir / "test.csv"
        if test_csv_path.exists():
            print("\n" + "-" * 70)
            print("Validating test.csv...")
            try:
                test_df = pd.read_csv(test_csv_path, nrows=5)
                missing = [c for c in EXPECTED_TEST_COLUMNS if c not in test_df.columns]
                if missing:
                    print(f"Warning: Missing columns in test.csv: {missing}")
                else:
                    print(f"test.csv validated ({len(pd.read_csv(test_csv_path)):,} rows)")
            except Exception as e:
                print(f"Warning: Could not validate test.csv: {e}")

        test_labels_path = extract_dir / "test_labels.csv"
        if test_labels_path.exists():
            print("\n" + "-" * 70)
            print("Validating test_labels.csv...")
            try:
                test_labels_df = pd.read_csv(test_labels_path, nrows=5)
                missing = [c for c in EXPECTED_TEST_LABELS_COLUMNS if c not in test_labels_df.columns]
                if missing:
                    print(f"Warning: Missing columns in test_labels.csv: {missing}")
                else:
                    print(f"test_labels.csv validated ({len(pd.read_csv(test_labels_path)):,} rows)")
            except Exception as e:
                print(f"Warning: Could not validate test_labels.csv: {e}")

        print("\n" + "=" * 70)
        print("Dataset downloaded and validated successfully!")
        print("=" * 70)

    return csv_path


def main() -> int:
    """Download and validate the Jigsaw dataset (CLI entry point)."""
    try:
        project_root = Path(__file__).resolve().parent.parent
        download_and_validate(project_root, verbose=True)
        return 0
    except Exception as e:
        print(f"FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
