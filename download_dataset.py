#!/usr/bin/env python3
"""
Download and validate the Jigsaw Toxic Comment Classification dataset from Kaggle.

This script is part of the data preparation pipeline. It downloads the dataset,
extracts CSV files, and validates their structure.

Usage:
    python download_dataset.py

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

import pandas as pd

from data.downloader import download_jigsaw_dataset, check_dataset_exists, ensure_train_csv_ready

EXPECTED_TRAIN_COLUMNS = [
    "id", "comment_text",
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
]

EXPECTED_TEST_COLUMNS = ["id", "comment_text"]

EXPECTED_TEST_LABELS_COLUMNS = [
    "id", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
]


def main():
    """Download and validate the Jigsaw dataset."""
    # Paths relative to script location
    project_root = Path(__file__).resolve().parent
    raw_dir = (project_root / "data" / "raw").resolve()
    extract_dir = (project_root / "data" / "processed").resolve()
    
    print(f"\nDirectories:")
    print(f"  Raw (zips):     {raw_dir}")
    print(f"  Extracted CSV:  {extract_dir}")
    print("\n" + "-"*70)
    
    if check_dataset_exists(str(raw_dir), str(extract_dir)):
        print("Dataset already exists. Extracting CSV files from zips...")
        try:
            csv_path = ensure_train_csv_ready(str(raw_dir), str(extract_dir))
        except Exception as e:
            print(f"FAILED to extract: {e}")
            return 1
    else:
        print("Dataset not found. Downloading from Kaggle...")
        try:
            csv_path = Path(download_jigsaw_dataset(str(raw_dir), extract_to=str(extract_dir)))
            print(f"Downloaded to: {csv_path}")
        except Exception as e:
            print(f"FAILED to download: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Validate train.csv
    print("\n" + "-"*70)
    print("Validating train.csv...")
    
    try:
        train_df = pd.read_csv(csv_path, nrows=None)
        print(f"Loaded train.csv successfully")
    except Exception as e:
        print(f"FAILED to load train.csv: {e}")
        return 1

    # Check columns
    missing = [c for c in EXPECTED_TRAIN_COLUMNS if c not in train_df.columns]
    if missing:
        print(f"FAILED: Missing columns: {missing}")
        print(f"Found columns: {list(train_df.columns)}")
        return 1

    print(f"All {len(EXPECTED_TRAIN_COLUMNS)} expected columns present")
    print(f"Dataset contains {len(train_df):,} rows")
    
    # Show label distribution
    print(f"\nLabel Distribution:")
    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    for label in label_cols:
        count = train_df[label].sum()
        pct = (count / len(train_df)) * 100
        print(f"  {label:15s}: {count:6,} ({pct:5.2f}%)")
    
    # Show sample
    print(f"\nSample Comment:")
    sample_text = train_df['comment_text'].iloc[0]
    if len(sample_text) > 150:
        sample_text = sample_text[:150] + "..."
    print(f"  {sample_text}")

    # Validate test.csv if exists
    test_csv_path = extract_dir / "test.csv"
    if test_csv_path.exists():
        print("\n" + "-"*70)
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
    
    # Validate test_labels.csv if exists
    test_labels_path = extract_dir / "test_labels.csv"
    if test_labels_path.exists():
        print("\n" + "-"*70)
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

    print("\n" + "="*70)
    print("Dataset downloaded and validated successfully!")
    print("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
