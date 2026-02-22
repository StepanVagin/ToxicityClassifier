#!/usr/bin/env python3
"""
Data loading and splitting utilities for Jigsaw Toxic Comment Classification.

This module handles:
    - Train/validation/test splitting with stratification
    - Saving and loading split indices for reproducibility
    - Preparing data for Logistic Regression models
    - Integration with preprocessor for text cleaning

Usage:
    from data.data_loader import split_data, prepare_data_for_logistic_regression
    
    # Split data
    train_df, val_df, test_df = split_data(df, config, seed=42)
    
    # Prepare for model training
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        prepare_data_for_logistic_regression(train_df, val_df, test_df, config)
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.preprocessor import clean_text_batch, encode_labels, get_label_statistics


def split_data(
    df: pd.DataFrame,
    config: Dict,
    seed: int = 42,
    stratify_column: str = "toxic"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets with stratification.
    
    Performs stratified splitting to maintain label distribution across splits.
    Uses two-step splitting: first train/temp split, then temp into val/test.
    
    Args:
        df: Full DataFrame with text and labels.
        config: Configuration dict containing 'data' section with:
                - train_ratio: Proportion for training (e.g., 0.8)
                - val_ratio: Proportion for validation (e.g., 0.1)
                - test_ratio: Proportion for test (e.g., 0.1)
        seed: Random seed for reproducibility.
        stratify_column: Column name to use for stratification (default: 'toxic').
                        This ensures balanced label distribution across splits.
    
    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames.
    
    Raises:
        KeyError: If required config keys or stratify_column not found.
        ValueError: If ratios don't sum to 1.0 or dataset is too small.
    
    Example:
        >>> config = {
        ...     'data': {
        ...         'train_ratio': 0.8,
        ...         'val_ratio': 0.1,
        ...         'test_ratio': 0.1
        ...     }
        ... }
        >>> train_df, val_df, test_df = split_data(df, config, seed=42)
        >>> print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    """
    # Validate config
    if 'data' not in config:
        raise KeyError("Config must contain 'data' section")
    
    data_config = config['data']
    train_ratio = data_config.get('train_ratio', 0.8)
    val_ratio = data_config.get('val_ratio', 0.1)
    test_ratio = data_config.get('test_ratio', 0.1)
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Train/val/test ratios must sum to 1.0, got {total_ratio}\n"
            f"train_ratio={train_ratio}, val_ratio={val_ratio}, test_ratio={test_ratio}"
        )
    
    # Validate stratify column exists
    if stratify_column not in df.columns:
        raise KeyError(
            f"Stratify column '{stratify_column}' not found in DataFrame.\n"
            f"Available columns: {list(df.columns)}"
        )
    
    # Check minimum dataset size
    min_samples = 100  # Arbitrary minimum for meaningful splits
    if len(df) < min_samples:
        raise ValueError(
            f"Dataset too small for splitting: {len(df)} samples.\n"
            f"Minimum recommended: {min_samples} samples."
        )
    
    print(f"Splitting {len(df):,} samples into train/val/test...")
    print(f"  Ratios: {train_ratio:.1%} / {val_ratio:.1%} / {test_ratio:.1%}")
    print(f"  Stratifying by: '{stratify_column}'")
    
    # First split: train vs (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df[stratify_column]
    )
    
    # Second split: val vs test
    # Adjust test_size to get correct proportions
    val_size_adjusted = val_ratio / temp_ratio
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size_adjusted),
        random_state=seed,
        stratify=temp_df[stratify_column]
    )
    
    # Print split statistics
    print(f"\n Split complete:")
    print(f"  Train: {len(train_df):,} samples ({len(train_df)/len(df):.1%})")
    print(f"  Val:   {len(val_df):,} samples ({len(val_df)/len(df):.1%})")
    print(f"  Test:  {len(test_df):,} samples ({len(test_df)/len(df):.1%})")
    
    # Show stratification balance
    if stratify_column in df.columns:
        train_pos = train_df[stratify_column].mean()
        val_pos = val_df[stratify_column].mean()
        test_pos = test_df[stratify_column].mean()
        print(f"\n  '{stratify_column}' positive rate:")
        print(f"    Train: {train_pos:.2%}")
        print(f"    Val:   {val_pos:.2%}")
        print(f"    Test:  {test_pos:.2%}")
    
    return train_df, val_df, test_df


def split_data_train_val(
    df: pd.DataFrame,
    config: Dict,
    seed: int = 42,
    stratify_column: str = "toxic"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and validation sets only (no test split).
    Used when test set comes from Kaggle test.csv + test_labels.csv.

    Args:
        df: Full DataFrame with text and labels.
        config: Configuration dict with train_ratio, val_ratio (must sum to 1.0).
        seed: Random seed for reproducibility.
        stratify_column: Column for stratification.

    Returns:
        Tuple of (train_df, val_df).
    """
    if 'data' not in config:
        raise KeyError("Config must contain 'data' section")

    data_config = config['data']
    train_ratio = data_config.get('train_ratio', 0.9)
    val_ratio = data_config.get('val_ratio', 0.1)

    if not np.isclose(train_ratio + val_ratio, 1.0):
        raise ValueError(
            f"train_ratio + val_ratio must sum to 1.0, got {train_ratio + val_ratio}"
        )

    if stratify_column not in df.columns:
        raise KeyError(f"Stratify column '{stratify_column}' not found")

    if len(df) < 100:
        raise ValueError(f"Dataset too small: {len(df)} samples")

    print(f"Splitting {len(df):,} samples into train/val (no test split)...")
    print(f"  Ratios: {train_ratio:.1%} / {val_ratio:.1%}")
    print(f"  Stratifying by: '{stratify_column}'")

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        stratify=df[stratify_column]
    )

    print(f"\n  Train: {len(train_df):,} samples ({len(train_df)/len(df):.1%})")
    print(f"  Val:   {len(val_df):,} samples ({len(val_df)/len(df):.1%})")
    print(f"\n  '{stratify_column}' positive rate: Train {train_df[stratify_column].mean():.2%}, Val {val_df[stratify_column].mean():.2%}")

    return train_df, val_df


def load_kaggle_test_data(
    processed_dir: Union[str, Path],
    config: Dict,
    text_column: str = "comment_text",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load Kaggle test set from test.csv + test_labels.csv.
    Joins on id and filters out rows with -1 (not used for scoring).

    Args:
        processed_dir: Directory containing test.csv and test_labels.csv.
        config: Configuration dict with 'data.labels' list.
        text_column: Name of text column in test.csv.
        verbose: If True, prints statistics.

    Returns:
        DataFrame with comment_text and label columns, ready for encoding.
        Only includes rows where all labels are 0 or 1 (excludes -1 rows).
    """
    processed_dir = Path(processed_dir)
    test_csv_path = processed_dir / "test.csv"
    test_labels_path = processed_dir / "test_labels.csv"

    if not test_csv_path.exists():
        raise FileNotFoundError(
            f"test.csv not found at {test_csv_path}. "
            "Run download first (without --skip-download)."
        )
    if not test_labels_path.exists():
        raise FileNotFoundError(
            f"test_labels.csv not found at {test_labels_path}. "
            "Run download first (without --skip-download)."
        )

    label_columns = config.get('data', {}).get('labels', [])
    if not label_columns:
        raise KeyError("Config must contain data.labels")

    test_df = pd.read_csv(test_csv_path)
    test_labels_df = pd.read_csv(test_labels_path)

    # Filter out rows where any label is -1 (not used for Kaggle scoring)
    mask = (test_labels_df[label_columns] >= 0).all(axis=1)
    test_labels_filtered = test_labels_df[mask].copy()

    # Join test.csv (comment_text) with test_labels on id
    test_merged = test_df.merge(
        test_labels_filtered,
        on="id",
        how="inner"
    )

    if verbose:
        n_total = len(test_labels_df)
        n_scored = len(test_merged)
        n_excluded = n_total - n_scored
        print(f"\nKaggle test set:")
        print(f"  Total rows in test_labels.csv: {n_total:,}")
        print(f"  Rows with -1 (excluded): {n_excluded:,}")
        print(f"  Rows used for scoring: {n_scored:,}")

    return test_merged


def save_split_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    save_path: Union[str, Path]
) -> None:
    """
    Save train/val/test split indices to disk for reproducibility.
    
    Saves indices as a pickle file containing a dictionary with
    'train', 'val', and 'test' keys.
    
    Args:
        train_idx: Array of training indices.
        val_idx: Array of validation indices.
        test_idx: Array of test indices.
        save_path: Path to save the pickle file.
    
    Raises:
        IOError: If unable to write to save_path.
    
    Example:
        >>> save_split_indices(
        ...     train_df.index.values,
        ...     val_df.index.values,
        ...     test_df.index.values,
        ...     "./data/processed/split_indices.pkl"
        ... )
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    split_dict = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx,
        'metadata': {
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
            'total_size': len(train_idx) + len(val_idx) + len(test_idx)
        }
    }
    
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(split_dict, f)
        print(f"Saved split indices to: {save_path}")
    except Exception as e:
        raise IOError(f"Failed to save split indices: {e}") from e


def load_split_indices(
    load_path: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load previously saved train/val/test split indices.
    
    Args:
        load_path: Path to the pickle file containing split indices.
    
    Returns:
        Tuple of (train_idx, val_idx, test_idx) as numpy arrays.
    
    Raises:
        FileNotFoundError: If load_path doesn't exist.
        KeyError: If pickle file doesn't contain expected keys.
    
    Example:
        >>> train_idx, val_idx, test_idx = load_split_indices(
        ...     "./data/processed/split_indices.pkl"
        ... )
        >>> train_df = df.iloc[train_idx]
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Split indices file not found: {load_path}")
    
    try:
        with open(load_path, 'rb') as f:
            split_dict = pickle.load(f)
    except Exception as e:
        raise IOError(f"Failed to load split indices: {e}") from e
    
    # Validate structure
    required_keys = ['train', 'val', 'test']
    missing_keys = [k for k in required_keys if k not in split_dict]
    if missing_keys:
        raise KeyError(
            f"Split indices file missing required keys: {missing_keys}\n"
            f"Found keys: {list(split_dict.keys())}"
        )
    
    train_idx = split_dict['train']
    val_idx = split_dict['val']
    test_idx = split_dict['test']
    
    print(f"Loaded split indices from: {load_path}")
    if 'metadata' in split_dict:
        meta = split_dict['metadata']
        print(f"  Train: {meta['train_size']:,} samples")
        print(f"  Val:   {meta['val_size']:,} samples")
        print(f"  Test:  {meta['test_size']:,} samples")
    
    return train_idx, val_idx, test_idx


def prepare_data_for_logistic_regression(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict,
    text_column: str = "comment_text",
    verbose: bool = True
) -> Tuple[Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray]]:
    """
    Prepare data for Logistic Regression training.
    
    Applies text cleaning and label encoding to train/val/test splits.
    Returns data in format suitable for sklearn models with TfidfVectorizer.
    
    Args:
        train_df: Training DataFrame with text and labels.
        val_df: Validation DataFrame with text and labels.
        test_df: Test DataFrame with text and labels.
        config: Configuration dict containing 'data' section with 'labels' list.
        text_column: Name of column containing text (default: 'comment_text').
        verbose: If True, prints progress and statistics.
    
    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test)) where:
            - X_* are lists of cleaned text strings
            - y_* are binary label matrices of shape (n_samples, n_labels)
    
    Raises:
        KeyError: If required columns or config keys not found.
    
    Example:
        >>> (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        ...     prepare_data_for_logistic_regression(
        ...         train_df, val_df, test_df, config
        ...     )
        >>> print(f"X_train: {len(X_train)} texts")
        >>> print(f"y_train: {y_train.shape}")
    """
    # Validate config
    if 'data' not in config:
        raise KeyError("Config must contain 'data' section")
    
    label_columns = config['data'].get('labels')
    if not label_columns:
        raise KeyError("Config['data'] must contain 'labels' list")
    
    # Validate text column exists in all DataFrames
    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if text_column not in df.columns:
            raise KeyError(
                f"Text column '{text_column}' not found in {name} DataFrame.\n"
                f"Available columns: {list(df.columns)}"
            )
    
    if verbose:
        print("="*60)
        print("Preparing data for Logistic Regression")
        print("="*60)
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_df):,} samples")
        print(f"  Val:   {len(val_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")
        print(f"\nLabel columns: {label_columns}")
    
    # Clean text for each split
    if verbose:
        print("\n" + "-"*60)
        print("Cleaning text...")
    
    X_train = clean_text_batch(train_df[text_column].tolist())
    X_val = clean_text_batch(val_df[text_column].tolist())
    X_test = clean_text_batch(test_df[text_column].tolist())
    
    if verbose:
        print(f"Cleaned {len(X_train):,} training texts")
        print(f"Cleaned {len(X_val):,} validation texts")
        print(f"Cleaned {len(X_test):,} test texts")
    
    # Encode labels
    if verbose:
        print("\n" + "-"*60)
        print("Encoding labels...")
    
    y_train = encode_labels(train_df, label_columns)
    y_val = encode_labels(val_df, label_columns)
    y_test = encode_labels(test_df, label_columns)
    
    if verbose:
        print(f"Encoded labels: shape {y_train.shape}")
        print(f"  {len(label_columns)} labels per sample")
    
    # Print label statistics
    if verbose:
        print("\n" + "-"*60)
        print("Label distribution in training set:")
        train_stats = get_label_statistics(train_df, label_columns)
        print(train_stats.to_string(index=False))
        
        print("\n" + "-"*60)
        print("Sample cleaned texts:")
        for i in range(min(3, len(X_train))):
            original = train_df[text_column].iloc[i]
            cleaned = X_train[i]
            print(f"\n  Original: {original[:80]}...")
            print(f"  Cleaned:  {cleaned[:80]}...")
    
    if verbose:
        print("\n" + "="*60)
        print("Data preparation complete!")
        print("="*60)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_and_prepare_data(
    csv_path: Union[str, Path],
    config: Dict,
    seed: int = 42,
    save_indices: bool = True,
    indices_path: Optional[Union[str, Path]] = None
) -> Tuple[Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray]]:
    """
    Complete pipeline: load train.csv, split 90/10 train/val, load Kaggle test set.

    When config has test_ratio=0, uses:
    - train.csv: 90% train, 10% validation
    - test.csv + test_labels.csv: Kaggle test set (rows with -1 excluded)

    Args:
        csv_path: Path to the train.csv file.
        config: Configuration dictionary.
        seed: Random seed for reproducibility.
        save_indices: If True, saves train/val split indices to disk.
        indices_path: Path to save split indices. If None, uses default.

    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test)).
    """
    csv_path = Path(csv_path)
    processed_dir = csv_path.parent

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading train data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} samples")

    # Split train/val only (90/10)
    train_df, val_df = split_data_train_val(df, config, seed=seed)

    # Load Kaggle test set (test.csv + test_labels.csv)
    test_df = load_kaggle_test_data(processed_dir, config, verbose=True)

    # Save train/val indices if requested (test comes from separate files)
    if save_indices:
        if indices_path is None:
            indices_path = processed_dir / "split_indices.pkl"
        save_split_indices(
            train_df.index.values,
            val_df.index.values,
            np.array([], dtype=np.int64),  # Test from Kaggle, not from train split
            indices_path
        )

    # Prepare data (clean text, encode labels)
    return prepare_data_for_logistic_regression(
        train_df, val_df, test_df, config
    )


if __name__ == "__main__":
    # Example usage and testing
    import json
    
    print("Testing data loading and splitting...")
    
    # Load config from file
    config_path = Path(__file__).parent.parent / "configs" / "base_config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from: {config_path}")
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}")
        print("Using default config for testing...")
        config = {
            'data': {
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1,
                'labels': ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            }
        }
    
    # Create sample DataFrame
    np.random.seed(42)
    n_samples = 1000
    
    sample_df = pd.DataFrame({
        'id': range(n_samples),
        'comment_text': [
            f"Sample comment {i} with some text http://example.com"
            for i in range(n_samples)
        ],
        'toxic': np.random.binomial(1, 0.1, n_samples),
        'severe_toxic': np.random.binomial(1, 0.01, n_samples),
        'obscene': np.random.binomial(1, 0.05, n_samples),
        'threat': np.random.binomial(1, 0.005, n_samples),
        'insult': np.random.binomial(1, 0.05, n_samples),
        'identity_hate': np.random.binomial(1, 0.01, n_samples)
    })
    
    print("\n" + "="*60)
    print("Testing split_data...")
    train_df, val_df, test_df = split_data(sample_df, config, seed=42)
    
    print("\n" + "="*60)
    print("Testing save/load split indices...")
    temp_path = Path("./temp_split_indices.pkl")
    save_split_indices(
        train_df.index.values,
        val_df.index.values,
        test_df.index.values,
        temp_path
    )
    
    train_idx, val_idx, test_idx = load_split_indices(temp_path)
    print(f"Loaded indices match: {np.array_equal(train_idx, train_df.index.values)}")
    
    # Clean up
    temp_path.unlink()
    
    print("\n" + "="*60)
    print("Testing prepare_data_for_logistic_regression...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        prepare_data_for_logistic_regression(train_df, val_df, test_df, config)
    
    print("\nAll tests passed!")
    print(f"\nFinal shapes:")
    print(f"  X_train: {len(X_train)} texts")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {len(X_val)} texts")
    print(f"  y_val: {y_val.shape}")
    print(f"  X_test: {len(X_test)} texts")
    print(f"  y_test: {y_test.shape}")
