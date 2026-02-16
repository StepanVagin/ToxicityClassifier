"""
Data module for Jigsaw Toxic Comment Classification.

This module provides utilities for downloading, preprocessing, and loading
the Jigsaw dataset for machine learning models.

Modules:
    - downloader: Download dataset from Kaggle
    - preprocessor: Clean text and encode labels
    - data_loader: Split data and prepare for training

Example usage:
    from data import (
        download_jigsaw_dataset,
        clean_text,
        split_data,
        prepare_data_for_logistic_regression
    )
    
    # Download dataset
    csv_path = download_jigsaw_dataset("./data/raw", extract_to="./data/processed")
    
    # Load and prepare data
    df = pd.read_csv(csv_path)
    train_df, val_df, test_df = split_data(df, config, seed=42)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        prepare_data_for_logistic_regression(train_df, val_df, test_df, config)
"""

from data.downloader import (
    download_jigsaw_dataset,
    check_dataset_exists,
    ensure_train_csv_ready,
    validate_dataset
)

from data.preprocessor import (
    clean_text,
    clean_text_batch,
    encode_labels,
    preprocess_dataframe,
    get_label_statistics
)

from data.data_loader import (
    split_data,
    save_split_indices,
    load_split_indices,
    prepare_data_for_logistic_regression,
    load_and_prepare_data
)

__all__ = [
    # Downloader
    'download_jigsaw_dataset',
    'check_dataset_exists',
    'ensure_train_csv_ready',
    'validate_dataset',
    
    # Preprocessor
    'clean_text',
    'clean_text_batch',
    'encode_labels',
    'preprocess_dataframe',
    'get_label_statistics',
    
    # Data Loader
    'split_data',
    'save_split_indices',
    'load_split_indices',
    'prepare_data_for_logistic_regression',
    'load_and_prepare_data',
]

__version__ = '0.1.0'
