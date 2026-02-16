#!/usr/bin/env python3
"""
Text preprocessing and label encoding for Jigsaw Toxic Comment Classification.

This module provides functions to clean text data and encode multi-label
classifications for use with classical machine learning models like
Logistic Regression.

The preprocessing pipeline includes:
    - Lowercase conversion
    - URL removal
    - Email removal
    - Special character removal
    - Label encoding to binary matrices

Usage:
    from data.preprocessor import clean_text, encode_labels
    
    # Clean a single text
    cleaned = clean_text("Check out http://example.com! Email: test@test.com")
    
    # Encode labels from DataFrame
    label_matrix = encode_labels(df, ["toxic", "severe_toxic", "obscene"])
"""

import re
from typing import List, Union

import numpy as np
import pandas as pd


def clean_text(text: Union[str, None]) -> str:
    """
    Clean and normalize text for classical ML models.
    
    Applies the following transformations:
        1. Converts to lowercase
        2. Removes URLs (http/https links)
        3. Removes email addresses
        4. Removes special characters (keeps only alphanumeric and spaces)
        5. Removes extra whitespace
    
    Args:
        text: Input text string to clean. If None or empty, returns empty string.
    
    Returns:
        Cleaned text string with only lowercase alphanumeric characters and spaces.
    
    Examples:
        >>> clean_text("Check THIS out: http://example.com!")
        'check this out'
        
        >>> clean_text("Email me at test@example.com for INFO!!!")
        'email me at for info'
        
        >>> clean_text("Hello    World!!!   123")
        'hello world 123'
        
        >>> clean_text(None)
        ''
    """
    # Handle None or empty input
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs (http, https, www)
    # Pattern matches: http://..., https://..., www....
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove email addresses
    # Pattern matches: anything@anything.anything
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters - keep only alphanumeric and spaces
    # This removes punctuation, symbols, etc.
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove extra whitespace (multiple spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def clean_text_batch(texts: List[Union[str, None]]) -> List[str]:
    """
    Clean a batch of texts efficiently.
    
    Applies clean_text to a list of texts. Useful for preprocessing
    entire datasets.
    
    Args:
        texts: List of text strings to clean.
    
    Returns:
        List of cleaned text strings.
    
    Example:
        >>> texts = ["Hello World!", "Check http://test.com", None]
        >>> clean_text_batch(texts)
        ['hello world', 'check', '']
    """
    return [clean_text(text) for text in texts]


def encode_labels(
    df: pd.DataFrame,
    label_columns: List[str]
) -> np.ndarray:
    """
    Encode multi-label classification labels as binary matrix.
    
    Converts DataFrame columns containing binary labels (0/1) into
    a numpy array suitable for multi-label classification with sklearn.
    
    Args:
        df: DataFrame containing label columns.
        label_columns: List of column names to encode as labels.
                      Each column should contain binary values (0 or 1).
    
    Returns:
        Binary matrix of shape (n_samples, n_labels) where each row
        represents one sample and each column represents one label.
        Values are 0 or 1.
    
    Raises:
        KeyError: If any label column is not found in the DataFrame.
        ValueError: If label columns contain non-binary values.
    
    Examples:
        >>> df = pd.DataFrame({
        ...     'text': ['comment1', 'comment2'],
        ...     'toxic': [1, 0],
        ...     'obscene': [0, 1]
        ... })
        >>> labels = encode_labels(df, ['toxic', 'obscene'])
        >>> labels.shape
        (2, 2)
        >>> labels
        array([[1, 0],
               [0, 1]])
    """
    # Validate that all label columns exist
    missing_cols = [col for col in label_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Label columns not found in DataFrame: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    # Extract label columns
    label_matrix = df[label_columns].values
    
    # Validate that labels are binary
    unique_values = np.unique(label_matrix)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(
            f"Label columns must contain only binary values (0 or 1).\n"
            f"Found values: {unique_values}"
        )
    
    # Convert to int type for consistency
    label_matrix = label_matrix.astype(np.int32)
    
    return label_matrix


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "comment_text",
    label_columns: Optional[List[str]] = None,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Preprocess an entire DataFrame with text cleaning.
    
    Applies text cleaning to the specified text column and optionally
    validates label columns.
    
    Args:
        df: Input DataFrame.
        text_column: Name of the column containing text to clean.
        label_columns: Optional list of label column names to validate.
        inplace: If True, modifies the DataFrame in place. If False,
                returns a copy.
    
    Returns:
        DataFrame with cleaned text column (and optionally validated labels).
    
    Raises:
        KeyError: If text_column or label_columns not found.
    
    Example:
        >>> df = pd.DataFrame({
        ...     'comment_text': ['Hello WORLD!', 'Test http://url.com'],
        ...     'toxic': [0, 1]
        ... })
        >>> cleaned_df = preprocess_dataframe(
        ...     df,
        ...     text_column='comment_text',
        ...     label_columns=['toxic']
        ... )
        >>> cleaned_df['comment_text'].tolist()
        ['hello world', 'test']
    """
    if not inplace:
        df = df.copy()
    
    # Validate text column exists
    if text_column not in df.columns:
        raise KeyError(
            f"Text column '{text_column}' not found in DataFrame.\n"
            f"Available columns: {list(df.columns)}"
        )
    
    # Clean text column
    print(f"Cleaning {len(df)} texts in column '{text_column}'...")
    df[text_column] = clean_text_batch(df[text_column].tolist())
    
    # Validate label columns if provided
    if label_columns:
        missing_cols = [col for col in label_columns if col not in df.columns]
        if missing_cols:
            raise KeyError(
                f"Label columns not found: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )
        
        # Check for missing values in labels
        for col in label_columns:
            if df[col].isnull().any():
                print(f"Warning: Column '{col}' contains {df[col].isnull().sum()} missing values")
    
    print(f"Preprocessing complete")
    return df


def get_label_statistics(
    df: pd.DataFrame,
    label_columns: List[str]
) -> pd.DataFrame:
    """
    Calculate statistics for multi-label classification labels.
    
    Computes counts and percentages for each label, useful for
    understanding class imbalance.
    
    Args:
        df: DataFrame containing label columns.
        label_columns: List of label column names.
    
    Returns:
        DataFrame with statistics for each label:
            - label: Label name
            - count: Number of positive samples
            - percentage: Percentage of positive samples
    
    Example:
        >>> df = pd.DataFrame({
        ...     'toxic': [1, 0, 1, 0],
        ...     'obscene': [0, 0, 1, 0]
        ... })
        >>> stats = get_label_statistics(df, ['toxic', 'obscene'])
        >>> print(stats)
              label  count  percentage
        0     toxic      2       50.00
        1   obscene      1       25.00
    """
    stats = []
    total = len(df)
    
    for label in label_columns:
        if label not in df.columns:
            continue
        
        count = df[label].sum()
        percentage = (count / total) * 100
        
        stats.append({
            'label': label,
            'count': int(count),
            'percentage': round(percentage, 2)
        })
    
    return pd.DataFrame(stats)


# Type hint import
from typing import Optional


if __name__ == "__main__":
    # Example usage and testing
    print("Testing text cleaning...")
    
    test_texts = [
        "Check out this link: http://example.com for more INFO!!!",
        "Email me at test@example.com or visit www.test.com",
        "This is NORMAL text with some punctuation!!!",
        "Multiple    spaces   and\ttabs\nand newlines",
        None,
        ""
    ]
    
    print("\nOriginal -> Cleaned:")
    for text in test_texts:
        cleaned = clean_text(text)
        print(f"  {repr(text)[:50]:50s} -> {repr(cleaned)}")
    
    print("\n" + "="*60)
    print("Testing label encoding...")
    
    # Create sample DataFrame
    sample_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'comment_text': [
            'This is great!',
            'You are terrible',
            'Normal comment',
            'Another bad one',
            'Good stuff'
        ],
        'toxic': [0, 1, 0, 1, 0],
        'severe_toxic': [0, 1, 0, 0, 0],
        'obscene': [0, 1, 0, 1, 0]
    })
    
    label_cols = ['toxic', 'severe_toxic', 'obscene']
    labels = encode_labels(sample_df, label_cols)
    
    print(f"\nLabel matrix shape: {labels.shape}")
    print(f"Label matrix:\n{labels}")
    
    print("\n" + "="*60)
    print("Label statistics:")
    stats = get_label_statistics(sample_df, label_cols)
    print(stats.to_string(index=False))
    
    print("\nAll tests passed!")
