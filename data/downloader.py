#!/usr/bin/env python3
"""
Download Jigsaw Toxic Comment Classification dataset from Kaggle.

This module provides functions to download and verify the Jigsaw dataset
using the Kaggle API. It handles authentication, download, extraction,
and validation of the dataset files.

Requirements:
    - kaggle package installed (pip install kaggle)
    - Kaggle API token: export KAGGLE_API_TOKEN="your_token"

Usage:
    from data.downloader import download_jigsaw_dataset, check_dataset_exists
    
    # Download dataset
    csv_path = download_jigsaw_dataset(save_dir="./data/raw")
    
    # Check if already downloaded
    if check_dataset_exists("./data/raw"):
        print("Dataset already exists")
"""

import os
import zipfile
from pathlib import Path
from typing import Optional, Union

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    raise ImportError(
        "Kaggle API not found. Install with: pip install kaggle"
    )


def download_jigsaw_dataset(
    save_dir: Union[str, Path],
    kaggle_credentials: Optional[dict] = None,
    extract_to: Optional[Union[str, Path]] = None,
) -> str:
    """
    Download Jigsaw Toxic Comment Classification dataset from Kaggle.
    
    Downloads the dataset using the Kaggle API and optionally extracts
    the train.csv file to a specified directory.
    
    Args:
        save_dir: Directory to save the downloaded zip files.
        kaggle_credentials: Optional dict with 'username' and 'key' for Kaggle API.
                          If None, uses default Kaggle authentication.
        extract_to: Optional directory to extract train.csv to. If None, files
                   remain zipped in save_dir.
    
    Returns:
        Path to the extracted train.csv file (if extract_to provided) or
        path to the downloaded zip directory.
    
    Raises:
        FileNotFoundError: If Kaggle credentials are not found.
        Exception: If download or extraction fails.
    
    Example:
        >>> csv_path = download_jigsaw_dataset(
        ...     save_dir="./data/raw",
        ...     extract_to="./data/processed"
        ... )
        >>> print(f"Dataset ready at: {csv_path}")
    """
    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle custom credentials if provided (takes precedence)
    if kaggle_credentials:
        os.environ['KAGGLE_USERNAME'] = kaggle_credentials.get('username', '')
        os.environ['KAGGLE_KEY'] = kaggle_credentials.get('key', '')
    
    # Initialize and authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download dataset
    dataset_name = "jigsaw-toxic-comment-classification-challenge"
    print(f"Downloading dataset: {dataset_name}")
    print(f"Saving to: {save_dir}")
    
    try:
        api.competition_download_files(
            competition=dataset_name,
            path=str(save_dir),
            quiet=False
        )
        print(f"Download complete")
    except Exception as e:
        raise Exception(f"Failed to download dataset: {e}") from e
    
    # If extract_to is specified, extract train.csv
    if extract_to:
        return ensure_train_csv_ready(str(save_dir), str(extract_to))
    
    return str(save_dir)


def ensure_train_csv_ready(
    zip_dir: Union[str, Path],
    extract_dir: Union[str, Path]
) -> str:
    """
    Extract train.csv, test.csv, and test_labels.csv from downloaded Kaggle zip files.
    
    Searches for CSV files in the main competition zip or in nested zips
    and extracts them to the specified directory.
    
    Args:
        zip_dir: Directory containing downloaded zip files.
        extract_dir: Directory to extract CSV files to.
    
    Returns:
        Path to the extracted train.csv file.
    
    Raises:
        FileNotFoundError: If no zip files or train.csv found.
        zipfile.BadZipFile: If zip files are corrupted.
    
    Example:
        >>> csv_path = ensure_train_csv_ready(
        ...     zip_dir="./data/raw",
        ...     extract_dir="./data/processed"
        ... )
    """
    zip_dir = Path(zip_dir).resolve()
    extract_dir = Path(extract_dir).resolve()
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv_path = extract_dir / "train.csv"
    test_csv_path = extract_dir / "test.csv"
    test_labels_path = extract_dir / "test_labels.csv"
    
    # Check which files are already extracted
    files_to_extract = []
    if not train_csv_path.exists():
        files_to_extract.append('train.csv')
    else:
        print(f"train.csv already exists at: {train_csv_path}")
    
    if not test_csv_path.exists():
        files_to_extract.append('test.csv')
    else:
        print(f"test.csv already exists at: {test_csv_path}")
    
    if not test_labels_path.exists():
        files_to_extract.append('test_labels.csv')
    else:
        print(f"test_labels.csv already exists at: {test_labels_path}")
    
    # If all files exist, return train.csv path
    if not files_to_extract:
        return str(train_csv_path)
    
    print(f"Extracting {', '.join(files_to_extract)} from zips in {zip_dir}")
    
    # Look for zip files
    zip_files = list(zip_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip files found in {zip_dir}")
    
    # Try to find and extract CSV files
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # Extract files that are directly in this zip
                for csv_file in files_to_extract.copy():
                    if csv_file in file_list:
                        print(f"Extracting {csv_file} from {zip_path.name}")
                        zip_ref.extract(csv_file, extract_dir)
                        print(f"Extracted to: {extract_dir / csv_file}")
                        files_to_extract.remove(csv_file)
                
                # Check for nested zips (e.g., train.csv.zip)
                for csv_file in files_to_extract.copy():
                    nested_zip_name = f"{csv_file}.zip"
                    if nested_zip_name in file_list:
                        print(f"Found {nested_zip_name} in {zip_path.name}, extracting...")
                        zip_ref.extract(nested_zip_name, extract_dir)
                        
                        # Extract CSV from the nested zip
                        nested_zip = extract_dir / nested_zip_name
                        with zipfile.ZipFile(nested_zip, 'r') as nested_ref:
                            nested_ref.extract(csv_file, extract_dir)
                        
                        # Clean up nested zip
                        nested_zip.unlink()
                        print(f"Extracted to: {extract_dir / csv_file}")
                        files_to_extract.remove(csv_file)
        
        except zipfile.BadZipFile:
            print(f"Warning: {zip_path.name} is not a valid zip file, skipping")
            continue
        except Exception as e:
            print(f"Warning: Error processing {zip_path.name}: {e}")
            continue
    
    # Check if train.csv was extracted (required)
    if not train_csv_path.exists():
        raise FileNotFoundError(
            f"train.csv not found in any zip files in {zip_dir}\n"
            f"Available zips: {[z.name for z in zip_files]}"
        )
    
    # Warn if optional files weren't found
    if not test_csv_path.exists():
        print(f"Warning: test.csv not found in zip files")
    if not test_labels_path.exists():
        print(f"Warning: test_labels.csv not found in zip files")
    
    return str(train_csv_path)


def check_dataset_exists(
    zip_dir: Union[str, Path],
    extract_dir: Optional[Union[str, Path]] = None
) -> bool:
    """
    Check if the Jigsaw dataset has already been downloaded.
    
    Checks for either:
    1. Extracted train.csv in extract_dir (if provided), or
    2. Zip files in zip_dir
    
    Args:
        zip_dir: Directory where zip files would be saved.
        extract_dir: Optional directory where train.csv would be extracted.
    
    Returns:
        True if dataset exists (either as zip or extracted CSV), False otherwise.
    
    Example:
        >>> if check_dataset_exists("./data/raw", "./data/processed"):
        ...     print("Dataset already downloaded")
        ... else:
        ...     download_jigsaw_dataset("./data/raw")
    """
    zip_dir = Path(zip_dir).resolve()
    
    # First check if CSV files are already extracted
    if extract_dir:
        extract_dir = Path(extract_dir).resolve()
        train_csv = extract_dir / "train.csv"
        test_csv = extract_dir / "test.csv"
        test_labels = extract_dir / "test_labels.csv"
        
        if train_csv.exists():
            print(f"Found extracted train.csv at: {train_csv}")
            if test_csv.exists():
                print(f"Found extracted test.csv at: {test_csv}")
            if test_labels.exists():
                print(f"Found extracted test_labels.csv at: {test_labels}")
            return True
    
    # Check if zip files exist
    if zip_dir.exists():
        zip_files = list(zip_dir.glob("*.zip"))
        if zip_files:
            print(f"Found {len(zip_files)} zip file(s) in {zip_dir}")
            return True
    
    return False


def validate_dataset(csv_path: Union[str, Path]) -> bool:
    """
    Validate that the downloaded dataset has the expected structure.
    
    Checks for required columns and basic data integrity.
    
    Args:
        csv_path: Path to the train.csv file.
    
    Returns:
        True if dataset is valid, False otherwise.
    
    Example:
        >>> if validate_dataset("./data/processed/train.csv"):
        ...     print("Dataset is valid")
    """
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not installed, skipping validation")
        return True
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return False
    
    expected_columns = [
        "id", "comment_text",
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
    ]
    
    try:
        # Read just the header to check columns
        df = pd.read_csv(csv_path, nrows=5)
        
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing columns: {missing_cols}")
            print(f"Found columns: {list(df.columns)}")
            return False
        
        print(f"Dataset validation passed")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Expected columns present: {len(expected_columns)}")
        return True
    
    except Exception as e:
        print(f"Error validating dataset: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    import sys
    
    save_dir = "./data/raw"
    extract_dir = "./data/processed"
    
    if check_dataset_exists(save_dir, extract_dir):
        print("Dataset already exists!")
        csv_path = ensure_train_csv_ready(save_dir, extract_dir)
    else:
        print("Downloading dataset...")
        csv_path = download_jigsaw_dataset(save_dir, extract_to=extract_dir)
    
    print(f"\nDataset ready at: {csv_path}")
    
    # Validate
    if validate_dataset(csv_path):
        print("All checks passed!")
        sys.exit(0)
    else:
        print("✗ Validation failed")
        sys.exit(1)
