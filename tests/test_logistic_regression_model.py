"""
Test suite for LogisticRegressionModel
Run: python test_logistic_regression_model.py
"""

import os
import numpy as np
import tempfile


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root

from models.logistic_regression_model import LogisticRegressionModel


LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
N_LABELS = len(LABELS)

MOCK_CONFIG = {
    "data": {"labels": LABELS},
    "preprocessing": {
        "max_features": 500,
        "ngram_range": [1, 2],
        "lowercase": True,
    },
    "hyperparameters": {
        "C": 1.0,
        "max_iter": 100,
        "class_weight": "balanced",
        "solver": "lbfgs",
    },
    "training": {"n_jobs": 1},
}

# Small synthetic dataset
TRAIN_TEXTS = [
    "you are stupid and horrible",
    "i will kill you",
    "this is a great article",
    "nice work everyone",
    "what an idiot you are",
    "go die in a fire",
    "great contribution to the topic",
    "thanks for sharing this",
    "you are the worst person ever",
    "i hate everything about you",
    "very informative post",
    "well written article",
]

TRAIN_LABELS = np.array([
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 1],  # ← added identity_hate
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],  # ← added identity_hate
    [0, 0, 0, 0, 0, 0],
])


VAL_TEXTS = ["you are awful", "great job"]
VAL_LABELS = np.array([[1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]])

TEST_TEXTS = ["i hate you so much", "wonderful explanation"]



# Helpers

def passed(test_name: str):
    print(f"  ✓ {test_name}")

def failed(test_name: str, reason: str):
    print(f"  ✗ {test_name}: {reason}")

def section(title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


# Tests

def test_instantiation():
    section("1. Instantiation")
    try:
        model = LogisticRegressionModel(MOCK_CONFIG)
        assert model.pipeline is not None
        passed("Model created successfully")
        assert model.labels == LABELS
        passed("Labels loaded from config")
        assert hasattr(model, "vectorizer")
        passed("Vectorizer attribute exists")
    except Exception as e:
        failed("Instantiation", str(e))
    return model


def test_train(model):
    section("2. Training")
    try:
        model.train((TRAIN_TEXTS, TRAIN_LABELS), val_data=(VAL_TEXTS, VAL_LABELS))
        passed("train() completed without errors")
    except Exception as e:
        failed("train()", str(e))


def test_predict(model):
    section("3. Predict (binary)")
    try:
        preds = model.predict(TEST_TEXTS)
        passed("predict() returned without errors")

        assert isinstance(preds, np.ndarray), "Output must be ndarray"
        passed("Output is np.ndarray")

        assert preds.shape == (len(TEST_TEXTS), N_LABELS), \
            f"Expected shape ({len(TEST_TEXTS)}, {N_LABELS}), got {preds.shape}"
        passed(f"Output shape correct: {preds.shape}")

        assert set(np.unique(preds)).issubset({0, 1}), "Predictions must be binary"
        passed("Values are binary {0, 1}")

        print(f"\n  Sample predictions:")
        for text, pred in zip(TEST_TEXTS, preds):
            print(f"    '{text[:40]}' → {dict(zip(LABELS, pred))}")

    except AssertionError as e:
        failed("predict()", str(e))
    except Exception as e:
        failed("predict()", str(e))


def test_predict_proba(model):
    section("4. Predict Probabilities")
    try:
        proba = model.predict_proba(TEST_TEXTS)
        passed("predict_proba() returned without errors")

        assert isinstance(proba, np.ndarray), "Output must be ndarray"
        passed("Output is np.ndarray")

        assert proba.shape == (len(TEST_TEXTS), N_LABELS), \
            f"Expected shape ({len(TEST_TEXTS)}, {N_LABELS}), got {proba.shape}"
        passed(f"Output shape correct: {proba.shape}")

        assert np.all(proba >= 0) and np.all(proba <= 1), "Probs must be in [0, 1]"
        passed("All values in [0.0, 1.0]")

        print(f"\n  Sample probabilities:")
        for text, prob in zip(TEST_TEXTS, proba):
            print(f"    '{text[:40]}'")
            for label, p in zip(LABELS, prob):
                print(f"      {label}: {p:.4f}")

    except AssertionError as e:
        failed("predict_proba()", str(e))
    except Exception as e:
        failed("predict_proba()", str(e))


def test_save_load(model):
    section("5. Save & Load")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test path with directory
            model_path = os.path.join(tmpdir, "saved_model", "model.pkl")
            model.save(model_path)
            assert os.path.exists(model_path)
            passed("save() with nested path works")

            # Test path without directory
            simple_path = os.path.join(tmpdir, "model.pkl")
            model.save(simple_path)
            assert os.path.exists(simple_path)
            passed("save() with simple path works")

            # Test load and predict
            model.load(model_path)
            passed("load() completed without errors")

            preds_after_load = model.predict(TEST_TEXTS)
            assert preds_after_load.shape == (len(TEST_TEXTS), N_LABELS)
            passed("predict() works correctly after load")

    except AssertionError as e:
        failed("save/load", str(e))
    except Exception as e:
        failed("save/load", str(e))


def test_model_size(model):
    section("6. Model Size")
    try:
        size = model.get_model_size()
        passed("get_model_size() returned without errors")

        assert isinstance(size, float), "Size must be a float"
        passed("Size is a float")

        assert size > 0, "Size must be > 0"
        passed(f"Model size: {size} MB")

    except AssertionError as e:
        failed("get_model_size()", str(e))
    except Exception as e:
        failed("get_model_size()", str(e))


def test_edge_cases(model):
    section("7. Edge Cases")

    # Single sample
    try:
        preds = model.predict(["test single input"])
        assert preds.shape == (1, N_LABELS)
        passed("Single sample prediction works")
    except Exception as e:
        failed("Single sample", str(e))

    # Empty string
    try:
        preds = model.predict([""])
        assert preds.shape == (1, N_LABELS)
        passed("Empty string prediction works")
    except Exception as e:
        failed("Empty string", str(e))

    # Very long text
    try:
        long_text = "word " * 500
        preds = model.predict([long_text])
        assert preds.shape == (1, N_LABELS)
        passed("Very long text prediction works")
    except Exception as e:
        failed("Long text", str(e))


# Main

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  LogisticRegressionModel Test Suite")
    print("="*50)

    model = test_instantiation()
    test_train(model)
    test_predict(model)
    test_predict_proba(model)
    test_save_load(model)
    test_model_size(model)
    test_edge_cases(model)

    print("\n" + "="*50)
    print("  All tests completed!")
    print("="*50 + "\n")
