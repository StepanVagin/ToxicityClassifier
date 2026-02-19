from typing import List, Tuple
import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
import joblib

from .abc_model import ModelABC


class LogisticRegressionModel(ModelABC):
    """
    TF-IDF + MultiOutput Logistic Regression baseline for multilabel toxicity.
    """

    def __init__(self, config: dict) -> None:
        """
        Parameters
        ----------
        config : dict
            Merged config (base_config + logistic_regression.json).
        """
        self.config = config
        self.labels = config["data"]["labels"]

        # TF-IDF params
        prep_cfg = config["preprocessing"]
        self.vectorizer = TfidfVectorizer(
            max_features=prep_cfg.get("max_features", 10000),
            ngram_range=tuple(prep_cfg.get("ngram_range", [1, 2])),
            lowercase=prep_cfg.get("lowercase", True),
            strip_accents="unicode",
        )

        # Logistic Regression params
        hp = config["hyperparameters"]
        base_lr = LogisticRegression(
            C=hp.get("C", 1.0),
            max_iter=hp.get("max_iter", 100),
            class_weight=hp.get("class_weight", "balanced"),
            solver=hp.get("solver", "lbfgs"),
        )

        # Multi-output wrapper
        clf = MultiOutputClassifier(base_lr, n_jobs=config.get("training", {}).get("n_jobs", -1))

        # Full pipeline: text → TF-IDF → 6 LR heads
        self.pipeline: Pipeline = Pipeline(
            [
                ("tfidf", self.vectorizer),
                ("clf", clf),
            ]
        )

    def train(self, train_data: Tuple, val_data: Tuple = None) -> None:

        """
        Fit TF-IDF + LR on training data.

        Parameters
        ----------
        train_data : (X_train, y_train)
            X_train: list of raw or cleaned texts
            y_train: ndarray of shape (n_samples, n_labels)
        """
        X_train, y_train = train_data
        self.pipeline.fit(X_train, y_train)

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict binary labels for given texts.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_labels) with {0,1} predictions.
        """
        preds = self.pipeline.predict(texts)
        return np.asarray(preds)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities for each label.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_labels) with probabilities in [0,1].
        """
        proba_per_label = self.pipeline.predict_proba(texts)
        return np.column_stack([p[:, 1] for p in proba_per_label])

    def save(self, path: str) -> None:
        dir_name = os.path.dirname(path)
        if dir_name:  
            os.makedirs(dir_name, exist_ok=True)
        joblib.dump(self.pipeline, path)


    def load(self, path: str) -> None:
        """
        Load pipeline from disk.
        """
        self.pipeline = joblib.load(path)

    def get_model_size(self) -> float:
        """Returns model size in MB after saving."""
        tmp_path = "/tmp/_model_size_check.pkl"
        joblib.dump(self.pipeline, tmp_path)
        size = os.path.getsize(tmp_path) / (1024 * 1024)
        os.remove(tmp_path)
        return round(size, 2)


