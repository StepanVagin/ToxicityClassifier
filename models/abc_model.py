import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

class ModelABC(ABC):

    @abstractmethod
    def train(self, train_data: Tuple, val_data: Tuple = None) -> None:
        """Train the model on train_data, monitor on val_data"""
        pass

    @abstractmethod
    def predict(self, texts: list) -> np.ndarray:
        """Return binary predictions (n_samples, 6)"""
        pass

    @abstractmethod
    def predict_proba(self, texts: list) -> np.ndarray:
        """Return probabilities (n_samples, 6)"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk"""
        pass

    @abstractmethod
    def get_model_size(self) -> float:
        """Return model size in MB"""
        pass
