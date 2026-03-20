from abc import ABC, abstractmethod

import joblib


class BaseModel(ABC):
    def __init__(self):
        self.model = None
        self._basic_model = None
        self._best_model  = None

    @abstractmethod
    def train_basic(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError
    
    @abstractmethod
    def train_best(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    def predict_basic(self, X):
        if self._basic_model is None:
            raise RuntimeError("No basic model trained yet. Call train_basic() first.")
        return self._basic_model.predict(X)

    def predict_best(self, X):
        if self._best_model is None:
            raise RuntimeError("No best model trained yet. Call train_best() first.")
        return self._best_model.predict(X)

    def save_basic(self, path):
        if self._basic_model is None:
            raise RuntimeError("No basic model trained yet. Call train_basic() first.")
        joblib.dump(self._basic_model, path)

    def save_best(self, path):
        if self._best_model is None:
            raise RuntimeError("No best model trained yet. Call train_best() first.")
        joblib.dump(self._best_model, path)

    def load_basic(self, path):
        self._basic_model = joblib.load(path)
        self.model = self._basic_model

    def load_best(self, path):
        self._best_model = joblib.load(path)
        self.model = self._best_model
