from abc import ABC, abstractmethod

import joblib


class BaseModel(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    def get_feature_importance(self, feature_names):
        return None

    def get_training_curve(self):
        return None

    def get_visual_payload(self, feature_names=None):
        return {
            "model_name": self.get_name(),
            "feature_importance": self.get_feature_importance(feature_names) if feature_names is not None else None,
            "training_curve": self.get_training_curve(),
        }

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
