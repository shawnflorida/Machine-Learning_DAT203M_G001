import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from backend.src.architecture.visualizer import Visualizer
from .base_model import BaseModel
from sklearn.model_selection import ParameterGrid
from backend.src.architecture.ml_tasks import EDA, Evaluator, Predictor
from backend import config

class LogisticRegressionModel(BaseModel):
    _N_EPOCHS = 200  # epochs used for partial_fit loss tracking

    def __init__(self):
        super().__init__()
        self.model = SGDClassifier(random_state=42, loss="log_loss")
        self.best_grid = None
        self.best_model = None
        self.best_metrics = None
        self._train_score: float = 0.0
        self._val_score:   float = 0.0
        self.improved_model = None
        self._loss_curve: list = []

    def train_basic(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.metrics import accuracy_score, log_loss
        import numpy as np
        import copy

        classes = np.unique(y_train)
        self._loss_curve = []
        for _ in range(self._N_EPOCHS):
            self.model.partial_fit(X_train, y_train, classes=classes)
            self._loss_curve.append(
                log_loss(y_train, self.model.predict_proba(X_train))
            )

        self._train_score = accuracy_score(y_train, self.model.predict(X_train))
        print(f"Trained: {self.get_name()}")
        print(f"Train acc: {self._train_score:.4f}", end="")
        if X_val is not None and y_val is not None:
            self._val_score = accuracy_score(y_val, self.model.predict(X_val))
            print(f"  Val acc: {self._val_score:.4f}", end="")
        print()
        self._basic_model = copy.deepcopy(self.model)

    def train_best(self, X_train, y_train, X_val, y_val):
        from sklearn.metrics import accuracy_score, log_loss
        import numpy as np

        self.grid_search(X_train, y_train, X_val, y_val)
        self._train_score = self.best_metrics["training accuracy"]
        self._val_score   = self.best_metrics["validation accuracy"]

        # Re-train the best configuration epoch-by-epoch to record a loss curve
        classes = np.unique(y_train)
        tracker = SGDClassifier(**self.best_grid)
        self._loss_curve = []
        for _ in range(self._N_EPOCHS):
            tracker.partial_fit(X_train, y_train, classes=classes)
            self._loss_curve.append(
                log_loss(y_train, tracker.predict_proba(X_train))
            )

        print(f"Trained: {self.get_name()}")
        print(f"Best params: {self.best_grid}")
        print(f"Model Fitting: {self.best_metrics}")
        self._best_model = self.best_model

    def predict(self, X):
        active = self.best_model if self.best_model is not None else self.model
        return active.predict(X)

    def predict_basic(self, X):
        if self._basic_model is None:
            raise RuntimeError("No basic model trained yet. Call train_basic() first.")
        return self._basic_model.predict(X)

    def predict_best(self, X):
        active = self._best_model if self._best_model is not None else self.best_model
        if active is None:
            raise RuntimeError("No best model trained yet. Call train_best() first.")
        return active.predict(X)

    def save_best(self, path):
        import joblib
        joblib.dump(self.best_model, path)

    def load_best(self, path):
        import joblib
        self.best_model = joblib.load(path)
        self.model = self.best_model

    def get_name(self) -> str:
        return "Multinomial Logistic Regression"

    def get_loss_curve(self):
        # Return the manually tracked per-epoch log-loss recorded during training
        return self._loss_curve if self._loss_curve else None

    def plot_loss_curve(self):
        import matplotlib.pyplot as plt

        curve = self.get_loss_curve()
        if curve is None or len(curve) == 0:
            print(f"[{self.get_name()}] No loss curve available — "
                  "train with loss='log_loss' to record per-epoch loss.")
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(curve) + 1), curve,
                color="#5C6BC0", linewidth=1.8, label="Train loss")
        ax.set(title=f"{self.get_name()} — Training Loss",
               xlabel="Epoch", ylabel="Log Loss")
        ax.legend(fontsize=9)
        ax.grid(True, linewidth=0.4)
        plt.tight_layout()
        plt.show()

    grid_search_params = [{
        "loss": ["log_loss"],
        "penalty": ["l2", "l1", "elasticnet", None],
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
        "eta0": [0.1, 0.01, 0.05, 0.001],
        "random_state": [42],
        "max_iter": [200]
    }]

    def grid_search(self, X_train, y_train, X_val, y_val):
        best_score = 0
        best_grid = {}
        for g in ParameterGrid(self.grid_search_params):
            print(g)

            self.model.set_params(**g)

            # Model Training
            self.model.fit(X_train, y_train)
            train_acc = self.model.score(X_train, y_train)

            # Validations
            val_acc = self.model.score(X_val, y_val)

            print(
                f"Train acc: {train_acc}% \t Val acc: {val_acc}%", end="\n\n")

            if val_acc > best_score:
                best_score = val_acc
                best_model = self.model
                best_grid = g
                evaluator = Evaluator()
                best_metrics = {"training accuracy": train_acc,
                                "validation accuracy": val_acc}

        pred_cats = {self.get_name(): self.model.predict(X_val)}
        class_reports = evaluator.classification_report_all(
            y_val, pred_cats, config.CATEGORY_ORDER)
        v = Visualizer()
        v.plot_confusion_matrices(class_reports, config.CATEGORY_ORDER)

        print("Best accuracy: ", best_score, "%")
        print("Best grid: ", best_grid)
        self.best_grid = best_grid
        self.best_model = best_model
        self.best_metrics = best_metrics
