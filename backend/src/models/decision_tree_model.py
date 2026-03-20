import copy
from itertools import product
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from .base_model import BaseModel

OPTIMAL_TRAIN_SCORE = 1.00
VAL_GAP_TOLERANCE = 0.05
MAX_ROUNDS = 5


class DecisionTreeModel(BaseModel):

    HYPERPARAMETERS = {
        "criterion":             ["gini", "entropy"],
        "max_depth":             [3, 4, 5, 7, 10, 15],
        "min_samples_split":     [2, 4, 6, 10, 15, 20],
        "min_samples_leaf":      [1, 2, 4, 8],
        "max_leaf_nodes":        [10, 20, 30, 50, None],
        "min_impurity_decrease": [0.0, 0.001, 0.005, 0.01],
    }

    def __init__(self):
        super().__init__()

        self.model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )
        self._train_score: float = 0.0
        self._val_score:   float = 0.0
        self._basic_model = None
        self._best_model  = None
        self.improved_model = None
        
    @staticmethod
    def compute_accuracy(predictions, actual) -> float:

        return float(np.mean(np.array(predictions) == np.array(actual)) * 100)

    def train_basic(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.metrics import f1_score as _f1
        import copy
        self.model.fit(X_train, y_train)
        self._train_score = _f1(y_train, self.model.predict(X_train), average='weighted', zero_division=0)
        print(f"Trained (basic): {self.get_name()}")
        print(f"Train F1: {self._train_score:.4f}", end="")
        if X_val is not None and y_val is not None:
            self._val_score = _f1(y_val, self.model.predict(X_val), average='weighted', zero_division=0)
            print(f"  Val F1: {self._val_score:.4f}", end="")
        print()
        self._basic_model = copy.deepcopy(self.model)

    def train_best(self, X_train, y_train, X_val=None, y_val=None):

        print(f"\n{'─' * 52}")
        print(f"  Training: {self.get_name()}")
        print(f"{'─' * 52}")

        # Phase 1: Baseline → observe overfitting, then regularize
        print("\n  [Phase 1] Baseline overfitting check + depth sweep")
        print(f"  {'depth':>9} | {'train_f1':>9} | {'val_f1':>8} | {'gap':>8}")
        print(f"  {'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}")

        # None first = unconstrained baseline
        depth_steps = [None, 5, 7, 10, 13, None]
        best_round_model = None

        for round_num, max_depth in enumerate(depth_steps[:MAX_ROUNDS + 1], start=0):
            candidate = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=self.model.min_samples_split,
                min_samples_leaf=self.model.min_samples_leaf,
                random_state=42,
            )
            candidate.fit(X_train, y_train)

            tr = f1_score(y_train, candidate.predict(X_train), average='weighted', zero_division=0)
            va = f1_score(y_val,   candidate.predict(
                X_val), average='weighted', zero_division=0) if X_val is not None else float("nan")
            gap = abs(tr - va) if X_val is not None else float("nan")

            depth_label = str(
                max_depth) if max_depth is not None else "unlimited"

            if round_num == 0:
                print(
                    f"  {'baseline':>9} | {tr:>9.4f} | {va:>8.4f} | {gap:>8.4f}  ← overfitting baseline")
                continue

            print(f"  {depth_label:>9} | {tr:>9.4f} | {va:>8.4f} | {gap:>8.4f}")

            self._train_score = tr
            self.model = candidate
            best_round_model = candidate

            if tr >= OPTIMAL_TRAIN_SCORE:
                print(f"\n Optimal training score reached "
                      f"({tr:.4f} >= {OPTIMAL_TRAIN_SCORE}). Stopping early.")
                break
        else:
            print(f"\n Max training rounds ({MAX_ROUNDS}) reached. "
                  f"Final train_acc = {self._train_score:.4f}")

        # Phase 2: Validation gap check
        if X_val is not None and y_val is not None:
            self._val_score = f1_score(y_val, self.model.predict(X_val), average='weighted', zero_division=0)
            gap = abs(self._train_score - self._val_score)

            print(f"\n  [Phase 2] Validation check")
            print(f"  train_f1 = {self._train_score:.4f} | "
                  f"val_f1 = {self._val_score:.4f} | "
                  f"gap = {gap:.4f} (tolerance = {VAL_GAP_TOLERANCE})")

            if gap <= VAL_GAP_TOLERANCE:
                print(f"  Gap within tolerance. Proceeding to test set.")
            else:
                print(f"  Gap too large. Starting manual grid search hypertuning ...")
                self._hypertune(X_train, y_train, X_val, y_val)
        else:
            print("\n  No validation data provided — skipping gap check.")

        import copy
        self._best_model = copy.deepcopy(self.model)
        print(f"\n{'─' * 52}\n")

    def _hypertune(self, X_train, y_train, X_val, y_val):

        param_names = list(self.HYPERPARAMETERS.keys())
        param_values = list(self.HYPERPARAMETERS.values())
        combos = list(product(*param_values))
        print(f"\n  [Phase 3] Manual grid search — {len(combos)} combinations")

        best_val_score = -1.0
        best_params = {}
        best_model = None

        for combo in combos:
            params = dict(zip(param_names, combo))
            candidate = DecisionTreeClassifier(random_state=42, **params)
            candidate.fit(X_train, y_train)
            val_acc = f1_score(y_val, candidate.predict(X_val), average='weighted', zero_division=0)
            if val_acc > best_val_score:
                best_val_score = val_acc
                best_params = params
                best_model = candidate

        self.model = best_model
        self._train_score = f1_score(y_train, self.model.predict(X_train), average='weighted', zero_division=0)
        self._val_score = best_val_score
        gap = abs(self._train_score - self._val_score)

        print(f"  Best hyperparameters : {best_params}")
        print(f"  train_f1={self._train_score:.4f}  "
              f"val_f1={self._val_score:.4f}  gap={gap:.4f}")
        print(f"  Tree depth={self.model.get_depth()}  "
              f"n_leaves={self.model.get_n_leaves()}")

        if gap <= VAL_GAP_TOLERANCE:
            print(f"  Gap within tolerance. Proceeding to test set.")
        else:
            print(f"  Gap still large after full grid search "
                  f"(val_acc={self._val_score:.4f}). Proceeding with best found.")

    def val(self, X_val, y_val) -> float:
        preds = self.model.predict(X_val)
        acc = self.compute_accuracy(preds, y_val)
        score = acc / 100
        print(f"  [{self.get_name()}]  "
              f"Validation accuracy: {acc:.4f}%")
        self._val_score = score
        return score

    def test(self, X_test, y_test) -> float:
        preds = self.model.predict(X_test)
        acc = self.compute_accuracy(preds, y_test)
        score = acc / 100
        print(f"  [{self.get_name()}]  "
              f"Test accuracy: {acc:.4f}%")
        return score

    def get_classification_report(self, X_test, y_test,
                                  labels=("Low", "Average", "High")) -> str:
        preds = self.model.predict(X_test)
        return classification_report(y_test, preds, labels=list(labels))

    def get_confusion_matrix(self, X_test, y_test,
                             labels=("Low", "Average", "High")):
        preds = self.model.predict(X_test)
        return confusion_matrix(y_test, preds, labels=list(labels))

    def predict(self, X):
        return self.model.predict(X)

    def get_loss_curve(self):
        # Decision Trees have no iterative loss — return None
        return None

    def plot_loss_curve(self):
        """Decision Trees have no iterative training loss.
        Prints a note instead of rendering an empty graph.
        """
        print(f"[{self.get_name()}] No loss curve available — "
              "Decision Trees are trained in a single pass with no iterative loss.")

    def predict_basic(self, X):
        if self._basic_model is None:
            raise RuntimeError("No basic model trained yet. Call train_basic() first.")
        return self._basic_model.predict(X)

    def predict_best(self, X):
        if self._best_model is None:
            raise RuntimeError("No best model trained yet. Call train_best() first.")
        return self._best_model.predict(X)

    def get_feature_importances(self):
        return self.model.feature_importances_

    def get_name(self) -> str:
        return "Decision Tree"
