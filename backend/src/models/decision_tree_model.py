import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .base_model import BaseModel

OPTIMAL_TRAIN_SCORE = 1.00
VAL_GAP_TOLERANCE   = 0.05
MAX_ROUNDS          = 5

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


    @staticmethod
    def compute_accuracy(predictions, actual) -> float:

        return float(np.mean(np.array(predictions) == np.array(actual)) * 100)

    def train(self, X_train, y_train, X_val=None, y_val=None):

        print(f"\n{'─' * 52}")
        print(f"  Training: {self.get_name()}")
        print(f"{'─' * 52}")

        # Phase 1: Baseline → observe overfitting, then regularize
        print("\n  [Phase 1] Baseline overfitting check + depth sweep")
        print(f"  {'depth':>9} | {'train_acc':>9} | {'val_acc':>8} | {'gap':>8}")
        print(f"  {'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}")

        depth_steps = [None, 5, 7, 10, 13, None]   # None first = unconstrained baseline
        best_round_model = None

        for round_num, max_depth in enumerate(depth_steps[:MAX_ROUNDS + 1], start=0):
            candidate = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=self.model.min_samples_split,
                min_samples_leaf=self.model.min_samples_leaf,
                random_state=42,
            )
            candidate.fit(X_train, y_train)

            tr = accuracy_score(y_train, candidate.predict(X_train))
            va = accuracy_score(y_val,   candidate.predict(X_val)) if X_val is not None else float("nan")
            gap = abs(tr - va) if X_val is not None else float("nan")

            depth_label = str(max_depth) if max_depth is not None else "unlimited"

            if round_num == 0:
                print(f"  {'baseline':>9} | {tr:>9.4f} | {va:>8.4f} | {gap:>8.4f}  ← overfitting baseline")
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
            self._val_score = accuracy_score(y_val, self.model.predict(X_val))
            gap = abs(self._train_score - self._val_score)

            print(f"\n  [Phase 2] Validation check")
            print(f"  train_acc = {self._train_score:.4f} | "
                  f"val_acc = {self._val_score:.4f} | "
                  f"gap = {gap:.4f} (tolerance = {VAL_GAP_TOLERANCE})")

            if gap <= VAL_GAP_TOLERANCE:
                print(f"  Gap within tolerance. Proceeding to test set.")
            else:
                print(f"  Gap too large. Starting RandomizedSearchCV hypertuning ...")
                self._hypertune(X_train, y_train, X_val, y_val)
        else:
            print("\n  No validation data provided — skipping gap check.")

        print(f"\n{'─' * 52}\n")

    def _hypertune(self, X_train, y_train, X_val, y_val):
        
        print(f"\n  [Phase 3] RandomizedSearchCV (n_iter=50, cv=5)")

        for tune_round in range(1, MAX_ROUNDS + 1):
            print(f"\n  [Hypertune {tune_round}/{MAX_ROUNDS}]  Searching ...")

            base = DecisionTreeClassifier(random_state=42)
            rsc = RandomizedSearchCV(
                estimator=base,
                param_distributions=self.HYPERPARAMETERS,
                n_iter=50,
                cv=5,
                scoring="accuracy",
                random_state=42,
                n_jobs=-1,
            )
            rsc.fit(X_train, y_train)

            self.model        = rsc.best_estimator_
            self._train_score = accuracy_score(y_train, self.model.predict(X_train))
            self._val_score   = accuracy_score(y_val,   self.model.predict(X_val))
            gap               = abs(self._train_score - self._val_score)

            print(f"  Best hyperparameters : {rsc.best_params_}")
            print(f"  CV best score        : {rsc.best_score_:.4f}")
            print(f"  train_acc={self._train_score:.4f}  "
                  f"val_acc={self._val_score:.4f}  gap={gap:.4f}")
            print(f"  Tree depth={self.model.get_depth()}  "
                  f"n_leaves={self.model.get_n_leaves()}")

            if gap <= VAL_GAP_TOLERANCE:
                print(f"Hypertune round {tune_round}: gap within tolerance. "
                      f"Proceeding to test set.")
                return

        print(f"\n Hypertuning exhausted ({MAX_ROUNDS} rounds). "
              f"Proceeding to test set with best model found "
              f"(val_acc = {self._val_score:.4f}).")

    def val(self, X_val, y_val) -> float:
        preds = self.model.predict(X_val)
        acc   = self.compute_accuracy(preds, y_val)
        score = acc / 100
        print(f"  [{self.get_name()}]  "
              f"Validation accuracy: {acc:.4f}%")
        self._val_score = score
        return score

    def test(self, X_test, y_test) -> float:
        preds = self.model.predict(X_test)
        acc   = self.compute_accuracy(preds, y_test)
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

    def get_feature_importances(self):
        return self.model.feature_importances_

    def get_name(self) -> str:
        return "Decision Tree"