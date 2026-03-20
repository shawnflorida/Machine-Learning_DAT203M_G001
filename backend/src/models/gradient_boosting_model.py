import copy
from itertools import product
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from .base_model import BaseModel


MAX_ROUNDS = 5
OPTIMAL_TRAIN_SCORE = 1.0
VAL_GAP_TOLERANCE = 0.05
HYPERTUNE_N_ITER = 3

class GradientBoostingModel(BaseModel):
        
    grid_search_params = {
        "n_estimators":     [100, 200],
        "learning_rate":    [0.05],
        "max_depth":        [3, 5],
        "min_samples_leaf": [1],
        "subsample":        [0.8],
        "max_features":     ["sqrt"],
    }

    def __init__(self):
        super().__init__()
        self.model = GradientBoostingClassifier(random_state=42)
        self._train_score: float = 0.0
        self._val_score:   float = 0.0
        self._basic_model = None
        self._best_model  = None
        self.improved_model = None

  
    def grid_search(self, X_train, y_train, X_val, y_val):
        print(f"  Running manual grid search for {self.get_name()} ...")
        param_names = list(self.grid_search_params.keys())
        param_values = list(self.grid_search_params.values())
        combos = list(product(*param_values))
        print(f"  {len(combos)} combinations to evaluate")

        best_val_score = -1.0
        best_params = {}
        best_model = None

        for combo in combos:
            params = dict(zip(param_names, combo))
            candidate = GradientBoostingClassifier(random_state=42, **params)
            candidate.fit(X_train, y_train)
            val_acc = f1_score(y_val, candidate.predict(X_val), average='weighted', zero_division=0)
            print(f"  {params}  val_f1={val_acc:.4f}")
            if val_acc > best_val_score:
                best_val_score = val_acc
                best_params = params
                best_model = candidate

        self.model = best_model
        print(f"  Best params : {best_params}  (val_f1={best_val_score:.4f})")
        


    def train_basic(self, X_train, y_train, X_val, y_val):
        import copy
        self.model.fit(X_train, y_train)
        self._train_score = f1_score(y_train, self.model.predict(X_train), average='weighted', zero_division=0)
        self._val_score = f1_score(y_val, self.model.predict(X_val), average='weighted', zero_division=0)
        print(f"Trained: {self.get_name()}")
        print(f"Train F1: {self._train_score:.4f}  Val F1: {self._val_score:.4f}")
        self._basic_model = copy.deepcopy(self.model)

    def train_best(self, X_train, y_train, X_val=None, y_val=None):
        print(f"\n{'─' * 52}")
        print(f"  Training: {self.get_name()} (grid search)")
        print(f"{'─' * 52}")

        self.grid_search(X_train, y_train, X_val, y_val)

        self._train_score = f1_score(y_train, self.model.predict(X_train), average='weighted', zero_division=0)
        if X_val is not None and y_val is not None:
            self._val_score = f1_score(y_val, self.model.predict(X_val), average='weighted', zero_division=0)
        print(f"  Train F1: {self._train_score:.4f}  Val F1: {self._val_score:.4f}")
        self._best_model = copy.deepcopy(self.model)

        # if X_val is not None and y_val is not None:
        #     self._val_score = accuracy_score(y_val, self.model.predict(X_val))
        #     gap = abs(self._train_score - self._val_score)
        #     print(f"  Val acc:   {self._val_score:.4f}  "
        #           f"gap={gap:.4f}  (tolerance={VAL_GAP_TOLERANCE})")
        #     if gap <= VAL_GAP_TOLERANCE:
        #         print(f"  Validation gap is within tolerance.  Ready for test set.")
        #     else:
        #         print(f"  Validation gap too large.  Starting hyper-tuning ...")
        #         self._hypertune(X_train, y_train, X_val, y_val)
        # else:
        #     print("  No validation data provided, skipping gap check.")

        print(f"{'─' * 52}\n")

   

    def val(self, X_val, y_val) -> float:
        score = accuracy_score(y_val, self.model.predict(X_val))
        print(f"  [{self.get_name()}]  val_acc={score:.4f}")
        return score

    def test(self, X_test, y_test) -> float:
        score = accuracy_score(y_test, self.model.predict(X_test))
        print(f"  [{self.get_name()}]  test_acc={score:.4f}")
        return score

    def predict(self, X):
        return self.model.predict(X)

    def predict_basic(self, X):
        if self._basic_model is None:
            raise RuntimeError("No basic model trained yet. Call train_basic() first.")
        return self._basic_model.predict(X)

    def predict_best(self, X):
        if self._best_model is None:
            raise RuntimeError("No best model trained yet. Call train_best() first.")
        return self._best_model.predict(X)

    def get_loss_curve(self):
        return self.model.train_score_

    def plot_loss_curve(self):
        """Plot the training deviance (loss) curve across boosting iterations."""
        import matplotlib.pyplot as plt

        curve = self.get_loss_curve()
        if curve is None or len(curve) == 0:
            print(f"[{self.get_name()}] No loss curve available — train the model first.")
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(curve) + 1), curve,
                color="#EF5350", linewidth=1.8, label="Train deviance")
        ax.set(title=f"{self.get_name()} — Training Loss (Deviance)",
               xlabel="Boosting Iteration", ylabel="Deviance")
        ax.legend(fontsize=9)
        ax.grid(True, linewidth=0.4)
        plt.tight_layout()
        plt.show()

    def get_feature_importances(self):
        return self.model.feature_importances_

    def get_name(self) -> str:
        return "Gradient Boosting"
