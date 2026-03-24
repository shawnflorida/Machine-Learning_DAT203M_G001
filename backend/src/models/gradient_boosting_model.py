from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

from .base_model import BaseModel


MAX_ROUNDS          = 5     
OPTIMAL_TRAIN_SCORE = 0.90  
VAL_GAP_TOLERANCE   = 0.05  
HYPERTUNE_N_ITER    = 20    


class GradientBoostingModel(BaseModel):

    PARAM_DISTRIBUTIONS = {
        "n_estimators":     [100, 150, 200, 250, 300],
        "learning_rate":    [0.01, 0.05, 0.08, 0.1, 0.15],
        "max_depth":        [3, 4, 5, 6],
        "min_samples_leaf": [1, 5, 10, 15],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "max_features":     ["sqrt", "log2", None],
    }

    grid_search_params = {
        "n_estimators":     [100, 150, 200, 250, 300],
        "learning_rate":    [0.01, 0.05, 0.08, 0.1, 0.15],
        "max_depth":        [3, 4, 5, 6],
        "min_samples_leaf": [1, 5, 10, 15],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "max_features":     ["sqrt", "log2", None],
    }

    def grid_search(self, X_train, y_train):
        print(f"  Running GridSearchCV for {self.get_name()} ...")
        base = GradientBoostingClassifier(random_state=42)
        grid = GridSearchCV(
            estimator=base,
            param_grid=self.grid_search_params,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        print(f"  Best params : {grid.best_params_}")

    def __init__(self):
        super().__init__()
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=10,
            subsample=0.8,
            warm_start=True,    
            random_state=42,
        )
        self._train_score: float = 0.0
        self._val_score:   float = 0.0


    def train(self, X_train, y_train, X_val=None, y_val=None):
        print(f"\n{'─' * 52}")
        print(f"  Training: {self.get_name()}")
        print(f"{'─' * 52}")

        estimator_step = 100       
        n_estimators   = self.model.n_estimators

        for round_num in range(1, MAX_ROUNDS + 1):
            self.model.n_estimators = n_estimators
            self.model.fit(X_train, y_train)

            self._train_score = accuracy_score(y_train, self.model.predict(X_train))
            print(f"  [Round {round_num}/{MAX_ROUNDS}]  "
                  f"n_estimators={n_estimators}  "
                  f"train_acc={self._train_score:.4f}")

            if self._train_score >= OPTIMAL_TRAIN_SCORE:
                print(f"Optimal training score reached ({self._train_score:.4f} "
                      f">= {OPTIMAL_TRAIN_SCORE}).  Stopping early.")
                break

            n_estimators += estimator_step
        else:
            print(f"Max training rounds ({MAX_ROUNDS}) reached.  "
                  f"Final train_acc={self._train_score:.4f}")

        if X_val is not None and y_val is not None:
            self._val_score = accuracy_score(y_val, self.model.predict(X_val))
            gap             = abs(self._train_score - self._val_score)
            print(f"\n  Validation:  val_acc={self._val_score:.4f}  "
                  f"gap={gap:.4f}  (tolerance={VAL_GAP_TOLERANCE})")

            if gap <= VAL_GAP_TOLERANCE:
                print(f" Validation gap is within tolerance.  Ready for test set.")
            else:
                print(f" Validation gap too large.  Starting hyper-tuning ...")
                self._hypertune(X_train, y_train, X_val, y_val)
        else:
            print(" No validation data provided, skipping gap check.")

        print(f"{'─' * 52}\n")

    def _hypertune(self, X_train, y_train, X_val, y_val):

        for tune_round in range(1, MAX_ROUNDS + 1):
            print(f"  [Hypertune {tune_round}/{MAX_ROUNDS}]  "
                  f"RandomizedSearchCV (n_iter={HYPERTUNE_N_ITER}) ...")

            base = GradientBoostingClassifier(random_state=42)
            search = RandomizedSearchCV(
                estimator=base,
                param_distributions=self.PARAM_DISTRIBUTIONS,
                n_iter=HYPERTUNE_N_ITER,
                cv=5,
                scoring="accuracy",
                n_jobs=1,      
                random_state=42,
                verbose=0,
            )
            search.fit(X_train, y_train)

            self.model        = search.best_estimator_
            self._train_score = accuracy_score(y_train, self.model.predict(X_train))
            self._val_score   = accuracy_score(y_val,   self.model.predict(X_val))
            gap               = abs(self._train_score - self._val_score)

            print(f"  Best params : {search.best_params_}")
            print(f"  train_acc={self._train_score:.4f}  "
                  f"val_acc={self._val_score:.4f}  gap={gap:.4f}")

            if gap <= VAL_GAP_TOLERANCE:
                print(f"Hypertune round {tune_round}: gap within tolerance.  "
                      f"Proceeding to test set.")
                return

        print(f"Hypertune exhausted ({MAX_ROUNDS} rounds).  "
              f"Proceeding to test set with best model found "
              f"(val_acc={self._val_score:.4f}).")

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

    def get_loss_curve(self):
        return self.model.train_score_

    def get_feature_importances(self):
        return self.model.feature_importances_

    def get_name(self) -> str:
        return "Gradient Boosting"
