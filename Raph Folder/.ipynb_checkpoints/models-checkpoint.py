"""
models.py
---------
All four ML model classes for the DAT203M stress-prediction project.

Each class follows the same interface shown in the project scaffold:
    __init__   – stores hyperparameters, creates a Visualisation instance
    clean      – dataset-specific cleaning / imputation / outlier handling
    data_engineering – feature selection, encoding, scaling, binning
    train      – fits the model; populates self.model and history attributes
    visualize_model – calls visualizer.plot() or custom charts
    return_model    – returns the trained model object

Import with:
    from models import SoftmaxRegressionModel, KNNModel, DecisionTreeModel, NeuralNetworkModel
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from visual import Visualisation


# ── shared constants ─────────────────────────────────────────────────────────

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

NUMERICAL_FEATURES = [
    "hours_work", "social_media_use", "rent", "friends_count",
    "dates", "standard_drinks", "countries", "semesters",
    "commute", "data_interest", "mark_goal", "hours_studying",
]
CATEGORICAL_FEATURES = [
    "relationship_status", "drug_use_ans", "student_type",
    "lecture_mode", "study_type", "learner_style",
]
TARGET = "stress"


# ── shared helpers ────────────────────────────────────────────────────────────

def _winsorize(series, lower_pct=1, upper_pct=99):
    lo = np.percentile(series.dropna(), lower_pct)
    hi = np.percentile(series.dropna(), upper_pct)
    return series.clip(lo, hi)


def _bin_stress(val):
    if val <= 3:
        return 0      # Low
    elif val <= 6:
        return 1      # Average
    return 2          # High


def _one_hot(y, n_classes=3):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1
    return Y


def _compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts)
    return dict(zip(classes, weights))


def _softmax(Z):
    Z_s = Z - np.max(Z, axis=1, keepdims=True)
    e = np.exp(Z_s)
    return e / e.sum(axis=1, keepdims=True)


def _relu(Z):
    return np.maximum(0, Z)


def _relu_deriv(Z):
    return (Z > 0).astype(float)


def _cross_entropy(Y_pred, Y_true, class_w, y_labels):
    eps = 1e-9
    sw = np.array([class_w[l] for l in y_labels])
    log_p = -np.sum(Y_true * np.log(Y_pred + eps), axis=1)
    return np.sum(sw * log_p) / np.sum(sw)


# ══════════════════════════════════════════════════════════════════════════════
# BASE CLASS  (shared clean / data_engineering logic)
# ══════════════════════════════════════════════════════════════════════════════

class _BaseModel:
    """
    Shared data-preparation pipeline inherited by all four model classes.
    Subclasses only need to implement train() and visualize_model().
    """

    def __init__(self):
        self.visualizer   = Visualisation()
        self.scaler       = StandardScaler()
        self.label_encoders: dict = {}
        self.df_clean     = None       # set after clean()
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.model        = None       # set after train()

    # ── clean ─────────────────────────────────────────────────────────────────

    def clean(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        1. Filter to consenting students only.
        2. Select model features + target.
        3. Drop rows with missing target.
        4. Impute: numerical → median, categorical → mode.
        5. Winsorise numerical columns (1st–99th percentile).
        6. Bin stress into 3 classes.

        Stores result in self.df_clean and returns it.
        """
        # Filter consent
        df = df_raw[df_raw["consent"] ==
                    "I consent to take part in the study"].copy()
        df.reset_index(drop=True, inplace=True)

        # Select columns
        keep = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET]
        df = df[keep].copy()

        # Drop missing target
        df.dropna(subset=[TARGET], inplace=True)

        # Impute numerical
        for col in NUMERICAL_FEATURES:
            df[col] = df[col].fillna(df[col].median())

        # Impute categorical
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].fillna(df[col].mode()[0])

        # Winsorise
        for col in NUMERICAL_FEATURES:
            df[col] = _winsorize(df[col])

        # Bin target
        df["stress_class"] = df[TARGET].apply(_bin_stress)

        self.df_clean = df
        print(f"[clean] Dataset shape after cleaning: {df.shape}")
        print(f"[clean] Class distribution:\n{df['stress_class'].value_counts().sort_index()}")
        return df

    # ── data_engineering ──────────────────────────────────────────────────────

    def data_engineering(self,
                         test_size: float = 0.15,
                         val_size:  float = 0.15,
                         scale: bool = True) -> tuple:
        """
        1. Label-encode categorical features.
        2. Assemble feature matrix X and label vector y.
        3. Stratified train / val / test split (default 70 / 15 / 15).
        4. StandardScaler fit on train only (no data leakage).

        Returns (X_train, X_val, X_test, y_train, y_val, y_test).
        Stores arrays on self for use inside train().
        """
        if self.df_clean is None:
            raise RuntimeError("Call clean() before data_engineering().")

        df = self.df_clean.copy()

        # Label-encode categoricals
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        enc_cols   = [c + "_enc" for c in CATEGORICAL_FEATURES]
        all_cols   = NUMERICAL_FEATURES + enc_cols
        self._feature_cols = all_cols

        X = df[all_cols].values.astype(float)
        y = df["stress_class"].values.astype(int)

        # Split off test
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=RANDOM_SEED, stratify=y)

        # Split train / val from remaining
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_ratio,
            random_state=RANDOM_SEED, stratify=y_tmp)

        # Scale (fit on train only)
        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_val   = self.scaler.transform(X_val)
            X_test  = self.scaler.transform(X_test)

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        print(f"[data_engineering] Train: {len(y_train)}  "
              f"Val: {len(y_val)}  Test: {len(y_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ── return_model ──────────────────────────────────────────────────────────

    def return_model(self):
        """Return the trained model object."""
        if self.model is None:
            raise RuntimeError("Call train() before return_model().")
        return self.model

    # ── convenience: evaluate ─────────────────────────────────────────────────

    def evaluate(self, X, y_true, split_label=""):
        """Return accuracy, macro-F1, weighted-F1, precision, recall dict."""
        y_pred = self.model.predict(X)
        return {
            "split":            split_label,
            "accuracy":         round(accuracy_score(y_true, y_pred),   4),
            "macro_f1":         round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4),
            "weighted_f1":      round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "macro_precision":  round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "macro_recall":     round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        }



# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — k-NEAREST NEIGHBOURS  (from scratch)
# ══════════════════════════════════════════════════════════════════════════════

class _KNNCore:
    """Inner estimator for k-NN."""

    def __init__(self, X_train, y_train, k):
        self._X = X_train
        self._y = y_train
        self.k  = k

    def _euclidean(self, X):
        # ||a-b||² = ||a||² + ||b||² - 2 a·b   (vectorised, memory-efficient)
        sq_a = (self._X ** 2).sum(axis=1)           # (n_train,)
        sq_b = (X      ** 2).sum(axis=1)            # (n_query,)
        dot  = X @ self._X.T                        # (n_query, n_train)
        dists = sq_b[:, None] + sq_a[None, :] - 2 * dot
        return np.sqrt(np.clip(dists, 0, None))

    def predict(self, X):
        dists = self._euclidean(X)                  # (n_query, n_train)
        nn_idx = np.argpartition(dists, self.k, axis=1)[:, :self.k]
        nn_labels = self._y[nn_idx]                 # (n_query, k)
        # majority vote
        preds = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=3).argmax(), 1, nn_labels)
        return preds

    def predict_proba(self, X):
        dists = self._euclidean(X)
        nn_idx = np.argpartition(dists, self.k, axis=1)[:, :self.k]
        nn_labels = self._y[nn_idx]
        proba = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=3) / self.k, 1, nn_labels)
        return proba


class KNNModel(_BaseModel):
    """
    k-Nearest Neighbours classifier implemented from scratch.
    Hyperparameter: k (number of neighbours) — tuned via grid search.
    Distance metric: Euclidean (L2).
    No gradient descent — completely different algorithmic family from MLP/Softmax.
    """

    def __init__(self, k_range=None):
        super().__init__()
        self.k_range    = k_range if k_range else list(range(1, 32, 2))
        self.best_k     = None
        self._k_accs    = []   # val accuracies per k (for plot)

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self):
        """Grid-search over k; refit best model on full train set."""
        if self.X_train is None:
            raise RuntimeError("Call data_engineering() before train().")

        print("[KNNModel.train] Searching over k =", self.k_range)
        best_acc = -1
        self._k_accs = []

        for k in self.k_range:
            candidate = _KNNCore(self.X_train, self.y_train, k)
            acc = accuracy_score(self.y_val, candidate.predict(self.X_val))
            self._k_accs.append(acc)
            if acc > best_acc:
                best_acc  = acc
                self.best_k = k

        self.model = _KNNCore(self.X_train, self.y_train, self.best_k)
        print(f"[KNNModel.train] Best k = {self.best_k}  "
              f"Val accuracy = {best_acc:.4f}")
        return self

    # ── visualize_model ───────────────────────────────────────────────────────

    def visualize_model(self):
        """k search curve + validation confusion matrix."""
        self.visualizer.plot_knn_k_search(self.k_range, self._k_accs)
        y_pred = self.model.predict(self.X_val)
        self.visualizer.plot_confusion_matrix(
            self.y_val, y_pred, f"k-NN (k={self.best_k}, Validation)")

    def correlation(self):
        """
        Feature-relevance proxy for k-NN:
        plots the mean absolute difference in each feature between
        correctly and incorrectly classified validation samples.
        (k-NN has no weights, so this is the closest analogue to feature importance.)
        """
        if self.model is None:
            raise RuntimeError("Call train() before correlation().")

        y_pred  = self.model.predict(self.X_val)
        correct = (y_pred == self.y_val)
        feat_names = self._feature_cols

        diff = np.abs(self.X_val[correct].mean(0) -
                      self.X_val[~correct].mean(0))

        order = np.argsort(diff)[::-1]
        plt.figure(figsize=(10, 5))
        plt.barh([feat_names[i] for i in order], diff[order],
                 color="darkorange", edgecolor="black", alpha=0.85)
        plt.xlabel("Mean |feature difference| (correct vs incorrect)")
        plt.title("k-NN Feature Relevance Proxy",
                  fontweight="bold")
        plt.tight_layout()
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — DECISION TREE  (from scratch, CART-style Gini impurity)
# ══════════════════════════════════════════════════════════════════════════════

class _DTNode:
    """A node in the decision tree."""
    __slots__ = ("feature", "threshold", "left", "right",
                 "is_leaf", "prediction", "gini", "n_samples")

    def __init__(self):
        self.feature   = None
        self.threshold = None
        self.left      = None
        self.right     = None
        self.is_leaf   = False
        self.prediction = None
        self.gini      = None
        self.n_samples = None


class _DTCore:
    """Decision Tree using Gini impurity (CART), implemented from scratch."""

    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, n_classes=3):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.n_classes         = n_classes
        self.root              = None
        self.feature_importances_ = None

    # ── Gini helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=3)
        probs  = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        best_gain = -np.inf
        best_feat = best_thresh = None
        G_parent  = self._gini(y)
        n         = len(y)

        for feat in range(X.shape[1]):
            vals    = X[:, feat]
            threshs = np.unique(vals)

            for thresh in threshs:
                left_mask  = vals <= thresh
                right_mask = ~left_mask
                n_l, n_r   = left_mask.sum(), right_mask.sum()

                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue

                gain = G_parent - (n_l / n * self._gini(y[left_mask]) +
                                   n_r / n * self._gini(y[right_mask]))
                if gain > best_gain:
                    best_gain  = gain
                    best_feat  = feat
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(self, X, y, depth):
        node = _DTNode()
        node.n_samples = len(y)
        node.gini      = self._gini(y)

        # Leaf conditions
        if (len(y) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth) or
                node.gini == 0.0):
            node.is_leaf    = True
            node.prediction = np.bincount(y, minlength=self.n_classes).argmax()
            return node

        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain <= 0:
            node.is_leaf    = True
            node.prediction = np.bincount(y, minlength=self.n_classes).argmax()
            return node

        node.feature   = feat
        node.threshold = thresh
        mask           = X[:, feat] <= thresh
        node.left      = self._build(X[mask],  y[mask],  depth + 1)
        node.right     = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        self.root = self._build(X, y, 0)
        # compute feature importances (by weighted Gini reduction)
        n_feat = X.shape[1]
        imps   = np.zeros(n_feat)
        self._accumulate_importance(self.root, len(y), imps)
        total = imps.sum()
        self.feature_importances_ = imps / total if total > 0 else imps
        return self

    def _accumulate_importance(self, node, n_total, imps):
        if node is None or node.is_leaf:
            return
        n_node = node.n_samples
        feat   = node.feature
        g_l = node.left.gini  if node.left  else 0
        g_r = node.right.gini if node.right else 0
        n_l = node.left.n_samples  if node.left  else 0
        n_r = node.right.n_samples if node.right else 0

        reduction = (n_node / n_total) * (
            node.gini - (n_l / n_node * g_l + n_r / n_node * g_r)
        )
        imps[feat] += reduction
        self._accumulate_importance(node.left,  n_total, imps)
        self._accumulate_importance(node.right, n_total, imps)

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def predict_proba(self, X):
        preds = self.predict(X)
        proba = np.zeros((len(preds), self.n_classes))
        proba[np.arange(len(preds)), preds] = 1.0
        return proba


class DecisionTreeModel(_BaseModel):
    """
    CART Decision Tree trained with Gini impurity, implemented from scratch.
    Hyperparameter: max_depth — tuned via grid search.
    No gradient descent — completely different from MLP/Softmax.
    """

    def __init__(self, depth_range=None, min_samples_split=5,
                 min_samples_leaf=3):
        super().__init__()
        self.depth_range       = depth_range if depth_range else list(range(2, 21))
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.best_depth        = None
        self._depth_accs       = []

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self):
        """Grid-search over max_depth; refit best model."""
        if self.X_train is None:
            raise RuntimeError("Call data_engineering() before train().")

        print("[DecisionTreeModel.train] Searching over depths:", self.depth_range)
        best_acc = -1
        self._depth_accs = []

        for depth in self.depth_range:
            candidate = _DTCore(max_depth=depth,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf)
            candidate.fit(self.X_train, self.y_train)
            acc = accuracy_score(self.y_val, candidate.predict(self.X_val))
            self._depth_accs.append(acc)
            if acc > best_acc:
                best_acc        = acc
                self.best_depth = depth

        # Refit best
        self.model = _DTCore(max_depth=self.best_depth,
                             min_samples_split=self.min_samples_split,
                             min_samples_leaf=self.min_samples_leaf)
        self.model.fit(self.X_train, self.y_train)
        print(f"[DecisionTreeModel.train] Best depth = {self.best_depth}  "
              f"Val accuracy = {best_acc:.4f}")
        return self

    # ── visualize_model ───────────────────────────────────────────────────────

    def visualize_model(self):
        """Depth search curve + validation confusion matrix."""
        self.visualizer.plot_dt_depth_search(self.depth_range, self._depth_accs)
        y_pred = self.model.predict(self.X_val)
        self.visualizer.plot_confusion_matrix(
            self.y_val, y_pred,
            f"Decision Tree (depth={self.best_depth}, Validation)")

    def correlation(self):
        """Bar chart of Gini-based feature importances."""
        if self.model is None:
            raise RuntimeError("Call train() before correlation().")

        imps      = self.model.feature_importances_
        feat_names = self._feature_cols
        order     = np.argsort(imps)[::-1]

        plt.figure(figsize=(10, 5))
        plt.barh([feat_names[i] for i in order], imps[order],
                 color="mediumseagreen", edgecolor="black", alpha=0.85)
        plt.xlabel("Gini-based Feature Importance")
        plt.title(f"Decision Tree Feature Importances (depth={self.best_depth})",
                  fontweight="bold")
        plt.tight_layout()
        plt.show()
