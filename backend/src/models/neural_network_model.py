import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from typing import Tuple, Optional, Dict, List
from itertools import product
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from .base_model import BaseModel


class NeuralNetworkModel(BaseModel):

    def __init__(self,
                 hidden_layers: Tuple = (128, 64, 32),
                 activation: str = 'relu',
                 learning_rate: float = 0.001,
                 max_iterations: int = 500,
                 random_state: int = 42,
                 dropout_rate: float = 0.5,
                 weight_decay: float = 1e-4,
                 early_stopping_patience: int = 10,
                 use_batch_norm: bool = False):

        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.use_batch_norm = use_batch_norm
        self._is_fitted = False
        self._input_size = None
        self._num_classes = None
        self.loss_curve_ = []
        self.val_loss_curve_ = []

        # FIX 1: initialize snapshot and score attributes to avoid AttributeError
        self._basic_model = None
        self._best_model = None
        self._basic_label_encoder = None
        self._best_label_encoder = None
        self._train_score = None
        self._val_score = None

        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = None
        self.optimizer = None
        self._label_encoder: Optional[LabelEncoder] = None

    def _create_network(self, input_size: int, num_classes: int):
        self._input_size = input_size
        self._num_classes = num_classes

        layers = []

        # Input layer -> first hidden
        layers.append(nn.Linear(input_size, self.hidden_layers[0]))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.hidden_layers[0]))
        layers.append(self._get_activation(self.activation))
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))

        # Intermediate hidden layers
        for i in range(len(self.hidden_layers) - 1):
            layers.append(
                nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_layers[i + 1]))
            layers.append(self._get_activation(self.activation))
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

        # Output layer
        layers.append(nn.Linear(self.hidden_layers[-1], num_classes))

        self.model = nn.Sequential(*layers)
        self._init_weights()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def _get_activation(self, activation: str) -> nn.Module:
        if activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'relu':
            return nn.ReLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def _init_weights(self):
        torch.manual_seed(self.random_state)
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.constant_(module.bias, 0)

    def forward(self, X: torch.Tensor, verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Model must be created before forward pass")
        x = X.clone().float()
        logits = self.model(x)
        probabilities = torch.softmax(logits, dim=1)
        if verbose:
            print(f'Logits: {logits}')
            print(f'Probabilities: {probabilities}')
        return logits, probabilities

    def _fit(self, X_train: torch.Tensor, y_train: torch.Tensor,
             epochs: int,
             X_val: Optional[torch.Tensor] = None,
             y_val: Optional[torch.Tensor] = None):
        """Training loop with optional validation and early stopping."""
        loss_fn = nn.CrossEntropyLoss()
        self.loss_curve_ = []
        self.val_loss_curve_ = []

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # Training step
            self.model.train()
            logits, _ = self.forward(X_train)
            loss = loss_fn(logits, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_curve_.append(loss.item())

            # Validation step (if data provided)
            val_loss = None
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits, _ = self.forward(X_val)
                    val_loss = loss_fn(val_logits, y_val).item()
                self.val_loss_curve_.append(val_loss)

                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone()
                                        for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            # Optional progress printing
            if (epoch + 1) % max(1, epochs // 10) == 0:
                msg = f"  Epoch {epoch + 1}/{epochs}  loss={loss.item():.4f}"
                if val_loss is not None:
                    msg += f"  val_loss={val_loss:.4f}"
                print(msg)

        # Restore best model if early stopping was used and a better model was found
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(
                f"  Restored best model from epoch with val_loss={best_val_loss:.4f}")

        self._is_fitted = True

    def train_basic(self, X_train, y_train, X_val=None, y_val=None):
        """Basic training without hyperparameter search."""
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data cannot be empty")
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train/y_train length mismatch: {len(X_train)} vs {len(y_train)}")

        X_t = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
        y_arr = np.asarray(y_train)
        if y_arr.dtype.kind in ('U', 'O'):
            self._label_encoder = LabelEncoder()
            y_arr = self._label_encoder.fit_transform(y_arr)
        y_t = torch.from_numpy(y_arr.astype(np.int64))

        # Convert validation data if provided
        X_val_t = None
        y_val_t = None
        if X_val is not None and y_val is not None:
            X_val_t = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
            y_val_arr = np.asarray(y_val)
            if self._label_encoder is not None and y_val_arr.dtype.kind in ('U', 'O'):
                y_val_arr = self._label_encoder.transform(y_val_arr)
            y_val_t = torch.from_numpy(y_val_arr.astype(np.int64))

        if self.model is None:
            self._create_network(X_t.shape[1], len(np.unique(y_t.numpy())))

        print(f"\nTraining {self.get_name()} (basic) ...")
        try:
            self._fit(X_t, y_t, self.max_iterations, X_val_t, y_val_t)
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")

        # Snapshot the trained module as the basic model
        import copy
        self._basic_model = copy.deepcopy(self.model)
        self._basic_label_encoder = copy.deepcopy(self._label_encoder)

        # Compute and store accuracy scores
        train_preds = self.predict(X_train)
        self._train_score = accuracy_score(y_train, train_preds)
        print(f"  Train acc: {self._train_score:.4f}", end="")
        if X_val is not None and y_val is not None:
            val_preds = self.predict(X_val)
            self._val_score = accuracy_score(y_val, val_preds)
            print(f"  Val acc: {self._val_score:.4f}", end="")
        print()

    def train_best(self, X_train, y_train, X_val=None, y_val=None):
        """Hyperparameter search using grid search."""
        if X_val is None or y_val is None:
            print("No validation data provided — falling back to train_basic.")
            self.train_basic(X_train, y_train)
            return

        default_grid = {
            "hidden_layers": [(128, 64, 32), (64, 32), (256, 128, 64)],
            "learning_rate": [0.001, 0.005, 0.01],
            "max_iterations": [100, 300],
            "dropout_rate": [0.3, 0.5],
            "weight_decay": [1e-4, 1e-3],
        }
        print(f"\nTraining {self.get_name()} (best) — running grid search ...")
        result = self.grid_search(
            X_train, y_train, X_val, y_val, param_grid=default_grid)
        label_encoder_from_grid = self._label_encoder

        best = result["best_model"]
        if best is not None:
            self.hidden_layers = best.hidden_layers
            self.learning_rate = best.learning_rate
            self.max_iterations = best.max_iterations
            self.dropout_rate = best.dropout_rate
            self.weight_decay = best.weight_decay
            self.early_stopping_patience = best.early_stopping_patience
            self.use_batch_norm = best.use_batch_norm
            self.model = best.model
            self.optimizer = best.optimizer
            self.loss_curve_ = best.loss_curve_
            self.val_loss_curve_ = best.val_loss_curve_
            self._is_fitted = best._is_fitted
            self._input_size = best._input_size
            self._num_classes = best._num_classes
            self._label_encoder = best._label_encoder or label_encoder_from_grid
            print(f"  Best params adopted: hidden={self.hidden_layers}  lr={self.learning_rate}  "
                  f"dropout={self.dropout_rate}  wd={self.weight_decay}")

            import copy
            self._best_model = copy.deepcopy(self.model)
            self._best_label_encoder = copy.deepcopy(self._label_encoder)

            self._train_score = accuracy_score(
                y_train, self.predict_best(X_train))
            if X_val is not None and y_val is not None:
                self._val_score = accuracy_score(
                    y_val, self.predict_best(X_val))
            print(
                f"  Train acc: {self._train_score:.4f}  Val acc: {self._val_score:.4f}")
        else:
            print("  Grid search yielded no valid model — falling back to train_basic.")
            self.train_basic(X_train, y_train)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs: Optional[int] = None):
        """Public training method: either use grid search or direct fit."""
        if epochs is not None:
            X_t = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
            y_t = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
            if self.model is None:
                self._create_network(X_t.shape[1], len(np.unique(y_t.numpy())))

            # FIX 2: pass validation data through so early stopping works in grid search
            X_val_t = torch.from_numpy(np.asarray(
                X_val, dtype=np.float32)) if X_val is not None else None
            y_val_t = torch.from_numpy(np.asarray(
                y_val, dtype=np.int64)) if y_val is not None else None

            self._fit(X_t, y_t, epochs, X_val_t, y_val_t)
        else:
            self.train_best(X_train, y_train, X_val, y_val)

    def predict(self, X) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            raise RuntimeError(
                "Model must be trained before making predictions")
        try:
            X = torch.from_numpy(np.asarray(X, dtype=np.float32))
            with torch.no_grad():
                _, probabilities = self.forward(X)
                predictions = torch.argmax(probabilities, dim=1)
            preds = predictions.numpy()
            if self._label_encoder is not None:
                preds = self._label_encoder.inverse_transform(preds)
            return preds
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def get_name(self) -> str:
        return "Neural Network"

    def _predict_with_module(self, module: nn.Module, encoder, X) -> np.ndarray:
        """Run inference using a stored nn.Module snapshot."""
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        module.eval()
        with torch.no_grad():
            logits = module(X_t)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).numpy()
        if encoder is not None:
            preds = encoder.inverse_transform(preds)
        return preds

    def predict_basic(self, X) -> np.ndarray:
        # FIX 1: _basic_model is now always defined, so this raises RuntimeError cleanly
        if self._basic_model is None:
            raise RuntimeError(
                "No basic model trained yet. Call train_basic() first.")
        return self._predict_with_module(self._basic_model, self._basic_label_encoder, X)

    def predict_best(self, X) -> np.ndarray:
        # FIX 1: _best_model is now always defined, so this raises RuntimeError cleanly
        if self._best_model is None:
            raise RuntimeError(
                "No best model trained yet. Call train_best() first.")
        return self._predict_with_module(self._best_model, self._best_label_encoder, X)

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    def load(self, path):
        import joblib
        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)

    def save_basic(self, path):
        if self._basic_model is None:
            raise RuntimeError(
                "No basic model trained yet. Call train_basic() first.")
        import joblib
        joblib.dump({
            "module":        self._basic_model,
            "label_encoder": self._basic_label_encoder,
            "input_size":    self._input_size,
            "num_classes":   self._num_classes,
        }, path)

    def save_best(self, path):
        if self._best_model is None:
            raise RuntimeError(
                "No best model trained yet. Call train_best() first.")
        import joblib
        joblib.dump({
            "module":        self._best_model,
            "label_encoder": self._best_label_encoder,
            "input_size":    self._input_size,
            "num_classes":   self._num_classes,
        }, path)

    def _restore_snapshot(self, data: dict):
        self.model = data["module"]
        self._label_encoder = data["label_encoder"]
        self._input_size = data["input_size"]
        self._num_classes = data["num_classes"]
        self._is_fitted = True

    def load_basic(self, path):
        import joblib
        data = joblib.load(path)
        self._restore_snapshot(data)
        self._basic_model = self.model
        self._basic_label_encoder = self._label_encoder

    def load_best(self, path):
        import joblib
        data = joblib.load(path)
        self._restore_snapshot(data)
        self._best_model = self.model
        self._best_label_encoder = self._label_encoder

    def get_loss_curve(self) -> Optional[np.ndarray]:
        if not self.loss_curve_:
            return None
        return np.array(self.loss_curve_)

    def get_val_loss_curve(self) -> Optional[np.ndarray]:
        if not self.val_loss_curve_:
            return None
        return np.array(self.val_loss_curve_)

    def plot_loss_curve(self):
        """Plot training and (if available) validation loss curves over epochs."""
        import matplotlib.pyplot as plt

        train_curve = self.get_loss_curve()
        val_curve = self.get_val_loss_curve()

        if train_curve is None or len(train_curve) == 0:
            print(
                f"[{self.get_name()}] No loss curve available — train the model first.")
            return

        epochs = range(1, len(train_curve) + 1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, train_curve,
                color="#5C6BC0", linewidth=1.8, label="Train loss")
        if val_curve is not None and len(val_curve) > 0:
            # FIX 4: use a distinct color so val loss is actually visible
            ax.plot(range(1, len(val_curve) + 1), val_curve,
                    color="#EF5350", linewidth=1.4, linestyle="--",
                    alpha=0.7, label="Val loss")
        ax.set(title=f"{self.get_name()} — Training Loss",
               xlabel="Epoch", ylabel="Loss")
        ax.legend(fontsize=9)
        ax.grid(True, linewidth=0.4)
        plt.tight_layout()
        plt.show()

    def get_model_info(self) -> dict:
        return {
            "name": self.get_name(),
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "max_iterations": self.max_iterations,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "early_stopping_patience": self.early_stopping_patience,
            "use_batch_norm": self.use_batch_norm,
            "is_fitted": self._is_fitted,
            "input_size": self._input_size,
            "num_classes": self._num_classes,
            "loss_curve_length": len(self.loss_curve_),
            "val_loss_curve_length": len(self.val_loss_curve_),
        }

    def grid_search(self, X_train, y_train, X_val, y_val,
                    param_grid: Dict[str, List], verbose: bool = True) -> Dict:
        """Hyperparameter grid search with validation."""
        X_train = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
        y_train_arr = np.asarray(y_train)
        if y_train_arr.dtype.kind in ('U', 'O'):
            self._label_encoder = LabelEncoder()
            y_train_arr = self._label_encoder.fit_transform(y_train_arr)
        y_train = torch.from_numpy(y_train_arr.astype(np.int64))

        X_val = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
        y_val_arr = np.asarray(y_val)
        if self._label_encoder is not None and y_val_arr.dtype.kind in ('U', 'O'):
            y_val_arr = self._label_encoder.transform(y_val_arr)
        y_val = torch.from_numpy(y_val_arr.astype(np.int64))

        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        all_combinations = list(product(*param_values))

        if verbose:
            print(
                f"\nGrid Search: {len(all_combinations)} combinations to evaluate")
            print(f"Parameters: {param_names}\n")

        results = []
        best_score = -np.inf
        best_params = None
        best_model = None

        for idx, combo in enumerate(all_combinations):
            params = dict(zip(param_names, combo))
            if verbose:
                print(f"[{idx + 1}/{len(all_combinations)}] Testing: {params}")

            try:
                model = NeuralNetworkModel(
                    hidden_layers=params.get(
                        'hidden_layers', self.hidden_layers),
                    activation=params.get('activation', self.activation),
                    learning_rate=params.get(
                        'learning_rate', self.learning_rate),
                    max_iterations=params.get(
                        'max_iterations', self.max_iterations),
                    random_state=self.random_state,
                    dropout_rate=params.get('dropout_rate', self.dropout_rate),
                    weight_decay=params.get('weight_decay', self.weight_decay),
                    early_stopping_patience=params.get(
                        'early_stopping_patience', self.early_stopping_patience),
                    use_batch_norm=params.get(
                        'use_batch_norm', self.use_batch_norm)
                )

                # FIX 2: pass val data so early stopping actually runs per candidate
                model.train(X_train.numpy(), y_train.numpy(),
                            X_val=X_val.numpy(), y_val=y_val.numpy(),
                            epochs=params.get('max_iterations', self.max_iterations))

                with torch.no_grad():
                    val_probs = model.forward(X_val)[1]
                    val_preds = torch.argmax(val_probs, dim=1)
                    val_accuracy = (val_preds == y_val).float().mean().item()

                results.append({
                    'params': params,
                    'val_accuracy': val_accuracy,
                    'model': model
                })

                if verbose:
                    print(f"  → Validation Accuracy: {val_accuracy:.4f}\n")

                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_params = params.copy()
                    best_model = model

            except Exception as e:
                if verbose:
                    print(f"  → Error: {str(e)}\n")
                results.append({
                    'params': params,
                    'val_accuracy': -1,
                    'error': str(e)
                })

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Best Parameters: {best_params}")
            print(f"Best Validation Accuracy: {best_score:.4f}")
            print(f"{'=' * 60}\n")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_model': best_model,
            'results': results
        }
