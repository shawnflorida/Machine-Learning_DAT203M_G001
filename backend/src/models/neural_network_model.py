import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from typing import Tuple, Optional, Dict, List
from itertools import product

from .base_model import BaseModel


class NeuralNetworkModel(BaseModel):
    """
    PyTorch-based Neural Network classifier with multi-layer support.
    
    Supports configurable architectures with various activation functions
    and manual/automatic forward propagation.
    """
    
    def __init__(self, hidden_layers: Tuple = (128, 64, 32), activation: str = 'relu', 
                 learning_rate: float = 0.001, max_iterations: int = 500, random_state: int = 42):
        """
        Initialize Neural Network model.
        
        Args:
            hidden_layers: Tuple of hidden layer sizes
            activation: Activation function ('sigmoid', 'tanh', 'relu')
            learning_rate: Initial learning rate for optimizer
            max_iterations: Maximum number of training iterations
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.random_state = random_state
        self._is_fitted = False
        self._input_size = None
        self._num_classes = None
        self.loss_curve_ = []
        
        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.optimizer = None

    def _create_network(self, input_size: int, num_classes: int):
        """
        Create the network architecture dynamically.
        
        Args:
            input_size: Number of input features
            num_classes: Number of output classes
        """
        self._input_size = input_size
        self._num_classes = num_classes
        
        layers = []
        
        # First hidden layer
        layers.append(nn.Linear(input_size, self.hidden_layers[0]))
        layers.append(self._get_activation(self.activation))
        
        # Intermediate hidden layers
        for i in range(len(self.hidden_layers) - 1):
            layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            layers.append(self._get_activation(self.activation))
        
        # Output layer
        layers.append(nn.Linear(self.hidden_layers[-1], num_classes))
        
        self.model = nn.Sequential(*layers)
        self._init_weights()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module."""
        if activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'relu':
            return nn.ReLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def _init_weights(self):
        """Initialize weights from normal distribution and bias to zeros."""
        torch.manual_seed(self.random_state)
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.constant_(module.bias, 0)

    def forward_manual(self, X: torch.Tensor, verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Manual forward propagation implementation.
        
        Args:
            X: Input tensor
            verbose: Whether to print layer outputs
            
        Returns:
            Tuple of (logits, probabilities)
        """
        x = X.clone().float()
        
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                x = torch.matmul(x, layer.weight.t()) + layer.bias
            else:
                x = layer(x)
            
            if verbose:
                print(f'Output of layer {i}: {x.shape}')
                print(x, '\n')
        
        # Apply softmax for probabilities
        probabilities = torch.softmax(x, dim=1)
        
        return x, probabilities

    def forward(self, X: torch.Tensor, verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PyTorch forward propagation.
        
        Args:
            X: Input tensor
            verbose: Whether to print layer outputs
            
        Returns:
            Tuple of (logits, probabilities)
        """
        if self.model is None:
            raise RuntimeError("Model must be created before forward pass")
        
        x = X.clone().float()
        logits = self.model(x)
        probabilities = torch.softmax(logits, dim=1)
        
        if verbose:
            print(f'Logits: {logits}')
            print(f'Probabilities: {probabilities}')
        
        return logits, probabilities

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs: Optional[int] = None):
        """
        Train the neural network model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs (uses max_iterations if not specified)
            
        Raises:
            ValueError: If input data is invalid
        """
        # Validate input data
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data cannot be empty")
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}")
        
        # Convert to tensors
        X_train = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
        y_train = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
        
        # Create network on first training call
        if self.model is None:
            num_classes = len(np.unique(y_train.numpy()))
            self._create_network(X_train.shape[1], num_classes)
        
        epochs = epochs or self.max_iterations
        loss_fn = nn.CrossEntropyLoss()
        self.loss_curve_ = []
        
        try:
            for epoch in range(epochs):
                # Forward pass
                logits, _ = self.forward(X_train)
                loss = loss_fn(logits, y_train)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.loss_curve_.append(loss.item())
                
                if (epoch + 1) % max(1, epochs // 10) == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
            self._is_fitted = True
        except Exception as e:
            raise RuntimeError(f"Model training failed: {str(e)}")

    def predict(self, X) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted class indices
            
        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be trained before making predictions")
        
        try:
            X = torch.from_numpy(np.asarray(X, dtype=np.float32))
            with torch.no_grad():
                _, probabilities = self.forward(X)
                predictions = torch.argmax(probabilities, dim=1)
            
            return predictions.numpy()
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def get_name(self) -> str:
        """Get model name."""
        return "Neural Network"

    def get_loss_curve(self) -> Optional[np.ndarray]:
        """
        Get training loss curve.
        
        Returns:
            Loss values during training, or None if not available
        """
        if not self.loss_curve_:
            return None
        return np.array(self.loss_curve_)
    
    def get_model_info(self) -> dict:
        """Get model configuration and training info."""
        return {
            "name": self.get_name(),
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "max_iterations": self.max_iterations,
            "is_fitted": self._is_fitted,
            "input_size": self._input_size,
            "num_classes": self._num_classes,
            "loss_curve_length": len(self.loss_curve_),
        }

    def grid_search(self, X_train, y_train, X_val, y_val, 
                    param_grid: Dict[str, List], verbose: bool = True) -> Dict:
        """
        Perform grid search over hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            param_grid: Dictionary mapping parameter names to lists of values to try
                Example: {
                    'hidden_layers': [(64, 32), (128, 64, 32), (256, 128)],
                    'activation': ['relu', 'tanh', 'sigmoid'],
                    'learning_rate': [0.001, 0.01, 0.1],
                    'max_iterations': [100, 500]
                }
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing:
                'best_params': Best hyperparameters found
                'best_score': Best validation accuracy
                'results': List of all results with params and score
        """
        # Convert to tensors
        X_train = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
        y_train = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
        X_val = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
        y_val = torch.from_numpy(np.asarray(y_val, dtype=np.int64))
        
        # Get all parameter combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        all_combinations = list(product(*param_values))
        
        if verbose:
            print(f"\nGrid Search: {len(all_combinations)} combinations to evaluate")
            print(f"Parameters: {param_names}\n")
        
        results = []
        best_score = -np.inf
        best_params = None
        best_model_state = None
        
        for idx, combo in enumerate(all_combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, combo))
            
            if verbose:
                print(f"[{idx+1}/{len(all_combinations)}] Testing: {params}")
            
            # Create new model with these parameters
            try:
                model = NeuralNetworkModel(
                    hidden_layers=params.get('hidden_layers', self.hidden_layers),
                    activation=params.get('activation', self.activation),
                    learning_rate=params.get('learning_rate', self.learning_rate),
                    max_iterations=params.get('max_iterations', self.max_iterations),
                    random_state=self.random_state
                )
                
                # Train model
                model.train(X_train.numpy(), y_train.numpy(), 
                           epochs=params.get('max_iterations', self.max_iterations))
                
                # Evaluate on validation set
                with torch.no_grad():
                    val_logits, val_probs = model.forward(X_val)
                    val_preds = torch.argmax(val_probs, dim=1)
                    val_accuracy = (val_preds == y_val).float().mean().item()
                
                results.append({
                    'params': params,
                    'val_accuracy': val_accuracy,
                    'model': model
                })
                
                if verbose:
                    print(f"  → Validation Accuracy: {val_accuracy:.4f}\n")
                
                # Track best
                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_params = params.copy()
                    best_model_state = model.model.state_dict()
            
            except Exception as e:
                if verbose:
                    print(f"  → Error: {str(e)}\n")
                results.append({
                    'params': params,
                    'val_accuracy': -1,
                    'error': str(e)
                })
        
        # Update current model with best params
        if best_params is not None:
            self.hidden_layers = best_params.get('hidden_layers', self.hidden_layers)
            self.activation = best_params.get('activation', self.activation)
            self.learning_rate = best_params.get('learning_rate', self.learning_rate)
            self.max_iterations = best_params.get('max_iterations', self.max_iterations)
            
            # Retrain with best params on full training set
            self.train(X_train.numpy(), y_train.numpy(), 
                      epochs=self.max_iterations)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Best Parameters: {best_params}")
            print(f"Best Validation Accuracy: {best_score:.4f}")
            print(f"{'='*60}\n")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'results': results
        }
