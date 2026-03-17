from sklearn.neural_network import MLPRegressor

from .base_model import BaseModel


class NeuralNetworkModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def get_name(self) -> str:
        return "Neural Network"

    def get_loss_curve(self):
        return self.model.loss_curve_
