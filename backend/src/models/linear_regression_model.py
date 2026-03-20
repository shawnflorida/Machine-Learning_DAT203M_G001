from sklearn.linear_model import LinearRegression

from .base_model import BaseModel


class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def get_name(self) -> str:
        return "Linear Regression"

    def get_coefficients(self, feature_names):
        import pandas as pd

        return pd.Series(self.model.coef_, index=feature_names)

class StochasticDescentModel(Model):
    
    def __init__(self):
        super().__init__()

    