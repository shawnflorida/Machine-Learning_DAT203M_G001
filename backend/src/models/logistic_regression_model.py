from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(
            max_iter=1000,
            C=0.5,
            random_state=42,
            solver="lbfgs",
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def get_name(self) -> str:
        return "Logistic Regression"