from sklearn.ensemble import GradientBoostingRegressor

from .base_model import BaseModel


class GradientBoostingModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def get_name(self) -> str:
        return "Gradient Boosting"
