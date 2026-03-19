from turtle import pd
from sklearn.linear_model import LogisticRegression
from backend.src.architecture.visualizer import Visualizer
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

    def print_model_info(self):
        v = Visualizer()
   

    grid_search_params = {
        "C": [0.01, 0.1, 1, 10],
        "options": ["lbfgs", "liblinear", "sag"],
        "solver": ["lbfgs", "liblinear", "sag"],
    }
    
    def grid_search(self, X_train, y_train, X_val, y_val):
        best_score = 0
        best_params = {}
        validation_set = pd.DataFrame(X_val, columns=[f"feature_{i}" for i in range(X_val.shape[1])])
        for C in self.grid_search_params["C"]:
            for solver in self.grid_search_params["solver"]:
                model = LogisticRegression(max_iter=1000, C=C, random_state=42, solver=solver)
                model.fit(X_train, y_train)
                val_score = model.score(X_val, y_val)
                print(f"C={C}, Solver={solver}: Validation Accuracy={val_score:.4f}")
                if val_score > best_score:
                    best_score = val_score
                    best_params = {"C": C, "solver": solver}
                    v = Visualizer()
                    v.plot_confusion_matrix(y_val, model.predict(X_val), title=f"Logistic
    Regression Confusion Matrix (C={C}, Solver={solver})")
                    folder_location = "logistic_regression_results"
                    v.save_plot(folder_location, f"confusion_matrix_C{C}_solver{solver}.png")
        print(f"Best Params: {best_params}, Best Validation Accuracy: {best_score:.4f}")
    
    
    def return_best_model(self, X_train, y_train, X_val, y_val):
        best_score = 0
        best_model = None
        for C in self.grid_search_params["C"]:
            for solver in self.grid_search_params["solver"]:
                model = LogisticRegression(max_iter=1000, C=C, random_state=42, solver=solver)
                model.fit(X_train, y_train)
                val_score = model.score(X_val, y_val)
                if val_score > best_score:
                    best_score = val_score
                    best_model = model
        save_path = "logistic_regression_results/best_model.pkl"
        return best_model
    
    1. grid search 4x4x4 = 64x 4 x 5 = 1280 models trained and evaluated on validation set
    2. model1 = metrics(gridsearch)
    3. folder/images/confusaion_matrix_C{C}_solver{solver}.png
    4. l.return_best_model() to return best model with best params for final evaluation on test set
    5. csv model, params, val acc, test acc, precision, recall, f1, support, confusion matrix saved for all models in grid search
    
    