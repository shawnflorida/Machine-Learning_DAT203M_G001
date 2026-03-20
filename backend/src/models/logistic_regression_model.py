import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from backend.src.architecture.visualizer import Visualizer
from .base_model import BaseModel
from sklearn.model_selection import ParameterGrid
from backend.src.architecture.ml_tasks import EDA, Evaluator, Predictor
from backend import config

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = SGDClassifier()
        self.best_grid = None
        self.best_model = None
        self.best_metrics = None

    def train(self, X_train, y_train, X_val, y_val):
        self.grid_search(X_train, y_train, X_val, y_val)
        print(f"Trained: {self.get_name()}")
        print(f"Best params: {self.best_grid}")
        print(f"Model Fitting: {self.best_metrics}")

    def predict(self, X):
        return self.model.predict(X)

    def get_name(self) -> str:
        return "Multinomial Logistic Regression"
    
    
    grid_search_params = [{
        "loss": ["log_loss"],
        "penalty": ["l2", "l1", "elasticnet", None],
        "learning_rate": ["constant","optimal","invscaling","adaptive"],
        "eta0": [0.1,0.01,0.001],
        "random_state": [42],
        "max_iter": [500,1000,1500]
    }]
    
    def grid_search(self, X_train, y_train, X_val, y_val):
        best_score = 0
        best_grid = {}
        for g in ParameterGrid(self.grid_search_params):
            print(g)
    
            self.model.set_params(**g)
            
            # Model Training
            self.model.fit(X_train, y_train)
            train_acc = self.model.score(X_train, y_train)
            
            # Validations
            val_acc = self.model.score(X_val, y_val)
            
            print(f"Train acc: {train_acc}% \t Val acc: {val_acc}%", end="\n\n")
            
            if val_acc > best_score:
                best_score = val_acc
                best_model = self.model
                best_grid = g
                evaluator = Evaluator()
                best_metrics = { "training accuracy": train_acc, "validation accuracy": val_acc }
            
        pred_cats = {self.get_name(): self.model.predict(X_val)}
        class_reports = evaluator.classification_report_all(y_val, pred_cats, config.CATEGORY_ORDER)
        v = Visualizer()
        v.plot_confusion_matrices(class_reports, config.CATEGORY_ORDER)
        
        print("Best accuracy: ", best_score, "%")
        print("Best grid: ", best_grid)
        self.best_grid = best_grid
        self.best_model = best_model
        self.best_metrics = best_metrics
        
        
        