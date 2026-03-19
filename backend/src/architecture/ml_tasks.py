import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class EDA:
    def __init__(self, visualizer):
        self.visualizer = visualizer

    def run(self, df, df_eda, feature_cols, target, target_category, category_order):
        self.visualizer.plot_stress_distribution(df, target, target_category, category_order)
        self.visualizer.plot_correlation_heatmap(df_eda, feature_cols, target)

        key_features = [
            "hours_work", "social_media_use", "rent", "friends_count",
            "standard_drinks", "hours_studying", "semesters", "dates",
        ]
        self.visualizer.plot_boxplots(df, key_features, target_category, category_order)


class Evaluator:
    def regression_report(self, y_true, preds: dict):
        metrics = {}
        for name, y_pred in preds.items():
            metrics[name] = {
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "R2": r2_score(y_true, y_pred),
            }
        return metrics

    def print_regression_report(self, metrics: dict):
        print(f"\n{'=' * 60}")
        print(f"  {'MODEL':<22} {'MAE':>6} {'RMSE':>7} {'R2':>9}")
        print(f"{'=' * 60}")
        for name, m in metrics.items():
            print(f"  {name:<22} {m['MAE']:>6.3f} {m['RMSE']:>7.3f} {m['R2']:>9.4f}")
        print(f"{'=' * 60}")

    def classification_report_all(self, y_true_cat, pred_cats, category_order):
        reports = {}
        for name, y_pred_cat in pred_cats.items():
            reports[name] = {
                "accuracy": accuracy_score(y_true_cat, y_pred_cat),
                "report": classification_report(y_true_cat, y_pred_cat, labels=category_order),
                "cm": confusion_matrix(y_true_cat, y_pred_cat, labels=category_order),
            }
        return reports

    def print_classification_reports(self, reports: dict):
        for name, data in reports.items():
            print(f"\n{'=' * 52}")
            print(f"  {name}  |  Accuracy: {data['accuracy']:.1%}")
            print(f"{'=' * 52}")
            print(data["report"])


class Predictor:
    def __init__(self, converters):
        self.converters = converters

    def predict(self, profile_df, models: list, pipeliner):
        X = pipeliner.transform(profile_df)
        results = []
        for model in models:
            score = float(np.clip(model.predict(X)[0], 0, 10))
            category = self.converters.bin_stress(score)
            results.append(
                {
                    "model_name": model.get_name(),
                    "score": round(score, 4),
                    "category": category,
                }
            )
        return results

    @staticmethod
    def print_results(results: list[dict]):
        print("=" * 58)
        print("    NEW STUDENT STRESS PREDICTION - ALL MODELS")
        print("=" * 58)
        for row in results:
            print(f"  {row['model_name']:<22}: {row['score']:.2f}/10  ->  {row['category']}")
        print("=" * 58)
