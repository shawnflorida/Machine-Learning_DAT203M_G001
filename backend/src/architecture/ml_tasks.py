from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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
    def predict(self, profile_df, models: list, pipeliner, label_encoder=None):
        def normalize_category(category: str) -> str:
            if "Low" in category:
                return "Low"
            if "Average" in category:
                return "Average"
            if "High" in category:
                return "High"
            return category

        X = pipeliner.transform(profile_df)
        results = []
        for model in models:
            raw_pred = model.predict(X)[0]
            
            # If label_encoder is provided, decode numeric index to category name
            if label_encoder is not None:
                try:
                    raw_category = label_encoder.inverse_transform([raw_pred])[0]
                except (ValueError, TypeError):
                    # If decoding fails, use the raw prediction as string
                    raw_category = str(raw_pred)
            else:
                raw_category = str(raw_pred)
            
            results.append({
                "model_name": model.get_name(),
                "category": normalize_category(raw_category),
            })
        return results

    @staticmethod
    def print_results(results: list[dict]):
        print("=" * 50)
        print("  STUDENT STRESS PREDICTION")
        print("=" * 50)
        for row in results:
            print(f"  {row['model_name']:<22}: {row['category']}")
        print("=" * 50)
