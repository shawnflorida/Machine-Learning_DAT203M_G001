from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
class EDA:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        
    
    def plot_distributions(self, df, target, target_category, category_order):
        self.visualizer.plot_stress_distribution(df, target, target_category, category_order)
    
    def plot_correlation_heatmap(self, df_eda, feature_cols, target):
        self.visualizer.plot_correlation_heatmap(df_eda, feature_cols, target)
    
    def boxplot_key_features(self, df, target_category, category_order):
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
                "f1": f1_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0),
                "report": classification_report(y_true_cat, y_pred_cat, labels=category_order),
                "cm": confusion_matrix(y_true_cat, y_pred_cat, labels=category_order),
            }
        return reports

    def print_classification_reports(self, reports: dict):
        for name, data in reports.items():
            print(f"\n{'=' * 52}")
            print(f"  {name}  |  F1 (weighted): {data['f1']:.1%}")
            print(f"{'=' * 52}")
            print(data["report"])

    def print_score_table(self, models: list, phase: str = "", reports: dict = None):
        """Print a train / val / test accuracy and final loss summary table."""
        import math
        title = f"MODEL RESULTS SUMMARY \u2014 {phase}" if phase else "MODEL RESULTS SUMMARY"
        w = 83
        print(f"\n{'\u2550' * w}")
        print(f"  {title}")
        print(f"{'\u2550' * w}")
        print(f"  {'Model':<34} {'Train F1':>9}  {'Val F1':>9}  {'Test F1':>9}  {'Final Loss':>10}")
        print(f"  {'\u2500' * 34}  {'\u2500' * 9}  {'\u2500' * 9}  {'\u2500' * 9}  {'\u2500' * 10}")
        for m in models:
            t = getattr(m, '_train_score', float('nan'))
            v = getattr(m, '_val_score', float('nan'))
            test_f1 = reports[m.get_name()]["f1"] if reports and m.get_name() in reports else float('nan')
            # Support both _loss_curve (list) and loss_curve_ (NN attribute)
            curve = getattr(m, '_loss_curve', None) or getattr(m, 'loss_curve_', None)
            final_loss = curve[-1] if curve else float('nan')
            t_str  = f"{t:.2%}"          if not math.isnan(t)          else "      n/a"
            v_str  = f"{v:.2%}"          if not math.isnan(v)          else "      n/a"
            ts_str = f"{test_f1:.2%}"    if not math.isnan(test_f1)    else "      n/a"
            fl_str = f"{final_loss:.4f}" if not math.isnan(final_loss) else "       n/a"
            print(f"  {m.get_name():<34} {t_str:>9}  {v_str:>9}  {ts_str:>9}  {fl_str:>10}")  # Train F1 / Val F1 / Test F1 / Final Loss
        print(f"{'═' * w}\n")


class Predictor:
    def predict(self, profile_df, models: list, pipeliner):
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
            raw_category = str(model.predict(X)[0])
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
