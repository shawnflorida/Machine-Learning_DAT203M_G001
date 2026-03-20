import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Visualizer:
    def __init__(self):
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({"figure.dpi": 90, "font.size": 9})
        self.palette = {"Low": "#4CAF50", "Average": "#FF9800", "High": "#F44336"}

    def plot_stress_distribution(self, df, target, category_col, category_order):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        cat_counts = df[category_col].value_counts().reindex(category_order)
        sns.barplot(x=cat_counts.index, y=cat_counts.values, palette=self.palette, ax=axes[0], order=category_order)
        axes[0].set(title="Stress Category Distribution", xlabel="Category", ylabel="Count")
        axes[1].hist(df[target].dropna(), bins=11, color="#5C6BC0", edgecolor="white", linewidth=0.8)
        axes[1].set(title="Raw Stress Score Distribution", xlabel="Score (0-10)", ylabel="Frequency")
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, df_eda, feature_cols, target):
        stress_corr = df_eda[feature_cols + [target]].corr()[[target]].drop(target).sort_values(target, ascending=False)
        fig, ax = plt.subplots(figsize=(3.5, 7))
        sns.heatmap(stress_corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                    linewidths=0.5, linecolor="white", ax=ax, cbar=False)
        ax.set_title("Feature Correlation with Stress", fontsize=10)
        plt.tight_layout()
        plt.show()

    def plot_boxplots(self, df, key_features, category_col, category_order):
        fig, axes = plt.subplots(2, 4, figsize=(12, 5))
        for ax, feat in zip(axes.flatten(), key_features):
            sns.boxplot(data=df[[feat, category_col]].dropna(), x=category_col, y=feat,
                        order=category_order, palette=self.palette, ax=ax,
                        flierprops={"marker": "o", "markersize": 2, "alpha": 0.4})
            ax.set(title=feat.replace("_", " ").title(), xlabel="")
            ax.tick_params(axis="x", rotation=20, labelsize=8)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, class_reports: dict, category_order: list[str]):
        n = len(class_reports)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, (name, data) in zip(axes, class_reports.items()):
            sns.heatmap(pd.DataFrame(data["cm"], index=category_order, columns=category_order),
                        annot=True, fmt="d", cmap="Blues", linewidths=0.5,
                        linecolor="white", ax=ax, cbar=False)
            ax.set(title=f"{name} | Acc: {data['accuracy']:.1%}", xlabel="Predicted", ylabel="True")
        plt.tight_layout()
        plt.show()

    def plot_model_comparison(self, model_names: list[str], accs: list[float]):
        x = np.arange(len(model_names))
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.bar(x, accs, 0.5, color=["#5C6BC0", "#EF5350", "#66BB6A"][:len(model_names)])
        ax.set(xticks=x, xticklabels=model_names, ylabel="Accuracy",
               ylim=(0, 1.05), title="Model Comparison: Classification Accuracy")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.show()

    def plot_loss_curves(self, models: list, title: str = "Training Loss Curves"):
        """Plot loss curves for all models that expose get_loss_curve().
        Models with no iterative loss (e.g. Decision Tree, Logistic Regression)
        are shown as a note rather than skipped silently.
        """
        curves = []
        for m in models:
            if hasattr(m, "get_loss_curve"):
                curve = m.get_loss_curve()
                if curve is not None and len(curve) > 0:
                    val_curve = m.get_val_loss_curve() if hasattr(m, "get_val_loss_curve") else None
                    curves.append((m.get_name(), np.asarray(curve), val_curve))

        if not curves:
            print("No loss curves available — models may not have iterative training.")
            return

        colors = ["#5C6BC0", "#EF5350", "#66BB6A", "#FF9800", "#AB47BC"]
        fig, axes = plt.subplots(1, len(curves), figsize=(5.5 * len(curves), 4), squeeze=False)
        for ax, (name, curve, val_curve), color in zip(axes[0], curves, colors):
            ax.plot(curve, color=color, linewidth=1.8, label="Train loss")
            if val_curve is not None and len(val_curve) > 0:
                ax.plot(np.asarray(val_curve), color=color, linewidth=1.4,
                        linestyle="--", alpha=0.7, label="Val loss")
                ax.legend(fontsize=8)
            ax.set(title=name, xlabel="Epoch / Iteration", ylabel="Loss")
            ax.grid(True, linewidth=0.4)
        fig.suptitle(title, fontsize=11, y=1.01)
        plt.tight_layout()
        plt.show()
