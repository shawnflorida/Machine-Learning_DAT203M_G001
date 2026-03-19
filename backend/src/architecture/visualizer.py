import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Visualizer:
    def __init__(self):
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams["figure.dpi"] = 110
        self.palette = {
            "Low (1-3)": "#4CAF50",
            "Average (4-6)": "#FF9800",
            "High (7-10)": "#F44336",
        }

    def plot_stress_distribution(self, df: pd.DataFrame, target: str, category_col: str, category_order: list[str]):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        cat_counts = df[category_col].value_counts().reindex(category_order)
        sns.barplot(
            x=cat_counts.index,
            y=cat_counts.values,
            palette=self.palette,
            ax=axes[0],
            order=category_order,
        )
        axes[0].set_title("Stress Category Distribution", fontsize=13)
        axes[0].set_xlabel("Stress Category")
        axes[0].set_ylabel("Count")

        axes[1].hist(df[target].dropna(), bins=11, color="#5C6BC0", edgecolor="white", linewidth=0.8)
        axes[1].set_title("Raw Stress Score Distribution", fontsize=13)
        axes[1].set_xlabel("Stress Score (0-10)")
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, df_eda: pd.DataFrame, feature_cols: list[str], target: str):
        corr_df = df_eda[feature_cols + [target]].corr()
        stress_corr = corr_df[[target]].drop(target).sort_values(target, ascending=False)

        fig, ax = plt.subplots(figsize=(5, 9))
        sns.heatmap(
            stress_corr,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Feature Correlation with Stress Score", fontsize=12)
        ax.set_xlabel("")
        plt.tight_layout()
        plt.show()

    def plot_boxplots(self, df: pd.DataFrame, key_features: list[str], category_col: str, category_order: list[str]):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i, feat in enumerate(key_features):
            plot_df = df[[feat, category_col]].dropna()
            sns.boxplot(
                data=plot_df,
                x=category_col,
                y=feat,
                order=category_order,
                palette=self.palette,
                ax=axes[i],
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
            )
            axes[i].set_title(feat.replace("_", " ").title(), fontsize=11)
            axes[i].set_xlabel("")
            axes[i].tick_params(axis="x", rotation=20)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, pred_cats: dict, y_true_category, category_order, accuracy_fn, confusion_fn):
        fig, axes = plt.subplots(1, len(pred_cats), figsize=(18, 5))
        if len(pred_cats) == 1:
            axes = [axes]

        for ax, (name, y_pred_cat) in zip(axes, pred_cats.items()):
            cm = confusion_fn(y_true_category, y_pred_cat, labels=category_order)
            sns.heatmap(
                pd.DataFrame(cm, index=category_order, columns=category_order),
                annot=True,
                fmt="d",
                cmap="Blues",
                linewidths=0.5,
                linecolor="white",
                ax=ax,
                cbar=False,
            )
            acc = accuracy_fn(y_true_category, y_pred_cat)
            ax.set_title(f"{name} | Accuracy: {acc:.1%}", fontsize=11)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        plt.tight_layout()
        plt.show()

    def plot_model_comparison(self, model_names: list[str], accs: list[float], r2s: list[float]):
        x = np.arange(len(model_names))
        w = 0.35
        colors_acc = ["#5C6BC0", "#EF5350", "#66BB6A"][: len(model_names)]
        colors_r2 = ["#9FA8DA", "#EF9A9A", "#A5D6A7"][: len(model_names)]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars1 = ax.bar(x - w / 2, accs, w, color=colors_acc, label="Category Accuracy")
        bars2 = ax.bar(x + w / 2, r2s, w, color=colors_r2, label="Regression R2")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison: Accuracy & R2", fontsize=12)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend()

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.show()
