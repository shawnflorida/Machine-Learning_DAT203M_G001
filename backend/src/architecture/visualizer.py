import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Visualizer:
    def __init__(self):
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams["figure.dpi"] = 110
        self.palette = {
            "Low":     "#4CAF50",
            "Average": "#FF9800",
            "High":    "#F44336",
        }

    def plot_stress_distribution(self, df, target, category_col, category_order):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        cat_counts = df[category_col].value_counts().reindex(category_order)
        sns.barplot(x=cat_counts.index, y=cat_counts.values, palette=self.palette,
                    ax=axes[0], order=category_order)
        axes[0].set_title("Stress Category Distribution", fontsize=13)
        axes[0].set_xlabel("Stress Category")
        axes[0].set_ylabel("Count")

        axes[1].hist(df[target].dropna(), bins=11, color="#5C6BC0", edgecolor="white", linewidth=0.8)
        axes[1].set_title("Raw Stress Score Distribution", fontsize=13)
        axes[1].set_xlabel("Stress Score (0-10)")
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, df_eda, feature_cols, target):
        corr_df = df_eda[feature_cols + [target]].corr()
        stress_corr = corr_df[[target]].drop(target).sort_values(target, ascending=False)

        fig, ax = plt.subplots(figsize=(5, 9))
        sns.heatmap(stress_corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                    linewidths=0.5, linecolor="white", ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlation with Stress Score", fontsize=12)
        ax.set_xlabel("")
        plt.tight_layout()
        plt.show()

    def plot_boxplots(self, df, key_features, category_col, category_order):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i, feat in enumerate(key_features):
            plot_df = df[[feat, category_col]].dropna()
            sns.boxplot(data=plot_df, x=category_col, y=feat, order=category_order,
                        palette=self.palette, ax=axes[i],
                        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
            axes[i].set_title(feat.replace("_", " ").title(), fontsize=11)
            axes[i].set_xlabel("")
            axes[i].tick_params(axis="x", rotation=20)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, class_reports: dict, category_order: list[str]):
        fig, axes = plt.subplots(1, len(class_reports), figsize=(18, 5))
        if len(class_reports) == 1:
            axes = [axes]

        for ax, (name, data) in zip(axes, class_reports.items()):
            sns.heatmap(
                pd.DataFrame(data["cm"], index=category_order, columns=category_order),
                annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, linecolor="white", ax=ax, cbar=False,
            )
            ax.set_title(f"{name} | Accuracy: {data['accuracy']:.1%}", fontsize=11)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        plt.tight_layout()
        plt.show()

    def plot_model_comparison(self, model_names: list[str], accs: list[float]):
        x = np.arange(len(model_names))
        colors = ["#5C6BC0", "#EF5350", "#66BB6A"][: len(model_names)]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(x, accs, 0.5, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("Model Comparison: Classification Accuracy", fontsize=12)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.1%}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.show()
