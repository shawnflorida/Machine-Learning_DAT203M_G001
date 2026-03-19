import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


class Visualizer:
    def __init__(self):
        sns.set_theme(style="whitegrid", palette="husl")
        plt.rcParams["figure.dpi"] = 120
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.labelsize"] = 11
        plt.rcParams["axes.titlesize"] = 13
        plt.rcParams["xtick.labelsize"] = 9
        plt.rcParams["ytick.labelsize"] = 9
        plt.rcParams["figure.facecolor"] = "#f8f9fa"
        plt.rcParams["axes.facecolor"] = "#ffffff"
        plt.rcParams["axes.edgecolor"] = "#e0e0e0"
        plt.rcParams["legend.frameon"] = True
        plt.rcParams["legend.fancybox"] = True
        plt.rcParams["legend.shadow"] = False
        
        self.palette = {
            "Low":     "#10B981",      # Green
            "Average": "#F59E0B",      # Amber
            "High":    "#EF4444",      # Red
        }
        self.model_colors = ["#3B82F6", "#8B5CF6", "#EC4899"]  # Blue, Purple, Pink

    def plot_stress_distribution(self, df, target, category_col, category_order):
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        
        # Category distribution - bar chart
        ax1 = fig.add_subplot(gs[0, 0:2])
        cat_counts = df[category_col].value_counts().reindex(category_order)
        bars = ax1.bar(range(len(cat_counts)), cat_counts.values, 
                       color=[self.palette[cat] for cat in cat_counts.index],
                       edgecolor="#2d3748", linewidth=1.5, alpha=0.85)
        
        ax1.set_xticks(range(len(cat_counts)))
        ax1.set_xticklabels(cat_counts.index, fontsize=11, fontweight="bold")
        ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax1.set_title("Stress Category Distribution", fontsize=13, fontweight="bold", pad=15)
        ax1.grid(axis="y", alpha=0.3, linestyle="--")
        ax1.set_axisbelow(True)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight="bold", fontsize=10)
        
        # Raw score distribution - histogram
        ax2 = fig.add_subplot(gs[0, 2])
        n, bins, patches = ax2.hist(df[target].dropna(), bins=11, edgecolor="#2d3748", linewidth=1, alpha=0.8)
        
        # Gradient colors for histogram
        for i, patch in enumerate(patches):
            patch.set_facecolor(plt.cm.RdYlGn_r(i / len(patches)))
        
        ax2.set_title("Raw Stress Scores", fontsize=13, fontweight="bold", pad=15)
        ax2.set_xlabel("Score (0-10)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3, linestyle="--")
        ax2.set_axisbelow(True)
        
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, df_eda, feature_cols, target):
        corr_df = df_eda[feature_cols + [target]].corr()
        stress_corr = corr_df[[target]].drop(target).sort_values(target, ascending=False)

        fig, ax = plt.subplots(figsize=(7, 11))
        sns.heatmap(stress_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    linewidths=1, linecolor="#e0e0e0", ax=ax, 
                    cbar_kws={"shrink": 0.8, "label": "Correlation"},
                    vmin=-1, vmax=1,
                    annot_kws={"size": 9, "weight": "bold"})
        ax.set_title("Feature Correlation with Stress", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("", fontsize=0)
        ax.set_ylabel("", fontsize=0)
        plt.tight_layout()
        plt.show()

    def plot_boxplots(self, df, key_features, category_col, category_order):
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes = axes.flatten()

        for i, feat in enumerate(key_features):
            plot_df = df[[feat, category_col]].dropna()
            sns.boxplot(data=plot_df, x=category_col, y=feat, order=category_order,
                        palette=self.palette, ax=axes[i],
                        flierprops={"marker": "D", "markersize": 4, "alpha": 0.6, "markeredgecolor": "#2d3748"},
                        width=0.6,
                        linewidth=1.5)
            axes[i].set_title(feat.replace("_", " ").title(), fontsize=12, fontweight="bold", pad=10)
            axes[i].set_xlabel("")
            axes[i].set_ylabel("Value", fontsize=10)
            axes[i].tick_params(axis="x", rotation=0)
            axes[i].grid(axis="y", alpha=0.25, linestyle="--")
            axes[i].set_axisbelow(True)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, class_reports: dict, category_order: list[str]):
        n_models = len(class_reports)
        fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))
        if n_models == 1:
            axes = [axes]

        for ax, (name, data) in zip(axes, class_reports.items()):
            cm_df = pd.DataFrame(data["cm"], index=category_order, columns=category_order)
            
            # Create heatmap with custom colors
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", 
                       linewidths=2, linecolor="#ffffff", ax=ax, 
                       cbar_kws={"label": "Count", "shrink": 0.8},
                       annot_kws={"size": 12, "weight": "bold"},
                       vmin=0)
            
            accuracy = data['accuracy']
            ax.set_title(f"{name}\nAccuracy: {accuracy:.1%}", 
                        fontsize=13, fontweight="bold", pad=15)
            ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
            ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

        plt.tight_layout()
        plt.show()

    def plot_model_comparison(self, model_names: list[str], accs: list[float]):
        fig, ax = plt.subplots(figsize=(11, 6))
        
        x = np.arange(len(model_names))
        colors = self.model_colors[:len(model_names)]
        
        # Create gradient effect with alpha
        bars = ax.bar(x, accs, width=0.6, color=colors, edgecolor="#2d3748", 
                     linewidth=2, alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.set_title("Model Comparison: Classification Accuracy", fontsize=14, fontweight="bold", pad=20)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        
        # Add value labels on bars with background box
        for bar, acc in zip(bars, accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{acc:.1%}', ha='center', va='bottom', fontweight="bold", 
                   fontsize=11, bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='white', edgecolor=bar.get_facecolor(), alpha=0.8))
        
        # Add a horizontal line for average
        avg_acc = np.mean(accs)
        ax.axhline(y=avg_acc, color='gray', linestyle='--', linewidth=2, alpha=0.5, label=f'Average: {avg_acc:.1%}')
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        plt.show()
