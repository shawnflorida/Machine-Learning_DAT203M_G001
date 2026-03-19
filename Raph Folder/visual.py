import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, f1_score
)

CLASS_NAMES  = ['Low (0-3)', 'Average (4-6)', 'High (7-10)']
CLASS_COLORS = ['#4CAF50', '#FF9800', '#F44336']


class Visualisation:
    """
    Centralised visualisation utility shared across all model classes.
    Import via:  from visual import Visualisation
    """

    # ------------------------------------------------------------------ #
    #  EDA / Dataset-level                                                 #
    # ------------------------------------------------------------------ #

    def plot_class_distribution(self, y, title='Target Class Distribution'):
        classes, counts = np.unique(y, return_counts=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar([CLASS_NAMES[c] for c in classes], counts,
                    color=CLASS_COLORS, edgecolor='black')
        axes[0].set_title('Count')
        axes[0].set_ylabel('Count')
        for i, v in enumerate(counts):
            axes[0].text(i, v + 3, str(v), ha='center', fontweight='bold')
        axes[1].pie(counts, labels=[CLASS_NAMES[c] for c in classes],
                    colors=CLASS_COLORS, autopct='%1.1f%%', startangle=90,
                    wedgeprops=dict(edgecolor='white'))
        axes[1].set_title('Proportion')
        plt.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout(); plt.show()

    def plot_feature_distributions(self, df, numerical_features):
        n = len(numerical_features)
        cols = 4; rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
        axes = axes.flatten()
        for i, col in enumerate(numerical_features):
            axes[i].hist(df[col], bins=30, color='steelblue',
                         edgecolor='white', alpha=0.85)
            axes[i].set_title(col, fontsize=10); axes[i].set_ylabel('Count')
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('Numerical Feature Distributions (after winsorisation)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(); plt.show()

    def plot_boxplots_by_class(self, df, numerical_features,
                               target_col='stress_class'):
        n = len(numerical_features)
        cols = 4; rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
        axes = axes.flatten()
        classes = sorted(df[target_col].unique())
        for i, col in enumerate(numerical_features):
            groups = [df[df[target_col] == k][col].values for k in classes]
            bp = axes[i].boxplot(groups,
                                 labels=[CLASS_NAMES[k] for k in classes],
                                 patch_artist=True)
            for patch, color in zip(bp['boxes'], CLASS_COLORS):
                patch.set_facecolor(color); patch.set_alpha(0.6)
            axes[i].set_title(col, fontsize=10)
            axes[i].tick_params(axis='x', rotation=15)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('Numerical Features by Stress Class',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(); plt.show()

    def plot_correlation_heatmap(self, df, cols):
        corr = df[cols].corr()
        plt.figure(figsize=(13, 10))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, linewidths=0.4, square=True,
                    cbar_kws={'shrink': 0.8})
        plt.title('Correlation Heatmap', fontsize=13, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout(); plt.show()

    def plot_categorical_vs_class(self, df, categorical_features,
                                  target_col='stress_class'):
        n = len(categorical_features)
        cols = 3; rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
        axes = axes.flatten()
        for i, col in enumerate(categorical_features):
            cross = (pd.crosstab(df[col], df[target_col],
                                 normalize='index') * 100)
            cross.columns = [CLASS_NAMES[c] for c in cross.columns]
            cross.plot(kind='bar', ax=axes[i], color=CLASS_COLORS,
                       edgecolor='black', alpha=0.8)
            axes[i].set_title(col); axes[i].set_ylabel('%')
            axes[i].tick_params(axis='x', rotation=30)
            axes[i].legend(fontsize=7)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('Categorical Features by Stress Class (%)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(); plt.show()

    def plot_split_distribution(self, y_train, y_val, y_test):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, y_split, title in [
            (axes[0], y_train, 'Training'),
            (axes[1], y_val,   'Validation'),
            (axes[2], y_test,  'Test'),
        ]:
            classes, counts = np.unique(y_split, return_counts=True)
            ax.bar([CLASS_NAMES[c] for c in classes], counts,
                   color=CLASS_COLORS, edgecolor='black')
            ax.set_title(f'{title} Set'); ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=15)
            for idx, v in enumerate(counts):
                ax.text(idx, v + 1, str(v), ha='center',
                        fontweight='bold', fontsize=9)
        plt.suptitle('Class Distribution Across Splits',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------ #
    #  Model-level                                                         #
    # ------------------------------------------------------------------ #

    def plot_learning_curves(self, train_losses, val_losses,
                             train_accs, val_accs, model_name):
        epochs = range(1, len(train_losses) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(epochs, train_losses, label='Train Loss', color='royalblue')
        axes[0].plot(epochs, val_losses,   label='Val Loss',
                     color='tomato', linestyle='--')
        axes[0].set_title(f'{model_name} — Loss')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-Entropy Loss')
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(epochs, train_accs, label='Train Acc', color='royalblue')
        axes[1].plot(epochs, val_accs,   label='Val Acc',
                     color='tomato', linestyle='--')
        axes[1].set_title(f'{model_name} — Accuracy')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim([0, 1]); axes[1].legend(); axes[1].grid(alpha=0.3)
        plt.tight_layout(); plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, model_name,
                              split_label='Validation', cmap='Blues'):
        cm   = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=CLASS_NAMES)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, colorbar=False, cmap=cmap)
        ax.set_title(f'{model_name}\n({split_label} Set)', fontweight='bold')
        ax.set_xticklabels(CLASS_NAMES, rotation=15, ha='right')
        plt.tight_layout(); plt.show()

    def plot_confusion_matrix_grid(self, results_list, split_label='Test'):
        """results_list: list of (model_name, y_true, y_pred)"""
        n = len(results_list)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1: axes = [axes]
        for ax, (name, y_true, y_pred) in zip(axes, results_list):
            cm   = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=CLASS_NAMES)
            disp.plot(ax=ax, colorbar=False, cmap='Blues')
            ax.set_title(name, fontweight='bold')
            ax.set_xticklabels(CLASS_NAMES, rotation=15, ha='right')
        plt.suptitle(f'Confusion Matrices — {split_label} Set',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(); plt.show()

    def plot_per_class_f1(self, results_dict, split_label='Validation'):
        """results_dict: {model_name: (y_true, y_pred)}"""
        model_names = list(results_dict.keys())
        n_models    = len(model_names)
        x     = np.arange(len(CLASS_NAMES))
        width = 0.8 / n_models
        palette = ['steelblue', 'darkorange', 'mediumseagreen', 'orchid']
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, name in enumerate(model_names):
            y_true, y_pred = results_dict[name]
            f1s    = f1_score(y_true, y_pred, average=None, zero_division=0)
            offset = (i - n_models / 2 + 0.5) * width
            bars   = ax.bar(x + offset, f1s, width, label=name,
                            color=palette[i % len(palette)],
                            edgecolor='black', alpha=0.85)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f'{bar.get_height():.2f}',
                        ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylabel('F1 Score'); ax.set_ylim([0, 1.1])
        ax.set_title(f'Per-Class F1 Score — {split_label} Set',
                     fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout(); plt.show()

    def plot_metric_summary(self, summary_df):
        metrics = ['Accuracy', 'Macro F1', 'Weighted F1',
                   'Macro Precision', 'Macro Recall']
        x = np.arange(len(metrics)); n = len(summary_df)
        width   = 0.7 / n
        palette = ['steelblue', 'darkorange', 'mediumseagreen', 'orchid']
        fig, ax = plt.subplots(figsize=(13, 5))
        for i, (_, row) in enumerate(summary_df.iterrows()):
            vals   = [row[m] for m in metrics]
            offset = (i - n / 2 + 0.5) * width
            bars   = ax.bar(x + offset, vals, width, label=row['Model'],
                            color=palette[i % len(palette)],
                            edgecolor='black', alpha=0.85)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004,
                        f'{bar.get_height():.3f}',
                        ha='center', va='bottom', fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels(metrics)
        ax.set_ylabel('Score'); ax.set_ylim([0, 1.1])
        ax.set_title('Model Performance Comparison (Test Set)',
                     fontsize=13, fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout(); plt.show()

    def plot_knn_k_search(self, k_values, val_accs):
        best_k = k_values[int(np.argmax(val_accs))]
        plt.figure(figsize=(8, 4))
        plt.plot(k_values, val_accs, marker='o',
                 color='darkorange', linewidth=2)
        plt.axvline(best_k, color='red', linestyle='--',
                    label=f'Best k = {best_k}')
        plt.xlabel('k (neighbours)'); plt.ylabel('Validation Accuracy')
        plt.title('k-NN — k vs Validation Accuracy', fontweight='bold')
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.show()

    def plot_dt_depth_search(self, depths, val_accs):
        best_d = depths[int(np.argmax(val_accs))]
        plt.figure(figsize=(8, 4))
        plt.plot(depths, val_accs, marker='s',
                 color='mediumseagreen', linewidth=2)
        plt.axvline(best_d, color='red', linestyle='--',
                    label=f'Best depth = {best_d}')
        plt.xlabel('Max Depth'); plt.ylabel('Validation Accuracy')
        plt.title('Decision Tree — Depth vs Validation Accuracy',
                  fontweight='bold')
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.show()

    def plot_feature_importance(self, importances, feature_names,
                                model_name='Decision Tree', top_n=15):
        idx  = np.argsort(importances)[::-1][:top_n]
        vals = importances[idx]
        lbls = [feature_names[i] for i in idx]
        plt.figure(figsize=(9, 5))
        plt.barh(lbls[::-1], vals[::-1],
                 color='mediumseagreen', edgecolor='black', alpha=0.85)
        plt.xlabel('Importance (Gini)')
        plt.title(f'{model_name} — Top {top_n} Feature Importances',
                  fontweight='bold')
        plt.tight_layout(); plt.show()

    def plot_before_after(self, before_dict, after_dict,
                          split_label='Validation'):
        """
        before_dict / after_dict: {model_name: (y_true, y_pred)}
        """
        all_items = list(before_dict.items()) + list(after_dict.items())
        labels    = (['Before'] * len(before_dict) +
                     ['After']  * len(after_dict))
        cmaps     = ['Oranges'] * len(before_dict) + ['Greens'] * len(after_dict)
        n = len(all_items)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1: axes = [axes]
        for ax, (name, (y_true, y_pred)), lbl, cmc in zip(
                axes, all_items, labels, cmaps):
            cm   = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=CLASS_NAMES)
            disp.plot(ax=ax, colorbar=False, cmap=cmc)
            ax.set_title(f'{lbl}: {name}', fontweight='bold')
            ax.set_xticklabels(CLASS_NAMES, rotation=15, ha='right')
        plt.suptitle(f'Before vs After Improvement — {split_label} Set',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(); plt.show()

    # Placeholder aliases (matching your screenshot structure)
    def plot(self):
        """Alias — call specific plot_* methods directly."""
        pass

    def correlation(self):
        """Alias — call plot_correlation_heatmap() directly."""
        pass
