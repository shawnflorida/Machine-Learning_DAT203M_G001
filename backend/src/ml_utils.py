import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class Converters:
    def __init__(self, stress_bins):
        self.stress_bins = stress_bins

    def bin_stress(self, score: float) -> str:
        for threshold, label in self.stress_bins:
            if score <= threshold:
                return label
        return self.stress_bins[-1][1]

    def to_stress_categories(self, scores) -> list[str]:
        arr = np.asarray(scores, dtype=float)
        return [self.bin_stress(float(s)) for s in arr]

    @staticmethod
    def label_encode(df: pd.DataFrame, cols: list[str]):
        df_copy = df.copy()
        encoders = {}
        for col in cols:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            encoders[col] = le
        return df_copy, encoders


class Pipeliner:
    def __init__(self, numeric_cols: list[str], categorical_cols: list[str]):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        self.pipeline = ColumnTransformer([
            ("num", numeric_transformer, self.numeric_cols),
            ("cat", categorical_transformer, self.categorical_cols),
        ])

    def fit_transform(self, X):
        return self.pipeline.fit_transform(X)

    def transform(self, X):
        return self.pipeline.transform(X)

    def get_ohe_feature_names(self) -> list[str]:
        return (
            self.pipeline.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(self.categorical_cols)
            .tolist()
        )

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


class ProfileGenerator:
    def __init__(self, numeric_cols: list[str], categorical_cols: list[str], all_numeric: list[str], all_cats: list[str]):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.all_numeric = all_numeric
        self.all_cats = all_cats

    def validate(self, profile: dict):
        required = set(self.numeric_cols + self.categorical_cols)
        missing = sorted(required - set(profile.keys()))
        if missing:
            raise ValueError(f"Missing profile fields: {missing}")

    def build(self, profile: dict) -> pd.DataFrame:
        self.validate(profile)

        r = dict(profile)
        hours_work = r["hours_work"] if r["hours_work"] != 0 else 1
        hours_studying = r["hours_studying"] if r["hours_studying"] != 0 else 1
        social_media = r["social_media_use"] if r["social_media_use"] != 0 else 1

        r["financial_pressure"] = float(r["rent"]) / float(hours_work)
        r["work_study_ratio"] = float(r["hours_work"]) / float(hours_studying)
        r["social_engagement"] = float(r["friends_count"]) / float(social_media)

        row = pd.DataFrame([r])[self.all_numeric + self.all_cats]

        for col in self.all_numeric:
            row[col] = pd.to_numeric(row[col], errors="coerce")
        row = row.replace([np.inf, -np.inf], np.nan)

        return row

    def generate_profile(self, df: pd.DataFrame, seed: int | None = None, mode: str = "random") -> dict:
        missing_cols = [c for c in (self.numeric_cols + self.categorical_cols) if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in source dataframe: {missing_cols}")

        rng = np.random.default_rng(seed)
        profile = {}

        for col in self.numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                profile[col] = 0.0
                continue

            if mode == "typical":
                profile[col] = float(series.median())
                continue

            col_min = float(series.min())
            col_max = float(series.max())

            if np.isclose(col_min, col_max):
                sampled = col_min
            elif np.all(np.isclose(series.values, np.round(series.values))):
                sampled = int(rng.integers(int(np.floor(col_min)), int(np.ceil(col_max)) + 1))
            else:
                sampled = float(rng.uniform(col_min, col_max))

            profile[col] = float(sampled)

        for col in self.categorical_cols:
            values = df[col].dropna().astype(str).unique().tolist()
            if not values:
                profile[col] = "Unknown"
                continue

            if mode == "typical":
                mode_series = df[col].mode(dropna=True)
                profile[col] = str(mode_series.iloc[0]) if not mode_series.empty else str(values[0])
            else:
                profile[col] = str(rng.choice(values))

        return profile


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
