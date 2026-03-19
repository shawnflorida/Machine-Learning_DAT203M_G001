import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class Converters:
    @staticmethod
    def label_encoder_independent(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Create stress_category from the raw numeric stress column."""
        def categorize(value):
            if value <= 3:
                return "Low"
            elif value <= 6:
                return "Average"
            else:
                return "High"

        result = df.copy()
        result["stress_category"] = result[col].apply(categorize)
        return result

    @staticmethod
    def label_encode(df: pd.DataFrame, cols: list[str]):
        encoded_df = df.copy()
        encoders = {}
        for col in cols:
            encoder = LabelEncoder()
            encoded_df[col] = encoder.fit_transform(encoded_df[col].astype(str))
            encoders[col] = encoder
        return encoded_df, encoders


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

        profile_data = dict(profile)
        hours_work     = profile_data["hours_work"]     if profile_data["hours_work"]     != 0 else 1
        hours_studying = profile_data["hours_studying"] if profile_data["hours_studying"] != 0 else 1
        social_media   = profile_data["social_media_use"] if profile_data["social_media_use"] != 0 else 1

        profile_data["financial_pressure"] = float(profile_data["rent"])        / float(hours_work)
        profile_data["work_study_ratio"]   = float(profile_data["hours_work"])  / float(hours_studying)
        profile_data["social_engagement"]  = float(profile_data["friends_count"]) / float(social_media)

        row = pd.DataFrame([profile_data])[self.all_numeric + self.all_cats]

        for col in self.all_numeric:
            row[col] = pd.to_numeric(row[col], errors="coerce")
        row = row.replace([np.inf, -np.inf], np.nan)

        return row

    def generate_profile(self, df: pd.DataFrame, seed: int | None = None, mode: str = "random") -> dict:
        missing_cols = [col for col in (self.numeric_cols + self.categorical_cols) if col not in df.columns]
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
