import numpy as np
import pandas as pd

import config


class DataLoader:
    @staticmethod
    def load(path=None) -> pd.DataFrame:
        return pd.read_csv(path or config.DATA_PATH)

    @staticmethod
    def filter_consent(df: pd.DataFrame, consent_col: str = "consent") -> pd.DataFrame:
        return df[df[consent_col].str.contains("consent to take part", na=False, case=False)].copy()


class DataCleaner:
    def clean(self, df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], target_col: str) -> pd.DataFrame:
        cleaned = df.copy()

        # Clip outliers to 1st–99th percentile
        for col in numeric_cols:
            lower_bound = cleaned[col].quantile(0.01)
            upper_bound = cleaned[col].quantile(0.99)
            cleaned[col] = cleaned[col].clip(lower_bound, upper_bound)

        # Impute missing values
        for col in numeric_cols:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        for col in categorical_cols:
            mode_val = cleaned[col].mode()
            cleaned[col] = cleaned[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")

        # Drop rows where target is missing
        return cleaned.dropna(subset=[target_col]).copy()


class FeatureEngineer:
    def engineer(self, df: pd.DataFrame, derived_cols: list[str], target: str, converters, target_category: str) -> pd.DataFrame:
        result = df.copy()

        # Derived features (avoid division by zero)
        result["financial_pressure"] = (result["rent"] / result["hours_work"].replace(0, np.nan)).fillna(result["rent"])
        result["work_study_ratio"]   = (result["hours_work"] / result["hours_studying"].replace(0, np.nan)).fillna(result["hours_work"])
        result["social_engagement"]  = (result["friends_count"] / result["social_media_use"].replace(0, np.nan)).fillna(result["friends_count"])

        # Cap derived cols at 99th percentile
        for col in derived_cols:
            result[col] = result[col].clip(0, result[col].quantile(0.99))

        # Build stress_category — the classification target
        result = converters.label_encoder_independent(result, target)

        return result
