import numpy as np
import pandas as pd

import config


class DataLoader:
    @staticmethod
    def load(path=None) -> pd.DataFrame:
        if path is None:
            path = config.DATA_PATH
        return pd.read_csv(path)

    @staticmethod
    def filter_consent(df: pd.DataFrame, consent_col: str = "consent") -> pd.DataFrame:
        return df[
            df[consent_col].str.contains("consent to take part", na=False, case=False)
        ].copy()


class DataCleaner:
    @staticmethod
    def clip_outliers(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for col in numeric_cols:
            lo, hi = out[col].quantile(0.01), out[col].quantile(0.99)
            out[col] = out[col].clip(lo, hi)
        return out

    @staticmethod
    def impute_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for col in numeric_cols:
            out[col] = out[col].fillna(out[col].median())
        return out

    @staticmethod
    def impute_categorical(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for col in categorical_cols:
            mode_val = out[col].mode()
            out[col] = out[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")
        return out

    @staticmethod
    def drop_missing_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        return df.dropna(subset=[target_col]).copy()

    def clean(self, df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], target_col: str) -> pd.DataFrame:
        out = self.clip_outliers(df, numeric_cols)
        out = self.impute_numeric(out, numeric_cols)
        out = self.impute_categorical(out, categorical_cols)
        out = self.drop_missing_target(out, target_col)
        return out


class FeatureEngineer:
    @staticmethod
    def add_derived(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["financial_pressure"] = (
            out["rent"] / out["hours_work"].replace(0, np.nan)
        ).fillna(out["rent"])

        out["work_study_ratio"] = (
            out["hours_work"] / out["hours_studying"].replace(0, np.nan)
        ).fillna(out["hours_work"])

        out["social_engagement"] = (
            out["friends_count"] / out["social_media_use"].replace(0, np.nan)
        ).fillna(out["friends_count"])
        return out

    @staticmethod
    def clip_derived(df: pd.DataFrame, derived_cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for col in derived_cols:
            out[col] = out[col].clip(0, out[col].quantile(0.99))
        return out

    def build_stress_categories(self, df: pd.DataFrame, target: str, converters, target_category: str) -> pd.DataFrame:
        out = df.copy()
        out[target_category] = out[target].apply(converters.bin_stress)
        return out

    def engineer(self, df: pd.DataFrame, derived_cols: list[str], target: str, converters, target_category: str) -> pd.DataFrame:
        out = self.add_derived(df)
        out = self.clip_derived(out, derived_cols)
        out = self.build_stress_categories(out, target, converters, target_category)
        return out
