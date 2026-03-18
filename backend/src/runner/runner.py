import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.data_pipeline import DataCleaner, DataLoader, FeatureEngineer
from src.ml_tasks import EDA, Evaluator, Predictor
from src.ml_utils import Converters, Pipeliner, ProfileGenerator, Visualizer
from src.models import GradientBoostingModel, LinearRegressionModel, NeuralNetworkModel


class Runner:
    def __init__(self):
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.converters = Converters(config.STRESS_BINS)
        self.visualizer = Visualizer()
        self.eda = EDA(self.visualizer)
        self.evaluator = Evaluator()

        self.pipeliner = Pipeliner(config.ALL_NUMERIC, config.ALL_CATS)
        self.models = [
            LinearRegressionModel(),
            NeuralNetworkModel(),
            GradientBoostingModel(),
        ]

        self.predictor = Predictor(self.converters)
        self.profile_generator = ProfileGenerator(
            config.NUMERIC_COLS,
            config.CATEGORICAL_COLS,
            config.ALL_NUMERIC,
            config.ALL_CATS,
        )

    def _split(self, X_raw, y_continuous, y_category):
        X_temp, X_test, y_temp, y_test, ycat_temp, ycat_test = train_test_split(
            X_raw,
            y_continuous,
            y_category,
            test_size=config.TEST_SIZE,
            random_state=config.SEED,
            stratify=y_category,
        )

        X_train, X_val, y_train, y_val, ycat_train, ycat_val = train_test_split(
            X_temp,
            y_temp,
            ycat_temp,
            test_size=config.VAL_SIZE_FROM_REMAINING,
            random_state=config.SEED,
            stratify=ycat_temp,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test, ycat_train, ycat_val, ycat_test

    def run(self):
        raw = self.loader.load()
        raw = self.loader.filter_consent(raw)

        cols_needed = config.NUMERIC_COLS + config.CATEGORICAL_COLS + [config.TARGET]
        df = raw[cols_needed].copy()

        df = self.cleaner.clean(df, config.NUMERIC_COLS, config.CATEGORICAL_COLS, config.TARGET)
        df = self.engineer.engineer(df, config.DERIVED_COLS, config.TARGET, self.converters, config.TARGET_CATEGORY)

        df_eda, _ = self.converters.label_encode(df, config.CATEGORICAL_COLS)

        X_num = df[config.ALL_NUMERIC].copy()
        X_cat = raw.loc[df.index, config.ALL_CATS].fillna("Unknown")
        X_raw = pd.concat([X_num, X_cat], axis=1)

        y_continuous = df[config.TARGET]
        y_category = df[config.TARGET_CATEGORY]

        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, ycat_train, ycat_val, ycat_test = self._split(
            X_raw, y_continuous, y_category
        )

        X_train = self.pipeliner.fit_transform(X_train_raw)
        X_val = self.pipeliner.transform(X_val_raw)
        X_test = self.pipeliner.transform(X_test_raw)

        print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        feature_cols = config.NUMERIC_COLS + config.DERIVED_COLS + config.CATEGORICAL_COLS
        self.eda.run(df, df_eda, feature_cols, config.TARGET, config.TARGET_CATEGORY, config.CATEGORY_ORDER)

        for model in self.models:
            if model.get_name() == "Neural Network":
                model.train(X_train, y_train, X_val, y_val)
            else:
                model.train(X_train, y_train)
            print(f"Trained: {model.get_name()}")

        preds = {
            m.get_name(): np.clip(m.predict(X_test), 0, 10)
            for m in self.models
        }
        pred_cats = {
            name: self.converters.to_stress_categories(values)
            for name, values in preds.items()
        }

        metrics = self.evaluator.regression_report(y_test, preds)
        self.evaluator.print_regression_report(metrics)

        class_reports = self.evaluator.classification_report_all(ycat_test.values, pred_cats, config.CATEGORY_ORDER)
        self.evaluator.print_classification_reports(class_reports)

        self.visualizer.plot_confusion_matrices(
            pred_cats,
            ycat_test.values,
            config.CATEGORY_ORDER,
            accuracy_fn=lambda y_true, y_pred: class_reports[next(k for k,v in pred_cats.items() if list(v) == list(y_pred))]["accuracy"],
            confusion_fn=lambda y_true, y_pred, labels: next(v["cm"] for k, v in class_reports.items() if list(pred_cats[k]) == list(y_pred)),
        )

        model_names = list(pred_cats.keys())
        accs = [class_reports[n]["accuracy"] for n in model_names]
        r2s = [metrics[n]["R2"] for n in model_names]
        self.visualizer.plot_model_comparison(model_names, accs, r2s)

        config.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for model in self.models:
            filename = config.MODEL_FILE_MAP[model.get_name()]
            model.save(config.SAVED_MODELS_DIR / filename)
        self.pipeliner.save(config.PIPERLINER_FILE)

        sample_profile = self.profile_generator.generate_profile(df, seed=config.SEED, mode="random")
        profile_df = self.profile_generator.build(sample_profile)
        prediction = self.predictor.predict(profile_df, self.models, self.pipeliner)
        self.predictor.print_results(prediction)
