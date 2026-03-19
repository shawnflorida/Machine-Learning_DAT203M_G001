import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.architecture.data_pipeline import DataCleaner, DataLoader, FeatureEngineer
from src.architecture.ml_tasks import EDA, Evaluator, Predictor
from src.architecture.ml_utils import Converters, Pipeliner, ProfileGenerator
from src.models import GradientBoostingModel, LogisticRegressionModel, NeuralNetworkModel
from src.architecture.visualizer import Visualizer


class Runner:
    def __init__(self):
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.converters = Converters()
        self.visualizer = Visualizer()
        self.eda = EDA(self.visualizer)
        self.evaluator = Evaluator()
        self.pipeliner = Pipeliner(config.ALL_NUMERIC, config.ALL_CATS)
        self.models = [
            LogisticRegressionModel(),
            NeuralNetworkModel(),
            GradientBoostingModel(),
        ]
        self.predictor = Predictor()
        self.profile_generator = ProfileGenerator(
            config.NUMERIC_COLS, config.CATEGORICAL_COLS,
            config.ALL_NUMERIC, config.ALL_CATS,
        )

    def run(self):
        # ── Load & prepare ────────────────────────────────────────────────
        raw = self.loader.load()
        raw = self.loader.filter_consent(raw)

        cols_needed = config.NUMERIC_COLS + config.CATEGORICAL_COLS + [config.TARGET]
        df = raw[cols_needed].copy()
        df = self.cleaner.clean(df, config.NUMERIC_COLS, config.CATEGORICAL_COLS, config.TARGET)
        df = self.engineer.engineer(df, config.DERIVED_COLS, config.TARGET, self.converters, config.TARGET_CATEGORY)

        df_eda, _ = self.converters.label_encode(df, config.CATEGORICAL_COLS)

        X_num = df[config.ALL_NUMERIC].copy()
        X_cat = raw.loc[df.index, config.ALL_CATS].fillna("Unknown")
        X = pd.concat([X_num, X_cat], axis=1)
        y = df[config.TARGET_CATEGORY]

        # ── Split ─────────────────────────────────────────────────────────
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y,
        )
        X_train = self.pipeliner.fit_transform(X_train_raw)
        X_test  = self.pipeliner.transform(X_test_raw)
        print(f"Train: {len(X_train)} | Test: {len(X_test)}")

        # ── EDA ──────────────────────────────────────────────────────────
        feature_cols = config.NUMERIC_COLS + config.DERIVED_COLS + config.CATEGORICAL_COLS
        self.eda.run(df, df_eda, feature_cols, config.TARGET, config.TARGET_CATEGORY, config.CATEGORY_ORDER)

        # ── Train ─────────────────────────────────────────────────────────
        for model in self.models:
            model.train(X_train, y_train.values)
            print(f"Trained: {model.get_name()}")

        # ── Evaluate ──────────────────────────────────────────────────────
        pred_cats = {m.get_name(): m.predict(X_test) for m in self.models}
        class_reports = self.evaluator.classification_report_all(y_test.values, pred_cats, config.CATEGORY_ORDER)
        self.evaluator.print_classification_reports(class_reports)
        self.visualizer.plot_confusion_matrices(class_reports, config.CATEGORY_ORDER)
        self.visualizer.plot_model_comparison(
            list(class_reports.keys()),
            [v["accuracy"] for v in class_reports.values()],
        )

        # ── Save ──────────────────────────────────────────────────────────
        config.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for model in self.models:
            model.save(config.SAVED_MODELS_DIR / config.MODEL_FILE_MAP[model.get_name()])
        self.pipeliner.save(config.PIPERLINER_FILE)

        # ── Sample prediction ─────────────────────────────────────────────
        sample = self.profile_generator.generate_profile(df, seed=config.SEED, mode="random")
        prediction = self.predictor.predict(self.profile_generator.build(sample), self.models, self.pipeliner)
        self.predictor.print_results(prediction)
