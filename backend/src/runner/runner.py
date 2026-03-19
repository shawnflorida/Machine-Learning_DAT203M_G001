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
        train_ratio = 0.75
        validation_ratio = 0.15
        test_ratio = 0.10

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

        # ── Use designated split files when available ────────────────────
        train_path = config.DATA_DIR / "train.csv"
        validation_path = config.DATA_DIR / "validation.csv"
        test_path = config.DATA_DIR / "test.csv"
        split_cols = config.ALL_NUMERIC + config.ALL_CATS + [config.TARGET_CATEGORY]

        if train_path.exists() and validation_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            validation_df = pd.read_csv(validation_path)
            test_df = pd.read_csv(test_path)

            missing = [c for c in split_cols if c not in train_df.columns or c not in validation_df.columns or c not in test_df.columns]
            if missing:
                raise ValueError(f"Split files are missing required columns: {missing}")

            train_df = train_df[split_cols]
            validation_df = validation_df[split_cols]
            test_df = test_df[split_cols]
            print("Using existing split files: train.csv, validation.csv, test.csv")
        else:
            # First split creates 75% train and 25% temp.
            X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
                X,
                y,
                test_size=1 - train_ratio,
                random_state=config.SEED,
                stratify=y,
            )

            # Split temp into validation/test so final proportions are 15%/10%.
            X_val_raw, X_test_raw, y_val, y_test = train_test_split(
                X_temp_raw,
                y_temp,
                test_size=test_ratio / (test_ratio + validation_ratio),
                random_state=config.SEED,
                stratify=y_temp,
            )

            # Save tabular splits for reuse/debugging.
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            train_df = X_train_raw.copy()
            train_df[config.TARGET_CATEGORY] = y_train.values
            train_df.to_csv(train_path, index=False)

            validation_df = X_val_raw.copy()
            validation_df[config.TARGET_CATEGORY] = y_val.values
            validation_df.to_csv(validation_path, index=False)

            test_df = X_test_raw.copy()
            test_df[config.TARGET_CATEGORY] = y_test.values
            test_df.to_csv(test_path, index=False)
            print("Created split files: train.csv, validation.csv, test.csv")

        X_train_raw = train_df[config.ALL_NUMERIC + config.ALL_CATS]
        y_train = train_df[config.TARGET_CATEGORY]
        X_val_raw = validation_df[config.ALL_NUMERIC + config.ALL_CATS]
        y_val = validation_df[config.TARGET_CATEGORY]
        X_test_raw = test_df[config.ALL_NUMERIC + config.ALL_CATS]
        y_test = test_df[config.TARGET_CATEGORY]
        
        X_train = self.pipeliner.fit_transform(X_train_raw)
        X_val = self.pipeliner.transform(X_val_raw)
        X_test  = self.pipeliner.transform(X_test_raw)
        print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

        # ── EDA ──────────────────────────────────────────────────────────
        feature_cols = config.NUMERIC_COLS + config.DERIVED_COLS + config.CATEGORICAL_COLS
        self.eda.run(df, df_eda, feature_cols, config.TARGET, config.TARGET_CATEGORY, config.CATEGORY_ORDER)

        # ── Train ─────────────────────────────────────────────────────────
        # for model in self.models:
        #     model.train(X_train, y_train.values, X_val, y_val.values)
        #     print(f"Trained: {model.get_name()}")

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
