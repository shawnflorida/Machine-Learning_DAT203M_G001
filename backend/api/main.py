from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from api.routes import meta_router, predict_router, profile_router
from api.state import ModelState
from src.architecture.data_pipeline import DataCleaner, DataLoader
from src.architecture.ml_tasks import Predictor
from src.architecture.ml_utils import Converters, Pipeliner, ProfileGenerator
from src.models import GradientBoostingModel, LogisticRegressionModel, NeuralNetworkModel, DecisionTreeModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    models = [LogisticRegressionModel(), NeuralNetworkModel(), GradientBoostingModel(), DecisionTreeModel()]
    for m in models:
        m.load_best(config.SAVED_MODELS_BEST_DIR / config.MODEL_FILE_MAP[m.get_name()])

    pipeliner = Pipeliner.load(config.PIPERLINER_FILE)
    profile_generator = ProfileGenerator(
        config.NUMERIC_COLS, config.CATEGORICAL_COLS,
        config.ALL_NUMERIC, config.ALL_CATS,
    )

    # Determine best model by accuracy on the test split
    test_df = pd.read_csv(config.DATA_DIR / "test.csv")
    X_test = pipeliner.transform(test_df[config.ALL_NUMERIC + config.ALL_CATS])
    y_test = test_df[config.TARGET_CATEGORY].values
    best_model = max(models, key=lambda m: (m.predict(X_test) == y_test).mean())
    print(f"Best model: {best_model.get_name()}")

    # Load source data for profile generation
    raw = DataLoader.load()
    raw = DataLoader.filter_consent(raw)
    cols = config.NUMERIC_COLS + config.CATEGORICAL_COLS + [config.TARGET]
    source_df = DataCleaner().clean(raw[cols].copy(), config.NUMERIC_COLS, config.CATEGORICAL_COLS, config.TARGET)

    app.state.model_state = ModelState(
        models=models,
        best_model=best_model,
        pipeliner=pipeliner,
        predictor=Predictor(),
        profile_generator=profile_generator,
        source_df=source_df,
    )
    yield


app = FastAPI(title="Student Stress Prediction API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(profile_router)
app.include_router(meta_router)
