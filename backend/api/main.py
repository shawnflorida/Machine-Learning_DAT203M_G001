from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from api.routes import predict_router
from api.state import ModelState
from src.architecture.ml_tasks import Predictor
from src.architecture.ml_utils import Converters, Pipeliner, ProfileGenerator
from src.models import GradientBoostingModel, LinearRegressionModel, NeuralNetworkModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    converters = Converters(config.STRESS_BINS)
    models = [LinearRegressionModel(), NeuralNetworkModel(), GradientBoostingModel()]

    for m in models:
        file_name = config.MODEL_FILE_MAP[m.get_name()]
        m.load(config.SAVED_MODELS_DIR / file_name)

    pipeliner = Pipeliner.load(config.PIPERLINER_FILE)
    predictor = Predictor(converters)
    profile_generator = ProfileGenerator(
        config.NUMERIC_COLS,
        config.CATEGORICAL_COLS,
        config.ALL_NUMERIC,
        config.ALL_CATS,
    )

    app.state.model_state = ModelState(
        models=models,
        pipeliner=pipeliner,
        predictor=predictor,
        profile_generator=profile_generator,
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
