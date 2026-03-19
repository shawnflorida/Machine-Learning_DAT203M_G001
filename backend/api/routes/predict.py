from fastapi import APIRouter, Request

from api.schemas import PredictionResponse, StudentProfile

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(profile: StudentProfile, request: Request):
    state = request.app.state.model_state

    profile_df = state.profile_generator.build(profile.model_dump())
    predictions = state.predictor.predict(profile_df, state.models, state.pipeliner)

    return PredictionResponse(predictions=predictions)
