from fastapi import APIRouter, Request

from api.schemas import StudentProfileResponse

router = APIRouter(prefix="/api", tags=["profile"])


@router.get("/profile/random", response_model=StudentProfileResponse)
def random_profile(request: Request):
    """Return a randomly sampled student profile from the training dataset."""
    state = request.app.state.model_state
    profile = state.profile_generator.generate_profile(state.source_df, mode="random")
    return StudentProfileResponse(**profile)


@router.get("/profile/typical", response_model=StudentProfileResponse)
def typical_profile(request: Request):
    """Return a typical (median/mode) student profile derived from the dataset."""
    state = request.app.state.model_state
    profile = state.profile_generator.generate_profile(state.source_df, mode="typical")
    return StudentProfileResponse(**profile)
