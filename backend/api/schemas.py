from pydantic import BaseModel, Field


class StudentProfile(BaseModel):
    age: float = Field(..., gt=0)
    hours_work: float = Field(..., ge=0)
    social_media_use: float = Field(..., ge=0)
    rent: float = Field(..., ge=0)
    friends_count: float = Field(..., ge=0)
    highest_speed: float = Field(..., ge=0)
    dates: float = Field(..., ge=0)
    standard_drinks: float = Field(..., ge=0)
    countries: float = Field(..., ge=0)
    semesters: float = Field(..., ge=0)
    commute: float = Field(..., ge=0)
    data_interest: float = Field(..., ge=0)
    mark_goal: float = Field(..., ge=0)
    hours_studying: float = Field(..., ge=0)

    gender: str
    relationship_status: str
    drug_use_ans: str
    student_type: str
    mainstream_advanced: str
    lecture_mode: str
    study_type: str
    learner_style: str


class ModelPrediction(BaseModel):
    model_name: str
    category: str


class PredictionResponse(BaseModel):
    predictions: list[ModelPrediction]


class StudentProfileResponse(BaseModel):
    age: float
    hours_work: float
    social_media_use: float
    rent: float
    friends_count: float
    highest_speed: float
    dates: float
    standard_drinks: float
    countries: float
    semesters: float
    commute: float
    data_interest: float
    mark_goal: float
    hours_studying: float
    gender: str
    relationship_status: str
    drug_use_ans: str
    student_type: str
    mainstream_advanced: str
    lecture_mode: str
    study_type: str
    learner_style: str
