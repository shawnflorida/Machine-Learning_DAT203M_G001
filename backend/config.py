from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE_NAME = "data1001_survey_data_2025_S2-1.csv"
DATA_PATH = DATA_DIR / DATA_FILE_NAME

# Pre-split dataset files
TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
PIPERLINER_FILE = SAVED_MODELS_DIR / "pipeliner.pkl"

NUMERIC_COLS = [
    "age", "hours_work", "social_media_use", "rent",
    "friends_count", "highest_speed", "dates", "standard_drinks",
    "countries", "semesters", "commute", "data_interest",
    "mark_goal", "hours_studying",
]

CATEGORICAL_COLS = [
    "gender", "relationship_status", "drug_use_ans",
    "student_type", "mainstream_advanced", "lecture_mode",
    "study_type", "learner_style",
]

DERIVED_COLS = ["financial_pressure", "work_study_ratio", "social_engagement"]
TARGET = "stress"
TARGET_CATEGORY = "stress_category"
ALL_NUMERIC = NUMERIC_COLS + DERIVED_COLS
ALL_CATS = CATEGORICAL_COLS

CATEGORY_ORDER = ["Low", "Average", "High"]

SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.15  # 15% validation from training data, or use pre-split files

# Set to True to use pre-split files (train.csv, validation.csv, test.csv)
# Set to False to use dynamic split from raw data
USE_PRESPLIT_FILES = True

MODEL_FILE_MAP = {
    "Logistic Regression": "logistic_regression.pkl",
    "Neural Network":      "neural_network.pkl",
    "Gradient Boosting":   "gradient_boosting.pkl",
    "Decision Tree":       "decision_tree.pkl",
}

FRONTEND_ORIGIN = "http://localhost:5173"
