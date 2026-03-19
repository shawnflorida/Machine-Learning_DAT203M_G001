from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/models")
def list_models():
    from config import MODEL_FILE_MAP
    return {"models": list(MODEL_FILE_MAP.keys())}
