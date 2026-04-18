"""Model metadata endpoints."""

from fastapi import APIRouter

from src.api.model_loader import get_models_info
from src.api.schemas import ModelsInfoResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/info", response_model=ModelsInfoResponse)
async def models_info() -> dict[str, object]:
    """Return configured model tasks and loading status."""

    return get_models_info()
