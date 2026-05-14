"""Health / liveness endpoint."""

from fastapi import APIRouter
from app.schemas.response import HealthResponse
from app.services.model_loader import ModelRegistry

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return API liveness status and model readiness."""
    return HealthResponse(status="ok", model_loaded=ModelRegistry.is_loaded())
