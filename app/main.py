"""FastAPI application factory for the spectralcrop inference API.

To run locally:
    uv sync --extra pytorch-cu126 --extra api
    uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health        — liveness probe
    POST /predict       — classify a single 5×5×63 spectral patch
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.routers import health, inference
from app.services.model_loader import ModelRegistry


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Load the CNN-2D model at startup; release at shutdown."""
    ModelRegistry.load()
    yield
    ModelRegistry.unload()


app = FastAPI(
    title="spectralcrop — P-deficiency Classifier",
    description=(
        "Non-invasive diagnosis of phosphorus deficiency stress in common beans "
        "(Phaseolus vulgaris L.) using hyperspectral imagery and CNN-2D.\n\n"
        "**Thesis:** Montoya Zapata, J. M. (2026). UNAL.\n"
        "**Model:** SpectralSpatialCNN2D — PR-AUC = 0.9635 on held-out test set."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["Health"])
app.include_router(inference.router, tags=["Inference"])
