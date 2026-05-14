"""Inference endpoint — POST /predict."""

from __future__ import annotations

import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from app.schemas.request import PatchRequest
from app.schemas.response import PredictionResponse
from app.services.model_loader import ModelRegistry

router = APIRouter()

_LABEL_NAMES = {0: "Non-stressed", 1: "Stressed"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PatchRequest) -> PredictionResponse:
    """Classify a single 5×5×63 spectral patch.

    The patch must be pre-scaled with ``models/robust_scaler.pkl`` before
    sending.  Raw reflectance values will produce incorrect results.

    Returns the predicted class (0 = Non-stressed, 1 = Stressed) along
    with the model probability and decision threshold used.
    """
    if not ModelRegistry.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Convert nested list → (1, 63, 5, 5) tensor
    patch_np = np.array(request.patch, dtype=np.float32)  # (63, 5, 5)
    patch_t  = torch.from_numpy(patch_np).unsqueeze(0)    # (1, 63, 5, 5)

    prob_stressed, label = ModelRegistry.predict(patch_t)

    return PredictionResponse(
        label=label,
        label_name=_LABEL_NAMES[label],
        probability_stressed=round(prob_stressed, 4),
        threshold_used=round(ModelRegistry._threshold, 4),
    )
