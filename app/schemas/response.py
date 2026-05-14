"""Pydantic response schemas for the inference endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Classification result for a single spectral patch."""

    label: int = Field(
        ...,
        description="Predicted class: 0 = Non-stressed, 1 = Stressed.",
        examples=[1],
    )
    label_name: str = Field(
        ...,
        description="Human-readable class name.",
        examples=["Stressed"],
    )
    probability_stressed: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model probability for class 1 (stressed).",
        examples=[0.87],
    )
    threshold_used: float = Field(
        ...,
        description="Decision threshold applied to the probability.",
        examples=[0.3218],
    )
    model_version: str = Field(
        default="bean_stress_classifier/1",
        description="MLflow registered model name and version.",
    )


class HealthResponse(BaseModel):
    """API liveness response."""

    status: str = Field(default="ok")
    model_loaded: bool = Field(
        ..., description="Whether the CNN-2D model is loaded and ready."
    )
