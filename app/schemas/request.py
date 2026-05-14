"""Pydantic request schemas for the inference endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PatchRequest(BaseModel):
    """A single 5×5×63 spectral patch for stress classification.

    The patch must be centred on the pixel of interest and pre-scaled
    with the RobustScaler fitted during training
    (``models/robust_scaler.pkl``).

    Shape convention: channel-first → ``[channel][row][col]``
    with ``channel ∈ [0, 62]``, ``row ∈ [0, 4]``, ``col ∈ [0, 4]``.
    """

    patch: list[list[list[float]]] = Field(
        ...,
        description=(
            "Scaled spectral patch of shape [63][5][5]. "
            "Channel order: [NDVI, NDRE, CIgreen, PRI, PSRI, band_0 … band_57]."
        ),
        examples=[
            {
                "patch": [[[0.0] * 5 for _ in range(5)] for _ in range(63)]
            }
        ],
    )

    @field_validator("patch")
    @classmethod
    def validate_shape(cls, v: list) -> list:
        if len(v) != 63:
            raise ValueError(f"Expected 63 channels, got {len(v)}.")
        for ch_idx, channel in enumerate(v):
            if len(channel) != 5:
                raise ValueError(f"Channel {ch_idx}: expected 5 rows, got {len(channel)}.")
            for row_idx, row in enumerate(channel):
                if len(row) != 5:
                    raise ValueError(
                        f"Channel {ch_idx}, row {row_idx}: expected 5 cols, got {len(row)}."
                    )
        return v
