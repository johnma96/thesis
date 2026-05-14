"""Vegetation index names and position utilities.

The five indices used in this project are stored at fixed positions
within the 63-channel feature vector:

    pos 0: NDVI    (Normalised Difference Vegetation Index)
    pos 1: NDRE    (Normalised Difference Red-Edge Index)
    pos 2: CIgreen (Chlorophyll Index Green)
    pos 3: PRI     (Photochemical Reflectance Index)
    pos 4: PSRI    (Plant Senescence Reflectance Index)

The actual index computation is performed in notebooks/203-jmmz-vegetation-indices.ipynb
and stored in the Zarr dataset variables ["NDVI", "NDRE", "CIgreen", "PRI", "PSRI"].
This module provides lookup utilities consumed by the ablation pipeline.
"""

from __future__ import annotations

from spectralcrop.config.constants import VI_INDICES, VI_NAMES

__all__ = ["VI_NAMES", "VI_INDICES", "vi_position", "vi_name"]


def vi_position(name: str) -> int:
    """Return the channel index of a vegetation index by name.

    Parameters
    ----------
    name : str
        One of ``["NDVI", "NDRE", "CIgreen", "PRI", "PSRI"]``.

    Raises
    ------
    KeyError
        If *name* is not a recognised vegetation index.
    """
    try:
        return VI_NAMES.index(name)
    except ValueError:
        raise KeyError(f"Unknown vegetation index: '{name}'. Valid: {VI_NAMES}") from None


def vi_name(position: int) -> str:
    """Return the vegetation index name at *position*."""
    if position not in VI_INDICES:
        raise IndexError(f"Position {position} is not a vegetation index channel (0–4).")
    return VI_NAMES[position]
