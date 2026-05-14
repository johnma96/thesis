"""Patch extraction for CNN-2D.

Extracts fixed-size spatial patches centred on each valid pixel of a
given split.  Pixels whose neighbourhood extends beyond the labelled
region or contains NaN values are excluded.
"""

from __future__ import annotations

import numpy as np


def build_patch_dataset(
    cube: np.ndarray,
    labels_2d: np.ndarray,
    split_2d: np.ndarray,
    split_value: int,
    patch: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract CNN-2D patch dataset from a pre-scaled spectral cube.

    Parameters
    ----------
    cube : ndarray of shape (H, W, C)
        Scaled feature cube (H×W spatial, C channels).  NaN where invalid.
    labels_2d : ndarray of shape (H, W)
        Binary label map (0 / 1 / NaN).
    split_2d : ndarray of shape (H, W)
        Split assignment map (1=train, 2=val, 3=test, 0=unassigned).
    split_value : int
        Which split to extract patches for.
    patch : int, optional
        Spatial patch size (square).  Defaults to 5.

    Returns
    -------
    X_patches : ndarray of shape (N, C, patch, patch)
        Patch tensor in (channel-first) format expected by PyTorch Conv2d.
    y : ndarray of shape (N,) dtype int64
        Corresponding binary labels.
    """
    Hc, Wc, Bc = cube.shape
    r = patch // 2
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for i in range(r, Hc - r):
        for j in range(r, Wc - r):
            if split_2d[i, j] != split_value:
                continue
            if np.isnan(labels_2d[i, j]):
                continue
            patch_cube = cube[i - r : i + r + 1, j - r : j + r + 1, :]
            if np.isnan(patch_cube).any():
                continue
            X_list.append(np.transpose(patch_cube, (2, 0, 1)))  # (C, p, p)
            y_list.append(int(labels_2d[i, j]))

    if not X_list:
        raise ValueError(f"No valid patches found for split_value={split_value}.")

    return np.stack(X_list), np.array(y_list, dtype=np.int64)
