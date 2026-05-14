"""Smoke tests for patch extraction (CNN-2D)."""

import numpy as np
import pytest

from spectralcrop.features.patches import build_patch_dataset


def _make_synthetic_cube(h=20, w=20, c=63, seed=42):
    rng = np.random.default_rng(seed)
    cube = rng.random((h, w, c)).astype(np.float32)
    labels = np.zeros((h, w))
    labels[5:15, 5:15] = 1.0  # stressed region
    split = np.zeros((h, w), dtype=np.uint8)
    split[5:15, 5:15] = 3  # test split
    return cube, labels, split


def test_patch_shape():
    cube, labels, split = _make_synthetic_cube()
    X, y = build_patch_dataset(cube, labels, split, split_value=3, patch=5)
    assert X.ndim == 4, "Expected 4-D array (N, C, H, W)"
    assert X.shape[1] == 63
    assert X.shape[2] == 5
    assert X.shape[3] == 5
    assert y.ndim == 1
    assert len(X) == len(y)
    assert X.dtype == np.float32


def test_patch_excludes_border():
    """Pixels within 2 of the IMAGE border must not appear in the output.

    Image: 20×20. Split region: [5:15, 5:15] = 100 pixels. patch=5 → r=2.
    Valid center range: [2, 18). The split region is entirely within [2, 18),
    so all 100 pixels pass the border check.

    To test actual exclusion, put pixels at the image edge inside the split.
    """
    # Put split region touching the image border
    rng = np.random.default_rng(42)
    cube = rng.random((10, 10, 63)).astype(np.float32)
    labels = np.ones((10, 10))  # all stressed
    split = np.full((10, 10), 3, dtype=np.uint8)  # all test split
    X, y = build_patch_dataset(cube, labels, split, split_value=3, patch=5)
    # patch=5, r=2: valid centers are [2, 8) × [2, 8) = 6×6 = 36 pixels
    assert len(X) == 36, f"Expected 36 valid patch centers (image 10×10, r=2), got {len(X)}"


def test_nan_patches_excluded():
    """Patches whose 5×5 neighbourhood contains a NaN must be excluded."""
    rng = np.random.default_rng(42)
    cube = rng.random((10, 10, 63)).astype(np.float32)
    labels = np.ones((10, 10))
    split = np.full((10, 10), 3, dtype=np.uint8)
    # Inject NaN at position (5, 5) — inside the valid center range [2, 8)
    cube[5, 5, 0] = np.nan
    X, y = build_patch_dataset(cube, labels, split, split_value=3, patch=5)
    # Without NaN: 36 patches; NaN at (5,5) removes patches centred on
    # [3,8) × [3,8) = 25 pixels (all centres whose 5×5 window includes (5,5))
    assert len(X) < 36, "Patches containing NaN should be excluded"


def test_empty_split_raises():
    cube, labels, split = _make_synthetic_cube()
    with pytest.raises(ValueError, match="No valid patches"):
        build_patch_dataset(cube, labels, split, split_value=99, patch=5)
