"""Tests for the vectorised patch extraction in spectralcrop/inference/predict.py."""

import numpy as np

from spectralcrop.inference.predict import _extract_patches_vectorized


def _cube(h: int = 20, w: int = 20, c: int = 63, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((h, w, c)).astype(np.float32)


def _full_valid(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


# ---------------------------------------------------------------------------
# Shape correctness
# ---------------------------------------------------------------------------


def test_output_shape_all_valid():
    cube = _cube(20, 20)
    valid = _full_valid(20, 20)
    patches, rows, cols = _extract_patches_vectorized(cube, valid, patch_size=5)
    # 5×5 patch, r=2 → valid centres are [2,18)×[2,18) = 16×16 = 256
    assert patches.shape == (256, 63, 5, 5)
    assert rows.shape == (256,)
    assert cols.shape == (256,)


def test_output_dtype_float32():
    cube = _cube().astype(np.float64)  # intentionally float64 input
    patches, _, _ = _extract_patches_vectorized(cube, _full_valid(20, 20))
    assert patches.dtype == np.float32


def test_channel_first_ordering():
    """Verify channel ordering: patches[:, c, :, :] == cube at that channel."""
    cube = _cube(10, 10, c=3)
    valid = _full_valid(10, 10)
    patches, rows, cols = _extract_patches_vectorized(cube, valid, patch_size=3)
    # For the first valid patch centre (row=1, col=1), verify the 3×3 patch
    r, c_idx = 1, 1
    expected = np.transpose(cube[r - 1 : r + 2, c_idx - 1 : c_idx + 2, :], (2, 0, 1))
    actual = patches[0]
    np.testing.assert_allclose(actual, expected)


# ---------------------------------------------------------------------------
# Mask handling
# ---------------------------------------------------------------------------


def test_border_pixels_excluded():
    """All output centres must be strictly inside the border strip."""
    cube = _cube(15, 15)
    valid = _full_valid(15, 15)
    patches, rows, cols = _extract_patches_vectorized(cube, valid, patch_size=5)
    r = 2
    assert np.all(rows >= r) and np.all(rows < 15 - r)
    assert np.all(cols >= r) and np.all(cols < 15 - r)


def test_validity_mask_respected():
    cube = _cube(10, 10)
    valid = np.zeros((10, 10), dtype=bool)
    valid[5, 5] = True  # single valid centre
    patches, rows, cols = _extract_patches_vectorized(cube, valid, patch_size=3)
    assert len(patches) == 1
    assert rows[0] == 5 and cols[0] == 5


def test_nan_neighbourhood_excluded():
    """A patch centre whose 5×5 window contains a NaN must be excluded."""
    cube = _cube(15, 15)
    cube[7, 7, 0] = np.nan  # inject NaN
    valid = _full_valid(15, 15)
    patches_with_nan, _, _ = _extract_patches_vectorized(cube, valid, patch_size=5)

    clean_cube = _cube(15, 15)
    patches_clean, _, _ = _extract_patches_vectorized(clean_cube, valid, patch_size=5)

    # Fewer patches when NaN is present
    assert len(patches_with_nan) < len(patches_clean)
    # No remaining patch should contain NaN
    assert not np.isnan(patches_with_nan).any()


def test_empty_mask_returns_empty():
    cube = _cube(10, 10)
    valid = np.zeros((10, 10), dtype=bool)
    patches, rows, cols = _extract_patches_vectorized(cube, valid, patch_size=5)
    assert len(patches) == 0
    assert patches.shape[1:] == (63, 5, 5)  # shape consistent even when empty


# ---------------------------------------------------------------------------
# Numerical consistency vs. loop implementation
# ---------------------------------------------------------------------------


def test_matches_loop_implementation():
    """Vectorised output must match the original loop-based extraction."""
    cube = _cube(12, 12)
    # inject one NaN to test NaN handling consistency
    cube[6, 6, 2] = np.nan
    valid = _full_valid(12, 12)
    patch_size = 5
    r = patch_size // 2

    # Reference: loop implementation
    ref_patches, ref_rows, ref_cols = [], [], []
    for i in range(r, 12 - r):
        for j in range(r, 12 - r):
            if not valid[i, j]:
                continue
            p = cube[i - r : i + r + 1, j - r : j + r + 1, :]
            if np.isnan(p).any():
                continue
            ref_patches.append(np.transpose(p, (2, 0, 1)).astype(np.float32))
            ref_rows.append(i)
            ref_cols.append(j)

    vec_patches, vec_rows, vec_cols = _extract_patches_vectorized(cube, valid, patch_size)

    assert len(vec_patches) == len(ref_patches)
    np.testing.assert_array_equal(vec_rows, np.array(ref_rows))
    np.testing.assert_array_equal(vec_cols, np.array(ref_cols))
    np.testing.assert_allclose(vec_patches, np.stack(ref_patches), atol=1e-6)
