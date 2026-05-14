"""Tests for spectralcrop/data/preprocessing.py — vegetation index computation."""

import numpy as np
import xarray as xr

from spectralcrop.data.preprocessing import _nearest_band_in_range, add_vegetation_indices

# ---------------------------------------------------------------------------
# _nearest_band_in_range
# ---------------------------------------------------------------------------


def test_nearest_band_exact_match():
    wl = np.array([400.0, 550.0, 660.0, 800.0])
    assert _nearest_band_in_range(wl, 540.0, 560.0) == 1  # 550 nm


def test_nearest_band_fallback_outside_range():
    """When no band falls in the range, return the globally nearest band."""
    wl = np.array([400.0, 500.0, 900.0])
    idx = _nearest_band_in_range(wl, 660.0, 680.0)
    # Nearest to centre 670 nm is 500 nm (closer than 900 nm)
    assert idx == 1


def test_nearest_band_selects_centre():
    """When multiple bands are in range, pick the one closest to the centre."""
    wl = np.array([700.0, 710.0, 730.0, 740.0])
    # Range 705–735, centre = 720, closest = 710 (Δ10) vs 730 (Δ10) — tie → first
    idx = _nearest_band_in_range(wl, 705.0, 735.0)
    assert idx in (1, 2)  # both 710 and 730 are equidistant


# ---------------------------------------------------------------------------
# add_vegetation_indices (with a synthetic zarr store)
# ---------------------------------------------------------------------------


def _make_synthetic_zarr(tmp_path, n_bands: int = 10):
    """Create a minimal zarr store that mimics the real dataset structure."""
    import zarr

    H, W = 8, 8
    rng = np.random.default_rng(42)

    reflectance = rng.uniform(0.01, 0.5, (n_bands, H, W)).astype("float32")
    wavelengths = np.linspace(400, 900, n_bands).astype("float32")

    veg_mask = np.ones((H, W), dtype="uint8")
    mask_nodata = np.zeros((H, W), dtype="uint8")
    ndvi = np.full((H, W), 0.5, dtype="float32")

    store_path = str(tmp_path / "synthetic.zarr")
    # Use zarr.open_group with dimension_names (same pattern as hypercube_processor.py)
    root = zarr.open_group(store_path, mode="w")

    arr = root.create_array(
        "reflectance",
        shape=reflectance.shape,
        chunks=(n_bands, H, W),
        dtype="float32",
        dimension_names=["band", "y", "x"],
    )
    arr[:] = reflectance

    wl_arr = root.create_array(
        "wavelength", shape=wavelengths.shape, dtype="float32", dimension_names=["band"]
    )
    wl_arr[:] = wavelengths

    vm = root.create_array(
        "veg_mask", shape=veg_mask.shape, dtype="uint8", dimension_names=["y", "x"]
    )
    vm[:] = veg_mask

    nd = root.create_array(
        "mask_nodata", shape=mask_nodata.shape, dtype="uint8", dimension_names=["y", "x"]
    )
    nd[:] = mask_nodata

    nv = root.create_array("NDVI", shape=ndvi.shape, dtype="float32", dimension_names=["y", "x"])
    nv[:] = ndvi

    # Add dimension coordinates so xarray can open it
    xr.open_zarr(store_path)  # validates store is xarray-compatible
    return store_path, H, W


def test_add_vegetation_indices_creates_variables(tmp_path):
    """add_vegetation_indices must write NDRE, CIgreen, PRI, PSRI to the zarr."""
    from pathlib import Path

    import zarr

    store_path, H, W = _make_synthetic_zarr(tmp_path, n_bands=20)

    add_vegetation_indices(Path(store_path))

    store = zarr.open(store_path, mode="r")
    for vi in ("NDRE", "CIgreen", "PRI", "PSRI"):
        assert vi in store, f"{vi} not written to zarr"
        assert store[vi].shape == (H, W), f"Unexpected shape for {vi}"


def test_add_vegetation_indices_produces_finite_values(tmp_path):
    """All VI values for valid (non-zero reflectance) pixels must be finite."""
    from pathlib import Path

    import zarr

    store_path, H, W = _make_synthetic_zarr(tmp_path, n_bands=20)
    add_vegetation_indices(Path(store_path))

    store = zarr.open(store_path, mode="r")
    for vi in ("NDRE", "CIgreen", "PRI", "PSRI"):
        arr = store[vi][:]
        finite = arr[~np.isnan(arr)]
        assert len(finite) > 0, f"{vi} has no finite values"
        assert np.all(np.isfinite(finite)), f"{vi} contains inf values"
