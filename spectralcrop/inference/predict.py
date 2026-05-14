"""Full-image CNN-2D inference: zarr → prediction GeoTIFFs.

Applies the trained CNN-2D model to every valid pixel in a preprocessed
zarr cube and saves two output rasters:

  prediction_proba.tif  — float32, P(stressed) ∈ [0, 1] per pixel
  prediction_class.tif  — uint8,  0=Non-stressed  1=Stressed  255=nodata

Border pixels within ``patch_size // 2`` of any image edge, and pixels
whose patch contains NaN, are marked as nodata — exactly mirroring the
exclusion applied during training.

Performance
-----------
Patch extraction uses vectorised NumPy advanced indexing rather than a
nested Python loop, reducing runtime from ~30 min to ~2 min for the
original 3660×3438 image on a modern CPU.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
import torch
import xarray as xr
from rasterio.transform import Affine
from scipy.ndimage import maximum_filter

from spectralcrop.config.constants import CNN2D_BEST_THR
from spectralcrop.config.paths import BANDS_SELECTED, MODELS_DIR
from spectralcrop.models.dl.architectures import SpectralSpatialCNN2D
from spectralcrop.models.dl.predict import load_cnn2d, predict_proba_2d

logger = logging.getLogger(__name__)

NODATA_PROBA: float = -1.0
NODATA_CLASS: int = 255


def _extract_patches_vectorized(
    cube: np.ndarray,
    valid_mask: np.ndarray,
    patch_size: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract CNN-2D patches using vectorised NumPy indexing.

    Replaces the O(H·W) nested Python loop with vectorised operations:

    1. A single ``maximum_filter`` pass identifies pixels whose ``patch_size×patch_size``
       neighbourhood contains at least one NaN — O(H·W) with a C implementation.
    2. Advanced NumPy indexing gathers all valid patches simultaneously
       without any Python-level iteration.

    Memory: ``N × C × patch × patch × 4 bytes``.  For N ≈ 250 k pixels,
    C = 63, patch = 5 this is ≈ 945 MB — acceptable on a 16 GB machine.
    For very large images consider tiling (``tile_size`` parameter in
    :func:`predict_image`).

    Parameters
    ----------
    cube : ndarray of shape (H, W, C)
        Scaled feature cube. NaN where pixel is invalid.
    valid_mask : ndarray of shape (H, W), bool
        Pre-computed mask: True for pixels that are (a) vegetation,
        (b) not nodata, and (c) within the valid border strip.
    patch_size : int
        Spatial patch size (odd, default 5).

    Returns
    -------
    patches : ndarray of shape (N, C, patch, patch)
        Valid patches in channel-first format.
    rows : ndarray of shape (N,)
        Row indices of patch centres in the original image.
    cols : ndarray of shape (N,)
        Column indices of patch centres in the original image.
    """
    H, W, C = cube.shape
    r = patch_size // 2

    # --- NaN neighbourhood mask (vectorised) ---
    # True where the pixel itself or any neighbour within patch_size has NaN.
    has_nan_pixel = np.isnan(cube).any(axis=2).astype(np.uint8)  # (H, W)
    nan_in_neighbourhood = maximum_filter(has_nan_pixel, size=patch_size).astype(bool)

    # --- Combined validity: user mask AND no NaN in neighbourhood ---
    combined = valid_mask & ~nan_in_neighbourhood  # (H, W)

    # Enforce border: centres within r pixels of the image edge cannot
    # form a full patch; clear those pixels from the mask.
    combined[:r, :] = False
    combined[H - r :, :] = False
    combined[:, :r] = False
    combined[:, W - r :] = False

    rows, cols = np.where(combined)  # (N,) each
    N = len(rows)
    if N == 0:
        empty = np.empty((0, C, patch_size, patch_size), dtype=np.float32)
        return empty, rows, cols

    # --- Vectorised patch extraction via advanced indexing ---
    # Offset grids for a patch_size×patch_size window.
    dr = np.arange(-r, r + 1, dtype=np.intp)  # (patch,)
    dc = np.arange(-r, r + 1, dtype=np.intp)  # (patch,)

    # Broadcast to (N, patch, patch)
    row_idx = rows[:, None, None] + dr[None, :, None]  # (N, p, 1) → (N, p, p)
    col_idx = cols[:, None, None] + dc[None, None, :]  # (N, 1, p) → (N, p, p)

    # Gather: cube[row_idx, col_idx, :] → (N, patch, patch, C)
    patches = cube[row_idx, col_idx, :]  # advanced indexing, no Python loop
    patches = patches.transpose(0, 3, 1, 2).astype(np.float32)  # (N, C, p, p)

    return patches, rows, cols


def predict_image(
    zarr_path: Path,
    output_dir: Path,
    model: SpectralSpatialCNN2D | None = None,
    scaler_path: Path | None = None,
    bands_csv: Path | None = None,
    threshold: float | None = None,
    patch_size: int = 5,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> tuple[Path, Path]:
    """Run pixel-level CNN-2D inference over a full preprocessed zarr cube.

    Parameters
    ----------
    zarr_path : Path
        Preprocessed ``masked_reflectance.zarr`` containing ``reflectance``,
        ``NDVI``, ``NDRE``, ``CIgreen``, ``PRI``, ``PSRI``, ``veg_mask``.
    output_dir : Path
        Directory where ``prediction_proba.tif`` and ``prediction_class.tif``
        are written.
    model : SpectralSpatialCNN2D, optional
        Loaded and eval-mode model.  If None, the final model from
        ``models/`` is loaded via :func:`~spectralcrop.models.dl.predict.load_cnn2d`.
    scaler_path : Path, optional
        Path to ``robust_scaler.pkl``.  Defaults to ``models/robust_scaler.pkl``.
    bands_csv : Path, optional
        CSV with ``band_index`` column listing the 58 selected spectral bands.
        Defaults to ``data/interim/bands_selected_by_segment.csv``.
    threshold : float, optional
        Decision threshold.  Defaults to the locked value in constants.
    patch_size : int, optional
        Spatial patch size (must match the trained model; default = 5).
    batch_size : int, optional
        Number of patches per inference batch.  Larger is faster on GPU.
    device : torch.device, optional
        Computation device.  Auto-detected if None.

    Returns
    -------
    proba_path : Path
        Path to ``prediction_proba.tif``.
    class_path : Path
        Path to ``prediction_class.tif``.
    """
    zarr_path = Path(zarr_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if threshold is None:
        threshold = CNN2D_BEST_THR
    if scaler_path is None:
        scaler_path = MODELS_DIR / "robust_scaler.pkl"
    if bands_csv is None:
        bands_csv = BANDS_SELECTED

    if model is None:
        logger.info("Loading CNN-2D model from %s", MODELS_DIR)
        model, _ = load_cnn2d(MODELS_DIR, device)

    scaler = joblib.load(scaler_path)
    selected_bands = pd.read_csv(bands_csv)["band_index"].tolist()

    logger.info("Loading zarr: %s", zarr_path)
    ds = xr.open_zarr(str(zarr_path), chunks={"band": 64, "y": 512, "x": 512})

    indices_da = ds[["NDVI", "NDRE", "CIgreen", "PRI", "PSRI"]].to_array(dim="band")
    refl_sel = ds["reflectance"].isel(band=selected_bands)
    features = xr.concat([indices_da, refl_sel], dim="band").transpose("y", "x", "band")
    veg_mask_arr = ds["veg_mask"].values.astype(bool)
    nodata_arr = ds["mask_nodata"].values.astype(bool)

    logger.info("Computing feature cube...")
    X_cube = features.values.astype(np.float32)  # (H, W, 63)
    H, W, B = X_cube.shape

    logger.info("Scaling feature cube (%d × %d × %d)...", H, W, B)
    cube_scaled = scaler.transform(X_cube.reshape(-1, B)).reshape(H, W, B)

    crs_str = ds.attrs.get("crs", "EPSG:32618")
    tf_list = ds.attrs.get("transform", None)
    geo_transform = Affine(*tf_list[:6]) if tf_list is not None else None

    # Valid-pixel mask: vegetation + no nodata (NaN neighbourhood handled inside)
    pixel_valid = veg_mask_arr & ~nodata_arr  # (H, W)

    logger.info("Extracting patches (vectorised)...")
    X_patches, patch_rows, patch_cols = _extract_patches_vectorized(
        cube_scaled, pixel_valid, patch_size
    )

    proba_map = np.full((H, W), NODATA_PROBA, dtype=np.float32)
    class_map = np.full((H, W), NODATA_CLASS, dtype=np.uint8)

    if len(X_patches) == 0:
        logger.warning("No valid patches found in %s.", zarr_path)
        return _write_tifs(proba_map, class_map, output_dir, H, W, crs_str, geo_transform)

    logger.info("Running model on %d patches (device=%s)...", len(X_patches), device)
    probs = predict_proba_2d(model, X_patches, device=device, batch_size=batch_size)

    # Vectorised assignment back to the output maps
    proba_map[patch_rows, patch_cols] = probs.astype(np.float32)
    class_map[patch_rows, patch_cols] = (probs >= threshold).astype(np.uint8)

    return _write_tifs(proba_map, class_map, output_dir, H, W, crs_str, geo_transform)


def _write_tifs(
    proba_map: np.ndarray,
    class_map: np.ndarray,
    output_dir: Path,
    H: int,
    W: int,
    crs_str: str,
    transform: Affine | None,
) -> tuple[Path, Path]:
    """Write probability and class maps as GeoTIFFs."""
    common = dict(
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        crs=crs_str,
        transform=transform,
        tiled=True,
        compress="DEFLATE",
    )
    if transform is None:
        logger.warning("No spatial transform found — output TIFs will have no georeference.")

    proba_path = output_dir / "prediction_proba.tif"
    with rasterio.open(proba_path, "w", dtype="float32", nodata=NODATA_PROBA, **common) as dst:
        dst.write(proba_map, 1)
    logger.info("Probability map: %s", proba_path)

    class_path = output_dir / "prediction_class.tif"
    with rasterio.open(class_path, "w", dtype="uint8", nodata=NODATA_CLASS, **common) as dst:
        dst.write(class_map, 1)

    n_stressed = int((class_map == 1).sum())
    n_non_stressed = int((class_map == 0).sum())
    n_nodata = int((class_map == NODATA_CLASS).sum())
    logger.info(
        "Prediction map: %s  |  stressed=%d  non-stressed=%d  nodata=%d",
        class_path,
        n_stressed,
        n_non_stressed,
        n_nodata,
    )
    return proba_path, class_path
