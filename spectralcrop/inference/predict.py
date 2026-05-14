"""Full-image CNN-2D inference: zarr → prediction GeoTIFFs.

Applies the trained CNN-2D model to every valid pixel in a preprocessed
zarr cube and saves two output rasters:

  prediction_proba.tif  — float32, P(stressed) ∈ [0, 1] per pixel
  prediction_class.tif  — uint8,  0=Non-stressed  1=Stressed  255=nodata

Border pixels within ``patch_size // 2`` of any image edge, and pixels
whose patch contains NaN, are marked as nodata — exactly mirroring the
exclusion applied during training.
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

from spectralcrop.config.constants import CNN2D_BEST_THR
from spectralcrop.config.paths import BANDS_SELECTED, MODELS_DIR
from spectralcrop.models.dl.architectures import SpectralSpatialCNN2D
from spectralcrop.models.dl.predict import load_cnn2d, predict_proba_2d

logger = logging.getLogger(__name__)

NODATA_PROBA: float = -1.0
NODATA_CLASS: int = 255


def predict_image(
    zarr_path: Path,
    output_dir: Path,
    model: SpectralSpatialCNN2D | None = None,
    scaler_path: Path | None = None,
    bands_csv: Path | None = None,
    threshold: float | None = None,
    patch_size: int = 5,
    batch_size: int = 256,
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
        Number of patches per inference batch.
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

    # Load model
    if model is None:
        logger.info("Loading CNN-2D model from %s", MODELS_DIR)
        model, _ = load_cnn2d(MODELS_DIR, device)

    scaler = joblib.load(scaler_path)
    selected_bands = pd.read_csv(bands_csv)["band_index"].tolist()

    # Load zarr
    logger.info("Loading zarr: %s", zarr_path)
    ds = xr.open_zarr(str(zarr_path), chunks={"band": 64, "y": 512, "x": 512})

    indices_da = ds[["NDVI", "NDRE", "CIgreen", "PRI", "PSRI"]].to_array(dim="band")
    refl_sel = ds["reflectance"].isel(band=selected_bands)
    features = xr.concat([indices_da, refl_sel], dim="band").transpose("y", "x", "band")
    veg_mask = ds["veg_mask"].astype("bool")
    nodata_mask = ds["mask_nodata"].astype("uint8")

    logger.info("Computing feature cube (may take a moment)...")
    X_cube = features.values.astype(np.float32)  # (H, W, 63)
    H, W, B = X_cube.shape

    # Scale entire cube (NaN propagates through linear scaler)
    cube_scaled = scaler.transform(X_cube.reshape(-1, B)).reshape(H, W, B)

    # Read spatial metadata for output TIFs
    crs_str = ds.attrs.get("crs", "EPSG:32618")
    tf_list = ds.attrs.get("transform", None)
    geo_transform = Affine(*tf_list[:6]) if tf_list is not None else None

    # Build output arrays (initialise with nodata)
    proba_map = np.full((H, W), NODATA_PROBA, dtype=np.float32)
    class_map = np.full((H, W), NODATA_CLASS, dtype=np.uint8)

    # Extract patches and run inference
    r = patch_size // 2
    logger.info("Extracting patches and running inference (r=%d, device=%s)...", r, device)

    patches, coords = [], []
    for i in range(r, H - r):
        for j in range(r, W - r):
            if not bool(veg_mask.values[i, j]):
                continue
            if bool(nodata_mask.values[i, j]):
                continue
            patch = cube_scaled[i - r : i + r + 1, j - r : j + r + 1, :]
            if np.isnan(patch).any():
                continue
            patches.append(np.transpose(patch, (2, 0, 1)))  # (C, p, p)
            coords.append((i, j))

    if not patches:
        logger.warning("No valid patches found in %s.", zarr_path)
        return _write_tifs(proba_map, class_map, output_dir, H, W, crs_str, geo_transform)

    X_patches = np.stack(patches)  # (N, C, p, p)
    logger.info("Running model on %d patches...", len(X_patches))
    probs = predict_proba_2d(model, X_patches, device=device, batch_size=batch_size)

    for (i, j), prob in zip(coords, probs, strict=False):
        proba_map[i, j] = float(prob)
        class_map[i, j] = 1 if prob >= threshold else 0

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
