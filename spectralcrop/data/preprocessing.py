"""Preprocessing pipeline: raw ENVI hypercube → masked Zarr + vegetation indices.

Orchestrates two steps:
  1. `HypercubeProcessor.save_reflectance_to_zarr_fast` → reflectance + NDVI + veg_mask
  2. `add_vegetation_indices` → writes NDRE, CIgreen, PRI, PSRI to the existing zarr

The output zarr contains 7 + N variables ready for downstream modelling:
  reflectance (band, y, x), wavelength, fwhm, NDVI, NDRE, CIgreen, PRI, PSRI,
  veg_mask, mask_nodata.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def _nearest_band_in_range(
    wavelengths: np.ndarray,
    wl_min: float,
    wl_max: float,
) -> int:
    """Return the band index whose wavelength is closest to the centre of [wl_min, wl_max]."""
    centre = (wl_min + wl_max) / 2.0
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    if mask.any():
        idxs = np.where(mask)[0]
        return int(idxs[np.argmin(np.abs(wavelengths[idxs] - centre))])
    # Fallback: globally nearest band
    return int(np.argmin(np.abs(wavelengths - centre)))


def add_vegetation_indices(zarr_path: Path) -> None:
    """Compute NDRE, CIgreen, PRI, PSRI and write them to an existing zarr store.

    NDVI is already present from `save_reflectance_to_zarr_fast` (add_ndvi=True).
    This function adds the four remaining indices required by the 63-feature vector.

    Parameters
    ----------
    zarr_path : Path
        Path to an existing ``masked_reflectance.zarr`` produced by
        :func:`preprocess_envi_to_zarr`.
    """
    logger.info("Opening zarr: %s", zarr_path)
    ds = xr.open_zarr(str(zarr_path), chunks={"band": 64, "y": 512, "x": 512})
    refl_byx = ds["reflectance"].transpose("y", "x", "band")
    wl = ds["wavelength"].values.astype(float)
    veg_mask = ds["veg_mask"].astype(bool)
    nodata = ds["mask_nodata"].astype(np.uint8)
    valid = veg_mask & (nodata == 0)

    # Band lookup
    b_green = _nearest_band_in_range(wl, 540, 560)
    b_re = _nearest_band_in_range(wl, 705, 740)
    b_nir = _nearest_band_in_range(wl, 780, 840)
    b_531 = _nearest_band_in_range(wl, 526, 536)
    b_570 = _nearest_band_in_range(wl, 565, 575)
    b_500 = _nearest_band_in_range(wl, 495, 505)
    b_680 = _nearest_band_in_range(wl, 675, 685)
    b_750 = _nearest_band_in_range(wl, 745, 755)

    logger.info(
        "Band lookup — green=%.0f  RE=%.0f  NIR=%.0f  531=%.0f  570=%.0f  "
        "500=%.0f  680=%.0f  750=%.0f nm",
        wl[b_green],
        wl[b_re],
        wl[b_nir],
        wl[b_531],
        wl[b_570],
        wl[b_500],
        wl[b_680],
        wl[b_750],
    )

    def _eps_div(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
        return a / xr.where(np.abs(b) < 1e-6, np.nan, b)

    R_green = refl_byx.isel(band=b_green)
    R_re = refl_byx.isel(band=b_re)
    R_nir = refl_byx.isel(band=b_nir)
    R_531 = refl_byx.isel(band=b_531)
    R_570 = refl_byx.isel(band=b_570)
    R_500 = refl_byx.isel(band=b_500)
    R_680 = refl_byx.isel(band=b_680)
    R_750 = refl_byx.isel(band=b_750)

    indices = xr.Dataset(
        {
            "NDRE": _eps_div(R_nir - R_re, R_nir + R_re)
            .where(valid)
            .astype("float32")
            .rename("NDRE"),
            "CIgreen": (_eps_div(R_nir, R_green) - 1.0)
            .where(valid)
            .astype("float32")
            .rename("CIgreen"),
            "PRI": _eps_div(R_531 - R_570, R_531 + R_570)
            .where(valid)
            .astype("float32")
            .rename("PRI"),
            "PSRI": _eps_div(R_680 - R_500, R_750).where(valid).astype("float32").rename("PSRI"),
        }
    )

    logger.info("Writing NDRE, CIgreen, PRI, PSRI to zarr...")
    indices.to_zarr(str(zarr_path), mode="a")
    logger.info("Vegetation indices written.")


def preprocess_envi_to_zarr(
    hdr_path: Path,
    output_zarr: Path,
    ndvi_threshold: float = 0.3,
    chunks: tuple[int, int, int] = (64, 512, 512),
    water_windows: tuple[tuple[float, float], ...] = ((1340.0, 1440.0), (1800.0, 1950.0)),
    scale: float = 10000.0,
    ignore_vals: tuple[float, ...] = (-1.0, 15000.0),
    resume: bool = False,
) -> None:
    """Process a raw ENVI hypercube to a Zarr store with reflectance + all 5 VI.

    This function orchestrates the complete preprocessing pipeline:
    1. Opens the ENVI file (.hdr + .bsq/.bil/.bip) via the ``spectral`` library.
    2. Creates a :class:`~spectralcrop.data.HypercubeProcessor` and writes the
       full reflectance cube + NDVI + vegetation mask to ``output_zarr``.
    3. Appends NDRE, CIgreen, PRI, PSRI via :func:`add_vegetation_indices`.

    Parameters
    ----------
    hdr_path : Path
        Path to the ENVI ``.hdr`` header file.  The companion binary file must
        be in the same directory.
    output_zarr : Path
        Destination zarr store (created or updated if ``resume=True``).
    ndvi_threshold : float, optional
        NDVI threshold above which pixels are considered vegetation.
    chunks : tuple of int, optional
        Zarr chunk shape ``(band, y, x)``.
    water_windows : tuple of (float, float), optional
        NIR water-absorption windows (nm) to exclude from the exported bands.
    scale : float, optional
        Reflectance scale factor from the ENVI header (typically 10 000).
    ignore_vals : tuple of float, optional
        Raw DN values to mask as no-data (e.g. background = −1, ignore = 15 000).
    resume : bool, optional
        If True, skip bands that were already written to the zarr.
    """
    try:
        import spectral
    except ImportError as err:
        raise ImportError(
            "The 'spectral' package is required for ENVI reading. "
            "Install it with: uv sync --extra notebooks"
        ) from err

    from spectralcrop.data.hypercube_processor import HypercubeProcessor

    hdr_path = Path(hdr_path)
    if not hdr_path.exists():
        raise FileNotFoundError(f"ENVI header not found: {hdr_path}")

    logger.info("Opening ENVI file: %s", hdr_path)
    img = spectral.open_image(str(hdr_path))
    cube = img.load()

    # Extract metadata from header
    hdr_meta = img.metadata
    wavelengths = np.array([float(w) for w in hdr_meta.get("wavelength", [])], dtype=np.float32)
    fwhm = np.array([float(f) for f in hdr_meta.get("fwhm", [])], dtype=np.float32)
    scale_hdr = float(hdr_meta.get("reflectance scale factor", scale))

    logger.info(
        "Cube: %d bands × %d lines × %d samples  |  scale=%.0f  |  λ=[%.0f–%.0f nm]",
        img.nbands,
        img.nrows,
        img.ncols,
        scale_hdr,
        wavelengths[0] if len(wavelengths) else 0,
        wavelengths[-1] if len(wavelengths) else 0,
    )

    processor = HypercubeProcessor(
        img=img,
        cube=cube,
        wavelengths=wavelengths if len(wavelengths) else None,
        fwhm=fwhm if len(fwhm) else None,
        scale=scale_hdr,
        ignore_vals=tuple(ignore_vals),
        water_windows=water_windows,
    )

    logger.info("Writing reflectance + NDVI to zarr: %s", output_zarr)
    processor.save_reflectance_to_zarr_fast(
        zarr_path=str(output_zarr),
        exclude_water=True,
        chunks=chunks,
        add_coords=True,
        add_ndvi=True,
        ndvi_threshold=ndvi_threshold,
        strategy="tiled",
        tile=(512, 512),
        resume=resume,
        atomic_swap=False,
    )
    logger.info("Reflectance written.")

    # Step 2: add the remaining 4 vegetation indices
    add_vegetation_indices(output_zarr)
    logger.info("Preprocessing complete: %s", output_zarr)
