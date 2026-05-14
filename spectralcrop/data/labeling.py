"""Label rasterisation: GeoPackage parcel polygons → labels_multiclass.tif.

Replicates the logic of notebooks/103-jmmz-labels.ipynb programmatically.

Label mapping (from the original field experiment):
  Raw GeoPackage column ``class`` → TIF pixel value
    0 → 0  (control: 100 % P dose — non-stressed)
    1 → 1  (treatment: 25 % P dose — stressed)
    2 → 2  (treatment: 50 % P dose — stressed)
    3 → 3  (treatment: 75 % P dose — stressed)
    no geometry / outside → 255 (nodata)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine

logger = logging.getLogger(__name__)

# Pixel values in the output TIF
NODATA_LABEL: int = 255


def rasterize_labels(
    gpkg_path: Path,
    output_tif: Path,
    reference_zarr: Path | None = None,
    reference_tif: Path | None = None,
    layer: str | None = None,
    class_col: str = "class",
    crs: str = "EPSG:32618",
    transform: Affine | None = None,
    shape: tuple[int, int] | None = None,
    all_touched: bool = True,
) -> None:
    """Rasterise GeoPackage parcel polygons to a multiclass label GeoTIFF.

    The spatial reference (CRS, transform, shape) is read from
    ``reference_tif`` or ``reference_zarr`` if either is provided.
    Otherwise, ``crs``, ``transform``, and ``shape`` must be supplied
    explicitly (useful when you know the raster grid in advance).

    Parameters
    ----------
    gpkg_path : Path
        GeoPackage file with labelled parcel polygons.
    output_tif : Path
        Destination GeoTIFF (uint8).  Parent directory is created if needed.
    reference_zarr : Path, optional
        An existing ``masked_reflectance.zarr`` whose spatial metadata
        (stored as root attributes ``crs``, ``transform``, ``shape``) is used
        as the reference grid.
    reference_tif : Path, optional
        Any georeferenced raster from which to read CRS / transform / shape.
    layer : str, optional
        GeoPackage layer name.  If None, the first layer is used.
    class_col : str, optional
        Column in the GeoPackage that holds the integer class labels.
    crs : str, optional
        Fallback CRS (EPSG string) when no reference file is provided.
    transform : Affine, optional
        Fallback affine transform when no reference file is provided.
    shape : tuple (height, width), optional
        Fallback raster shape when no reference file is provided.
    all_touched : bool, optional
        If True, all pixels touched by a polygon are rasterised (inclusive
        border).  Defaults to True to match the original notebook.
    """
    try:
        import geopandas as gpd
    except ImportError as err:
        raise ImportError("geopandas is required for label rasterisation.") from err

    import zarr

    gpkg_path = Path(gpkg_path)
    output_tif = Path(output_tif)
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    # Resolve spatial reference
    if reference_tif is not None:
        with rasterio.open(reference_tif) as src:
            crs = src.crs.to_string()
            transform = src.transform
            shape = (src.height, src.width)
        logger.info("Spatial reference from TIF: %s  shape=%s", reference_tif, shape)

    elif reference_zarr is not None:
        store = zarr.open(str(reference_zarr), mode="r")
        attrs = dict(store.attrs)
        crs = attrs.get("crs", crs)
        tf_list = attrs.get("transform", None)
        if tf_list is not None:
            transform = Affine(*tf_list[:6])
        ny = store["reflectance"].shape[-2] if "reflectance" in store else None
        nx = store["reflectance"].shape[-1] if "reflectance" in store else None
        if ny and nx:
            shape = (ny, nx)
        logger.info("Spatial reference from zarr: %s  shape=%s", reference_zarr, shape)

    if transform is None or shape is None:
        raise ValueError(
            "Spatial reference could not be determined. "
            "Provide reference_tif, reference_zarr, or (transform + shape)."
        )

    height, width = shape

    # Load and reproject labels
    logger.info("Reading GeoPackage: %s", gpkg_path)
    read_kwargs = {"filename": str(gpkg_path)}
    if layer is not None:
        read_kwargs["layer"] = layer
    gdf = gpd.read_file(**read_kwargs)

    target_crs = rasterio.crs.CRS.from_string(crs)
    if gdf.crs is None:
        logger.warning("GeoPackage has no CRS — assuming %s.", crs)
        gdf = gdf.set_crs(target_crs)
    elif gdf.crs != target_crs:
        logger.info("Reprojecting GeoPackage from %s to %s", gdf.crs, target_crs)
        gdf = gdf.to_crs(target_crs)

    if class_col not in gdf.columns:
        raise KeyError(
            f"Column '{class_col}' not found in GeoPackage. Available columns: {list(gdf.columns)}"
        )

    # Build (geometry, class_value) pairs
    shapes = [
        (row.geometry, int(row[class_col]))
        for _, row in gdf.iterrows()
        if row.geometry is not None and not np.isnan(row[class_col])
    ]

    logger.info("Rasterising %d polygons onto %d × %d grid...", len(shapes), height, width)
    label_array = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=NODATA_LABEL,
        all_touched=all_touched,
        dtype="uint8",
    )

    # Write GeoTIFF
    with rasterio.open(
        output_tif,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=target_crs,
        transform=transform,
        tiled=True,
        compress="DEFLATE",
        nodata=NODATA_LABEL,
    ) as dst:
        dst.write(label_array, 1)

    unique, counts = np.unique(label_array[label_array != NODATA_LABEL], return_counts=True)
    summary = "  ".join(f"class={u}:{c:,}" for u, c in zip(unique, counts, strict=False))
    logger.info("Label raster written: %s  |  %s", output_tif, summary)
