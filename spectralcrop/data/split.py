"""Spatial train/val/test split: parcel polygons → split_id GeoTIFF.

Replicates the logic of notebooks/302-jmmz-spatial-split.ipynb.

Strategy: stratified split at the (class, plot) group level — never at the
pixel level — so that entire parcels end up in the same fold and there is no
spatial leakage between splits.

Output TIF values: 1 = train  |  2 = val  |  3 = test  |  0 = outside
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def make_spatial_split(
    gpkg_path: Path,
    output_tif: Path,
    reference_tif: Path | None = None,
    layer: str | None = None,
    class_col: str = "class",
    plot_col: str = "plot",
    crs: str = "EPSG:32618",
    transform: Affine | None = None,
    shape: tuple[int, int] | None = None,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    seed: int = 42,
    all_touched: bool = True,
) -> pd.DataFrame:
    """Generate a spatially stratified train/val/test split and rasterise it.

    Groups parcels by ``(class_col, plot_col)``, then splits the groups
    with stratification on ``class_col`` so that class balance is roughly
    preserved across splits.

    Parameters
    ----------
    gpkg_path : Path
        GeoPackage with labelled parcel polygons.
    output_tif : Path
        Destination GeoTIFF (uint8).
    reference_tif : Path, optional
        Georeferenced raster used to read CRS / transform / shape.
    layer : str, optional
        GeoPackage layer name; if None, first layer is used.
    class_col : str
        Column with integer class labels (0–3).
    plot_col : str
        Column identifying each physical parcel / plot.
    crs, transform, shape
        Fallback spatial reference when ``reference_tif`` is not given.
    train_frac : float
        Fraction of groups assigned to the training split.
    val_frac : float
        Fraction of groups assigned to the validation split.
        ``test_frac = 1 - train_frac - val_frac``.
    seed : int
        Random seed for reproducibility.
    all_touched : bool
        Passed to ``rasterio.features.rasterize``.

    Returns
    -------
    pd.DataFrame
        Group-level split assignment table with columns
        ``group_id, split, class, plot``.
    """
    try:
        import geopandas as gpd
    except ImportError as err:
        raise ImportError("geopandas is required for split generation.") from err

    gpkg_path = Path(gpkg_path)
    output_tif = Path(output_tif)
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    # Spatial reference
    if reference_tif is not None:
        with rasterio.open(reference_tif) as src:
            crs = src.crs.to_string()
            transform = src.transform
            shape = (src.height, src.width)

    if transform is None or shape is None:
        raise ValueError("Provide reference_tif or explicit (transform + shape).")

    height, width = shape

    # Load GeoPackage
    read_kw = {"filename": str(gpkg_path)}
    if layer is not None:
        read_kw["layer"] = layer
    gdf = gpd.read_file(**read_kw)

    target_crs = rasterio.crs.CRS.from_string(crs)
    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    elif gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    for col in (class_col, plot_col):
        if col not in gdf.columns:
            raise KeyError(f"Column '{col}' not found. Available: {list(gdf.columns)}")

    # Build group table
    gdf["_group_id"] = gdf[class_col].astype(str) + "_" + gdf[plot_col].astype(str)
    groups = (
        gdf[["_group_id", class_col, plot_col]].drop_duplicates("_group_id").reset_index(drop=True)
    )

    group_ids = groups["_group_id"].values
    classes = groups[class_col].values.astype(int)

    # Stratified split
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    gid_tr, gid_tmp, y_tr, y_tmp = train_test_split(
        group_ids,
        classes,
        test_size=(val_frac + test_frac),
        stratify=classes,
        random_state=seed,
    )
    val_ratio = val_frac / (val_frac + test_frac)
    gid_val, gid_te, _, _ = train_test_split(
        gid_tmp,
        y_tmp,
        test_size=(1.0 - val_ratio),
        stratify=y_tmp,
        random_state=seed,
    )

    assign = pd.DataFrame(
        {
            "_group_id": np.concatenate([gid_tr, gid_val, gid_te]),
            "split": (["train"] * len(gid_tr) + ["val"] * len(gid_val) + ["test"] * len(gid_te)),
        }
    )
    assign = assign.merge(groups, on="_group_id", how="left")
    assign = assign.rename(columns={"_group_id": "group_id", class_col: "class", plot_col: "plot"})

    split2id = {"train": 1, "val": 2, "test": 3}

    # Merge split ids back to geometries
    gdf2 = gdf.merge(
        assign[["group_id", "split"]], left_on="_group_id", right_on="group_id", how="left"
    )
    gdf2["_split_id"] = gdf2["split"].map(split2id).fillna(0).astype("uint8")

    shapes = [
        (row.geometry, int(row["_split_id"]))
        for _, row in gdf2.iterrows()
        if row.geometry is not None
    ]

    split_array = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )

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
        nodata=0,
    ) as dst:
        dst.write(split_array, 1)

    for split_name, sid in split2id.items():
        n = int((split_array == sid).sum())
        logger.info("  %s (id=%d): %d px", split_name, sid, n)

    logger.info("Split raster written: %s", output_tif)
    return assign[["group_id", "split", "class", "plot"]]
