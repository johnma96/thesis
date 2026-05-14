"""
spectralcrop — CNN-2D pipeline orchestrator.

Complete, reproducible pipeline for phosphorus-deficiency detection in common
beans using UAV hyperspectral imagery and the CNN-2D architecture.

----------------------------------------------------------------------------
RETRAINING WORKFLOW (new labeled data)
----------------------------------------------------------------------------
    uv run python main.py preprocess   --envi path/to/image.hdr
    uv run python main.py make-labels  --gpkg path/to/labels.gpkg
    uv run python main.py make-split   --gpkg path/to/labels.gpkg
    uv run python main.py train-cnn2d  --use-locked-hparams
    uv run python main.py evaluate     --model cnn2d --split test

Or in one shot:
    uv run python main.py full-pipeline --envi path/to/image.hdr \\
                                        --gpkg path/to/labels.gpkg

----------------------------------------------------------------------------
PRODUCTION WORKFLOW (new unlabelled image -> prediction map)
----------------------------------------------------------------------------
    uv run python main.py predict-pipeline --envi path/to/new_image.hdr

Or step by step:
    uv run python main.py preprocess --envi path/to/new_image.hdr
    uv run python main.py predict

----------------------------------------------------------------------------
All commands require data to be available locally. Run `dvc pull` first
if you are starting from a clean clone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="spectralcrop",
    help=(
        "Non-invasive P-deficiency detection in common beans — "
        "full CNN-2D pipeline (preprocess -> train -> evaluate / predict)."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("spectralcrop.main")

# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------


def _default_paths():
    from spectralcrop.config.paths import (
        BANDS_SELECTED,
        INTERIM_DIR,
        LABELS_TIF,
        MODELS_DIR,
        PROCESSED_DIR,
        RAW_DIR,
        SPLIT_BINARY_TIF,
        ZARR_CUBE,
    )

    return (
        ZARR_CUBE,
        LABELS_TIF,
        SPLIT_BINARY_TIF,
        BANDS_SELECTED,
        MODELS_DIR,
        PROCESSED_DIR,
        INTERIM_DIR,
        RAW_DIR,
    )


def _check_zarr_ready(zarr_path: Path) -> None:
    if not zarr_path.exists():
        typer.secho(
            f"Zarr not found: {zarr_path}\n"
            "Run `uv run python main.py preprocess --envi <path_to_hdr>` first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)


def _check_labels_ready(labels_tif: Path, split_tif: Path) -> None:
    missing = [p for p in [labels_tif, split_tif] if not p.exists()]
    if missing:
        typer.secho(
            "Missing label/split files:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\nRun make-labels and make-split first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# 1. PREPROCESS
# ---------------------------------------------------------------------------


@app.command("preprocess")
def preprocess(
    envi: Annotated[
        Path,
        typer.Option("--envi", help="Path to the ENVI .hdr header file."),
    ],
    output_zarr: Annotated[
        Path | None,
        typer.Option(
            "--output-zarr",
            help="Destination zarr path. Defaults to data/interim/masked_reflectance.zarr.",
        ),
    ] = None,
    ndvi_threshold: Annotated[
        float,
        typer.Option("--ndvi-threshold", help="NDVI threshold for vegetation mask."),
    ] = 0.3,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Resume an interrupted zarr write."),
    ] = False,
) -> None:
    """Process a raw ENVI hypercube -> masked Zarr with reflectance + all 5 VI.

    Steps performed:
      1. Open the ENVI file with the spectral library.
      2. Convert raw DN -> physical reflectance [0–1.2], mask nodata.
      3. Apply vegetation mask (NDVI > threshold).
      4. Exclude water-absorption bands (~1340–1440 nm, ~1800–1950 nm).
      5. Export reflectance cube + NDVI + veg_mask to zarr.
      6. Compute and append NDRE, CIgreen, PRI, PSRI to the zarr.

    The output zarr is the starting point for both the training and
    production prediction pipelines.
    """
    from spectralcrop.config.paths import ZARR_CUBE
    from spectralcrop.data.preprocessing import preprocess_envi_to_zarr

    dest = output_zarr if output_zarr is not None else ZARR_CUBE
    typer.echo(f"Preprocessing {envi} -> {dest}")
    preprocess_envi_to_zarr(
        hdr_path=envi,
        output_zarr=dest,
        ndvi_threshold=ndvi_threshold,
        resume=resume,
    )
    typer.secho("✅  Preprocessing complete.", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# 2. MAKE-LABELS
# ---------------------------------------------------------------------------


@app.command("make-labels")
def make_labels(
    gpkg: Annotated[
        Path | None,
        typer.Option(
            "--gpkg",
            help="GeoPackage with labelled parcel polygons. Defaults to data/raw/labels_export.gpkg.",
        ),
    ] = None,
    output_tif: Annotated[
        Path | None,
        typer.Option(
            "--output-tif", help="Output label TIF. Defaults to data/interim/labels_multiclass.tif."
        ),
    ] = None,
    layer: Annotated[
        str | None,
        typer.Option("--layer", help="GeoPackage layer name (if there are multiple layers)."),
    ] = None,
    class_col: Annotated[
        str,
        typer.Option("--class-col", help="Column with integer class labels (0–3)."),
    ] = "class",
    zarr_reference: Annotated[
        Path | None,
        typer.Option(
            "--zarr-ref", help="Zarr from which to read the spatial grid (CRS/transform/shape)."
        ),
    ] = None,
) -> None:
    """Rasterise GeoPackage parcel polygons -> labels_multiclass.tif.

    Label mapping:  0 = control (100% P)  |  1,2,3 = stressed (25/50/75% P)
    Nodata value = 255.

    The spatial grid (CRS, pixel size, extent) is taken from the zarr
    produced by the preprocess step. If no zarr is provided, it defaults
    to data/interim/masked_reflectance.zarr.
    """
    from spectralcrop.config.paths import LABELS_TIF, RAW_DIR, ZARR_CUBE
    from spectralcrop.data.labeling import rasterize_labels

    gpkg_path = gpkg if gpkg is not None else RAW_DIR / "labels_export.gpkg"
    out_tif = output_tif if output_tif is not None else LABELS_TIF
    zarr_ref = zarr_reference if zarr_reference is not None else ZARR_CUBE

    if not gpkg_path.exists():
        typer.secho(f"GeoPackage not found: {gpkg_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Rasterising {gpkg_path} -> {out_tif}")
    rasterize_labels(
        gpkg_path=gpkg_path,
        output_tif=out_tif,
        reference_zarr=zarr_ref,
        layer=layer,
        class_col=class_col,
    )
    typer.secho("✅  Labels written.", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# 3. MAKE-SPLIT
# ---------------------------------------------------------------------------


@app.command("make-split")
def make_split(
    gpkg: Annotated[
        Path | None,
        typer.Option(
            "--gpkg",
            help="GeoPackage with parcel polygons. Defaults to data/raw/labels_export.gpkg.",
        ),
    ] = None,
    output_tif: Annotated[
        Path | None,
        typer.Option(
            "--output-tif",
            help="Output split TIF. Defaults to data/processed/splits/by_plot_split_id_binary.tif.",
        ),
    ] = None,
    reference_tif: Annotated[
        Path | None,
        typer.Option(
            "--ref-tif",
            help="Reference raster for CRS/transform/shape. Defaults to labels_multiclass.tif.",
        ),
    ] = None,
    class_col: Annotated[str, typer.Option("--class-col")] = "class",
    plot_col: Annotated[str, typer.Option("--plot-col")] = "plot",
    train_frac: Annotated[float, typer.Option("--train")] = 0.60,
    val_frac: Annotated[float, typer.Option("--val")] = 0.20,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Generate a stratified spatial train/val/test split -> split_id GeoTIFF.

    Parcels are grouped by (class, plot) and split at the group level so
    that whole parcels land in a single fold (no pixel-level spatial leakage).

    Output values: 1=train  2=val  3=test  0=outside.
    """
    from spectralcrop.config.paths import LABELS_TIF, RAW_DIR, SPLIT_BINARY_TIF
    from spectralcrop.data.split import make_spatial_split

    gpkg_path = gpkg if gpkg is not None else RAW_DIR / "labels_export.gpkg"
    out_tif = output_tif if output_tif is not None else SPLIT_BINARY_TIF
    ref_tif = reference_tif if reference_tif is not None else LABELS_TIF

    if not gpkg_path.exists():
        typer.secho(f"GeoPackage not found: {gpkg_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Creating split raster -> {out_tif}")
    assign = make_spatial_split(
        gpkg_path=gpkg_path,
        output_tif=out_tif,
        reference_tif=ref_tif if ref_tif.exists() else None,
        class_col=class_col,
        plot_col=plot_col,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )
    typer.echo(assign.groupby("split").size().to_string())
    typer.secho("✅  Split raster written.", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# 4. TRAIN-CNN2D
# ---------------------------------------------------------------------------


@app.command("train-cnn2d")
def train_cnn2d(
    use_locked_hparams: Annotated[
        bool,
        typer.Option(
            "--use-locked-hparams/--custom-hparams",
            help="Use the hyperparameters locked in constants.py (recommended).",
        ),
    ] = True,
    mlflow_run_name: Annotated[str, typer.Option("--run-name")] = "cnn2d_retrain",
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("models"),
) -> None:
    """Train (or retrain) the CNN-2D model on the processed data.

    Prerequisites: preprocess, make-labels, make-split must have been run
    (or the DVC-tracked artefacts must be present via `dvc pull`).

    When --use-locked-hparams (default), uses the hyperparameters from the
    thesis MLflow run (run_id 61a3cc05f39d46f79f2e3fa3d29fae7f), guaranteeing
    a comparable model.  Expected result: val PR-AUC ≈ 0.96.
    """
    import json

    import joblib
    import mlflow
    import numpy as np
    import pandas as pd
    import rasterio
    import torch
    import xarray as xr

    from spectralcrop.config.constants import CNN2D_HPARAMS, MLFLOW_TRACKING_URI, RANDOM_SEED
    from spectralcrop.config.paths import (
        BANDS_SELECTED,
        LABELS_TIF,
        SPLIT_BINARY_TIF,
        ZARR_CUBE,
    )
    from spectralcrop.features.patches import build_patch_dataset
    from spectralcrop.models.dl.architectures import SpectralSpatialCNN2D
    from spectralcrop.models.dl.train import fit_cnn2d

    _check_zarr_ready(ZARR_CUBE)
    _check_labels_ready(LABELS_TIF, SPLIT_BINARY_TIF)

    hparams = CNN2D_HPARAMS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading data...")
    ds = xr.open_zarr(str(ZARR_CUBE), chunks={"band": 64, "y": 512, "x": 512})
    bands_df = pd.read_csv(BANDS_SELECTED)
    selected_bands = bands_df["band_index"].tolist()

    indices_da = ds[["NDVI", "NDRE", "CIgreen", "PRI", "PSRI"]].to_array(dim="band")
    refl_sel = ds["reflectance"].isel(band=selected_bands)
    features = xr.concat([indices_da, refl_sel], dim="band").transpose("y", "x", "band")

    with rasterio.open(str(LABELS_TIF)) as src:
        labels_raw = src.read(1)
    labels_bin = np.where(labels_raw == 0, 0, np.where(np.isin(labels_raw, [1, 2, 3]), 1, np.nan))
    with rasterio.open(str(SPLIT_BINARY_TIF)) as src:
        split_map = src.read(1)

    from sklearn.preprocessing import RobustScaler

    X_cube = features.values
    H, W, B = X_cube.shape

    # Extract flat training pixels to fit a fresh scaler on this dataset.
    # Re-using the original robust_scaler.pkl would be incorrect if the
    # input image has different radiometric characteristics.
    X_flat = X_cube.reshape(-1, B)
    labels_flat = labels_bin.reshape(-1)
    split_flat = split_map.reshape(-1).astype(float)
    train_mask = (split_flat == 1) & ~np.isnan(labels_flat) & ~np.isnan(X_flat).any(axis=1)
    X_train_flat = X_flat[train_mask]

    logger.info("Fitting RobustScaler on %d training pixels...", len(X_train_flat))
    scaler = RobustScaler().fit(X_train_flat)
    scaler_path = output_dir / "robust_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved: %s", scaler_path)

    cube_scaled = scaler.transform(X_flat).reshape(H, W, B)
    logger.info("Cube: %d × %d × %d  |  building patches...", H, W, B)

    X_train2d, y_train2d = build_patch_dataset(
        cube_scaled, labels_bin, split_map, 1, hparams["patch_size"]
    )
    X_val2d, y_val2d = build_patch_dataset(
        cube_scaled, labels_bin, split_map, 2, hparams["patch_size"]
    )
    logger.info("train=%d  val=%d", len(X_train2d), len(X_val2d))

    torch.manual_seed(RANDOM_SEED)
    model = SpectralSpatialCNN2D(
        n_channels=hparams["n_channels"],
        n_classes=2,
        kernel_size=hparams["kernel_size"],
        dropout=hparams["dropout"],
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name=mlflow_run_name):
        mlflow.log_params(hparams)

        def _on_epoch(ep: int, tr: float, vl: float) -> None:
            logger.info("epoch %3d | train_loss=%.4f  val_prauc=%.4f", ep, tr, vl)
            mlflow.log_metrics({"train_loss": tr, "val_prauc": vl}, step=ep)

        model, _, val_praucs = fit_cnn2d(
            model=model,
            X_train=X_train2d,
            y_train=y_train2d,
            X_val=X_val2d,
            y_val=y_val2d,
            lr=hparams["lr"],
            batch_size=hparams["batch_size"],
            max_epochs=hparams["max_epochs"],
            patience=hparams["patience"],
            weight_decay=hparams["weight_decay"],
            device=device,
            seed=RANDOM_SEED,
            on_epoch_end=_on_epoch,
        )
        best_vl = max(val_praucs)
        mlflow.log_metric("best_val_prauc", best_vl)

    weights_path = output_dir / "cnn2d_retrain_weights.pt"
    info_path = output_dir / "cnn2d_retrain_info.json"
    torch.save(model.state_dict(), weights_path)
    with open(info_path, "w") as f:
        json.dump(
            {**hparams, "best_val_prauc": best_vl, "scaler": str(scaler_path)},
            f,
            indent=2,
        )

    typer.secho(
        f"✅  Model saved -> {weights_path}  (val PR-AUC={best_vl:.4f})\n"
        f"   Scaler saved -> {scaler_path}",
        fg=typer.colors.GREEN,
    )


# ---------------------------------------------------------------------------
# 5. EVALUATE
# ---------------------------------------------------------------------------


@app.command("evaluate")
def evaluate(
    split: Annotated[
        str, typer.Option("--split", help="Dataset split: train | val | test.")
    ] = "test",
    weights_path: Annotated[
        Path | None,
        typer.Option(
            "--weights",
            help="Path to .pt weights. Defaults to models/cnn2d_final_model_weights.pt.",
        ),
    ] = None,
    output_metrics: Annotated[
        Path,
        typer.Option(
            "--output-metrics",
            help="JSON file where metrics are written for DVC tracking.",
        ),
    ] = Path("reports/metrics_retrain.json"),
) -> None:
    """Evaluate the CNN-2D model on a dataset split, print metrics, and write JSON.

    The JSON output is compatible with `dvc metrics diff` for comparing runs.
    Uses the locked decision threshold (0.3218) from the thesis.
    Expected test PR-AUC with the final model: 0.9635.
    """
    import joblib
    import numpy as np
    import pandas as pd
    import rasterio
    import torch
    import xarray as xr

    from spectralcrop.config.paths import (
        BANDS_SELECTED,
        LABELS_TIF,
        MODELS_DIR,
        SPLIT_BINARY_TIF,
        ZARR_CUBE,
    )
    from spectralcrop.evaluation.metrics import compute_all_metrics
    from spectralcrop.features.patches import build_patch_dataset
    from spectralcrop.models.dl.predict import load_cnn2d, predict_proba_2d

    _check_zarr_ready(ZARR_CUBE)
    _check_labels_ready(LABELS_TIF, SPLIT_BINARY_TIF)

    split_map_val = {"train": 1, "val": 2, "test": 3}
    if split not in split_map_val:
        typer.secho(f"Unknown split '{split}'. Use: train, val, test", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading data...")
    ds = xr.open_zarr(str(ZARR_CUBE), chunks={"band": 64, "y": 512, "x": 512})
    bands_df = pd.read_csv(BANDS_SELECTED)
    selected_bands = bands_df["band_index"].tolist()

    indices_da = ds[["NDVI", "NDRE", "CIgreen", "PRI", "PSRI"]].to_array(dim="band")
    refl_sel = ds["reflectance"].isel(band=selected_bands)
    features = xr.concat([indices_da, refl_sel], dim="band").transpose("y", "x", "band")

    with rasterio.open(str(LABELS_TIF)) as src:
        labels_raw = src.read(1)
    labels_bin = np.where(labels_raw == 0, 0, np.where(np.isin(labels_raw, [1, 2, 3]), 1, np.nan))
    with rasterio.open(str(SPLIT_BINARY_TIF)) as src:
        split_arr = src.read(1)

    X_cube = features.values
    H, W, B = X_cube.shape
    scaler = joblib.load(MODELS_DIR / "robust_scaler.pkl")
    cube_scaled = scaler.transform(X_cube.reshape(-1, B)).reshape(H, W, B)

    logger.info("Building patches for split='%s'...", split)
    X_eval, y_eval = build_patch_dataset(cube_scaled, labels_bin, split_arr, split_map_val[split])
    logger.info("Evaluating %d patches...", len(X_eval))

    md = MODELS_DIR if weights_path is None else weights_path.parent
    cnn2d_model, thr = load_cnn2d(md, device)
    if weights_path is not None:
        state = torch.load(weights_path, map_location=device, weights_only=False)
        cnn2d_model.load_state_dict(state)

    y_prob = predict_proba_2d(cnn2d_model, X_eval, device)
    metrics = compute_all_metrics(y_eval, y_prob, thr)

    typer.echo(f"\n=== CNN-2D evaluation — split={split} ===")
    for k, v in metrics.items():
        typer.echo(f"  {k:<14} {v:.4f}")
    typer.echo()

    # Write metrics JSON for DVC tracking (dvc metrics diff, dvc repro)
    import json as _json

    output_metrics = Path(output_metrics)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    payload = {split: {k: round(v, 4) for k, v in metrics.items()}}
    output_metrics.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    typer.secho(f"Metrics written: {output_metrics}", fg=typer.colors.BRIGHT_BLACK)


# ---------------------------------------------------------------------------
# 6. PREDICT  (production: zarr -> prediction TIFs)
# ---------------------------------------------------------------------------


@app.command("predict")
def predict(
    zarr_path: Annotated[
        Path | None,
        typer.Option(
            "--zarr", help="Preprocessed zarr. Defaults to data/interim/masked_reflectance.zarr."
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Where to write prediction TIFs. Defaults to reports/figures/predictions/.",
        ),
    ] = None,
    weights_path: Annotated[
        Path | None,
        typer.Option(
            "--weights",
            help="Custom model weights (.pt). Defaults to models/cnn2d_final_model_weights.pt.",
        ),
    ] = None,
    threshold: Annotated[
        float | None,
        typer.Option("--threshold", help="Decision threshold. Defaults to locked value (0.3218)."),
    ] = None,
) -> None:
    """Run production inference over a full preprocessed zarr -> GeoTIFFs.

    Outputs:
      prediction_proba.tif  — P(stressed) float32 per pixel
      prediction_class.tif  — binary classification uint8 (0/1/255=nodata)

    Border pixels (within 2px of any image edge or NaN region) are set to
    nodata, matching the exclusion applied during training.
    """
    import torch

    from spectralcrop.config.paths import FIGURES_DIR, MODELS_DIR, ZARR_CUBE
    from spectralcrop.inference.predict import predict_image
    from spectralcrop.models.dl.predict import load_cnn2d

    zp = zarr_path if zarr_path is not None else ZARR_CUBE
    odir = output_dir if output_dir is not None else FIGURES_DIR / "predictions"

    _check_zarr_ready(zp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    md = MODELS_DIR if weights_path is None else weights_path.parent
    model, locked_thr = load_cnn2d(md, device)
    if weights_path is not None:
        import torch as _torch

        model.load_state_dict(_torch.load(weights_path, map_location=device, weights_only=False))
    thr = threshold if threshold is not None else locked_thr

    typer.echo(f"Predicting {zp} -> {odir}  (threshold={thr:.4f})")
    proba_path, class_path = predict_image(
        zarr_path=zp,
        output_dir=odir,
        model=model,
        threshold=thr,
        device=device,
    )
    typer.secho("✅  Prediction maps written:", fg=typer.colors.GREEN)
    typer.echo(f"   Probability  : {proba_path}")
    typer.echo(f"   Classification: {class_path}")


# ---------------------------------------------------------------------------
# 7. PREDICT-PIPELINE  (production end-to-end: ENVI -> prediction TIFs)
# ---------------------------------------------------------------------------


@app.command("predict-pipeline")
def predict_pipeline(
    envi: Annotated[
        Path,
        typer.Option("--envi", help="Path to the ENVI .hdr file of the new image."),
    ],
    output_zarr: Annotated[
        Path | None,
        typer.Option(
            "--output-zarr",
            help="Zarr destination. Defaults to data/interim/masked_reflectance.zarr.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Where to write prediction TIFs."),
    ] = None,
    ndvi_threshold: Annotated[float, typer.Option("--ndvi-threshold")] = 0.3,
) -> None:
    """End-to-end production pipeline: ENVI image -> stress prediction maps.

    Step 1: preprocess  (ENVI -> zarr + all 5 VI)
    Step 2: predict     (zarr -> prediction_proba.tif + prediction_class.tif)

    Use this command when you have a new field image and want predictions
    without retraining the model.
    """
    typer.echo("-- Step 1/2: Preprocessing ---------------------------------")
    preprocess(envi=envi, output_zarr=output_zarr, ndvi_threshold=ndvi_threshold, resume=False)

    typer.echo("-- Step 2/2: Predicting ------------------------------------")
    predict(zarr_path=output_zarr, output_dir=output_dir)

    typer.secho("✅  predict-pipeline complete.", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# 8. FULL-PIPELINE  (retraining end-to-end)
# ---------------------------------------------------------------------------


@app.command("full-pipeline")
def full_pipeline(
    envi: Annotated[
        Path,
        typer.Option("--envi", help="Path to the ENVI .hdr file."),
    ],
    gpkg: Annotated[
        Path | None,
        typer.Option(
            "--gpkg",
            help="GeoPackage with labelled parcels. Defaults to data/raw/labels_export.gpkg.",
        ),
    ] = None,
    output_zarr: Annotated[
        Path | None,
        typer.Option("--output-zarr"),
    ] = None,
    ndvi_threshold: Annotated[float, typer.Option("--ndvi-threshold")] = 0.3,
) -> None:
    """End-to-end retraining pipeline: raw ENVI + labels -> trained CNN-2D.

    Step 1: preprocess    (ENVI -> zarr + VI)
    Step 2: make-labels   (GeoPackage -> labels_multiclass.tif)
    Step 3: make-split    (labels -> spatial train/val/test split)
    Step 4: train-cnn2d   (patches -> trained model + MLflow log)
    Step 5: evaluate      (test set metrics)

    Use this when you have a new labelled dataset and want to retrain
    the CNN-2D from scratch.
    """
    typer.echo("-- Step 1/5: Preprocessing ---------------------------------")
    preprocess(envi=envi, output_zarr=output_zarr, ndvi_threshold=ndvi_threshold, resume=False)

    typer.echo("-- Step 2/5: Rasterising labels ----------------------------")
    make_labels(gpkg=gpkg, zarr_reference=output_zarr)

    typer.echo("-- Step 3/5: Creating spatial split ------------------------")
    make_split(gpkg=gpkg)

    typer.echo("-- Step 4/5: Training CNN-2D -------------------------------")
    train_cnn2d(use_locked_hparams=True)

    typer.echo("-- Step 5/5: Evaluating on test set ------------------------")
    evaluate(split="test")

    typer.secho("✅  full-pipeline complete.", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
