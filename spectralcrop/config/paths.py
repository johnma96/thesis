"""Centralised path resolution for the spectralcrop project.

All paths are derived from the project root (detected via pyprojroot).
No hardcoded absolute paths anywhere in the codebase — import from here.
"""

from __future__ import annotations

from pathlib import Path

from pyprojroot import here

# Project root (the directory containing pyproject.toml / .git)
ROOT: Path = here()

# Data layers
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"

# Key data files
ZARR_CUBE: Path = INTERIM_DIR / "masked_reflectance.zarr"
LABELS_TIF: Path = INTERIM_DIR / "labels_multiclass.tif"
BANDS_SELECTED: Path = INTERIM_DIR / "bands_selected_by_segment.csv"
SPLIT_BINARY_TIF: Path = PROCESSED_DIR / "splits" / "by_plot_split_id_binary.tif"

# Models
MODELS_DIR: Path = ROOT / "models"

# Reports / figures
REPORTS_DIR: Path = ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# MLflow (remote on DagsHub — no local mlruns/)
MLFLOW_TRACKING_URI: str = "https://dagshub.com/johnma96/thesis.mlflow"

# Notebooks
NOTEBOOKS_DIR: Path = ROOT / "notebooks"


def ensure_dirs(*paths: Path) -> None:
    """Create directories (including parents) if they do not exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
