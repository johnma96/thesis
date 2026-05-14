"""Project constants: spectral layout, label mapping, CNN-2D locked hyperparameters.

These values are derived from the thesis experimental setup and the final
MLflow run (run_id = 61a3cc05f39d46f79f2e3fa3d29fae7f).  They must NOT
be changed without explicit authorisation — changing them invalidates the
reproducibility of the reported results.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Spectral feature layout (63 channels)
# ---------------------------------------------------------------------------
N_FEATURES: int = 63  # total features fed to every model
N_BANDS: int = 58  # selected spectral bands
N_VI: int = 5  # vegetation indices

# Channel positions
VI_NAMES: list[str] = ["NDVI", "NDRE", "CIgreen", "PRI", "PSRI"]
VI_INDICES: list[int] = list(range(N_VI))  # 0 – 4
SPECTRAL_INDICES: list[int] = list(range(N_VI, N_FEATURES))  # 5 – 62

# Split identifiers
SPLIT_TRAIN: int = 1
SPLIT_VAL: int = 2
SPLIT_TEST: int = 3

# Label mapping (binary classification)
LABEL_STRESSED: int = 1  # 25 / 50 / 75 % P dose
LABEL_NON_STRESSED: int = 0  # 100 % P dose (optimal)
CLASS_NAMES: list[str] = ["Non-stressed (0)", "Stressed (1)"]

# ---------------------------------------------------------------------------
# CNN-2D — final locked hyperparameters
# MLflow run_id: 61a3cc05f39d46f79f2e3fa3d29fae7f
# Registered model: bean_stress_classifier v1 (Production)
# ---------------------------------------------------------------------------
CNN2D_HPARAMS: dict = {
    "patch_size": 5,
    "n_channels": 63,
    "kernel_size": 5,
    "dropout": 0.3727780074568463,
    "lr": 1.8559980846490597e-4,
    "batch_size": 64,
    "max_epochs": 80,
    "patience": 15,
    "weight_decay": 3.752055855124284e-5,
    "seed": 42,
}

CNN2D_BEST_THR: float = 0.32181818181818184
CNN2D_TEST_PRAUC: float = 0.9635  # locked — do not modify

MLFLOW_CNN2D_RUN_ID: str = "61a3cc05f39d46f79f2e3fa3d29fae7f"
MLFLOW_REGISTERED_MODEL: str = "bean_stress_classifier"
MLFLOW_MODEL_VERSION: str = "1"
MLFLOW_MODEL_STAGE: str = "Production"

# ---------------------------------------------------------------------------
# Random seed — must remain fixed for all reproducible runs
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
