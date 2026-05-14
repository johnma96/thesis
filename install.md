# Installation guide

## Prerequisites

- **Python 3.12** (exactly — `requires-python = ">=3.12,<3.13"`)
- **[uv](https://docs.astral.sh/uv/)** ≥ 0.4 — recommended package manager:
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **git** + **DVC remote credentials** (DagsHub) — see [docs/README_DVC_GoogleDrive.md](docs/README_DVC_GoogleDrive.md)

---

## Quick start with uv

### 1. Clone the repository

```bash
git clone https://github.com/johnma96/thesis.git
cd thesis
```

### 2. Install dependencies

**CPU-only** (CI, machines without GPU):
```bash
uv sync --extra pytorch-cpu --extra notebooks
```

**GPU — CUDA 12.6** (PC B: NVIDIA GeForce RTX 3050):
```bash
uv sync --extra pytorch-cu126 --extra notebooks
```

**Development environment** (adds linting, testing, type-checking):
```bash
uv sync --extra pytorch-cu126 --extra notebooks --extra dev
```

**API serving** (FastAPI stubs):
```bash
uv sync --extra pytorch-cu126 --extra api
```

### 3. Pull data and models from DVC

```bash
uv run dvc pull
```

> For the full pipeline: `uv run dvc pull data/raw.dvc data/processed.dvc models.dvc`

### 4. Verify installation

```bash
uv run python -c "import spectralcrop; import torch; print('OK', torch.__version__)"
uv run python main.py --help
```

---

## Alternative: pip install (no uv required)

Pip-compatible frozen exports are generated from the lockfile:

```bash
# CPU:
pip install -r requirements-export-cpu.txt
# GPU (CUDA 12.6):
pip install -r requirements-export-cu126.txt
```

> These files include all transitive dependencies pinned to exact versions
> for full reproducibility without uv.

---

## Reproducing thesis results

### Retrain final CNN-2D model (locked hyperparameters from MLflow)

```bash
uv run python main.py train-cnn2d --use-locked-hparams
```

### Evaluate on test set

```bash
uv run python main.py evaluate --model cnn2d --split test
```

### Run full pipeline

```bash
uv run python main.py full-pipeline
```

---

## DagsHub credentials (DVC remote)

Credentials are stored in `.dvc/config.local` (gitignored — never committed).

```bash
dvc remote modify origin --local access_key_id <dagshub_token>
dvc remote modify origin --local secret_access_key <dagshub_token>
```

---

## Environment summary

| Machine | GPU | torch extra |
|---|---|---|
| PC A | None (CPU) | `pytorch-cpu` |
| PC B | NVIDIA RTX 3050 | `pytorch-cu126` |
