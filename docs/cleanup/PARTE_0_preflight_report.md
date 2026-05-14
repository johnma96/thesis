# Pre-flight Report — Repo Reorganization

**Date:** 2026-05-14  
**Branch:** `chore/repo-cleanup-uv`  
**DVC status:** Cache and remote 'origin' are in sync ✅

---

## Current repo structure (summary)

```
thesis/
├── spectralcrop/          # source package (partially populated)
├── notebooks/             # 18 notebooks (000–407)
├── data/                  # DVC-tracked raw/interim/processed/external
├── models/                # 20 model files (DVC-tracked)
├── reports/               # tesis.docx, figures/, pdfs/
├── docs/                  # thesis_corrections/, DVC guides
├── references/            # papers (DVC-tracked), technical reports
├── app/                   # only .gitkeep (empty)
├── conf/                  # config.yml
├── credentials/           # only .gitkeep
├── queries/               # develop/ + production/ (SQL template, unused)
├── prompts/               # prompt files for LLM assistants
├── tests/                 # only .gitkeep (no tests)
├── main.py                # leftover from template (financial model)
├── main.ipynb             # leftover from template
├── setup.py               # setuptools-based (deprecated path)
├── run.sh                 # references wrong project path
├── requirements.txt       # 30 pinned packages
├── requirements-pytorch-cpu.txt
├── requirements-pytorch-cu126.txt
└── environment.yml
```

---

## Differences vs. template (johnma96/machine_learning_project_template)

| Item | Template | Current repo | Action needed |
|---|---|---|---|
| Dependency manager | requirements.txt + environment.yml | Same | **Migrate to uv + pyproject.toml** |
| `spectralcrop/` subfolders | archive, data, features, models, performance, utils, visualization | Same + evaluation/ | Add: config/, models/ml, models/dl, evaluation/metrics.py, etc. |
| `queries/` | Present (SQL template) | Present but empty | **Archive / remove** (no SQL in this project) |
| `tests/` | `.gitkeep` only | Same | **Add smoke tests** |
| `app/` | `.gitkeep` only | Same | **Add FastAPI stubs** |
| `main.py` | Financial model orchestrator | Same (wrong content) | **Rewrite as CNN-2D pipeline CLI** |
| `setup.py` | setuptools-based | Same | **Replace with pyproject.toml** |
| `run.sh` | Template stub | Wrong paths | **Update or archive** |
| `.github/workflows/` | Not present | Not present | **Add CI stub** |
| `pyproject.toml` | Not present | Not present | **Create** |
| `uv.lock` | Not present | Not present | **Generate** |
| `main.ipynb` | Template stub | Present | **Archive or remove** |

---

## Root-level files that appear obsolete

| File | Reason | Proposed action |
|---|---|---|
| `main.py` | Contains a financial model (`run_prediction`) with wrong imports | Rewrite for thesis CNN-2D pipeline |
| `main.ipynb` | Template stub, no meaningful content | Archive to `archive/legacy_dependencies/` |
| `setup.py` | Superseded by pyproject.toml | Archive to `archive/legacy_dependencies/` |
| `run.sh` | References `/path_to_project/thesis/` + wrong Python | Rewrite with uv invocation |
| `environment.yml` | Superseded by pyproject.toml | Archive to `archive/legacy_dependencies/` |
| `requirements*.txt` | Superseded by pyproject.toml | Archive to `archive/legacy_dependencies/` |

---

## Key pinned versions to preserve in pyproject.toml

Taken from `.venv` (actual installed state, PC B):

| Package | Version | Notes |
|---|---|---|
| numpy | 2.3.5 | Core |
| pandas | 2.2.3 | Core |
| scipy | 1.16.2 | Core |
| torch | 2.9.1+cu126 | PC B (GPU); CPU variant for PC A |
| scikit-learn | 1.7.2 | Models |
| xgboost | 3.1.2 | Models |
| lightgbm | 4.6.0 | Models |
| mlflow | 3.8.1 | Tracking |
| dvc | 3.63.0 | Versioning |
| rasterio | 1.4.3 | Geospatial |
| xarray | 2025.12.0 | Data cubes |
| zarr | 3.1.5 | Storage |
| lazypredict | 0.2.12 | Exploration only (not in production pipeline) |

---

## spectralcrop/ current state

```
spectralcrop/
├── __init__.py
├── pipeline_envi_cpu.py       ← legacy script (ENVI format reader)
├── pipeline_envi_dask_zarr.py ← legacy script
├── pipeline_original.md       ← documentation
├── archive/
├── data/
│   ├── hypercube_processor.py ← has functions used in notebooks 101/102
│   └── make_dataset.py
├── evaluation/                ← created this corrections cycle
│   ├── confusion_matrices.py
│   └── feature_ablation.py
├── features/
│   └── generate_features.py   ← stub
├── models/
│   ├── predict_model.py       ← stub
│   └── train_model.py         ← stub
├── performance/
│   └── backtesting.py         ← stub (wrong name for this project)
├── utils/
│   └── path_manager.py        ← actively used by all notebooks
└── visualization/
    └── visualize.py           ← stub
```

**Gaps:** no `config/`, no `models/dl/architectures.py`, no `features/patches.py`, no `features/vegetation_indices.py`.
