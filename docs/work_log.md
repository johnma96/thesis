# Work Log — spectralcrop thesis

> Private per-machine log. Not synced via git. Each PC keeps its own copy.

---

### 2026-05-14 (continuación, sesión 2) — PC B

- **Developer context:** Segunda sesión del mismo día. Objetivo: mejoras técnicas al repositorio post-limpieza y arrancar el despliegue en GCP.

- **Session work:**

  - **Privacy cleanup:** Reescritura de historia con `git filter-repo` para eliminar 5 PDFs con datos personales (firmas y cédulas). Archivos eliminados: `Carta Respuesta - Observaciones jurado.pdf`, `Propuesta Trabajo Final.pdf`, `Observaciones TFM John Montoya.pdf`, `Trabajo Final John Montoya - Feb23-pre-jury.docx`, `archive/legacy_dependencies/...docx`. Creado `docs/thesis_corrections/jury_observations.md` con el texto de las 20 observaciones sin datos personales. Eliminadas carpetas `prompts/`, `credentials/`, `conf/` (scaffolding sin uso).

  - **Git tags:** Creados 5 tags anotados alineados con la tabla de versiones del README:
    `v0.1.0` (2025-04-14 — initial), `v1.0.0` (2026-01-22 — thesis submission),
    `v1.1.0` (2026-05-12 — jury corrections approved), `v1.2.0` (2026-05-14 — repo cleanup PR),
    `v1.3.0` (2026-05-14 — vectorisation + DVC pipeline).

  - **Technical improvements (PR #3):**
    - `spectralcrop/inference/predict.py`: vectorised patch extraction con `scipy.maximum_filter` + NumPy advanced indexing. Elimina bucle Python anidado → ~30 min reducido a ~2 min.
    - `main.py train-cnn2d`: refit de `RobustScaler` sobre los píxeles de entrenamiento del nuevo dataset. Guarda `robust_scaler.pkl` junto al modelo.
    - `main.py evaluate`: escribe `reports/metrics_retrain.json` compatible con `dvc metrics diff`.
    - `dvc.yaml` + `params.yaml`: pipeline declarativo de 5 etapas (preprocess → make_labels → make_split → train → evaluate).
    - 6 stubs vacíos eliminados de `spectralcrop/`.
    - 3 nuevos archivos de tests: `test_predict_vectorized.py` (7 tests), `test_preprocessing.py` (5 tests), `test_training_loop.py` (4 tests). Cobertura 18% → 40%.
    - `docs/gcp_deployment_proposal.md`: propuesta arquitectural completa (Cloud Run Jobs + Cloud Run Service, estimación de costos, plan de 4 sprints).

  - **GCP Sprint 1 (en develop, pendiente de PR):**
    - `Dockerfile.batch`: imagen CPU para Cloud Run Jobs (python:3.12-slim + uv + pytorch-cpu + libgdal).
    - `scripts/batch_predict.py`: script de inferencia batch (descarga ENVI de GCS → preprocesa → predice → sube TIFs + metadata.json).
    - `.github/workflows/docker-publish.yml`: CI que construye y pushea imagen a Artifact Registry al mergear a `main` (Workload Identity Federation, sin JSON keys).
    - `pyproject.toml`: nuevo extra `[gcp]` con `google-cloud-storage>=2.18`.
    - `docs/gcp_setup.md`: guía de configuración única en GCP (proyecto, APIs, AR, WIF, buckets).

- **Git state:**
  - Ramas: `develop` (activa), `main` (sincronizado hasta PR #3)
  - Tags en remoto: v0.1.0, v1.0.0, v1.1.0, v1.2.0, v1.3.0
  - Último commit en develop: `0835238 feat(gcp): Sprint 1 — Dockerfile.batch, CI publish workflow, batch script`
  - Tests: 33 passed, cobertura 40%
  - Lint: All checks passed (ruff)

- **Pending for next session:**
  - PR `develop` → `main` para GCP Sprint 1
  - Configuración única en GCP (guía en `docs/gcp_setup.md`)
  - GCP Sprint 2: Cloud Run Job + Eventarc trigger automático al subir archivo a GCS
  - GCP Sprint 3: Cloud Run Service (API REST, `app/`)
  - GCP Sprint 4: Terraform + dominio + seguridad

- **Open questions:**
  - ¿Qué nombre de proyecto GCP usar? (`spectralcrop-prod` sugerido en docs)
  - ¿Se necesita GPU (T4) en el batch job o CPU es suficiente para el volumen esperado?

---

### 2026-05-14 — PC B

- **Developer context:** Post-thesis session. Thesis approved May 12, 2026. All 20 jury corrections addressed. Goal for this session: clean up and reorganize the repository for long-term maintainability.

- **Session work:**
  - Completed full repository reorganization via PR #1 (`chore/repo-cleanup-uv` → `main`), CI green.
  - **Part 1 — uv migration:** Created `pyproject.toml` + `uv.lock` (340 packages). Fixed DVC incompatibility: `pathspec>=0.10.3,<0.12` (DVC 3.63 uses `_DIR_MARK` removed in pathspec 0.12+). Added `required-environments` for Linux x86_64 + Windows AMD64 so the lockfile works in CI.
  - **Part 2 — spectralcrop reorganization:** Added `config/` (paths.py, constants.py with locked CNN-2D hparams), `data/preprocessing.py` (ENVI→zarr+VI), `data/labeling.py` (GeoPackage rasterization), `data/split.py` (spatial split), `features/patches.py`, `features/vegetation_indices.py`, `models/dl/` (architectures, train, predict), `models/ml/predict.py`, `evaluation/metrics.py`, `inference/predict.py` (full-image pixel prediction). Rewrote `main.py` as 8-command typer CLI (preprocess, make-labels, make-split, train-cnn2d, evaluate, predict, predict-pipeline, full-pipeline). Replaced `run.sh` with `Makefile`.
  - **Part 3 — app/ stubs:** FastAPI scaffold for future CNN-2D serving (health + predict endpoints, Pydantic schemas, MLflow model loader).
  - **Part 4 — cleanup:** Archived `queries/`, `pipeline_envi_*.py`, `pipeline_original.md`, legacy dependency files. Updated `.gitignore`. Removed `.docx` from git (PDFs only). Removed Google Drive DVC remote.
  - **Part 5 — README + install.md:** Full rewrite with uv-based install, pipeline usage, reproducibility statement.
  - **Part 6 — CI + tests:** 16 smoke tests (patches, CNN-2D inference, metrics, imports), branch coverage in `pyproject.toml`, GitHub Actions workflow (ruff + pytest on ubuntu-latest).
  - **Branch cleanup:** Deleted 9 stale branches (all `feature/*`, `dev`, `docs/thesis-results`, `corrections-jury-2026`). Created `develop` from merged `main`.
  - **Bug fix:** CI torchvision wheel failure on Linux — removed pytorch-cpu from CI sync (torch tests skip via `importorskip`), added `required-environments` for multi-platform lock.

- **Git state:**
  - Branch: `develop` (current working branch going forward)
  - `main`: stable, CI green, PR #1 merged
  - Last commit on main: `e658253 Merge pull request #1`
  - DVC: in sync with DagsHub S3 (`dvc status -c` → "Cache and remote 'origin' are in sync")

- **Pending for next session:**
  - Run `make evaluate` on a clean clone to verify end-to-end reproducibility
  - Consider setting up GitHub branch protection rules for `main` (require PR + CI pass)
  - `app/` MLflow Registry loader is a stub — implement remote model pull when needed
  - Dependabot vulnerabilities on `main` should auto-close after the PR merge resolves the legacy requirements

- **Open questions:**
  - None blocking.
