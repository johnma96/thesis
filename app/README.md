# spectralcrop Inference API

FastAPI application serving the final CNN-2D model (`bean_stress_classifier` v1, Production).

## Purpose

Exposes a REST endpoint to classify individual 5×5×63 spectral patches as
**stressed** (P-deficient) or **non-stressed** (optimal P dose).

## Quick start (local)

```bash
# Install API dependencies
uv sync --extra pytorch-cu126 --extra api

# Launch development server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/predict` | Classify a single spectral patch |

## Request format

`POST /predict` expects a JSON body with a single `patch` field:

```json
{
  "patch": [
    [[v00, v01, v02, v03, v04],
     [v10, v11, v12, v13, v14],
     ...],
    ...
  ]
}
```

Shape: `[63 channels][5 rows][5 cols]`.  
Channel order: `[NDVI, NDRE, CIgreen, PRI, PSRI, band_0 … band_57]`.

> ⚠️ **The patch must be pre-scaled** with `models/robust_scaler.pkl` before
> sending.  Raw reflectance values will produce incorrect predictions.

## Response format

```json
{
  "label": 1,
  "label_name": "Stressed",
  "probability_stressed": 0.8734,
  "threshold_used": 0.3218,
  "model_version": "bean_stress_classifier/1"
}
```

## Docker deployment (reference)

See `app/Dockerfile.example` for a reference container image.

## Implementation status

| Component | Status |
|---|---|
| FastAPI app factory | ✅ |
| `/health` endpoint | ✅ |
| `/predict` endpoint | ✅ |
| Pydantic schemas (request + response) | ✅ |
| Model loader (from local weights) | ✅ |
| MLflow Registry loader (remote pull) | ⏳ Future work |
| Authentication / API key | ⏳ Future work |
| Batch endpoint `/predict-batch` | ⏳ Future work |
