"""GCP batch prediction script.

Entrypoint for the Cloud Run Job that processes a full hyperspectral
image end-to-end:

  1. Download ENVI files from GCS input bucket.
  2. Preprocess: ENVI -> masked_reflectance.zarr + vegetation indices.
  3. Run CNN-2D inference: zarr -> prediction_proba.tif + prediction_class.tif.
  4. Upload results to GCS output bucket.
  5. Write a metadata JSON with run details.

Environment variables (injected by Cloud Run / Secret Manager):
  GCS_INPUT_BUCKET   — bucket where ENVI files are uploaded
  GCS_OUTPUT_BUCKET  — bucket where prediction TIFs are written
  GCS_ENVI_BLOB      — path within the input bucket, e.g. raw/2026-05-14/image.hdr
  MODEL_WEIGHTS_BLOB — path to .pt weights in input bucket (optional)
                       defaults to models/cnn2d_final_model_weights.pt
  MODEL_SCALER_BLOB  — path to scaler in input bucket (optional)
                       defaults to models/robust_scaler.pkl

Usage (local test):
  GCS_INPUT_BUCKET=spectralcrop-inputs \
  GCS_OUTPUT_BUCKET=spectralcrop-outputs \
  GCS_ENVI_BLOB=raw/2026-05-14/image.hdr \
  uv run python scripts/batch_predict.py
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("spectralcrop.batch")


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Required environment variable not set: {name}")
    return val


def _download_blob(client, bucket_name: str, blob_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(str(local_path))
    logger.info("Downloaded gs://%s/%s -> %s", bucket_name, blob_path, local_path)


def _upload_blob(client, bucket_name: str, blob_path: str, local_path: Path) -> None:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(local_path))
    logger.info("Uploaded %s -> gs://%s/%s", local_path, bucket_name, blob_path)


def main() -> None:
    try:
        from google.cloud import storage as gcs
    except ImportError as err:
        raise ImportError(
            "google-cloud-storage is required for GCP batch jobs. Install with: uv sync --extra api"
        ) from err

    import torch

    from spectralcrop.config.constants import CNN2D_BEST_THR
    from spectralcrop.data.preprocessing import preprocess_envi_to_zarr
    from spectralcrop.inference.predict import predict_image
    from spectralcrop.models.dl.predict import load_cnn2d

    # ── Read configuration from environment ──────────────────────────────
    input_bucket = _require_env("GCS_INPUT_BUCKET")
    output_bucket = _require_env("GCS_OUTPUT_BUCKET")
    envi_blob = _require_env("GCS_ENVI_BLOB")  # e.g. raw/2026-06-01/image.hdr

    weights_blob = os.environ.get("MODEL_WEIGHTS_BLOB", "models/cnn2d_final_model_weights.pt")
    scaler_blob = os.environ.get("MODEL_SCALER_BLOB", "models/robust_scaler.pkl")

    # Derive companion .bsq path from .hdr path (same name, different extension)
    bsq_blob = envi_blob.replace(".hdr", ".bsq")

    run_ts = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    run_dir = Path(envi_blob).parent.name  # e.g. "2026-06-01"
    out_prefix = f"predictions/{run_dir}/{run_ts}"

    logger.info("=== spectralcrop batch job started ===")
    logger.info("Input : gs://%s/%s", input_bucket, envi_blob)
    logger.info("Output: gs://%s/%s/", output_bucket, out_prefix)

    client = gcs.Client()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # ── Download inputs ───────────────────────────────────────────────
        logger.info("Downloading ENVI files...")
        local_hdr = tmp_path / "image.hdr"
        local_bsq = tmp_path / "image.bsq"
        local_weights = tmp_path / "weights.pt"
        local_scaler = tmp_path / "scaler.pkl"
        local_zarr = tmp_path / "cube.zarr"
        local_out_dir = tmp_path / "output"

        _download_blob(client, input_bucket, envi_blob, local_hdr)
        _download_blob(client, input_bucket, bsq_blob, local_bsq)
        _download_blob(client, input_bucket, weights_blob, local_weights)
        _download_blob(client, input_bucket, scaler_blob, local_scaler)

        # ── Preprocess ────────────────────────────────────────────────────
        logger.info("Preprocessing ENVI -> zarr...")
        preprocess_envi_to_zarr(hdr_path=local_hdr, output_zarr=local_zarr)

        # ── Inference ─────────────────────────────────────────────────────
        logger.info("Running CNN-2D inference...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, thr = load_cnn2d(
            models_dir=local_weights.parent,
            device=device,
        )

        # Override weights with the downloaded file
        import torch as _torch

        state = _torch.load(local_weights, map_location=device, weights_only=False)
        model.load_state_dict(state)

        proba_path, class_path = predict_image(
            zarr_path=local_zarr,
            output_dir=local_out_dir,
            model=model,
            scaler_path=local_scaler,
            threshold=CNN2D_BEST_THR,
            device=device,
        )

        # ── Upload results ────────────────────────────────────────────────
        logger.info("Uploading results...")
        for local_file in [proba_path, class_path]:
            _upload_blob(
                client,
                output_bucket,
                f"{out_prefix}/{local_file.name}",
                local_file,
            )

        # ── Metadata JSON ─────────────────────────────────────────────────
        metadata = {
            "run_timestamp": run_ts,
            "input_envi": f"gs://{input_bucket}/{envi_blob}",
            "model_weights": f"gs://{input_bucket}/{weights_blob}",
            "threshold": CNN2D_BEST_THR,
            "device": str(device),
            "output_proba": f"gs://{output_bucket}/{out_prefix}/prediction_proba.tif",
            "output_class": f"gs://{output_bucket}/{out_prefix}/prediction_class.tif",
        }
        meta_path = local_out_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _upload_blob(client, output_bucket, f"{out_prefix}/metadata.json", meta_path)

    logger.info("=== Batch job completed: %s ===", out_prefix)


if __name__ == "__main__":
    main()
