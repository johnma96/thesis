# spectralcrop — project automation
#
# Usage: make <target>
# Requires: uv (https://docs.astral.sh/uv/), DVC credentials in .dvc/config.local
#
# RETRAINING WORKFLOW (new labeled data):
#   make preprocess  ENVI=path/to/image.hdr
#   make labels      GPKG=path/to/labels.gpkg
#   make split       GPKG=path/to/labels.gpkg
#   make train
#   make evaluate
#   -- or in one shot --
#   make retrain     ENVI=path/to/image.hdr  GPKG=path/to/labels.gpkg
#
# PRODUCTION WORKFLOW (new image -> prediction maps):
#   make predict-new ENVI=path/to/new_image.hdr

.PHONY: help install install-gpu sync lint format test test-cov \
        preprocess labels split train evaluate predict predict-new retrain clean

# Default target
help:
	@uv run python main.py --help

# ── Environment ──────────────────────────────────────────────────────────────

install:
	uv sync --extra pytorch-cpu --extra notebooks --extra dev

install-gpu:
	uv sync --extra pytorch-cu126 --extra notebooks --extra dev

sync:
	uv run dvc pull

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	uv run ruff check spectralcrop/ tests/ main.py
	uv run ruff format --check spectralcrop/ tests/ main.py

format:
	uv run ruff format spectralcrop/ tests/ main.py
	uv run ruff check --fix spectralcrop/ tests/ main.py

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -v --tb=short --cov=spectralcrop --cov-report=term-missing

test-html:
	uv run pytest tests/ --cov=spectralcrop --cov-report=html:reports/coverage
	@echo "Coverage report: reports/coverage/index.html"

# ── Pipeline steps ────────────────────────────────────────────────────────────

preprocess:   ## ENVI=path/to/image.hdr
	uv run python main.py preprocess --envi $(ENVI)

labels:       ## GPKG=path/to/labels.gpkg
	uv run python main.py make-labels --gpkg $(GPKG)

split:        ## GPKG=path/to/labels.gpkg
	uv run python main.py make-split --gpkg $(GPKG)

train:
	uv run python main.py train-cnn2d --use-locked-hparams

evaluate:
	uv run python main.py evaluate --split test

predict:
	uv run python main.py predict

# ── Composite workflows ───────────────────────────────────────────────────────

retrain:      ## Full retraining: ENVI=... GPKG=...
	uv run python main.py full-pipeline --envi $(ENVI) --gpkg $(GPKG)

predict-new:  ## Production prediction on new image: ENVI=...
	uv run python main.py predict-pipeline --envi $(ENVI)

# ── Utilities ─────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
