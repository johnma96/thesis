#!/bin/bash
# spectralcrop — convenience wrapper for common pipeline commands
# Usage: ./run.sh [COMMAND]
#   ./run.sh train       → train CNN-2D with locked hyperparameters
#   ./run.sh evaluate    → evaluate CNN-2D on test set
#   ./run.sh pipeline    → full train + evaluate

set -euo pipefail
COMMAND="${1:-help}"

case "$COMMAND" in
  train)
    uv run python main.py train-cnn2d --use-locked-hparams
    ;;
  evaluate)
    uv run python main.py evaluate --model cnn2d --split test
    ;;
  pipeline)
    uv run python main.py full-pipeline
    ;;
  *)
    uv run python main.py --help
    ;;
esac