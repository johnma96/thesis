"""CNN-2D model loader for the inference API.

The model is loaded once at API startup (via FastAPI lifespan) and kept
in memory.  This avoids per-request I/O overhead.

The registered MLflow model ``bean_stress_classifier`` v1 (Production) is
loaded via the local weights file for now.  A future improvement would
pull directly from the MLflow Model Registry at startup.
"""

from __future__ import annotations

import logging

import torch

from spectralcrop.config.constants import CNN2D_BEST_THR
from spectralcrop.config.paths import MODELS_DIR
from spectralcrop.models.dl.architectures import SpectralSpatialCNN2D
from spectralcrop.models.dl.predict import load_cnn2d

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Singleton holder for the loaded CNN-2D model."""

    _model: SpectralSpatialCNN2D | None = None
    _threshold: float = CNN2D_BEST_THR
    _device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def load(cls) -> None:
        """Load the final CNN-2D weights from models/."""
        logger.info("Loading CNN-2D model from %s ...", MODELS_DIR)
        cls._model, cls._threshold = load_cnn2d(MODELS_DIR, cls._device)
        logger.info(
            "Model ready. threshold=%.4f  device=%s", cls._threshold, cls._device
        )

    @classmethod
    def unload(cls) -> None:
        cls._model = None
        logger.info("Model unloaded.")

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None

    @classmethod
    def predict(cls, patch_tensor: torch.Tensor) -> tuple[float, int]:
        """Run inference on a single pre-processed patch.

        Parameters
        ----------
        patch_tensor : torch.Tensor of shape (1, 63, 5, 5)
            Scaled patch tensor (channel-first, batch dimension included).

        Returns
        -------
        prob_stressed : float
            P(class=1 | patch).
        label : int
            Hard prediction: 0 or 1.
        """
        if cls._model is None:
            raise RuntimeError("Model not loaded. Call ModelRegistry.load() first.")

        cls._model.eval()
        with torch.no_grad():
            logits = cls._model(patch_tensor.to(cls._device))
            prob = float(torch.softmax(logits, dim=1)[0, 1].item())

        label = int(prob >= cls._threshold)
        return prob, label
