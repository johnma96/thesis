"""Inference utilities for DL models.

Handles batched probability prediction for both the CNN-1D and CNN-2D
architectures without reloading the model each call.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from spectralcrop.models.dl.architectures import SpectralSpatialCNN2D


def predict_proba_1d(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """Return positive-class probabilities from a CNN-1D model.

    Parameters
    ----------
    model : nn.Module
        Loaded and eval-mode CNN-1D instance.
    X : ndarray of shape (N, n_features)
        Scaled feature matrix (external scaler already applied).
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.
    batch_size : int, optional
        Mini-batch size for inference.

    Returns
    -------
    ndarray of shape (N,)
        Softmax probability for class 1 (stressed).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    probs: list[float] = []
    Xt = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)  # (N, 1, L)
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            logits = model(Xt[i : i + batch_size].to(device))
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.array(probs)


def predict_proba_2d(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Return positive-class probabilities from a CNN-2D model.

    Parameters
    ----------
    model : nn.Module
        Loaded and eval-mode CNN-2D instance.
    X : ndarray of shape (N, C, patch_h, patch_w)
        Patch tensor (channel-first, scaled).
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.
    batch_size : int, optional
        Mini-batch size for inference.

    Returns
    -------
    ndarray of shape (N,)
        Softmax probability for class 1 (stressed).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    probs: list[float] = []
    Xt = torch.from_numpy(X.astype(np.float32))
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            logits = model(Xt[i : i + batch_size].to(device))
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.array(probs)


def load_cnn2d(
    models_dir: Path,
    device: torch.device | None = None,
) -> tuple[SpectralSpatialCNN2D, float]:
    """Load the final CNN-2D model and its optimal decision threshold.

    Parameters
    ----------
    models_dir : Path
        Directory containing ``cnn2d_final_model_weights.pt`` and
        ``cnn2d_final_model_info.json``.
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.

    Returns
    -------
    model : SpectralSpatialCNN2D
        Model in eval mode, moved to *device*.
    best_thr : float
        Optimal decision threshold (maximises macro-F1 on validation set).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(models_dir / "cnn2d_final_model_info.json") as f:
        info = json.load(f)

    model = SpectralSpatialCNN2D(
        n_channels=info["n_channels"],
        n_classes=2,
        kernel_size=info["kernel_size"],
        dropout=info["dropout"],
    ).to(device)

    state = torch.load(
        models_dir / "cnn2d_final_model_weights.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state)
    model.eval()
    return model, float(info["best_thr"])
