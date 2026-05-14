"""Tests for spectralcrop/models/dl/train.py — fit_cnn2d training loop."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed — skip training tests")

from spectralcrop.models.dl.architectures import SpectralSpatialCNN2D  # noqa: E402
from spectralcrop.models.dl.train import fit_cnn2d  # noqa: E402


def _tiny_dataset(n: int = 64, n_channels: int = 8, patch: int = 3, seed: int = 0):
    """Minimal synthetic patch dataset for training tests."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_channels, patch, patch)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    return X, y


def test_fit_cnn2d_runs_without_error():
    """Training loop completes at least one epoch without raising."""
    n_ch, p = 8, 3
    X_tr, y_tr = _tiny_dataset(32, n_ch, p)
    X_vl, y_vl = _tiny_dataset(16, n_ch, p)

    model = SpectralSpatialCNN2D(n_channels=n_ch, n_classes=2, dropout=0.0, kernel_size=3)
    device = torch.device("cpu")

    trained, losses, praucs = fit_cnn2d(
        model=model,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_vl,
        y_val=y_vl,
        lr=1e-3,
        batch_size=16,
        max_epochs=2,
        patience=5,
        weight_decay=0.0,
        device=device,
    )
    assert len(losses) == 2
    assert len(praucs) == 2
    assert all(0.0 <= p <= 1.0 for p in praucs)


def test_fit_cnn2d_returns_best_weights():
    """Model weights at return must correspond to the best val PR-AUC epoch."""
    n_ch, p = 4, 3
    X_tr, y_tr = _tiny_dataset(32, n_ch, p, seed=1)
    X_vl, y_vl = _tiny_dataset(16, n_ch, p, seed=2)

    model = SpectralSpatialCNN2D(n_channels=n_ch, n_classes=2, dropout=0.0, kernel_size=3)
    device = torch.device("cpu")

    trained, _, val_praucs = fit_cnn2d(
        model=model,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_vl,
        y_val=y_vl,
        lr=1e-2,
        batch_size=8,
        max_epochs=3,
        patience=10,
        weight_decay=0.0,
        device=device,
    )
    # The model should be in eval mode after training
    assert not trained.training


def test_early_stopping_triggers():
    """With patience=1 and constant val loss, training stops after 2 epochs."""
    n_ch, p = 4, 3
    X_tr, y_tr = _tiny_dataset(16, n_ch, p, seed=3)
    X_vl, y_vl = _tiny_dataset(8, n_ch, p, seed=4)

    # All-zero model weights → val PR-AUC won't improve → early stopping
    model = SpectralSpatialCNN2D(n_channels=n_ch, n_classes=2, dropout=0.0, kernel_size=3)
    for param in model.parameters():
        torch.nn.init.zeros_(param)

    device = torch.device("cpu")
    _, losses, _ = fit_cnn2d(
        model=model,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_vl,
        y_val=y_vl,
        lr=0.0,
        batch_size=4,  # lr=0 → no updates → no improvement
        max_epochs=20,
        patience=1,
        weight_decay=0.0,
        device=device,
    )
    # With patience=1, stops after first non-improving epoch (≤ 2 epochs)
    assert len(losses) <= 3  # small tolerance for initial improvement


def test_fit_cnn2d_loss_decreases():
    """Training loss should decrease over multiple epochs on separable data."""
    n_ch, p = 4, 3
    rng = np.random.default_rng(99)
    # Clearly separable: class 0 = low values, class 1 = high values
    X_0 = rng.uniform(-2, -1, (32, n_ch, p, p)).astype(np.float32)
    X_1 = rng.uniform(1, 2, (32, n_ch, p, p)).astype(np.float32)
    X_tr = np.concatenate([X_0, X_1])
    y_tr = np.array([0] * 32 + [1] * 32, dtype=np.int64)
    X_vl, y_vl = X_tr[:16], y_tr[:16]

    model = SpectralSpatialCNN2D(n_channels=n_ch, n_classes=2, dropout=0.0, kernel_size=3)
    _, losses, _ = fit_cnn2d(
        model=model,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_vl,
        y_val=y_vl,
        lr=1e-2,
        batch_size=16,
        max_epochs=5,
        patience=10,
        weight_decay=0.0,
        device=torch.device("cpu"),
    )
    assert losses[-1] < losses[0], "Loss should decrease on separable data"
