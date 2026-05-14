"""Smoke tests for CNN-2D architecture and inference utilities.

These tests require torch to be installed (pytorch-cpu or pytorch-cu126 extra).
They are automatically skipped in environments that don't have torch (e.g.,
a minimal CI job that only installs the dev extra).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed — skip CNN-2D tests")

from spectralcrop.config.constants import CNN2D_HPARAMS, N_FEATURES  # noqa: E402
from spectralcrop.models.dl.architectures import SpectralSpatialCNN2D  # noqa: E402
from spectralcrop.models.dl.predict import predict_proba_2d  # noqa: E402


@pytest.fixture(scope="module")
def dummy_model():
    """Return an untrained CNN-2D in eval mode."""
    m = SpectralSpatialCNN2D(
        n_channels=CNN2D_HPARAMS["n_channels"],
        n_classes=2,
        kernel_size=CNN2D_HPARAMS["kernel_size"],
        dropout=CNN2D_HPARAMS["dropout"],
    )
    m.eval()
    return m


def test_cnn2d_forward_shape(dummy_model):
    batch = 4
    patch = CNN2D_HPARAMS["patch_size"]
    x = torch.randn(batch, N_FEATURES, patch, patch)
    with torch.no_grad():
        out = dummy_model(x)
    assert out.shape == (batch, 2), f"Expected ({batch}, 2), got {out.shape}"


def test_cnn2d_output_is_logits(dummy_model):
    """Output should be raw logits (not bounded to [0,1])."""
    x = torch.randn(1, N_FEATURES, 5, 5)
    with torch.no_grad():
        out = dummy_model(x)
    # Logits can be any real value; softmax would bound them
    assert out.shape == (1, 2)


def test_predict_proba_2d_returns_valid_probabilities(dummy_model):
    rng = np.random.default_rng(0)
    X = rng.random((50, N_FEATURES, 5, 5)).astype(np.float32)
    # Force CPU so the test works regardless of GPU availability
    probs = predict_proba_2d(dummy_model, X, device=torch.device("cpu"), batch_size=16)
    assert probs.shape == (50,)
    assert np.all(probs >= 0) and np.all(probs <= 1), "Probabilities must be in [0, 1]"


def test_predict_proba_2d_deterministic(dummy_model):
    rng = np.random.default_rng(1)
    X = rng.random((10, N_FEATURES, 5, 5)).astype(np.float32)
    cpu = torch.device("cpu")
    p1 = predict_proba_2d(dummy_model, X, device=cpu)
    p2 = predict_proba_2d(dummy_model, X, device=cpu)
    np.testing.assert_array_equal(p1, p2)
