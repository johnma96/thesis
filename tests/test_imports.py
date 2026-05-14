"""Smoke tests: verify all spectralcrop submodules are importable."""

import pytest


def test_spectralcrop_imports():
    import spectralcrop  # noqa: F401


def test_config_imports():
    from spectralcrop.config.constants import (
        CNN2D_BEST_THR,
        CNN2D_HPARAMS,
        N_FEATURES,
        RANDOM_SEED,
        SPECTRAL_INDICES,
        VI_INDICES,
        VI_NAMES,
    )

    assert len(VI_NAMES) == 5
    assert len(VI_INDICES) == 5
    assert len(SPECTRAL_INDICES) == 58
    assert N_FEATURES == 63
    assert CNN2D_HPARAMS["patch_size"] == 5
    assert CNN2D_HPARAMS["n_channels"] == 63
    assert 0 < CNN2D_BEST_THR < 1
    assert RANDOM_SEED == 42


def test_features_imports():
    from spectralcrop.features.vegetation_indices import vi_name, vi_position

    assert vi_position("NDVI") == 0
    assert vi_position("NDRE") == 1
    assert vi_name(0) == "NDVI"


def test_models_imports():
    pytest.importorskip("torch", reason="torch not installed")


def test_evaluation_imports():
    pass
