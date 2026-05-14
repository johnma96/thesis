"""CNN architectures for hyperspectral stress classification.

Two architectures are defined here, matching exactly the implementations
trained in notebooks/305-jmmz-dl-binary-modeling.ipynb:

- ``Spectral1DCNN``       — spectral-only (1D convolutions along band axis)
- ``SpectralSpatialCNN2D``— spectro-spatial (2D convolutions on 5×5 patches)

The ``SpectralSpatialCNN2D`` is the final production model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Spectral1DCNN(nn.Module):
    """1D CNN operating on the spectral signature of a single pixel.

    Input shape : ``(batch, 1, n_features)``  — 1 channel, length = 63
    Output shape: ``(batch, n_classes)``
    """

    def __init__(
        self,
        n_classes: int = 2,
        kernel_size: int = 9,
        dropout: float = 0.3628,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(1, 16, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(32 * 8, 64)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


class SpectralSpatialCNN2D(nn.Module):
    """2D CNN operating on spectro-spatial patches (the production model).

    Input shape : ``(batch, n_channels, patch_h, patch_w)``
                  with ``n_channels=63, patch_h=patch_w=5``
    Output shape: ``(batch, n_classes)``

    MLflow run_id: 61a3cc05f39d46f79f2e3fa3d29fae7f
    PR-AUC (test): 0.9635
    """

    def __init__(
        self,
        n_channels: int = 63,
        n_classes: int = 2,
        dropout: float = 0.3728,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=pad)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)
