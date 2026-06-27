"""Encoder for the fixed 20-point contour condition."""

from __future__ import annotations

import torch
from torch import nn


class ContourEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(60, 128),
            nn.Tanh(),
            nn.Linear(128, embedding_dim),
            nn.Tanh(),
        )

    def forward(self, contour_features: torch.Tensor) -> torch.Tensor:
        if contour_features.ndim == 2:
            contour_features = contour_features.unsqueeze(0)
        flat = contour_features.reshape(contour_features.shape[0], -1)
        return self.net(flat)
