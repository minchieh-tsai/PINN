"""Loss functions for level-set endpoint and geometry objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def endpoint_sdf_loss(phi_pred: torch.Tensor, phi_target: torch.Tensor, alpha: float = 2.0, sigma: float = 8.0) -> torch.Tensor:
    weights = 1.0 + alpha * torch.exp(-torch.abs(phi_target) / sigma)
    return torch.mean(weights * F.smooth_l1_loss(phi_pred, phi_target, reduction="none"))


def soft_mask(phi: torch.Tensor, epsilon_h: float = 1.0) -> torch.Tensor:
    return torch.sigmoid(-phi / epsilon_h)


def dice_loss(phi_pred: torch.Tensor, phi_target: torch.Tensor, epsilon_h: float = 1.0, eps: float = 1.0e-8) -> torch.Tensor:
    pred = soft_mask(phi_pred, epsilon_h)
    target = soft_mask(phi_target, epsilon_h)
    numerator = 2.0 * torch.sum(pred * target) + eps
    denominator = torch.sum(pred * pred) + torch.sum(target * target) + eps
    return 1.0 - numerator / denominator


def eikonal_loss(phi_x: torch.Tensor, phi_y: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    grad_norm = torch.sqrt(phi_x * phi_x + phi_y * phi_y + eps)
    return torch.mean((grad_norm - 1.0) ** 2)


def sign_loss(phi_t: torch.Tensor, process_sign: float) -> torch.Tensor:
    if process_sign > 0:
        return torch.mean(torch.relu(phi_t) ** 2)
    return torch.mean(torch.relu(-phi_t) ** 2)
