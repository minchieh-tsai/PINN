"""Loss functions for level-set endpoint and geometry objectives."""

from __future__ import annotations

from typing import Tuple

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


def levelset_derivatives(
    phi: torch.Tensor,
    features: torch.Tensor,
    duration_s: torch.Tensor,
    length_x: float,
    length_y: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return physical derivatives phi_x, phi_y, phi_t.

    The model outputs phi in physical level-set units.  The first three feature
    columns are normalized xi, eta, and tau, so only coordinate chain-rule
    factors are needed here.
    """

    grads = torch.autograd.grad(
        outputs=phi,
        inputs=features,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    phi_x = grads[:, 0] * (2.0 / float(length_x))
    phi_y = grads[:, 1] * (2.0 / float(length_y))
    phi_t = grads[:, 2] / duration_s
    return phi_x, phi_y, phi_t


def pde_residual(
    phi_x: torch.Tensor,
    phi_y: torch.Tensor,
    phi_t: torch.Tensor,
    normal_velocity: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    grad_norm = torch.sqrt(phi_x * phi_x + phi_y * phi_y + eps)
    return phi_t + normal_velocity * grad_norm


def pde_residual_loss(
    phi_x: torch.Tensor,
    phi_y: torch.Tensor,
    phi_t: torch.Tensor,
    normal_velocity: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    residual = pde_residual(phi_x, phi_y, phi_t, normal_velocity, eps=eps)
    return torch.mean(residual * residual)


def eikonal_loss(phi_x: torch.Tensor, phi_y: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    grad_norm = torch.sqrt(phi_x * phi_x + phi_y * phi_y + eps)
    return torch.mean((grad_norm - 1.0) ** 2)


def sign_loss(phi_t: torch.Tensor, process_sign: float) -> torch.Tensor:
    if process_sign > 0:
        return torch.mean(torch.relu(phi_t) ** 2)
    return torch.mean(torch.relu(-phi_t) ** 2)


def velocity_jacobian_loss(
    normal_velocity: torch.Tensor,
    features: torch.Tensor,
    length_x: float,
    length_y: float,
) -> torch.Tensor:
    """Penalize spatial roughness in predicted normal velocity."""

    if not normal_velocity.requires_grad:
        return torch.zeros((), dtype=normal_velocity.dtype, device=normal_velocity.device)

    grads = torch.autograd.grad(
        outputs=normal_velocity,
        inputs=features,
        grad_outputs=torch.ones_like(normal_velocity),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    if grads is None:
        return torch.zeros((), dtype=normal_velocity.dtype, device=normal_velocity.device)
    velocity_x = grads[:, 0] * (2.0 / float(length_x))
    velocity_y = grads[:, 1] * (2.0 / float(length_y))
    return torch.mean(velocity_x * velocity_x + velocity_y * velocity_y)
