"""Conditional level-set PINN with hard initial condition."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from epi_pinn.sampling import FEATURE_NAMES
from epi_pinn.models.contour_encoder import ContourEncoder


def _mlp(input_dim: int, hidden_dim: int, depth: int, output_dim: int) -> nn.Sequential:
    layers = []
    current = input_dim
    for _ in range(max(1, depth)):
        layers.append(nn.Linear(current, hidden_dim))
        layers.append(nn.Tanh())
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


def _logit_clamped(value: float, eps: float = 1.0e-6) -> float:
    clipped = max(min(float(value), 1.0 - eps), eps)
    return math.log(clipped / (1.0 - clipped))


def _softplus_inverse(value: float, eps: float = 1.0e-8) -> float:
    positive = max(float(value), eps)
    return math.log(math.expm1(positive))


class LevelSetPINN(nn.Module):
    def __init__(self, process_sign: float, model_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.process_sign = float(process_sign)
        self.correction_scale = float(model_config.get("correction_scale", 0.5))
        self.velocity_residual_fraction = float(model_config.get("velocity_residual_fraction", 0.5))

        self.use_transport_velocity = bool(model_config.get("use_transport_velocity", False))
        self.learn_transport_alpha = bool(model_config.get("learn_transport_alpha", False))
        initial_transport_alpha = float(model_config.get("transport_alpha", model_config.get("transport_alpha_init", 0.0)))
        self.transport_alpha_max = float(model_config.get("transport_alpha_max", 1.0))
        if self.transport_alpha_max <= 0.0:
            raise ValueError("transport_alpha_max must be positive")
        if self.learn_transport_alpha:
            if initial_transport_alpha < 0.0:
                raise ValueError("transport_alpha must be non-negative when learn_transport_alpha is true")
            raw_alpha = _logit_clamped(initial_transport_alpha / self.transport_alpha_max)
            self.raw_transport_alpha = nn.Parameter(torch.tensor(raw_alpha, dtype=torch.float32))
        else:
            self.transport_alpha = initial_transport_alpha

        self.learn_transport_ld = bool(model_config.get("learn_transport_ld", False))
        self.transport_ld_min = float(model_config.get("transport_ld_min", 1.0))
        initial_transport_ld = float(model_config.get("transport_ld", model_config.get("transport_ld_init", 50.0)))
        if self.transport_ld_min <= 0.0:
            raise ValueError("transport_ld_min must be positive")
        if initial_transport_ld < self.transport_ld_min:
            raise ValueError("transport_ld must be greater than or equal to transport_ld_min")
        if self.learn_transport_ld:
            raw_ld = _softplus_inverse(initial_transport_ld - self.transport_ld_min)
            self.raw_transport_ld = nn.Parameter(torch.tensor(raw_ld, dtype=torch.float32))
        else:
            self.transport_ld = initial_transport_ld
        self.transport_depth_unit = str(model_config.get("transport_depth_unit", "pixel")).lower()

        self.use_curvature_velocity = bool(model_config.get("use_curvature_velocity", False))
        initial_curvature_velocity_weight = float(model_config.get("curvature_velocity_weight", 0.0))
        self.learn_curvature_velocity_weight = bool(model_config.get("learn_curvature_velocity_weight", False))
        default_curvature_velocity_weight_max = max(abs(initial_curvature_velocity_weight) * 5.0, 1.0e-6)
        self.curvature_velocity_weight_max = float(
            model_config.get("curvature_velocity_weight_max", default_curvature_velocity_weight_max)
        )
        if self.curvature_velocity_weight_max <= 0.0:
            raise ValueError("curvature_velocity_weight_max must be positive")
        if self.learn_curvature_velocity_weight:
            if initial_curvature_velocity_weight < 0.0:
                raise ValueError(
                    "curvature_velocity_weight must be non-negative when "
                    "learn_curvature_velocity_weight is true; use curvature_velocity_sign for direction"
                )
            raw_weight = _logit_clamped(initial_curvature_velocity_weight / self.curvature_velocity_weight_max)
            self.raw_curvature_velocity_weight = nn.Parameter(torch.tensor(raw_weight, dtype=torch.float32))
        else:
            self.curvature_velocity_weight = initial_curvature_velocity_weight
        self.curvature_velocity_sign = float(model_config.get("curvature_velocity_sign", 1.0))
        self.curvature_reference = float(model_config.get("curvature_reference", 0.1))
        self.curvature_velocity_form = str(model_config.get("curvature_velocity_form", "tanh")).lower()
        self.curvature_mcoeff = float(model_config.get("curvature_mcoeff", 1.0))
        self.hard_initial_condition = bool(model_config.get("hard_initial_condition", True))
        embedding_dim = int(model_config.get("contour_embedding_dim", 64))
        solution_hidden = int(model_config.get("solution_hidden_dim", 128))
        solution_depth = int(model_config.get("solution_depth", 6))
        velocity_hidden = int(model_config.get("velocity_hidden_dim", 64))
        velocity_depth = int(model_config.get("velocity_depth", 4))

        base_dim = len(FEATURE_NAMES)
        self.contour_encoder = ContourEncoder(embedding_dim=embedding_dim)
        self.solution_net = _mlp(base_dim + embedding_dim, solution_hidden, solution_depth, 1)
        self.velocity_net = _mlp(base_dim + embedding_dim, velocity_hidden, velocity_depth, 1)

    def curvature_velocity_weight_value(self) -> torch.Tensor:
        if self.learn_curvature_velocity_weight:
            return self.curvature_velocity_weight_max * torch.sigmoid(self.raw_curvature_velocity_weight)
        return torch.tensor(float(self.curvature_velocity_weight), dtype=torch.float32)

    def transport_alpha_value(self) -> torch.Tensor:
        if self.learn_transport_alpha:
            return self.transport_alpha_max * torch.sigmoid(self.raw_transport_alpha)
        return torch.tensor(float(self.transport_alpha), dtype=torch.float32)

    def transport_ld_value(self) -> torch.Tensor:
        if self.learn_transport_ld:
            return self.transport_ld_min + F.softplus(self.raw_transport_ld)
        return torch.tensor(float(self.transport_ld), dtype=torch.float32)

    def _curvature_velocity_weight_for(self, features: torch.Tensor) -> torch.Tensor:
        return self.curvature_velocity_weight_value().to(dtype=features.dtype, device=features.device)

    def _transport_alpha_for(self, features: torch.Tensor) -> torch.Tensor:
        return self.transport_alpha_value().to(dtype=features.dtype, device=features.device)

    def _transport_ld_for(self, features: torch.Tensor) -> torch.Tensor:
        return self.transport_ld_value().to(dtype=features.dtype, device=features.device)

    def _compose_features(self, features: torch.Tensor, contour_features: torch.Tensor) -> torch.Tensor:
        if contour_features.ndim == 2:
            contour_features = contour_features.unsqueeze(0)
        embedding = self.contour_encoder(contour_features)
        if embedding.shape[0] == 1:
            embedding = embedding.expand(features.shape[0], -1)
        elif embedding.shape[0] != features.shape[0]:
            raise ValueError("Contour embedding batch size must be 1 or match feature batch size")
        return torch.cat([features, embedding], dim=1)

    def _transport_depth(self, features: torch.Tensor, length_y: Optional[float]) -> torch.Tensor:
        eta = features[:, 1]
        contour_eta = features[:, 12]
        normalized_depth = torch.relu(eta - contour_eta)
        if self.transport_depth_unit == "pixel" and length_y is not None:
            return normalized_depth * (0.5 * float(length_y))
        return normalized_depth

    def _velocity_components(
        self,
        features: torch.Tensor,
        composed: torch.Tensor,
        average_rate: torch.Tensor,
        length_y: Optional[float],
    ) -> Dict[str, torch.Tensor]:
        v_psi = self.velocity_net(composed).squeeze(-1)
        base = (
            self.process_sign
            * average_rate
            * (1.0 + self.velocity_residual_fraction * torch.tanh(v_psi))
        )
        transport = torch.zeros_like(base)
        if self.use_transport_velocity:
            depth = self._transport_depth(features, length_y)
            alpha = self._transport_alpha_for(features)
            ld = torch.clamp(self._transport_ld_for(features), min=1.0e-12)
            transport = alpha * base * (torch.exp(-depth / ld) - 1.0)

        curvature = torch.zeros_like(base)
        if self.use_curvature_velocity:
            kappa0 = features[:, 11]
            if self.curvature_velocity_form in {"linear", "raw"}:
                curvature_effect = self.curvature_mcoeff * kappa0
            else:
                curvature_reference = max(self.curvature_reference, 1.0e-12)
                curvature_effect = self.curvature_mcoeff * torch.tanh(kappa0 / curvature_reference)
            curvature = (
                self.curvature_velocity_sign
                * self.process_sign
                * average_rate
                * self._curvature_velocity_weight_for(features)
                * curvature_effect
            )
        total = base + transport + curvature
        return {"base": base, "transport": transport, "curvature": curvature, "total": total}

    def velocity_components(
        self,
        features: torch.Tensor,
        contour_features: torch.Tensor,
        average_rate: torch.Tensor,
        length_y: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        composed = self._compose_features(features, contour_features)
        return self._velocity_components(features, composed, average_rate, length_y)

    def forward(
        self,
        features: torch.Tensor,
        contour_features: torch.Tensor,
        raw_phi0: torch.Tensor,
        duration_s: torch.Tensor,
        average_rate: torch.Tensor,
        clip_distance: float,
        length_y: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        composed = self._compose_features(features, contour_features)
        tau = features[:, 2]
        g_theta = self.solution_net(composed).squeeze(-1)

        nominal = raw_phi0 - tau * duration_s * self.process_sign * average_rate
        correction = tau * float(clip_distance) * self.correction_scale * torch.tanh(g_theta)
        if self.hard_initial_condition:
            phi = nominal + correction
        else:
            phi = raw_phi0 + correction

        velocity = self._velocity_components(features, composed, average_rate, length_y)["total"]
        return phi, velocity

    @torch.no_grad()
    def predict_numpy(
        self,
        features_np,
        contour_features_np,
        raw_phi0_np,
        duration_s: float,
        average_rate: float,
        clip_distance: float,
        device: str,
        dtype: torch.dtype,
        batch_size: int = 16384,
        length_y: Optional[float] = None,
    ):
        self.eval()
        outputs = []
        contour = torch.as_tensor(contour_features_np, dtype=dtype, device=device)
        duration = torch.tensor(float(duration_s), dtype=dtype, device=device)
        rate = torch.tensor(float(average_rate), dtype=dtype, device=device)
        for start in range(0, features_np.shape[0], batch_size):
            end = start + batch_size
            features = torch.as_tensor(features_np[start:end], dtype=dtype, device=device)
            raw_phi0 = torch.as_tensor(raw_phi0_np[start:end], dtype=dtype, device=device)
            phi, _velocity = self.forward(features, contour, raw_phi0, duration, rate, clip_distance, length_y=length_y)
            outputs.append(phi.detach().cpu())
        return torch.cat(outputs, dim=0).numpy()


class DepositionPINN(LevelSetPINN):
    def __init__(self, model_config: Mapping[str, Any]) -> None:
        super().__init__(process_sign=1.0, model_config=model_config)


class EtchPINN(LevelSetPINN):
    def __init__(self, model_config: Mapping[str, Any]) -> None:
        super().__init__(process_sign=-1.0, model_config=model_config)
