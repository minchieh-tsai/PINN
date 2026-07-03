"""Training entry points for deposition and etch PINN models."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch

from epi_pinn.baseline import estimate_average_rate_from_pair
from epi_pinn.config import (
    average_rate,
    device_name,
    load_config,
    output_dir,
    process_config,
    project_root_from_config_path,
    rate_reference,
    schedule_seconds,
)
from epi_pinn.contour import extract_contour20
from epi_pinn.excel_io import load_state_arrays
from epi_pinn.losses import (
    curvature_velocity_loss,
    dice_loss,
    eikonal_loss,
    endpoint_mae_loss,
    endpoint_sdf_loss,
    levelset_derivatives,
    pde_residual_loss,
    sign_loss,
    velocity_jacobian_loss,
)
from epi_pinn.models import DepositionPINN, EtchPINN
from epi_pinn.sampling import (
    build_collocation_pools,
    build_features,
    full_grid_query,
    sample_collocation_indices,
    sample_endpoint_indices,
    transition_key,
)
from epi_pinn.sdf import ensure_signed_distance


LossRow = Tuple[int, str, float, float, float, float, float, float, float, float, float, float, float, float]


def torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if str(name).lower() == "float64" else torch.float32


def make_model(process_name: str, config: Mapping[str, Any]) -> torch.nn.Module:
    if process_name == "deposition":
        return DepositionPINN(config.get("model", {}))
    if process_name == "etch":
        return EtchPINN(config.get("model", {}))
    raise ValueError("process_name must be 'deposition' or 'etch'")


def _current_scalar_parameter(model: torch.nn.Module, method_name: str) -> float:
    value_fn = getattr(model, method_name, None)
    if value_fn is None:
        return float("nan")
    with torch.no_grad():
        return float(value_fn().detach().cpu())


def current_beta_kappa(model: torch.nn.Module) -> float:
    return _current_scalar_parameter(model, "curvature_velocity_weight_value")


def current_transport_alpha(model: torch.nn.Module) -> float:
    return _current_scalar_parameter(model, "transport_alpha_value")


def current_transport_ld(model: torch.nn.Module) -> float:
    return _current_scalar_parameter(model, "transport_ld_value")


def _prepare_transitions(
    config: Mapping[str, Any],
    states: Mapping[str, np.ndarray],
    process_name: str,
    infer_missing_rates: bool,
) -> List[Dict[str, Any]]:
    transitions = config.get("transitions", {}).get(transition_key(config, process_name), [])
    if not transitions:
        raise ValueError(f"No configured training transitions for {process_name}")

    proc = process_config(config, process_name)
    process_sign = float(proc["sign"])
    level_cfg = config.get("level_set", {})
    contour_cfg = config.get("contour", {})
    spatial_cfg = config.get("spatial", {})
    narrow_band = float(level_cfg.get("narrow_band_distance", 8.0))
    clip_distance = float(level_cfg.get("phi_clip_distance", 32.0))
    duration_reference = float(proc.get("duration_reference_s", 1.0))
    pixel_size_x = float(spatial_cfg.get("pixel_size_x", 1.0))
    pixel_size_y = float(spatial_cfg.get("pixel_size_y", 1.0))
    prepared: List[Dict[str, Any]] = []

    for transition in transitions:
        cycle = int(transition["cycle"])
        input_state = str(transition["input_state"])
        target_state = str(transition["target_state"])
        phi_initial = ensure_signed_distance(states[input_state], level_cfg)
        phi_target = ensure_signed_distance(states[target_state], level_cfg)
        duration = schedule_seconds(config, process_name, cycle)
        inferred_rate = None
        if infer_missing_rates:
            inferred_rate = estimate_average_rate_from_pair(
                phi_initial,
                phi_target,
                duration,
                process_sign,
                narrow_band_distance=narrow_band,
            )
        rate = average_rate(config, process_name, cycle, fallback=inferred_rate)
        rate_ref = rate_reference(config, process_name, rate)
        contour = extract_contour20(
            phi_initial,
            num_points=int(contour_cfg.get("num_points", 20)),
            min_valid_points=int(contour_cfg.get("min_valid_points", 10)),
            crossing_policy=str(contour_cfg.get("crossing_policy", "closest_to_previous")),
            first_crossing_policy=str(contour_cfg.get("first_crossing_policy", "topmost")),
        )

        height, width = phi_initial.shape
        xi, eta, _x, _y = full_grid_query(height, width)
        tau = np.ones_like(xi)
        features, raw_phi0 = build_features(
            phi_initial,
            contour,
            xi,
            eta,
            tau,
            duration,
            rate,
            duration_reference,
            rate_ref,
            clip_distance,
            process_sign,
        )
        prepared.append(
            {
                "id": transition["id"],
                "cycle": cycle,
                "phi_target": phi_target.reshape(-1),
                "features": features,
                "raw_phi0": raw_phi0,
                "contour": contour.as_features(),
                "duration_s": duration,
                "average_rate": rate,
                "narrow_band": narrow_band,
                "clip_distance": clip_distance,
                "process_sign": process_sign,
                "length_x": max(pixel_size_x, (width - 1) * pixel_size_x),
                "length_y": max(pixel_size_y, (height - 1) * pixel_size_y),
                "collocation_pools": build_collocation_pools(phi_initial, contour, narrow_band),
            }
        )
    return prepared


def _sample_training_batch(
    item: Mapping[str, Any],
    batch_size: int,
    endpoint_fraction: float,
    collocation_batch_size: int,
    collocation_interface_fraction: float,
    collocation_contour_fraction: float,
    collocation_global_fraction: float,
    use_physics_terms: bool,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    batch = {
        "endpoint_indices": sample_endpoint_indices(
            item["phi_target"],
            batch_size,
            endpoint_fraction,
            item["narrow_band"],
            rng,
        )
    }
    if use_physics_terms:
        collocation_indices, collocation_tau = sample_collocation_indices(
            item["collocation_pools"],
            collocation_batch_size,
            collocation_interface_fraction,
            collocation_contour_fraction,
            collocation_global_fraction,
            rng,
        )
        batch["collocation_indices"] = collocation_indices
        batch["collocation_tau"] = collocation_tau
    return batch


def _compute_training_losses(
    model: torch.nn.Module,
    item: Mapping[str, Any],
    batch: Mapping[str, np.ndarray],
    dtype: torch.dtype,
    device: str,
    loss_weights: Mapping[str, float],
    use_physics_terms: bool,
) -> Dict[str, torch.Tensor]:
    endpoint_indices = batch["endpoint_indices"]
    features = torch.as_tensor(item["features"][endpoint_indices], dtype=dtype, device=device)
    raw_phi0 = torch.as_tensor(item["raw_phi0"][endpoint_indices], dtype=dtype, device=device)
    phi_target = torch.as_tensor(item["phi_target"][endpoint_indices], dtype=dtype, device=device)
    contour = torch.as_tensor(item["contour"], dtype=dtype, device=device)
    duration = torch.tensor(float(item["duration_s"]), dtype=dtype, device=device)
    rate = torch.tensor(float(item["average_rate"]), dtype=dtype, device=device)

    phi_pred, _velocity = model(features, contour, raw_phi0, duration, rate, item["clip_distance"])
    sdf = endpoint_sdf_loss(phi_pred, phi_target)
    dice = dice_loss(phi_pred, phi_target)
    mae = endpoint_mae_loss(phi_pred, phi_target)
    zero = torch.zeros((), dtype=dtype, device=device)
    pde = zero
    eikonal = zero
    sign = zero
    velocity_jacobian = zero
    curvature_velocity = zero
    loss = loss_weights["sdf"] * sdf + loss_weights["dice"] * dice + loss_weights["mae"] * mae

    if use_physics_terms:
        collocation_indices = batch["collocation_indices"]
        collocation_tau = batch["collocation_tau"]
        collocation_features_np = item["features"][collocation_indices].copy()
        collocation_features_np[:, 2] = collocation_tau
        collocation_features = torch.as_tensor(collocation_features_np, dtype=dtype, device=device).requires_grad_(True)
        collocation_raw_phi0 = torch.as_tensor(item["raw_phi0"][collocation_indices], dtype=dtype, device=device)
        collocation_phi, collocation_velocity = model(
            collocation_features,
            contour,
            collocation_raw_phi0,
            duration,
            rate,
            item["clip_distance"],
            length_y=item["length_y"],
        )
        phi_x, phi_y, phi_t = levelset_derivatives(
            collocation_phi,
            collocation_features,
            duration,
            item["length_x"],
            item["length_y"],
        )
        pde = pde_residual_loss(phi_x, phi_y, phi_t, collocation_velocity)
        eikonal = eikonal_loss(phi_x, phi_y)
        sign = sign_loss(phi_t, float(item["process_sign"]))
        if loss_weights["velocity_jacobian"] > 0.0:
            velocity_jacobian = velocity_jacobian_loss(
                collocation_velocity,
                collocation_features,
                item["length_x"],
                item["length_y"],
            )
        if loss_weights["curvature_velocity"] > 0.0:
            velocity_components = model.velocity_components(
                collocation_features,
                contour,
                rate,
                length_y=item["length_y"],
            )
            curvature_velocity = curvature_velocity_loss(velocity_components["curvature"])
        loss = (
            loss
            + loss_weights["pde"] * pde
            + loss_weights["eikonal"] * eikonal
            + loss_weights["sign"] * sign
            + loss_weights["velocity_jacobian"] * velocity_jacobian
            + loss_weights["curvature_velocity"] * curvature_velocity
        )

    return {
        "loss": loss,
        "sdf": sdf,
        "dice": dice,
        "mae": mae,
        "pde": pde,
        "eikonal": eikonal,
        "sign": sign,
        "velocity_jacobian": velocity_jacobian,
        "curvature_velocity": curvature_velocity,
    }


def _append_log_row(rows: List[LossRow], step: int, item: Mapping[str, Any], losses: Mapping[str, torch.Tensor], model: torch.nn.Module) -> None:
    rows.append(
        (
            step,
            str(item["id"]),
            float(losses["loss"].detach().cpu()),
            float(losses["sdf"].detach().cpu()),
            float(losses["dice"].detach().cpu()),
            float(losses["mae"].detach().cpu()),
            float(losses["pde"].detach().cpu()),
            float(losses["eikonal"].detach().cpu()),
            float(losses["sign"].detach().cpu()),
            float(losses["velocity_jacobian"].detach().cpu()),
            float(losses["curvature_velocity"].detach().cpu()),
            current_beta_kappa(model),
            current_transport_alpha(model),
            current_transport_ld(model),
        )
    )


def _save_checkpoint(model: torch.nn.Module, path: Path, process_name: str, best_loss: float, config: Mapping[str, Any]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "process_name": process_name,
            "best_loss": best_loss,
            "config": config,
        },
        path,
    )


def train_process(config_path: str, process_name: str, infer_missing_rates: bool = False) -> Path:
    config = load_config(config_path)
    root = project_root_from_config_path(config_path)
    states = load_state_arrays(config, base_dir=root)
    training_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})
    sampling_cfg = config.get("sampling", {})
    device = device_name(config)
    dtype = torch_dtype(training_cfg.get("dtype", "float64"))
    torch.manual_seed(int(training_cfg.get("seed", 42)))
    rng = np.random.default_rng(int(training_cfg.get("seed", 42)))

    prepared = _prepare_transitions(config, states, process_name, infer_missing_rates)
    model = make_model(process_name, config).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg.get("adam_lr", 1.0e-3)))
    steps = int(training_cfg.get("adam_steps", 10000))
    batch_size = int(training_cfg.get("endpoint_batch_size", 4096))
    collocation_batch_size = int(training_cfg.get("collocation_batch_size", batch_size))
    log_every = int(training_cfg.get("log_every", 100))
    checkpoint_every = int(training_cfg.get("checkpoint_every", 500))
    grad_clip = float(training_cfg.get("grad_clip_norm", 10.0))
    endpoint_fraction = float(sampling_cfg.get("endpoint_interface_fraction", 0.70))
    collocation_interface_fraction = float(sampling_cfg.get("collocation_interface_fraction", 0.60))
    collocation_contour_fraction = float(sampling_cfg.get("collocation_contour_fraction", 0.20))
    collocation_global_fraction = float(sampling_cfg.get("collocation_global_fraction", 0.20))

    loss_weights = {
        "sdf": float(loss_cfg.get("sdf", 1.0)),
        "dice": float(loss_cfg.get("dice", 0.5)),
        "mae": float(loss_cfg.get("mae", 0.0)),
        "pde": float(loss_cfg.get("pde", 1.0)),
        "eikonal": float(loss_cfg.get("eikonal", 0.02)),
        "sign": float(loss_cfg.get("sign", 0.05)),
        "velocity_jacobian": float(loss_cfg.get("velocity_jacobian", 0.0)),
        "curvature_velocity": float(loss_cfg.get("curvature_velocity", 0.0)),
    }
    use_physics_terms = collocation_batch_size > 0 and (
        loss_weights["pde"] > 0.0
        or loss_weights["eikonal"] > 0.0
        or loss_weights["sign"] > 0.0
        or loss_weights["velocity_jacobian"] > 0.0
        or loss_weights["curvature_velocity"] > 0.0
    )

    out_dir = output_dir(config, root)
    checkpoint_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{process_name}_training.csv"
    best_path = checkpoint_dir / f"{process_name}_best.pt"

    best_loss = float("inf")
    rows: List[LossRow] = []
    for step in range(1, steps + 1):
        item = prepared[(step - 1) % len(prepared)]
        batch = _sample_training_batch(
            item,
            batch_size,
            endpoint_fraction,
            collocation_batch_size,
            collocation_interface_fraction,
            collocation_contour_fraction,
            collocation_global_fraction,
            use_physics_terms,
            rng,
        )
        optimizer.zero_grad(set_to_none=True)
        losses = _compute_training_losses(model, item, batch, dtype, device, loss_weights, use_physics_terms)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_value = float(losses["loss"].detach().cpu())
        if step % log_every == 0 or step == 1:
            _append_log_row(rows, step, item, losses, model)
            print(
                f"[{process_name}] step={step}/{steps} transition={item['id']} "
                f"loss={loss_value:.6g} beta_kappa={current_beta_kappa(model):.6g} "
                f"alpha_transport={current_transport_alpha(model):.6g} Ld={current_transport_ld(model):.6g}",
                flush=True,
            )
        if loss_value < best_loss or step % checkpoint_every == 0:
            if loss_value < best_loss:
                best_loss = loss_value
            _save_checkpoint(model, best_path, process_name, best_loss, config)

    use_lbfgs = bool(training_cfg.get("use_lbfgs", False))
    lbfgs_steps = int(training_cfg.get("lbfgs_steps", 0))
    if use_lbfgs and lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=float(training_cfg.get("lbfgs_lr", 1.0)),
            max_iter=int(training_cfg.get("lbfgs_max_iter", 20)),
            history_size=int(training_cfg.get("lbfgs_history_size", 50)),
            line_search_fn=str(training_cfg.get("lbfgs_line_search_fn", "strong_wolfe")),
        )
        for lbfgs_step in range(1, lbfgs_steps + 1):
            global_step = steps + lbfgs_step
            item = prepared[(global_step - 1) % len(prepared)]
            batch = _sample_training_batch(
                item,
                batch_size,
                endpoint_fraction,
                collocation_batch_size,
                collocation_interface_fraction,
                collocation_contour_fraction,
                collocation_global_fraction,
                use_physics_terms,
                rng,
            )
            latest_losses: Dict[str, torch.Tensor] = {}

            def closure() -> torch.Tensor:
                lbfgs.zero_grad(set_to_none=True)
                losses = _compute_training_losses(model, item, batch, dtype, device, loss_weights, use_physics_terms)
                losses["loss"].backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                latest_losses.clear()
                latest_losses.update(losses)
                return losses["loss"]

            lbfgs.step(closure)
            losses = latest_losses
            loss_value = float(losses["loss"].detach().cpu())
            if global_step % log_every == 0 or lbfgs_step == 1:
                _append_log_row(rows, global_step, item, losses, model)
                print(
                    f"[{process_name}] lbfgs_step={lbfgs_step}/{lbfgs_steps} global_step={global_step} "
                    f"transition={item['id']} loss={loss_value:.6g} "
                    f"beta_kappa={current_beta_kappa(model):.6g} "
                    f"alpha_transport={current_transport_alpha(model):.6g} Ld={current_transport_ld(model):.6g}",
                    flush=True,
                )
            if loss_value < best_loss or global_step % checkpoint_every == 0:
                if loss_value < best_loss:
                    best_loss = loss_value
                _save_checkpoint(model, best_path, process_name, best_loss, config)

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "transition_id",
                "loss",
                "sdf_loss",
                "dice_loss",
                "mae_loss",
                "pde_loss",
                "eikonal_loss",
                "sign_loss",
                "velocity_jacobian_loss",
                "curvature_velocity_loss",
                "beta_kappa",
                "transport_alpha",
                "transport_ld",
            ]
        )
        writer.writerows(rows)
    return best_path
