# Rollout Physics Loss Implementation Report

## Summary

This change implements the spec's process-specific effective velocity formula inside the existing deposition/etch PINN model and training loop. The PDE residual now uses an effective normal velocity composed from base velocity, transport depth decay, and curvature velocity. A curvature velocity regularization loss is also logged and weighted in training.

## Files Changed

- `src/epi_pinn/models/conditional_levelset_pinn.py`
- `src/epi_pinn/losses.py`
- `src/epi_pinn/train.py`
- `src/epi_pinn/rollout.py`
- `src/epi_pinn/training_plots.py`
- `configs/default.yaml`
- `tests/test_curvature_velocity.py`

## Data Leakage Prevention

The existing training loop still supervises only configured train transitions. `5M` and `5E` remain evaluation/rollout states and are not added as supervised training targets in this change.

## Process-Specific Velocity Model

For each process model, the effective velocity is:

```text
Veff = Vbase + Vtransport + Vcurvature
```

`Vbase` keeps the existing process sign convention and rate-conditioned neural residual. Deposition remains positive, etch remains negative through `process_sign`.

## Transport Term

The transport term uses the requested correction form:

```text
Vtransport = alpha * Vbase * (exp(-Zdepth / Ld) - 1)
```

`Zdepth` is computed from normalized `eta - contour_eta`; when `length_y` is available, it is converted back to pixel depth. `alpha` is bounded by `transport_alpha_max`, and `Ld` is constrained positive by `transport_ld_min + softplus(raw_ld)` when learning is enabled.

## Curvature Term

The curvature velocity term is exposed as a separate velocity component and can use either the legacy tanh-bounded form or the spec-oriented linear form:

```text
Vcurvature = sign * process_sign * average_rate * beta * Mcoeff * kappa
```

The default config now selects `curvature_velocity_form: linear` and keeps `beta` bounded by `curvature_velocity_weight_max`.

## Loss Implementation

The PDE loss uses `Veff` directly:

```text
phi_t + Veff * |grad(phi)|
```

The training loss also supports:

```text
curvature_velocity_loss = mean(Vcurvature^2)
```

with weight `loss.curvature_velocity`.

## Config Options

Added model options:

```yaml
use_transport_velocity: true
learn_transport_alpha: true
transport_alpha: 0.05
transport_alpha_max: 1.0
learn_transport_ld: true
transport_ld: 50.0
transport_ld_min: 1.0
transport_depth_unit: pixel
curvature_velocity_form: linear
curvature_mcoeff: 1.0
```

Added loss option:

```yaml
curvature_velocity: 1.0e-3
```

## Training Log Keys

Training CSVs now include:

```text
curvature_velocity_loss
transport_alpha
transport_ld
```

The plotting helper also includes `curvature_velocity_loss` in raw, weighted, and share plots.

## Verified Behavior

Static/compile verification passed in this environment. Torch-based behavior tests were added, but could not be executed here because neither available Python runtime has `torch` installed.

## Known Limitations

This is not yet a full joint differentiable rollout trainer. The existing repository trains deposition and etch models separately, so this change does not yet backpropagate a multi-step `2E -> ... -> 5E` rollout physics loss across both models in one optimizer step.

## Next Recommendations

Add a joint training script that owns both models, rolls from `2E` through `5E`, applies physics-only losses on each rollout transition, and keeps `5M`/`5E` out of supervised endpoint loss.