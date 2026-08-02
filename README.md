# EPI Level-Set PINN POC

This repository is generated from `PINN_EPI_CODEX_SPEC_H350_W200_POC.md`.
It provides a proof-of-concept Python package for reading EPI level-set
workbooks, preprocessing signed-distance fields, training separate deposition
and etch PINN models, rolling out from `2E` to `5E`, and evaluating holdout
states `5M` and `5E`.

## Requirements

- Python 3.11+
- XLSX input at `data/raw/deposition.xlsx`
- Average interface rates in `configs/default.yaml`, or `--infer-missing-rates`

Install dependencies:

```bash
python -m pip install -e .
```

## Data Layout

If all states are in one workbook, configure `configs/default.yaml` like this:

```yaml
data:
  workbooks:
    all: data/raw/deposition.xlsx
  state_sources:
    init: {workbook: all, sheet: "init"}
    1M: {workbook: all, sheet: "1M"}
    1E: {workbook: all, sheet: "1E"}
    2M: {workbook: all, sheet: "2M"}
    2E: {workbook: all, sheet: "2E"}
    5M: {workbook: all, sheet: "5M"}
    5E: {workbook: all, sheet: "5E"}
```

All arrays are handled as `(H, W) = (350, 200)` with `phi[y, x]` indexing.

## Workflow

```bash
python scripts/inspect_xlsx.py --config configs/default.yaml
python scripts/preprocess_data.py --config configs/default.yaml --split all
python scripts/train_deposition.py --config configs/default.yaml --infer-missing-rates
python scripts/train_etch.py --config configs/default.yaml --infer-missing-rates
python scripts/run_rollout.py --config configs/default.yaml --infer-missing-rates
python scripts/evaluate_holdout.py --config configs/default.yaml
```

## Rate-Velocity Smoothing Ablation

Run three controlled 5,000-step rate-velocity experiments: restored-capacity
control, normal-consistency only, and combined mild smoothing. Each checkpoint
is evaluated with rollout smoothing `sigma=0` and `sigma=0.5` without
retraining:

```bash
python scripts/run_rate_velocity_smoothing_ablation.py \
  --infer-missing-rates
```

Add `--full-train-winner` to automatically retrain the selected configuration
for 20,000 Adam steps after the short experiments:

```bash
python scripts/run_rate_velocity_smoothing_ablation.py \
  --infer-missing-rates \
  --full-train-winner
```

To recompute predictions and metrics from existing short-run checkpoints:

```bash
python scripts/run_rate_velocity_smoothing_ablation.py \
  --infer-missing-rates \
  --skip-data-prep \
  --skip-training \
  --overwrite
```

The selector requires both in-sample and rollout Chamfer/MAE to stay within 5%
of the unsmoothed control and rejects additional zero-contour components. Among
eligible candidates, it selects the lowest mean curvature total variation.
Combined results and the selection decision are saved in:

```text
artifacts/rate_velocity_smoothing_ablation/
|-- metrics.csv
|-- selection.json
`-- selected_full_config.yaml  # created with --full-train-winner
```

Per-experiment in-sample metrics and isolated rollout snapshots are stored in
`artifacts/ablation_rate_velocity_{control,normal,smooth}/`.

## CMA-ES Time Optimization

Optimize the eight rollout durations from `1M` through `4E` while keeping the
trained deposition and etch models fixed. The objective is:

```text
zero_chamfer(predicted 4M, GT 5M)
+ zero_chamfer(predicted 4E, GT 5E)
```

Install the project first so that the `cma` dependency is available, then run:

```bash
python -m pip install -e .

python scripts/optimize_1m_to_4e_times_cmaes.py \
  --config configs/ablation_full_physics.yaml \
  --workbook data/raw/deposition.xlsx \
  --infer-missing-rates
```

Resume an interrupted or extended search from the saved CMA-ES state:

```bash
python scripts/optimize_1m_to_4e_times_cmaes.py \
  --config configs/ablation_full_physics.yaml \
  --workbook data/raw/deposition.xlsx \
  --infer-missing-rates \
  --resume
```

The optimizer uses the training-time duration references instead of
normalizing each candidate duration by itself. With the full-physics config,
deposition uses `duration / 9000` and etch uses `duration / 50`. Rates are
inferred once and kept fixed throughout the search.

Results are written below the configured artifact directory:

```text
time_optimization/
|-- optimization_history.csv
|-- best_result.json
|-- cmaes_state.pkl
`-- best/
    |-- 1M.npy ... 4E.npy
    |-- predictions_1m_to_4e.xlsx
    |-- comparison_metrics.json
    |-- manifest.json
    `-- 4M_4E_on_5M_5E.png
```

The `5M` and `5E` sheets are calibration targets for this optimization and
must not be treated as unbiased holdout data afterward.

## Visualization and Train-Range Replay

Plot rollout predictions `3M` through `5E` after `run_rollout.py` has created
`artifacts/predictions/*.npy`:

```bash
python scripts/plot_rollout_contours.py --config configs/default.yaml
```

This saves:

```text
artifacts/figures/rollout_zero_contours_3M_to_5E.png
```

Predict training-range states from `init` through `2E`, save `1M.npy`, `1E.npy`,
`2M.npy`, `2E.npy`, and plot predicted/GT zero contours:

```bash
python scripts/predict_train_range.py --config configs/default.yaml --infer-missing-rates
```

This saves:

```text
artifacts/train_range_predictions/
artifacts/figures/train_range_zero_contours_1M_to_2E.png
```

If you want to run without trained checkpoints, add `--allow-baseline-fallback`.
That uses the known-average-rate baseline instead of the PINN checkpoints.

Plot training loss curves and weighted contribution ratios after training:

```bash
python scripts/plot_training_losses.py --config configs/default.yaml
```

This saves a three-column figure with raw losses, weighted losses, and weighted
loss share:

```text
artifacts/figures/training_loss_breakdown.png
```

## Tests

Run the CMA-ES optimizer unit and resume tests with:

```bash
python -m unittest tests/test_cmaes_time_optimization.py
```
