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

## Notes

The generated code intentionally does not create `tests/`, pytest files, or CI
configuration because the source spec excludes those artifacts.


## CMA-ES Time Optimization

The independent optimizer searches eight continuous schedule multipliers for
`1M, 1E, 2M, 2E, 3M, 3E, 4M, 4E`. Its objective is the sum of the
`4M -> GT 5M` and `4E -> GT 5E` zero-contour symmetric Chamfer distances,
plus the weighted penalty hooks in `src/epi_pinn/time_objective.py`.

Two normalization stages are available and write to separate directories:

- `legacy` (default) exactly follows the inference behavior at base commit
  `2ab4856`: `duration_normalized = duration / duration = 1` and
  `rate_normalized = rate / rate = 1`. Time still changes nominal
  displacement through `rate * duration`, but the network duration feature
  remains constant.
- `training-reference` passes each candidate duration through the configured
  `duration_reference_s`. The rate uses a fixed configured reference, or a
  rate inferred once before optimization when `rate_reference` is null. This
  exposes candidate duration changes to the network in the same normalization
  convention used during training.

Install the optimizer and test dependencies:

```bash
python -m pip install -e ".[dev]"
```

Stage 1, reproduce legacy inference:

```bash
python scripts/optimize_1m_to_4e_times_cmaes.py \
  --config configs/ablation_full_physics.yaml \
  --workbook data/raw/deposition.xlsx \
  --infer-missing-rates \
  --normalization-mode legacy
```

Resume the same run:

```bash
python scripts/optimize_1m_to_4e_times_cmaes.py \
  --config configs/ablation_full_physics.yaml \
  --workbook data/raw/deposition.xlsx \
  --infer-missing-rates \
  --normalization-mode legacy \
  --resume
```

Stage 2, use training-reference normalization:

```bash
python scripts/optimize_1m_to_4e_times_cmaes.py \
  --config configs/ablation_full_physics.yaml \
  --workbook data/raw/deposition.xlsx \
  --infer-missing-rates \
  --normalization-mode training-reference
```

The two result roots are:

```text
artifacts/ablation_full_physics/time_optimization/legacy/
artifacts/ablation_full_physics/time_optimization/training-reference/
```

Each contains `optimization_history.csv`, `current_best.json`,
`best_result.json`, `run_manifest.json`, the CMA state, and final prediction
artifacts under `best/`. Resume verifies config, workbook, checkpoints,
normalization mode, bounds, weights, and the penalty source hash.

The `merge_penalty()` and `down_penalty()` functions are deliberately marked
`TO-DO` and currently return zero. After implementing either penalty, start a
new output directory because the penalty source hash intentionally invalidates
old resume state.

The workbook `5M` and `5E` sheets are calibration targets in these runs.
They must not also be reported as unbiased holdout results.

## Cycle Visualization

Render the standard in-sample and rollout sequence. The default rollout starts
from GT `2E`; use `--rollout-start predicted-2e` to carry the predicted
training-range state forward.

```bash
python scripts/visualize_cycle_predictions.py \
  --mode standard \
  --config configs/ablation_full_physics.yaml \
  --workbook data/raw/deposition.xlsx \
  --infer-missing-rates \
  --normalization-mode legacy \
  --rollout-start gt-2e
```

Render the exact settings saved by a legacy CMA-ES result:

```bash
python scripts/visualize_cycle_predictions.py \
  --mode cmaes \
  --config configs/ablation_full_physics.yaml \
  --workbook data/raw/deposition.xlsx \
  --infer-missing-rates \
  --best-result artifacts/ablation_full_physics/time_optimization/legacy/best_result.json
```

For the second stage, point `--best-result` at the
`training-reference/best_result.json` file. The script reads and enforces the
saved normalization mode and prediction smoothing value.

Every panel uses the workbook `init` contour as the black solid initial
boundary. The blue solid line is the PINN prediction and the red dashed line is
GT when a target is available. GT pixel bands are not drawn.

Useful display controls are `--plot-gaussian-sigma`, `--contour-mode`,
`--min-contour-points`, `--border-margin`, `--panel-width`, and
`--figure-height`. The default canvas uses 4.8 inches per panel and an
8.5-inch height to keep titles and legends separated. Plot smoothing only
affects temporary display copies; it never changes CMA scores, prediction
arrays, or workbooks.

Before every inference step, the sequence runtime calls the Gaussian interface
helper. `--prediction-gaussian-sigma 0` is an identity copy with no smoothing;
use a nonzero value such as `--prediction-gaussian-sigma 0.75` to apply actual
Gaussian smoothing before every model prediction. CMA-ES visualization uses
the value stored in `best_result.json` unless the CLI explicitly overrides it.

Figures and manifests are written under:

```text
artifacts/ablation_full_physics/cycle_visualization/legacy/
artifacts/ablation_full_physics/cycle_visualization/training-reference/
```