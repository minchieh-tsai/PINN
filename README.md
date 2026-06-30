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
