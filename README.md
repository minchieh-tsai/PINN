# EPI Level-Set PINN POC

This repository is generated from `PINN_EPI_CODEX_SPEC_H350_W200_POC.md`.
It provides a proof-of-concept Python package for reading EPI level-set
workbooks, preprocessing signed-distance fields, training separate deposition
and etch PINN models, rolling out from `2E` to `5E`, and evaluating holdout
states `5M` and `5E`.

## Requirements

- Python 3.11+
- XLSX files at `data/raw/deposition.xlsx` and `data/raw/etch.xlsx`
- Average interface rates in `configs/default.yaml`

Install dependencies:

```bash
python -m pip install -e .
```

## Workflow

```bash
python scripts/inspect_xlsx.py --config configs/default.yaml
python scripts/preprocess_data.py --config configs/default.yaml --split train
python scripts/train_deposition.py --config configs/default.yaml
python scripts/train_etch.py --config configs/default.yaml
python scripts/run_rollout.py --config configs/default.yaml
python scripts/evaluate_holdout.py --config configs/default.yaml

--infer-missing-ratesc will use average rate
```

The generated code intentionally does not create `tests/`, pytest files, or CI
configuration because the source spec excludes those artifacts.

## Important Configuration Notes

`average_rate_default`, `average_rate_by_cycle`, and `rate_reference` are `null`
in the generated config. Fill them before training, or pass
`--infer-missing-rates` to the training scripts to estimate process rates from
the configured training state pairs.

All arrays are handled as `(H, W) = (350, 200)` with `phi[y, x]` indexing.
