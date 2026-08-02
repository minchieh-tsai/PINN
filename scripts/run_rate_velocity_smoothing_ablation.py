#!/usr/bin/env python
"""Run controlled rate-velocity smoothing experiments and select a winner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    "configs/ablation_rate_velocity_control.yaml",
    "configs/ablation_rate_velocity_normal.yaml",
    "configs/ablation_rate_velocity_smooth.yaml",
)
INSAMPLE_STATES = ("1M", "1E", "2M", "2E")
HOLDOUT_STATES = ("5M", "5E")
PRIMARY_METRICS = (
    "zero_contour_symmetric_chamfer_px",
    "contour20_y_mae_px",
)


def add_src_path() -> None:
    sys.path.insert(0, str(ROOT / "src"))


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return loaded


def dump_yaml(path: Path, config: Mapping[str, Any]) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)


def resolve_config(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Config does not exist: {candidate}")
    return candidate


def experiment_name(config_path: Path) -> str:
    prefix = "ablation_rate_velocity_"
    return config_path.stem[len(prefix):] if config_path.stem.startswith(prefix) else config_path.stem


def sigma_label(sigma: float) -> str:
    return str(float(sigma)).replace("-", "neg").replace(".", "p")


def run_command(command: Sequence[str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def script_command(script: str, config_path: Path, *extra: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / script), "--config", str(config_path), *extra]


@contextmanager
def temporary_config(
    source_path: Path,
    *,
    sigma: float,
    output_dir_value: str | None = None,
    adam_steps: int | None = None,
) -> Iterator[Path]:
    config = load_yaml(source_path)
    config.setdefault("rollout", {})["interface_smoothing_sigma_px"] = float(sigma)
    if output_dir_value is not None:
        config.setdefault("project", {})["output_dir"] = output_dir_value
    if adam_steps is not None:
        config.setdefault("training", {})["adam_steps"] = int(adam_steps)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix=f".{source_path.stem}_",
        dir=source_path.parent,
        delete=False,
        encoding="utf-8",
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        dump_yaml(temp_path, config)
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


def configured_output_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    value = Path(str(config.get("project", {}).get("output_dir", "artifacts")))
    return value if value.is_absolute() else ROOT / value


def evaluate_insample(config_path: Path) -> list[Dict[str, Any]]:
    import numpy as np

    add_src_path()
    from epi_pinn.evaluate import evaluate_pair
    from epi_pinn.excel_io import load_state_arrays
    from epi_pinn.sdf import ensure_signed_distance

    config = load_yaml(config_path)
    states = load_state_arrays(config, base_dir=ROOT)
    output_dir = configured_output_dir(config_path)
    prediction_dir = output_dir / "insample_predictions"
    metrics_dir = output_dir / "insample_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Dict[str, Any]] = []
    detail: Dict[str, Dict[str, float]] = {}
    for state in INSAMPLE_STATES:
        prediction_path = prediction_dir / f"{state}.npy"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing in-sample prediction: {prediction_path}")
        prediction = np.load(prediction_path)
        target = ensure_signed_distance(states[state], config.get("level_set", {}))
        metrics = evaluate_pair(prediction, target, config)
        detail[state] = metrics
        rows.append({"state": state, **metrics})

    (metrics_dir / "metrics.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    write_csv(metrics_dir / "summary.csv", rows)
    return rows


def read_holdout_metrics(output_dir: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for state in HOLDOUT_STATES:
        path = output_dir / "metrics" / f"{state}_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing holdout metrics: {path}")
        rows.append({"state": state, **json.loads(path.read_text(encoding="utf-8"))})
    return rows


def snapshot_rollout(output_dir: Path, config_path: Path, sigma: float, overwrite: bool) -> Path:
    destination = output_dir / f"rollout_sigma_{sigma_label(sigma)}"
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Snapshot already exists: {destination}; pass --overwrite")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in ("predictions", "contours", "metrics"):
        source = output_dir / name
        if source.exists():
            shutil.copytree(source, destination / name)
    report = output_dir / "report.md"
    if report.exists():
        shutil.copy2(report, destination / "report.md")
    shutil.copy2(config_path, destination / "config.yaml")
    return destination


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_metric(rows: Iterable[Mapping[str, Any]], metric: str) -> float:
    values = [float(row[metric]) for row in rows if metric in row and math.isfinite(float(row[metric]))]
    return sum(values) / len(values) if values else float("inf")


def _component_excess(rows: Iterable[Mapping[str, Any]]) -> float:
    return max(
        (
            float(row["zero_contour_component_count"])
            - float(row["target_zero_contour_component_count"])
            for row in rows
        ),
        default=float("inf"),
    )


def select_winner(rows: Sequence[Mapping[str, Any]], tolerance: float = 0.05) -> Dict[str, Any]:
    experiments = sorted({str(row["experiment"]) for row in rows})
    sigmas = sorted({float(row["sigma_px"]) for row in rows if row["scope"] == "rollout"})
    control_name = "control" if "control" in experiments else experiments[0]
    control_insample = [row for row in rows if row["experiment"] == control_name and row["scope"] == "insample"]
    control_rollout = [
        row for row in rows
        if row["experiment"] == control_name and row["scope"] == "rollout" and float(row["sigma_px"]) == 0.0
    ]
    if not control_insample or not control_rollout:
        raise ValueError("Control in-sample and sigma=0 rollout metrics are required")

    baseline = {
        "insample_chamfer": _mean_metric(control_insample, PRIMARY_METRICS[0]),
        "insample_mae": _mean_metric(control_insample, PRIMARY_METRICS[1]),
        "rollout_chamfer": _mean_metric(control_rollout, PRIMARY_METRICS[0]),
        "rollout_mae": _mean_metric(control_rollout, PRIMARY_METRICS[1]),
    }
    candidates: list[Dict[str, Any]] = []
    for experiment in experiments:
        insample = [row for row in rows if row["experiment"] == experiment and row["scope"] == "insample"]
        for sigma in sigmas:
            rollout = [
                row for row in rows
                if row["experiment"] == experiment
                and row["scope"] == "rollout"
                and float(row["sigma_px"]) == sigma
            ]
            if not insample or not rollout:
                continue
            values = {
                "experiment": experiment,
                "sigma_px": sigma,
                "insample_chamfer": _mean_metric(insample, PRIMARY_METRICS[0]),
                "insample_mae": _mean_metric(insample, PRIMARY_METRICS[1]),
                "rollout_chamfer": _mean_metric(rollout, PRIMARY_METRICS[0]),
                "rollout_mae": _mean_metric(rollout, PRIMARY_METRICS[1]),
                "curvature_total_variation": _mean_metric(
                    [*insample, *rollout], "curvature_total_variation"
                ),
                "component_excess": max(_component_excess(insample), _component_excess(rollout)),
            }
            values["passes_error_tolerance"] = all(
                values[key] <= baseline[key] * (1.0 + tolerance) for key in baseline
            )
            values["passes_component_check"] = values["component_excess"] <= 0.0
            values["eligible"] = bool(
                values["passes_error_tolerance"] and values["passes_component_check"]
            )
            candidates.append(values)

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if eligible:
        winner = min(eligible, key=lambda item: float(item["curvature_total_variation"]))
    else:
        winner = next(
            candidate for candidate in candidates
            if candidate["experiment"] == control_name and float(candidate["sigma_px"]) == 0.0
        )
    return {"tolerance": tolerance, "baseline": baseline, "candidates": candidates, "winner": winner}


def run_experiment(config_path: Path, args: argparse.Namespace) -> list[Dict[str, Any]]:
    name = experiment_name(config_path)
    infer_flag = ["--infer-missing-rates"] if args.infer_missing_rates else []
    if not args.skip_data_prep:
        run_command(script_command("inspect_xlsx.py", config_path), args.dry_run)
        run_command(script_command("preprocess_data.py", config_path, "--split", "all"), args.dry_run)
    if not args.skip_training:
        run_command(script_command("train_deposition.py", config_path, *infer_flag), args.dry_run)
        run_command(script_command("train_etch.py", config_path, *infer_flag), args.dry_run)
        run_command(script_command("plot_training_losses.py", config_path), args.dry_run)

    run_command(script_command("predict_insample.py", config_path, *infer_flag), args.dry_run)
    if args.dry_run:
        for sigma in args.sigmas:
            print(f"# would run and snapshot rollout for {name} with sigma={sigma:g}")
        return []

    rows: list[Dict[str, Any]] = [
        {"experiment": name, "scope": "insample", "sigma_px": "", **row}
        for row in evaluate_insample(config_path)
    ]
    output_dir = configured_output_dir(config_path)
    for sigma in args.sigmas:
        with temporary_config(config_path, sigma=sigma) as variant_path:
            run_command(script_command("run_rollout.py", variant_path, *infer_flag), False)
            run_command(script_command("evaluate_holdout.py", variant_path), False)
            holdout = read_holdout_metrics(output_dir)
            snapshot_rollout(output_dir, variant_path, sigma, args.overwrite)
        rows.extend(
            {"experiment": name, "scope": "rollout", "sigma_px": sigma, **row}
            for row in holdout
        )
    return rows


def run_full_training(winner: Mapping[str, Any], config_paths: Sequence[Path], args: argparse.Namespace) -> None:
    selected = next(path for path in config_paths if experiment_name(path) == winner["experiment"])
    infer_flag = ["--infer-missing-rates"] if args.infer_missing_rates else []
    output_value = "artifacts/ablation_rate_velocity_selected"
    with temporary_config(
        selected,
        sigma=float(winner["sigma_px"]),
        output_dir_value=output_value,
        adam_steps=20000,
    ) as full_config:
        record_dir = ROOT / "artifacts" / "rate_velocity_smoothing_ablation"
        record_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(full_config, record_dir / "selected_full_config.yaml")
        run_command(script_command("inspect_xlsx.py", full_config), False)
        run_command(script_command("preprocess_data.py", full_config, "--split", "all"), False)
        run_command(script_command("train_deposition.py", full_config, *infer_flag), False)
        run_command(script_command("train_etch.py", full_config, *infer_flag), False)
        run_command(script_command("plot_training_losses.py", full_config), False)
        run_command(script_command("predict_insample.py", full_config, *infer_flag), False)
        evaluate_insample(full_config)
        run_command(script_command("run_rollout.py", full_config, *infer_flag), False)
        run_command(script_command("evaluate_holdout.py", full_config), False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train three rate-velocity smoothing ablations and compare rollout sigmas."
    )
    parser.add_argument("configs", nargs="*", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--infer-missing-rates", action="store_true")
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.5])
    parser.add_argument("--skip-data-prep", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--full-train-winner", action="store_true")
    args = parser.parse_args(argv)
    if not args.sigmas or any(sigma != 0.0 and not 0.5 <= sigma <= 1.0 for sigma in args.sigmas):
        parser.error("each sigma must be 0 or between 0.5 and 1.0")
    if 0.0 not in args.sigmas:
        parser.error("--sigmas must include 0 for the control baseline")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_paths = [resolve_config(path) for path in args.configs]
    all_rows: list[Dict[str, Any]] = []
    for config_path in config_paths:
        print(f"\n=== {experiment_name(config_path)}: {config_path} ===", flush=True)
        all_rows.extend(run_experiment(config_path, args))
    if args.dry_run:
        return 0

    summary_dir = ROOT / "artifacts" / "rate_velocity_smoothing_ablation"
    write_csv(summary_dir / "metrics.csv", all_rows)
    selection = select_winner(all_rows)
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    winner = selection["winner"]
    print(
        "Selected "
        f"experiment={winner['experiment']} sigma={winner['sigma_px']} "
        f"eligible={winner['eligible']}",
        flush=True,
    )
    if args.full_train_winner:
        run_full_training(winner, config_paths, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
