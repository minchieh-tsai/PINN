#!/usr/bin/env python
"""Optimize eight init-to-4E process times with sequential CMA-ES."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


STEP_LABELS = ("1M", "1E", "2M", "2E", "3M", "3E", "4M", "4E")
HISTORY_FIELDS = (
    "evaluation", "generation", "candidate", "status", "error", "elapsed_s",
    *(f"multiplier_{state}" for state in STEP_LABELS),
    *(f"time_{state}_s" for state in STEP_LABELS),
    "chamfer_4m_vs_5m", "chamfer_4e_vs_5e", "chamfer_score",
    "merge_penalty", "down_penalty", "chamfer_weight", "merge_weight",
    "down_weight", "weighted_chamfer", "weighted_merge", "weighted_down",
    "objective",
)


@dataclass
class CandidateEvaluation:
    multipliers: Tuple[float, ...]
    times_s: Tuple[float, ...]
    score: Any
    objective: float
    status: str
    error: str
    elapsed_s: float


def add_src_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    return root


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        (json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HISTORY_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_history(path: Path) -> list[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def times_from_multipliers(
    base_times: Sequence[float], multipliers: Sequence[float], lower: float, upper: float
) -> Tuple[float, ...]:
    if len(base_times) != len(STEP_LABELS) or len(multipliers) != len(STEP_LABELS):
        raise ValueError("exactly eight base times and multipliers are required")
    values = []
    for base, multiplier in zip(base_times, multipliers):
        base_value = float(base)
        scale = float(multiplier)
        if not math.isfinite(base_value) or base_value <= 0.0:
            raise ValueError("base times must be positive and finite")
        if not math.isfinite(scale) or scale < lower - 1e-12 or scale > upper + 1e-12:
            raise ValueError("candidate multiplier is outside configured bounds")
        value = base_value * scale
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("candidate time must be positive and finite")
        values.append(value)
    return tuple(values)


def evaluate_candidate(
    runtime: Any,
    base_times: Sequence[float],
    multipliers: Sequence[float],
    lower: float,
    upper: float,
    prediction_sigma: float,
    weights: Any,
    invalid_objective: float,
) -> CandidateEvaluation:
    from epi_pinn.prediction_runtime import PREDICT_1M_TO_4E_STEPS, run_prediction_sequence
    from epi_pinn.time_objective import evaluate_time_objective

    started = time.perf_counter()
    scales = tuple(float(value) for value in multipliers)
    score = None
    try:
        times = times_from_multipliers(base_times, scales, lower, upper)
        result = run_prediction_sequence(
            runtime,
            runtime.states["init"],
            PREDICT_1M_TO_4E_STEPS,
            times,
            prediction_gaussian_sigma=prediction_sigma,
            start_state="init",
        )
        time_map = dict(zip(STEP_LABELS, times))
        score = evaluate_time_objective(
            result.predictions, runtime.states, time_map, runtime.config, weights
        )
        objective = float(score.total_score)
        if not math.isfinite(objective) or objective < 0.0 or objective >= invalid_objective:
            raise ValueError("candidate objective is invalid")
        status, error = "ok", ""
    except Exception as exc:
        times = tuple(float(base) * float(scale) for base, scale in zip(base_times, scales))
        objective = float(invalid_objective)
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    return CandidateEvaluation(
        multipliers=scales,
        times_s=times,
        score=score,
        objective=objective,
        status=status,
        error=error,
        elapsed_s=time.perf_counter() - started,
    )


def evaluation_row(
    evaluation: int, generation: int, candidate: str, result: CandidateEvaluation
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "evaluation": evaluation,
        "generation": generation,
        "candidate": candidate,
        "status": result.status,
        "error": result.error,
        "elapsed_s": result.elapsed_s,
    }
    row.update({f"multiplier_{state}": value for state, value in zip(STEP_LABELS, result.multipliers)})
    row.update({f"time_{state}_s": value for state, value in zip(STEP_LABELS, result.times_s)})
    if result.score is None:
        for field in HISTORY_FIELDS[6 + 2 * len(STEP_LABELS):-1]:
            row[field] = float("nan")
    else:
        score_values = result.score.to_dict()
        score_values.pop("total_score", None)
        row.update(score_values)
    row["objective"] = result.objective
    return row


def row_float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key, float("nan"))
    return float(value) if value not in (None, "") else float("nan")


def row_times(row: Mapping[str, Any]) -> Tuple[float, ...]:
    return tuple(row_float(row, f"time_{state}_s") for state in STEP_LABELS)


def row_multipliers(row: Mapping[str, Any]) -> Tuple[float, ...]:
    return tuple(row_float(row, f"multiplier_{state}") for state in STEP_LABELS)


def best_valid_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    valid = [row for row in rows if str(row.get("status")) == "ok" and math.isfinite(row_float(row, "objective"))]
    if not valid:
        raise RuntimeError("no valid CMA-ES candidate was produced")
    return min(valid, key=lambda row: row_float(row, "objective"))


def workbook_identity(runtime: Any) -> Dict[str, Dict[str, Optional[str]]]:
    if runtime.workbook_path is not None:
        paths = {"override": Path(runtime.workbook_path).resolve()}
    else:
        paths = {}
        for key, value in runtime.config.get("data", {}).get("workbooks", {}).items():
            path = Path(str(value))
            if not path.is_absolute():
                path = runtime.config_root / path
            paths[str(key)] = path.resolve()
    return {
        key: {"path": str(path), "sha256": sha256_file(path)}
        for key, path in sorted(paths.items())
    }

def build_identity(runtime: Any, base_times: Sequence[float], args: argparse.Namespace) -> Dict[str, Any]:
    import epi_pinn.time_objective as objective_module
    from epi_pinn.prediction_runtime import normalization_metadata

    return {
        "base_commit": "2ab4856c44d4fe1ce7c189f0701cc190bd1b5397",
        "config": str(runtime.config_path),
        "config_sha256": sha256_file(runtime.config_path),
        "workbooks": workbook_identity(runtime),
        "checkpoints": {
            process: {"path": str(path), "sha256": sha256_file(path)}
            for process, path in runtime.checkpoint_paths.items()
        },
        "penalty_source_sha256": sha256_file(Path(objective_module.__file__).resolve()),
        "normalization": normalization_metadata(runtime),
        "runtime_rates": {
            "inferred": dict(runtime.inferred_rates),
            "duration_references_s": dict(runtime.duration_references),
            "rate_references": dict(runtime.rate_references),
        },
        "execution_flags": {
            "infer_missing_rates": bool(args.infer_missing_rates),
            "allow_baseline_fallback": bool(args.allow_baseline_fallback),
        },
        "base_times_s": list(base_times),
        "time_scale_bounds": list(args.time_scale_bounds),
        "prediction_gaussian_sigma": args.prediction_gaussian_sigma,
        "invalid_objective": args.invalid_objective,
        "weights": {
            "chamfer": args.chamfer_weight,
            "merge": args.merge_weight,
            "down": args.down_weight,
        },
        "cma_sigma": args.cma_sigma,
        "cma_popsize": args.cma_popsize,
        "cma_seed": args.cma_seed,
    }


def save_strategy(path: Path, strategy: Any) -> None:
    payload = strategy.pickle_dumps() if hasattr(strategy, "pickle_dumps") else pickle.dumps(strategy)
    atomic_write(path, payload)


def load_strategy(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing CMA-ES state: {path}")
    return pickle.loads(path.read_bytes())


def current_best_payload(row: Mapping[str, Any]) -> Dict[str, Any]:
    score_keys = (
        "chamfer_4m_vs_5m", "chamfer_4e_vs_5e", "chamfer_score",
        "merge_penalty", "down_penalty", "weighted_chamfer", "weighted_merge",
        "weighted_down", "objective",
    )
    return {
        "times_s": dict(zip(STEP_LABELS, row_times(row))),
        "multipliers": dict(zip(STEP_LABELS, row_multipliers(row))),
        "score": {key: row_float(row, key) for key in score_keys},
    }


def write_best_artifacts(
    output_dir: Path,
    runtime: Any,
    result: Any,
    times: Sequence[float],
    score: Any,
    identity: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    stop_reasons: Mapping[str, Any],
) -> None:
    import numpy as np
    from epi_pinn.evaluate import evaluate_pair
    from epi_pinn.excel_io import write_prediction_workbook
    from epi_pinn.prediction_runtime import normalization_metadata
    from epi_pinn.time_objective import PENALTY_PLACEHOLDERS_ACTIVE

    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    for state, array in result.predictions.items():
        np.save(best_dir / f"{state}.npy", array)
    write_prediction_workbook(result.predictions, str(best_dir / "predictions_1m_to_4e.xlsx"))
    comparisons = {
        "4M_vs_5M": evaluate_pair(result.predictions["4M"], runtime.states["5M"], runtime.config),
        "4E_vs_5E": evaluate_pair(result.predictions["4E"], runtime.states["5E"], runtime.config),
    }
    write_json(best_dir / "comparison_metrics.json", comparisons)
    payload = {
        "base_commit": identity["base_commit"],
        "normalization": normalization_metadata(runtime),
        "time_order": list(STEP_LABELS),
        "best_times_s": dict(zip(STEP_LABELS, times)),
        "score": score.to_dict(),
        "weights": {
            "chamfer": args.chamfer_weight,
            "merge": args.merge_weight,
            "down": args.down_weight,
        },
        "prediction_gaussian_sigma": args.prediction_gaussian_sigma,
        "invalid_objective": args.invalid_objective,
        "penalty_placeholders_active": PENALTY_PLACEHOLDERS_ACTIVE,
        "evaluations": len(rows),
        "stop_reasons": dict(stop_reasons),
        "calibration_warning": "5M and 5E are optimization targets, not unbiased holdout data.",
    }
    write_json(output_dir / "best_result.json", payload)
    write_json(
        output_dir / "run_manifest.json",
        {
            "identity": identity,
            "best_artifact_dir": str(best_dir),
            "steps": [
                {
                    "process": step.process_name,
                    "cycle": step.cycle,
                    "state": step.output_state,
                    "input_state": step.input_state,
                    "duration_s": step.duration_s,
                    "average_rate": step.average_rate,
                }
                for step in result.steps
            ],
        },
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize 1M-to-4E process times with CMA-ES.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--workbook", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--infer-missing-rates", action="store_true")
    parser.add_argument("--allow-baseline-fallback", action="store_true")
    parser.add_argument(
        "--normalization-mode", choices=("legacy", "training-reference"), default="legacy"
    )
    parser.add_argument("--prediction-gaussian-sigma", type=float, default=0.0)
    parser.add_argument("--times", nargs=8, type=float, metavar="SECONDS")
    parser.add_argument("--time-scale-bounds", nargs=2, type=float, default=(0.5, 1.5), metavar=("LOWER", "UPPER"))
    parser.add_argument("--cma-sigma", type=float, default=0.2)
    parser.add_argument("--cma-generations", type=int, default=30)
    parser.add_argument("--cma-popsize", type=int, default=10)
    parser.add_argument("--cma-seed", type=int, default=42)
    parser.add_argument("--chamfer-weight", type=float, default=1.0)
    parser.add_argument("--merge-weight", type=float, default=1.0)
    parser.add_argument("--down-weight", type=float, default=1.0)
    parser.add_argument("--invalid-objective", type=float, default=1.0e9)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    lower, upper = args.time_scale_bounds
    if not 0.0 < lower < upper:
        parser.error("--time-scale-bounds must satisfy 0 < LOWER < UPPER")
    if args.cma_sigma <= 0.0 or args.cma_generations <= 0 or args.cma_popsize < 2:
        parser.error("CMA sigma/generations must be positive and popsize must be at least 2")
    if not math.isfinite(args.prediction_gaussian_sigma) or args.prediction_gaussian_sigma < 0.0:
        parser.error("--prediction-gaussian-sigma must be finite and nonnegative")
    if not math.isfinite(args.invalid_objective) or args.invalid_objective <= 0.0:
        parser.error("--invalid-objective must be positive and finite")
    if args.times is not None and any(not math.isfinite(value) or value <= 0.0 for value in args.times):
        parser.error("all --times values must be positive and finite")
    for name in ("chamfer_weight", "merge_weight", "down_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = add_src_path()
    try:
        import cma
    except ImportError as exc:
        raise SystemExit("Missing dependency 'cma'; run: python -m pip install -e .") from exc

    from epi_pinn.config import load_config, output_dir, project_root_from_config_path, schedule_seconds
    from epi_pinn.prediction_runtime import (
        PREDICT_1M_TO_4E_STEPS,
        load_prediction_runtime,
        run_prediction_sequence,
    )
    from epi_pinn.time_objective import (
        PENALTY_PLACEHOLDERS_ACTIVE,
        ScoreWeights,
        evaluate_time_objective,
    )

    if PENALTY_PLACEHOLDERS_ACTIVE:
        print("WARNING: merge_penalty and down_penalty are placeholders returning 0.0.", flush=True)
    runtime = load_prediction_runtime(
        args.config,
        workbook_path=args.workbook,
        checkpoint_dir=args.checkpoint_dir,
        infer_missing_rates=args.infer_missing_rates,
        allow_baseline_fallback=args.allow_baseline_fallback,
        normalization_mode=args.normalization_mode,
    )
    config = load_config(args.config)
    config_root = project_root_from_config_path(args.config)
    base_times = tuple(args.times) if args.times is not None else tuple(
        schedule_seconds(config, process, cycle)
        for process, cycle, _state in PREDICT_1M_TO_4E_STEPS
    )
    base_artifacts = output_dir(config, config_root)
    optimization_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir else base_artifacts / "time_optimization" / args.normalization_mode
    )
    optimization_dir.mkdir(parents=True, exist_ok=True)
    history_path = optimization_dir / "optimization_history.csv"
    state_path = optimization_dir / "cmaes_state.pkl"
    identity_path = optimization_dir / "run_identity.json"
    current_best_path = optimization_dir / "current_best.json"
    weights = ScoreWeights(args.chamfer_weight, args.merge_weight, args.down_weight).validated()
    identity = build_identity(runtime, base_times, args)

    if args.resume:
        if not identity_path.exists():
            raise FileNotFoundError(f"Missing resume identity: {identity_path}")
        if json.loads(identity_path.read_text(encoding="utf-8")) != _json_safe(identity):
            raise ValueError("Resume inputs differ from the saved run identity")
        strategy = load_strategy(state_path)
        strategy.opts.set({"maxiter": int(args.cma_generations)})
        strategy.stop().clear()
        rows: list[Mapping[str, Any]] = read_history(history_path)
    else:
        if any(path.exists() for path in (history_path, state_path, identity_path)):
            raise FileExistsError(f"Optimization already exists: {optimization_dir}; use --resume")
        strategy = cma.CMAEvolutionStrategy(
            [1.0] * len(STEP_LABELS),
            args.cma_sigma,
            {
                "bounds": [[args.time_scale_bounds[0]] * 8, [args.time_scale_bounds[1]] * 8],
                "popsize": args.cma_popsize,
                "seed": args.cma_seed,
                "maxiter": args.cma_generations,
                "verb_log": 0,
            },
        )
        baseline = evaluate_candidate(
            runtime, base_times, [1.0] * 8, *args.time_scale_bounds,
            args.prediction_gaussian_sigma, weights, args.invalid_objective,
        )
        if baseline.status != "ok":
            raise RuntimeError(f"Baseline candidate failed: {baseline.error}")
        rows = [evaluation_row(0, 0, "baseline", baseline)]
        write_history(history_path, rows)
        write_json(identity_path, identity)
        save_strategy(state_path, strategy)
        write_json(current_best_path, current_best_payload(rows[0]))

    evaluation_index = max((int(row["evaluation"]) for row in rows), default=0)
    cache: Dict[Tuple[float, ...], CandidateEvaluation] = {}
    while strategy.countiter < args.cma_generations and not strategy.stop():
        generation = int(strategy.countiter) + 1
        solutions = strategy.ask()
        results = []
        generation_rows = []
        for candidate_index, solution in enumerate(solutions):
            key = tuple(round(float(value), 12) for value in solution)
            result = cache.get(key)
            if result is None:
                result = evaluate_candidate(
                    runtime, base_times, solution, *args.time_scale_bounds,
                    args.prediction_gaussian_sigma, weights, args.invalid_objective,
                )
                cache[key] = result
            evaluation_index += 1
            results.append(result)
            generation_rows.append(
                evaluation_row(evaluation_index, generation, str(candidate_index), result)
            )
            print(
                f"generation={generation}/{args.cma_generations} "
                f"candidate={candidate_index + 1}/{len(solutions)} "
                f"objective={result.objective:.6g} status={result.status}",
                flush=True,
            )
        strategy.tell(solutions, [result.objective for result in results])
        rows.extend(generation_rows)
        write_history(history_path, rows)
        save_strategy(state_path, strategy)
        write_json(current_best_path, current_best_payload(best_valid_row(rows)))

    best_row = best_valid_row(rows)
    best_times = row_times(best_row)
    final_result = run_prediction_sequence(
        runtime, runtime.states["init"], PREDICT_1M_TO_4E_STEPS, best_times,
        prediction_gaussian_sigma=args.prediction_gaussian_sigma, start_state="init",
    )
    final_score = evaluate_time_objective(
        final_result.predictions,
        runtime.states,
        dict(zip(STEP_LABELS, best_times)),
        runtime.config,
        weights,
    )
    if not math.isclose(final_score.total_score, row_float(best_row, "objective"), rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError("Recomputed best objective does not match optimization history")
    write_best_artifacts(
        optimization_dir, runtime, final_result, best_times, final_score,
        identity, rows, args, strategy.stop(),
    )
    print(f"Best objective: {final_score.total_score:.8g}")
    print(f"Saved optimization: {optimization_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())