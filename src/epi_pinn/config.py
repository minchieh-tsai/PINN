"""Configuration loading and convenience accessors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


Config = Dict[str, Any]


def load_config(path: str) -> Config:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return config


def project_root_from_config_path(path: str) -> Path:
    config_path = Path(path).resolve()
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return Path.cwd()


def save_config(config: Mapping[str, Any], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)


def output_dir(config: Mapping[str, Any], base_dir: Optional[Path] = None) -> Path:
    path = Path(config.get("project", {}).get("output_dir", "artifacts"))
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def expected_shape(config: Mapping[str, Any]) -> tuple:
    data = config.get("data", {})
    return (
        int(data.get("expected_height", 350)),
        int(data.get("expected_width", 200)),
    )


def process_config(config: Mapping[str, Any], process_name: str) -> Mapping[str, Any]:
    processes = config.get("processes", {})
    if process_name not in processes:
        raise KeyError(f"Unknown process {process_name!r}; expected one of {sorted(processes)}")
    return processes[process_name]


def schedule_seconds(config: Mapping[str, Any], process_name: str, cycle: int) -> float:
    schedule = config.get("schedule", {})
    key = "deposition_seconds" if process_name == "deposition" else "etch_seconds"
    values = schedule.get(key, {})
    try:
        return float(values[int(cycle)])
    except KeyError:
        try:
            return float(values[str(cycle)])
        except KeyError as exc:
            raise KeyError(f"Missing {key} for cycle {cycle}") from exc


def average_rate(
    config: Mapping[str, Any],
    process_name: str,
    cycle: int,
    fallback: Optional[float] = None,
) -> float:
    proc = process_config(config, process_name)
    by_cycle = proc.get("average_rate_by_cycle") or {}
    value = by_cycle.get(cycle, by_cycle.get(str(cycle)))
    if value is None:
        value = proc.get("average_rate_default")
    if value is None:
        value = fallback
    if value is None:
        raise ValueError(
            f"Missing average rate for {process_name} cycle {cycle}; "
            "set average_rate_by_cycle, average_rate_default, or use --infer-missing-rates."
        )
    value = float(value)
    if value <= 0:
        raise ValueError(f"Average rate must be positive for {process_name} cycle {cycle}: {value}")
    return value


def rate_reference(config: Mapping[str, Any], process_name: str, rate: float) -> float:
    proc = process_config(config, process_name)
    value = proc.get("rate_reference")
    if value is None:
        value = proc.get("average_rate_default")
    if value is None:
        value = rate
    value = float(value)
    if value <= 0:
        raise ValueError(f"Rate reference must be positive for {process_name}: {value}")
    return value


def device_name(config: Mapping[str, Any]) -> str:
    device = str(config.get("training", {}).get("device", "auto")).lower()
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"