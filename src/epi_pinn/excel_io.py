"""Excel input/output helpers for level-set state arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd


def _as_path(path: str, base_dir: Optional[Path] = None) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return candidate


def read_excel_array(
    workbook_path: str,
    sheet_name: str,
    expected_shape: tuple,
    allow_transpose: bool = True,
) -> np.ndarray:
    path = Path(workbook_path)
    if not path.exists():
        raise FileNotFoundError(f"Workbook does not exist: {path}")

    try:
        frame = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    except ValueError as exc:
        raise ValueError(f"Sheet {sheet_name!r} not found in workbook {path}") from exc

    array = frame.to_numpy(dtype=np.float64, copy=True)
    if array.shape == expected_shape:
        result = array
    elif allow_transpose and array.T.shape == expected_shape:
        result = array.T.copy()
    else:
        raise ValueError(
            f"Unexpected shape for {path} sheet {sheet_name!r}: {array.shape}; "
            f"expected {expected_shape}"
        )

    if not np.issubdtype(result.dtype, np.number):
        raise ValueError(f"Sheet {sheet_name!r} in {path} did not produce numeric values")
    if np.isnan(result).any():
        raise ValueError(f"Sheet {sheet_name!r} in {path} contains NaN values")
    if np.isinf(result).any():
        raise ValueError(f"Sheet {sheet_name!r} in {path} contains Inf values")
    return np.ascontiguousarray(result, dtype=np.float64)


def workbook_sheet_names(workbook_path: str) -> list:
    path = Path(workbook_path)
    if not path.exists():
        raise FileNotFoundError(f"Workbook does not exist: {path}")
    return list(pd.ExcelFile(path, engine="openpyxl").sheet_names)


def load_state_arrays(config: Mapping[str, Any], base_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
    data = config.get("data", {})
    expected_shape = (
        int(data.get("expected_height", 350)),
        int(data.get("expected_width", 200)),
    )
    allow_transpose = bool(data.get("allow_transpose", True))
    workbooks = data.get("workbooks", {})
    state_sources = data.get("state_sources", {})
    states: Dict[str, np.ndarray] = {}

    for state_name, source in state_sources.items():
        workbook_key = source["workbook"]
        workbook_path = _as_path(workbooks[workbook_key], base_dir)
        states[state_name] = read_excel_array(
            str(workbook_path),
            str(source["sheet"]),
            expected_shape=expected_shape,
            allow_transpose=allow_transpose,
        )
    return states


def inspect_configured_workbooks(config: Mapping[str, Any], base_dir: Optional[Path] = None) -> Dict[str, Any]:
    data = config.get("data", {})
    expected_shape = (
        int(data.get("expected_height", 350)),
        int(data.get("expected_width", 200)),
    )
    allow_transpose = bool(data.get("allow_transpose", True))
    workbooks = data.get("workbooks", {})
    state_sources = data.get("state_sources", {})

    report: Dict[str, Any] = {"workbooks": {}, "states": {}}
    for key, path_text in workbooks.items():
        path = _as_path(path_text, base_dir)
        workbook_info: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.exists():
            workbook_info["sheets"] = workbook_sheet_names(str(path))
        report["workbooks"][key] = workbook_info

    for state_name, source in state_sources.items():
        workbook_path = _as_path(workbooks[source["workbook"]], base_dir)
        state_info: Dict[str, Any] = {
            "workbook": source["workbook"],
            "path": str(workbook_path),
            "sheet": str(source["sheet"]),
        }
        try:
            array = read_excel_array(
                str(workbook_path),
                str(source["sheet"]),
                expected_shape=expected_shape,
                allow_transpose=allow_transpose,
            )
            state_info.update(
                {
                    "shape": list(array.shape),
                    "min": float(np.min(array)),
                    "max": float(np.max(array)),
                    "finite": bool(np.isfinite(array).all()),
                    "zero_crossing_columns": int(count_zero_crossing_columns(array)),
                }
            )
        except Exception as exc:
            state_info["error"] = str(exc)
        report["states"][state_name] = state_info
    return report


def count_zero_crossing_columns(phi: np.ndarray) -> int:
    count = 0
    for x_index in range(phi.shape[1]):
        column = phi[:, x_index]
        if np.any(column == 0) or np.any(column[:-1] * column[1:] < 0):
            count += 1
    return count


def write_prediction_workbook(predictions: Mapping[str, np.ndarray], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, array in predictions.items():
            pd.DataFrame(np.asarray(array)).to_excel(
                writer,
                sheet_name=str(name),
                header=False,
                index=False,
            )
