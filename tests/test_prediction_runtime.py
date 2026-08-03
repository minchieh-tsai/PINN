from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import sys
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = types.ModuleType("epi_pinn")
PACKAGE.__path__ = [str(ROOT / "src" / "epi_pinn")]
sys.modules.setdefault("epi_pinn", PACKAGE)

TORCH = types.ModuleType("torch")
TORCH.Tensor = type("Tensor", (), {})
TORCH.float64 = object()
TORCH.dtype = object
TORCH.nn = SimpleNamespace(Module=object)
TORCH.inference_mode = contextlib.nullcontext
sys.modules.setdefault("torch", TORCH)

ROLLOUT = types.ModuleType("epi_pinn.rollout")
ROLLOUT._infer_process_rate = lambda *args, **kwargs: None
ROLLOUT._load_model = lambda *args, **kwargs: None
ROLLOUT.predict_next_levelset = lambda *args, **kwargs: np.asarray(args[0]).copy()
sys.modules.setdefault("epi_pinn.rollout", ROLLOUT)

TRAIN = types.ModuleType("epi_pinn.train")
TRAIN.torch_dtype = lambda _name: TORCH.float64
sys.modules.setdefault("epi_pinn.train", TRAIN)

import torch
from epi_pinn import prediction_runtime
from epi_pinn.prediction_runtime import PredictionRuntime, run_prediction_sequence
from epi_pinn.sdf import gaussian_smooth_interface


def runtime_for(mode: str, model=None) -> PredictionRuntime:
    config = {
        "processes": {
            "deposition": {
                "sign": 1.0,
                "average_rate_default": 2.0,
                "duration_reference_s": 100.0,
            },
            "etch": {
                "sign": -1.0,
                "average_rate_default": 3.0,
                "duration_reference_s": 10.0,
            },
        },
        "level_set": {"phi_clip_distance": 32.0},
        "contour": {"num_points": 3, "min_valid_points": 1},
        "spatial": {"pixel_size_y": 1.0},
    }
    state = np.arange(20, dtype=np.float64).reshape(4, 5) - 8.0
    return PredictionRuntime(
        config_path=ROOT / "configs" / "default.yaml",
        config=config,
        config_root=ROOT,
        workbook_path=None,
        checkpoint_dir=ROOT,
        checkpoint_paths={},
        raw_states={"init": state},
        states={"init": state},
        models={"deposition": model, "etch": model},
        inferred_rates={"deposition": None, "etch": None},
        duration_references={"deposition": 100.0, "etch": 10.0},
        rate_references={"deposition": 4.0, "etch": 6.0},
        normalization_mode=mode,
        device="cpu",
        dtype=torch.float64,
        reinitialize_sdf_each_step=False,
    )


class PredictionRuntimeTests(unittest.TestCase):
    def test_sigma_zero_returns_contiguous_float64_copy(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::-1]
        result = gaussian_smooth_interface(source, 0.0)

        self.assertEqual(result.dtype, np.float64)
        self.assertTrue(result.flags.c_contiguous)
        self.assertFalse(np.shares_memory(result, source))
        np.testing.assert_array_equal(result, source)

    def test_gaussian_sigma_validation(self):
        with self.assertRaises(ValueError):
            gaussian_smooth_interface(np.zeros((3, 3)), -0.1)
        with self.assertRaises(ValueError):
            gaussian_smooth_interface(np.zeros((3, 3)), float("nan"))

    def test_legacy_mode_calls_historical_prediction_path(self):
        runtime = runtime_for("legacy", model=object())
        start = runtime.states["init"]
        expected = start + 0.25

        with mock.patch.object(
            prediction_runtime,
            "predict_next_levelset",
            return_value=expected,
        ) as legacy_predict, mock.patch.object(
            prediction_runtime,
            "_predict_training_reference",
        ) as training_predict:
            result = run_prediction_sequence(
                runtime,
                start,
                (("deposition", 1, "1M"),),
                (25.0,),
                prediction_gaussian_sigma=0.0,
            )

        legacy_predict.assert_called_once()
        training_predict.assert_not_called()
        args = legacy_predict.call_args.args
        self.assertEqual(args[1], 25.0)
        self.assertEqual(args[2], 2.0)
        self.assertEqual(args[3], 1.0)
        np.testing.assert_array_equal(result.predictions["1M"], expected)

    def test_training_reference_mode_uses_fixed_references(self):
        captured = {}

        class DummyModel:
            def predict_numpy(self, features, contour_features, raw_phi0, *args, **kwargs):
                captured["features"] = features
                return np.asarray(raw_phi0)

        runtime = runtime_for("training-reference", model=DummyModel())
        fake_contour = SimpleNamespace(
            as_features=lambda: np.zeros((3, 3), dtype=np.float64)
        )

        def fake_build_features(
            phi, contour, xi, eta, tau, duration, rate,
            duration_reference, rate_reference, clip_distance, process_sign,
        ):
            captured["duration"] = duration
            captured["rate"] = rate
            captured["duration_reference"] = duration_reference
            captured["rate_reference"] = rate_reference
            captured["duration_normalized"] = duration / duration_reference
            captured["rate_normalized"] = rate / rate_reference
            return np.zeros((phi.size, 14), dtype=np.float64), phi.reshape(-1)

        with mock.patch(
            "epi_pinn.contour.extract_contour20",
            return_value=fake_contour,
        ), mock.patch(
            "epi_pinn.sampling.build_features",
            side_effect=fake_build_features,
        ):
            result = run_prediction_sequence(
                runtime,
                runtime.states["init"],
                (("deposition", 1, "1M"),),
                (25.0,),
                prediction_gaussian_sigma=0.0,
            )

        self.assertEqual(captured["duration_reference"], 100.0)
        self.assertEqual(captured["rate_reference"], 4.0)
        self.assertEqual(captured["duration_normalized"], 0.25)
        self.assertEqual(captured["rate_normalized"], 0.5)
        np.testing.assert_array_equal(result.predictions["1M"], runtime.states["init"])


if __name__ == "__main__":
    unittest.main()