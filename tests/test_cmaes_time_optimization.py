from __future__ import annotations

import importlib.util
import json
import math
import os
import pickle
from pathlib import Path
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "optimize_1m_to_4e_times_cmaes.py"
SPEC = importlib.util.spec_from_file_location("cmaes_time_optimizer", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import optimizer script: {SCRIPT_PATH}")
optimizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = optimizer
SPEC.loader.exec_module(optimizer)


class TimeOptimizationPureTests(unittest.TestCase):
    def test_multipliers_convert_all_eight_schedule_times(self):
        base = [9000.0, 50.0, 8000.0, 50.0, 7000.0, 50.0, 6000.0, 50.0]
        multipliers = [0.5, 1.5, 1.0, 0.75, 1.25, 1.0, 1.5, 0.5]

        actual = optimizer.times_from_multipliers(base, multipliers, 0.5, 1.5)

        expected = (4500.0, 75.0, 8000.0, 37.5, 8750.0, 50.0, 9000.0, 25.0)
        self.assertEqual(actual, expected)

    def test_multiplier_outside_bounds_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "within"):
            optimizer.times_from_multipliers([1.0] * 8, [1.0] * 7 + [1.51], 0.5, 1.5)

    def test_duration_and_rate_references_are_fixed_by_process(self):
        config = {
            "processes": {
                "deposition": {"duration_reference_s": 9000.0, "rate_reference": None},
                "etch": {"duration_reference_s": 50.0, "rate_reference": 0.25},
            }
        }

        self.assertEqual(optimizer.duration_reference_for(config, "deposition"), 9000.0)
        self.assertEqual(optimizer.duration_reference_for(config, "etch"), 50.0)
        self.assertEqual(
            optimizer.fixed_rate_reference_for(config, "deposition", [0.1, 0.3, 0.2, 0.4]),
            0.25,
        )
        self.assertEqual(
            optimizer.fixed_rate_reference_for(config, "etch", [0.1, 0.2]),
            0.25,
        )

    def test_feature_builder_receives_training_references_not_candidate_values(self):
        captured = {}

        def fake_build_features(*args):
            captured["duration"] = args[5]
            captured["rate"] = args[6]
            captured["duration_reference"] = args[7]
            captured["rate_reference"] = args[8]
            return "features", "phi0"

        result = optimizer.build_features_with_fixed_references(
            fake_build_features,
            "phi",
            "contour",
            "xi",
            "eta",
            "tau",
            6000.0,
            0.2,
            9000.0,
            0.25,
            32.0,
            1.0,
        )

        self.assertEqual(result, ("features", "phi0"))
        self.assertEqual(captured["duration"], 6000.0)
        self.assertEqual(captured["duration_reference"], 9000.0)
        self.assertAlmostEqual(captured["duration"] / captured["duration_reference"], 2.0 / 3.0)
        self.assertEqual(captured["rate"], 0.2)
        self.assertEqual(captured["rate_reference"], 0.25)

    def test_objective_is_exact_sum_and_invalid_values_use_penalty(self):
        self.assertEqual(optimizer.combined_objective(12.5, 7.25, 1.0e9), 19.75)
        self.assertEqual(optimizer.combined_objective(math.nan, 7.25, 1.0e9), 1.0e9)

    def test_json_writer_converts_nonfinite_metrics_to_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            optimizer.write_json(path, {"valid": 1.5, "invalid": math.nan})
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["valid"], 1.5)
        self.assertIsNone(payload["invalid"])

    def test_duplicate_configured_targets_are_rejected_without_workbook_override(self):
        config = {
            "data": {
                "state_sources": {
                    "init": {"workbook": "all", "sheet": "init"},
                    "5M": {"workbook": "all", "sheet": "5"},
                    "5E": {"workbook": "all", "sheet": "5"},
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "same configured workbook/sheet"):
            optimizer.validate_target_sources(config, explicit_workbook=False)
        optimizer.validate_target_sources(config, explicit_workbook=True)

    def test_failed_candidate_returns_penalty_without_raising(self):
        def failing_rollout(_times):
            raise RuntimeError("synthetic inference failure")

        result = optimizer.evaluate_candidate(
            [1.0] * 8,
            [1.0] * 8,
            0.5,
            1.5,
            12345.0,
            failing_rollout,
            lambda _predictions: (1.0, 2.0),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.objective, 12345.0)
        self.assertIn("synthetic inference failure", result.error)

    def test_cma_state_helper_round_trips_pickle_bytes(self):
        class FakeStrategy:
            def pickle_dumps(self):
                return pickle.dumps({"countiter": 3, "seed": 42})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cmaes_state.pkl"
            optimizer.save_cma_state(path, FakeStrategy())
            restored = optimizer.load_cma_state(path)

        self.assertEqual(restored, {"countiter": 3, "seed": 42})


@unittest.skipUnless(importlib.util.find_spec("cma") is not None, "cma is not installed")
class CmaLibraryTests(unittest.TestCase):
    def test_seed_reproduces_first_population_and_state_resume(self):
        import cma

        options = {
            "bounds": [[0.5] * 8, [1.5] * 8],
            "popsize": 2,
            "seed": 42,
            "verbose": -9,
        }
        first = cma.CMAEvolutionStrategy([1.0] * 8, 0.2, options)
        first_population = first.ask()
        first.tell(
            first_population,
            [sum((x - 1.0) ** 2 for x in item) for item in first_population],
        )

        second = cma.CMAEvolutionStrategy([1.0] * 8, 0.2, options)
        second_population = second.ask()
        for left, right in zip(first_population, second_population):
            for left_value, right_value in zip(left, right):
                self.assertAlmostEqual(float(left_value), float(right_value), places=12)

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.pkl"
            optimizer.save_cma_state(state_path, first)
            resumed = optimizer.load_cma_state(state_path)
        self.assertEqual(resumed.countiter, 1)
        self.assertEqual(resumed.countevals, 2)

    def test_one_generation_main_loop_writes_history_and_state(self):
        fake_config_module = types.ModuleType("epi_pinn.config")
        fake_config_module.load_config = lambda _path: {}
        fake_config_module.project_root_from_config_path = lambda _path: ROOT
        fake_config_module.output_dir = lambda _config, root: root / "artifacts"
        schedule = {
            ("deposition", 1): 9000.0,
            ("etch", 1): 50.0,
            ("deposition", 2): 8000.0,
            ("etch", 2): 50.0,
            ("deposition", 3): 7000.0,
            ("etch", 3): 50.0,
            ("deposition", 4): 6000.0,
            ("etch", 4): 50.0,
        }
        fake_config_module.schedule_seconds = (
            lambda _config, process, cycle: schedule[(process, cycle)]
        )

        fake_runtime = types.SimpleNamespace(
            config={},
            config_path=ROOT / "synthetic-config.yaml",
            workbook_path=None,
            checkpoint_paths={},
            duration_references={"deposition": 9000.0, "etch": 50.0},
            rate_references={"deposition": 1.0, "etch": 1.0},
            rates=(1.0,) * 8,
        )
        originals = {
            "prepare_runtime": optimizer.prepare_runtime,
            "rollout_times": optimizer.rollout_times,
            "score_predictions": optimizer.score_predictions,
            "save_best_artifacts": optimizer.save_best_artifacts,
        }
        previous_config_module = sys.modules.get("epi_pinn.config")
        captured = {}

        def fake_score(_runtime, times):
            return (
                abs(sum(times[0::2]) - 30000.0) / 1000.0 + 1.0,
                abs(sum(times[1::2]) - 200.0) / 10.0 + 2.0,
            )

        try:
            sys.modules["epi_pinn.config"] = fake_config_module
            optimizer.prepare_runtime = lambda _args: fake_runtime
            optimizer.rollout_times = lambda _runtime, times: tuple(times)
            optimizer.score_predictions = fake_score

            def fake_save(_runtime, predictions, best_row, _baseline, output, *_args):
                captured["predictions"] = predictions
                captured["objective"] = float(best_row["objective"])
                (output / "best").mkdir(parents=True, exist_ok=True)
                (output / "best" / "probe.json").write_text(
                    json.dumps({"objective": captured["objective"]}),
                    encoding="utf-8",
                )

            optimizer.save_best_artifacts = fake_save
            with tempfile.TemporaryDirectory() as directory:
                exit_code = optimizer.main(
                    [
                        "--config",
                        "synthetic-config.yaml",
                        "--cma-generations",
                        "1",
                        "--cma-popsize",
                        "2",
                        "--output-dir",
                        directory,
                        "--no-plot",
                    ]
                )
                history = optimizer.read_history(
                    Path(directory) / "optimization_history.csv"
                )
                self.assertEqual(exit_code, 0)
                self.assertEqual(len(history), 3)
                self.assertTrue((Path(directory) / "cmaes_state.pkl").exists())
                self.assertTrue((Path(directory) / "best" / "probe.json").exists())

                resumed_exit_code = optimizer.main(
                    [
                        "--config",
                        "synthetic-config.yaml",
                        "--cma-generations",
                        "2",
                        "--cma-popsize",
                        "2",
                        "--output-dir",
                        directory,
                        "--no-plot",
                        "--resume",
                    ]
                )
                resumed_history = optimizer.read_history(
                    Path(directory) / "optimization_history.csv"
                )
                resumed_state = optimizer.load_cma_state(
                    Path(directory) / "cmaes_state.pkl"
                )
                self.assertEqual(resumed_exit_code, 0)
                self.assertEqual(len(resumed_history), 5)
                self.assertEqual(resumed_state.countiter, 2)
        finally:
            for name, value in originals.items():
                setattr(optimizer, name, value)
            if previous_config_module is None:
                sys.modules.pop("epi_pinn.config", None)
            else:
                sys.modules["epi_pinn.config"] = previous_config_module

        score_4m, score_4e = fake_score(fake_runtime, captured["predictions"])
        self.assertAlmostEqual(
            captured["objective"],
            optimizer.combined_objective(score_4m, score_4e, 1.0e9),
        )


INTEGRATION_CONFIG = os.environ.get("EPI_PINN_INTEGRATION_CONFIG")
INTEGRATION_WORKBOOK = os.environ.get("EPI_PINN_INTEGRATION_WORKBOOK")


@unittest.skipUnless(
    INTEGRATION_CONFIG and INTEGRATION_WORKBOOK,
    "set EPI_PINN_INTEGRATION_CONFIG and EPI_PINN_INTEGRATION_WORKBOOK",
)
class CmaOptimizerIntegrationTests(unittest.TestCase):
    def test_one_generation_artifacts_reproduce_best_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            exit_code = optimizer.main(
                [
                    "--config",
                    str(INTEGRATION_CONFIG),
                    "--workbook",
                    str(INTEGRATION_WORKBOOK),
                    "--infer-missing-rates",
                    "--cma-generations",
                    "1",
                    "--cma-popsize",
                    "2",
                    "--output-dir",
                    directory,
                    "--no-plot",
                ]
            )
            self.assertEqual(exit_code, 0)

            import numpy as np

            src = str(ROOT / "src")
            if src not in sys.path:
                sys.path.insert(0, src)
            from epi_pinn.config import load_config
            from epi_pinn.evaluate import evaluate_pair
            from epi_pinn.excel_io import read_excel_array
            from epi_pinn.sdf import ensure_signed_distance

            config = load_config(str(INTEGRATION_CONFIG))
            data_config = config.get("data", {})
            shape = (
                int(data_config.get("expected_height", 350)),
                int(data_config.get("expected_width", 200)),
            )
            level_config = config.get("level_set", {})
            pred_4m = np.load(Path(directory) / "best" / "4M.npy")
            pred_4e = np.load(Path(directory) / "best" / "4E.npy")
            gt_5m = ensure_signed_distance(
                read_excel_array(str(INTEGRATION_WORKBOOK), "5M", shape), level_config
            )
            gt_5e = ensure_signed_distance(
                read_excel_array(str(INTEGRATION_WORKBOOK), "5E", shape), level_config
            )
            recomputed = (
                evaluate_pair(pred_4m, gt_5m, config)["zero_contour_symmetric_chamfer_px"]
                + evaluate_pair(pred_4e, gt_5e, config)["zero_contour_symmetric_chamfer_px"]
            )
            result = json.loads(
                (Path(directory) / "best_result.json").read_text(encoding="utf-8")
            )
            self.assertAlmostEqual(recomputed, result["best"]["objective"], places=10)


if __name__ == "__main__":
    unittest.main()
