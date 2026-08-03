from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock
import sys
import types
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = types.ModuleType("epi_pinn")
PACKAGE.__path__ = [str(ROOT / "src" / "epi_pinn")]
sys.modules.setdefault("epi_pinn", PACKAGE)

from epi_pinn import time_objective
from epi_pinn.time_objective import ScoreWeights, evaluate_time_objective


def load_optimizer_module():
    path = ROOT / "scripts" / "optimize_1m_to_4e_times_cmaes.py"
    spec = importlib.util.spec_from_file_location("test_optimizer_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TimeObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.predictions = {
            "4M": np.zeros((3, 3), dtype=np.float64),
            "4E": np.ones((3, 3), dtype=np.float64),
        }
        self.states = {
            "5M": np.zeros((3, 3), dtype=np.float64),
            "5E": np.ones((3, 3), dtype=np.float64),
        }

    def test_placeholder_penalties_are_zero_and_marked_todo(self):
        self.assertTrue(time_objective.PENALTY_PLACEHOLDERS_ACTIVE)
        self.assertEqual(time_objective.merge_penalty({}, {}, {}, {}), 0.0)
        self.assertEqual(time_objective.down_penalty({}, {}, {}, {}), 0.0)
        source = Path(time_objective.__file__).read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("# TO-DO:"), 2)

    def test_objective_keeps_raw_and_weighted_components(self):
        metrics = [
            {"zero_contour_symmetric_chamfer_px": 2.0},
            {"zero_contour_symmetric_chamfer_px": 3.0},
        ]
        with mock.patch.object(
            time_objective, "evaluate_pair", side_effect=metrics
        ), mock.patch.object(
            time_objective, "merge_penalty", return_value=7.0
        ), mock.patch.object(
            time_objective, "down_penalty", return_value=11.0
        ):
            score = evaluate_time_objective(
                self.predictions,
                self.states,
                {"4M": 1.0, "4E": 1.0},
                {},
                ScoreWeights(chamfer=2.0, merge=3.0, down=5.0),
            )

        self.assertEqual(score.chamfer_score, 5.0)
        self.assertEqual(score.weighted_chamfer, 10.0)
        self.assertEqual(score.weighted_merge, 21.0)
        self.assertEqual(score.weighted_down, 55.0)
        self.assertEqual(score.total_score, 86.0)

    def test_nonfinite_or_negative_penalty_is_invalid(self):
        for invalid in (-1.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), mock.patch.object(
                time_objective,
                "evaluate_pair",
                return_value={"zero_contour_symmetric_chamfer_px": 1.0},
            ), mock.patch.object(
                time_objective, "merge_penalty", return_value=invalid
            ):
                with self.assertRaises(ValueError):
                    evaluate_time_objective(
                        self.predictions,
                        self.states,
                        {},
                        {},
                        ScoreWeights(),
                    )

    def test_multiplier_conversion_and_history_schema(self):
        optimizer = load_optimizer_module()
        base = (10.0,) * 8
        multipliers = (0.5, 0.75, 1.0, 1.25, 1.5, 1.0, 0.8, 1.2)
        times = optimizer.times_from_multipliers(base, multipliers, 0.5, 1.5)
        self.assertEqual(times, tuple(10.0 * value for value in multipliers))

        score = time_objective.ScoreBreakdown(
            chamfer_4m_vs_5m=1.0,
            chamfer_4e_vs_5e=2.0,
            chamfer_score=3.0,
            merge_penalty=0.0,
            down_penalty=0.0,
            chamfer_weight=1.0,
            merge_weight=1.0,
            down_weight=1.0,
            weighted_chamfer=3.0,
            weighted_merge=0.0,
            weighted_down=0.0,
            total_score=3.0,
        )
        evaluation = optimizer.CandidateEvaluation(
            multipliers=multipliers,
            times_s=times,
            score=score,
            objective=3.0,
            status="ok",
            error="",
            elapsed_s=0.1,
        )
        row = optimizer.evaluation_row(1, 1, "0", evaluation)
        self.assertEqual(set(row), set(optimizer.HISTORY_FIELDS))
        self.assertEqual(row["objective"], 3.0)
        self.assertNotIn("total_score", row)

    def test_candidate_failure_uses_penalty(self):
        optimizer = load_optimizer_module()
        runtime = mock.Mock()
        runtime.states = {"init": np.zeros((2, 2))}
        fake_runtime_module = types.ModuleType("epi_pinn.prediction_runtime")
        fake_runtime_module.PREDICT_1M_TO_4E_STEPS = ()
        fake_runtime_module.run_prediction_sequence = mock.Mock(
            side_effect=RuntimeError("inference failed")
        )
        with mock.patch.dict(
            sys.modules,
            {"epi_pinn.prediction_runtime": fake_runtime_module},
        ):
            result = optimizer.evaluate_candidate(
                runtime,
                (10.0,) * 8,
                (1.0,) * 8,
                0.5,
                1.5,
                0.0,
                ScoreWeights(),
                1.0e9,
            )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.objective, 1.0e9)
        self.assertIn("inference failed", result.error)


    def test_config_workbook_is_hashed_for_resume_identity(self):
        optimizer = load_optimizer_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = root / "input.xlsx"
            workbook.write_bytes(b"first workbook content")
            runtime = types.SimpleNamespace(
                workbook_path=None,
                config_root=root,
                config={"data": {"workbooks": {"all": "input.xlsx"}}},
            )
            first = optimizer.workbook_identity(runtime)
            workbook.write_bytes(b"changed workbook content")
            second = optimizer.workbook_identity(runtime)

        self.assertEqual(first["all"]["path"], str(workbook.resolve()))
        self.assertNotEqual(first["all"]["sha256"], second["all"]["sha256"])

    def test_cma_checkpoint_resume_and_seed_are_deterministic(self):
        try:
            import cma
        except ImportError:
            self.skipTest("cma is not installed")
        optimizer = load_optimizer_module()
        options = {
            "bounds": [[0.5] * 8, [1.5] * 8],
            "popsize": 2,
            "seed": 42,
            "maxiter": 1,
            "verb_log": 0,
            "verbose": -9,
        }
        first = cma.CMAEvolutionStrategy([1.0] * 8, 0.2, options)
        first_solutions = first.ask()
        second = cma.CMAEvolutionStrategy([1.0] * 8, 0.2, options)
        second_solutions = second.ask()
        np.testing.assert_allclose(first_solutions, second_solutions)
        objectives = [
            float(np.sum((np.asarray(x) - 1.0) ** 2))
            for x in first_solutions
        ]
        first.tell(first_solutions, objectives)

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "cmaes_state.pkl"
            optimizer.save_strategy(state_path, first)
            restored = optimizer.load_strategy(state_path)

        self.assertEqual(restored.countiter, 1)
        restored.opts.set({"maxiter": 2})
        restored.stop().clear()
        self.assertFalse(restored.stop())
        solutions = restored.ask()
        restored.tell(
            solutions,
            [float(np.sum((np.asarray(x) - 1.0) ** 2)) for x in solutions],
        )
        self.assertEqual(restored.countiter, 2)

if __name__ == "__main__":
    unittest.main()