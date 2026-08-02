from pathlib import Path
import sys
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Import evaluate.py without loading package __init__, which pulls in training dependencies.
PACKAGE = types.ModuleType("epi_pinn")
PACKAGE.__path__ = [str(ROOT / "src" / "epi_pinn")]
sys.modules.setdefault("epi_pinn", PACKAGE)
sys.modules.setdefault(
    "yaml",
    types.SimpleNamespace(safe_load=lambda _handle: {}, safe_dump=lambda _config, _handle, sort_keys=False: None),
)

from epi_pinn import evaluate


class EvaluateContourMetricTests(unittest.TestCase):
    def test_zero_contour_points_flattens_all_contours_to_pixel_xy(self):
        original = evaluate._find_zero_contours
        try:
            evaluate._find_zero_contours = lambda _phi: [
                np.array([[1.25, 2.5], [3.0, 4.0]], dtype=np.float64),
                np.array([[0.0, 1.0]], dtype=np.float64),
            ]

            points = evaluate._zero_contour_points(np.zeros((5, 5), dtype=np.float64))

            expected = np.array([[2.5, 1.25], [4.0, 3.0], [1.0, 0.0]], dtype=np.float64)
            np.testing.assert_allclose(points, expected)
        finally:
            evaluate._find_zero_contours = original

    def test_contour_metrics_keep_contour20_and_add_full_zero_contour_chamfer(self):
        y = np.arange(5, dtype=np.float64).reshape(5, 1)
        pred = np.repeat(y - 1.0, 5, axis=1)
        target = np.repeat(y - 2.0, 5, axis=1)
        full_pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float64)
        full_target = np.array(
            [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]],
            dtype=np.float64,
        )
        calls = []
        original = evaluate._zero_contour_points
        try:
            def fake_zero_contour_points(phi):
                calls.append(phi)
                return full_pred if len(calls) == 1 else full_target

            evaluate._zero_contour_points = fake_zero_contour_points
            metrics = evaluate._contour_metrics(pred, target, {"num_points": 3, "min_valid_points": 1})
        finally:
            evaluate._zero_contour_points = original

        self.assertIn("contour20_y_mae_px", metrics)
        self.assertIn("contour20_symmetric_chamfer_px", metrics)
        self.assertIn("zero_contour_symmetric_chamfer_px", metrics)
        self.assertAlmostEqual(metrics["contour20_y_mae_px"], 1.0)
        self.assertAlmostEqual(
            metrics["zero_contour_symmetric_chamfer_px"],
            evaluate._symmetric_chamfer(full_pred, full_target),
        )
        self.assertNotEqual(metrics["zero_contour_symmetric_chamfer_px"], metrics["contour20_symmetric_chamfer_px"])


    def test_curvature_metrics_distinguish_straight_and_oscillating_contours(self):
        x = np.linspace(0.0, 40.0, 161)
        straight = np.stack([np.zeros_like(x), x], axis=1)
        oscillating = np.stack([2.0 * np.sin(1.2 * x), x], axis=1)

        straight_metrics = evaluate._contour_curvature_statistics([straight])
        oscillating_metrics = evaluate._contour_curvature_statistics([oscillating])

        self.assertAlmostEqual(straight_metrics["curvature_rms"], 0.0)
        self.assertAlmostEqual(straight_metrics["curvature_total_variation"], 0.0)
        self.assertGreater(oscillating_metrics["curvature_rms"], straight_metrics["curvature_rms"])
        self.assertGreater(
            oscillating_metrics["curvature_total_variation"],
            straight_metrics["curvature_total_variation"],
        )

    def test_contour_metrics_report_roughness_and_component_counts(self):
        y = np.arange(9, dtype=np.float64).reshape(9, 1)
        pred = np.repeat(y - 4.0, 9, axis=1)
        target = np.repeat(y - 5.0, 9, axis=1)

        metrics = evaluate._contour_metrics(pred, target, {"num_points": 5, "min_valid_points": 1})

        self.assertIn("contour_curvature_rms", metrics)
        self.assertIn("curvature_total_variation", metrics)
        self.assertEqual(metrics["zero_contour_component_count"], 1.0)
        self.assertEqual(metrics["target_zero_contour_component_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
