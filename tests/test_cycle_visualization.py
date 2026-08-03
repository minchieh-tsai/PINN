from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import types
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = types.ModuleType("epi_pinn")
PACKAGE.__path__ = [str(ROOT / "src" / "epi_pinn")]
sys.modules.setdefault("epi_pinn", PACKAGE)

from epi_pinn.cycle_visualization import (
    GroundTruthOverlay,
    create_cycle_figure,
    gt_band_mask,
    save_cycle_figure,
)


def circle_phi(height=40, width=30, center_y=22.0, radius=10.0):
    y, x = np.mgrid[:height, :width]
    return np.sqrt((x - width / 2.0) ** 2 + (y - center_y) ** 2) - radius


def step(state, input_state, duration, input_phi, prediction_phi):
    return types.SimpleNamespace(
        process_name="deposition" if state.endswith("M") else "etch",
        cycle=int(state[0]),
        output_state=state,
        input_state=input_state,
        duration_s=duration,
        average_rate=1.0,
        input_phi=input_phi,
        model_input_phi=input_phi.copy(),
        prediction_phi=prediction_phi,
    )


class CycleVisualizationTests(unittest.TestCase):
    def test_gt_band_is_inclusive_and_validated(self):
        phi = np.array([[-3.1, -3.0, 0.0, 3.0, 3.1]])
        np.testing.assert_array_equal(
            gt_band_mask(phi, 3.0),
            np.array([[False, True, True, True, False]]),
        )
        with self.assertRaises(ValueError):
            gt_band_mask(phi, -1.0)

    def test_figure_has_one_axis_per_state_and_does_not_mutate_arrays(self):
        phi0 = circle_phi(center_y=20.0)
        phi1 = circle_phi(center_y=21.0)
        phi2 = circle_phi(center_y=22.0)
        snapshots = [array.copy() for array in (phi0, phi1, phi2)]
        steps = [
            step("1M", "init", 100.0, phi0, phi1),
            step("2M", "predicted 1E", 120.0, phi1, phi2),
        ]
        overlays = {
            "1M": GroundTruthOverlay(phi1.copy(), "1M"),
            "2M": GroundTruthOverlay(phi2.copy(), "5M"),
        }

        fig = create_cycle_figure(
            steps,
            ("1M", "2M"),
            overlays,
            plot_gaussian_sigma=0.75,
            gt_band_px=3.0,
            contour_mode="all",
            min_contour_points=2,
        )
        try:
            self.assertEqual(len(fig.axes), 2)
            self.assertIn("input: init", fig.axes[0].get_title())
            self.assertIn("GT target: 5M", fig.axes[1].get_title())
        finally:
            plt.close(fig)

        for actual, expected in zip((phi0, phi1, phi2), snapshots):
            np.testing.assert_array_equal(actual, expected)

    def test_save_closes_figure_and_creates_nonempty_png(self):
        phi0 = circle_phi(center_y=20.0)
        phi1 = circle_phi(center_y=21.0)
        steps = [step("1E", "predicted 1M", 10.0, phi0, phi1)]
        overlays = {
            "1E": GroundTruthOverlay(phi1.copy(), "5E", band_color="#2f9e44")
        }
        before = set(plt.get_fignums())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cycle.png"
            result = save_cycle_figure(
                steps,
                ("1E",),
                overlays,
                output,
                dpi=72,
                contour_mode="all",
                min_contour_points=2,
            )
            self.assertEqual(result, output.resolve())
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)
        self.assertEqual(set(plt.get_fignums()), before)


if __name__ == "__main__":
    unittest.main()