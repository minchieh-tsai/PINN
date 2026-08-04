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
    def test_every_panel_uses_fixed_initial_boundary_without_gt_band(self):
        initial = circle_phi(center_y=18.0)
        step_input_1 = circle_phi(center_y=20.0)
        step_input_2 = circle_phi(center_y=21.0)
        prediction_1 = circle_phi(center_y=22.0)
        prediction_2 = circle_phi(center_y=23.0)
        arrays = (initial, step_input_1, step_input_2, prediction_1, prediction_2)
        snapshots = [array.copy() for array in arrays]
        steps = [
            step("1M", "init", 100.0, step_input_1, prediction_1),
            step("2M", "predicted 1E", 120.0, step_input_2, prediction_2),
        ]
        overlays = {
            "1M": GroundTruthOverlay(prediction_1.copy(), "1M"),
            "2M": GroundTruthOverlay(prediction_2.copy(), "5M"),
        }

        fig = create_cycle_figure(
            steps,
            ("1M", "2M"),
            overlays,
            initial_phi=initial,
            initial_label="init",
            plot_gaussian_sigma=0.75,
            contour_mode="all",
            min_contour_points=2,
        )
        try:
            self.assertEqual(len(fig.axes), 2)
            self.assertEqual(tuple(fig.get_size_inches()), (9.6, 8.5))
            for axis in fig.axes:
                self.assertEqual(len(axis.images), 0)
                self.assertGreaterEqual(len(axis.lines), 2)
                self.assertEqual(axis.lines[0].get_color(), "#202124")
                self.assertEqual(axis.lines[0].get_linestyle(), "-")
                self.assertNotIn("input:", axis.get_title())
            np.testing.assert_allclose(
                fig.axes[0].lines[0].get_xdata(),
                fig.axes[1].lines[0].get_xdata(),
            )
            np.testing.assert_allclose(
                fig.axes[0].lines[0].get_ydata(),
                fig.axes[1].lines[0].get_ydata(),
            )
            self.assertIn("GT target: 5M", fig.axes[1].get_title())
        finally:
            plt.close(fig)

        for actual, expected in zip(arrays, snapshots):
            np.testing.assert_array_equal(actual, expected)

    def test_figure_size_validation(self):
        phi = circle_phi()
        steps = [step("1M", "init", 100.0, phi, phi)]
        with self.assertRaises(ValueError):
            create_cycle_figure(
                steps,
                ("1M",),
                {},
                initial_phi=phi,
                panel_width=0.0,
            )
        with self.assertRaises(ValueError):
            create_cycle_figure(
                steps,
                ("1M",),
                {},
                initial_phi=phi,
                figure_height=float("nan"),
            )

    def test_save_closes_figure_and_creates_nonempty_png(self):
        initial = circle_phi(center_y=18.0)
        prediction = circle_phi(center_y=21.0)
        steps = [step("1E", "predicted 1M", 10.0, prediction, prediction)]
        overlays = {"1E": GroundTruthOverlay(prediction.copy(), "5E")}
        before = set(plt.get_fignums())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cycle.png"
            result = save_cycle_figure(
                steps,
                ("1E",),
                overlays,
                output,
                initial_phi=initial,
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