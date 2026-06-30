from pathlib import Path
import sys
import types
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Import training_plots.py without loading package __init__, which pulls in training dependencies.
PACKAGE = types.ModuleType("epi_pinn")
PACKAGE.__path__ = [str(ROOT / "src" / "epi_pinn")]
sys.modules.setdefault("epi_pinn", PACKAGE)

from epi_pinn import training_plots


class TrainingPlotTests(unittest.TestCase):
    def test_raw_loss_components_reads_known_log_columns(self):
        frame = pd.DataFrame(
            {
                "sdf_loss": [2.0],
                "dice_loss": [3.0],
                "unrelated": [99.0],
            }
        )

        raw = training_plots.raw_loss_components(frame)

        self.assertEqual(list(raw.columns), ["sdf", "dice"])
        self.assertAlmostEqual(raw.loc[0, "sdf"], 2.0)
        self.assertAlmostEqual(raw.loc[0, "dice"], 3.0)

    def test_weighted_loss_components_use_config_weights_and_skip_missing(self):
        frame = pd.DataFrame(
            {
                "sdf_loss": [2.0],
                "dice_loss": [3.0],
                "pde_loss": [4.0],
            }
        )

        weighted = training_plots.weighted_loss_components(
            frame,
            {"sdf": 0.5, "dice": 2.0, "pde": 0.25},
        )

        self.assertEqual(list(weighted.columns), ["sdf", "dice", "pde"])
        self.assertAlmostEqual(weighted.loc[0, "sdf"], 1.0)
        self.assertAlmostEqual(weighted.loc[0, "dice"], 6.0)
        self.assertAlmostEqual(weighted.loc[0, "pde"], 1.0)

    def test_loss_component_fractions_sum_to_one_and_handle_zero_rows(self):
        weighted = pd.DataFrame({"sdf": [1.0, 0.0], "dice": [3.0, 0.0]})

        fractions = training_plots.loss_component_fractions(weighted)

        self.assertAlmostEqual(fractions.loc[0].sum(), 1.0)
        self.assertAlmostEqual(fractions.loc[0, "sdf"], 0.25)
        self.assertAlmostEqual(fractions.loc[0, "dice"], 0.75)
        self.assertAlmostEqual(fractions.loc[1].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
