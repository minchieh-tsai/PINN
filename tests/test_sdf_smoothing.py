from pathlib import Path
import sys
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = types.ModuleType("epi_pinn")
PACKAGE.__path__ = [str(ROOT / "src" / "epi_pinn")]
sys.modules.setdefault("epi_pinn", PACKAGE)

from epi_pinn.sdf import rebuild_sdf_from_mask, smooth_and_rebuild_sdf


class SdfSmoothingTests(unittest.TestCase):
    def test_zero_sigma_matches_plain_sdf_reinitialization(self):
        phi = np.ones((9, 9), dtype=np.float64)
        phi[2:7, 3:6] = -1.0

        expected = rebuild_sdf_from_mask(phi < 0.0)
        actual = smooth_and_rebuild_sdf(phi, 0.0)

        np.testing.assert_allclose(actual, expected)

    def test_gaussian_smoothing_removes_isolated_negative_pixel(self):
        phi = np.ones((9, 9), dtype=np.float64)
        phi[4, 4] = -1.0

        actual = smooth_and_rebuild_sdf(phi, 0.75)

        self.assertTrue(np.all(actual > 0.0))

    def test_negative_sigma_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            smooth_and_rebuild_sdf(np.ones((5, 5)), -0.1)


if __name__ == "__main__":
    unittest.main()
