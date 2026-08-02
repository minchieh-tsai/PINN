from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epi_pinn.losses import normal_consistency_loss, velocity_neighbor_smoothness_loss
from epi_pinn.sampling import sample_neighbor_stencils


class SmoothnessLossTests(unittest.TestCase):
    def test_normal_consistency_is_small_for_matching_normals(self):
        pred = torch.tensor([[0.0, -1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
        target = pred.clone()

        loss = normal_consistency_loss(pred, target)

        self.assertLess(float(loss), 1.0e-7)

    def test_normal_consistency_penalizes_opposite_normals(self):
        pred = torch.tensor([[0.0, -1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
        target = torch.tensor([[0.0, 1.0, -1.0, 0.0, 0.0]], dtype=torch.float64)

        loss = normal_consistency_loss(pred, target)

        self.assertGreater(float(loss), 1.9)

    def test_velocity_smoothness_uses_rate_normalized_neighbor_changes(self):
        smooth = torch.full((2, 5), 4.0, dtype=torch.float64)
        rough = smooth.clone()
        rough[:, 2] = 6.0
        rate = torch.tensor(2.0, dtype=torch.float64)

        smooth_loss = velocity_neighbor_smoothness_loss(smooth, rate)
        rough_loss = velocity_neighbor_smoothness_loss(rough, rate)

        self.assertEqual(float(smooth_loss), 0.0)
        self.assertGreater(float(rough_loss), 0.0)

    def test_neighbor_stencils_stay_interior_and_in_narrow_band(self):
        phi = np.full((7, 8), 10.0, dtype=np.float64)
        phi[2:5, 2:6] = 0.0
        stencils = sample_neighbor_stencils(phi, 32, 1.0, np.random.default_rng(7))

        centers = stencils[:, 0]
        y, x = np.divmod(centers, phi.shape[1])
        self.assertTrue(np.all((x > 0) & (x < phi.shape[1] - 1)))
        self.assertTrue(np.all((y > 0) & (y < phi.shape[0] - 1)))
        self.assertTrue(np.all(np.abs(phi.reshape(-1)[centers]) <= 1.0))


if __name__ == "__main__":
    unittest.main()
