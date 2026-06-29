from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epi_pinn.models import DepositionPINN
from epi_pinn.sampling import FEATURE_NAMES
import epi_pinn.losses as losses


class CurvatureVelocityTests(unittest.TestCase):
    def test_curvature_velocity_adds_signed_curvature_effect(self):
        dtype = torch.float64
        model = DepositionPINN(
            {
                "solution_hidden_dim": 4,
                "solution_depth": 1,
                "velocity_hidden_dim": 4,
                "velocity_depth": 1,
                "contour_embedding_dim": 4,
                "velocity_residual_fraction": 0.0,
                "use_curvature_velocity": True,
                "curvature_velocity_weight": 0.25,
                "curvature_velocity_sign": 1.0,
                "curvature_reference": 0.5,
            }
        ).to(dtype=dtype)
        features = torch.zeros((2, len(FEATURE_NAMES)), dtype=dtype)
        features[:, 11] = torch.tensor([0.5, -0.5], dtype=dtype)
        contour = torch.zeros((20, 3), dtype=dtype)
        raw_phi0 = torch.zeros(2, dtype=dtype)
        duration = torch.tensor(1.0, dtype=dtype)
        average_rate = torch.tensor(2.0, dtype=dtype)

        _phi, velocity = model(features, contour, raw_phi0, duration, average_rate, 32.0)

        expected = average_rate * (1.0 + 0.25 * torch.tanh(features[:, 11] / 0.5))
        torch.testing.assert_close(velocity, expected)

    def test_velocity_jacobian_loss_penalizes_spatial_velocity_gradients(self):
        self.assertTrue(hasattr(losses, "velocity_jacobian_loss"))
        features = torch.zeros((3, len(FEATURE_NAMES)), dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            features[:, 0] = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
            features[:, 1] = torch.tensor([0.0, 0.25, -0.25], dtype=torch.float64)
        scale = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        velocity = scale * features[:, 0] ** 2 + 2.0 * features[:, 1]

        loss = losses.velocity_jacobian_loss(velocity, features, length_x=2.0, length_y=4.0)

        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(scale.grad)
        self.assertNotEqual(float(scale.grad.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
