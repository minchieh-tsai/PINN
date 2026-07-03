from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read_repo_file(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def loss_block(config_text: str) -> str:
    match = re.search(r"(?ms)^loss:\n(?P<body>.*?)(?=^\S)", config_text)
    if match is None:
        raise AssertionError("missing loss block")
    return match.group("body")


class MaeLossWiringTests(unittest.TestCase):
    def test_training_loop_logs_and_weights_endpoint_mae_loss(self):
        train = read_repo_file("src/epi_pinn/train.py")

        self.assertIn("endpoint_mae_loss", train)
        self.assertIn('"mae": float(loss_cfg.get("mae", 0.0))', train)
        self.assertIn("mae = endpoint_mae_loss(phi_pred, phi_target)", train)
        self.assertIn('loss_weights["mae"] * mae', train)
        self.assertIn('"mae_loss"', train)

    def test_training_loss_plot_knows_mae_component(self):
        training_plots = read_repo_file("src/epi_pinn/training_plots.py")

        self.assertIn('LossComponent("mae", "mae_loss", "mae", "MAE")', training_plots)

    def test_three_ablation_configs_define_mae_weight(self):
        for path in (
            "configs/ablation_data_only.yaml",
            "configs/ablation_rate_velocity.yaml",
            "configs/ablation_full_physics.yaml",
        ):
            with self.subTest(path=path):
                config = read_repo_file(path)
                self.assertRegex(loss_block(config), r"(?m)^  mae: ")


if __name__ == "__main__":
    unittest.main()
