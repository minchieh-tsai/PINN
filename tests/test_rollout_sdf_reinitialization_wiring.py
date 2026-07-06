from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read_repo_file(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def rollout_block(config_text: str) -> str:
    match = re.search(r"(?ms)^rollout:\n(?P<body>.*?)(?=^\S)", config_text)
    if match is None:
        raise AssertionError("missing rollout block")
    return match.group("body")


class RolloutSdfReinitializationWiringTests(unittest.TestCase):
    def test_rollout_reinitializes_sdf_after_each_prediction_when_enabled(self):
        rollout = read_repo_file("src/epi_pinn/rollout.py")

        self.assertIn("rebuild_sdf_from_mask", rollout)
        self.assertIn("reinitialize_sdf_each_step", rollout)
        self.assertIn("bool(config.get(\"rollout\", {}).get(\"reinitialize_sdf_each_step\", False))", rollout)
        self.assertIn("phi = rebuild_sdf_from_mask(phi < 0.0)", rollout)

    def test_configs_enable_rollout_sdf_reinitialization(self):
        for path in (
            "configs/default.yaml",
            "configs/ablation_data_only.yaml",
            "configs/ablation_rate_velocity.yaml",
            "configs/ablation_full_physics.yaml",
        ):
            with self.subTest(path=path):
                config = read_repo_file(path)
                self.assertRegex(rollout_block(config), r"(?m)^  reinitialize_sdf_each_step: true$")


if __name__ == "__main__":
    unittest.main()
