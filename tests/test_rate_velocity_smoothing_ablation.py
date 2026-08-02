from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rate_velocity_ablation",
    ROOT / "scripts" / "run_rate_velocity_smoothing_ablation.py",
)
assert SPEC is not None and SPEC.loader is not None
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


class RateVelocityConfigTests(unittest.TestCase):
    def test_short_configs_isolate_smoothing_terms(self):
        expected = {
            "control": (0.0, 0.0),
            "normal": (0.02, 0.0),
            "smooth": (0.02, 0.001),
        }
        for name, (normal_weight, velocity_weight) in expected.items():
            config = ablation.load_yaml(
                ROOT / "configs" / f"ablation_rate_velocity_{name}.yaml"
            )
            with self.subTest(name=name):
                self.assertEqual(config["training"]["adam_steps"], 5000)
                self.assertFalse(config["training"]["use_lbfgs"])
                self.assertEqual(config["model"]["correction_scale"], 0.20)
                self.assertEqual(config["model"]["velocity_residual_fraction"], 0.20)
                self.assertFalse(config["model"]["use_transport_velocity"])
                self.assertFalse(config["model"]["use_curvature_velocity"])
                self.assertEqual(config["loss"]["eikonal"], 0.10)
                self.assertEqual(config["loss"]["velocity_jacobian"], 1.0e-3)
                self.assertEqual(config["loss"]["normal_consistency"], normal_weight)
                self.assertEqual(config["loss"]["velocity_smoothness"], velocity_weight)
                self.assertEqual(config["rollout"]["interface_smoothing_sigma_px"], 0.0)

    def test_temporary_config_applies_overrides_and_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.yaml"
            source.write_text(
                "project:\n"
                "  output_dir: artifacts/original\n"
                "training:\n"
                "  adam_steps: 5000\n"
                "rollout:\n"
                "  interface_smoothing_sigma_px: 0.0\n",
                encoding="utf-8",
            )
            with ablation.temporary_config(
                source,
                sigma=0.5,
                output_dir_value="artifacts/selected",
                adam_steps=20000,
            ) as variant:
                config = ablation.load_yaml(variant)
                self.assertTrue(variant.exists())
                self.assertEqual(config["project"]["output_dir"], "artifacts/selected")
                self.assertEqual(config["training"]["adam_steps"], 20000)
                self.assertEqual(config["rollout"]["interface_smoothing_sigma_px"], 0.5)
            self.assertFalse(variant.exists())


def metric_row(
    experiment: str,
    scope: str,
    state: str,
    chamfer: float,
    mae: float,
    curvature: float,
    sigma: float | str = "",
    components: float = 1.0,
) -> dict[str, object]:
    return {
        "experiment": experiment,
        "scope": scope,
        "sigma_px": sigma,
        "state": state,
        "zero_contour_symmetric_chamfer_px": chamfer,
        "contour20_y_mae_px": mae,
        "curvature_total_variation": curvature,
        "zero_contour_component_count": components,
        "target_zero_contour_component_count": 1.0,
    }


class WinnerSelectionTests(unittest.TestCase):
    def base_rows(self) -> list[dict[str, object]]:
        rows = []
        for state in ablation.INSAMPLE_STATES:
            rows.append(metric_row("control", "insample", state, 10.0, 8.0, 10.0))
            rows.append(metric_row("smooth", "insample", state, 10.2, 8.1, 6.0))
        for state in ablation.HOLDOUT_STATES:
            rows.append(metric_row("control", "rollout", state, 12.0, 9.0, 12.0, 0.0))
            rows.append(metric_row("control", "rollout", state, 12.1, 9.1, 8.0, 0.5))
            rows.append(metric_row("smooth", "rollout", state, 12.2, 9.2, 5.0, 0.0))
            rows.append(metric_row("smooth", "rollout", state, 12.3, 9.3, 4.0, 0.5))
        return rows

    def test_selects_lowest_curvature_candidate_within_five_percent(self):
        result = ablation.select_winner(self.base_rows(), tolerance=0.05)
        self.assertEqual(result["winner"]["experiment"], "smooth")
        self.assertEqual(result["winner"]["sigma_px"], 0.5)
        self.assertTrue(result["winner"]["eligible"])

    def test_rejects_candidate_with_large_error(self):
        rows = self.base_rows()
        for row in rows:
            if row["experiment"] == "smooth" and row["scope"] == "rollout":
                row["zero_contour_symmetric_chamfer_px"] = 20.0
        result = ablation.select_winner(rows, tolerance=0.05)
        self.assertEqual(result["winner"]["experiment"], "control")

    def test_rejects_extra_zero_contour_components(self):
        rows = self.base_rows()
        for row in rows:
            if row["experiment"] == "smooth":
                row["zero_contour_component_count"] = 2.0
        result = ablation.select_winner(rows, tolerance=0.05)
        self.assertEqual(result["winner"]["experiment"], "control")


if __name__ == "__main__":
    unittest.main()
