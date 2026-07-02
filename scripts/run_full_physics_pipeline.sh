#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
DEFAULT_CONFIGS=(
  "configs/ablation_data_only.yaml"
  "configs/ablation_rate_velocity.yaml"
  "configs/ablation_full_physics.yaml"
)

if [[ "$#" -gt 0 ]]; then
  CONFIGS=("$@")
else
  CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

run_step() {
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  "$@"
}

run_pipeline_for_config() {
  local config="$1"

  if [[ ! -f "$config" ]]; then
    echo "Config file not found: $config" >&2
    exit 1
  fi

  echo
  echo "============================================================"
  echo "Running pipeline for: $config"
  echo "============================================================"

  run_step "$PYTHON_BIN" scripts/inspect_xlsx.py --config "$config"
  run_step "$PYTHON_BIN" scripts/preprocess_data.py --config "$config" --split all

  run_step "$PYTHON_BIN" scripts/train_deposition.py --config "$config" --infer-missing-rates
  run_step "$PYTHON_BIN" scripts/train_etch.py --config "$config" --infer-missing-rates

  run_step "$PYTHON_BIN" scripts/run_rollout.py --config "$config" --infer-missing-rates
  run_step "$PYTHON_BIN" scripts/evaluate_holdout.py --config "$config"

  run_step "$PYTHON_BIN" scripts/plot_rollout_contours.py --config "$config"
  run_step "$PYTHON_BIN" scripts/plot_training_losses.py --config "$config"
}

echo "Project root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Configs: ${CONFIGS[*]}"

for config in "${CONFIGS[@]}"; do
  run_pipeline_for_config "$config"
done

echo
echo "All pipelines completed successfully."