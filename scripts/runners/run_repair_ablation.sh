#!/usr/bin/env bash
# Ablation over the three residual-failure repairs, on the frozen architecture.
#
#   depot_only     : schema-v3 depot visibility, original shaping
#   +stranding     : adds the failure-type-weighted penalty
#   +margin        : adds the terminal energy-margin bonus
#   full           : all three
#
# Every arm shares the architecture, budget, seeds, and demonstration archive,
# so the only thing that varies is the repair under test. Selection stays on the
# validation split.
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-800000}"
NUM_ENVS="${NUM_ENVS:-16}"
DEVICE="${DEVICE:-cuda:1}"
TORCH_THREADS="${TORCH_THREADS:-4}"
OUTPUT="${OUTPUT:-results/canonical/repair}"
LOGS="${LOGS:-results/canonical/logs_repair}"
DEMOS="${DEMOS:-results/canonical/demonstrations/ensemble.npz}"
ENCODER="${ENCODER:-hetero_graph}"
HEAD="${HEAD:-self_attention}"
SEED="${SEED:-0}"

mkdir -p "$LOGS"

launch() {
  local name="$1"; shift
  echo "launching $name"
  PYTHONPATH=. MPLCONFIGDIR=/tmp/evrp-matplotlib OMP_NUM_THREADS="$TORCH_THREADS" \
    CUDA_VISIBLE_DEVICES=1 \
    nohup .venv/bin/python scripts/training/train_canonical_ppo.py \
      --state-encoder "$ENCODER" --action-head "$HEAD" \
      --total-timesteps "$TIMESTEPS" --num-envs "$NUM_ENVS" \
      --learning-rate 1e-4 --entropy-coefficient 0.005 \
      --demonstrations "$DEMOS" --pretrain-epochs 20 \
      --validation-scenarios 40 --validate-every 25 \
      --device cuda:0 --torch-threads "$TORCH_THREADS" \
      --seed "$SEED" --output "$OUTPUT" --run-name "$name" \
      "$@" > "$LOGS/${name}.log" 2>&1 &
}

# CUDA_VISIBLE_DEVICES=1 remaps the allowed GPU to local index 0, so the
# process can only ever touch the requested physical device.
launch depot_only
launch stranding   --stranding-penalty 3000.0
launch margin      --energy-margin-bonus 1500.0
launch full        --stranding-penalty 3000.0 --energy-margin-bonus 1500.0

wait
echo "repair ablation complete"
