#!/usr/bin/env bash
# Classic PPO on the proposed setting: no demonstrations, no behaviour cloning.
#
# The policy starts from random initialisation and learns only from its own
# closed-loop experience. Reward shaping is retained -- it is a training signal,
# not a demonstrator -- and the depot-visibility fix (schema v3) is in place, so
# the return leg the agent must plan for is actually observable.
#
# Exploration is the known obstacle: an episode needs roughly a hundred
# consecutive well-chosen actions before it succeeds, so the arms below vary the
# entropy bonus and learning rate rather than the architecture.
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-3000000}"
NUM_ENVS="${NUM_ENVS:-16}"
TORCH_THREADS="${TORCH_THREADS:-4}"
OUTPUT="${OUTPUT:-results/canonical/pure_ppo}"
LOGS="${LOGS:-results/canonical/logs_pure}"
ENCODER="${ENCODER:-hetero_graph}"
HEAD="${HEAD:-self_attention}"

mkdir -p "$LOGS"

launch() {
  local name="$1"; shift
  echo "launching $name"
  PYTHONPATH=. MPLCONFIGDIR=/tmp/evrp-matplotlib OMP_NUM_THREADS="$TORCH_THREADS" \
    CUDA_VISIBLE_DEVICES=1 \
    nohup .venv/bin/python scripts/training/train_canonical_ppo.py \
      --state-encoder "$ENCODER" --action-head "$HEAD" \
      --total-timesteps "$TIMESTEPS" --num-envs "$NUM_ENVS" \
      --validation-scenarios 40 --validate-every 50 \
      --stranding-penalty 3000.0 --energy-margin-bonus 1500.0 \
      --device cuda:0 --torch-threads "$TORCH_THREADS" \
      --output "$OUTPUT" --run-name "$name" \
      "$@" > "$LOGS/${name}.log" 2>&1 &
}

# No --demonstrations and no --pretrain-demonstrations: the behaviour-cloning
# stage is skipped entirely and PPO starts from random weights.
launch entropy_low     --entropy-coefficient 0.005 --learning-rate 3e-4 --seed 0
launch entropy_mid     --entropy-coefficient 0.02  --learning-rate 3e-4 --seed 0
launch entropy_high    --entropy-coefficient 0.05  --learning-rate 3e-4 --seed 0
launch entropy_mid_s1  --entropy-coefficient 0.02  --learning-rate 3e-4 --seed 1

wait
echo "pure PPO sweep complete"
