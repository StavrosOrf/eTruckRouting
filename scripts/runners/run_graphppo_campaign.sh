#!/usr/bin/env bash
# GraphPPO campaign: classic PPO from random initialisation, no imitation.
#
# Measured obstacle: from scratch the agent reaches ~0.82 completion under the
# per-delivery bonus but essentially never closes the mandatory depot return, so
# the terminal success bonus is never sampled. With a stranding penalty on top,
# training collapses to risk-avoidance (completion falls to 0.26-0.33).
#
# The arms separate the two candidate remedies:
#   milestone  - reward serving every customer, an intermediate the agent does
#                reach, so the last mile carries gradient
#   curriculum - ramp physical difficulty toward the target configuration
#   both       - milestone plus curriculum
#   control    - neither, to show the wall is real
#
# Every arm trains the proposed GraphPPO architecture (hetero_graph state graph
# + complete_gcn action graph) and is validated on the *target* configuration.
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-4000000}"
NUM_ENVS="${NUM_ENVS:-16}"
TORCH_THREADS="${TORCH_THREADS:-3}"
OUTPUT="${OUTPUT:-results/canonical/graphppo}"
LOGS="${LOGS:-results/canonical/logs_graphppo}"
CURRICULUM="${CURRICULUM:-configs/curriculum/energy_ramp.json}"

mkdir -p "$LOGS"

launch() {
  local name="$1"; shift
  echo "launching $name"
  PYTHONPATH=. MPLCONFIGDIR=/tmp/evrp-matplotlib OMP_NUM_THREADS="$TORCH_THREADS" \
    CUDA_VISIBLE_DEVICES=1 \
    nohup .venv/bin/python scripts/training/train_canonical_ppo.py \
      --state-encoder hetero_graph --action-head complete_gcn \
      --total-timesteps "$TIMESTEPS" --num-envs "$NUM_ENVS" \
      --learning-rate 3e-4 --entropy-coefficient 0.02 \
      --validation-scenarios 40 --validate-every 50 \
      --device cuda:0 --torch-threads "$TORCH_THREADS" \
      --output "$OUTPUT" --run-name "$name" \
      "$@" > "$LOGS/${name}.log" 2>&1 &
}

launch milestone   --all-served-bonus 4000.0
launch curriculum  --curriculum "$CURRICULUM"
launch both        --all-served-bonus 4000.0 --curriculum "$CURRICULUM"
launch control

wait
echo "GraphPPO campaign complete"
