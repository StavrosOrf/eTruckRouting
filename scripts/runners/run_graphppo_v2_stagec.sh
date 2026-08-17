#!/usr/bin/env bash
# GraphPPO v2 stage C: buy feasibility back without giving up the route length.
#
# Stage B reached 124.9 fleet travel hours on the test split -- statistically
# tied with the CP-SAT planner on jointly solved scenarios -- but at 0.800
# success against 0.863 for the makespan-era policy. The trade-off is governed
# by the *ratio* of the terminal success bonus to the dense per-leg travel
# penalty, so the way to recover feasibility without lengthening routes is to
# raise the numerator rather than lower the denominator.
#
#   c_sb6000 / c_sb9000   heavier success bonus at the stage-B travel weight
#   c_tm15                the alternative: lower the travel weight instead
#   c_sb6000_margin       heavier success bonus plus a terminal charge-margin
#                         bonus, since the remaining failures are strandings
#
# INIT_FROM must name the stage-B run selected on validation.
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-2000000}"
NUM_ENVS="${NUM_ENVS:-16}"
WORKERS="${WORKERS:-6}"
TORCH_THREADS="${TORCH_THREADS:-2}"
INIT_FROM="${INIT_FROM:?set INIT_FROM to the selected stage-B run directory}"
OUTPUT="${OUTPUT:-results/canonical/graphppo_v2_stagec}"
LOGS="${LOGS:-results/canonical/logs_v2_stagec}"
GPU="${GPU:-1}"

mkdir -p "$LOGS"

launch() {
  local name="$1"; shift
  echo "launching $name (from $INIT_FROM)"
  PYTHONPATH=. MPLCONFIGDIR=/tmp/evrp-matplotlib OMP_NUM_THREADS="$TORCH_THREADS" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    nohup .venv/bin/python scripts/training/train_canonical_ppo.py \
      --state-encoder hetero_graph --action-head complete_gcn \
      --total-timesteps "$TIMESTEPS" --num-envs "$NUM_ENVS" --rollout-steps 128 \
      --rollout-workers "$WORKERS" \
      --learning-rate 1e-4 --entropy-coefficient 0.01 \
      --validation-scenarios 40 --validate-every 50 \
      --selection-objective travel_time \
      --init-from "$INIT_FROM" \
      --device cuda:0 --torch-threads "$TORCH_THREADS" \
      --output "$OUTPUT" --run-name "$name" \
      "$@" > "$LOGS/${name}.log" 2>&1 &
}

launch c_sb6000        --time-multiplier 20 --success-bonus 6000
launch c_sb9000        --time-multiplier 20 --success-bonus 9000
launch c_tm15          --time-multiplier 15
launch c_sb6000_margin --time-multiplier 20 --success-bonus 6000 --energy-margin-bonus 1000

wait
echo "GraphPPO v2 stage C complete"
