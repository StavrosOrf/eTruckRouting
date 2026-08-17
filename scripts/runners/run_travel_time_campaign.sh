#!/usr/bin/env bash
# GraphPPO refinement against the fleet travel-time objective.
#
# SUPERSEDED, AND NO LONGER RUNNABLE AS WRITTEN. Every arm refines a checkpoint
# trained on the pre-`joint-fleet-v4` observation, which is narrower than the one
# the environment now emits, so `--init-from` will refuse it. The findings this
# sweep produced are recorded in document 10 section 4, and the ablation it was
# reaching for is done properly by the `v2_ablate` arm of
# `run_graphppo_v2_campaign.sh`, which zeroes the new action columns while
# holding the observation width, network shape, and budget fixed.
#
# Kept for the record of what was run.
#
# The campaign objective is the total hours driven by all trucks on a plan that
# delivers everything -- not makespan. Those differ: a plan can finish early in
# wall-clock terms while both trucks take long detours to chargers, which is
# exactly what the incumbent policy does (33% more distance than the CP-SAT
# planner on jointly solved scenarios, from 10.2 charging stops against 6.8).
#
# Two levers are separated here:
#   time_multiplier     dense, per-leg: every hour driven is charged at the
#                       moment the leg is chosen, so credit assignment is local.
#                       The config default of 1.0 makes travel worth ~150 reward
#                       against 5000 for deliveries, i.e. effectively ignored.
#   travel_time_bonus   terminal, paid only on a complete plan, so it can never
#                       make abandoning deliveries look profitable.
#
# `control` re-runs the incumbent makespan objective on the identical budget, so
# a win cannot be attributed to the extra training alone.
#
# Every arm refines the same frozen from-scratch checkpoint with the recipe that
# produced it (lr 1e-4, entropy 0.01, 16 envs), and keeps checkpoints on
# validation travel time.
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-1500000}"
NUM_ENVS="${NUM_ENVS:-16}"
TORCH_THREADS="${TORCH_THREADS:-2}"
INIT_FROM="${INIT_FROM:-results/canonical/graphppo_speed/speed1500}"
OUTPUT="${OUTPUT:-results/canonical/graphppo_travel}"
LOGS="${LOGS:-results/canonical/logs_travel}"
GPU="${GPU:-1}"

mkdir -p "$LOGS"

launch() {
  local name="$1"; shift
  echo "launching $name"
  PYTHONPATH=. MPLCONFIGDIR=/tmp/evrp-matplotlib OMP_NUM_THREADS="$TORCH_THREADS" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    nohup .venv/bin/python scripts/training/train_canonical_ppo.py \
      --state-encoder hetero_graph --action-head complete_gcn \
      --total-timesteps "$TIMESTEPS" --num-envs "$NUM_ENVS" --rollout-steps 128 \
      --learning-rate 1e-4 --entropy-coefficient 0.01 \
      --validation-scenarios 40 --validate-every 50 \
      --selection-objective travel_time \
      --init-from "$INIT_FROM" \
      --device cuda:0 --torch-threads "$TORCH_THREADS" \
      --output "$OUTPUT" --run-name "$name" \
      "$@" > "$LOGS/${name}.log" 2>&1 &
}

# Dense penalty alone, spanning two orders of magnitude around the point where
# travel hours become comparable to the delivery bonus.
launch tm5          --time-multiplier 5
launch tm10         --time-multiplier 10
launch tm20         --time-multiplier 20
launch tm40         --time-multiplier 40

# Dense plus terminal.
launch tm10_tb3000  --time-multiplier 10 --travel-time-bonus 3000
launch tm20_tb3000  --time-multiplier 20 --travel-time-bonus 3000

# Terminal alone, to show whether local credit assignment is what matters.
launch tb6000       --travel-time-bonus 6000

# Incumbent objective on the identical budget.
launch control      --speed-bonus 1500

wait
echo "travel-time campaign complete"
