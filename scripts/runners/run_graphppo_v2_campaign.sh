#!/usr/bin/env bash
# GraphPPO v2 stage A: routing-aware action nodes, from scratch, no imitation.
#
# Why a new from-scratch campaign rather than another refinement:
#
# The action head scores every candidate from its own feature row plus a pooled
# state embedding. Before this campaign a row carried no travel cost at all --
# only ``required_energy`` as an indirect proxy -- so the policy was asked to
# minimize fleet travel hours while unable to see how long any leg takes, or
# whether a charger sits on the way or far off it. ``ROUTING_ACTION_FEATURES``
# adds the leg hours, the return leg from the target, the target's distance to
# the work that remains, and the insertion detour. That widens the canonical
# observation, so no earlier checkpoint transfers and the curriculum has to run
# again from random initialisation.
#
# The arms isolate the two changes independently:
#
#   arm            routing features   dense travel penalty
#   v2_base        yes                1.0 (config default, i.e. negligible)
#   v2_tm10        yes                10
#   v2_tm20        yes                20
#   v2_ablate      zeroed             10
#
# `v2_ablate` keeps the observation width, the network shape, and the budget
# identical and only blanks the new columns, so any gap to `v2_tm10` is what the
# features bought rather than what the reward change bought.
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-2000000}"
NUM_ENVS="${NUM_ENVS:-16}"
WORKERS="${WORKERS:-6}"
TORCH_THREADS="${TORCH_THREADS:-2}"
OUTPUT="${OUTPUT:-results/canonical/graphppo_v2}"
LOGS="${LOGS:-results/canonical/logs_v2}"
CURRICULUM="${CURRICULUM:-configs/curriculum/energy_ramp.json}"
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
      --rollout-workers "$WORKERS" \
      --learning-rate 3e-4 --entropy-coefficient 0.02 \
      --validation-scenarios 40 --validate-every 50 \
      --selection-objective travel_time \
      --curriculum "$CURRICULUM" \
      --device cuda:0 --torch-threads "$TORCH_THREADS" \
      --output "$OUTPUT" --run-name "$name" \
      "$@" > "$LOGS/${name}.log" 2>&1 &
}

launch v2_base    --seed 0
launch v2_tm10    --seed 0 --time-multiplier 10
launch v2_tm20    --seed 0 --time-multiplier 20
launch v2_ablate  --seed 0 --time-multiplier 10 --disable-routing-action-features

wait
echo "GraphPPO v2 stage A complete"
