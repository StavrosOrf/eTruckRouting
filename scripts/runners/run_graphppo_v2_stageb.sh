#!/usr/bin/env bash
# GraphPPO v2 stage B: refine the stage-A winner against fleet travel hours.
#
# Stage A has to learn feasibility first, so it cannot carry an aggressive
# travel penalty from random initialisation -- at high weight a policy that
# drives less outscores one that finishes, which is the failure `tm40` showed in
# the document-10 sweep. Once a feasible policy exists that trade-off is safe to
# push, which is the same two-stage shape that produced `speed1500`.
#
# INIT_FROM must name the stage-A run selected on validation.
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-2000000}"
NUM_ENVS="${NUM_ENVS:-16}"
WORKERS="${WORKERS:-6}"
TORCH_THREADS="${TORCH_THREADS:-2}"
INIT_FROM="${INIT_FROM:?set INIT_FROM to the selected stage-A run directory}"
OUTPUT="${OUTPUT:-results/canonical/graphppo_v2_stageb}"
LOGS="${LOGS:-results/canonical/logs_v2_stageb}"
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

# Three weights from the identical checkpoint on identical budgets, so the
# comparison isolates penalty strength. Note this is not a zero-penalty control:
# whichever stage-A run is selected may itself carry a penalty.
launch b_tm10         --time-multiplier 10
launch b_tm20         --time-multiplier 20
launch b_tm30         --time-multiplier 30
launch b_tm20_tb3000  --time-multiplier 20 --travel-time-bonus 3000

wait
echo "GraphPPO v2 stage B complete"
