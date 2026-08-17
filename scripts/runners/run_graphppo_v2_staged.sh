#!/usr/bin/env bash
# GraphPPO v2 stage D: attack the residual failure mode directly.
#
# Stage C reached 0.817 test success at 122.0 fleet travel hours. Of its 55
# failures, 38 are `no_feasible_action` -- a truck driven into a state where the
# mask allows nothing, i.e. stranded mid-route -- and only 17 are anything else.
# So the remaining feasibility gap is not "routes left unfinished", it is
# "trucks that ran their margin too thin", which is a different lever from the
# success bonus Stage C raised.
#
#   d_str2000 / d_str4000   penalise the stranding terminations themselves
#   d_margin2000            pay for terminal charge in hand instead
#   d_tm10_sb6000           simply make detouring to charge cheaper
#
# All refine the stage-C run with the highest validation success.
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-2000000}"
NUM_ENVS="${NUM_ENVS:-16}"
WORKERS="${WORKERS:-6}"
TORCH_THREADS="${TORCH_THREADS:-2}"
INIT_FROM="${INIT_FROM:?set INIT_FROM to the selected stage-C run directory}"
OUTPUT="${OUTPUT:-results/canonical/graphppo_v2_staged}"
LOGS="${LOGS:-results/canonical/logs_v2_staged}"
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

launch d_str2000      --time-multiplier 15 --success-bonus 6000 --stranding-penalty 2000
launch d_str4000      --time-multiplier 15 --success-bonus 6000 --stranding-penalty 4000
launch d_margin2000   --time-multiplier 15 --success-bonus 6000 --energy-margin-bonus 2000
launch d_tm10_sb6000  --time-multiplier 10 --success-bonus 6000

wait
echo "GraphPPO v2 stage D complete"
