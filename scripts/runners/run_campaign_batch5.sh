#!/usr/bin/env bash
# Fifth training batch: stage C of the seed replication, the duration-action
# comparison, and a size-generalizing policy.
#
#   seed1_C / seed2_C   the last stage of the headline ladder at seeds 1 and 2,
#                       with seed 0's selected hyperparameters (`c_tm15`).
#   duration            charging expressed as 15/30/60-minute durations instead
#                       of target SoC (R2.6). Another action space, so another
#                       policy.
#   scale_envelope      a policy trained on a *variable* instance size inside a
#                       larger envelope -- up to 4 trucks and 14 customers. The
#                       headline policy has a fixed observation width and can
#                       only be evaluated at or below its trained envelope, so
#                       upward size transfer and any real charger contention
#                       need a policy trained for them. This is what the scale
#                       grid and the congestion sensitivity are scored on.
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-2000000}"
NUM_ENVS="${NUM_ENVS:-16}"
WORKERS="${WORKERS:-5}"
TORCH_THREADS="${TORCH_THREADS:-2}"
LOGS="${LOGS:-results/canonical/logs_batch5}"
CURRICULUM="${CURRICULUM:-configs/curriculum/energy_ramp.json}"
GPU="${GPU:-1}"

mkdir -p "$LOGS"

refine() {
  local name="$1"; local output="$2"; local init="$3"; shift 3
  echo "launching $name (from $init)"
  PYTHONPATH=. MPLCONFIGDIR=/tmp/evrp-matplotlib OMP_NUM_THREADS="$TORCH_THREADS" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    nohup .venv/bin/python scripts/training/train_canonical_ppo.py \
      --state-encoder hetero_graph --action-head complete_gcn \
      --total-timesteps "$TIMESTEPS" --num-envs "$NUM_ENVS" --rollout-steps 128 \
      --rollout-workers "$WORKERS" \
      --learning-rate 1e-4 --entropy-coefficient 0.01 \
      --validation-scenarios 40 --validate-every 50 \
      --selection-objective travel_time \
      --init-from "$init" \
      --device cuda:0 --torch-threads "$TORCH_THREADS" \
      --output "$output" --run-name "$name" \
      "$@" > "$LOGS/${name}.log" 2>&1 &
}

scratch() {
  local name="$1"; local output="$2"; shift 2
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
      --time-multiplier 10 \
      --device cuda:0 --torch-threads "$TORCH_THREADS" \
      --output "$output" --run-name "$name" \
      "$@" > "$LOGS/${name}.log" 2>&1 &
}

refine seed1_C results/canonical/graphppo_seeds \
  results/canonical/graphppo_seeds/seed1_B --time-multiplier 15
refine seed2_C results/canonical/graphppo_seeds \
  results/canonical/graphppo_seeds/seed2_B --time-multiplier 15
scratch duration results/canonical/charging_actions --seed 0 \
  --config EVRoutingEnv/config_files/config_joint_duration.yaml
scratch scale_envelope results/canonical/scale --seed 0 \
  --config EVRoutingEnv/config_files/config_joint_scale.yaml

wait
echo "batch 5 complete"
