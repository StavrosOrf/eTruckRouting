#!/usr/bin/env bash
# Fourth training batch: stage B of the seed replication, the remaining feature
# ablations, and the charging action-space comparison.
#
#   seed1_B / seed2_B   stage B of the headline ladder at seeds 1 and 2, from
#                       their own stage-A checkpoints. The hyperparameters are
#                       the ones seed 0 selected (`b_tm20`), not re-searched:
#                       this replicates the selected *chain*, which is what a
#                       seed-variance claim needs, rather than re-running the
#                       whole search at every seed.
#   ablate_queue        charger queue state -- ports, occupancy, waitlist,
#   ablate_active_truck known workload -- and the flag marking which truck is
#                       deciding, each blanked in turn (E3).
#   soc5                target-SoC actions at 5% granularity instead of 10%
#                       (R2.6). The action space changes, so this is a separate
#                       policy rather than a re-evaluation.
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-2000000}"
NUM_ENVS="${NUM_ENVS:-16}"
WORKERS="${WORKERS:-5}"
TORCH_THREADS="${TORCH_THREADS:-2}"
LOGS="${LOGS:-results/canonical/logs_batch4}"
CURRICULUM="${CURRICULUM:-configs/curriculum/energy_ramp.json}"
GPU="${GPU:-1}"

mkdir -p "$LOGS"

# Stage B refines an existing policy: lower learning rate and entropy, no
# curriculum, initialised from the stage-A run of the same seed.
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

refine seed1_B results/canonical/graphppo_seeds \
  results/canonical/graphppo_seeds/baseA_seed1 --time-multiplier 20
refine seed2_B results/canonical/graphppo_seeds \
  results/canonical/graphppo_seeds/baseA_seed2 --time-multiplier 20
scratch ablate_queue results/canonical/ablations --seed 0 --ablate-features queue
scratch ablate_active_truck results/canonical/ablations --seed 0 \
  --ablate-features active_truck
scratch soc5 results/canonical/charging_actions --seed 0 \
  --config EVRoutingEnv/config_files/config_joint_soc5.yaml

wait
echo "batch 4 complete"
