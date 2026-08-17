#!/usr/bin/env bash
# Does the hard feasibility mask explain GraphPPO's gain?  (R1.2, E3)
#
# The masked control is *not* trained here: `v2_tm10` from
# `run_graphppo_v2_campaign.sh` already is this exact configuration --
# hetero_graph + complete_gcn, energy-ramp curriculum, dense travel penalty 10,
# 2M steps, seed 0, from random initialisation with no imitation.  These arms
# change one thing and inherit everything else, so the comparison isolates the
# mask rather than the budget, the architecture, or the reward.
#
#   arm                    policy mask   infeasible action does
#   v2_tm10 (control)      hard          cannot be selected
#   mask_none_terminate    structural    strands the truck (simulator semantics)
#   mask_none_penalize     structural    refused, penalised, episode continues
#
# Both arms keep the identical observation and candidate set: `structural` hides
# only slots that denote no action at all.  `penalize` exists so the result
# cannot be dismissed as an artefact of the harshest possible treatment of a
# mistake -- if unmasked PPO fails under both, the mask is doing real work.
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-2000000}"
NUM_ENVS="${NUM_ENVS:-16}"
WORKERS="${WORKERS:-5}"
TORCH_THREADS="${TORCH_THREADS:-2}"
OUTPUT="${OUTPUT:-results/canonical/mask_ablation}"
LOGS="${LOGS:-results/canonical/logs_mask_ablation}"
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
      --time-multiplier 10 \
      --device cuda:0 --torch-threads "$TORCH_THREADS" \
      --output "$OUTPUT" --run-name "$name" \
      "$@" > "$LOGS/${name}.log" 2>&1 &
}

launch mask_none_terminate --seed 0 \
  --policy-action-mask structural --invalid-action-mode terminate
launch mask_none_penalize  --seed 0 \
  --policy-action-mask structural --invalid-action-mode penalize

wait
echo "mask ablation complete"
