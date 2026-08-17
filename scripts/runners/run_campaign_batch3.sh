#!/usr/bin/env bash
# Third training batch: seed replication for both sides of the mask ablation,
# the headline ladder's stage A at two further seeds, and a fairness check on
# the penalized variant.
#
#   mask_seed1 / mask_seed2   `mask_none_terminate` at seeds 1 and 2. The mask
#                             claim is the most load-bearing in the revision and
#                             was single-seed; its control (`v2_tm10`) is
#                             replicated by seed1_A/seed2_A from batch 2.
#   baseA_seed1 / baseA_seed2 stage A of the *headline* ladder, which selected
#                             `v2_base` -- the config-default travel penalty,
#                             not tm10 -- so replicating the pipeline means
#                             replicating that arm, not the ablation control.
#   penalize_p1000            the penalized mask variant collapsed at a penalty
#                             of 100. Ten times the penalty tests whether that
#                             was the magnitude rather than the mechanism, so
#                             the arm cannot be dismissed as a straw man.
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-2000000}"
NUM_ENVS="${NUM_ENVS:-16}"
WORKERS="${WORKERS:-5}"
TORCH_THREADS="${TORCH_THREADS:-2}"
LOGS="${LOGS:-results/canonical/logs_batch3}"
CURRICULUM="${CURRICULUM:-configs/curriculum/energy_ramp.json}"
GPU="${GPU:-1}"

mkdir -p "$LOGS"

launch() {
  local name="$1"; local output="$2"; shift 2
  echo "launching $name -> $output"
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
      --output "$output" --run-name "$name" \
      "$@" > "$LOGS/${name}.log" 2>&1 &
}

launch mask_seed1 results/canonical/mask_ablation --seed 1 --time-multiplier 10 \
  --policy-action-mask structural --invalid-action-mode terminate
launch mask_seed2 results/canonical/mask_ablation --seed 2 --time-multiplier 10 \
  --policy-action-mask structural --invalid-action-mode terminate
launch penalize_p1000 results/canonical/mask_ablation --seed 0 --time-multiplier 10 \
  --policy-action-mask structural --invalid-action-mode penalize \
  --invalid-action-penalty 1000
launch baseA_seed1 results/canonical/graphppo_seeds --seed 1
launch baseA_seed2 results/canonical/graphppo_seeds --seed 2

wait
echo "batch 3 complete"
