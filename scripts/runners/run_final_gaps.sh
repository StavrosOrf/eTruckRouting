#!/usr/bin/env bash
# The two runs left after batch 5.
#
#   ppo_flat_unmasked  flat encoder, independent action head, *structural* mask
#                      -- the literal "PPO" row of the manuscript's tables, as
#                      distinct from ppo_flat which is the MaskPPO analogue.
#                      Without it the replacement table cannot reuse the row
#                      names the paper's other tables use.
set -euo pipefail
cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-2000000}"
LOGS="${LOGS:-results/canonical/logs_final_gaps}"
mkdir -p "$LOGS"

PYTHONPATH=. MPLCONFIGDIR=/tmp/evrp-matplotlib OMP_NUM_THREADS=2 \
  CUDA_VISIBLE_DEVICES=1 \
  nohup .venv/bin/python scripts/training/train_canonical_ppo.py \
    --state-encoder flat --action-head independent \
    --total-timesteps "$TIMESTEPS" --num-envs 16 --rollout-steps 128 \
    --rollout-workers 6 \
    --learning-rate 3e-4 --entropy-coefficient 0.02 \
    --validation-scenarios 40 --validate-every 50 \
    --selection-objective travel_time \
    --curriculum configs/curriculum/energy_ramp.json \
    --time-multiplier 10 \
    --policy-action-mask structural --invalid-action-mode terminate \
    --device cuda:0 --torch-threads 2 \
    --output results/canonical/learned_baselines --run-name ppo_flat_unmasked \
    > "$LOGS/ppo_flat_unmasked.log" 2>&1 &

echo "launched ppo_flat_unmasked"
