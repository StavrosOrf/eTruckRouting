#!/usr/bin/env bash
# The learned-baseline family under equal information (R1.6, R1.2, R2.9).
#
# Every arm shares the observation, the hard mask, the curriculum, the reward,
# the seed stream, and the 2M-step budget.  Only the architecture differs, so
# the comparison measures the architecture:
#
#   arm                      encoder        action head    stands in for
#   flat__independent        flat MLP       independent    MaskPPO on a flat state
#   deep_sets__independent   DeepSets       independent    DeepSets-PPO
#   hetero__independent      hetero graph   independent    state-GNN PPO
#   v2_tm10 (control)        hetero graph   complete_gcn   GraphPPO (proposed)
#
# The control is not retrained: `v2_tm10` from `run_graphppo_v2_campaign.sh` is
# already this configuration.  `independent` scores each candidate on its own,
# so the three baselines isolate what the action-graph interaction adds on top
# of whatever the state encoder already provides.
#
# DeepSets was implemented and unit-tested during the earlier campaign but never
# trained to completion for lack of compute; this closes that gap.
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-2000000}"
NUM_ENVS="${NUM_ENVS:-16}"
WORKERS="${WORKERS:-5}"
TORCH_THREADS="${TORCH_THREADS:-2}"
OUTPUT="${OUTPUT:-results/canonical/learned_baselines}"
LOGS="${LOGS:-results/canonical/logs_learned_baselines}"
CURRICULUM="${CURRICULUM:-configs/curriculum/energy_ramp.json}"
GPU="${GPU:-1}"

mkdir -p "$LOGS"

launch() {
  local name="$1"; local encoder="$2"; local head="$3"; shift 3
  echo "launching $name"
  PYTHONPATH=. MPLCONFIGDIR=/tmp/evrp-matplotlib OMP_NUM_THREADS="$TORCH_THREADS" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    nohup .venv/bin/python scripts/training/train_canonical_ppo.py \
      --state-encoder "$encoder" --action-head "$head" \
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

launch flat__independent      flat         independent --seed 0
launch deep_sets__independent deep_sets    independent --seed 0
launch hetero__independent    hetero_graph independent --seed 0

wait
echo "learned baseline family complete"
