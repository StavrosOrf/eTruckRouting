#!/usr/bin/env bash
# Second training batch: seed replication, the attention baseline, and the two
# architecture ablations that need their own runs.
#
#   seed1_A / seed2_A   stage A of the travel ladder at seeds 1 and 2. Document
#                       10 ran the whole A->B->C ladder at seed 0 only, so the
#                       headline carries unquantified seed variance (R1.7).
#                       These replicate the *procedure*, with the stage
#                       hyperparameters frozen rather than re-selected.
#   attention           the constructive attention baseline (R1.6, R2.8): the
#                       transformer state encoder against the same action head,
#                       budget, curriculum, and reward as the proposed model.
#   ablate_pooling      the action head loses the pooled fleet embedding, so a
#                       candidate is scored from its own row alone (E3).
#   ablate_edges        every pairwise relation is blanked, which is the
#                       relation-type ablation in its strongest form (E3).
#
# Every arm matches `v2_tm10` from run_graphppo_v2_campaign.sh in budget,
# curriculum, reward, and optimizer settings, so each is read against that run.
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-2000000}"
NUM_ENVS="${NUM_ENVS:-16}"
WORKERS="${WORKERS:-5}"
TORCH_THREADS="${TORCH_THREADS:-2}"
LOGS="${LOGS:-results/canonical/logs_batch2}"
CURRICULUM="${CURRICULUM:-configs/curriculum/energy_ramp.json}"
GPU="${GPU:-1}"

mkdir -p "$LOGS"

launch() {
  local name="$1"; local output="$2"; local encoder="$3"; local head="$4"; shift 4
  echo "launching $name -> $output"
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
      --output "$output" --run-name "$name" \
      "$@" > "$LOGS/${name}.log" 2>&1 &
}

launch seed1_A        results/canonical/graphppo_seeds   hetero_graph complete_gcn --seed 1
launch seed2_A        results/canonical/graphppo_seeds   hetero_graph complete_gcn --seed 2
launch attention      results/canonical/learned_baselines attention   complete_gcn --seed 0
launch ablate_pooling results/canonical/ablations        hetero_graph complete_gcn --seed 0 \
  --ablate-state-pooling
launch ablate_edges   results/canonical/ablations        hetero_graph complete_gcn --seed 0 \
  --ablate-features edges

wait
echo "batch 2 complete"
