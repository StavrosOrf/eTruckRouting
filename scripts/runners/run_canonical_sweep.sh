#!/usr/bin/env bash
# Launch the canonical architecture sweep: every state encoder crossed with
# every approved action head, on an identical budget, identical seeds, and the
# identical cached demonstration set. The winner is selected on validation only.
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-800000}"
NUM_ENVS="${NUM_ENVS:-16}"
OUTPUT="${OUTPUT:-results/canonical/training}"
LOGS="${LOGS:-results/canonical/logs}"
SEED="${SEED:-0}"
DEMOS="${DEMOS:-results/canonical/demonstrations/mpc.npz}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-40}"
LR="${LR:-1e-4}"
ENTROPY="${ENTROPY:-0.005}"
VALIDATION_SCENARIOS="${VALIDATION_SCENARIOS:-40}"
VALIDATE_EVERY="${VALIDATE_EVERY:-25}"
TORCH_THREADS="${TORCH_THREADS:-6}"
ENCODERS="${ENCODERS:-flat deep_sets hetero_graph}"
HEADS="${HEADS:-independent complete_gcn self_attention}"

mkdir -p "$LOGS"

for encoder in $ENCODERS; do
  for head in $HEADS; do
    name="${encoder}__${head}"
    echo "launching $name"
    PYTHONPATH=. MPLCONFIGDIR=/tmp/evrp-matplotlib OMP_NUM_THREADS="$TORCH_THREADS" \
      nohup .venv/bin/python scripts/training/train_canonical_ppo.py \
        --state-encoder "$encoder" \
        --action-head "$head" \
        --total-timesteps "$TIMESTEPS" \
        --num-envs "$NUM_ENVS" \
        --learning-rate "$LR" \
        --entropy-coefficient "$ENTROPY" \
        --demonstrations "$DEMOS" \
        --pretrain-epochs "$PRETRAIN_EPOCHS" \
        --validation-scenarios "$VALIDATION_SCENARIOS" \
        --validate-every "$VALIDATE_EVERY" \
        --seed "$SEED" \
        --output "$OUTPUT" \
        --torch-threads "$TORCH_THREADS" \
        --run-name "$name" \
        > "$LOGS/${name}.log" 2>&1 &
  done
done

wait
echo "sweep complete"
