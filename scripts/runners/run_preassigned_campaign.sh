#!/usr/bin/env bash
# Do the three revision findings hold in the eTFRP-style setting?
#
# The main benchmark pre-assigns customers to trucks, so the policy decides
# sequencing, charging and routing but not assignment. That setting is now
# expressible inside the canonical stack (problem.assignment=preassigned), which
# means the mask ablation and the seed study can be run there with the *same*
# architecture, observation, mask, curriculum, reward and artifact contract as
# the joint study -- the controlled comparison the original PPO-vs-MaskPPO
# figures could not provide, because those arms differ in architecture as well
# as in masking.
#
#   pre_masked_s0/s1     hard feasibility mask
#   pre_unmasked_s0/s1   structural mask; infeasible actions strand the truck
set -euo pipefail
cd "$(dirname "$0")/../.."

TIMESTEPS="${TIMESTEPS:-2000000}"
LOGS="${LOGS:-results/canonical/logs_preassigned}"
CONFIG="${CONFIG:-EVRoutingEnv/config_files/config_joint_preassigned.yaml}"
mkdir -p "$LOGS"

launch() {
  local name="$1"; shift
  PYTHONPATH=. MPLCONFIGDIR=/tmp/evrp-matplotlib OMP_NUM_THREADS=2 \
    CUDA_VISIBLE_DEVICES=1 setsid nohup .venv/bin/python \
    scripts/training/train_canonical_ppo.py \
      --config "$CONFIG" \
      --state-encoder hetero_graph --action-head complete_gcn \
      --total-timesteps "$TIMESTEPS" --num-envs 16 --rollout-steps 128 \
      --rollout-workers 5 \
      --learning-rate 3e-4 --entropy-coefficient 0.02 \
      --validation-scenarios 40 --validate-every 50 \
      --selection-objective travel_time \
      --curriculum configs/curriculum/energy_ramp.json \
      --time-multiplier 10 \
      --device cuda:0 --torch-threads 2 \
      --output results/canonical/preassigned --run-name "$name" \
      "$@" > "$LOGS/${name}.log" 2>&1 < /dev/null &
  echo "launched $name"
}

launch pre_masked_s0   --seed 0
launch pre_masked_s1   --seed 1
launch pre_unmasked_s0 --seed 0 --policy-action-mask structural --invalid-action-mode terminate
launch pre_unmasked_s1 --seed 1 --policy-action-mask structural --invalid-action-mode terminate
