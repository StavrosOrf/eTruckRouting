#!/bin/bash
# Example training script for curriculum learning with mixed strategy

python scripts/training/train_curriculum.py \
    --config EVRoutingEnv/config_files/config.yaml \
    --curriculum-config EVRoutingEnv/config_files/curriculum_config_mixed.json \
    --max-timesteps 2000000 \
    --ppo-steps-per-update 512 \
    --ppo-epochs 10 \
    --ppo-minibatch-size 256 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --ppo-clip 0.2 \
    --ppo-entropy-coef 0.01 \
    --ppo-value-coef 0.5 \
    --gnn-hidden-dim 64 \
    --mlp-hidden-dim 256 \
    --actor-gcn-layers 3 \
    --lr 3e-4 \
    --eval-freq 1000 \
    --eval-episodes 5 \
    --eval-configs "1,3,5,7,10" \
    --wandb-project evpr-curriculum \
    --wandb-entity stavrosorf \
    --exp-name curriculum_mixed \
    --seed 42
