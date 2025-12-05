# Curriculum Learning for PPO-Variable Action GNN

This guide explains how to train PPO-variable agents on episodes with varying numbers of trucks and stops using curriculum learning.

## Overview

The curriculum learning system enables training a single agent that can generalize across different problem sizes. Three strategies are available:

1. **Uniform Random**: Uniformly sample from ranges every episode
2. **Staged Curriculum**: Gradually increase difficulty through stages
3. **Mixed Curriculum**: Sample from multiple difficulty levels with specified weights

## Quick Start

### Basic Training with Uniform Random Strategy

```bash
python scripts/training/train_curriculum.py \
    --curriculum-strategy uniform \
    --truck-range 3 8 \
    --stop-range 3 8 \
    --seed 42
```

This will sample uniformly from 3-8 trucks and 3-8 stops every episode.

### Using Pre-configured Strategies

You can use the pre-defined curriculum configurations:

```bash
# Uniform random sampling
python scripts/training/train_curriculum.py \
    --curriculum-config EVRoutingEnv/config_files/curriculum_config_uniform.json

# Staged curriculum (easy → hard)
python scripts/training/train_curriculum.py \
    --curriculum-config EVRoutingEnv/config_files/curriculum_config_staged.json

# Mixed difficulty levels
python scripts/training/train_curriculum.py \
    --curriculum-config EVRoutingEnv/config_files/curriculum_config_mixed.json
```

### Using Example Scripts

Pre-configured bash scripts are provided:

```bash
# Make scripts executable
chmod +x scripts/training/run_curriculum_*.sh

# Run uniform strategy
./scripts/training/run_curriculum_uniform.sh

# Run staged strategy
./scripts/training/run_curriculum_staged.sh

# Run mixed strategy
./scripts/training/run_curriculum_mixed.sh
```

## Curriculum Strategies

### 1. Uniform Random Strategy

**When to use**: Good baseline for general robustness. Agent sees diverse problems from the start.

**Configuration**:
```json
{
  "strategy": "uniform",
  "truck_range": [3, 8],
  "stop_range": [3, 8]
}
```

**Command line**:
```bash
python scripts/training/train_curriculum.py \
    --curriculum-strategy uniform \
    --truck-range 3 8 \
    --stop-range 3 8
```

**Pros**:
- Simple and robust
- Agent exposed to full range early
- No hyperparameters to tune

**Cons**:
- May struggle initially with hard problems
- No progressive difficulty increase

### 2. Staged Curriculum Strategy

**When to use**: When you want gradual difficulty increase. Best for stable learning from easy to hard problems.

**Configuration**:
```json
{
  "strategy": "staged",
  "stages": [
    {
      "episodes": 500,
      "truck_range": [1, 3],
      "stop_range": [3, 5]
    },
    {
      "episodes": 1000,
      "truck_range": [3, 6],
      "stop_range": [5, 8]
    },
    {
      "episodes": -1,
      "truck_range": [5, 10],
      "stop_range": [5, 10]
    }
  ]
}
```

**Pros**:
- Stable learning progression
- Agent masters easy problems first
- Can prevent early training collapse

**Cons**:
- Requires careful stage design
- May overfit to early stages
- Takes longer to see full range

### 3. Mixed Curriculum Strategy

**When to use**: When you want to emphasize certain difficulty levels while still exposing the agent to all levels.

**Configuration**:
```json
{
  "strategy": "mixed",
  "difficulty_levels": [
    {
      "truck_range": [1, 3],
      "stop_range": [3, 5],
      "weight": 0.3
    },
    {
      "truck_range": [4, 7],
      "stop_range": [5, 8],
      "weight": 0.5
    },
    {
      "truck_range": [8, 10],
      "stop_range": [8, 10],
      "weight": 0.2
    }
  ]
}
```

**Pros**:
- Balanced exposure to all difficulties
- Can weight towards important sizes
- Stable gradients from easier problems

**Cons**:
- Requires tuning weights
- May under-sample hard problems

## Key Training Arguments

### Curriculum Parameters
- `--curriculum-strategy`: Choose 'uniform', 'staged', or 'mixed'
- `--truck-range MIN MAX`: Range of truck numbers (e.g., `3 8`)
- `--stop-range MIN MAX`: Range of stop numbers (e.g., `3 8`)
- `--curriculum-config PATH`: Load detailed config from JSON

### Training Parameters
- `--max-timesteps`: Total training steps (recommend 2M+ for curriculum)
- `--ppo-steps-per-update`: Steps before PPO update (recommend 512)
- `--eval-freq`: How often to evaluate (recommend 1000)
- `--eval-episodes`: Episodes per evaluation config (recommend 5)
- `--eval-configs`: Comma-separated truck counts to evaluate on

### PPO Hyperparameters
- `--gamma`: Discount factor (default: 0.99)
- `--gae-lambda`: GAE lambda (default: 0.95)
- `--ppo-clip`: Clipping coefficient (default: 0.2)
- `--ppo-entropy-coef`: Entropy bonus (default: 0.01)
- `--ppo-value-coef`: Value loss coefficient (default: 0.5)
- `--lr`: Learning rate (default: 3e-4)

### Network Architecture
- `--gnn-hidden-dim`: GNN hidden dimension (default: 64)
- `--mlp-hidden-dim`: MLP hidden dimension (default: 256)
- `--actor-gcn-layers`: Number of GCN layers (default: 3)

## Evaluation

The training script automatically evaluates on multiple fixed problem sizes:

```bash
--eval-configs "1,3,5,7,10"  # Evaluates on 1, 3, 5, 7, and 10 trucks
```

For each configuration, it logs:
- Mean reward and standard deviation
- Success rate (% of episodes completing all deliveries)
- Mean episode length (steps)
- Mean episode time (simulation hours)

Results are logged to Weights & Biases under keys like:
- `eval/3t_5s/mean_reward`
- `eval/7t_5s/success_rate`
- `eval/aggregate_reward` (average across all configs)

## Monitoring Training

### Wandb Logging

All metrics are logged to Weights & Biases:

**Training metrics** (per episode):
- `train/episode_reward`
- `train/success`
- `train/num_trucks` (current episode size)
- `train/num_stops` (current episode size)

**Evaluation metrics** (periodic):
- `eval/{N}t_{M}s/mean_reward` (per configuration)
- `eval/{N}t_{M}s/success_rate` (per configuration)
- `eval/aggregate_reward` (average across all)
- `eval/aggregate_success` (average across all)

**Curriculum statistics**:
- `curriculum/total_episodes`
- `curriculum/unique_configs`

### Model Saving

Models are saved to `saved_models/{exp_name}/`:
- `ppo_curriculum_best`: Best model based on aggregate success rate
- `ppo_curriculum_final`: Final model after training
- `ppo_network_config.json`: Network configuration

## Tips and Best Practices

### 1. Start with Narrow Ranges

Begin with a small range to validate the approach:
```bash
--truck-range 3 5 --stop-range 3 5
```

Then gradually expand:
```bash
--truck-range 3 8 --stop-range 3 8
```

### 2. Increase Training Time

Curriculum learning requires more training than fixed-size:
- Fixed size: 1M timesteps
- Curriculum: 2-3M timesteps

### 3. Use Larger Buffers

Collect more diverse experiences before updating:
```bash
--ppo-steps-per-update 512  # or even 1024
```

### 4. Monitor Per-Size Performance

Check Wandb to identify which problem sizes are struggling. You can then:
- Adjust curriculum weights to emphasize weak sizes
- Add stages focusing on problematic ranges

### 5. Evaluation Strategy

Evaluate on fixed sizes that span your training range:
```bash
--eval-configs "1,3,5,7,10"  # Good coverage
```

### 6. Seed Management

Use different seeds for reproducibility:
```bash
--seed 42  # or 0, 1, 2, etc.
```

## Creating Custom Curriculum Configs

Create a JSON file with your desired curriculum:

```json
{
  "strategy": "staged",
  "stages": [
    {
      "episodes": 1000,
      "truck_range": [2, 4],
      "stop_range": [3, 6],
      "description": "Stage 1: Warm-up"
    },
    {
      "episodes": -1,
      "truck_range": [3, 10],
      "stop_range": [3, 10],
      "description": "Stage 2: Full training"
    }
  ]
}
```

Then use it:
```bash
python scripts/training/train_curriculum.py \
    --curriculum-config path/to/your/config.json
```

## Troubleshooting

### Issue: Training is unstable

**Solution**: 
- Reduce the problem size range
- Use staged curriculum starting with easier problems
- Increase `--ppo-steps-per-update` to 1024

### Issue: Poor performance on large problems

**Solution**:
- Use mixed curriculum with higher weight on hard problems
- Extend training time
- Increase network capacity (`--gnn-hidden-dim 128`)

### Issue: Agent overfits to certain sizes

**Solution**:
- Use uniform random strategy
- Check evaluation metrics to identify overfitted sizes
- Adjust curriculum to under-sample those sizes

### Issue: Out of memory errors

**Solution**:
- Reduce `--ppo-minibatch-size`
- Reduce `--ppo-steps-per-update`
- Limit maximum problem size in range

## Example Full Training Command

```bash
python scripts/training/train_curriculum.py \
    --config EVRoutingEnv/config_files/config.yaml \
    --curriculum-strategy uniform \
    --truck-range 3 10 \
    --stop-range 3 10 \
    --max-timesteps 3000000 \
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
    --exp-name my_curriculum_experiment \
    --group-name curriculum_3-10trucks \
    --seed 42 \
    --verbose
```

## References

For more details on the implementation:
- Environment wrapper: `EVRoutingEnv/models/curriculum_env.py`
- Training script: `scripts/training/train_curriculum.py`
- Config examples: `EVRoutingEnv/config_files/curriculum_config_*.json`
