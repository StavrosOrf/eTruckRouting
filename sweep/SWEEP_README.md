# Hyperparameter Sweep for EVPR TD3-GNN

This directory contains scripts and configurations for running hyperparameter sweeps on SLURM clusters using Weights & Biases.

## Files

- **sweep_config.yaml** - Bayesian optimization sweep (recommended for finding optimal hyperparameters)
- **sweep_config_grid.yaml** - Grid search over focused parameter space
- **sweep_config_random.yaml** - Random search (good for initial exploration)
- **launch_sweep.sh** - Main script to initialize and launch a sweep
- **run_sweep.sh** - SLURM batch script for GPU jobs
- **run_sweep_cpu.sh** - SLURM batch script for CPU-only jobs

## Quick Start

### 1. Configure SLURM Scripts

Edit `run_sweep.sh` or `run_sweep_cpu.sh` to match your cluster configuration:

```bash
#SBATCH --partition=gpu        # Your GPU partition name
#SBATCH --gres=gpu:1           # GPU type (e.g., gpu:v100:1)
#SBATCH --time=24:00:00        # Adjust based on expected runtime
#SBATCH --mem=16G              # Memory per job

# Load appropriate modules for your cluster
module load python/3.9
module load cuda/11.8
```

### 2. Launch a Sweep

```bash
# Make scripts executable
chmod +x launch_sweep.sh run_sweep.sh run_sweep_cpu.sh

# Launch sweep with default configuration (Bayesian optimization, GPU)
./launch_sweep.sh

# Or launch with specific config
wandb sweep sweep_config_random.yaml  # Get sweep ID
sbatch --array=0-9 run_sweep.sh <sweep_id>
```

### 3. Monitor Progress

The launch script will output monitoring commands and URLs:

```bash
# View W&B dashboard
https://wandb.ai/stavrosorf/evpr-td3-gnn-sweep/sweeps/<sweep_id>

# Check SLURM job status
squeue -u $USER

# View logs
tail -f logs/sweep_<job_id>_0.out

# Cancel jobs
scancel <job_id>
```

## Sweep Configurations

### Bayesian Optimization (sweep_config.yaml)
- **Method**: Gaussian Process-based optimization
- **Best for**: Finding optimal hyperparameters with fewer runs
- **Features**: Early termination with Hyperband
- **Parameters**: Explores ~20 key hyperparameters

### Grid Search (sweep_config_grid.yaml)
- **Method**: Exhaustive search over discrete values
- **Best for**: Systematic exploration of key parameters
- **Parameters**: Focused set with 2-3 values each
- **Total runs**: ~100-200 combinations

### Random Search (sweep_config_random.yaml)
- **Method**: Random sampling
- **Best for**: Initial broad exploration
- **Run cap**: 50 combinations
- **Parameters**: Wide ranges for all hyperparameters

## Adding More Agents

You can dynamically add more sweep agents to speed up the search:

```bash
# Add 5 more agents to existing sweep
sbatch --array=0-4 run_sweep.sh <sweep_id>
```

## Customizing Sweeps

### Modify Hyperparameter Ranges

Edit the sweep config YAML files:

```yaml
parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-4
    max: 1e-3
  
  batch-size:
    values: [16, 32, 64, 128]
```

### Change Optimization Strategy

```yaml
method: bayes  # Options: grid, random, bayes

metric:
  name: eval/mean_reward  # Metric to optimize
  goal: maximize          # or minimize
```

### Add Early Termination

```yaml
early_terminate:
  type: hyperband
  min_iter: 10   # Minimum iterations before termination
  eta: 3         # Downsampling rate
  s: 2           # Number of brackets
```

## Resource Management

### GPU Jobs
```bash
# Adjust in run_sweep.sh
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
```

### CPU Jobs
```bash
# Use run_sweep_cpu.sh
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
```

## Best Practices

1. **Start small**: Run a few trials manually before launching full sweep
2. **Check logs**: Monitor first few jobs to ensure proper configuration
3. **Resource limits**: Set appropriate time limits to avoid wasting resources
4. **Early termination**: Use Hyperband for faster convergence
5. **Seed variation**: Include multiple seeds for robust evaluation

## Troubleshooting

### Jobs fail immediately
- Check SLURM partition names
- Verify module loads
- Test with single job first: `sbatch --array=0 run_sweep.sh <sweep_id>`

### Out of memory errors
- Increase `--mem` in SLURM script
- Reduce `--batch-size` in sweep config
- Use smaller `--gnn-hidden-dim`

### Wandb authentication issues
- Run `wandb login` before launching
- Check `~/.netrc` has correct credentials
- Set `WANDB_API_KEY` environment variable in SLURM script

### Logs not appearing
- Ensure `logs/` directory exists
- Check SLURM output/error paths
- Verify file permissions

## Example Workflow

```bash
# 1. Test single run locally
python train.py --num-stops 3 --max-timesteps 10000

# 2. Test single SLURM job
sbatch --array=0 run_sweep_cpu.sh <sweep_id>

# 3. Check logs
tail -f logs/sweep_*_0.out

# 4. Launch full sweep
./launch_sweep.sh

# 5. Monitor on W&B
# Visit the URL printed by launch_sweep.sh

# 6. Add more agents if needed
sbatch --array=0-9 run_sweep.sh <sweep_id>
```

## Output

Results are logged to:
- **W&B Dashboard**: Real-time metrics, visualizations, comparisons
- **SLURM logs**: `logs/sweep_<job_id>_<task_id>.{out,err}`
- **Model checkpoints**: `saved_models/<exp_name>/`
- **Sweep info**: `logs/sweep_<sweep_id>_info.txt`

## Advanced: Multi-Stage Sweeps

For complex hyperparameter optimization:

1. **Stage 1**: Random search (50 runs) - broad exploration
2. **Stage 2**: Bayesian optimization on top-10 region - refinement
3. **Stage 3**: Grid search around best config - final tuning

```bash
# Stage 1
wandb sweep sweep_config_random.yaml
sbatch --array=0-19 run_sweep.sh <sweep_id_1>

# After Stage 1 completes, update sweep_config.yaml with narrower ranges

# Stage 2
wandb sweep sweep_config.yaml
sbatch --array=0-9 run_sweep.sh <sweep_id_2>
```
