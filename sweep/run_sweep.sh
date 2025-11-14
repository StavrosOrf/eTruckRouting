#!/bin/bash
#SBATCH --job-name=evpr-sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-9  # Number of sweep agents to run in parallel

# Load required modules (adjust based on your cluster)
# module load python/3.9
# module load cuda/11.8

# Set up environment
cd /home/sorfanouda/EVPR
source .venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Starting time: $(date)"
echo "=========================================="

# Set CUDA device (if using GPU)
export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID

# Get the sweep path from command line argument or environment variable
# Format: entity/project/sweep_id
SWEEP_PATH=${1:-$WANDB_SWEEP_PATH}

if [ -z "$SWEEP_PATH" ]; then
    echo "ERROR: No sweep path provided!"
    echo "Usage: sbatch run_sweep.sh <entity>/<project>/<sweep_id>"
    echo "   or: WANDB_SWEEP_PATH=<entity>/<project>/<sweep_id> sbatch run_sweep.sh"
    exit 1
fi

echo "Running sweep agent for sweep: $SWEEP_PATH"
echo "=========================================="

# Run the wandb agent
wandb agent $SWEEP_PATH

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
