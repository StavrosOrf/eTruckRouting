#!/bin/bash
#SBATCH --account=tdsei17344
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@240
set -euo pipefail
campaign_dir="$1"
transition_budget="$2"
export PYTHONPATH="$campaign_dir/source"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# The supervisor preserves SLURM GPU visibility and forwards signals to every
# trainer. Each trainer writes a complete checkpoint at its next update.
/gpfs/home2/sorfanoudakis/EVRP/.venv/bin/python \
  "$campaign_dir/source/scripts/campaigns/california_graphppo.py" train \
  --directory "$campaign_dir" --bundle "$SLURM_ARRAY_TASK_ID" \
  --budget "$transition_budget" &
supervisor_pid=$!
trap 'kill -USR1 "$supervisor_pid" 2>/dev/null || true' USR1 TERM
while true; do
  set +e
  wait "$supervisor_pid"
  supervisor_status=$?
  set -e
  if ! kill -0 "$supervisor_pid" 2>/dev/null; then
    exit "$supervisor_status"
  fi
done
