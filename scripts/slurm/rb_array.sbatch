#!/bin/bash
# RoutingBlocks baseline plans, one scenario seed per array task, validated by replay.
#
#   sbatch --array=0-19 scripts/slurm/rb_array.sbatch 50 100 test results/rb_alns/ca_50x100
#   sbatch --array=0-19 scripts/slurm/rb_array.sbatch 128 200 test results/rb_alns/ca_128x200 1800
#
# One A100 share is 18 cores; the search uses them as parallel workers that share
# a route pool (no licence, so no array throttle is needed). Needs the solver
# interpreter once: bash scripts/setup_rb_env.sh
#
# Arguments: TRUCKS CUSTOMERS SPLIT OUT_DIR [TIME_LIMIT_SECONDS] [CONFIG] [DRIVER ARGS...]
# SPLIT is train/validation/test, or "seeds=7,11,13" for explicit scenario seeds.
# DRIVER ARGS go to scripts.evaluation.solve_routingblocks unchanged, e.g.
#   ... 1800 EVRoutingEnv/config_files/config_joint.yaml --bound-from results/branch_and_price/ca_128x200
# (Snellius does not export the submitting shell's environment into jobs.)
#SBATCH --job-name=rb_alns
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rb_alns_%A_%a.out

set -euo pipefail
TRUCKS=${1:?trucks}
CUSTOMERS=${2:?customers}
SPLIT=${3:?split}
OUT=${4:?output directory}
LIMIT=${5:-1200}
CONFIG=${6:-EVRoutingEnv/config_files/config_joint.yaml}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source /home/sorfanoudakis/EVRP/.venv/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1

if [[ "$SPLIT" == seeds=* ]]; then
    read -ra SEEDS <<< "${SPLIT#seeds=}"; SEEDS=(${SEEDS[@]//,/ })
    SELECT=(--seeds "${SEEDS[@]}" --index "${SLURM_ARRAY_TASK_ID:-0}")
else
    SELECT=(--split "$SPLIT" --count "${SLURM_ARRAY_TASK_COUNT:-1}" --index "${SLURM_ARRAY_TASK_ID:-0}")
fi
python -m scripts.evaluation.solve_routingblocks \
    --config "$CONFIG" --trucks "$TRUCKS" --customers "$CUSTOMERS" "${SELECT[@]}" \
    --time-limit "$LIMIT" --workers "${SLURM_CPUS_PER_TASK:-1}" --out "$OUT" "${@:7}"
