#!/bin/bash
# Exact branch-and-price certificates, one scenario seed per array task.
#
#   sbatch --array=0-19 scripts/slurm/bnp_array.sbatch 50 100 test results/branch_and_price/ca_50x100
#   sbatch --array=0-19 --time=48:00:00 scripts/slurm/bnp_array.sbatch 128 200 test results/branch_and_price/ca_128x200
#
# Arguments: TRUCKS CUSTOMERS SPLIT OUT_DIR [TIME_LIMIT_SECONDS] [CONFIG]
# SPLIT is train/val/test, or "seeds=7,11,13" for explicit scenario seeds.
# Run scripts/slurm/bnp_precompile.sbatch once first so tasks do not race to
# precompile the Julia package.
#SBATCH --job-name=bnp
#SBATCH --partition=cbuild
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bnp_%A_%a.out

set -euo pipefail
TRUCKS=${1:?trucks}
CUSTOMERS=${2:?customers}
SPLIT=${3:?split}
OUT=${4:?output directory}
LIMIT=${5:-82800}
CONFIG=${6:-EVRoutingEnv/config_files/config_joint.yaml}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source /etc/profile.d/modules.sh
module load 2024 Julia/1.10.4-linux-x86_64
source /home/sorfanoudakis/EVRP/.venv/bin/activate
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
export PYTHONPATH=.

if [[ "$SPLIT" == seeds=* ]]; then
    read -ra SEEDS <<< "${SPLIT#seeds=}"; SEEDS=(${SEEDS[@]//,/ })
    SELECT=(--seeds "${SEEDS[@]}" --index "${SLURM_ARRAY_TASK_ID:-0}")
else
    SELECT=(--split "$SPLIT" --count "${SLURM_ARRAY_TASK_COUNT:-1}" --index "${SLURM_ARRAY_TASK_ID:-0}")
fi
python -m scripts.evaluation.solve_branch_and_price \
    --config "$CONFIG" --trucks "$TRUCKS" --customers "$CUSTOMERS" "${SELECT[@]}" \
    --time-limit "$LIMIT" --threads "${SLURM_CPUS_PER_TASK:-1}" --out "$OUT"
