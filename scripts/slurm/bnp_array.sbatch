#!/bin/bash
# Exact branch-and-price certificates, one scenario seed per array task.
#
#   sbatch --array=0-19%2 scripts/slurm/bnp_array.sbatch 50 100 test results/branch_and_price/ca_50x100
#   sbatch --array=0-19%2 --time=48:00:00 scripts/slurm/bnp_array.sbatch 128 200 test results/branch_and_price/ca_128x200
#
# Runs on the GPU partition (one A100 = 18 cores, ~120 GB) although the solver
# is CPU-only: GPU allocations are plentiful on this account. Use
# "--partition=cbuild --gpus=0 --cpus-per-task=16 --mem=64G" to run on CPU nodes.
#
# Keep "%2": the Gurobi WLS licence admits only two concurrent sessions and
# every solver process holds one (Gurobi error 10030 otherwise), counting
# solves running anywhere else under the same licence.
#
# Arguments: TRUCKS CUSTOMERS SPLIT OUT_DIR [TIME_LIMIT_SECONDS] [CONFIG] [DRIVER ARGS...]
# SPLIT is train/validation/test, or "seeds=7,11,13" for explicit scenario seeds.
# Everything after CONFIG goes to the driver unchanged, e.g.
#   sbatch ... bnp_array.sbatch 50 100 test OUT 84000 EVRoutingEnv/config_files/config_joint.yaml \
#       --alns-seconds 120 --solver-arg lns_time 2400
# (Snellius does not export the submitting shell's environment into jobs, so
# settings must travel as arguments.)
# Run scripts/slurm/bnp_precompile.sbatch once first so tasks do not race to
# precompile the Julia package.
#SBATCH --job-name=bnp
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
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
    --time-limit "$LIMIT" --threads "${SLURM_CPUS_PER_TASK:-1}" --out "$OUT" "${@:7}"
