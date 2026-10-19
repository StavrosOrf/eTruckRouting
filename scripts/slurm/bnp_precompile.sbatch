#!/bin/bash
# Instantiate and precompile the EVRPBranchPrice Julia package once.
#   sbatch scripts/slurm/bnp_precompile.sbatch
#SBATCH --job-name=bnp-precompile
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=00:45:00
#SBATCH --output=logs/bnp_precompile_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source /etc/profile.d/modules.sh
module load 2024 Julia/1.10.4-linux-x86_64
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
julia --project=EVRoutingEnv/baselines/optimization/branch_and_price/julia/EVRPBranchPrice \
    -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using EVRPBranchPrice; println("ready")'
