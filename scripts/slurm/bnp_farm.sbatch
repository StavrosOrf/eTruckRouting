#!/bin/bash
# One allocation working for a branch-and-price farm (scripts/evaluation/bnp_farm.py).
#
#   python -m scripts.evaluation.bnp_farm init --farm FARM --instance OUT/instances/seed_1.json \
#       --split OUT/split_1.json --skip-closed OUT/subtrees_1 --incumbents OUT/incumbents_1
#   sbatch scripts/slurm/bnp_farm.sbatch FARM            # as many allocations as there is room for
#   python -m scripts.evaluation.bnp_farm status --farm FARM
#
# Arguments: FARM [WORKERS] [WORKER_SECONDS] [PRIMAL_WORKERS] [RUNNER] [SOLVER ARGS...].
# WORKERS defaults to the allocation's CPUs less PRIMAL_WORKERS; every worker is
# one single-threaded solver process (HiGHS master, about 1-2 GB at 50 x 100).
# The Slurm environment is not passed on here, so the runner (default: the
# package's run_bp.sh) is an argument. Defaults to a gpu_mig slice, which is
# usually free when the larger GPU partitions are not; override on the command
# line, e.g. "sbatch --partition=cbuild --gpus=0 --cpus-per-task=32 --mem=120G".
#SBATCH --job-name=bnp-farm
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bnp_farm_%j.out

set -uo pipefail
FARM=${1:?farm directory}
PRIMAL=${4:-0}
WORKERS=${2:-$(( ${SLURM_CPUS_PER_TASK:-1} - PRIMAL ))}
WORKER_TIME=${3:-7200}
RUNNER=${5:-EVRoutingEnv/baselines/optimization/branch_and_price/julia/run_bp.sh}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source /etc/profile.d/modules.sh
module load 2024 Julia/1.10.4-linux-x86_64
source /home/sorfanoudakis/EVRP/.venv/bin/activate
export PYTHONPATH=.
python -m scripts.evaluation.bnp_farm work --farm "$FARM" --workers "$WORKERS" --worker-time "$WORKER_TIME" \
    --primal-workers "$PRIMAL" --runner "$RUNNER" -- "${@:6}"
