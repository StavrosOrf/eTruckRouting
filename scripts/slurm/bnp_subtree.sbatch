#!/bin/bash
# One subtree of a split branch-and-price search per array task.
#
#   # 1. split run: root, then branching until 32 nodes are open
#   sbatch scripts/slurm/bnp_array.sbatch 50 100 seeds=1 OUT 84000 EVRoutingEnv/config_files/config_joint.yaml \
#       --split 32 --solver-arg lp_solver highs
#   # 2. the subtrees, all at once (they share incumbents through OUT/incumbents_<SEED>)
#   sbatch --array=0-31 scripts/slurm/bnp_subtree.sbatch OUT 1 [TIME_LIMIT_SECONDS] [SOLVER ARGS...]
#   # 3. combine (any time; unfinished subtrees count with their split bound)
#   python -m scripts.evaluation.combine_bnp_subtrees --out OUT --seed 1 --trucks 50 --customers 100
#
# Reads OUT/instances/seed_<SEED>.json and OUT/split_<SEED>.json from the split
# run, and writes OUT/subtrees_<SEED>/sub_<i>.result.json plus its log. The
# master is solved with HiGHS unless SOLVER ARGS say "--lp_solver gurobi": the
# Gurobi WLS licence admits only two sessions. Runs on the GPU partition for
# the reasons given in bnp_array.sbatch.
#SBATCH --job-name=bnp-sub
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bnp_sub_%A_%a.out

set -euo pipefail
OUT=${1:?output directory of the split run}
SEED=${2:?scenario seed}
LIMIT=${3:-82800}
I=${SLURM_ARRAY_TASK_ID:-0}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source /etc/profile.d/modules.sh
module load 2024 Julia/1.10.4-linux-x86_64
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
export EVRP_BP_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$OUT/subtrees_$SEED"
EVRoutingEnv/baselines/optimization/branch_and_price/julia/run_bp.sh \
    "$OUT/instances/seed_$SEED.json" "$OUT/subtrees_$SEED/sub_$I.result.json" \
    --lp_solver highs --time_limit "$LIMIT" --threads "$EVRP_BP_THREADS" --verbose true \
    --subtree_file "$OUT/split_$SEED.json" --subtree_index "$I" \
    --incumbent_dir "$OUT/incumbents_$SEED" --incumbent_tag "sub_$I" "${@:4}" \
    > "$OUT/subtrees_$SEED/sub_$I.log" 2>&1
