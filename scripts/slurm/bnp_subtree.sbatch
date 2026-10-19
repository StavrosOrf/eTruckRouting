#!/bin/bash
# Subtrees of a split branch-and-price search, PER_TASK of them per array task.
#
#   # 1. split run: root, then branching until 32 nodes are open
#   sbatch scripts/slurm/bnp_array.sbatch 50 100 seeds=1 OUT 84000 EVRoutingEnv/config_files/config_joint.yaml \
#       --split-nodes 32 --solver-arg lp_solver highs
#   # 2. the subtrees, all at once: 4 tasks x 8 processes (they share incumbents through OUT/incumbents_<SEED>)
#   sbatch --array=0-3 scripts/slurm/bnp_subtree.sbatch OUT 1 82800 8 [SOLVER ARGS...]
#   # 3. combine (any time; unfinished subtrees count with their split bound)
#   python -m scripts.evaluation.combine_bnp_subtrees --out OUT --seed 1 --trucks 50 --customers 100
#
# Arguments: OUT SEED [TIME_LIMIT_SECONDS] [PER_TASK] [SOLVER ARGS...]. Task t
# searches subtrees t*PER_TASK .. t*PER_TASK+PER_TASK-1 of OUT/split_<SEED>.json
# concurrently, each with CPUS/PER_TASK threads: a solver process keeps only one
# or two cores busy (the bidirectional pricing halves) and needs 2-4 GB at
# 50 x 100, so one allocation holds several. Results go to
# OUT/subtrees_<SEED>/sub_<i>.result.json, with a log next to each.
#
# The master is solved with HiGHS unless SOLVER ARGS say "--lp_solver gurobi":
# the Gurobi WLS licence admits only two sessions. Runs on the GPU partition
# for the reasons given in bnp_array.sbatch; when it is full, override, e.g.
# "sbatch --partition=gpu_h100 --cpus-per-task=16" or
# "--partition=cbuild --gpus=0 --cpus-per-task=16".
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
PER_TASK=${4:-1}
TASK=${SLURM_ARRAY_TASK_ID:-0}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source /etc/profile.d/modules.sh
module load 2024 Julia/1.10.4-linux-x86_64
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
THREADS=$(( ${SLURM_CPUS_PER_TASK:-1} / PER_TASK ))
(( THREADS >= 1 )) || THREADS=1
export EVRP_BP_THREADS=$THREADS
SPLIT="$OUT/split_$SEED.json"
COUNT=$(python3 -c "import json, sys; print(len(json.load(open(sys.argv[1]))['nodes']))" "$SPLIT")
mkdir -p "$OUT/subtrees_$SEED"
for (( k = 0; k < PER_TASK; k++ )); do
    I=$(( TASK * PER_TASK + k ))
    (( I < COUNT )) || break
    EVRoutingEnv/baselines/optimization/branch_and_price/julia/run_bp.sh \
        "$OUT/instances/seed_$SEED.json" "$OUT/subtrees_$SEED/sub_$I.result.json" \
        --lp_solver highs --time_limit "$LIMIT" --threads "$THREADS" --verbose true \
        --subtree_file "$SPLIT" --subtree_index "$I" \
        --incumbent_dir "$OUT/incumbents_$SEED" --incumbent_tag "sub_$I" "${@:5}" \
        > "$OUT/subtrees_$SEED/sub_$I.log" 2>&1 &
done
wait
