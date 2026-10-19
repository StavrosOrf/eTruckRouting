#!/bin/bash
# Split branch-and-price search of one scenario inside one allocation: the
# split run, then its subtrees side by side, then the combined certificate.
#
#   sbatch scripts/slurm/bnp_parallel.sbatch 50 100 seeds=1 results/branch_and_price/ca_50x100_par
#   sbatch --array=0-7 scripts/slurm/bnp_parallel.sbatch 50 100 test results/branch_and_price/ca_50x100_par
#
# Arguments: TRUCKS CUSTOMERS SPLIT OUT [TIME_LIMIT_SECONDS] [NODES] [CONFIG] [DRIVER ARGS...]
# SPLIT as in bnp_array.sbatch (train/validation/test, or "seeds=7,11").
# The split run stops once NODES (default 16) nodes are open, or after a
# quarter of TIME_LIMIT with whatever is open then; then all subtrees run at
# once, one Julia thread each (a solver process keeps about one core busy),
# sharing incumbents through OUT/incumbents_<seed>, until TIME_LIMIT. All of
# them solve the master with HiGHS (the Gurobi licence admits only two
# sessions). Use bnp_subtree.sbatch instead to spread the subtrees over
# several allocations.
#SBATCH --job-name=bnp-par
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bnp_par_%A_%a.out

set -euo pipefail
TRUCKS=${1:?trucks}
CUSTOMERS=${2:?customers}
SPLIT=${3:?split}
OUT=${4:?output directory}
LIMIT=${5:-84000}
NODES=${6:-16}
CONFIG=${7:-EVRoutingEnv/config_files/config_joint.yaml}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source /etc/profile.d/modules.sh
module load 2024 Julia/1.10.4-linux-x86_64
source /home/sorfanoudakis/EVRP/.venv/bin/activate
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
export PYTHONPATH=.
START=$(date +%s)

if [[ "$SPLIT" == seeds=* ]]; then
    read -ra SEEDS <<< "${SPLIT#seeds=}"; SEEDS=(${SEEDS[@]//,/ })
    SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}
else
    SEED=$(python -c "from scripts.evaluation.canonical_harness import split_seeds; import sys; \
print(split_seeds(sys.argv[1], int(sys.argv[2]))[int(sys.argv[3])])" \
        "$SPLIT" "${SLURM_ARRAY_TASK_COUNT:-1}" "${SLURM_ARRAY_TASK_ID:-0}")
fi
SIZE=(--config "$CONFIG" --trucks "$TRUCKS" --customers "$CUSTOMERS")

# 1. split run (writes OUT/seed_<seed>.json; OUT/split_<seed>.json unless it closed the search)
python -m scripts.evaluation.solve_branch_and_price "${SIZE[@]}" --seeds "$SEED" \
    --time-limit $(( LIMIT / 4 )) --threads 2 --split-nodes "$NODES" --out "$OUT" \
    --solver-arg lp_solver highs "${@:8}" || true
[[ -f "$OUT/split_$SEED.json" ]] || exit 0

# 2. subtrees, all at once, until the job's time limit
LEFT=$(( LIMIT - ($(date +%s) - START) - 600 ))
COUNT=$(python -c "import json, sys; print(len(json.load(open(sys.argv[1]))['nodes']))" "$OUT/split_$SEED.json")
mkdir -p "$OUT/subtrees_$SEED"
export EVRP_BP_THREADS=1
for (( I = 0; I < COUNT; I++ )); do
    EVRoutingEnv/baselines/optimization/branch_and_price/julia/run_bp.sh \
        "$OUT/instances/seed_$SEED.json" "$OUT/subtrees_$SEED/sub_$I.result.json" \
        --lp_solver highs --time_limit "$LEFT" --threads 1 --verbose true --lns_time 300 \
        --subtree_file "$OUT/split_$SEED.json" --subtree_index "$I" \
        --incumbent_dir "$OUT/incumbents_$SEED" --incumbent_tag "sub_$I" \
        > "$OUT/subtrees_$SEED/sub_$I.log" 2>&1 &
done
wait || true

# 3. combined certificate
python -m scripts.evaluation.combine_bnp_subtrees --out "$OUT" --seed "$SEED" "${SIZE[@]}"
