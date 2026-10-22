#!/bin/bash
# Best-known solutions (BKS) of the generalist suite: one job works through a
# list of driver commands, each using every core of its allocation (episodes in
# parallel). On Snellius every job on a GPU partition holds a GPU, so this CPU
# job takes one gpu_a100 share (18 cores, 117 GB) and leaves its GPU idle.
#   sbatch --job-name bks_<problem>_<split> scripts/campaigns/generalist_bks.sbatch SOURCE_DIR BUILDER PLAN_FILE [BKS_OUT]
# SOURCE_DIR: a snapshot made by `generalist_bks_static.py snapshot` (the code
# the job runs; edits to the live tree do not reach it). BUILDER: static or
# dynamic, selecting scripts/campaigns/generalist_bks_<BUILDER>.py. PLAN_FILE:
# one argument list per line for that script; "{W}" is replaced by the number of
# cores. BKS_OUT: the output root (default: the live results tree). Settings
# travel as arguments because Snellius jobs do not inherit the shell environment.
# The --output directory must exist: `generalist_bks_static.py snapshot` creates it.
#SBATCH --account=tdsei17344
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=6500M
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/home2/sorfanoudakis/EVRP/results/generalist_graphppo/bks/logs/%x_%j.out
set -euo pipefail
source_dir="$1"
builder="$2"
plan_file="$3"
repo="/gpfs/home2/sorfanoudakis/EVRP"
bks_out="${4:-$repo/results/generalist_graphppo/bks}"
python="$repo/.venv/bin/python"
case "$builder" in
  static|dynamic) ;;
  *) echo "BUILDER must be static or dynamic, not '$builder'" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="${TMPDIR:-/tmp}"
workers="${SLURM_CPUS_PER_TASK:-18}"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) with ${workers} cores, source ${source_dir}, builder ${builder}, out ${bks_out}"
cd "$source_dir"
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  line="${line//\{W\}/$workers}"
  echo "=== $(date -Is) $line"
  # shellcheck disable=SC2086
  BKS_OUT="$bks_out" PYTHONPATH="$source_dir" "$python" \
    "scripts/campaigns/generalist_bks_${builder}.py" $line
done < "$plan_file"
echo "=== $(date -Is) done"
