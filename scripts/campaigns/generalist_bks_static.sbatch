#!/bin/bash
# Static long-haul BKS: one full compute node works through a list of driver
# commands, each using every core (episodes in parallel).
#   sbatch --job-name bks_s_<what> scripts/campaigns/generalist_bks_static.sbatch SOURCE_DIR PLAN_FILE
# SOURCE_DIR: a snapshot made by `generalist_bks_static.py snapshot` (code the
# job runs; edits to the live tree do not reach it). PLAN_FILE: one argument
# list per line for scripts/campaigns/generalist_bks_static.py; "{W}" is
# replaced by the number of cores. Outputs go to the live results tree.
#SBATCH --account=research-eemcs-ese
#SBATCH --partition=compute-p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=results/generalist_graphppo/bks/logs/%x_%j.out
set -euo pipefail
source_dir="$1"
plan_file="$2"
repo="/scratch/sorfanoudakis/eTruckRouting"
python="$repo/.venv/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="${TMPDIR:-/tmp}"
workers="${SLURM_CPUS_PER_TASK:-48}"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) with ${workers} cores, source ${source_dir}"
cd "$source_dir"
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  line="${line//\{W\}/$workers}"
  echo "=== $(date -Is) $line"
  # shellcheck disable=SC2086
  BKS_OUT="$repo/results/generalist_graphppo/bks" PYTHONPATH="$source_dir" "$python" \
    scripts/campaigns/generalist_bks_static.py $line
done < "$plan_file"
echo "=== $(date -Is) done"
