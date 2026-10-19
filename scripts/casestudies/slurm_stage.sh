#!/bin/bash
# One stage of a staged (output_format: npy) case-study build on a cbuild node (32 cores). Normally submitted by
# submit_npy_build.sh; by hand:
#   sbatch scripts/casestudies/slurm_stage.sh prepare  scripts/casestudies/presets/europe_10k.yaml
#   sbatch --array=0-31 --export=ALL,N_SHARDS=32 scripts/casestudies/slurm_stage.sh od scripts/casestudies/presets/europe_10k.yaml
#   sbatch scripts/casestudies/slurm_stage.sh assemble scripts/casestudies/presets/europe_10k.yaml
# PYTHON=/path/to/python and RAW_DIR=/path/to/.casestudy_raw override the defaults.
#SBATCH --job-name=casestage
#SBATCH --partition=cbuild
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
stage="$1"
preset="$2"
extra=()
if [ "$stage" = od ]; then
    extra=(--shard "${SLURM_ARRAY_TASK_ID}" --n-shards "${N_SHARDS}")
fi
"${PYTHON:-python}" scripts/casestudies/build_case_study.py "$preset" ${RAW_DIR:+--raw-dir "$RAW_DIR"} \
    --workers "${SLURM_CPUS_PER_TASK:-8}" --stage "$stage" "${extra[@]}"
