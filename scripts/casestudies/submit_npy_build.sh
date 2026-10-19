#!/bin/bash
# Submit a staged case-study build as three dependent Slurm jobs: prepare -> od (array of N shards) -> assemble.
#   bash scripts/casestudies/submit_npy_build.sh scripts/casestudies/presets/europe_10k.yaml 32
# The road-network extraction and the charger/place scans must be cached first (slurm_extract.sh, one job per
# region); otherwise `prepare` does them serially. PYTHON and RAW_DIR are passed through to the jobs.
set -euo pipefail
preset="$1"
shards="${2:-32}"
export_vars="ALL,N_SHARDS=${shards}${PYTHON:+,PYTHON=$PYTHON}${RAW_DIR:+,RAW_DIR=$RAW_DIR}"
here="$(dirname "$0")"
prep=$(sbatch --parsable --job-name=cs_prepare --export="$export_vars" "$here/slurm_stage.sh" prepare "$preset")
od=$(sbatch --parsable --job-name=cs_od --dependency=afterok:"$prep" --array=0-$((shards - 1)) --time=04:00:00 \
    --export="$export_vars" "$here/slurm_stage.sh" od "$preset")
asm=$(sbatch --parsable --job-name=cs_assemble --dependency=afterok:"$od" --export="$export_vars" \
    "$here/slurm_stage.sh" assemble "$preset")
echo "prepare=$prep od=$od (array 0-$((shards - 1))) assemble=$asm"
