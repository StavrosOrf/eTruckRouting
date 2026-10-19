#!/bin/bash
# Cache the per-region inputs of a preset (one job per Geofabrik region, run in parallel):
#   sbatch --cpus-per-task=24 scripts/casestudies/slurm_extract.sh scripts/casestudies/presets/europe_10k.yaml europe/france
#   sbatch --export=ALL,MODE=scan scripts/casestudies/slurm_extract.sh scripts/casestudies/presets/europe_10k.yaml europe/france
# MODE=extract (default) caches the road network; MODE=scan caches the charging stations and places.
# PYTHON=/path/to/python and RAW_DIR=/path/to/.casestudy_raw override the defaults (pass them with sbatch --export=ALL,...).
#SBATCH --job-name=extract
#SBATCH --partition=cbuild
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=slurm-extract-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
"${PYTHON:-python}" scripts/casestudies/build_case_study.py "$1" ${RAW_DIR:+--raw-dir "$RAW_DIR"} "--${MODE:-extract}-only" "$2"
