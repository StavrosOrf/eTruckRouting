#!/bin/bash
# Build a case study on a compute node (partition cbuild: the only CPU product on this account, 32 cores).
# State-scale cases run for an hour or more and were killed on
# login nodes, so submit them:
#   sbatch scripts/casestudies/slurm_build.sh scripts/casestudies/presets/texas.yaml
# PYTHON=/path/to/python overrides the interpreter (needs osmium, pillow, pyyaml, scipy, numpy).
#SBATCH --job-name=casestudy
#SBATCH --partition=cbuild
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=slurm-casestudy-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
"${PYTHON:-python}" scripts/casestudies/build_case_study.py "$1" ${RAW_DIR:+--raw-dir "$RAW_DIR"} --workers "${SLURM_CPUS_PER_TASK:-8}"
