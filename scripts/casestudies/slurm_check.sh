#!/bin/bash
# Scenario check of a case-study config on a compute node (the first load of a large case computes
# all-pairs routes in pure Python, which takes tens of minutes and is cached in $TMPDIR):
#   TMPDIR=/some/persistent/dir sbatch scripts/casestudies/slurm_check.sh EVRoutingEnv/config_files/config_joint_texas.yaml 20
#SBATCH --job-name=cscheck
#SBATCH --partition=cbuild
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=slurm-cscheck-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=1
"${PYTHON:-.venv/bin/python}" scripts/casestudies/check_scenarios.py "$1" --episodes "${2:-20}" --plot-episodes "${3:-2}"
