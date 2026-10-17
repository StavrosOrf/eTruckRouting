#!/bin/bash
# Scenario check of a case-study config on a compute node (the first load of a large case computes
# all-pairs routes in pure Python, which takes tens of minutes and is cached in $TMPDIR):
#   CASE_TMPDIR=/some/persistent/dir sbatch --export=ALL,CASE_TMPDIR=/some/persistent/dir scripts/casestudies/slurm_check.sh EVRoutingEnv/config_files/config_joint_texas.yaml 20
#SBATCH --job-name=cscheck
#SBATCH --partition=cbuild
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=slurm-cscheck-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=1
# SLURM points TMPDIR at node-local storage; keep the route cache somewhere that survives the job
export TMPDIR="${CASE_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "$TMPDIR"
"${PYTHON:-.venv/bin/python}" scripts/casestudies/check_scenarios.py "$1" --episodes "${2:-20}" --plot-episodes "${3:-2}"
