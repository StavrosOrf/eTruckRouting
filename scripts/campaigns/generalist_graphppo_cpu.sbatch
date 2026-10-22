#!/bin/bash
# CPU stages of the generalist GraphPPO campaign: one array element works
# through a group of reference shards, one after another, on all its cores.
# On Snellius the allocation is one gpu_a100 share (18 cores plus a GPU the
# stage leaves idle, since GPU partitions refuse jobs without one).
#   submitted by: scripts/campaigns/generalist_graphppo.py submit --stage references [--pack J]
# Arguments: CAMPAIGN_DIR STAGE (references) GROUP [GROUP ...]. Each GROUP is a
# comma-separated list of shard indices; the array index picks one GROUP.
# Finished shards are skipped, so a resubmission only does what is missing.
set -euo pipefail
campaign_dir="$1"
stage="$2"
shift 2
groups=("$@")
group="${groups[${SLURM_ARRAY_TASK_ID:-0}]}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
campaign_python="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$campaign_dir/manifest.json")"
campaign_source="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$campaign_dir/manifest.json")"
cd "$campaign_source"
echo "job ${SLURM_JOB_ID:-local} on $(hostname): ${stage} ${group} on ${SLURM_CPUS_PER_TASK} cores"
PYTHONPATH="$campaign_source" "$campaign_python" scripts/campaigns/generalist_graphppo.py "$stage" \
  --directory "$campaign_dir" --tasks "$group" --workers "$SLURM_CPUS_PER_TASK"
