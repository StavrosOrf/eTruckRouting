#!/bin/bash
# Generalist GraphPPO training: one array element trains a group of runs at once,
# each on its own GPU (or R runs per GPU) and its own disjoint CPU slice.
#   submitted by: scripts/campaigns/generalist_graphppo.py submit --stage train --budget N [--pack K] [--runs-per-gpu R]
# Arguments: CAMPAIGN_DIR BUDGET RUNS_PER_GPU GROUP [GROUP ...]. Each GROUP is a
# comma-separated list of run indices; the array index picks one GROUP.
# Partition, account, GPUs, CPUs and memory come from the manifest via sbatch options.
set -euo pipefail
campaign_dir="$1"
budget="$2"
runs_per_gpu="$3"
shift 3
groups=("$@")
group="${groups[${SLURM_ARRAY_TASK_ID:-0}]}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# sbatch runs a spool copy, so the manifest locates the interpreter and checkout.
campaign_python="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$campaign_dir/manifest.json")"
campaign_source="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$campaign_dir/manifest.json")"
cd "$campaign_source"
echo "job ${SLURM_JOB_ID:-local} on $(hostname): runs ${group}, ${runs_per_gpu} per GPU, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
# GPU utilisation and memory every 30 s, for sizing --runs-per-gpu.
monitor_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total --format=csv -l 30 \
    > "$campaign_dir/logs/gpu_${SLURM_JOB_ID:-local}.csv" 2>&1 &
  monitor_pid=$!
fi
PYTHONPATH="$campaign_source" "$campaign_python" scripts/campaigns/generalist_graphppo.py train-pack \
  --directory "$campaign_dir" --runs "$group" --budget "$budget" --runs-per-gpu "$runs_per_gpu" &
supervisor_pid=$!
# The supervisor forwards the pre-timeout signal to every run; each trainer
# checkpoints at its next update and exits, and the next submission resumes
# them from their training.pt.
trap 'kill -USR1 "$supervisor_pid" 2>/dev/null || true' USR1 TERM
while true; do
  set +e
  wait "$supervisor_pid"
  status=$?
  set -e
  if ! kill -0 "$supervisor_pid" 2>/dev/null; then
    if [ -n "$monitor_pid" ]; then
      kill "$monitor_pid" 2>/dev/null || true
    fi
    exit "$status"
  fi
done
