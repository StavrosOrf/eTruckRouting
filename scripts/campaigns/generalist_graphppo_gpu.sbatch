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
# A twin array queues the same groups on another partition (submit --twin-of ARRAY_JOB_ID), since a job
# cannot name two GPU partitions here. Only one element of a pair may train. This one yields to a twin
# that is already running; otherwise it takes the twin out of the queue and waits until it is gone.
twin_array="${EVRP_TWIN_ARRAY:-}"
if [ -n "$twin_array" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  twin="${twin_array}_${SLURM_ARRAY_TASK_ID}"
  removed=0
  state="unknown"
  for attempt in $(seq 1 60); do
    if listing="$(squeue -h -r -u "$USER" -o '%i %T')"; then
      state="$(awk -v twin="$twin" '$1 == twin { print $2; exit }' <<< "$listing")"
    else
      state="unknown"  # the scheduler did not answer: never taken for an absent twin
    fi
    case "$state" in
      "") break ;;
      PENDING) scancel "$twin" && removed=1 ;;
      RUNNING|CONFIGURING|SUSPENDED)
        if [ "$removed" = 0 ]; then
          echo "twin ${twin} is ${state} and has the runs ${group}: this job stops"
          exit 0
        fi
        ;;
    esac
    sleep "${EVRP_TWIN_POLL:-10}"
  done
  if [ -n "$state" ]; then
    echo "twin ${twin} is still ${state}: this job does not start" >&2
    exit 1
  fi
  echo "twin ${twin} is not queued (removed here: ${removed}): this job trains the runs ${group}"
fi
# The demonstration store is read one observation at a time, in random order. Shared scratch served 16-27
# such reads a second (2026-11-13), where a cloning epoch needs 340 thousand and a PPO update 100 thousand.
# A campaign can therefore keep the built store at demonstrations_shared and make demonstrations a link
# into /dev/shm: each job then copies the store into its own memory-backed /dev/shm (private to the job,
# charged to its memory, gone when it ends) before it trains, and the trainer reads that copy.
store_link="$campaign_dir/demonstrations"
store_shared="$campaign_dir/demonstrations_shared"
if [ -L "$store_link" ] && [ -d "$store_shared/" ] && [[ "$(readlink "$store_link")" == /dev/shm/* ]]; then
  store_local="$(readlink "$store_link")"
  measure() { (cd "$1/" && find . -type f -printf '%s\n' | awk '{ bytes += $1 } END { printf "%d %d\n", NR, bytes }'); }
  read -r store_files store_bytes <<< "$(measure "$store_shared")"
  job_mb="${SLURM_MEM_PER_NODE:-$(( ${SLURM_MEM_PER_CPU:-0} * ${SLURM_CPUS_ON_NODE:-0} ))}"
  needed_mb=$(( store_bytes / 1048576 + ${EVRP_TRAINER_MEMORY_MB:-90000} ))
  if [ "$store_files" = 0 ]; then
    echo "no demonstration store at ${store_shared}: this job does not start" >&2
    exit 1
  fi
  if [ "$job_mb" -gt 0 ] && [ "$job_mb" -lt "$needed_mb" ]; then
    echo "the job has ${job_mb} MB; the store in memory and the trainer need ${needed_mb} MB: this job does not start" >&2
    exit 1
  fi
  copy_started=$SECONDS
  mkdir -p "$store_local"
  (cd "$store_shared/" && find . -type d -exec mkdir -p "$store_local/{}" \; \
    && find . -type f -print0 | xargs -0 -P "${EVRP_STORE_COPIES:-16}" -I{} cp {} "$store_local/{}")
  read -r local_files local_bytes <<< "$(measure "$store_local")"
  if [ "$local_files $local_bytes" != "$store_files $store_bytes" ]; then
    echo "copied ${local_files} files (${local_bytes} bytes) of ${store_files} (${store_bytes}): this job does not start" >&2
    exit 1
  fi
  echo "demonstration store in ${store_local}: ${store_files} files, ${store_bytes} bytes, copied in $((SECONDS - copy_started)) s"
fi
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
