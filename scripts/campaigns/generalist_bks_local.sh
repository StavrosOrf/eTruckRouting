#!/bin/bash
# Local-box equivalent of generalist_bks.sbatch (iepg-compute1, no SLURM): one
# process works through a plan file from a frozen snapshot, pinned to a CPU set.
#   scripts/campaigns/generalist_bks_local.sh SOURCE_DIR {static|dynamic} PLAN_FILE BKS_OUT CPUSET
# CPUSET is a taskset list (e.g. 40-47); {W} in the plan becomes its core count.
set -euo pipefail
source_dir="$1"; builder="$2"; plan_file="$3"; bks_out="$4"; cpuset="$5"
repo="/home/sorfanoudakis/EVRP"
python="$repo/.venv/bin/python"
case "$builder" in static|dynamic) ;; *) echo "BUILDER must be static or dynamic" >&2; exit 2;; esac
workers=$(taskset -c "$cpuset" nproc)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="${TMPDIR:-/tmp}"
echo "local bks on $(hostname) cpus ${cpuset} (${workers} cores), source ${source_dir}, builder ${builder}, out ${bks_out}"
cd "$source_dir"
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  line="${line//\{W\}/$workers}"
  echo "=== $(date -Is) $line"
  # shellcheck disable=SC2086
  BKS_OUT="$bks_out" PYTHONPATH="$source_dir" taskset -c "$cpuset" "$python" \
    "scripts/campaigns/generalist_bks_${builder}.py" $line
done < "$plan_file"
echo "=== $(date -Is) done"
