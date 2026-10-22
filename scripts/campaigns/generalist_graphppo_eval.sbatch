#!/bin/bash
# Evaluation of the generalist GraphPPO campaign: baselines, then every run's
# best.pt, in one allocation of one GPU and its CPU share.
#   submitted by: scripts/campaigns/generalist_graphppo.py submit --stage evaluate --split S --tag T [--runs ...]
# Arguments: CAMPAIGN_DIR SPLIT TAG [RUN ...]. Rerunning skips finished baselines and runs.
set -euo pipefail
campaign_dir="$1"
split="$2"
tag="$3"
shift 3
runs=()
if [ "$#" -gt 0 ]; then
  runs=(--runs "$@")
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
campaign_python="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$campaign_dir/manifest.json")"
campaign_source="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$campaign_dir/manifest.json")"
cd "$campaign_source"
# Workers fork inside evaluate before any policy moves to the GPU.
PYTHONPATH="$campaign_source" "$campaign_python" scripts/campaigns/generalist_graphppo.py evaluate \
  --directory "$campaign_dir" --split "$split" --tag "$tag" --device cuda:0 \
  --workers "$((SLURM_CPUS_PER_TASK - 1))" "${runs[@]}"
