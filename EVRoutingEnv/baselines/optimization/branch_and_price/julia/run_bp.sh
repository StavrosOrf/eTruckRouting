#!/usr/bin/env bash
# Run the EVRPBranchPrice solver with the cluster toolchain.
#   run_bp.sh IN.json OUT.json [--key value ...]
#   run_bp.sh --batch LIST.txt [--key value ...]   (one "IN OUT" pair per line)
# Honours EVRP_BP_THREADS (Julia threads) and GRB_LICENSE_FILE; falls back to
# ~/gurobi.lic, and loads the Snellius Julia module (or a generic `julia` module) when available.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v julia >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
    module load 2024 Julia/1.10.4-linux-x86_64 >/dev/null 2>&1 || module load julia >/dev/null 2>&1 || true
fi
if [ -z "${GRB_LICENSE_FILE:-}" ] && [ -f "$HOME/gurobi.lic" ]; then
    export GRB_LICENSE_FILE="$HOME/gurobi.lic"
fi
script="$here/EVRPBranchPrice/bin/run_bp.jl"
if [ "${1:-}" = "--batch" ]; then
    script="$here/EVRPBranchPrice/bin/run_batch.jl"
    shift
fi
exec julia --project="$here/EVRPBranchPrice" -t "${EVRP_BP_THREADS:-1}" "$script" "$@"
