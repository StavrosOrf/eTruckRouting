#!/bin/bash
# Create the solver interpreter of the RoutingBlocks baseline (Python 3.11: the
# routingblocks wheel has no 3.13 build) and compile the native pricing core.
#
#   bash scripts/setup_rb_env.sh [TARGET_DIR]       # default: <repo>/.venv-rb
#
# The bridge finds the interpreter as $EVRP_RB_PYTHON or the nearest .venv-rb.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$ROOT/.venv-rb}"
uv venv --python 3.11 "$TARGET"
uv pip install --python "$TARGET/bin/python" "routingblocks==0.2.1" numpy pybind11 highspy pytest
PYTHONPATH="$ROOT" "$TARGET/bin/python" "$ROOT/EVRoutingEnv/baselines/optimization/routingblocks_alns/build.py" --force
echo "solver interpreter: $TARGET/bin/python"
