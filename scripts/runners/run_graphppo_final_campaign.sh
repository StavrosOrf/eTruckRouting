#!/usr/bin/env bash
# Final evaluation for the from-scratch GraphPPO campaign.
#
# Selects among the curriculum arms on validation only, re-scores the finalists
# on held-out validation scenarios, then scores every method on the test split
# and reports both feasibility and optimality against the best-known reference.
#
# The baselines are already frozen from validation tuning and are re-run here so
# every method in the table is scored on an identical scenario set.
set -euo pipefail

cd "$(dirname "$0")/../.."

TRAINING_ROOT="${TRAINING_ROOT:-results/canonical/graphppo}"
SCENARIOS="${SCENARIOS:-300}"
SPLIT="${SPLIT:-test}"
OUTPUT="${OUTPUT:-results/canonical/campaign_graphppo}"
FROZEN="${FROZEN:-results/canonical/frozen_baselines.json}"
METHODS_FILE="${METHODS_FILE:-results/canonical/graphppo_methods.json}"
SELECTION="${SELECTION:-results/canonical/selected_graphppo.json}"

export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/evrp-matplotlib

echo "== 1. selecting the GraphPPO run on validation only =="
.venv/bin/python scripts/evaluation/select_architecture.py \
  --training-root "$TRAINING_ROOT" \
  --rescore-top "${RESCORE_TOP:-4}" \
  --rescore-scenarios "${RESCORE_SCENARIOS:-150}" \
  --rescore-offset "${RESCORE_OFFSET:-40}" \
  --output "$SELECTION"

echo
echo "== 2. assembling the frozen method set =="
.venv/bin/python - "$FROZEN" "$METHODS_FILE" "$SELECTION" <<'PYEOF'
import json
import sys

frozen_path, methods_path, selection_path = sys.argv[1], sys.argv[2], sys.argv[3]
methods = json.load(open(frozen_path))
selection = json.load(open(selection_path))
methods["graphppo"] = {
    "checkpoint": selection["checkpoint"],
    "prefix": "best",
    "deterministic": True,
    "state_encoder": selection["state_encoder"],
    "action_head": selection["action_head"],
    "trained_from_scratch": True,
    "imitation_learning": False,
    "validation_success_rate": selection["validation_summary"]["success_rate"],
}
with open(methods_path, "w") as handle:
    json.dump(methods, handle, indent=2, sort_keys=True)
print(f"frozen methods -> {methods_path}")
for name in sorted(methods):
    print(f"  {name}")
PYEOF

echo
echo "== 3. scoring every method on the ${SPLIT} split =="
.venv/bin/python scripts/evaluation/run_canonical_campaign.py \
  --split "$SPLIT" --scenarios "$SCENARIOS" \
  --methods "$METHODS_FILE" --output "$OUTPUT"

echo
echo "== 4. feasibility comparison (paired) =="
.venv/bin/python scripts/evaluation/compare_campaign.py \
  --campaign "$OUTPUT/$SPLIT" --reference mpc --candidate graphppo

echo
echo "== 5. optimality against the best-known reference =="
.venv/bin/python scripts/evaluation/build_best_known.py --campaign "$OUTPUT/$SPLIT"
