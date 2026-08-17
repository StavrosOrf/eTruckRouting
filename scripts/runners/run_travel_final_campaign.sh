#!/usr/bin/env bash
# Final evaluation under the fleet travel-time objective.
#
# Ranks GraphPPO arms on validation only, then scores every method on the test
# split and reports the paired travel-hour difference on jointly solved
# scenarios -- the only like-for-like speed comparison, since methods that
# decline different scenarios cannot be compared on their own conditional means.
#
# The baselines are re-tuned for this objective in
# `results/canonical/frozen_baselines_travel.json`; in particular the CP-SAT
# planner minimizes total route time rather than makespan. Comparing against a
# makespan-tuned planner would measure the objective mismatch, not the method.
set -euo pipefail

cd "$(dirname "$0")/../.."

TRAINING_ROOT="${TRAINING_ROOT:-results/canonical/graphppo_v2}"
SCENARIOS="${SCENARIOS:-300}"
SPLIT="${SPLIT:-test}"
OUTPUT="${OUTPUT:-results/canonical/campaign_travel}"
FROZEN="${FROZEN:-results/canonical/frozen_baselines_travel.json}"
METHODS_FILE="${METHODS_FILE:-results/canonical/travel_methods.json}"
SELECTION="${SELECTION:-results/canonical/selected_travel.json}"
WORKERS="${WORKERS:-12}"

export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/evrp-matplotlib

echo "== 1. selecting the GraphPPO run on validation only =="
.venv/bin/python scripts/evaluation/select_architecture.py \
  --training-root "$TRAINING_ROOT" \
  --objective travel_time \
  --rescore-top "${RESCORE_TOP:-4}" \
  --rescore-scenarios "${RESCORE_SCENARIOS:-150}" \
  --rescore-offset "${RESCORE_OFFSET:-40}" \
  --workers "$WORKERS" \
  --success-tolerance "${SUCCESS_TOLERANCE:-0.03}" \
  --output "$SELECTION"

echo
echo "== 2. assembling the frozen method set =="
.venv/bin/python - "$FROZEN" "$METHODS_FILE" "$SELECTION" <<'PYEOF'
import json
import sys

frozen_path, methods_path, selection_path = sys.argv[1], sys.argv[2], sys.argv[3]
methods = json.load(open(frozen_path))
selection = json.load(open(selection_path))
summary = selection["validation_summary"]
methods["graphppo"] = {
    "checkpoint": selection["checkpoint"],
    "prefix": "best",
    "deterministic": True,
    "state_encoder": selection["state_encoder"],
    "action_head": selection["action_head"],
    "trained_from_scratch": True,
    "imitation_learning": False,
    "objective": "fleet travel hours",
    "validation_success_rate": summary["success_rate"],
    "validation_travel_hours": summary.get("mean_travel_time_successful"),
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
echo "== 3b. carrying the makespan-era policy in as a reference =="
# The earlier GraphPPO was trained on a narrower observation and cannot be run
# under the routing-aware one, but it was scored on these same 300 test seeds
# and the simulator dynamics are unchanged, so its stored rows are directly
# comparable. Copying them in lets the paired statistics include it.
PRIOR="${PRIOR:-results/canonical/campaign_final/test/graphppo}"
if [ -d "$PRIOR" ] && [ ! -d "$OUTPUT/$SPLIT/graphppo_makespan" ]; then
  cp -r "$PRIOR" "$OUTPUT/$SPLIT/graphppo_makespan"
  echo "copied $PRIOR -> $OUTPUT/$SPLIT/graphppo_makespan"
fi

echo
echo "== 4. paired comparison against each baseline =="
for reference in cpsat mpc heuristic; do
  echo "--- reference: $reference"
  .venv/bin/python scripts/evaluation/compare_campaign.py \
    --campaign "$OUTPUT/$SPLIT" --reference "$reference" --candidate graphppo \
    --output "$OUTPUT/$SPLIT/comparison_vs_${reference}.json"
done

echo
echo "== 5. best-known travel-hour reference =="
.venv/bin/python scripts/evaluation/build_best_known.py \
  --campaign "$OUTPUT/$SPLIT" --objective travel_time
