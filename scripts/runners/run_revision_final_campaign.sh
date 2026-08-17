#!/usr/bin/env bash
# The revision's final test-split campaign.
#
# Differences from run_travel_final_campaign.sh, which produced document 10's
# headline:
#
#   * 500 scenarios rather than 300, because R1.7 asks for at least 500 paired
#     scenarios per main setting and the GraphPPO-versus-CP-SAT difference is
#     small enough that the larger sample matters;
#   * the corrected CP-SAT planner, which could not previously leave a truck
#     idle and is stronger now, plus ALNS as a second strong optimizer;
#   * the learned baseline family and the unmasked ablation, each scored in the
#     environment it was trained in;
#   * seed replicates of the headline ladder, so the reported figure carries a
#     spread rather than a single number.
#
# Nothing is selected here. Every checkpoint and every baseline setting arrives
# frozen from validation.
set -euo pipefail

cd "$(dirname "$0")/../.."

SCENARIOS="${SCENARIOS:-500}"
SPLIT="${SPLIT:-test}"
OUTPUT="${OUTPUT:-results/canonical/campaign_revision}"
FROZEN="${FROZEN:-results/canonical/frozen_baselines_revision.json}"
METHODS_FILE="${METHODS_FILE:-results/canonical/revision_methods.json}"
WORKERS="${WORKERS:-8}"

export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/evrp-matplotlib

echo "== 1. assembling the frozen method set =="
.venv/bin/python scripts/evaluation/assemble_revision_methods.py \
  --frozen "$FROZEN" --output "$METHODS_FILE"

echo
echo "== 2. scoring every method on the ${SPLIT} split (${SCENARIOS} scenarios) =="
.venv/bin/python scripts/evaluation/run_canonical_campaign.py \
  --split "$SPLIT" --scenarios "$SCENARIOS" \
  --methods "$METHODS_FILE" --output "$OUTPUT" --workers "$WORKERS"

echo
echo "== 3. paired comparisons on jointly solved scenarios =="
# Methods decline different instances, so their own conditional means are not
# comparable; pairing by scenario seed is the only like-for-like reading.
for reference in cpsat alns mpc heuristic ppo_flat ppo_deepsets ppo_stategnn mask_none; do
  if [ -d "$OUTPUT/$SPLIT/$reference" ]; then
    echo "--- reference: $reference"
    .venv/bin/python scripts/evaluation/compare_campaign.py \
      --campaign "$OUTPUT/$SPLIT" --reference "$reference" --candidate graphppo \
      --output "$OUTPUT/$SPLIT/comparison_vs_${reference}.json"
  fi
done

echo
echo "== 4. best-known travel-hour reference =="
.venv/bin/python scripts/evaluation/build_best_known.py \
  --campaign "$OUTPUT/$SPLIT" --objective travel_time

echo
echo "artifacts under $OUTPUT/$SPLIT"
