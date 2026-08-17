#!/usr/bin/env bash
# End-to-end travel-time campaign: wait for stage A, select, refine, score.
#
# Stage A must already be running (or finished). This waits for it, freezes the
# stage-A winner on validation evidence only, launches the stage-B refinement
# from that checkpoint, waits again, and then runs the test-split campaign.
#
# Nothing here reads a test scenario until the final step, and the checkpoint
# that reaches it was chosen entirely from the validation namespace.
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/evrp-matplotlib

STAGE_A="${STAGE_A:-results/canonical/graphppo_v2}"
STAGE_B="${STAGE_B:-results/canonical/graphppo_v2_stageb}"
STAGE_A_SELECTION="${STAGE_A_SELECTION:-results/canonical/selected_stagea.json}"
WORKERS="${WORKERS:-12}"

# Match the interpreter invocation, not the bare script name: any shell that
# mentions the script -- including a monitoring loop that greps for it, or this
# script itself -- would otherwise look like a live training run and the wait
# would never return.
TRAINER_PATTERN='python[^ ]* scripts/training/train_canonical_ppo\.py'

wait_for_training() {
  local label="$1"
  echo "[$(date +%H:%M:%S)] waiting for $label to finish..."
  while pgrep -f "$TRAINER_PATTERN" > /dev/null; do
    sleep 60
  done
  echo "[$(date +%H:%M:%S)] $label finished"
}

wait_for_training "stage A"

echo
echo "== freezing the stage-A winner on validation only =="
# Re-scored on validation scenarios 40..99, disjoint from the 0..39 that chose
# the checkpoints during training, so the finalist comparison is independent of
# the selection that produced them.
.venv/bin/python scripts/evaluation/select_architecture.py \
  --training-root "$STAGE_A" \
  --objective travel_time \
  --rescore-top 4 --rescore-scenarios 60 --rescore-offset 40 \
  --workers "$WORKERS" --success-tolerance 0.03 \
  --output "$STAGE_A_SELECTION"

WINNER=$(.venv/bin/python -c "import json;print(json.load(open('$STAGE_A_SELECTION'))['checkpoint'])")
echo "stage-A winner: $WINNER"

echo
echo "== stage B: refining against fleet travel hours =="
INIT_FROM="$WINNER" OUTPUT="$STAGE_B" bash scripts/runners/run_graphppo_v2_stageb.sh &
STAGE_B_PID=$!
sleep 120
wait_for_training "stage B"
wait "$STAGE_B_PID" 2>/dev/null || true

echo
echo "== final campaign on the test split =="
# Stage B refines stage A, so both are candidates; the selector reads only
# validation and picks whichever is better under the campaign objective.
mkdir -p results/canonical/graphppo_travel_all
for run in "$STAGE_A"/*/ "$STAGE_B"/*/; do
  [ -f "${run}validation_history.json" ] || continue
  name=$(basename "$run")
  target="results/canonical/graphppo_travel_all/$name"
  [ -e "$target" ] || ln -s "$(realpath "$run")" "$target"
done

TRAINING_ROOT=results/canonical/graphppo_travel_all \
  WORKERS="$WORKERS" \
  bash scripts/runners/run_travel_final_campaign.sh
