#!/usr/bin/env bash
# Freeze the validation-selected architecture, then score every method on the
# held-out test split and publish the paired comparison.
#
# Order matters and is enforced here: the architecture is selected from
# validation histories BEFORE any test scenario is touched, and the baseline
# settings come from the frozen validation tuning file.
set -euo pipefail

cd "$(dirname "$0")/../.."

TRAINING_ROOT="${TRAINING_ROOT:-results/canonical/training}"
SCENARIOS="${SCENARIOS:-200}"
OUTPUT="${OUTPUT:-results/canonical/campaign}"
SPLIT="${SPLIT:-test}"
FROZEN="${FROZEN:-results/canonical/frozen_baselines.json}"
METHODS_FILE="${METHODS_FILE:-results/canonical/headline_methods.json}"

export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/evrp-matplotlib

echo "== 1. selecting architecture on validation only =="
.venv/bin/python scripts/evaluation/select_architecture.py \
  --training-root "$TRAINING_ROOT" \
  --rescore-top "${RESCORE_TOP:-3}" \
  --rescore-scenarios "${RESCORE_SCENARIOS:-150}" \
  --output results/canonical/selected_architecture.json

echo
echo "== 2. freezing tuned baseline settings from the validation grids =="
.venv/bin/python - "$FROZEN" <<'PYEOF'
import json
import pathlib
import sys

root = pathlib.Path("results/canonical/tuning")
frozen = {"random": {"seed": 0}}
for name, filename in (
    ("heuristic", "heuristic_tuning.json"),
    ("mpc", "mpc_tuning.json"),
    ("cpsat", "cpsat_tuning.json"),
):
    path = root / filename
    if not path.exists():
        # Silently dropping a baseline would quietly weaken the comparison, so
        # a missing tuning file is a hard error rather than a warning.
        raise SystemExit(
            f"{path} is missing; run tune_heuristics.py for {name} before the "
            "headline campaign, or the comparison will omit it"
        )
    best = json.loads(path.read_text())["best"]
    frozen[name] = {
        "parameters": best["parameters"],
        "validation_success_rate": best["summary"]["success_rate"],
        "validation_scenarios": best["summary"]["episodes"],
    }
    print(f"  {name}: validation success {best['summary']['success_rate']:.3f}")
pathlib.Path(sys.argv[1]).write_text(json.dumps(frozen, indent=2, sort_keys=True))
PYEOF

echo
echo "== 3. assembling frozen method set =="
.venv/bin/python - "$FROZEN" "$METHODS_FILE" <<'PYEOF'
import json
import sys

frozen_path, methods_path = sys.argv[1], sys.argv[2]
methods = json.load(open(frozen_path))
selection = json.load(open("results/canonical/selected_architecture.json"))
methods["rl"] = {
    "checkpoint": selection["checkpoint"],
    "prefix": "best",
    "deterministic": True,
    "state_encoder": selection["state_encoder"],
    "action_head": selection["action_head"],
    "validation_success_rate": selection["validation_summary"]["success_rate"],
}
with open(methods_path, "w") as handle:
    json.dump(methods, handle, indent=2, sort_keys=True)
print(f"frozen methods -> {methods_path}")
for name, settings in sorted(methods.items()):
    marker = " (selected on validation)" if name == "rl" else ""
    print(f"  {name}{marker}")
PYEOF

echo
echo "== 4. scoring every method on the ${SPLIT} split =="
.venv/bin/python scripts/evaluation/run_canonical_campaign.py \
  --split "$SPLIT" \
  --scenarios "$SCENARIOS" \
  --methods "$METHODS_FILE" \
  --output "$OUTPUT"

echo
echo "== 5. paired comparison =="
.venv/bin/python scripts/evaluation/compare_campaign.py \
  --campaign "$OUTPUT/$SPLIT" \
  --reference mpc \
  --candidate rl
