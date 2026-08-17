"""Turn the generalization artifacts into the tables the response letter needs.

Three questions, kept apart because they license different claims:

1. *How much does each method degrade outside its tuning distribution?*  Every
   regime is reported next to the in-distribution control for the same method,
   so a regime that is simply harder for everybody is not read as a failure of
   any one method.
2. *Does the learned policy degrade faster than the classical baselines?*  The
   ranking within each regime answers that, and it is the question R1.7 is
   actually asking.
3. *Which regimes are interpolation, which are size transfer, and which are
   genuinely out of distribution?*  The campaign labels each one, and the
   summary keeps the labels attached so a size-transfer result is never quoted
   as evidence of out-of-distribution generalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluation.run_generalization_campaign import REGIMES


def _conditional(summary: dict, metric: str) -> float | None:
    entry = summary["aggregate"]["metrics"].get(metric)
    return None if entry is None else entry.get("mean")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", default="results/canonical/generalization"
    )
    parser.add_argument("--control", default="in_distribution")
    parser.add_argument(
        "--output", default="results/canonical/generalization/summary.json"
    )
    arguments = parser.parse_args()

    # Read the artifact tree rather than the index. The index reflects only the
    # methods of the invocation that wrote it, so a campaign completed in two
    # passes -- baselines first, learned methods second -- would silently report
    # whichever pass ran last. The per-method summaries on disk are the record.
    campaign = Path(arguments.campaign)
    index_path = campaign / "generalization_index.json"
    labels: dict[str, dict] = {}
    if index_path.exists():
        labels = json.loads(index_path.read_text()).get("regimes", {})

    regimes: dict[str, dict] = {}
    for summary_path in sorted(campaign.glob("*/*/summary.json")):
        regime = summary_path.parent.parent.name
        method = summary_path.parent.name
        entry = regimes.setdefault(
            regime,
            {
                "kind": labels.get(regime, {}).get("kind")
                or REGIMES.get(regime, {}).get("kind", "unknown"),
                "description": labels.get(regime, {}).get("description")
                or REGIMES.get(regime, {}).get("description", ""),
                "methods": {},
            },
        )
        entry["methods"][method] = json.loads(summary_path.read_text())

    if arguments.control not in regimes:
        raise SystemExit(f"control regime {arguments.control!r} was not scored")

    control = {
        method: summary["aggregate"]["success_rate"]
        for method, summary in regimes[arguments.control]["methods"].items()
    }

    rows = []
    for name, regime in regimes.items():
        for method, summary in regime["methods"].items():
            aggregate = summary["aggregate"]
            success = aggregate["success_rate"]
            baseline = control.get(method)
            rows.append(
                {
                    "regime": name,
                    "kind": regime["kind"],
                    "description": regime["description"],
                    "method": method,
                    "success_rate": success,
                    "success_wilson_95": aggregate["success_wilson_95"],
                    "delta_vs_control": (
                        None if baseline is None else success - baseline
                    ),
                    "travel_hours_successful": _conditional(
                        summary, "total_travel_time"
                    ),
                    "completed_fraction": _conditional(summary, "completed_fraction"),
                    "episodes": aggregate["episode_count"],
                }
            )

    by_regime: dict[str, list[dict]] = {}
    for row in rows:
        by_regime.setdefault(row["regime"], []).append(row)

    order = {"interpolation": 0, "size_transfer": 1, "ood": 2}
    print(
        f"{'regime':<22}{'kind':<14}{'method':<12}"
        f"{'success':>9}{'delta':>8}{'travel h':>10}"
    )
    for name in sorted(
        by_regime, key=lambda key: (order.get(regimes[key]["kind"], 9), key)
    ):
        for row in sorted(
            by_regime[name], key=lambda item: -item["success_rate"]
        ):
            delta = row["delta_vs_control"]
            travel = row["travel_hours_successful"]
            print(
                f"{row['regime']:<22}{row['kind']:<14}{row['method']:<12}"
                f"{row['success_rate']:>9.3f}"
                f"{'   n/a' if delta is None else f'{delta:>+8.3f}'}"
                f"{'     n/a' if travel is None else f'{travel:>10.1f}'}"
            )

    # Who degrades least, averaged over the regimes that are genuinely OOD.
    ood_deltas: dict[str, list[float]] = {}
    for row in rows:
        if row["kind"] == "ood" and row["delta_vs_control"] is not None:
            ood_deltas.setdefault(row["method"], []).append(row["delta_vs_control"])
    ranking = {
        method: sum(values) / len(values) for method, values in ood_deltas.items()
    }
    print("\nmean success change across out-of-distribution regimes:")
    for method, value in sorted(ranking.items(), key=lambda item: -item[1]):
        print(f"  {method:<12}{value:>+8.3f}")

    destination = Path(arguments.output)
    destination.write_text(
        json.dumps(
            {
                "control_regime": arguments.control,
                "rows": rows,
                "mean_ood_success_change": ranking,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"\nwrote {destination}", flush=True)


if __name__ == "__main__":
    main()
