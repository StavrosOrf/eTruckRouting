"""Summarize only the corrected 400 kWh experiments, excluding cap pilots."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from EVRoutingEnv.models.core.transportation_graph import TransportationGraph
from EVRoutingEnv.models.environment.energy_reachability import EnergyReachability
from EVRoutingEnv.utils.utils import get_graph, load_config


def network_limits():
    graph = TransportationGraph(
        get_graph(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    )
    envelope = EnergyReachability.build(graph, graph.get_charging_nodes(), 400.0, 1.2)
    candidates = [
        node
        for node in envelope.nodes
        if node not in envelope.chargers
        and graph.graph.in_degree(node) > 0
        and graph.graph.out_degree(node) > 0
    ]
    depots = [
        {
            "depot": depot,
            "max_individually_energy_safe_customers": len(
                envelope.customers(depot, candidates)
            ),
        }
        for depot in candidates
    ]
    return {
        "battery_kwh": 400.0,
        "execution_energy_multiplier": 1.2,
        "topology_customer_ceiling": len(candidates) - 1,
        "generator_customer_ceiling": max(
            row["max_individually_energy_safe_customers"] for row in depots
        ),
        "assumption": "Individual depot/charger approach and exit on one battery, not a full fleet routing certificate; negative net-energy legs rejected consistently with execution.",
        "depots": depots,
    }


def main():
    notes = Path("notes")
    groups = defaultdict(list)
    for path in sorted(notes.glob("default400_scale_*_2026-10-17.jsonl")):
        if "pilot" in path.name:
            continue
        meta = json.loads(path.with_suffix(".metadata.json").read_text())
        assert meta["arguments"]["battery_kwh"] == 400.0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            groups[(row["trucks"], row["customers"], row["horizon_hours"])].append(
                row
                | {"source": path.name, "guard_mode": meta["arguments"]["step_mode"]}
            )
    result = []
    for (trucks, customers, horizon), records in sorted(
        groups.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        # Final configured-budget confirmations supersede earlier cap/load
        # experiments for the same case, without discarding their evidence.
        final = [r for r in records if "_final_" in r["source"]]
        prior = [r for r in records if "_final_" not in r["source"]] if final else []
        if final:
            records = final
        # Confirmation sweeps can repeat seeds. Count a seed only once.
        ended = {r["seed"]: r for r in records if r["status"] == "completed"}
        cuts = {
            r["seed"]: r
            for r in records
            if r["status"] == "episode_wall_budget" and r["seed"] not in ended
        }
        successful = [
            r for r in ended.values() if r["success"] and r["episode_seconds"] <= 60
        ]
        row = {
            "trucks": trucks,
            "customers": customers,
            "horizon_hours": horizon,
            "completed_seeds": len(ended),
            "successful_within_60s": len(successful),
            "episode_budget_stops": len(cuts),
            "termination_counts": dict(
                Counter(r["termination_reason"] for r in ended.values())
            ),
            "errors": list(
                dict.fromkeys(r["error"] for r in records if r["status"] == "error")
            ),
            "experiment_stops": [
                r["status"]
                for r in records
                if r["status"] in {"wall_budget", "rss_budget"}
            ],
            "sources": sorted({r["source"] for r in records}),
            "guard_modes": sorted({r["guard_mode"] for r in records}),
            "prior_sources": sorted({r["source"] for r in prior}),
            "prior_budget_stops": sum(
                r["status"] == "episode_wall_budget" for r in prior
            ),
            "prior_termination_counts": dict(
                Counter(
                    r["termination_reason"] for r in prior if r["status"] == "completed"
                )
            ),
            "interrupted_seeds": len(
                {r["seed"] for r in records if r["status"] == "initialized"}
                - set(ended)
                - set(cuts)
            ),
            "peak_rss_mib": max(
                (r.get("peak_rss_mib", r.get("rss_mib", 0)) for r in records), default=0
            ),
            "steps_checked_ended": sum(r["steps"] for r in ended.values()),
        }
        for field in (
            "episode_seconds",
            "simulated_hours",
            "steps",
            "vehicles_used",
            "queue_hours",
        ):
            values = [r[field] for r in successful]
            row[field + "_median"] = median(values) if values else None
            row[field + "_max"] = max(values) if values else None
        result.append(row)
    base = notes / "default400_scale_summary_2026-10-17"
    base.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    with base.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            for row in result
        )
    (notes / "default400_network_limits_2026-10-17.json").write_text(
        json.dumps(network_limits(), indent=2) + "\n"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for count in sorted({row["customers"] for row in result}):
        measured = [
            r for r in result if r["customers"] == count and r["successful_within_60s"]
        ]
        if not measured:
            continue
        axes[0].plot(
            [r["trucks"] for r in measured],
            [r["episode_seconds_median"] for r in measured],
            "o-",
            label=f"{count} customers",
        )
    axes[0].axhline(60, color="red", linestyle="--", label="Practical episode budget")
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Configured trucks",
        ylabel="Wall seconds (successful seeds)",
        title="Success requires service AND depot return within 200 h",
    )
    axes[0].legend(fontsize=8, ncols=2)
    for row in result:
        trials = row["completed_seeds"] + row["episode_budget_stops"]
        if trials:
            fraction = row["successful_within_60s"] / trials
            axes[1].scatter(
                row["customers"],
                row["trucks"],
                c=[fraction],
                vmin=0,
                vmax=1,
                cmap="RdYlGn",
                s=90,
            )
            axes[1].annotate(
                f"{row['successful_within_60s']}/{trials}",
                (row["customers"], row["trucks"]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7,
            )
    axes[1].set(
        yscale="log",
        xlabel="Fleet-total customers",
        ylabel="Configured trucks",
        title="Observed successes / distinct tested seeds (60 s budget)",
    )
    for axis in axes:
        axis.grid(alpha=0.2, which="both")
    fig.suptitle("Corrected stochastic simulator • fixed 400 kWh • 17 October 2026")
    fig.savefig(notes / "default400_scaling_2026-10-17.png", dpi=180)
    fig.savefig(notes / "default400_scaling_2026-10-17.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {
                "cases": len(result),
                "ended_episodes": sum(r["completed_seeds"] for r in result),
                "checked_decisions": sum(r["steps_checked_ended"] for r in result),
                "successful_within_60s": sum(
                    r["successful_within_60s"] for r in result
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
