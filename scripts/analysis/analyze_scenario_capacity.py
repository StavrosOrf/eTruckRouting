"""Find energy-only map limits with one station per depot/customer segment.

For each eligible depot, count feasible one-delivery round trips and solve a
maximum-cardinality matching of feasible two-delivery round trips. These bounds
retain conservative traffic energy but ignore time, queues and payload. They
are distinct from observed success rates of the bounded random-walk generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.models.environment.joint_instance import GENERATOR_VERSION
from EVRoutingEnv.models.environment.route_walk import RouteWalkTopology
from EVRoutingEnv.utils.utils import load_config


def feasible_pairs(topology, depot):
    """Ordered pairs and one-customer trips under full-battery charging.

    Arrival is the largest battery attainable at the first customer, with zero
    or one station since the depot. Each pair preserves a direct/one-station
    return after its second customer. Diagonal entries exclude repeated tasks.
    """
    t = topology
    customers = t.customers[t.customers != depot]
    home = t._return_paths(depot, np.full(len(t.chargers), t.capacity))
    reachable = t.chargers[t.energy[depot, t.chargers] <= t.capacity + 1e-9]
    arrival = t.capacity - t.energy[depot, customers]
    if len(reachable):
        arrival = np.maximum(
            arrival,
            (t.capacity - t.energy[np.ix_(reachable, customers)]).max(axis=0),
        )
    single = arrival >= home.escape[customers] - 1e-9
    pairs = (
        t.energy[np.ix_(customers, customers)] + home.escape[customers]
        <= arrival[:, None] + 1e-9
    )
    if len(t.chargers):
        to_station = t.energy[np.ix_(customers, t.chargers)] <= arrival[:, None] + 1e-9
        from_station = (
            t.energy[np.ix_(t.chargers, customers)] + home.escape[customers]
            <= t.capacity + 1e-9
        )
        pairs |= to_station @ from_station
    np.fill_diagonal(pairs, False)
    return customers, single, pairs


def analyze(config_file):
    config = load_config(config_file)
    start = perf_counter()
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    initialization_ms = 1000 * (perf_counter() - start)
    traffic = config["traffic"]
    factor = (
        max(1.0, traffic["max_energy_multiplier"])
        if traffic["enable_traffic"] and traffic["enable_energy_uncertainty"]
        else 1.0
    )
    t = RouteWalkTopology.for_graph(
        env.transport_graph,
        env.charging_nodes,
        config["truck"]["battery_capacity"],
        factor,
    )
    rows, best_pairs = [], []
    try:
        for depot in t.customers:
            customers, single, pairs = feasible_pairs(t, int(depot))
            matching = nx.max_weight_matching(
                nx.from_numpy_array(pairs | pairs.T), maxcardinality=True
            )
            row = {
                "depot": t.nodes[int(depot)],
                "one_delivery_trucks": int(single.sum()),
                "two_delivery_trucks": len(matching),
                "two_delivery_deliveries": 2 * len(matching),
                "potential_pair_nodes": int(
                    (pairs.any(axis=0) | pairs.any(axis=1)).sum()
                ),
            }
            if not rows or row["two_delivery_trucks"] > max(
                r["two_delivery_trucks"] for r in rows
            ):
                best_pairs = []
                for a, b in sorted(tuple(sorted(pair)) for pair in matching):
                    if not pairs[a, b]:
                        a, b = b, a
                    best_pairs.append(
                        [t.nodes[int(customers[a])], t.nodes[int(customers[b])]]
                    )
            rows.append(row)
        best = max(rows, key=lambda r: r["two_delivery_trucks"])
        return {
            "config_file": config_file,
            "base_config": config,
            "generator_version": GENERATOR_VERSION,
            "map": Path(config["network"]["data_path"]).name,
            "map_nodes": len(t.nodes),
            "chargers": len(t.chargers),
            "eligible_depots_customers": len(t.customers),
            "node_delivery_ceiling": len(t.customers) - 1,
            "battery_kwh": t.capacity,
            "energy_multiplier": factor,
            "initialization_ms": initialization_ms,
            "total_analysis_ms": 1000 * (perf_counter() - start),
            "maximum_one_delivery_trucks": max(r["one_delivery_trucks"] for r in rows),
            "maximum_two_delivery_trucks": best["two_delivery_trucks"],
            "maximum_two_delivery_deliveries": best["two_delivery_deliveries"],
            "best_two_delivery_depot": best["depot"],
            "best_two_delivery_pairs": best_pairs,
            "depots_attaining_two_delivery_maximum": sum(
                r["two_delivery_trucks"] == best["two_delivery_trucks"] for r in rows
            ),
            "per_depot": rows,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", nargs="+", default=["EVRoutingEnv/config_files/config_joint.yaml"]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists")
    args.output.mkdir(parents=True)
    summaries = []
    for config in args.config:
        result = analyze(config)
        summaries.append(result)
        (args.output / f"{result['map']}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        print(
            f"{result['map']}: node ceiling {result['node_delivery_ceiling']}; "
            f"one-delivery maximum {result['maximum_one_delivery_trucks']} trucks; "
            f"two-delivery maximum {result['maximum_two_delivery_trucks']} trucks / "
            f"{result['maximum_two_delivery_deliveries']} deliveries; "
            f"{result['depots_attaining_two_delivery_maximum']} best depots",
            flush=True,
        )
    manifest = {
        "command": sys.argv,
        "definition": "Exact energy-only maxima for one or two deliveries per truck, full initial battery and full charging, one station per segment, unique customers and one shared depot. Ignore payload, event time, queues and generator search limits; not a seed-level certification claim.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "topology_sha256": hashlib.sha256(
            (ROOT / "EVRoutingEnv/models/environment/route_walk.py").read_bytes()
        ).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = [
        "# Energy-only generation limits",
        "",
        manifest["definition"],
        "",
        "| Map | Node delivery ceiling | Maximum trucks × 1 delivery | Maximum trucks × 2 deliveries | Best depot |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        rows.append(
            f"| {s['map']} | {s['node_delivery_ceiling']} | {s['maximum_one_delivery_trucks']} | "
            f"{s['maximum_two_delivery_trucks']} ({s['maximum_two_delivery_deliveries']} deliveries) | "
            f"{s['best_two_delivery_depot']} |"
        )
    rows += [
        "",
        "Each map JSON records every depot and the distinct customer pairs attaining its maximum. Limits apply to these delivery counts per truck; they do not bound longer routes with additional deliveries on the way home. Audit actual reset success separately with `analyze_scenario_resets.py`.",
    ]
    (args.output / "report.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
