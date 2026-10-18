"""Audit reset distributions using the generator's certified fleet witnesses.

Example (deliveries are explicitly per truck):
    .venv/bin/python scripts/analysis/analyze_scenario_resets.py \
        --case 10:10 --case 100:5 --resets 100 \
        --output experiments/scenario_generation_ablation/results/texas_reset_audit

Inspired by analyze_delivery_routes.py and the scenario-generation ablation.
Reuses each environment, counts actual witness ownership/order rather than
shared delivery_sequence lists, and retains every failed reset in the report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.core.transportation_graph import ROUTING_POLICY_VERSION
from EVRoutingEnv.models.environment import event_driven_env as module
from EVRoutingEnv.models.environment.joint_instance import GENERATOR_VERSION
from EVRoutingEnv.utils.utils import load_config, resolve_network_data_dir
from scripts.analysis.benchmark_scenario_generation import load_road_distances


HOP_FIELDS = (
    "case",
    "seed",
    "truck_id",
    "kind",
    "delivery_position",
    "from_node",
    "to_node",
    "origin_kind",
    "target_kind",
    "energy_kwh",
    "travel_hours",
    "road_km",
    "distance_proxy_km",
    "planned_charges",
)
TRUCK_FIELDS = (
    "case",
    "seed",
    "truck_id",
    "deliveries",
    "demand",
    "payload_utilization",
    "energy_kwh",
    "travel_hours",
    "road_km",
    "distance_proxy_km",
    "navigation_legs",
    "charger_visits",
    "unique_chargers",
    "max_chargers_between_stops",
    "planned_charges",
    "certified_charges",
    "certificate_actions",
)
RESET_FIELDS = (
    "case",
    "seed",
    "trucks",
    "deliveries_per_truck",
    "total_deliveries",
    "status",
    "certified",
    "reset_ms",
    "construction_ms",
    "certification_ms",
    "generation_attempts",
    "witness_steps",
    "depot_node",
    "scenario_sha256",
    "error",
)


def distribution(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    q1, median, q3, p90, p95, p99 = np.percentile(values, [25, 50, 75, 90, 95, 99])
    iqr = q3 - q1
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "q25": float(q1),
        "median": float(median),
        "q75": float(q3),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(values.max()),
        "iqr_outliers": int(
            ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()
        ),
    }


class LegLookup:
    """Cache nominal route costs and stored road kilometres outside reset timing."""

    def __init__(self, env):
        self.graph = env.transport_graph
        self.road = load_road_distances(env, env.config)
        self.speed = float(env.config["truck"]["base_speed"])
        self.cache = {}

    def __call__(self, origin, destination):
        key = (int(origin), int(destination))
        if key not in self.cache:
            energy, hours = self.graph.get_leg_costs(*key)
            km = None
            if self.road is not None and math.isfinite(energy):
                positions, distance = self.road
                path = self.graph.get_shortest_path(*key)
                if not path:
                    raise ValueError(f"no explicit path for finite-cost leg {key}")
                from itertools import pairwise

                km = sum(
                    float(distance[positions[a], positions[b]])
                    for a, b in pairwise(path)
                )
            self.cache[key] = {
                "energy_kwh": float(energy),
                "travel_hours": float(hours),
                "road_km": km,
                "distance_proxy_km": float(hours) * self.speed,
            }
        return self.cache[key]


def extract_witness(env, seed, case, lookup):
    """One direct delivery hop per customer; each navigation leg counted once.

    delivery_direct is the direct OD cost from the previous customer/depot.
    delivery_travel sums the witness's actual travel, including charger detours.
    return_travel is separate, so it cannot inflate delivery-hop statistics.
    All travel costs are nominal; charging counts use the certified action trace.
    """
    instance = env.joint_instance
    tasks = {t.node_id: t for t in instance.tasks}
    chargers = set(env.charging_nodes)
    depot = instance.depot_node
    hops, trucks = [], []
    certified_charges = Counter(
        tid for tid, a in env.feasibility_witness if a >= env.num_navigation_actions
    )
    actions = Counter(tid for tid, _ in env.feasibility_witness)

    def node_kind(node):
        return (
            "depot" if node == depot else "charger" if node in chargers else "customer"
        )

    def hop(tid, kind, position, origin, target, costs, charges=0):
        return {
            "case": case,
            "seed": seed,
            "truck_id": tid,
            "kind": kind,
            "delivery_position": position,
            "from_node": int(origin),
            "to_node": int(target),
            "origin_kind": node_kind(origin),
            "target_kind": node_kind(target),
            **costs,
            "planned_charges": charges,
        }

    def zero_costs():
        return {
            "energy_kwh": 0.0,
            "travel_hours": 0.0,
            "road_km": 0.0 if lookup.road is not None else None,
            "distance_proxy_km": 0.0,
        }

    owners = {}
    for task, owner in zip(instance.tasks, instance.witness_owners, strict=True):
        owners[task.node_id] = owner
    served = set()
    for tid, route in enumerate(instance.witness_routes):
        node = previous_delivery = depot
        totals, segment = zero_costs(), zero_costs()
        delivery_count = leg_count = charger_visits = planned_charges = (
            segment_charges
        ) = 0
        demand = 0.0
        visited = set()
        segment_stations = max_segment_stations = 0
        for target, _, charging in route:
            target = int(target)
            if charging:
                planned_charges += 1
                segment_charges += 1
                continue
            if target == node:
                continue  # retirement at the depot is not a travel hop
            costs = lookup(node, target)
            hops.append(hop(tid, "navigation", delivery_count, node, target, costs))
            leg_count += 1
            for key, value in costs.items():
                if value is not None:
                    totals[key] += value
                    segment[key] += value
            if target in chargers:
                charger_visits += 1
                visited.add(target)
                segment_stations += 1
                max_segment_stations = max(max_segment_stations, segment_stations)
                if segment_stations > 1:
                    raise ValueError("witness visits consecutive charging stations")
            elif target in tasks:
                if target in served or owners[target] != tid:
                    raise ValueError(
                        "witness repeats a customer or violates its saved partition"
                    )
                served.add(target)
                hops.append(
                    hop(
                        tid,
                        "delivery_direct",
                        delivery_count,
                        previous_delivery,
                        target,
                        lookup(previous_delivery, target),
                    )
                )
                hops.append(
                    hop(
                        tid,
                        "delivery_travel",
                        delivery_count,
                        previous_delivery,
                        target,
                        segment.copy(),
                        segment_charges,
                    )
                )
                demand += tasks[target].demand
                delivery_count += 1
                previous_delivery = target
                segment = zero_costs()
                segment_charges = 0
                segment_stations = 0
            elif target == depot:
                hops.append(
                    hop(
                        tid,
                        "return_travel",
                        delivery_count,
                        previous_delivery,
                        depot,
                        segment.copy(),
                        segment_charges,
                    )
                )
                segment = zero_costs()
                segment_charges = 0
                segment_stations = 0
            node = target
        if node != depot:
            raise ValueError("witness does not end at its depot")
        trucks.append(
            {
                "case": case,
                "seed": seed,
                "truck_id": tid,
                "deliveries": delivery_count,
                "demand": demand,
                "payload_utilization": demand / env.trucks[tid].payload_capacity,
                **totals,
                "navigation_legs": leg_count,
                "charger_visits": charger_visits,
                "unique_chargers": len(visited),
                "max_chargers_between_stops": max_segment_stations,
                "planned_charges": planned_charges,
                "certified_charges": certified_charges[tid],
                "certificate_actions": actions[tid],
            }
        )
    if served != set(tasks):
        raise ValueError("witness does not cover all customers")
    return hops, trucks


def summarize_case(name, resets, hops, trucks, metadata):
    good = [row for row in resets if row["certified"]]
    result = {
        "case": name,
        **metadata,
        "requested_resets": len(resets),
        "certified_resets": len(good),
        "failed_resets": len(resets) - len(good),
        "certification_rate": len(good) / len(resets),
        "statuses": dict(Counter(r["status"] for r in resets)),
        "errors": list(dict.fromkeys(r["error"] for r in resets if r["error"])),
        "timing_ms": {
            key: distribution(r[key] for r in resets)
            for key in ("reset_ms", "construction_ms", "certification_ms")
        },
        "generation_attempts": distribution(r["generation_attempts"] for r in good),
        "truck_statistics": {},
        "hop_statistics": {},
    }
    for field in (
        "deliveries",
        "demand",
        "payload_utilization",
        "energy_kwh",
        "travel_hours",
        "road_km",
        "distance_proxy_km",
        "certified_charges",
        "charger_visits",
        "max_chargers_between_stops",
    ):
        result["truck_statistics"][field] = distribution(t[field] for t in trucks)
    for kind in ("delivery_direct", "delivery_travel", "navigation", "return_travel"):
        group = [h for h in hops if h["kind"] == kind]
        result["hop_statistics"][kind] = {
            field: distribution(h[field] for h in group)
            for field in ("energy_kwh", "travel_hours", "road_km", "distance_proxy_km")
        }
    result["coverage"] = {
        "distinct_depots": len({r["depot_node"] for r in good}),
        "distinct_scenarios": len({r["scenario_sha256"] for r in good}),
        "distinct_delivery_nodes": len(
            {h["to_node"] for h in hops if h["kind"] == "delivery_direct"}
        ),
    }
    return result


def run_case(base, case, seeds, output, writers, details):
    truck_count, per_truck = case
    name = f"{truck_count}t_{per_truck}d_per_truck"
    config = deepcopy(base)
    config["environment"].update(
        num_trucks=truck_count,
        num_stops=truck_count * per_truck,
        allow_variable_num_stops=False,
    )
    config["environment"].pop("instance_num_stops", None)
    config["environment"].pop("scenario_cache_dir", None)
    config["problem"]["payload_capacity"] = (
        per_truck * config["problem"]["max_customer_demand"]
    )
    (output / f"{name}_config.json").write_text(json.dumps(config, indent=2) + "\n")
    start = perf_counter()
    env = module.EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    initialization_ms = (perf_counter() - start) * 1000
    lookup = LegLookup(env)
    eligible = [
        n
        for n in env.transport_graph.get_all_nodes()
        if n not in set(env.charging_nodes)
        and env.transport_graph.graph.in_degree(n) > 0
        and env.transport_graph.graph.out_degree(n) > 0
    ]
    ceiling = len(eligible) - 1
    print(
        f"{name}: {env.num_stops} total deliveries; map ceiling {ceiling}; initialization {initialization_ms:.0f} ms",
        flush=True,
    )
    resets, all_hops, all_trucks = [], [], []
    generation, certification = [], []
    original = module.generate_joint_routing_instance

    def timed_generate(*args, **kwargs):
        certify = kwargs["certify"]

        def timed_certify(instance):
            start = perf_counter()
            try:
                return certify(instance)
            finally:
                certification.append((perf_counter() - start) * 1000)

        kwargs["certify"] = timed_certify
        start = perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            generation.append((perf_counter() - start) * 1000)

    try:
        with patch.object(module, "generate_joint_routing_instance", timed_generate):
            for index, seed in enumerate(seeds):
                generation.clear()
                certification.clear()
                record = dict.fromkeys(RESET_FIELDS)
                record.update(
                    case=name,
                    seed=seed,
                    trucks=truck_count,
                    deliveries_per_truck=per_truck,
                    total_deliveries=truck_count * per_truck,
                    certified=False,
                    error="",
                )
                start = perf_counter()
                try:
                    _, info = env.reset(seed=seed)
                    record["reset_ms"] = (perf_counter() - start) * 1000
                    if (
                        not env.feasibility_witness
                        or not info["instance_feasibility"]["certified"]
                    ):
                        raise ValueError(
                            "reset returned no seeded event-engine certificate"
                        )
                    record.update(
                        status="certified",
                        certified=True,
                        generation_attempts=env.joint_instance.generation_attempts_used,
                        witness_steps=len(env.feasibility_witness),
                        depot_node=env.joint_instance.depot_node,
                        scenario_sha256=hashlib.sha256(
                            json.dumps(info["scenario"], sort_keys=True).encode()
                        ).hexdigest(),
                    )
                except (ValueError, RuntimeError) as error:
                    record["reset_ms"] = (perf_counter() - start) * 1000
                    record.update(
                        status="insufficient_map_nodes"
                        if env.num_stops > ceiling
                        else "generation_failed",
                        error=f"{type(error).__name__}: {error}",
                    )
                record["certification_ms"] = sum(certification)
                record["construction_ms"] = max(
                    0.0, sum(generation) - sum(certification)
                )
                if record["certified"]:
                    hops, trucks = extract_witness(env, seed, name, lookup)
                    for row in hops:
                        writers["hops"].writerow(row)
                    for row in trucks:
                        writers["trucks"].writerow(row)
                    all_hops.extend(hops)
                    all_trucks.extend(trucks)
                    details.write(
                        json.dumps(
                            {
                                "case": name,
                                "seed": seed,
                                "scenario": info["scenario"],
                                "witness_owners": env.joint_instance.witness_owners,
                                "witness_routes": env.joint_instance.witness_routes,
                                "certified_trace": env.feasibility_witness,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                writers["resets"].writerow(record)
                resets.append(record)
                if (index + 1) % 10 == 0 or index + 1 == len(seeds):
                    print(
                        f"  {index + 1}/{len(seeds)} resets; certified {sum(r['certified'] for r in resets)}",
                        flush=True,
                    )
        summary = summarize_case(
            name,
            resets,
            all_hops,
            all_trucks,
            {
                "trucks": truck_count,
                "deliveries_per_truck": per_truck,
                "total_deliveries": env.num_stops,
                "eligible_customer_ceiling": ceiling,
                "map_nodes": len(env.transport_graph.get_all_nodes()),
                "chargers": env.num_charging_nodes,
                "initialization_ms": initialization_ms,
                "road_km_available": lookup.road is not None,
                "resolved_config": f"{name}_config.json",
            },
        )
        return summary, resets, all_hops, all_trucks
    finally:
        env.close()


def make_report(output, summaries, datasets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    panels = [
        ("delivery_direct", "road_km", "Direct delivery-hop road km"),
        (
            "delivery_travel",
            "road_km",
            "Delivery travel with charger detours (road km)",
        ),
        ("delivery_direct", "energy_kwh", "Direct delivery-hop energy (kWh)"),
        ("truck", "road_km", "Per-truck full-route road km"),
        ("truck", "certified_charges", "Certified charging sessions per truck"),
        ("reset", "reset_ms", "Reset wall time (ms)"),
    ]
    for ax, (kind, field, title) in zip(axes.flat, panels, strict=True):
        for name, resets, hops, trucks in datasets:
            if kind == "reset":
                rows = [r for r in resets if r["certified"]]
            elif kind == "truck":
                rows = trucks
            else:
                rows = [r for r in hops if r["kind"] == kind]
            values = np.sort(
                [
                    r[field]
                    for r in rows
                    if r[field] is not None and math.isfinite(r[field])
                ]
            )
            if len(values):
                ax.step(
                    values,
                    np.arange(1, len(values) + 1) / len(values),
                    where="post",
                    label=name,
                )
        ax.set(xlabel=title, ylabel="Cumulative fraction", ylim=(0, 1.02))
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "No successful-reset data",
                ha="center",
                transform=ax.transAxes,
            )
    fig.suptitle("Certified reset distributions — witness routes, nominal travel costs")
    fig.savefig(output / "distributions.png", dpi=180)
    fig.savefig(output / "distributions.svg")
    plt.close(fig)
    lines = [
        "# Scenario reset audit",
        "",
        "Each case requests 100 resets unless overridden on the command line. Deliveries are per truck; environment.num_stops is the fleet total. Each environment is initialized once. Scenario disk caching is disabled; map lookup caching remains enabled. Timing excludes metric extraction and plotting.",
        "",
        "The routes analyzed are the hidden feasibility witnesses. In joint-fleet mode, ownership/order in these witnesses are not policy constraints. Successful reset certification uses the seeded event engine, including queues, stochastic travel/service, windows, payload, return-to-depot, and episode budgets. No heuristic rollout is substituted.",
        "",
        "Every generated witness is checked for at most one charging-station visit between consecutive depot/customer stops, including departure from and return to the depot. `trucks.csv` records `max_chargers_between_stops`; repeated station visits count even if no charging action was executed.",
        "",
        "`delivery_direct` measures direct previous-customer/depot → next-customer costs. `delivery_travel` accumulates actual navigation through intervening charging stations. `navigation` counts each physical witness leg once, and `return_travel` separates the final depot return. Direct delivery hops may exceed one battery when the witness uses chargers. Nominal kWh, nominal travel hours, road-route kilometres, and time-times-speed proxy kilometres are separate fields.",
        "",
        "| Case | Certified / requested | Reset median / p95 ms | Direct delivery hop mean / p95 road km | Delivery travel mean road km | Full route mean road km | Charges per truck mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def value(d, field="mean"):
        return f"{d[field]:.2f}" if d.get("count") else "—"

    for summary in summaries:
        hop = summary["hop_statistics"]["delivery_direct"]["road_km"]
        timing = summary["timing_ms"]["reset_ms"]
        lines.append(
            f"| {summary['case']} | {summary['certified_resets']} / {summary['requested_resets']} | "
            f"{value(timing, 'median')} / {value(timing, 'p95')} | {value(hop)} / {value(hop, 'p95')} | "
            f"{value(summary['hop_statistics']['delivery_travel']['road_km'])} | "
            f"{value(summary['truck_statistics']['road_km'])} | {value(summary['truck_statistics']['certified_charges'])} |"
        )
        if summary["failed_resets"]:
            lines.extend(
                [
                    "",
                    f"**{summary['case']}:** {summary['failed_resets']} rejected resets. "
                    f"The map has room for at most {summary['eligible_customer_ceiling']} distinct deliveries plus a depot, "
                    f"before energy/route constraints; this case requests {summary['total_deliveries']}. Errors: "
                    + "; ".join(summary["errors"]),
                    "",
                ]
            )
    lines += [
        "",
        "Empty cells indicate absent successful-reset data, not zero-length routes. Rejected-reset timings describe rejection, not successful generation.",
        "",
        "![Distribution plots](distributions.png)",
        "",
        "Files: `resets.csv`, `trucks.csv`, `hops.csv`, `scenarios.jsonl`, `summary.json`, `summary.csv`, per-case resolved configs, and `manifest.json`. Full percentiles, standard deviations and IQR outlier counts are in `summary.json`. Road km are summed from compatible stored OD lengths along the simulator's energy-optimal paths. When these lengths are unavailable, road-km fields remain empty.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def parse_case(value):
    try:
        trucks, deliveries = map(int, value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use TRUCKS:DELIVERIES_PER_TRUCK") from error
    if min(trucks, deliveries) < 1:
        raise argparse.ArgumentTypeError("trucks and deliveries must be positive")
    return trucks, deliveries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="EVRoutingEnv/config_files/config_joint_texas.yaml"
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help="TRUCKS:DELIVERIES_PER_TRUCK; repeat for multiple cases",
    )
    parser.add_argument("--resets", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.resets < 1:
        parser.error("resets must be positive")
    if args.output.exists():
        parser.error("output directory already exists; choose a new directory")
    args.output.mkdir(parents=True)
    os.environ.pop("EVRP_SCENARIO_CACHE", None)
    config = load_config(args.config)
    cases = args.case or [(10, 10), (100, 5)]
    seeds = list(range(args.seed_start, args.seed_start + args.resets))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    manifest = {
        "schema_version": "evrp-reset-audit-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "config_file": args.config,
        "base_config": config,
        "cases": cases,
        "seeds": seeds,
        "generator_version": GENERATOR_VERSION,
        "routing_policy": ROUTING_POLICY_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": commit.stdout.strip(),
        "scenario_disk_cache": False,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__).resolve(),
                ROOT / "EVRoutingEnv/models/environment/route_walk.py",
                ROOT / "EVRoutingEnv/models/environment/joint_instance.py",
                ROOT / "EVRoutingEnv/models/environment/event_driven_env.py",
                ROOT / "EVRoutingEnv/models/core/transportation_graph.py",
            ]
        },
    }
    folder = Path(resolve_network_data_dir(config["network"]))
    inputs = [
        folder / config["network"][key]
        for key in (
            "shortest_path_energy_file",
            "shortest_path_time_file",
            "station_info_file",
        )
    ]
    inputs += [folder / "metadata.json", folder / "vis_data/od_matrices.npz"]
    manifest["network_sha256"] = {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs
        if p.exists()
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    from contextlib import ExitStack

    summaries, datasets = [], []
    with ExitStack() as stack:
        writers = {}
        for name, fields in (
            ("resets", RESET_FIELDS),
            ("trucks", TRUCK_FIELDS),
            ("hops", HOP_FIELDS),
        ):
            stream = stack.enter_context(
                (args.output / f"{name}.csv").open("w", newline="")
            )
            writers[name] = csv.DictWriter(stream, fieldnames=fields)
            writers[name].writeheader()
        details = stack.enter_context((args.output / "scenarios.jsonl").open("w"))
        for case in cases:
            summary, resets, hops, trucks = run_case(
                config, case, seeds, args.output, writers, details
            )
            summaries.append(summary)
            datasets.append((summary["case"], resets, hops, trucks))
    (args.output / "summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False) + "\n"
    )
    with (args.output / "summary.csv").open("w", newline="") as file:
        columns = (
            "case",
            "section",
            "metric",
            "count",
            "mean",
            "std",
            "min",
            "q25",
            "median",
            "q75",
            "p90",
            "p95",
            "p99",
            "max",
            "iqr_outliers",
        )
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            sections = {
                "timing_ms": summary["timing_ms"],
                "truck": summary["truck_statistics"],
                **summary["hop_statistics"],
            }
            for section, metrics in sections.items():
                for name, stats in metrics.items():
                    writer.writerow(
                        {
                            "case": summary["case"],
                            "section": section,
                            "metric": name,
                            **stats,
                        }
                    )
    make_report(args.output, summaries, datasets)
    print(f"Saved {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
