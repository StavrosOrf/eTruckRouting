"""Compare California and Texas scenario construction, certification and reset.

    .venv/bin/python scripts/analysis/benchmark_scenario_generation.py --seeds 10
    .venv/bin/python scripts/analysis/benchmark_scenario_generation.py --synthetic \
        --trucks 1 10 100 --deliveries-per-truck 20 --seeds 5

Synthetic runs use the existing study's explicit hub-and-spoke map, never
silently expand geographic maps, and include stochastic travel and service.
Map initialization is reported separately. No scenario disk cache is used.
Charging-session counts and per-truck travel are measured outside reset timing.
Stored road-route distances are reported when available and compatible;
time-times-speed proxies remain explicitly labelled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.environment import event_driven_env as module
from EVRoutingEnv.models.environment.joint_instance import GENERATOR_VERSION
from EVRoutingEnv.utils.utils import load_config, resolve_network_data_dir


def route_metrics(env, road_distances=None):
    """Describe the certified witness outside the measured reset interval."""
    charges = [0] * env.num_trucks
    for tid, action in env.feasibility_witness:
        charges[tid] += int(action >= env.num_navigation_actions)
    energy, proxy, road = [], [], []
    for route in env.joint_instance.witness_routes:
        origin = env.joint_instance.depot_node
        kwh = hours = km = 0.0
        for target, _, charging in route:
            if charging:
                continue
            kwh += env.transport_graph.get_path_energy(origin, target)
            hours += env.transport_graph.get_time_distance(origin, target)
            if road_distances is not None:
                positions, distances = road_distances
                path = env.transport_graph.get_shortest_path(origin, target)
                km += sum(
                    float(distances[positions[a], positions[b]])
                    for a, b in pairwise(path)
                )
            origin = target
        energy.append(kwh)
        proxy.append(hours * env.config["truck"]["base_speed"])
        road.append(km)
    return {
        "charging_sessions_per_truck": charges,
        "nominal_kwh_per_truck": energy,
        "distance_proxy_km_per_truck": proxy,
        "road_km_per_truck": road if road_distances is not None else None,
    }


def load_road_distances(env, config):
    folder = Path(resolve_network_data_dir(config["network"]))
    source = folder / "vis_data" / "od_matrices.npz"
    metadata = folder / "metadata.json"
    # California's visualisation distances were reconstructed separately from
    # its original energy/time data; they do not describe simulator routes.
    if not source.exists() or (
        metadata.exists()
        and not json.loads(metadata.read_text()).get("generated", True)
    ):
        return None
    with np.load(source) as data:
        positions = {int(n): i for i, n in enumerate(data["node_ids"])}
        original = {
            n: env.transport_graph.graph.nodes[n].get("original_id", n)
            for n in env.transport_graph.graph
        }
        if not all(n in positions for n in original.values()):
            return None
        return (
            {n: positions[int(original_id)] for n, original_id in original.items()},
            data["dist_km"],
        )


def benchmark(config, seeds, *, map_name=None):
    config["environment"].pop("scenario_cache_dir", None)
    map_name = map_name or Path(config["network"]["data_path"]).name
    original = module.generate_joint_routing_instance
    generation, certification = [], []

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

    start = perf_counter()
    env = module.EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    initialization = (perf_counter() - start) * 1000
    road_distances = load_road_distances(env, config)
    records = []
    routes = []
    try:
        with patch.object(module, "generate_joint_routing_instance", timed_generate):
            for seed in range(seeds):
                generation.clear()
                certification.clear()
                start = perf_counter()
                env.reset(seed=seed)
                total = (perf_counter() - start) * 1000
                replay = sum(certification)
                records.append(
                    (
                        sum(generation) - replay,
                        replay,
                        total,
                        env.joint_instance.generation_attempts_used,
                    )
                )
                routes.append({"seed": seed, **route_metrics(env, road_distances)})
    finally:
        env.close()
    values = np.asarray(records)
    print(
        f"map={map_name} trucks={config['environment']['num_trucks']} customers={config['environment']['num_stops']} "
        f"initialization_ms={initialization:.2f} seeds={seeds}"
    )
    for column, name in enumerate(("construction", "certification", "complete_reset")):
        print(
            f"  {name:16s} p50={np.median(values[:, column]):9.2f} ms "
            f"p95={np.percentile(values[:, column], 95):9.2f} ms "
            f"max={values[:, column].max():9.2f} ms"
        )
    print(f"  candidate_attempts max={int(values[:, 3].max())}", flush=True)
    for key in (
        "charging_sessions_per_truck",
        "nominal_kwh_per_truck",
        "distance_proxy_km_per_truck",
        "road_km_per_truck",
    ):
        samples = [v for row in routes for v in (row[key] or [])]
        if samples:
            print(
                f"  {key} mean={np.mean(samples):.2f} min={min(samples):.2f} max={max(samples):.2f}",
                flush=True,
            )
    return {
        "map": map_name,
        "generator_version": GENERATOR_VERSION,
        "config": config,
        "initialization_ms": initialization,
        "timing_columns": [
            "construction_ms",
            "certification_ms",
            "reset_ms",
            "attempts",
        ],
        "timings": records,
        "routes": routes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        nargs="+",
        help="Configs to compare; defaults to California and Texas.",
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--trucks", type=int, nargs="+")
    parser.add_argument("--deliveries-per-truck", type=int)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the complete benchmark report to a new JSON file.",
    )
    args = parser.parse_args()
    if args.seeds < 1 or (args.trucks and min(args.trucks) < 1):
        parser.error("seeds and trucks must be positive")
    if args.deliveries_per_truck is not None and args.deliveries_per_truck < 1:
        parser.error("deliveries-per-truck must be positive")
    if args.output is not None and args.output.exists():
        parser.error("output file already exists")
    os.environ.pop("EVRP_SCENARIO_CACHE", None)
    configs = args.config or ["EVRoutingEnv/config_files/config_joint.yaml"]
    if args.config is None and not args.synthetic:
        configs.append("EVRoutingEnv/config_files/config_joint_texas.yaml")
    reports = []
    for name in configs:
        base = load_config(name)
        fleets = args.trucks or [base["environment"]["num_trucks"]]
        reports.extend(run_fleets(base, fleets, args))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as file:
            json.dump(reports, file, indent=2, allow_nan=False)


def run_fleets(base, fleets, args):
    reports = []
    for trucks in fleets:
        from copy import deepcopy

        config = deepcopy(base)
        config["environment"]["num_trucks"] = trucks
        if args.deliveries_per_truck is not None:
            config["environment"]["num_stops"] = trucks * args.deliveries_per_truck
            config["problem"]["payload_capacity"] = (
                args.deliveries_per_truck * config["problem"]["max_customer_demand"]
            )
        if args.synthetic:
            import networkx as nx

            from experiments.scenario_generation_ablation.run_scenario_grid import (
                SyntheticHubSpokeTransportationGraph,
                expand_customer_nodes,
            )

            graph, _, _ = expand_customer_nodes(
                nx.DiGraph(), config["environment"]["num_stops"]
            )
            config["environment"]["max_time"] = 5000.0
            with (
                patch.object(module, "get_graph", lambda _, graph=graph: graph.copy()),
                patch.object(
                    module,
                    "TransportationGraph",
                    SyntheticHubSpokeTransportationGraph,
                ),
            ):
                report = benchmark(config, args.seeds, map_name="synthetic_hub_spoke")
                reports.append(report)
        else:
            reports.append(benchmark(config, args.seeds))
    return reports


if __name__ == "__main__":
    main()
