"""Plot five certified preassigned truck paths on California and Texas.

Default scenarios use T3D3: three trucks and three deliveries per truck.
One route is selected from each of five seeded scenarios per map, rotating
across trucks so the figure shows five routes and all three assignments.
Routes are generator witnesses, not policy rollouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.models.environment.joint_instance import GENERATOR_VERSION
from EVRoutingEnv.utils.utils import load_config, resolve_network_data_dir


CONFIGS = {
    "California": "EVRoutingEnv/config_files/config_joint.yaml",
    "Texas": "EVRoutingEnv/config_files/config_joint_texas.yaml",
}
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def geographic_data(env, config):
    vis = Path(resolve_network_data_dir(config["network"])) / "vis_data"
    node_records = json.loads((vis / "nodes.json").read_text())
    coords = {
        int(row["node"]): (float(row["lon"]), float(row["lat"])) for row in node_records
    }
    graph = env.transport_graph.graph
    node_coords = {}
    for node, attributes in graph.nodes(data=True):
        original = int(attributes.get("original_id", node))
        if original in coords:
            node_coords[int(node)] = coords[original]

    roads_file = vis / "roads.npz"
    road_segments = np.empty((0, 2, 2), dtype=float)
    if roads_file.exists():
        with np.load(roads_file) as data:
            major = np.flatnonzero(data["cls"] <= 9)
            stride = max(1, math.ceil(len(major) / 18000))
            indices = major[::stride]
            road_segments = (
                np.stack(
                    [
                        np.column_stack((data["lon0"][indices], data["lat0"][indices])),
                        np.column_stack((data["lon1"][indices], data["lat1"][indices])),
                    ],
                    axis=1,
                ).astype(float)
                / 1e5
            )
    return vis, node_coords, road_segments


def collect_scenarios(name, config_file, seeds, trucks, deliveries):
    config = load_config(config_file)
    config["environment"].update(
        num_trucks=trucks,
        num_stops=trucks * deliveries,
        allow_variable_num_stops=False,
    )
    config["environment"].pop("scenario_cache_dir", None)
    config["problem"]["assignment"] = "preassigned"
    config["problem"]["payload_capacity"] = (
        deliveries * config["problem"]["max_customer_demand"]
    )
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    vis, coordinates, roads = geographic_data(env, config)
    scenarios, selected = [], []
    try:
        for seed in seeds:
            _, info = env.reset(seed=seed)
            if (
                not env.feasibility_witness
                or not info["instance_feasibility"]["certified"]
            ):
                raise RuntimeError(
                    f"{name} seed {seed} returned an uncertified scenario"
                )
            instance = env.joint_instance
            owners = {
                task.node_id: instance.witness_owners[task.task_id]
                for task in instance.tasks
            }
            if any(
                task.preassigned_to != owners[task.node_id] for task in instance.tasks
            ):
                raise RuntimeError(f"{name} seed {seed} lost its preassigned ownership")
            scenario = {
                "seed": seed,
                "depot": int(instance.depot_node),
                "tasks": [
                    {
                        "node": int(task.node_id),
                        "truck": int(task.preassigned_to),
                        "sequence_position": int(task.sequence_position)
                        if task.sequence_position is not None
                        else None,
                        "demand": float(task.demand),
                    }
                    for task in instance.tasks
                ],
                "routes": [],
            }
            for truck_id, route in enumerate(instance.witness_routes):
                stops = []
                origin = int(instance.depot_node)
                navigation_paths = []
                for target, _, charging in route:
                    target = int(target)
                    if charging:
                        if stops and stops[-1]["node"] == target:
                            stops[-1]["charge"] = True
                        continue
                    if target == origin:
                        continue
                    path = env.transport_graph.get_shortest_path(origin, target)
                    if not path:
                        raise RuntimeError(f"Missing map path for {origin} → {target}")
                    path = [int(node) for node in path]
                    missing = [node for node in path if node not in coordinates]
                    if missing:
                        raise RuntimeError(
                            f"Missing geographic coordinates for nodes {missing[:3]}"
                        )
                    navigation_paths.append(path)
                    stops.append(
                        {
                            "node": target,
                            "kind": (
                                "charger"
                                if target in env.charging_nodes
                                else "delivery"
                                if target in owners
                                else "depot"
                            ),
                            "charge": False,
                        }
                    )
                    origin = target
                delivery_stops = [stop for stop in stops if stop["kind"] == "delivery"]
                if len(delivery_stops) != deliveries:
                    raise RuntimeError(
                        f"{name} seed {seed} truck {truck_id}: expected {deliveries} deliveries, "
                        f"found {len(delivery_stops)}"
                    )
                item = {
                    "truck": truck_id,
                    "deliveries": [stop["node"] for stop in delivery_stops],
                    "charges": sum(int(stop["charge"]) for stop in stops),
                    "stops": stops,
                    "navigation_paths": navigation_paths,
                }
                scenario["routes"].append(item)
            scenarios.append(scenario)
            truck_id = len(selected) % trucks
            selected.append(
                {"seed": seed, "truck": truck_id, "route": scenario["routes"][truck_id]}
            )
    finally:
        env.close()
    return {
        "map": name,
        "config_file": config_file,
        "network_data_dir": str(vis.parent),
        "node_count": len(coordinates),
        "truck_count": trucks,
        "deliveries_per_truck": deliveries,
        "seeds": list(seeds),
        "scenarios": scenarios,
        "selected_paths": selected,
        "_coordinates": coordinates,
        "_roads": roads,
    }


def plot_maps(results, output, trucks, deliveries):
    fig, axes = plt.subplots(1, 2, figsize=(19, 10), dpi=180)
    for ax, result in zip(axes, results, strict=True):
        coordinates = result["_coordinates"]
        roads = result["_roads"]
        ax.add_collection(
            LineCollection(
                roads, colors="#cbd1d6", linewidths=0.34, alpha=0.7, zorder=1
            )
        )
        all_lons = [point[0] for point in coordinates.values()]
        all_lats = [point[1] for point in coordinates.values()]
        for index, selected in enumerate(result["selected_paths"]):
            route = selected["route"]
            color = PALETTE[index % len(PALETTE)]
            for path in route["navigation_paths"]:
                xy = np.asarray([coordinates[node] for node in path])
                ax.plot(
                    xy[:, 0],
                    xy[:, 1],
                    color=color,
                    linewidth=1.65,
                    alpha=0.84,
                    zorder=4,
                )
            delivery_stops = [
                stop for stop in route["stops"] if stop["kind"] == "delivery"
            ]
            for position, stop in enumerate(delivery_stops, start=1):
                lon, lat = coordinates[stop["node"]]
                ax.scatter(
                    lon,
                    lat,
                    s=34,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=6,
                )
                ax.annotate(
                    str(position),
                    (lon, lat),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    color="#202830",
                    zorder=7,
                )
            for stop in route["stops"]:
                if stop["kind"] == "charger":
                    lon, lat = coordinates[stop["node"]]
                    ax.scatter(
                        lon,
                        lat,
                        s=27,
                        marker="D",
                        color=color,
                        edgecolor="white",
                        linewidth=0.5,
                        zorder=5,
                    )
            depot = result["scenarios"][index]["depot"]
            lon, lat = coordinates[depot]
            ax.scatter(
                lon,
                lat,
                s=100,
                marker="*",
                color="#f4c542",
                edgecolor="#4c3b00",
                linewidth=0.7,
                zorder=8,
            )
        margin_x = max((max(all_lons) - min(all_lons)) * 0.035, 0.08)
        margin_y = max((max(all_lats) - min(all_lats)) * 0.035, 0.08)
        ax.set_xlim(min(all_lons) - margin_x, max(all_lons) + margin_x)
        ax.set_ylim(min(all_lats) - margin_y, max(all_lats) + margin_y)
        ax.set_aspect(1 / math.cos(math.radians((min(all_lats) + max(all_lats)) / 2)))
        ax.set_title(
            f"{result['map']} · five certified assigned paths",
            fontsize=14,
            weight="bold",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.18, linewidth=0.5)
    route_handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE[i],
            lw=2,
            label=f"Seed {p['seed']} · Truck {p['truck'] + 1}",
        )
        for i, p in enumerate(results[0]["selected_paths"])
    ]
    node_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            color="#f4c542",
            markeredgecolor="#4c3b00",
            label="Depot",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="#657786",
            label="Delivery (numbered in order)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            color="#657786",
            label="Charging stop",
        ),
        Line2D([0], [0], color="#cbd1d6", lw=2, label="Major roads"),
    ]
    fig.legend(
        handles=route_handles + node_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=9,
    )
    fig.suptitle(
        f"Preassigned certified witness routes · {trucks} trucks × {deliveries} deliveries per truck",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.5,
        0.935,
        "Five routes from seeds 0–4 per map; each scenario contains all three preassigned truck routes. "
        "Lines follow the energy optimal road path; charging diamonds are planned stops.",
        ha="center",
        fontsize=10,
        color="#38434d",
    )
    fig.tight_layout(rect=(0.015, 0.075, 0.985, 0.91))
    fig.savefig(output / "assigned_paths_T3D3.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "assigned_paths_T3D3.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trucks", type=int, default=3)
    parser.add_argument("--deliveries", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/scenario_generation_ablation/results/assigned_paths_T3D3_20261018"
        ),
    )
    args = parser.parse_args()
    if args.trucks < 1 or args.deliveries < 1 or not args.seeds:
        parser.error("trucks, deliveries and at least one seed must be positive")
    if args.output.exists():
        parser.error("output directory already exists; choose a new directory")
    args.output.mkdir(parents=True)
    results = [
        collect_scenarios(name, config, args.seeds, args.trucks, args.deliveries)
        for name, config in CONFIGS.items()
    ]
    plot_maps(results, args.output, args.trucks, args.deliveries)
    serializable = [
        {key: value for key, value in result.items() if not key.startswith("_")}
        for result in results
    ]
    (args.output / "assigned_paths.json").write_text(
        json.dumps(serializable, indent=2, allow_nan=False) + "\n"
    )
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "assignment": "preassigned",
        "trucks": args.trucks,
        "deliveries_per_truck": args.deliveries,
        "total_deliveries_per_scenario": args.trucks * args.deliveries,
        "seeds": args.seeds,
        "selected_trucks_by_seed": {
            str(seed): index % args.trucks for index, seed in enumerate(args.seeds)
        },
        "script_sha256": script_sha,
        "maps": [result["map"] for result in results],
        "route_note": "Plotted paths are certified generator witnesses through the map's minimum energy routes. They show one assigned truck route per seed; other trucks remain in the saved scenario JSON.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "README.md").write_text(
        "# California and Texas preassigned route plot\n\n"
        f"The figure overlays {len(args.seeds)} certified truck paths per map, selected from "
        f"{len(args.seeds)} seeded scenarios with {args.trucks} trucks and "
        f"{args.deliveries} deliveries per truck in `preassigned` mode. The displayed truck "
        "rotates across seeds. Every scenario assigns all its customers to witness trucks, "
        "and each reset is certified by the seeded event engine. Paths follow minimum energy "
        "routes in the road graph; they are feasible witness assignments, not executed policy traces.\n\n"
        "- `assigned_paths_T3D3.png` and `.svg`: joint California/Texas route figure.\n"
        "- `assigned_paths.json`: all generated scenarios and routes, including trucks not plotted.\n"
        "- `manifest.json`: settings and selected truck IDs.\n"
    )
    print(f"Saved {args.output / 'assigned_paths_T3D3.png'}", flush=True)


if __name__ == "__main__":
    main()
