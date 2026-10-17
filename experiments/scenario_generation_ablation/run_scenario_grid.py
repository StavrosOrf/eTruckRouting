"""Reproducible 3-mode, 100-seed scenario-generation comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from EVRoutingEnv.models.environment import event_driven_env as event_driven_module
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


STUDY = Path(__file__).resolve().parent
MODES = {
    "joint_fleet": ("joint_fleet", "fleet"),
    "preassigned": ("joint_fleet", "preassigned"),
    "sequential": ("sequential", "preassigned"),
}
TRUCK_COUNTS = (1, 10, 100)
DELIVERIES_PER_TRUCK = (3, 10, 20)
GRID = [
    (trucks, per_truck)
    for trucks in TRUCK_COUNTS
    for per_truck in DELIVERIES_PER_TRUCK
]
METRICS = (
    "graph_node_count",
    "synthetic_customer_nodes",
    "charger_count",
    "customer_count",
    "total_demand",
    "mean_demand",
    "max_demand",
    "payload_utilization",
    "initial_soc_mean",
    "generation_attempts",
    "witness_steps",
    "witness_trucks_used",
    "witness_empty_trucks",
    "witness_charger_visits",
    "witness_unique_chargers",
    "witness_charge_actions",
    "witness_navigation_hops",
    "witness_distance_proxy_km",
    "witness_travel_time_hours",
    "witness_energy_kwh",
    "delivery_hop_energy_mean_kwh",
    "delivery_hop_energy_max_kwh",
    "max_truck_payload_utilization",
)


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def expand_customer_nodes(
    base_graph: nx.DiGraph, num_customers: int
) -> tuple[nx.DiGraph, int, str]:
    """Use the bundled map when possible, otherwise make a scalable test graph."""
    eligible = [
        int(node)
        for node, data in base_graph.nodes(data=True)
        if not data.get("has_charger", False)
        and base_graph.in_degree(node) > 0
        and base_graph.out_degree(node) > 0
    ]
    if num_customers + 1 <= len(eligible):
        return base_graph.copy(), 0, "bundled_road_graph"

    # The bundled map has only 258 nodes and cannot provide 2,000 distinct
    # delivery tasks. This symmetric hub-and-spoke graph keeps tasks unique
    # and exercises charger routing at larger scale.
    graph = nx.DiGraph()
    graph.add_node(0, has_charger=False, charger_type={}, original_id=0)
    charger_count = 25
    leaves = []
    for node in range(1, charger_count + 1):
        graph.add_node(
            node,
            has_charger=True,
            charger_type={"DCFast": 1},
            original_id=node,
        )
        leaves.append(node)
    for node in range(charger_count + 1, charger_count + num_customers + 1):
        graph.add_node(
            node,
            has_charger=False,
            charger_type={},
            original_id=node,
            synthetic_customer=True,
        )
        leaves.append(node)
    for node in leaves:
        graph.add_edge(0, node, distance=40.0, time=0.25, terrain_factor=1.0)
        graph.add_edge(node, 0, distance=40.0, time=0.25, terrain_factor=1.0)
    return graph, num_customers, "synthetic_hub_spoke"


def scenario_row(
    env: EventDrivenTruckEnv,
    seed: int,
    mode: str,
    deliveries_per_truck: int,
    synthetic_nodes: int,
    network_variant: str,
) -> dict:
    instance = env.joint_instance
    graph = env.transport_graph
    tasks = instance.tasks
    owners = instance.witness_owners
    owner_nodes = {
        truck_id: [
            task.node_id
            for task, owner in zip(tasks, owners, strict=True)
            if owner == truck_id
        ]
        for truck_id in range(env.num_trucks)
    }
    owner_demands = {
        truck_id: sum(
            task.demand
            for task, owner in zip(tasks, owners, strict=True)
            if owner == truck_id
        )
        for truck_id in range(env.num_trucks)
    }

    distance_km = travel_hours = energy_kwh = 0.0
    charger_visits = charge_actions = navigation_hops = 0
    visited_chargers: set[int] = set()
    delivery_hop_energy = []
    for truck_id, route in enumerate(instance.witness_routes):
        node = instance.depot_node
        delivery_nodes = set(owner_nodes[truck_id])
        for target, _charge_value, charging in route:
            if charging:
                charge_actions += 1
                continue
            target = int(target)
            if target == node:
                continue
            edge_energy = float(graph.get_path_energy(node, target))
            edge_time = float(graph.get_time_distance(node, target))
            energy_kwh += edge_energy
            travel_hours += edge_time
            distance_km += edge_time * float(env.config["truck"]["base_speed"])
            navigation_hops += 1
            if target in env.charging_nodes:
                charger_visits += 1
                visited_chargers.add(target)
            if target in delivery_nodes:
                delivery_hop_energy.append(float(graph.get_path_energy(node, target)))
            node = target

    demands = [float(task.demand) for task in tasks]
    capacity = float(env.problem_config["payload_capacity"])
    initial_batteries = [
        float(value) for value in env.scenario_descriptor["initial_battery_kwh"]
    ]
    metric = {
        "graph_node_count": int(graph.num_nodes),
        "synthetic_customer_nodes": synthetic_nodes,
        "network_variant": network_variant,
        "charger_count": len(env.charging_nodes),
        "customer_count": len(tasks),
        "total_demand": sum(demands),
        "mean_demand": statistics.fmean(demands),
        "max_demand": max(demands),
        "payload_utilization": sum(demands) / (env.num_trucks * capacity),
        "initial_soc_mean": statistics.fmean(
            battery / float(env.config["truck"]["battery_capacity"])
            for battery in initial_batteries
        ),
        "generation_attempts": instance.generation_attempts_used,
        "witness_steps": len(env.feasibility_witness),
        "witness_trucks_used": sum(bool(nodes) for nodes in owner_nodes.values()),
        "witness_empty_trucks": sum(not nodes for nodes in owner_nodes.values()),
        "witness_charger_visits": charger_visits,
        "witness_unique_chargers": len(visited_chargers),
        "witness_charge_actions": charge_actions,
        "witness_navigation_hops": navigation_hops,
        "witness_distance_proxy_km": distance_km,
        "witness_travel_time_hours": travel_hours,
        "witness_energy_kwh": energy_kwh,
        "delivery_hop_energy_mean_kwh": (
            statistics.fmean(delivery_hop_energy) if delivery_hop_energy else 0.0
        ),
        "delivery_hop_energy_max_kwh": max(delivery_hop_energy, default=0.0),
        "max_truck_payload_utilization": max(owner_demands.values(), default=0.0)
        / capacity,
    }
    desc = env.scenario_descriptor
    descriptor_json = json.dumps(desc, sort_keys=True, separators=(",", ":"))
    return {
        "mode": mode,
        "seed": seed,
        "num_trucks": env.num_trucks,
        "deliveries_per_truck": deliveries_per_truck,
        "num_deliveries_total": env.num_stops,
        "certified": bool(env.feasibility_witness),
        "certificate_method": "seeded_event_replay",
        **metric,
        "depot_node": int(instance.depot_node),
        "customer_node_ids_json": json.dumps([int(task.node_id) for task in tasks]),
        "customer_demands_json": json.dumps([float(task.demand) for task in tasks]),
        "witness_owner_by_task_json": json.dumps(list(owners)),
        "witness_route_nodes_json": json.dumps(
            {
                str(tid): [int(action[0]) for action in route if not action[2]]
                for tid, route in enumerate(instance.witness_routes)
            },
            sort_keys=True,
        ),
        "scenario_sha256": hashlib.sha256(descriptor_json.encode()).hexdigest(),
        "error": "",
    }


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["mode"],
                int(row["num_trucks"]),
                int(row["deliveries_per_truck"]),
            )
        ].append(row)
    summaries = []
    for key in sorted(groups, key=lambda x: (x[1], x[2], tuple(MODES).index(x[0]))):
        mode, trucks, deliveries_per_truck = key
        group = groups[key]
        good = [row for row in group if row["certified"]]
        entry = {
            "mode": mode,
            "num_trucks": trucks,
            "deliveries_per_truck": deliveries_per_truck,
            "num_deliveries_total": trucks * deliveries_per_truck,
            "requested_scenarios": len(group),
            "certified_scenarios": len(good),
            "failed_generations": len(group) - len(good),
            "certificate_rate": len(good) / len(group),
        }
        for name in METRICS:
            values = [
                float(row[name]) for row in good if row.get(name) not in (None, "")
            ]
            entry[f"{name}_mean"] = statistics.fmean(values) if values else None
            entry[f"{name}_sd"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
            )
            entry[f"{name}_p90"] = (
                sorted(values)[math.ceil(0.9 * len(values)) - 1] if values else None
            )
        summaries.append(entry)
    return summaries


def _mean_label(item: dict, name: str) -> str:
    value = item[f"{name}_mean"]
    return "—" if value is None else f"{value:.2f}"


def make_plots(summaries: list[dict], output: Path) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir()
    colors = {"joint_fleet": "#3569a8", "preassigned": "#e28e2c", "sequential": "#3a9d5d"}
    font = "font-family='DejaVu Sans, sans-serif'"
    heatmap = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='1260' height='480' viewBox='0 0 1260 480'>",
        f"<rect width='1260' height='480' fill='white'/><text x='630' y='32' text-anchor='middle' font-size='22' {font}>Complete-witness certification rate</text>",
    ]
    for mode_index, mode in enumerate(MODES):
        left = 85 + mode_index * 400
        heatmap.append(f"<text x='{left + 150}' y='75' text-anchor='middle' font-size='17' {font}>{mode.replace('_', ' ')}</text>")
        for row, trucks in enumerate(TRUCK_COUNTS):
            heatmap.append(f"<text x='{left - 12}' y='{132 + row * 95}' text-anchor='end' font-size='13' {font}>{trucks}</text>")
            for column, per_truck in enumerate(DELIVERIES_PER_TRUCK):
                item = next(x for x in summaries if x["mode"] == mode and x["num_trucks"] == trucks and x["deliveries_per_truck"] == per_truck)
                x, y = left + column * 94, 100 + row * 95
                rate = item["certificate_rate"]
                red_blue = int(255 - 100 * rate)
                heatmap.append(f"<rect x='{x}' y='{y}' width='84' height='76' rx='4' fill='rgb({red_blue},255,{red_blue})' stroke='#b8c3cc'/>")
                heatmap.append(f"<text x='{x + 42}' y='{y + 43}' text-anchor='middle' font-size='17' {font}>{rate:.0%}</text>")
                if row == 0:
                    heatmap.append(f"<text x='{x + 42}' y='96' text-anchor='middle' font-size='12' {font}>{per_truck} del/truck</text>")
        heatmap.append(f"<text x='{left - 76}' y='240' transform='rotate(-90 {left - 76} 240)' text-anchor='middle' font-size='13' {font}>Fleet size (trucks)</text>")
    heatmap.append("</svg>")
    (plot_dir / "certificate_rate.svg").write_text("\n".join(heatmap), encoding="utf-8")

    offsets = {"joint_fleet": -0.18, "preassigned": 0.0, "sequential": 0.18}
    for filename, metric, ylabel, title in (
        ("witness_distance_proxy.svg", "witness_distance_proxy_km", "Mean distance proxy (km)", "Certified witness distance"),
        ("witness_energy.svg", "witness_energy_kwh", "Mean nominal energy (kWh)", "Certified witness energy"),
    ):
        width, height = 1260, 470
        svg = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>", f"<rect width='{width}' height='{height}' fill='white'/><text x='630' y='32' text-anchor='middle' font-size='22' {font}>{title} (mean ± 1 SD)</text>"]
        for panel, trucks in enumerate(TRUCK_COUNTS):
            left = 82 + panel * 405
            plot_top, plot_bottom = 95, 365
            panel_items = [item for item in summaries if item["num_trucks"] == trucks]
            maximums = [
                item[f"{metric}_mean"] + (item[f"{metric}_sd"] or 0.0)
                for item in panel_items
                if item[f"{metric}_mean"] is not None
            ]
            maximum = max(maximums, default=1.0)
            maximum = max(maximum * 1.12, 1.0)
            svg.append(f"<text x='{left + 160}' y='70' text-anchor='middle' font-size='17' {font}>{trucks} trucks</text>")
            if panel == 0:
                svg.append(f"<text x='20' y='235' transform='rotate(-90 20 235)' text-anchor='middle' font-size='12' {font}>{ylabel}</text>")
            svg.append(f"<line x1='{left}' y1='{plot_top}' x2='{left}' y2='{plot_bottom}' stroke='#68737d'/><line x1='{left}' y1='{plot_bottom}' x2='{left + 330}' y2='{plot_bottom}' stroke='#68737d'/>")
            for tick in range(5):
                value = maximum * tick / 4
                y = plot_bottom - (plot_bottom - plot_top) * tick / 4
                svg.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + 330}' y2='{y:.1f}' stroke='#e2e6ea'/><text x='{left - 8}' y='{y + 4:.1f}' text-anchor='end' font-size='10' {font}>{value:.0f}</text>")
            points_by_mode = {}
            for mode in MODES:
                points = []
                for index, per_truck in enumerate(DELIVERIES_PER_TRUCK):
                    item = next(x for x in summaries if x["mode"] == mode and x["num_trucks"] == trucks and x["deliveries_per_truck"] == per_truck)
                    value = item[f"{metric}_mean"]
                    if value is None:
                        continue
                    x = left + 45 + index * 120 + offsets[mode] * 70
                    sd = item[f"{metric}_sd"] or 0.0
                    y = plot_bottom - (plot_bottom - plot_top) * value / maximum
                    low = plot_bottom - (plot_bottom - plot_top) * max(0.0, value - sd) / maximum
                    high = plot_bottom - (plot_bottom - plot_top) * min(maximum, value + sd) / maximum
                    points.append(f"{x:.1f},{y:.1f}")
                    svg.append(f"<line x1='{x:.1f}' y1='{high:.1f}' x2='{x:.1f}' y2='{low:.1f}' stroke='{colors[mode]}' stroke-width='1.5'/><circle cx='{x:.1f}' cy='{y:.1f}' r='5' fill='{colors[mode]}'/>")
                points_by_mode[mode] = " ".join(points)
                if points_by_mode[mode]:
                    svg.append(f"<polyline points='{points_by_mode[mode]}' fill='none' stroke='{colors[mode]}' stroke-width='2.2'/>")
            for index, per_truck in enumerate(DELIVERIES_PER_TRUCK):
                x = left + 45 + index * 120
                svg.append(f"<text x='{x}' y='388' text-anchor='middle' font-size='12' {font}>{per_truck}</text>")
            svg.append(f"<text x='{left + 160}' y='414' text-anchor='middle' font-size='12' {font}>Deliveries per truck</text>")
        for index, mode in enumerate(MODES):
            x = 430 + index * 170
            svg.append(f"<line x1='{x}' y1='448' x2='{x + 24}' y2='448' stroke='{colors[mode]}' stroke-width='3'/><text x='{x + 30}' y='452' font-size='12' {font}>{mode.replace('_', ' ')}</text>")
        svg.append("</svg>")
        (plot_dir / filename).write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=STUDY / "results" / "seeds_1_100_per_truck",
        help="New output directory; existing directories are never overwritten.",
    )
    args = parser.parse_args()
    if args.seed_start < 0 or args.seed_count < 1:
        parser.error("seed-start must be non-negative and seed-count positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    rows: list[dict] = []
    resolved_settings = {}

    original_get_graph = event_driven_module.get_graph
    for trucks, deliveries_per_truck in GRID:
        total_deliveries = trucks * deliveries_per_truck
        template = load_config("EVRoutingEnv/config_files/config_joint.yaml")
        source_graph = original_get_graph(template)
        graph, synthetic_nodes, network_variant = expand_customer_nodes(
            source_graph, total_deliveries
        )
        event_driven_module.get_graph = lambda _config, graph=graph: graph.copy()
        try:
            for mode, (problem_mode, assignment) in MODES.items():
                config = load_config("EVRoutingEnv/config_files/config_joint.yaml")
                config["problem"]["mode"] = problem_mode
                config["problem"]["assignment"] = assignment
                config["problem"]["payload_capacity"] = float(4 * deliveries_per_truck)
                config["environment"].update(
                    num_trucks=trucks,
                    num_stops=total_deliveries,
                    allow_variable_num_stops=False,
                    max_time=5000.0,
                    verbose=False,
                )
                config.setdefault("environment", {}).pop("max_episode_steps", None)
                resolved_settings[
                    f"{mode}/{trucks}trucks/{deliveries_per_truck}deliveries_per_truck"
                ] = config
                env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
                try:
                    for seed in seeds:
                        try:
                            env.reset(seed=seed)
                            rows.append(
                                scenario_row(
                                    env,
                                    seed,
                                    mode,
                                    deliveries_per_truck,
                                    synthetic_nodes,
                                    network_variant,
                                )
                            )
                        except (ValueError, RuntimeError, KeyError, AttributeError) as error:
                            rows.append(
                                {
                                    "mode": mode,
                                    "seed": seed,
                                    "num_trucks": trucks,
                                    "deliveries_per_truck": deliveries_per_truck,
                                    "num_deliveries_total": total_deliveries,
                                    "certified": False,
                                    "certificate_method": "seeded_event_replay",
                                    "error": f"{type(error).__name__}: {error}",
                                }
                            )
                    print(
                        f"completed mode={mode}, trucks={trucks}, "
                        f"deliveries_per_truck={deliveries_per_truck}",
                        flush=True,
                    )
                finally:
                    env.close()
        finally:
            event_driven_module.get_graph = original_get_graph

    summaries = summarize(rows)
    raw_path = output / "scenario_records.csv"
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    make_plots(summaries, output)
    report = [
        "# Results: scenario-generation mode ablation",
        "",
        (
            f"Seeds: {seeds[0]}–{seeds[-1]} ({len(seeds)} per cell); records: {len(rows)}; "
            f"certified: {sum(bool(row.get('certified')) for row in rows)}; "
            f"failed: {sum(not bool(row.get('certified')) for row in rows)}."
        ),
        "",
        (
            "Values below are means over certified scenarios. Distance is the simulator's "
            "travel-time × speed proxy; full means, sample SDs, p90s, and all recorded "
            "fields are in `summary.csv` and `scenario_records.csv`."
        ),
        "",
        (
            "Cells larger than the bundled graph's unique-customer capacity use the "
            "synthetic hub-and-spoke graph; see the manifest and per-row `network_variant`. "
            "Its distance values are not geographic measurements."
        ),
        "",
    ]
    for trucks in TRUCK_COUNTS:
        report.extend(
            [
                f"## {trucks} trucks",
                "",
                (
                    "| Mode | Deliveries / truck | Total deliveries | Certificate rate | Nodes / chargers | Mean demand | "
                    "Witness trucks / empty | Distance proxy (km) | Travel (h) | Energy (kWh) | "
                    "Charger visits / charges | Attempts |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in summaries:
            if int(item["num_trucks"]) != trucks:
                continue
            mode = item["mode"]
            report.append(
                f"| {mode} | {item['deliveries_per_truck']} | {item['num_deliveries_total']} | "
                f"{item['certified_scenarios']}/{item['requested_scenarios']} "
                f"({100 * item['certificate_rate']:.1f}%) | "
                f"{_mean_label(item, 'graph_node_count')} / {_mean_label(item, 'charger_count')} | "
                f"{_mean_label(item, 'total_demand')} | {_mean_label(item, 'witness_trucks_used')} / "
                f"{_mean_label(item, 'witness_empty_trucks')} | {_mean_label(item, 'witness_distance_proxy_km')} | "
                f"{_mean_label(item, 'witness_travel_time_hours')} | {_mean_label(item, 'witness_energy_kwh')} | "
                f"{_mean_label(item, 'witness_charger_visits')} / {_mean_label(item, 'witness_charge_actions')} | "
                f"{_mean_label(item, 'generation_attempts')} |"
            )
        report.append("")
    report.extend(
        [
            "## Plots",
            "",
            "![Certification rate](plots/certificate_rate.svg)",
            "",
            "![Witness distance proxy](plots/witness_distance_proxy.svg)",
            "",
            "![Witness energy](plots/witness_energy.svg)",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    config_blob = json.dumps(resolved_settings, sort_keys=True, separators=(",", ":"))
    manifest = {
        "study": "scenario_generation_ablation",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "seeds": seeds,
        "grid": [
            {
                "num_trucks": t,
                "deliveries_per_truck": d,
                "total_deliveries": t * d,
            }
            for t, d in GRID
        ],
        "modes": {
            name: {"problem_mode": value[0], "assignment": value[1]}
            for name, value in MODES.items()
        },
        "scenario_count_requested": len(seeds) * len(GRID) * len(MODES),
        "scenario_count_recorded": len(rows),
        "config_sha256": hashlib.sha256(config_blob.encode()).hexdigest(),
        "resolved_configs": resolved_settings,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain"),
        "python": sys.version,
        "platform": platform.platform(),
        "distance_definition": "sum(shortest_path_travel_time_hours * truck.base_speed_km_per_hour) over the complete certified witness, including charger detours and final depot returns",
        "network_expansion": "Cells with more tasks than the bundled map supports use a synthetic bidirectional hub-and-spoke network with 25 chargers, one unique customer node per delivery, 40 kWh/0.25 h hub-to-leaf arcs, and zero measured-geography claims. Smaller cells use the bundled road graph.",
        "notes": [
            "Every row is an environment reset; success means its complete witness replay certified under this seed.",
            "Mode comparisons reuse the same integer seeds within each truck/delivery-size cell.",
            "Flexible fleet ownership has no prescribed owner/order; witness owner/routes are a hidden certificate only.",
            "Failed reset attempts remain in scenario_records.csv and are not dropped from certificate rates.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} records to {output}", flush=True)


if __name__ == "__main__":
    main()
