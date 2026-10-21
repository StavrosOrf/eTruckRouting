"""Reset analysis for the California campaign case: instance statistics and plots.

Generates many certified resets with the exact campaign configuration
(config_joint.yaml, 10 trucks, 30 customers, payload unchanged) and records
what each instance looks like before any policy acts: depot location,
customer geography, depot distances, demand versus fleet payload, how much
charging the generator's witness needs, and generation cost.

    python scripts/analysis/analyze_california_resets.py \
        --resets 1000 --output results/california_reset_analysis

Extra variants overlay on every plot, e.g. the long-haul depot ring:

    --variant 'ring225={"environment": {"min_depot_energy_kwh": 225}}'

All costs are nominal map costs. The witness is the generator's hidden
feasibility certificate, not an optimized plan: its charging counts are an
upper reference for what a good plan needs, not a measurement of one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv  # noqa: E402
from EVRoutingEnv.utils.utils import load_config, resolve_network_data_dir  # noqa: E402
from scripts.analysis.analyze_scenario_resets import (  # noqa: E402
    LegLookup,
    distribution,
    extract_witness,
)

RESET_FIELDS = (
    "variant", "seed", "certified", "error", "reset_ms", "generation_attempts",
    "depot_node", "depot_original_id", "depot_lat", "depot_lon", "depot_zone",
    "customers", "total_demand", "fleet_payload", "demand_over_payload",
    "min_trucks_by_payload", "depot_energy_min", "depot_energy_median",
    "depot_energy_max", "depot_hours_median", "customers_beyond_planning_reach",
    "customers_forcing_charge", "customer_nn_energy_median",
    "customer_pair_energy_mean", "customer_span_km", "distinct_zones",
    "witness_trucks_used", "witness_trucks_charging", "witness_charges",
    "witness_travel_hours", "witness_max_truck_hours", "witness_energy_kwh",
)
CUSTOMER_FIELDS = (
    "variant", "seed", "node", "original_id", "lat", "lon", "zone", "demand",
    "depot_energy_out", "depot_energy_in", "depot_hours_out", "depot_hours_in",
)


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def load_geography(env):
    folder = Path(resolve_network_data_dir(env.config["network"]))
    rows = json.loads((folder / "vis_data" / "nodes.json").read_text())
    by_original = {int(r["node"]): r for r in rows}
    geo = {}
    for node in env.transport_graph.graph:
        original = int(env.transport_graph.graph.nodes[node].get("original_id", node))
        row = by_original.get(original)
        if row is not None:
            geo[int(node)] = (original, float(row["lat"]), float(row["lon"]), row.get("zone_id") or "")
    return geo


def build_env(base, trucks, stops, overrides):
    config = deepcopy(base)
    config["environment"].update(num_trucks=trucks, num_stops=stops, allow_variable_num_stops=False)
    config["environment"].pop("scenario_cache_dir", None)
    for section, values in overrides.items():
        config.setdefault(section, {}).update(values)
    return config, EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)


def analyze_variant(name, config, env, seeds, writers):
    graph = env.transport_graph
    lookup = LegLookup(env)
    geo = load_geography(env)
    battery = float(config["truck"]["battery_capacity"])
    traffic = config["traffic"]
    uncertain = traffic["enable_traffic"] and traffic["enable_energy_uncertainty"]
    worst = max(1.0, float(traffic["max_energy_multiplier"])) if uncertain else 1.0
    best = min(1.0, float(traffic["min_energy_multiplier"])) if uncertain else 1.0
    resets, customers, trucks_all = [], [], []
    for index, seed in enumerate(seeds):
        row = dict.fromkeys(RESET_FIELDS)
        row.update(variant=name, seed=seed, certified=False, error="")
        start = perf_counter()
        try:
            _, info = env.reset(seed=seed)
            row["reset_ms"] = (perf_counter() - start) * 1000
            if not env.feasibility_witness or not info["instance_feasibility"]["certified"]:
                raise ValueError("reset returned no certificate")
        except (ValueError, RuntimeError) as error:
            row["reset_ms"] = (perf_counter() - start) * 1000
            row["error"] = f"{type(error).__name__}: {error}"[:300]
            writers["resets"].writerow(row)
            resets.append(row)
            continue
        instance = env.joint_instance
        depot = int(instance.depot_node)
        nodes = [int(t.node_id) for t in instance.tasks]
        demand = sum(float(t.demand) for t in instance.tasks)
        payload = sum(float(t.payload_capacity) for t in env.trucks)
        out_e = np.array([graph.get_leg_costs(depot, n)[0] for n in nodes])
        in_e = np.array([graph.get_leg_costs(n, depot)[0] for n in nodes])
        out_h = np.array([graph.get_leg_costs(depot, n)[1] for n in nodes])
        in_h = np.array([graph.get_leg_costs(n, depot)[1] for n in nodes])
        pair = np.array([[graph.get_leg_costs(a, b)[0] if a != b else np.nan for b in nodes] for a in nodes])
        pair_sym = np.fmin(pair, pair.T)
        round_trip = out_e + in_e
        hops, trucks = extract_witness(env, seed, name, lookup)
        used = [t for t in trucks if t["deliveries"] > 0]
        coords = [geo[n][1:3] for n in nodes if n in geo]
        span = max((haversine_km(a, b) for i, a in enumerate(coords) for b in coords[i + 1:]), default=0.0)
        dgeo = geo.get(depot, (None, None, None, ""))
        row.update(
            certified=True,
            generation_attempts=instance.generation_attempts_used,
            depot_node=depot, depot_original_id=dgeo[0], depot_lat=dgeo[1], depot_lon=dgeo[2], depot_zone=dgeo[3],
            customers=len(nodes), total_demand=demand, fleet_payload=payload,
            demand_over_payload=demand / payload,
            min_trucks_by_payload=math.ceil(demand / float(env.trucks[0].payload_capacity) - 1e-9),
            depot_energy_min=float(np.fmin(out_e, in_e).min()),
            depot_energy_median=float(np.median(np.fmin(out_e, in_e))),
            depot_energy_max=float(np.fmax(out_e, in_e).max()),
            depot_hours_median=float(np.median(out_h)),
            # Planning with the conservative multiplier cannot do the round trip on one battery.
            customers_beyond_planning_reach=float(np.mean(round_trip * worst > battery)),
            # Even the lowest realized energy draw exceeds one battery: charging is unavoidable.
            customers_forcing_charge=float(np.mean(round_trip * best > battery)),
            customer_nn_energy_median=float(np.median(np.nanmin(pair_sym, axis=1))),
            customer_pair_energy_mean=float(np.nanmean(pair)),
            customer_span_km=span,
            distinct_zones=len({geo[n][3] for n in nodes if n in geo}),
            witness_trucks_used=len(used),
            witness_trucks_charging=sum(t["certified_charges"] > 0 for t in used),
            witness_charges=sum(t["certified_charges"] for t in trucks),
            witness_travel_hours=sum(t["travel_hours"] for t in trucks),
            witness_max_truck_hours=max((t["travel_hours"] for t in trucks), default=0.0),
            witness_energy_kwh=sum(t["energy_kwh"] for t in trucks),
        )
        writers["resets"].writerow(row)
        resets.append(row)
        for task, eo, ei, ho, hi in zip(instance.tasks, out_e, in_e, out_h, in_h, strict=True):
            g = geo.get(int(task.node_id), (None, None, None, ""))
            record = dict(variant=name, seed=seed, node=int(task.node_id), original_id=g[0], lat=g[1], lon=g[2],
                          zone=g[3], demand=float(task.demand), depot_energy_out=float(eo), depot_energy_in=float(ei),
                          depot_hours_out=float(ho), depot_hours_in=float(hi))
            writers["customers"].writerow(record)
            customers.append(record)
        for t in trucks:
            t = {**t, "variant": name}
            writers["trucks"].writerow({k: t.get(k) for k in writers["trucks"].fieldnames})
            trucks_all.append(t)
        if (index + 1) % 100 == 0 or index + 1 == len(seeds):
            print(f"  {name}: {index + 1}/{len(seeds)} resets, "
                  f"{sum(r['certified'] for r in resets)} certified", flush=True)
    return dict(resets=resets, customers=customers, trucks=trucks_all, geo=geo,
                chargers=[int(c) for c in env.charging_nodes], battery=battery, worst=worst, best=best)


def summarize(name, data):
    good = [r for r in data["resets"] if r["certified"]]
    fields = [f for f in RESET_FIELDS if f not in (
        "variant", "seed", "certified", "error", "depot_node", "depot_original_id",
        "depot_lat", "depot_lon", "depot_zone")]
    return {
        "variant": name,
        "requested": len(data["resets"]),
        "certified": len(good),
        "errors": sorted({r["error"] for r in data["resets"] if r["error"]})[:10],
        "distinct_depots": len({r["depot_node"] for r in good}),
        "distinct_customers": len({c["node"] for c in data["customers"]}),
        "reset": {f: distribution(r[f] for r in data["resets"] if r[f] is not None) if f == "reset_ms"
                  else distribution(r[f] for r in good) for f in fields},
        "truck": {f: distribution(t[f] for t in data["trucks"] if t["deliveries"] > 0)
                  for f in ("deliveries", "demand", "payload_utilization", "energy_kwh",
                            "travel_hours", "certified_charges")},
    }


def plot(output, datasets, primary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1"]
    names = list(datasets)
    base = datasets[primary]
    battery, worst, best = base["battery"], base["worst"], base["best"]

    def good(name):
        return [r for r in datasets[name]["resets"] if r["certified"]]

    def hist(ax, values_by_name, title, xlabel, bins=40, discrete=False, vlines=()):
        allv = np.concatenate([np.asarray(v, float) for v in values_by_name.values() if len(v)] or [np.zeros(1)])
        if discrete:
            lo, hi = math.floor(allv.min()), math.ceil(allv.max())
            edges = np.arange(lo - 0.5, hi + 1.5, 1.0)
        else:
            edges = np.histogram_bin_edges(allv, bins=bins)
        for (name, values), color in zip(values_by_name.items(), colors, strict=False):
            if len(values):
                ax.hist(values, bins=edges, density=True, histtype="stepfilled", alpha=0.35, color=color,
                        label=f"{name} (n={len(values)})")
                ax.hist(values, bins=edges, density=True, histtype="step", color=color, linewidth=1.2)
        for k, (x, label) in enumerate(vlines):
            ax.axvline(x, color="#444", linestyle="--", linewidth=1)
            ax.text(x, ax.get_ylim()[1] * (0.95 - 0.08 * k), f" {label}", fontsize=7, va="top", color="#444")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)

    # Figure 1: geography of the primary variant.
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    geo, chargers = base["geo"], set(base["chargers"])
    all_lat = [v[1] for v in geo.values()]
    all_lon = [v[2] for v in geo.values()]
    depot_count, customer_count = {}, {}
    for r in good(primary):
        depot_count[r["depot_node"]] = depot_count.get(r["depot_node"], 0) + 1
    for c in base["customers"]:
        customer_count[c["node"]] = customer_count.get(c["node"], 0) + 1
    for ax, counts, title in ((axes[0], depot_count, "Depot frequency"),
                              (axes[1], customer_count, "Customer frequency")):
        ax.scatter(all_lon, all_lat, s=6, color="#bbb", label="map nodes")
        cx = [geo[n][2] for n in chargers if n in geo]
        cy = [geo[n][1] for n in chargers if n in geo]
        ax.scatter(cx, cy, marker="^", s=40, color="#2f855a", label="charging stations", zorder=3)
        nodes = [n for n in counts if n in geo]
        sc = ax.scatter([geo[n][2] for n in nodes], [geo[n][1] for n in nodes], c=[counts[n] for n in nodes],
                        s=[12 + 120 * counts[n] / max(counts.values()) for n in nodes], cmap="viridis",
                        edgecolor="k", linewidth=0.3, zorder=4)
        fig.colorbar(sc, ax=ax, shrink=0.7, label="resets in which the node appears")
        ax.set_title(f"{title} ({primary}, {len(good(primary))} resets)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect(1 / math.cos(math.radians(np.mean(all_lat))))
        ax.legend(fontsize=8, loc="lower left")
    fig.savefig(output / "geography.png", dpi=170)
    plt.close(fig)

    # Figure 2: instance-level distributions.
    panels = [
        ("depot_energy_median", "Median depot–customer energy per instance", "kWh (smaller direction)", False, ()),
        ("customer_nn_energy_median", "Median nearest-customer energy per instance", "kWh", False, ()),
        ("customer_span_km", "Customer geographic span", "km (great circle)", False, ()),
        ("demand_over_payload", "Total demand / fleet payload", "fraction", False, ()),
        ("min_trucks_by_payload", "Trucks needed by payload alone", "trucks", True, ()),
        ("customers_beyond_planning_reach", "Customers whose round trip exceeds one battery\nat the planning multiplier",
         "fraction of customers", False, ()),
        ("customers_forcing_charge", "Customers whose round trip forces a charge\neven at the lowest energy draw",
         "fraction of customers", False, ()),
        ("witness_trucks_charging", "Witness trucks that charge", "trucks", True, ()),
        ("witness_charges", "Witness charging sessions per instance", "sessions", True, ()),
        ("witness_travel_hours", "Witness total travel per instance", "hours", False, ()),
        ("witness_max_truck_hours", "Longest witness truck travel", "hours", False, ()),
        ("distinct_zones", "Distinct cities among customers", "zones", True, ()),
        ("generation_attempts", "Generation attempts per certified reset", "attempts", True, ()),
        ("reset_ms", "Reset wall time", "ms", False, ()),
    ]
    cols = 4
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), layout="constrained")
    for ax, (field, title, xlabel, discrete, vlines) in zip(axes.flat, panels, strict=False):
        hist(ax, {n: [r[field] for r in good(n) if r[field] is not None] for n in names}, title, xlabel,
             discrete=discrete, vlines=vlines)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("California reset analysis: instance statistics (nominal map costs; witness = generator certificate)")
    fig.savefig(output / "instance_distributions.png", dpi=150)
    plt.close(fig)

    # Figure 3: per-customer and per-truck views.
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    usable = battery / worst
    hist(axes[0, 0], {n: [min(c["depot_energy_out"], c["depot_energy_in"]) for c in datasets[n]["customers"]]
                      for n in names}, "Depot–customer energy (all customers)", "kWh (smaller direction)",
         vlines=((usable / 2, "half usable battery"), (battery / 2, "half battery")))
    hist(axes[0, 1], {n: [c["depot_hours_out"] for c in datasets[n]["customers"]] for n in names},
         "Depot → customer travel time", "hours")
    ax = axes[0, 2]
    for n, color in zip(names, colors, strict=False):
        cs = datasets[n]["customers"]
        ax.scatter([c["depot_energy_out"] for c in cs], [c["depot_energy_in"] for c in cs], s=3, alpha=0.15,
                   color=color, label=n)
    lim = max(max(c["depot_energy_out"], c["depot_energy_in"]) for n in names for c in datasets[n]["customers"])
    ax.plot([0, lim], [0, lim], color="#444", linewidth=0.8)
    ax.set(title="Depot leg asymmetry", xlabel="depot → customer kWh", ylabel="customer → depot kWh")
    ax.legend(fontsize=8, markerscale=4)
    ax.grid(alpha=0.2)
    trucks = {n: [t for t in datasets[n]["trucks"] if t["deliveries"] > 0] for n in names}
    hist(axes[1, 0], {n: [t["energy_kwh"] for t in v] for n, v in trucks.items()},
         "Witness route energy per active truck", "kWh", vlines=((battery, "one battery"),))
    hist(axes[1, 1], {n: [t["certified_charges"] for t in v] for n, v in trucks.items()},
         "Witness charging sessions per active truck", "sessions", discrete=True)
    hist(axes[1, 2], {n: [t["payload_utilization"] for t in v] for n, v in trucks.items()},
         "Witness payload utilization per active truck", "fraction of payload")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("California reset analysis: customers and witness trucks")
    fig.savefig(output / "customers_and_trucks.png", dpi=150)
    plt.close(fig)


def write_report(output, summaries, args, datasets, primary):
    def cell(summary, field, stat="median", section="reset", fmt="{:.2f}"):
        d = summary[section].get(field, {})
        return fmt.format(d[stat]) if d.get("count") else "—"

    lines = [
        "# California reset analysis",
        "",
        f"Config `{args.config}`, {args.trucks} trucks, {args.stops} customers, seeds "
        f"{args.seed_start}–{args.seed_start + args.resets - 1}. Costs are nominal map costs. "
        "The witness is the generator's certificate route, not an optimized plan.",
        "",
        "| Statistic (median unless noted) | " + " | ".join(s["variant"] for s in summaries) + " |",
        "| --- | " + " | ".join("---:" for _ in summaries) + " |",
        "| Certified / requested | " + " | ".join(f"{s['certified']} / {s['requested']}" for s in summaries) + " |",
        "| Distinct depots / customers used | " + " | ".join(
            f"{s['distinct_depots']} / {s['distinct_customers']}" for s in summaries) + " |",
    ]
    rows = [
        ("Depot–customer energy, kWh", "depot_energy_median", "median", "{:.0f}"),
        ("Depot → customer travel, h", "depot_hours_median", "median", "{:.1f}"),
        ("Nearest-customer energy, kWh", "customer_nn_energy_median", "median", "{:.0f}"),
        ("Customer span, km", "customer_span_km", "median", "{:.0f}"),
        ("Demand / fleet payload", "demand_over_payload", "median", "{:.2f}"),
        ("Trucks needed by payload (mean)", "min_trucks_by_payload", "mean", "{:.2f}"),
        ("Customers beyond one-battery round trip at planning multiplier (mean)",
         "customers_beyond_planning_reach", "mean", "{:.1%}"),
        ("Customers forcing a charge at lowest draw (mean)", "customers_forcing_charge", "mean", "{:.1%}"),
        ("Witness trucks charging (mean)", "witness_trucks_charging", "mean", "{:.2f}"),
        ("Witness charging sessions (mean)", "witness_charges", "mean", "{:.2f}"),
        ("Witness total travel, h", "witness_travel_hours", "median", "{:.0f}"),
        ("Longest witness truck travel, h", "witness_max_truck_hours", "median", "{:.1f}"),
        ("Generation attempts (mean)", "generation_attempts", "mean", "{:.2f}"),
        ("Reset time p95, ms", "reset_ms", "p95", "{:.0f}"),
    ]
    for label, field, stat, fmt in rows:
        lines.append(f"| {label} | " + " | ".join(cell(s, field, stat, fmt=fmt) for s in summaries) + " |")
    for s in summaries:
        if s["errors"]:
            lines += ["", f"**{s['variant']} rejected resets:** " + "; ".join(s["errors"])]
    lines += [
        "",
        "![Geography](geography.png)",
        "",
        "![Instance distributions](instance_distributions.png)",
        "",
        "![Customers and trucks](customers_and_trucks.png)",
        "",
        "Files: `resets.csv` (one row per reset), `customers.csv` (one row per customer per reset), "
        "`trucks.csv` (witness trucks), `summary.json` (full percentiles).",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def parse_variant(value):
    name, _, payload = value.partition("=")
    if not name or not payload:
        raise argparse.ArgumentTypeError("use NAME=JSON_OVERRIDES")
    return name, json.loads(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="EVRoutingEnv/config_files/config_joint.yaml")
    parser.add_argument("--trucks", type=int, default=10)
    parser.add_argument("--stops", type=int, default=30)
    parser.add_argument("--resets", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--primary-name", default="campaign_10t30c")
    parser.add_argument("--variant", action="append", type=parse_variant, default=[],
                        help="NAME=JSON config-section overrides; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists; choose a new directory")
    args.output.mkdir(parents=True)
    os.environ.pop("EVRP_SCENARIO_CACHE", None)
    base = load_config(args.config)
    seeds = list(range(args.seed_start, args.seed_start + args.resets))
    variants = [(args.primary_name, {})] + args.variant
    datasets, summaries, configs = {}, [], {}
    with (args.output / "resets.csv").open("w", newline="") as r_file, \
            (args.output / "customers.csv").open("w", newline="") as c_file, \
            (args.output / "trucks.csv").open("w", newline="") as t_file:
        writers = {
            "resets": csv.DictWriter(r_file, fieldnames=RESET_FIELDS),
            "customers": csv.DictWriter(c_file, fieldnames=CUSTOMER_FIELDS),
            "trucks": csv.DictWriter(t_file, fieldnames=(
                "variant", "seed", "truck_id", "deliveries", "demand", "payload_utilization", "energy_kwh",
                "travel_hours", "charger_visits", "certified_charges")),
        }
        for writer in writers.values():
            writer.writeheader()
        for name, overrides in variants:
            config, env = build_env(base, args.trucks, args.stops, overrides)
            configs[name] = config
            try:
                print(f"{name}: {len(seeds)} resets", flush=True)
                datasets[name] = analyze_variant(name, config, env, seeds, writers)
            finally:
                env.close()
            summaries.append(summarize(name, datasets[name]))
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (args.output / "configs.json").write_text(json.dumps(configs, indent=2, default=str) + "\n")
    plot(args.output, datasets, args.primary_name)
    write_report(args.output, summaries, args, datasets, args.primary_name)
    print(f"Saved {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
