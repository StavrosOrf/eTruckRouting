"""Record policy episodes and render them as one interactive trace report.

Two stages, so a report can combine runs made at different times or places:

    # 1. Run policies on scenarios and seeds; write the traces.
    PYTHONPATH=. python -m scripts.analysis.trace_report run \\
        --scenario tx_longhaul=EVRoutingEnv/config_files/generalist_suite/tx_longhaul.yaml \\
        --seeds 0 1 2 3 4 --policy heuristic --policy alns_planned --out traces.json

    # 2. Render one or more trace files into one HTML page.
    PYTHONPATH=. python -m scripts.analysis.trace_report render traces.json more.json --html report.html

``--policy`` takes a registered name (``--list-policies``), ``NAME=checkpoint:PATH[:PREFIX]``
for a trained canonical policy, or ``NAME=module:attribute`` for any callable
``policy(env, observation, info) -> action`` (a class is instantiated without
arguments). A policy that does not apply to a scenario, such as a day-ahead
planner on revealed-over-time requests, is recorded as not applicable with its
reason. From Python: ``record_episode(env, policy, seed)`` returns one trace and
``render([dataset, ...], path)`` writes the page.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import io
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("trace_report_template.html")
FORMAT = "etruck-trace-report-v1"
EPS = 1e-9


# ---------------------------------------------------------------- policies


# Where a registered policy applies: "static" (customers known at reset),
# "requests" (requests revealed over time or depot manifests) or "public"
# (static, with public stations shared with outside traffic).
SCOPES = {
    "heuristic": "static", "mpc": "static", "alns_planned": "static", "cpsat_planned": "static",
    "kullman": "public", "greedy_dispatch": "requests", "wait_at_depot": "requests",
    "rolling_alns": "requests", "random": "any", "bnp": "static",
}
SCOPE_TEXT = {
    "static": "it plans over customers known at reset, and this scenario reveals requests over time",
    "requests": "it dispatches revealed requests, and this scenario has only static customers",
    "public": "it is the public-station baseline, and this scenario has no public stations",
}


def applicability(name: str, config: dict) -> str | None:
    """Why a registered policy does not apply to ``config``, or None if it does."""
    scope = SCOPES.get(name, "any")
    requests = (config.get("requests") or {}).get("mode", "static") != "static" \
        or (config.get("problem") or {}).get("loading") == "at_depot"
    public = bool((config.get("public_charging") or {}).get("enabled", False))
    if (scope in ("static", "public") and requests) or (scope == "requests" and not requests) \
            or (scope == "public" and not public):
        return f"Not run: {SCOPE_TEXT[scope]}."
    return None


def _registered():
    """Name -> (label, description, factory(seed))."""
    from EVRoutingEnv.baselines.heuristics.alns import ALNSParameters, ALNSPolicy
    from EVRoutingEnv.baselines.heuristics.canonical_baselines import (
        GreedyHeuristicPolicy,
        HeuristicParameters,
        MPCParameters,
        RandomFeasiblePolicy,
        RollingHorizonMPCPolicy,
    )
    from EVRoutingEnv.baselines.heuristics.component_baselines import (
        DispatchParameters,
        GreedyDispatchPolicy,
        KullmanMyopicPolicy,
        RollingHorizonALNSPolicy,
        WaitAtDepotPolicy,
    )
    from EVRoutingEnv.baselines.optimization.exact_optimization import (
        ExactPlannerParameters,
        MathematicalProgrammingPolicy,
    )

    # Campaign settings (scripts/training/collect_demonstrations.py and the
    # generalist plan): day-ahead plans are solved once at reset and executed
    # open loop through the energy-safe navigator.
    day_ahead = {"replan_on_completion": False, "respect_planned_routes": True,
                 "energy_safety_factor": 1.2, "objective": "total_time"}
    return {
        "heuristic": ("Greedy heuristic", "Nearest serviceable customer, charges when the next leg needs it.",
                      lambda seed: GreedyHeuristicPolicy(HeuristicParameters(
                          energy_safety_factor=1.15, target_soc=1.0, demand_weight=2.0))),
        "mpc": ("Rolling-horizon MPC", "Short look-ahead tree over the next decisions, re-solved at every decision.",
                lambda seed: RollingHorizonMPCPolicy(MPCParameters(
                    horizon=6, branching=2, energy_safety_factor=1.15, target_soc=0.8))),
        "alns_planned": ("Day-ahead ALNS", "One nominal ALNS plan at reset (20,000 iterations, 10 s), executed open loop.",
                         lambda seed: ALNSPolicy(ALNSParameters(
                             iterations=20000, time_limit_seconds=10, average_charging_power_kw=135, **day_ahead))),
        "cpsat_planned": ("Day-ahead CP-SAT", "One nominal CP-SAT plan at reset (60 s), executed open loop.",
                          lambda seed: MathematicalProgrammingPolicy(ExactPlannerParameters(
                              time_limit_seconds=60, workers=4, average_charging_power_kw=135, **day_ahead))),
        "kullman": ("Kullman myopic", "Kullman et al. (2021) sec. 5.2.3: queue only behind a shorter-than-expected queue.",
                    lambda seed: KullmanMyopicPolicy()),
        "greedy_dispatch": ("Greedy dispatch", "EURO Meets NeurIPS 2022 greedy: load released requests nearest first, wait if none.",
                            lambda seed: GreedyDispatchPolicy()),
        "wait_at_depot": ("Wait at depot", "Waits at the depot until two requests are loadable, if deadlines allow.",
                          lambda seed: WaitAtDepotPolicy(DispatchParameters(wait_threshold=2))),
        "rolling_alns": ("Rolling-horizon ALNS", "Re-plans the active truck's next trip at every decision (200 iterations).",
                         lambda seed: RollingHorizonALNSPolicy(DispatchParameters(seed=seed))),
        "random": ("Random feasible", "Uniform over the feasible actions; a floor, not a method.",
                   lambda seed: RandomFeasiblePolicy(seed)),
        "bnp": ("Branch-and-price plan",
                ("Proven-optimal nominal plan (branch-and-price with Gurobi, 10 min), executed open loop "
                 "with energy-safe fallbacks. Needs Julia and a Gurobi licence."),
                _branch_and_price),
    }


def _branch_and_price(seed):
    from EVRoutingEnv.baselines.optimization.branch_and_price.julia_bridge import (
        BPParameters,
    )
    from EVRoutingEnv.baselines.optimization.branch_and_price.policy import (
        BranchAndPricePolicy,
    )

    licence = ROOT / "gurobi.lic"
    if not os.environ.get("GRB_LICENSE_FILE") and licence.exists():
        os.environ["GRB_LICENSE_FILE"] = str(licence)
    return BranchAndPricePolicy(BPParameters(time_limit_seconds=600, threads=4))


def resolve_policy(spec: str):
    """``spec`` -> (name, label, description, factory(seed))."""
    registered = _registered()
    if spec in registered:
        label, description, factory = registered[spec]
        return spec, label, description, factory
    if "=" not in spec:
        raise ValueError(f"unknown policy {spec!r}; registered: {', '.join(registered)}")
    name, target = spec.split("=", 1)
    if target.startswith("checkpoint:"):
        path, _, prefix = target[len("checkpoint:"):].partition(":")
        from scripts.evaluation.run_canonical_campaign import CanonicalPolicyRunner

        return name, name, f"Trained canonical policy {path}", (
            lambda seed: CanonicalPolicyRunner(Path(path), prefix=prefix or "best"))
    module, _, attribute = target.partition(":")
    obj = getattr(importlib.import_module(module), attribute)
    return name, name, f"{module}:{attribute}", (lambda seed: obj() if isinstance(obj, type) else obj)


# ---------------------------------------------------------------- recording


def _r(x, digits=3):
    if x is None:
        return None
    x = float(x)
    return round(x, digits) if math.isfinite(x) else None


def _truck_trace(truck, end):
    """Activity segments, legs (for the map), marks and charge samples of one truck."""
    events = truck.event_history
    segs, legs, marks, soc, open_at = [], [], [], [], {}
    roadside = [e for e in events if e["event_type"] == "HOS_REST" and e["details"].get("roadside")]
    for e in events:
        kind, t, node, d = e["event_type"], float(e["timestamp"]), int(e["location"]), e["details"] or {}
        sample = [_r(t), _r(e["battery_soc"], 1)]
        if not soc or soc[-1] != sample:
            soc.append(sample)
        if kind in ("ROUTING_START", "CHARGING_START", "WAITING_START", "UNLOADING_START",
                    "LOADING_START", "IDLE_START"):
            open_at[kind[:-6]] = e
        elif kind == "ROUTING_END":
            start = open_at.pop("ROUTING")
            s0 = float(start["timestamp"])
            drives, cursor = [], s0
            for rest in sorted(roadside, key=lambda x: x["details"]["start"]):
                rs, re_ = float(rest["details"]["start"]), float(rest["details"]["end"])
                if rs >= s0 - EPS and re_ <= t + EPS:
                    if rs > cursor + EPS:
                        drives.append([_r(cursor), _r(rs)])
                    segs.append([_r(rs), _r(re_), "rest", node, node, None, None,
                                 {"rest": rest["details"]["kind"], "roadside": True}])
                    cursor = re_
            if t > cursor + EPS or not drives:
                drives.append([_r(cursor), _r(t)])
            origin = int(d.get("origin", start["location"]))
            info = {"km": _r(d.get("distance_km"), 1), "kwh": _r(d.get("energy_consumed_kwh"), 1)}
            segs.extend([a, b, "drive", origin, node, None, None, info] for a, b in drives)
            legs.append([_r(s0), _r(t), origin, node, drives])
        elif kind == "CHARGING_END":
            start = open_at.pop("CHARGING")
            segs.append([_r(start["timestamp"]), _r(t), "charge", node, node, _r(start["battery_soc"], 1),
                         _r(e["battery_soc"], 1), {"kwh": _r(d.get("charge_amount_kwh"), 1)}])
        elif kind in ("WAITING_END", "UNLOADING_END", "LOADING_END", "IDLE_END"):
            label = kind[:-4]
            start = open_at.pop(label)
            name = {"WAITING": "queue", "UNLOADING": "service", "LOADING": "load", "IDLE": "wait"}[label]
            extra = {"request": d.get("request_id")} if label == "LOADING" else {}
            segs.append([_r(start["timestamp"]), _r(t), name, node, node, None, None, extra])
        elif kind == "TIME_WINDOW_WAIT":
            segs.append([_r(t - float(d.get("wait_duration_hours", 0.0))), _r(t), "window", node, node, None, None, {}])
        elif kind == "HOS_REST" and not d.get("roadside"):
            segs.append([_r(d["start"]), _r(d["end"]), "rest", node, node, None, None,
                         {"rest": d["kind"], "roadside": False}])
        elif kind == "CUSTOMER_SERVICE_COMPLETE":
            marks.append([_r(t), "served", node, d.get("task_id")])
        elif kind == "FAILED":
            marks.append([_r(t), "failed", node, d.get("reason")])
        elif kind == "COMPLETE":
            marks.append([_r(t), "retired", node, None])
        elif kind == "PORT_RELEASED_UNUSED":
            marks.append([_r(t), "released", node, None])
        elif kind == "REQUEST_LOST_ON_ARRIVAL":
            marks.append([_r(t), "lost", node, d.get("request_id")])
    rest_start = getattr(truck, "rest_start_time", None)
    if rest_start is not None and end > rest_start + EPS:  # an explicit rest cut by the horizon
        segs.append([_r(rest_start), _r(end), "rest", int(truck.current_node), int(truck.current_node),
                     None, None, {"rest": "?", "roadside": False, "cut": True}])
    names = {"ROUTING": "drive", "CHARGING": "charge", "WAITING": "queue", "UNLOADING": "service",
             "LOADING": "load", "IDLE": "wait"}
    for label, start in open_at.items():  # activities still running at the horizon
        s0 = float(start["timestamp"])
        if end > s0 + EPS:
            d = start["details"] or {}
            origin = int(start["location"])
            target = int(d.get("destination", origin)) if label == "ROUTING" else origin
            segs.append([_r(s0), _r(end), names[label], origin, target, None, None, {"cut": True}])
            if label == "ROUTING":
                legs.append([_r(s0), _r(end), origin, target, [[_r(s0), _r(end)]], True])
            marks.append([_r(end), "cut", origin, names[label]])
    segs.sort(key=lambda s: (s[0], s[1]))
    return segs, legs, marks, soc


MAX_STRIPS = 8  # stations shown per run, by fleet time spent there


def _pack_steps(steps) -> dict:
    """Occupancy steps [t, outside busy, fleet busy, queued, ports up] as integer
    centi-hour time deltas and a flat count list (about a third of the size)."""
    deltas, values, last = [], [], 0
    for t, *counts in steps:
        tick = round(float(t) * 100)
        deltas.append(tick - last)
        values.extend(int(c) for c in counts)
        last = tick
    return {"dt": deltas, "v": values}


def _station_strips(env, trucks, end):
    stations = env.public_stations
    if stations is None or not hasattr(stations, "queues"):
        return []
    fleet_hours = {}
    for truck in trucks:
        for seg in truck["segs"]:
            if seg[2] in ("charge", "queue") and seg[3] in stations.queues:
                fleet_hours[seg[3]] = fleet_hours.get(seg[3], 0.0) + seg[1] - seg[0]
    visited = sorted(fleet_hours, key=lambda n: -fleet_hours[n])[:MAX_STRIPS]
    strips = []
    for node in visited:
        queue = stations.queues[node]
        queue.advance(end)  # read-only use after the episode: extends the logged realization
        changes, initial_down = [], {}
        for _, t, _ in queue.log.arrivals:
            changes.append((max(0.0, t), 0, 1, 0, 0, 0))
        for kind, _, t, _ in queue.log.starts:
            outside = kind == "outside"
            changes.append((t, 1, -1 if outside else 0, int(outside), int(not outside), 0))
        for kind, _, t, _ in queue.log.departures:
            outside = kind == "outside"
            changes.append((t, 2, 0, -int(outside), -int(not outside), 0))
        for port, t, up in queue.log.port_changes:
            initial_down.setdefault(port, up)  # the first change flips the initial state
            changes.append((t, 3, 0, 0, 0, 1 if up else -1))
        changes.sort(key=lambda c: (c[0], c[1]))
        waiting = busy_out = busy_fleet = 0
        ups = sum(1 for port in range(len(queue.ports)) if not initial_down.get(port, False))
        steps = [[0.0, 0, 0, 0, ups]]
        for t, _, dq, dbo, dbf, du in changes:
            if t > end + EPS:
                break
            waiting, busy_out, busy_fleet, ups = waiting + dq, busy_out + dbo, busy_fleet + dbf, ups + du
            row = [_r(t), busy_out, busy_fleet, waiting, ups]
            if steps[-1][0] == row[0]:
                steps[-1] = row
            else:
                steps.append(row)
        params = stations.station_parameters(node)
        strips.append({"node": int(node), "ports": int(params.ports), "u": params.utilization,
                       "outages": bool(params.outages), **_pack_steps(steps)})
    return strips


def _requests(env, end):
    engine = env.request_engine
    if engine is None:
        return [], []
    rows = []
    for req in engine.registry.requests():
        if req.reveal_time > end + EPS:
            continue
        pickup = req.pickup
        rows.append({
            "id": req.request_id, "kind": req.kind.value,
            "pickup": None if pickup is None else [int(pickup.node), _r(pickup.earliest), _r(pickup.latest)],
            "delivery": [int(req.delivery.node), _r(req.delivery.earliest), _r(req.delivery.latest)],
            "reveal": _r(req.reveal_time), "release": _r(req.release_time), "status": req.status.value,
            "owner": req.owner, "loaded": _r(req.loaded_at), "served": _r(req.served_at),
            "lost": _r(req.lost_at), "delay": _r(req.delay_hours, 2),
        })
    return rows, [int(t) for t in getattr(engine, "terminals", ()) or ()]


def record_episode(env, policy, seed: int, *, max_steps: int = 50_000) -> dict:
    """Run one episode and return its trace (the report's run format)."""
    from EVRoutingEnv.baselines.heuristics.canonical_baselines import (
        decode_feasible_actions,
    )

    started = time.time()
    observation, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    decisions, done, steps = [], False, 0
    while not done and steps < max_steps:
        clock, truck_id = env.global_clock, env.active_truck_id
        candidates = decode_feasible_actions(env)
        action = int(policy(env, observation, info))
        chosen = next((c for c in candidates if c.index == action), None)
        kinds = {c.kind.value for c in candidates}
        decisions.append([
            _r(clock), int(truck_id), None if chosen is None else chosen.kind.value,
            None if chosen is None or chosen.target_node < 0 else int(chosen.target_node),
            len(candidates), "queue" in kinds,
            _r(chosen.charge_value, 2) if chosen is not None and chosen.kind.value == "charge" else None,
        ])
        observation, _, terminated, truncated, info = env.step(action)
        done, steps = terminated or truncated, steps + 1
    end = float(env.global_clock)
    trucks = []
    for truck in env.trucks:
        segs, legs, marks, soc = _truck_trace(truck, end)
        status = "failed" if truck.failed else ("retired" if truck.is_complete else "active")
        trucks.append({"id": truck.truck_id, "status": status, "failure": truck.failure_reason,
                       "start": int(truck.event_history[0]["location"]),
                       "segs": segs, "legs": legs, "marks": marks, "soc": soc})
    requests, terminals = _requests(env, end)
    customers = [] if env.request_engine is not None else [
        [int(task.node_id), _r(task.demand, 2)] for task in env.task_registry.tasks()]
    metrics = {k: (_r(v) if isinstance(v, float) else v) for k, v in info["operational_metrics"].items()}
    hos = None if env.hos is None else len(_hos_violations(env))
    return {
        "status": "ok", "termination": info["termination_reason"], "end": _r(end),
        "horizon": float(env.max_time), "steps": steps, "seconds": round(time.time() - started, 2),
        "metrics": metrics, "request_metrics": info.get("request_metrics"), "hos_violations": hos,
        "depot": int(env.joint_instance.depot_node), "terminals": terminals, "customers": customers,
        "requests": requests, "decisions": decisions, "trucks": trucks,
        "stations": _station_strips(env, trucks, end),
    }


def _hos_violations(env) -> list:
    """Independent audit of every driver's duty timeline (beyond counted roadside breaks)."""
    from EVRoutingEnv.models.core.hos_audit import (
        AuditSettings,
        hos_audit,
        intervals_from_event_log,
    )

    config = env.hos.config
    settings = AuditSettings(
        env.hos.regime.name,
        charging_counts_as_break=config.charging_counts_as_break,
        queue_wait_counts_as_break=config.queue_wait_counts_as_break,
        eu_split_break=config.eu_split_break,
        eu_extended_daily_driving=config.eu_extended_daily_driving,
        eu_reduced_daily_rest=config.eu_reduced_daily_rest,
    )
    violations = []
    for truck in env.trucks:
        intervals = intervals_from_event_log(truck.export_event_log(),
                                             handling_overhead_hours=config.handling_overhead_hours)
        violations.extend(hos_audit(intervals, settings))
    return violations if len(violations) > env.hos.unplanned_violations else []


def _network_table(env) -> dict:
    """Node index -> [lat, lon, role, label, station type, ports] for the map."""
    data = Path(env.config["network"]["data_path"]) / "vis_data"
    if not data.is_absolute():
        data = ROOT / data
    rows = {int(r["node"]): r for r in json.loads((data / "nodes.json").read_text())}
    stations = {int(s["node"]): s for s in json.loads((data / "stations.json").read_text())}
    chargers = set(map(int, env.charging_nodes))
    table = {}
    for index, attributes in env.transport_graph.graph.nodes(data=True):
        original = int(attributes["original_id"])
        row = rows.get(original)
        if row is None:
            continue
        if index in chargers:
            station = stations.get(original, {})
            names = station.get("names") or []
            table[int(index)] = [round(row["lat"], 5), round(row["lon"], 5), "charger",
                                 names[0] if names else "Charging station",
                                 station.get("station_type", ""), int(station.get("ports", 0) or 0)]
        else:
            table[int(index)] = [round(row["lat"], 5), round(row["lon"], 5), "customer",
                                 str(row.get("zone_id") or row.get("name") or "").title(), "", 0]
    return table


def _scenario_summary(config: dict) -> dict:
    keep = ("public_charging", "hours_of_service", "requests", "drayage", "depot_charging", "observation")
    summary = {k: config[k] for k in keep if k in config}
    summary["environment"] = {k: config["environment"].get(k) for k in ("num_trucks", "num_stops", "max_time")}
    if "loading" in config.get("problem", {}):
        summary["problem"] = {"loading": config["problem"]["loading"]}
    return summary


def _job(job):
    scenario, config_path, seed, spec = job
    from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
    from EVRoutingEnv.utils.utils import load_config

    name, _, _, factory = resolve_policy(spec)
    record = {"scenario": scenario, "policy": name, "seed": seed}
    config = deepcopy(load_config(config_path))
    reason = applicability(name, config)
    if reason is not None:
        record.update(status="not_applicable", reason=reason)
        return record
    try:
        env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
        record.update(record_episode(env, factory(seed), seed))
    except NotImplementedError as exc:
        record.update(status="not_applicable", reason=str(exc))
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden: the page shows the error
        record.update(status="error", reason=f"{type(exc).__name__}: {exc}",
                      trace=traceback.format_exc(limit=6))
    return record


def run(scenarios: dict[str, str], seeds: list[int], policies: list[str], *, workers: int = 1) -> dict:
    """Run every (scenario, seed, policy) and return a dataset."""
    from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
    from EVRoutingEnv.utils.utils import load_config

    dataset = {"format": FORMAT, "scenarios": {}, "networks": {}, "policies": {}, "runs": []}
    for scenario, path in scenarios.items():
        config = load_config(path)
        network = Path(config["network"]["data_path"].rstrip("/")).name
        dataset["scenarios"][scenario] = {"config": str(path), "network": network, "settings": _scenario_summary(config)}
        if network not in dataset["networks"]:
            env = EventDrivenTruckEnv(deepcopy(config), verbose=False, enable_plotting=False)
            dataset["networks"][network] = {"nodes": _network_table(env), "data_path": config["network"]["data_path"]}
    for spec in policies:
        name, label, description, _ = resolve_policy(spec)
        dataset["policies"][name] = {"label": label, "description": description, "spec": spec}
    jobs = [(s, str(p), seed, spec) for s, p in scenarios.items() for spec in policies for seed in seeds]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            for record in pool.map(_job, jobs):
                dataset["runs"].append(record)
                print(_line(record), flush=True)
    else:
        for job in jobs:
            record = _job(job)
            dataset["runs"].append(record)
            print(_line(record), flush=True)
    return dataset


def _line(record) -> str:
    head = f"{record['scenario']:24s} {record['policy']:16s} seed {record['seed']}: "
    if record["status"] != "ok":
        return head + f"{record['status']} ({record.get('reason', '')[:90]})"
    return head + f"{record['termination']} at {record['end']} h, {record['steps']} steps, {record['seconds']} s"


# ---------------------------------------------------------------- rendering


def _compact_strips(run: dict) -> None:
    """Bring strips of datasets written before packing to the current form."""
    strips = run.get("stations")
    if not strips:
        return
    hours = {}
    for truck in run["trucks"]:
        for seg in truck["segs"]:
            if seg[2] in ("charge", "queue"):
                hours[seg[3]] = hours.get(seg[3], 0.0) + seg[1] - seg[0]
    strips.sort(key=lambda strip: -hours.get(strip["node"], 0.0))
    del strips[MAX_STRIPS:]
    for strip in strips:
        if "steps" in strip:
            strip.update(_pack_steps(strip.pop("steps")))


def merge(datasets: list[dict]) -> dict:
    """Union of several datasets; a later run of the same (scenario, policy, seed) wins."""
    merged = {"format": FORMAT, "scenarios": {}, "networks": {}, "policies": {}, "runs": []}
    index = {}
    for dataset in datasets:
        if dataset.get("format") != FORMAT:
            raise ValueError(f"not a trace dataset ({dataset.get('format')!r})")
        for key in ("scenarios", "networks", "policies"):
            merged[key].update(dataset[key])
        for run in dataset["runs"]:
            _compact_strips(run)
            index[(run["scenario"], run["policy"], run["seed"])] = run
    merged["runs"] = list(index.values())
    return merged


def _view_box(dataset, scenario, k):
    nodes = dataset["networks"][dataset["scenarios"][scenario]["network"]]["nodes"]
    used = set()
    for run in dataset["runs"]:
        if run["scenario"] != scenario or run["status"] != "ok":
            continue
        used.update([run["depot"], *run["terminals"], *(c[0] for c in run["customers"])])
        for req in run["requests"]:
            used.add(req["delivery"][0])
            if req["pickup"]:
                used.add(req["pickup"][0])
        for truck in run["trucks"]:
            for seg in truck["segs"]:
                used.update((seg[3], seg[4]))
    points = [nodes[str(n)] if str(n) in nodes else nodes.get(n) for n in used]
    points = [p for p in points if p]
    if not points:
        return None
    lat = [p[0] for p in points]
    lon = [p[1] for p in points]
    lat0, lat1, lon0, lon1 = min(lat), max(lat), min(lon), max(lon)
    pad = 0.12 * max(lat1 - lat0, (lon1 - lon0) * k, 0.35)
    lat0, lat1, lon0, lon1 = lat0 - pad, lat1 + pad, lon0 - pad / k, lon1 + pad / k
    width, height = (lon1 - lon0) * k, lat1 - lat0
    if width / height < 1 / 1.1:
        grow = (height / 1.1 - width) / k / 2
        lon0, lon1 = lon0 - grow, lon1 + grow
    else:
        grow = (width * 1.1 - height) / 2
        lat0, lat1 = lat0 - grow, lat1 + grow
    return [lat0, lat1, lon0, lon1]


def _roads_png(data_path: str, box, k) -> str | None:
    """Road network of ``box`` as a transparent PNG data URI (OSM motorway..primary)."""
    roads_file = (ROOT / data_path / "vis_data" / "roads.npz") if not Path(data_path).is_absolute() \
        else Path(data_path) / "vis_data" / "roads.npz"
    if not roads_file.exists():
        return None
    import numpy as np

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    roads = np.load(roads_file)
    la0, la1, lo0, lo1 = box
    a_lat, a_lon = roads["lat0"] / 1e5, roads["lon0"] / 1e5
    b_lat, b_lon = roads["lat1"] / 1e5, roads["lon1"] / 1e5
    inside = ((np.maximum(a_lat, b_lat) >= la0) & (np.minimum(a_lat, b_lat) <= la1)
              & (np.maximum(a_lon, b_lon) >= lo0) & (np.minimum(a_lon, b_lon) <= lo1))
    width_px = 900
    height_px = round(width_px * (la1 - la0) / ((lo1 - lo0) * k))
    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(lo0 * k, lo1 * k)
    ax.set_ylim(la0, la1)
    ax.axis("off")
    for classes, lw, alpha in (((2, 3, 4, 5), 0.35 if la1 - la0 < 3 else 0.2, 0.55), ((1,), 0.6, 0.75), ((0,), 0.9, 1.0)):
        sel = inside & np.isin(roads["cls"], classes)
        segs = np.stack([np.stack([a_lon[sel] * k, a_lat[sel]], 1), np.stack([b_lon[sel] * k, b_lat[sel]], 1)], 1)
        ax.add_collection(LineCollection(segs, colors=[(0.5, 0.5, 0.5, alpha)], linewidths=lw))
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True, dpi=100)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# (key, label, better) for the per-scenario summary; None means "no ranking".
SUMMARY_COLUMNS = (
    ("runs", "Seeds", None), ("finished", "Finished", "max"), ("served", "Served %", "max"),
    ("late", "Late", "min"), ("lost", "Lost", "min"), ("stranded", "Stranded runs", "min"),
    ("end", "Run end h", "min"), ("operating", "Fleet h", "min"), ("travel", "Driving h", "min"),
    ("charging", "Charging h", "min"), ("queue", "Queue h", "min"), ("rest", "Rest h", "min"),
    ("waiting", "Waiting h", "min"), ("distance", "km", "min"), ("energy", "kWh", "min"),
    ("sessions", "Charges", None), ("seconds", "Run time s", None),
)


def _run_metrics(run: dict) -> dict:
    m, rm = run["metrics"], run.get("request_metrics")
    served = (rm["requests_served_on_time"] / max(1, rm["requests_released"]) if rm
              else m["customers_served"] / max(1, m["customers_total"]))
    return {
        "finished": float(run["termination"] == "success"), "served": 100.0 * served,
        "late": float(rm["requests_served_late"]) if rm else 0.0, "lost": float(rm["requests_lost"]) if rm else 0.0,
        "stranded": float(any(t["status"] == "failed" for t in run["trucks"])), "end": run["end"],
        "operating": m["total_operating_time"], "travel": m["total_travel_time"],
        "charging": m["total_charging_time"], "queue": m["total_queue_time"], "rest": m.get("total_rest_time", 0.0),
        "waiting": m.get("total_request_wait_time", 0.0) + m.get("total_time_window_waiting", 0.0),
        "distance": m["total_distance"], "energy": m["total_energy_consumed"],
        "sessions": float(m.get("charging_sessions", 0)), "seconds": run["seconds"],
    }


def summarize(dataset: dict) -> list[dict]:
    """Mean (and standard deviation) over seeds per scenario and policy; counts are sums."""
    groups: dict[tuple, list[dict]] = {}
    for run in dataset["runs"]:
        if run["status"] == "ok":
            groups.setdefault((run["scenario"], run["policy"]), []).append(_run_metrics(run))
    rows = []
    for (scenario, policy), runs in groups.items():
        row = {"scenario": scenario, "policy": policy, "runs": len(runs)}
        for key in runs[0]:
            values = [r[key] for r in runs]
            if key in ("finished", "stranded"):
                row[key] = sum(values)
            else:
                mean = sum(values) / len(values)
                row[key] = mean
                row[f"{key}_sd"] = (sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)) ** 0.5
        rows.append(row)
    order = list(dataset["policies"])
    rows.sort(key=lambda r: (list(dataset["scenarios"]).index(r["scenario"]), order.index(r["policy"])))
    return rows


def summary_markdown(dataset: dict) -> str:
    """One table per scenario; columns that are zero for every policy are left out."""
    rows, lines = summarize(dataset), []
    for scenario in dataset["scenarios"]:
        group = [r for r in rows if r["scenario"] == scenario]
        if not group:
            continue
        columns = [c for c in SUMMARY_COLUMNS if c[0] in ("runs", "served", "end") or any(r.get(c[0]) for r in group)]
        lines += [f"### {scenario}", "", "| Algorithm | " + " | ".join(label for _, label, _ in columns) + " |",
                  "|---|" + "---:|" * len(columns)]
        for r in group:
            cells = []
            for key, _, _ in columns:
                value = r[key]
                if key in ("runs", "finished", "stranded"):
                    cells.append(f"{value:.0f}")
                elif f"{key}_sd" in r and key in ("served", "end", "operating"):
                    cells.append(f"{value:.1f} ± {r[key + '_sd']:.1f}")
                else:
                    cells.append(f"{value:.1f}" if abs(value) < 1000 else f"{value:,.0f}")
            label = dataset["policies"].get(r["policy"], {}).get("label", r["policy"])
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def render(datasets: list[dict], html_path: str | Path, *, title: str = "eTruck Trace Report") -> Path:
    """Write the interactive page for one or more datasets."""
    dataset = merge(datasets)
    lats = [row[0] for net in dataset["networks"].values() for row in net["nodes"].values()]
    k = math.cos(math.radians(sum(lats) / len(lats))) if lats else 1.0
    views = {}
    for scenario, meta in dataset["scenarios"].items():
        box = _view_box(dataset, scenario, k)
        if box is not None:
            views[scenario] = {"box": box, "png": _roads_png(dataset["networks"][meta["network"]]["data_path"], box, k)}
    payload = {**dataset, "k": k, "views": views, "title": title,
               "generated": time.strftime("%Y-%m-%d %H:%M")}
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__TITLE__*/", title).replace("/*__DATA__*/", data)
    html_path = Path(html_path)
    html_path.write_text(html)
    # The same per-scenario averages as the page, for papers and notes.
    rows = summarize(dataset)
    if rows:
        import csv

        with html_path.with_suffix(".summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        html_path.with_suffix(".summary.md").write_text(summary_markdown(dataset))
    return html_path


# ---------------------------------------------------------------- command line


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    runp = sub.add_parser("run", help="run policies and write a trace dataset")
    runp.add_argument("--scenario", action="append", required=True, metavar="NAME=CONFIG")
    runp.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    runp.add_argument("--policy", action="append", required=True, metavar="SPEC")
    runp.add_argument("--workers", type=int, default=1)
    runp.add_argument("--out", type=Path, required=True)
    runp.add_argument("--html", type=Path, help="also render the page")
    renderp = sub.add_parser("render", help="render one or more datasets into one page")
    renderp.add_argument("datasets", type=Path, nargs="+")
    renderp.add_argument("--html", type=Path, required=True)
    renderp.add_argument("--title", default="eTruck Trace Report")
    summaryp = sub.add_parser("summary", help="print per-scenario averages over seeds (Markdown)")
    summaryp.add_argument("datasets", type=Path, nargs="+")
    sub.add_parser("list-policies", help="print the registered policies")
    args = parser.parse_args(argv)
    if args.command == "list-policies":
        for name, (label, description, _) in _registered().items():
            print(f"{name:16s} [{SCOPES.get(name, 'any')}] {label}: {description}")
        return
    if args.command == "run":
        scenarios = dict(item.split("=", 1) for item in args.scenario)
        dataset = run(scenarios, args.seeds, args.policy, workers=args.workers)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dataset, separators=(",", ":")))
        print(f"wrote {args.out}")
        if args.html:
            print(f"wrote {render([dataset], args.html)}")
        return
    datasets = [json.loads(path.read_text()) for path in args.datasets]
    if args.command == "summary":
        print(summary_markdown(merge(datasets)))
        return
    print(f"wrote {render(datasets, args.html, title=args.title)}")


if __name__ == "__main__":
    main()
