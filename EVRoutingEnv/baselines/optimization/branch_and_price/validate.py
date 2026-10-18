"""Check a branch-and-price plan against the environment itself.

``nominal_replay`` executes the plan on a private copy of the freshly reset
episode with every noise source switched off, through the public ``step``
and therefore through the same action mask, energy margin, payload, window,
horizon and FCFS charger-queue logic as any policy. Queues are the one effect
the solver relaxes, so the replay is also the port-capacity check: a plan
that waits at a busy station still succeeds as long as it keeps its windows
and the horizon, and its travel time is unchanged by waiting.
"""

from __future__ import annotations

import math

from EVRoutingEnv.baselines.optimization.perfect_information import _clone, _require_initial_state
from EVRoutingEnv.evaluation.metrics import extract_operational_metrics
from EVRoutingEnv.models.environment.route_certificate import NoFeasibleCertificate, replay_routes


def fleet_witness(env, routes) -> tuple:
    """Per-truck witness: planned routes first, idle trucks retire at the depot."""
    depot = int(env.joint_instance.depot_node)
    if len(routes) > env.num_trucks:
        raise ValueError(f"{len(routes)} routes for {env.num_trucks} trucks")
    idle = ((depot, 0.0, False),)
    return tuple(tuple(route) for route in routes) + (idle,) * (env.num_trucks - len(routes))


def nominal_shadow(env):
    """Private copy of the reset episode with deterministic legs, energy and service."""
    _require_initial_state(env)
    shadow = _clone(env)
    shadow.traffic_simulator.std_dev_factor = 0.0
    shadow.traffic_simulator.energy_uncertainty_factor = 0.0
    shadow.delivery_simulator.enable_stochastic_unloading = False
    return shadow


def nominal_replay(env, routes) -> dict:
    """Replay planned witness routes; returns feasibility and realized metrics."""
    shadow = nominal_shadow(env)
    witness = fleet_witness(env, routes)
    try:
        replay_routes(shadow, witness)
        feasible, reason = True, ""
    except NoFeasibleCertificate as error:
        feasible, reason = False, str(error)
    metrics = extract_operational_metrics(shadow)
    return {
        "feasible": feasible and metrics.success,
        "reason": reason,
        "total_travel_hours": metrics.total_travel_time,
        "total_queue_hours": metrics.total_queue_time,
        "makespan_hours": float(shadow.global_clock),
        "steps": int(shadow.episode_steps),
        "metrics": metrics.as_dict(),
    }


def port_conflicts(certificate: dict, ports: dict[int, int]) -> list[dict]:
    """Charging intervals of the relaxed schedule that exceed a station's ports."""
    intervals: dict[int, list[tuple[float, float]]] = {}
    for route in certificate["routes"]:
        for stop in route["stops"]:
            if stop["kind"] == "charger" and stop["charge_h"] > 0:
                end = stop["departure_h"]
                intervals.setdefault(int(stop["node"]), []).append((end - stop["charge_h"], end))
    conflicts = []
    for node, spans in intervals.items():
        events = sorted([(a, 1) for a, _ in spans] + [(b, -1) for _, b in spans])
        busy = peak = 0
        for _, change in events:
            busy += change
            peak = max(peak, busy)
        if peak > ports.get(node, math.inf):
            conflicts.append({"node": node, "peak": peak, "ports": ports[node]})
    return conflicts
