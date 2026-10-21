"""Export the nominal deterministic instance of a reset environment for the
Julia branch-and-price solver (schema ``evrp-bp-instance/v1``).

The solver optimizes the expected-value problem: every leg costs its nominal
table time and energy, service takes its base duration, and charging follows
the station curve the simulator executes. Nothing here re-derives physics:
times and energies come from ``transport_graph.cost_matrices()`` and charge
times from the environment's own ``ChargingCurveModel``, so the Julia side
never re-implements either.

Charge time is exported per station as a cumulative table ``F(soc)``, the hours
needed to charge from empty to ``soc``. The curve integrates ``1 / P(soc)``, so
the time from ``a`` to ``b`` is ``F(b) - F(a)``; ``tests`` check that additivity
against direct calls. Targets are always discrete action levels, which lie on
the grid, so only the arrival SoC needs rounding (up for a lower bound, down for
a conservative plan) and the Julia reader can bound the error exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = "evrp-bp-instance/v1"

# 0.001 SoC is 0.4 kWh on a 400 kWh battery: well below one leg's energy grain.
SOC_GRID_STEP = 1e-3


def energy_margin(env) -> float:
    """Per-leg multiplier the action mask applies (``state/feasibility.py``)."""
    traffic = env.traffic_config
    if traffic["enable_traffic"] and traffic["enable_energy_uncertainty"]:
        return max(1.0, float(traffic["max_energy_multiplier"]))
    return 1.0


def station_charger_settings(env, node: int) -> tuple[str, dict]:
    """The charger config the environment passes to the curve model for ``node``."""
    station = env.charging_station
    kind = station.charger_type[node]
    settings = dict(env.charging_config["dcfast" if kind == "DCFast" else "level2"])
    settings.update(
        use_realistic_curve=env.charging_config["use_realistic_curve"],
        charge_rate=float(station.charger_power_kw[node]),
        efficiency=float(station.charger_efficiency[node]),
    )
    # Same clamp as ``_execute_charge_action``: a station rated below the taper
    # floor never tapers below its own power.
    if "taper_power_min" in settings:
        settings["taper_power_min"] = min(
            float(settings["taper_power_min"]), float(settings["charge_rate"]),
        )
    return kind, settings


def cumulative_charge_hours(env, node: int, step: float = SOC_GRID_STEP) -> list[float]:
    """``F(k * step)`` for ``k = 0..1/step``: hours from empty at station ``node``."""
    capacity = float(env.config["truck"]["battery_capacity"])
    kind, settings = station_charger_settings(env, node)
    model = env.charging_curve_model
    count = round(1.0 / step)
    table = [0.0]
    for k in range(1, count + 1):
        _, details = model.calculate_charge_to_target(
            initial_soc=(k - 1) * step,
            target_soc=k * step,
            battery_capacity=capacity,
            charger_config=settings,
            charger_type=kind,
        )
        table.append(table[-1] + float(details["actual_charge_hours"]))
    return table


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _config_sha(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def export_nominal_instance(env, *, objective: str = "travel", warm_start=None, sequences=None) -> dict:
    """Build the solver input for the episode ``env`` was last reset to.

    ``objective`` is ``"travel"`` (fleet travel hours, the campaign metric) or
    ``"operating"`` (travel plus service and charging hours). ``warm_start``
    optionally adds routes in witness format; ``sequences`` adds customer-only
    routes (environment node ids, e.g. an ALNS plan) that the solver completes
    with its own optimal charging stops.
    """
    from EVRoutingEnv.baselines.optimization.exact_optimization import (
        refuse_unrevealed_requests,
    )

    refuse_unrevealed_requests(env, "the nominal instance export (branch-and-price, RoutingBlocks)")
    if objective not in ("travel", "operating"):
        raise ValueError(f"objective must be 'travel' or 'operating', got {objective!r}")
    if env.joint_instance is None or env.task_registry is None:
        raise RuntimeError("export needs a joint-routing environment after reset()")
    if env.preassigned_customers or env.sequential_routing:
        raise NotImplementedError("only the joint_fleet assignment mode is supported")
    if env.charging_action_mode != "target_soc":
        raise NotImplementedError("only charging.action_mode = target_soc is supported")

    depot = int(env.joint_instance.depot_node)
    tasks = list(env.task_registry.tasks())
    chargers = [int(node) for node in env.charging_nodes]
    station = env.charging_station

    nodes = [{"id": depot, "kind": "depot"}]
    for task in tasks:
        nodes.append({
            "id": int(task.node_id),
            "kind": "customer",
            "task_id": int(task.task_id),
            "demand": float(task.demand),
            "service_h": float(task.base_service_time),
            "tw": [float(task.earliest_service), _finite_or_none(task.latest_service)],
        })
    for node in chargers:
        nodes.append({"id": node, "kind": "charger"})
    ids = [entry["id"] for entry in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("depot, customer and charger nodes must be distinct")

    index, energy, hours = env.transport_graph.cost_matrices()
    rows = np.array([index[node] for node in ids])
    E = energy[np.ix_(rows, rows)]
    H = hours[np.ix_(rows, rows)]

    curve_cache: dict[tuple, list[float]] = {}
    charger_entries = []
    for node in chargers:
        kind, settings = station_charger_settings(env, node)
        key = (kind, tuple(sorted((k, str(v)) for k, v in settings.items())))
        if key not in curve_cache:
            curve_cache[key] = cumulative_charge_hours(env, node)
        charger_entries.append({
            "node": node,
            "ports": int(station.charger_capacity[node]),
            "type": kind,
            "power_kw": float(station.charger_power_kw[node]),
            "efficiency": float(station.charger_efficiency[node]),
            "available": bool(station.station_available[node]),
            "curve": list(curve_cache).index(key),
        })

    battery = float(env.config["truck"]["battery_capacity"])
    start_soc = [float(truck.current_battery) / battery for truck in env.trucks]
    witness = [
        [[int(node), float(value), bool(charging)] for node, value, charging in route]
        for route in env.joint_instance.witness_routes
    ]

    def matrix(values: np.ndarray) -> list[list[float | None]]:
        return [[_finite_or_none(v) for v in row] for row in values]

    return {
        "schema": SCHEMA,
        "meta": {
            "seed": getattr(env, "scenario_seed", None),
            "config_sha": _config_sha(env.config),
            "units": {"time": "h", "energy": "kWh", "load": "demand units"},
        },
        "objective": objective,
        "nodes": nodes,
        "chargers": charger_entries,
        "charge_curves": {
            "soc_step": SOC_GRID_STEP,
            "cumulative_hours": list(curve_cache.values()),
        },
        "vehicle": {
            "count": int(env.num_trucks),
            "battery_kwh": battery,
            "payload": float(env.trucks[0].payload_capacity),
            "start_soc": start_soc,
        },
        "energy_margin": energy_margin(env),
        "horizon_h": float(env.max_time),
        "step_budget": int(env.max_episode_steps),
        "charge_levels": [float(v) for v in env.charge_action_values],
        "energy_kwh": matrix(E),
        "time_h": matrix(H),
        "warm_start": {
            "witness": witness,
            "extra": list(warm_start or []),
            "sequences": [[int(node) for node in seq] for seq in (sequences or []) if seq],
        },
    }


def write_instance(env, path: str | Path, **kwargs) -> dict:
    instance = export_nominal_instance(env, **kwargs)
    Path(path).write_text(json.dumps(instance))
    return instance
