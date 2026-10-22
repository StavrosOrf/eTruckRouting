"""Nominal network data shared by the long-haul planner and executor.

Everything here is read from the environment once per map and cached: the
nominal leg tables (hours, kWh), the station charging curve, and the
*waypoint closure* of the leg costs. The closure matters on these maps: the
shipped shortest-path tables violate the triangle inequality (a leg through a
station can be faster than the direct table entry), and every leg longer than
``capacity / margin`` kWh needs at least one station on the way anyway.

Leg cost for planning is ``hours + kWh / rate``: every route here recharges
more than it starts with, so each extra kWh driven is an extra kWh charged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from EVRoutingEnv.models.simulation import queueing


_NETWORK_CACHE: dict[tuple, LonghaulNetwork] = {}
_CURVE_CACHE: dict[tuple, ChargeCurve] = {}


def energy_margin(env) -> float:
    """The mask's per-leg energy multiplier (1.2 when energy is uncertain)."""
    traffic = env.traffic_config
    if traffic["enable_traffic"] and traffic["enable_energy_uncertainty"]:
        return max(1.0, float(traffic["max_energy_multiplier"]))
    return 1.0


class ChargeCurve:
    """Hours to charge from one SoC to a target level, from the env's own curve."""

    GRID = 400

    def __init__(self, env, charger_node: int, levels):
        station = env.charging_station
        model = env.charging_curve_model
        charger_type = station.charger_type.get(int(charger_node), "DCFast")
        section = "dcfast" if charger_type == "DCFast" else "level2"
        config = dict(env.charging_config.get(section, {}))
        config["use_realistic_curve"] = env.charging_config.get("use_realistic_curve", False)
        power = float(station.charger_power_kw.get(int(charger_node), 150.0))
        config["charge_rate"] = power
        config["efficiency"] = float(
            station.charger_efficiency.get(int(charger_node), config.get("efficiency", 0.9))
        )
        if "taper_power_min" in config:
            config["taper_power_min"] = min(float(config["taper_power_min"]), power)
        self.capacity = float(env.config["truck"]["battery_capacity"])
        self.levels = tuple(float(level) for level in levels)
        self.grid = np.linspace(0.0, 1.0, self.GRID + 1)
        self.tables = {}
        for level in self.levels:
            values = np.zeros(self.GRID + 1)
            for index, soc in enumerate(self.grid):
                if soc >= level - 1e-12:
                    values[index] = 0.0
                    continue
                _, details = model.calculate_charge_to_target(
                    initial_soc=float(soc),
                    target_soc=float(level),
                    battery_capacity=self.capacity,
                    charger_config=dict(config),
                    charger_type=charger_type,
                )
                values[index] = float(details["actual_charge_hours"])
            self.tables[level] = values
        full = self.tables[max(self.levels)]
        # Effective kWh per hour at mid SoC, used by linear planning costs.
        mid = int(self.GRID * 0.5)
        self.rate = (max(self.levels) - 0.5) * self.capacity / max(full[mid], 1e-9)

    def hours(self, battery: float, level: float) -> float:
        soc = min(max(battery / self.capacity, 0.0), 1.0)
        if soc >= level - 1e-12:
            return 0.0
        position = soc * self.GRID
        low = int(position)
        high = min(low + 1, self.GRID)
        table = self.tables[level]
        weight = position - low
        return float(table[low] * (1.0 - weight) + table[high] * weight)


@dataclass
class LonghaulNetwork:
    """Dense nominal tables over every node of the map, plus the closure."""

    position: dict[int, int]
    nodes: np.ndarray
    hours: np.ndarray
    energy: np.ndarray
    chargers: np.ndarray  # matrix indices of the stations
    capacity: float
    margin: float
    rate: float
    max_hop_hours: float
    closure_cost: np.ndarray = field(repr=False, default=None)
    closure_hours: np.ndarray = field(repr=False, default=None)
    closure_energy: np.ndarray = field(repr=False, default=None)
    closure_next: np.ndarray = field(repr=False, default=None)

    @property
    def max_hop_energy(self) -> float:
        return self.capacity / self.margin

    def hop_ok(self, source: int, target: int) -> bool:
        energy = self.energy[source, target]
        return (
            math.isfinite(energy)
            and self.margin * energy <= self.capacity + 1e-9
            and self.hours[source, target] <= self.max_hop_hours + 1e-9
        )

    def path(self, source: int, target: int) -> list[int]:
        """Matrix indices of the closure path, excluding ``source``."""
        nodes, current, guard = [], int(source), 0
        while current != int(target):
            current = int(self.closure_next[current, target])
            if current < 0 or guard > len(self.nodes):
                return []
            nodes.append(current)
            guard += 1
        return nodes


def _closure(network: LonghaulNetwork) -> None:
    """Floyd-Warshall with stations as the only intermediate nodes."""
    hours, energy = network.hours, network.energy
    feasible = (
        np.isfinite(energy)
        & np.isfinite(hours)
        & (network.margin * energy <= network.capacity + 1e-9)
        & (hours <= network.max_hop_hours + 1e-9)
    )
    cost = np.where(feasible, hours + energy / network.rate, np.inf)
    total_hours = np.where(feasible, hours, np.inf)
    total_energy = np.where(feasible, energy, np.inf)
    size = len(network.nodes)
    nxt = np.where(feasible, np.arange(size)[None, :], -1).astype(np.int32)
    np.fill_diagonal(cost, 0.0)
    np.fill_diagonal(total_hours, 0.0)
    np.fill_diagonal(total_energy, 0.0)
    np.fill_diagonal(nxt, np.arange(size))
    for k in network.chargers:
        alternative = cost[:, k, None] + cost[None, k, :]
        better = alternative < cost - 1e-12
        if not better.any():
            continue
        cost = np.where(better, alternative, cost)
        total_hours = np.where(better, total_hours[:, k, None] + total_hours[None, k, :], total_hours)
        total_energy = np.where(better, total_energy[:, k, None] + total_energy[None, k, :], total_energy)
        nxt = np.where(better, nxt[:, k, None], nxt)
    network.closure_cost = cost
    network.closure_hours = total_hours
    network.closure_energy = total_energy
    network.closure_next = nxt


def network_for(env, *, max_hop_hours: float | None = None) -> LonghaulNetwork:
    """The (cached) network of ``env``'s map."""
    graph = env.transport_graph
    capacity = float(env.config["truck"]["battery_capacity"])
    margin = energy_margin(env)
    levels = tuple(float(level) for level in env.charge_action_values)
    curve = charge_curve(env)
    hop = float(max_hop_hours) if max_hop_hours is not None else math.inf
    key = (
        str(env.config["network"].get("data_path")),
        len(env.charging_nodes),
        capacity,
        margin,
        round(curve.rate, 6),
        hop,
        levels,
    )
    cached = _NETWORK_CACHE.get(key)
    if cached is not None:
        return cached
    position, energy, hours = graph.cost_matrices()
    nodes = np.zeros(len(position), dtype=np.int64)
    for node, index in position.items():
        nodes[index] = int(node)
    network = LonghaulNetwork(
        position={int(node): int(index) for node, index in position.items()},
        nodes=nodes,
        hours=np.asarray(hours, dtype=float),
        energy=np.asarray(energy, dtype=float),
        chargers=np.asarray([position[int(node)] for node in env.charging_nodes], dtype=np.int64),
        capacity=capacity,
        margin=margin,
        rate=curve.rate,
        max_hop_hours=hop,
    )
    _closure(network)
    _NETWORK_CACHE[key] = network
    return network


def charge_curve(env) -> ChargeCurve:
    """Charging-time table; every station on these maps shares one class."""
    station = env.charging_station
    node = int(env.charging_nodes[0])
    classes = {
        (
            station.charger_type.get(int(n), "DCFast"),
            float(station.charger_power_kw.get(int(n), 150.0)),
            float(station.charger_efficiency.get(int(n), 0.9)),
        )
        for n in env.charging_nodes
    }
    if len(classes) != 1:
        raise NotImplementedError(
            f"the long-haul planner assumes one station class, found {sorted(classes)}"
        )
    levels = tuple(float(level) for level in env.charge_action_values)
    key = (next(iter(classes)), float(env.config["truck"]["battery_capacity"]), levels,
           bool(env.charging_config.get("use_realistic_curve", False)))
    cached = _CURVE_CACHE.get(key)
    if cached is None:
        cached = _CURVE_CACHE[key] = ChargeCurve(env, node, levels)
    return cached


def station_waits(env, network: LonghaulNetwork) -> np.ndarray | None:
    """Expected newcomer wait (hours) per matrix index, or None without queues.

    Plan-time estimate: the long-run M/M/c wait with outside traffic at the
    configured utilization, averaged over the binomial number of ports that are
    up (port availability), each state's wait capped at ``cap`` hours since an
    overloaded state does not last for the whole horizon. This is the same
    information every policy has before observing a station (Kullman et al.
    2021: the expected queue); the executor replaces it by the observed queue
    on arrival.
    """
    stations = getattr(env, "public_stations", None)
    if stations is None:
        return None
    from math import comb

    waits = np.zeros(len(network.nodes))
    config = stations.config
    availability = float(config.outages.port_availability) if config.outages.enabled else 1.0
    cap = 4.0
    for node in env.charging_nodes:
        parameters = stations.station_parameters(int(node))
        ports, rate, service = parameters.ports, parameters.mean_arrival_rate, parameters.service_rate
        expected = 0.0
        for up in range(ports + 1):
            probability = comb(ports, up) * availability**up * (1 - availability) ** (ports - up)
            if probability < 1e-12:
                continue
            if up == 0 or rate >= up * service:
                wait = cap
            else:
                wait = min(cap, queueing.mean_wait(up, rate, service))
            expected += probability * wait
        waits[network.position[int(node)]] = expected
    return waits
