"""Constructive energy witnesses, followed by event-engine certification.

This is a feasibility heuristic, not an optimal solver. Failure to construct a
witness means resample/reject, never label an unverified instance feasible.
"""

from __future__ import annotations

import heapq
import math
from itertools import count

import numpy as np


class NoFeasibleCertificate(ValueError):
    """A candidate has no certificate from the bounded constructive search."""


def construct_energy_routes(
    instance,
    graph,
    charging_nodes,
    *,
    num_trucks,
    battery_capacity,
    initial_batteries,
    energy_multiplier=1.0,
    charge_transition=None,
    min_hop_energy_kwh=0.0,
    max_hop_energy_kwh=math.inf,
):
    """Return per-truck public tuple actions, including the final depot return.

    Partition/order are hidden witnesses in fleet mode, not imposed ownership.
    Reserve uses worst-case energy, including the exit from every customer.
    Chargers may be chained; no artificial one-charger-hop restriction applies.
    Bounds apply to nominal depot/customer->next-customer legs in this witness,
    not to charging detours or depot retirement.
    """
    depot = instance.depot_node
    chargers = tuple(sorted(charging_nodes))
    charger_set = frozenset(chargers)
    owners = instance.witness_owners
    if len(owners) != len(instance.tasks):
        raise NoFeasibleCertificate("missing feasible payload partition")

    # Vectorised lookups when the graph can provide float64 all-pairs matrices: the scan over every
    # charger below is the dominant cost of generating an instance on networks with hundreds of
    # chargers, and per-pair dictionary lookups made it take tens of seconds. The values are the
    # same numbers get_path_energy / get_time_distance return, so the witness is unchanged.
    matrices = graph.cost_matrices() if hasattr(graph, "cost_matrices") else None
    if matrices is not None:
        index, energy_matrix, hours_matrix = matrices
        charger_index = np.array([index[c] for c in chargers], dtype=np.int64)

        def energy(a, b):
            return float(energy_matrix[index[a], index[b]]) * energy_multiplier
    else:

        def energy(a, b):
            return graph.get_path_energy(a, b) * energy_multiplier

    charge_memo = {}

    def charge(node, battery):
        if charge_transition is None:
            return battery_capacity, 0.0, 1.0
        key = (node, battery)
        if key not in charge_memo:
            charge_memo[key] = charge_transition(node, battery)
        return charge_memo[key]

    charger_position = {c: i for i, c in enumerate(chargers)}

    def reachable_chargers(node, available):
        """Fallback scan for graphs without cost matrices, in sorted-charger order."""
        for target in chargers:
            if target == node:
                continue
            required = energy(node, target)
            if not math.isfinite(required) or required > available + 1e-9:
                continue
            hours = graph.get_time_distance(node, target) if hasattr(graph, "get_time_distance") else 0.0
            yield target, required, hours

    def path(origin, battery, destination, reserve):
        serial = count()
        pending = [(0.0, next(serial), origin, battery, ())]
        labels = {}
        # Pareto labels (battery, cost) per charger for the vectorised branch, as padded arrays.
        # An empty slot holds battery=-inf so it can never dominate anything.
        label_battery = np.full((len(chargers), 8), -np.inf)
        label_cost = np.full((len(chargers), 8), np.inf)
        label_count = np.zeros(len(chargers), dtype=np.int64)
        expansions = 0
        # Target-SoC search has at most one post-charge level per station.
        # Duration actions can need revisits; bound those explicitly as well.
        limit = max(32, 8 * (len(chargers) + 1) ** 2)
        while pending and expansions < limit:
            cost, _, node, available, actions = heapq.heappop(pending)
            expansions += 1
            to_destination = energy(node, destination)
            if to_destination + reserve <= available + 1e-9:
                remaining = max(0.0, available - to_destination)
                return (*actions, (destination, 0.0, False)), remaining
            departures = [(available, cost, actions)]
            if node in charger_set and (not actions or not actions[-1][2]):
                charged, hours, value = charge(node, available)
                if charged > available + 1e-9:
                    departures.append(
                        (charged, cost + hours, (*actions, (node, value, True)))
                    )
            if matrices is not None:
                row = energy_matrix[index[node]][charger_index] * energy_multiplier
                row_hours = hours_matrix[index[node]][charger_index]
                node_position = charger_position.get(node)
            for departure_battery, departure_cost, prefix in departures:
                if to_destination + reserve <= departure_battery + 1e-9:
                    return (*prefix, (destination, 0.0, False)), max(
                        0.0, departure_battery - to_destination
                    )
                if matrices is None:
                    for target, required, hours in reachable_chargers(node, departure_battery):
                        remaining = max(0.0, departure_battery - required)
                        new_cost = departure_cost + hours + 1e-6
                        prior = labels.setdefault(target, [])
                        if any(b >= remaining - 1e-9 and c <= new_cost for b, c in prior):
                            continue
                        prior[:] = [
                            (b, c)
                            for b, c in prior
                            if not (remaining >= b - 1e-9 and new_cost <= c)
                        ]
                        prior.append((remaining, new_cost))
                        heapq.heappush(
                            pending,
                            (new_cost, next(serial), target, remaining, (*prefix, (target, 0.0, False))),
                        )
                    continue
                # Vectorised candidate generation and dominance test. Every target appears at most
                # once per departure, so testing all candidates against the labels held before this
                # departure is the same as testing them one by one in charger order.
                ok = np.isfinite(row) & (row <= departure_battery + 1e-9)
                if node_position is not None:
                    ok[node_position] = False
                ks = np.nonzero(ok)[0]
                if ks.size == 0:
                    continue
                remaining_all = np.maximum(0.0, departure_battery - row[ks])
                cost_all = departure_cost + row_hours[ks] + 1e-6
                dominated = (
                    (label_battery[ks] >= (remaining_all - 1e-9)[:, None])
                    & (label_cost[ks] <= cost_all[:, None])
                ).any(axis=1)
                for j in np.nonzero(~dominated)[0]:
                    k = int(ks[j])
                    remaining = float(remaining_all[j])
                    new_cost = float(cost_all[j])
                    n = int(label_count[k])
                    keep = ~(
                        (remaining >= label_battery[k, :n] - 1e-9)
                        & (new_cost <= label_cost[k, :n])
                    )
                    kept = int(keep.sum())
                    if kept < n:
                        label_battery[k, :kept] = label_battery[k, :n][keep]
                        label_cost[k, :kept] = label_cost[k, :n][keep]
                        label_battery[k, kept:n] = -np.inf
                        label_cost[k, kept:n] = np.inf
                    if kept >= label_battery.shape[1]:
                        pad = np.full((len(chargers), label_battery.shape[1]), -np.inf)
                        label_battery = np.concatenate([label_battery, pad], axis=1)
                        label_cost = np.concatenate([label_cost, np.full_like(pad, np.inf)], axis=1)
                    label_battery[k, kept] = remaining
                    label_cost[k, kept] = new_cost
                    label_count[k] = kept + 1
                    heapq.heappush(
                        pending,
                        (new_cost, next(serial), chargers[k], remaining, (*prefix, (chargers[k], 0.0, False))),
                    )
        raise NoFeasibleCertificate(
            "no complete battery-feasible customer/charger path"
        )

    routes = []
    for truck_id in range(num_trucks):
        node, battery = depot, initial_batteries[truck_id]
        actions = []
        for task, owner in zip(instance.tasks, owners, strict=True):
            if owner != truck_id:
                continue
            hop = graph.get_path_energy(node, task.node_id)
            if not min_hop_energy_kwh - 1e-9 <= hop <= max_hop_energy_kwh + 1e-9:
                raise NoFeasibleCertificate(
                    "customer hop outside configured kWh bounds"
                )
            reserve = min(energy(task.node_id, anchor) for anchor in (*chargers, depot))
            segment, battery = path(node, battery, task.node_id, reserve)
            actions.extend(segment)
            node = task.node_id
        segment, battery = path(node, battery, depot, 0.0)
        actions.extend(segment)
        routes.append(tuple(actions))
    return tuple(routes)


def replay_routes(env, routes):
    """Run a witness on an isolated, freshly reset simulator state.

    Skip a planned recharge only if the realized SoC already exceeds its target.
    This handles conservative energy planning without issuing invalid actions.
    Every executed action belongs to the configured discrete action envelope.
    """
    positions = [0] * len(routes)
    trace = []
    while not env._episode_ended:
        tid = env.active_truck_id
        if tid is None or positions[tid] >= len(routes[tid]):
            raise NoFeasibleCertificate(
                "witness exhausted before successful termination"
            )
        node, value, charging = routes[tid][positions[tid]]
        positions[tid] += 1
        truck = env.trucks[tid]
        if charging and (
            (
                env.charging_action_mode == "target_soc"
                and truck.current_battery >= value * truck.battery_capacity - 1e-9
            )
            or truck.current_battery >= truck.battery_capacity - 1e-9
        ):
            continue
        if charging:
            action = env.num_navigation_actions + env.charge_action_values.index(value)
        elif node == env.joint_instance.depot_node:
            action = env.num_charging_nodes + env.fixed_num_stops
        elif node in env.charging_nodes:
            action = env.charging_nodes.index(node)
        else:
            task = env.task_registry.task_for_node(node)
            action = env.num_charging_nodes + task.task_id
        trace.append((tid, action))
        env.step(action)
        if env.invalid_action_count or any(truck.failed for truck in env.trucks):
            raise NoFeasibleCertificate(
                "witness failed under configured execution semantics"
            )
    if not env.task_registry.all_served() or not all(t.is_complete for t in env.trucks):
        raise NoFeasibleCertificate("witness missed a window, horizon or action budget")
    return tuple(trace)
