"""Constructive energy witnesses, followed by event-engine certification.

This is a feasibility heuristic, not an optimal solver. Failure to construct a
witness means resample/reject, never label an unverified instance feasible.
"""

from __future__ import annotations

import heapq
import math
from itertools import count


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
    owners = instance.witness_owners
    if len(owners) != len(instance.tasks):
        raise NoFeasibleCertificate("missing feasible payload partition")

    def energy(a, b):
        return graph.get_path_energy(a, b) * energy_multiplier

    def charge(node, battery):
        if charge_transition is None:
            return battery_capacity, 0.0, 1.0
        return charge_transition(node, battery)

    def path(origin, battery, destination, reserve):
        serial = count()
        pending = [(0.0, next(serial), origin, battery, ())]
        labels = {}
        expansions = 0
        # Target-SoC search has at most one post-charge level per station.
        # Duration actions can need revisits; bound those explicitly as well.
        limit = max(32, 8 * (len(chargers) + 1) ** 2)
        while pending and expansions < limit:
            cost, _, node, available, actions = heapq.heappop(pending)
            expansions += 1
            if energy(node, destination) + reserve <= available + 1e-9:
                remaining = max(0.0, available - energy(node, destination))
                return (*actions, (destination, 0.0, False)), remaining
            departures = [(available, cost, actions)]
            if node in chargers and (not actions or not actions[-1][2]):
                charged, hours, value = charge(node, available)
                if charged > available + 1e-9:
                    departures.append(
                        (charged, cost + hours, (*actions, (node, value, True)))
                    )
            for departure_battery, departure_cost, prefix in departures:
                if energy(node, destination) + reserve <= departure_battery + 1e-9:
                    return (*prefix, (destination, 0.0, False)), max(
                        0.0, departure_battery - energy(node, destination)
                    )
                for target in chargers:
                    if target == node:
                        continue
                    required = energy(node, target)
                    if (
                        not math.isfinite(required)
                        or required > departure_battery + 1e-9
                    ):
                        continue
                    remaining = max(0.0, departure_battery - required)
                    hours = (
                        graph.get_time_distance(node, target)
                        if hasattr(graph, "get_time_distance")
                        else 0.0
                    )
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
                        (
                            new_cost,
                            next(serial),
                            target,
                            remaining,
                            (*prefix, (target, 0.0, False)),
                        ),
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
