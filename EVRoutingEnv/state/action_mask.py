"""
Action feasibility mask generation for the truck routing environment.

This module provides functionality to determine which actions are feasible
for the active truck based on battery constraints, location, and state.
"""

from typing import TYPE_CHECKING

import numpy as np

from EVRoutingEnv.state.feasibility import (
    FeasibilityReason,
    joint_action_feasibility,
)


# Slots that denote no action at all rather than an action that happens to be
# infeasible right now.  An unmasked policy still may not select these: there is
# no customer behind an empty slot to route to, and with no active truck there is
# no decision to make.
STRUCTURALLY_UNDEFINED = frozenset(
    {
        FeasibilityReason.EMPTY_ACTION_SLOT,
        FeasibilityReason.NO_ACTIVE_TRUCK,
    }
)


if TYPE_CHECKING:
    from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv


def get_action_mask(env: "EventDrivenTruckEnv") -> np.ndarray:
    """
    Generate the hard mask from the shared routing feasibility engine.

    Args:
        env: EventDrivenTruckEnv instance

    Returns:
        np.ndarray: Boolean array where True indicates feasible actions.
                   Shape: (action_space.n,)
                   Order: chargers, global customer slots, depot, configured charge values.
    """
    decisions = joint_action_feasibility(env)
    env.last_action_feasibility = decisions
    return np.asarray([item.feasible for item in decisions], dtype=bool)


def get_structural_action_mask(env: "EventDrivenTruckEnv") -> np.ndarray:
    """Return the candidate set without any feasibility filtering.

    This is the mask for the no-mask ablation.  It differs from
    :func:`get_action_mask` in exactly one respect: an action that the
    feasibility engine rejects for a *dynamic* reason -- not enough energy, the
    task is already claimed, the truck is not at a charger -- stays selectable,
    and the environment rejects it at execution time under
    ``environment.invalid_action_mode``.  Slots that denote no action at all are
    still hidden, so both arms score the same number of real candidates and the
    comparison isolates the mask rather than the action space.
    """
    decisions = joint_action_feasibility(env)
    env.last_action_feasibility = decisions
    return np.asarray(
        [item.reason not in STRUCTURALLY_UNDEFINED for item in decisions],
        dtype=bool,
    )


def policy_action_mask(env: "EventDrivenTruckEnv") -> np.ndarray:
    """Return whichever mask ``env`` hands to a learning policy.

    Environments that predate the mask ablation -- and the lightweight stubs in
    the test suite -- only expose ``mask_fn``, so fall back to it.
    """
    accessor = getattr(env, "policy_mask_fn", None)
    mask = env.mask_fn() if accessor is None else accessor()
    if (
        not getattr(env, "config", {})
        .get("environment", {})
        .get("completion_shield", False)
    ):
        return mask
    return completion_action_mask(env, mask)


def completion_action_mask(env, legal_mask: np.ndarray) -> np.ndarray:
    """Optional robust escape/capacity restriction of the execution-legal set.

    This is a conservative policy shield, not a change to simulator legality.
    A fallback restores only legal choices when the restriction empties a set.
    """
    from EVRoutingEnv.state.feasibility import ActionKind

    if env.active_truck_id is None:
        return legal_mask
    truck = env.trucks[env.active_truck_id]
    multiplier = 1.0
    if (
        env.traffic_config["enable_traffic"]
        and env.traffic_config["enable_energy_uncertainty"]
    ):
        multiplier = max(1.0, float(env.traffic_config["max_energy_multiplier"]))
    decisions = joint_action_feasibility(env)
    safe_nodes = [env.joint_instance.depot_node] + [
        node
        for node in env.charging_nodes
        if env.charging_station.station_available[node]
    ]
    result = legal_mask.copy()
    all_tasks = list(env.task_registry.tasks())
    tasks = [task for task in all_tasks if task.is_available]
    other_payload = []
    for other in env.trucks:
        if (
            other.truck_id == truck.truck_id
            or other.failed
            or other.is_complete
            or getattr(other, "return_to_depot_pending", False)
            or getattr(other, "route_destination", None)
            == env.joint_instance.depot_node
        ):
            continue
        # Payload is consumed at service completion, so reserve demands already
        # claimed by an in-flight/servicing truck before counting its capacity.
        committed = sum(
            task.demand
            for task in all_tasks
            if not task.is_available
            and not getattr(task, "is_served", False)
            and getattr(task, "claimed_by", None) == other.truck_id
        )
        capacity = (
            float("inf")
            if other.remaining_payload is None
            else max(0.0, float(other.remaining_payload) - committed)
        )
        other_payload.append((other.truck_id, capacity))
    other_capacity = sum(capacity for _, capacity in other_payload)
    exact_lookahead = getattr(env, "config", {}).get("environment", {}).get(
        "completion_capacity_exact_lookahead", False
    )
    capacity_lookahead = exact_lookahead or getattr(env, "config", {}).get("environment", {}).get(
        "completion_capacity_lookahead", False
    )
    work_escape = getattr(env, "config", {}).get("environment", {}).get(
        "completion_work_escape_lookahead", False
    )
    for index in np.flatnonzero(legal_mask):
        decision = decisions[index]
        if decision.action_kind is ActionKind.CUSTOMER:
            remaining = [
                task for task in tasks if task.node_id != decision.target_node
            ]
            selected = next(
                task for task in tasks if task.node_id == decision.target_node
            )
            active_capacity = (
                float("inf")
                if truck.remaining_payload is None
                else max(0.0, float(truck.remaining_payload) - selected.demand)
            )
            escape_nodes = safe_nodes
            if work_escape and not remaining_payload_can_fit(
                remaining, other_payload, exact=exact_lookahead
            ) and needs_charge_before_retiring(env, truck, decision.target_node, active_capacity, remaining, other_payload):
                # Reaching the depot retires this truck. If the other trucks
                # provably cannot carry the residual work and even optimistic
                # energy use requires a recharge, reserve enough
                # battery to reach a charger instead of treating retirement as
                # a usable escape. A depot which is also a charger stays valid.
                escape_nodes = [node for node in safe_nodes if node in env.charging_nodes and env.charging_station.station_available[node]]
            escape = (
                min(
                    (env.transport_graph.get_path_energy(decision.target_node, node)
                     for node in escape_nodes),
                    default=float("inf"),
                )
                * multiplier
            )
            if (
                float(truck.current_battery) - float(decision.required_energy) + 1e-8
                < escape
            ):
                result[index] = False
            if capacity_lookahead:
                if not remaining_payload_can_fit(
                    remaining, [(truck.truck_id, active_capacity), *other_payload],
                    exact=exact_lookahead,
                ):
                    result[index] = False
        elif decision.action_kind is ActionKind.DEPOT and tasks:
            owned = any(task.preassigned_to == truck.truck_id for task in tasks)
            stranded_task = any(
                not any(
                    capacity + 1e-8 >= task.demand
                    and (task.preassigned_to is None or task.preassigned_to == other_id)
                    for other_id, capacity in other_payload
                )
                for task in tasks
            )
            if (
                owned
                or stranded_task
                or other_capacity + 1e-8 < sum(task.demand for task in tasks)
                or (
                    capacity_lookahead
                    and not remaining_payload_can_fit(tasks, other_payload, exact=exact_lookahead)
                )
            ):
                result[index] = False
    if not result.any() and legal_mask.any():
        return legal_mask
    return result


def needs_charge_before_retiring(env, truck, target, capacity, tasks, other_payload):
    """A lower bound on remaining work proves a recharge will be necessary.

    Crucially, payload necessity alone does not imply charging necessity: a
    truck may finish several nearby customers and return without any recharge.
    Use optimistic energy for both the proposed leg and the continuation, so
    this predicate never relies on an unrevealed disturbance realization.
    """
    multiplier = 1.0
    if env.traffic_config["enable_traffic"] and env.traffic_config["enable_energy_uncertainty"]:
        multiplier = float(env.traffic_config["min_energy_multiplier"])
    eligible = [task for task in tasks if task.demand <= capacity + 1e-8 and (task.preassigned_to is None or task.preassigned_to == truck.truck_id)]
    mandatory = [task for task in eligible if not any(other_capacity + 1e-8 >= task.demand and (task.preassigned_to is None or task.preassigned_to == other_id) for other_id, other_capacity in other_payload)]
    bounds = [env.transport_graph.get_path_energy(target, task.node_id) + env.transport_graph.get_path_energy(task.node_id, env.joint_instance.depot_node) for task in (mandatory or eligible)]
    # With mandatory customers every one must be visited. Otherwise at least
    # one eligible customer is needed because the other fleet cannot finish.
    continuation = max(bounds) if mandatory else min(bounds, default=float("inf"))
    optimistic_after = float(truck.current_battery) - multiplier * env.transport_graph.get_path_energy(truck.current_node, target)
    return optimistic_after + 1e-8 < multiplier * continuation


def remaining_payload_can_fit(tasks, payloads, *, exact=False) -> bool:
    """Necessary packing conditions; True does not certify an assignment.

    Every pending customer must fit an eligible remaining truck, total demand
    must fit total capacity, and for each demand threshold d there must be at
    least as many capacity slots of size d as customers with demand >= d.
    Threshold counts relax ownership; the individual-fit check respects it.
    All three conditions are necessary for any single-trip completion. Optional
    bounded search additionally rejects proven impossible residual packings.
    Its size/node limits return unknown, which never adds a rejection.
    """
    if not tasks:
        return True
    if not payloads:
        return False
    capacities = np.asarray([capacity for _, capacity in payloads], dtype=float)
    demands = np.asarray([float(task.demand) for task in tasks], dtype=float)
    if capacities.sum() + 1e-8 < demands.sum():
        return False
    for task in tasks:
        if not any(
            capacity + 1e-8 >= task.demand
            and (task.preassigned_to is None or task.preassigned_to == truck_id)
            for truck_id, capacity in payloads
        ):
            return False
    thresholds = np.sort(demands[demands > 1e-8])[::-1]
    slots = np.floor((capacities[None, :] + 1e-8) / thresholds[:, None]).sum(-1)
    if not bool(np.all(slots >= np.arange(1, len(thresholds) + 1))):
        return False
    return not exact or bounded_payload_packing(tasks, payloads) is not False


def bounded_payload_packing(tasks, payloads, *, max_tasks=12, max_nodes=4096):
    """Decide a small residual payload packing, or return None on a limit.

    A False result exhausts all assignments under a permissive numerical
    tolerance. True certifies payload packing only, never routing feasibility.
    Fixed owners are reserved first; remaining customers are indistinguishable
    with respect to truck eligibility, so equal-capacity bins are symmetric.
    """
    if len(tasks) > max_tasks:
        return None
    tolerance = 1e-8
    capacities = {truck_id: float(capacity) for truck_id, capacity in payloads}
    if len(capacities) != len(payloads):
        raise ValueError("duplicate truck identity in payload packing")
    reserved = {}
    demands = []
    for task in tasks:
        owner = task.preassigned_to
        if owner is None:
            demands.append(float(task.demand))
        else:
            reserved[owner] = reserved.get(owner, 0.0) + float(task.demand)
    for owner, demand in reserved.items():
        if owner not in capacities or demand > capacities[owner] + tolerance:
            return False
        capacities[owner] = max(0.0, capacities[owner] - demand)
    demands.sort(reverse=True)
    if not demands:
        return True
    initial = tuple(sorted(capacities.values(), reverse=True))
    if not initial:
        return False
    if np.isinf(initial[0]):
        return True
    suffix = np.cumsum(np.asarray(demands[::-1], dtype=float))[::-1]
    impossible = set()
    visited = 0

    def search(index, residual):
        nonlocal visited
        if index == len(demands):
            return True
        if visited >= max_nodes:
            return None
        visited += 1
        if demands[index] > residual[0] + tolerance or suffix[index] > sum(residual) + tolerance:
            return False
        state = (index, residual)
        if state in impossible:
            return False
        previous = None
        # Best-fitting bins first. Exact equality alone removes symmetric bins;
        # rounding capacities could turn a feasible assignment into a rejection.
        for position in range(len(residual) - 1, -1, -1):
            capacity = residual[position]
            if capacity == previous or demands[index] > capacity + tolerance:
                continue
            previous = capacity
            updated = list(residual)
            updated[position] = max(0.0, capacity - demands[index])
            result = search(index + 1, tuple(sorted(updated, reverse=True)))
            if result is True or result is None:
                return result
        impossible.add(state)
        return False

    return search(0, initial)
