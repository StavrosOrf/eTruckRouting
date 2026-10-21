"""Baselines for the public-station and dynamic-request components.

* ``KullmanMyopicPolicy`` -- Kullman, Goodson & Mendoza (2021) sec. 5.2.3: the
  cheapest immediate action with their added rules.
* ``GreedyDispatchPolicy`` -- the EURO Meets NeurIPS 2022 "greedy" baseline: at
  the depot, load every feasible released request, nearest first; wait if
  there is none.
* ``WaitAtDepotPolicy`` -- a parametric policy motivated by Voccia, Campbell &
  Thomas (2019)'s result on waiting (not a reproduction of it): wait while
  fewer than ``k`` released requests are loadable and deadlines allow.
* ``RollingHorizonALNSPolicy`` -- re-plans the active truck's next trip at
  every decision with on-board requests fixed (destroy/repair over the
  released requests), the standard dynamic baseline.

All of them read the hard-feasible action set and never relax it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from EVRoutingEnv.baselines.heuristics.canonical_baselines import (
    ActionKind,
    CandidateAction,
    GreedyHeuristicPolicy,
    HeuristicParameters,
    _energy,
    _travel_hours,
    component_action,
    decode_feasible_actions,
)
from EVRoutingEnv.models.core.requests import RequestStatus
from EVRoutingEnv.models.simulation import queueing


def _by_kind(candidates):
    grouped: dict[ActionKind, list[CandidateAction]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.kind, []).append(candidate)
    return grouped


def _slot(env, candidate: CandidateAction) -> int:
    return candidate.index - env.num_charging_nodes


class KullmanMyopicPolicy:
    """Kullman et al. (2021) sec. 5.2.3, rule by rule.

    1. Once every customer is served and the depot is reachable, return.
    2. At an available station, charge to full.
    3. At a busy station, queue only if the observed queue is shorter than the
       expected queue (the long-run M/M/c L_q).
    4. Prefer customers over other nodes whenever a customer is reachable.
    5. Visit at most two public stations between customers.
    Otherwise take the cheapest immediate action (travel hours, plus the
    expected wait at a station).
    """

    name = "kullman_myopic"
    MAX_STATIONS_BETWEEN_CUSTOMERS = 2

    def __call__(self, env, observation=None, info=None) -> int:
        candidates = decode_feasible_actions(env)
        grouped = _by_kind(candidates)
        truck = env.trucks[env.active_truck_id]
        node = int(truck.current_node)
        charges = grouped.get(ActionKind.CHARGE, [])
        stations = getattr(env, "public_stations", None)

        if grouped.get(ActionKind.QUEUE):
            observed = stations.queue_at(node, env.global_clock).queue_length()
            params = stations.station_parameters(node)
            expected = queueing.mean_queue_length(
                params.ports, params.mean_arrival_rate, params.service_rate
            )
            if observed < expected:
                return grouped[ActionKind.QUEUE][0].index
        elif charges:
            return max(charges, key=lambda action: action.charge_value).index

        depot = grouped.get(ActionKind.DEPOT, [])
        if depot and env.task_registry.all_served():
            return depot[0].index
        customers = grouped.get(ActionKind.CUSTOMER, [])
        if customers:
            return min(customers, key=lambda a: _travel_hours(env, node, a.target_node)).index
        chargers = grouped.get(ActionKind.CHARGER, [])
        if chargers and _stations_since_customer(truck) < self.MAX_STATIONS_BETWEEN_CUSTOMERS:
            def cost(action):
                wait = 0.0
                if stations is not None:
                    wait = stations.long_run_wait(action.target_node)
                return _travel_hours(env, node, action.target_node) + wait

            return min(chargers, key=cost).index
        component = component_action(env, truck, grouped)
        if component is not None:
            return component
        if depot:
            return depot[0].index
        others = [a for a in candidates if a.kind is not ActionKind.QUEUE]
        return (others or candidates)[0].index


def _stations_since_customer(truck) -> int:
    """Consecutive charger arrivals since the last customer or depot stop."""
    count = 0
    for event in reversed(truck.event_history):
        if event["event_type"] == "CUSTOMER_SERVICE_COMPLETE":
            break
        if event["event_type"] == "CHARGING_START":
            count += 1
    return count


@dataclass(frozen=True)
class DispatchParameters:
    energy_safety_factor: float = 1.25
    target_soc: float = 0.9
    # WaitAtDepotPolicy: wait while fewer than this many requests are loadable.
    wait_threshold: int = 1
    # RollingHorizonALNSPolicy search budget.
    iterations: int = 200
    seed: int = 0


class GreedyDispatchPolicy:
    """EURO 'greedy': load released requests nearest first, deliver, return."""

    name = "greedy_dispatch"

    def __init__(self, parameters: DispatchParameters | None = None):
        self.parameters = parameters or DispatchParameters()
        self.navigator = GreedyHeuristicPolicy(HeuristicParameters(
            energy_safety_factor=self.parameters.energy_safety_factor,
            target_soc=self.parameters.target_soc,
            require_continuation=False,
        ))

    def __call__(self, env, observation=None, info=None) -> int:
        candidates = decode_feasible_actions(env)
        grouped = _by_kind(candidates)
        truck = env.trucks[env.active_truck_id]
        component = component_action(env, truck, grouped)
        if component is not None:
            return component
        engine = env.request_engine
        depot = int(env.joint_instance.depot_node)
        node = int(truck.current_node)
        registry = engine.registry
        on_board = registry.on_board(truck.truck_id)
        customers = grouped.get(ActionKind.CUSTOMER, [])

        if engine.is_drayage:
            container = engine.container(truck.truck_id)
            if container is not None:
                return self._toward(env, truck, candidates, grouped, container.request_id,
                                    int(container.delivery.node))
            origins = [a for a in customers if registry.get(_slot(env, a)).status is RequestStatus.OPEN]
            if origins:
                best = min(origins, key=lambda a: _travel_hours(env, node, a.target_node))
                return best.index
            return self._idle(env, truck, candidates, grouped, depot)

        loadable = self._loadable(env, customers, node) if node == depot else []
        if loadable:
            choice = self.choose_load(env, truck, loadable)
            if choice is not None:
                return choice
        if on_board:
            target = min(on_board, key=lambda r: _travel_hours(env, node, int(r.delivery.node)))
            return self._toward(env, truck, candidates, grouped, target.request_id,
                                int(target.delivery.node))
        if node != depot:
            return self._toward_depot(env, truck, candidates, grouped, depot)
        return self._idle(env, truck, candidates, grouped, depot)

    def _loadable(self, env, customers, depot) -> list[CandidateAction]:
        registry = env.request_engine.registry
        result = []
        for action in customers:
            request = registry.get(_slot(env, action))
            if (
                request.status is RequestStatus.OPEN
                and request.release_time <= env.global_clock + 1e-9
                and int(request.pickup.node) == depot
            ):
                result.append(action)
        return result

    def choose_load(self, env, truck, loadable) -> int | None:
        """The next request to load, or None to stop loading and depart."""
        depot = int(env.joint_instance.depot_node)
        registry = env.request_engine.registry
        return min(
            loadable,
            key=lambda a: _travel_hours(env, depot, int(registry.get(_slot(env, a)).delivery.node)),
        ).index

    def _toward(self, env, truck, candidates, grouped, request_id, node) -> int:
        direct = [a for a in grouped.get(ActionKind.CUSTOMER, []) if _slot(env, a) == request_id]
        safety = self.parameters.energy_safety_factor
        if direct and truck.current_battery >= safety * _energy(env, int(truck.current_node), node):
            return direct[0].index
        return self.navigator.navigate_toward(env, truck, node, candidates, grouped)

    def _toward_depot(self, env, truck, candidates, grouped, depot) -> int:
        direct = grouped.get(ActionKind.DEPOT, [])
        if direct:
            return direct[0].index
        return self.navigator.navigate_toward(env, truck, depot, candidates, grouped)

    def _idle(self, env, truck, candidates, grouped, depot) -> int:
        node = int(truck.current_node)
        if node != depot:
            return self._toward_depot(env, truck, candidates, grouped, depot)
        charges = grouped.get(ActionKind.CHARGE, [])
        soc = truck.current_battery / truck.battery_capacity
        if charges and soc < self.parameters.target_soc:
            return GreedyHeuristicPolicy._pick_charge_action(charges, self.parameters.target_soc)
        if grouped.get(ActionKind.WAIT):
            return grouped[ActionKind.WAIT][0].index
        if grouped.get(ActionKind.END_SHIFT):
            return grouped[ActionKind.END_SHIFT][0].index
        return candidates[0].index


class WaitAtDepotPolicy(GreedyDispatchPolicy):
    """Wait at the depot until ``k`` requests are loadable, if deadlines allow."""

    name = "wait_at_depot"

    def choose_load(self, env, truck, loadable) -> int | None:
        engine = env.request_engine
        depot = int(env.joint_instance.depot_node)
        if len(loadable) < self.parameters.wait_threshold and engine.future_information_possible(
            env.global_clock
        ):
            wait = engine.config.wait_hours
            urgent = min(
                engine.registry.get(_slot(env, a)).delivery.latest
                - _travel_hours(env, depot, int(engine.registry.get(_slot(env, a)).delivery.node))
                for a in loadable
            )
            on_board = engine.registry.on_board(truck.truck_id)
            if not on_board and env.global_clock + wait + engine.config.loading_service_hours <= urgent:
                grouped = _by_kind(decode_feasible_actions(env))
                if grouped.get(ActionKind.WAIT):
                    return grouped[ActionKind.WAIT][0].index
        return super().choose_load(env, truck, loadable)


class RollingHorizonALNSPolicy(GreedyDispatchPolicy):
    """Re-plan the next trip at every decision; on-board requests stay fixed.

    The trip is a destroy/repair search over the released, loadable requests
    for the active truck: requests served on time first, then trip hours. It
    replaces the day-ahead ALNS, which cannot see unrevealed requests.
    """

    name = "rolling_alns"

    def choose_load(self, env, truck, loadable) -> int | None:
        engine = env.request_engine
        registry = engine.registry
        depot = int(env.joint_instance.depot_node)
        on_board = registry.on_board(truck.truck_id)
        capacity = float(truck.payload_capacity)
        pool = [registry.get(_slot(env, a)) for a in loadable]
        loading = engine.config.loading_service_hours
        rng = random.Random(hash((self.parameters.seed, int(env.scenario_seed), round(env.global_clock, 6))))

        def evaluate(trip):
            new = sum(request not in on_board for request in trip)
            time, node, on_time = env.global_clock + loading * new, depot, 0
            for request in trip:
                time += _travel_hours(env, node, int(request.delivery.node))
                time = max(time, request.delivery.earliest)
                on_time += time <= request.delivery.latest + 1e-9
                time += request.delivery.service_hours
                node = int(request.delivery.node)
            time += _travel_hours(env, node, depot)
            return (-on_time, time)

        def insert(trip, request):
            best, best_value = None, None
            for position in range(len(trip) + 1):
                candidate = trip[:position] + [request] + trip[position:]
                value = evaluate(candidate)
                if best_value is None or value < best_value:
                    best, best_value = candidate, value
            return best

        def repair(trip, removed):
            load = sum(r.demand for r in trip)
            for request in sorted(removed, key=lambda r: r.delivery.latest):
                if load + request.demand <= capacity + 1e-9 and evaluate(insert(trip, request)) < evaluate(trip):
                    trip, load = insert(trip, request), load + request.demand
            return trip

        fixed = list(on_board)
        current = repair(fixed, pool)
        best, best_value = current, evaluate(current)
        for _ in range(self.parameters.iterations):
            added = [r for r in current if r not in on_board]
            if not added:
                break
            keep = [r for r in current if r in on_board or rng.random() > 1 / 3]
            candidate = repair(keep, [r for r in pool if r not in keep])
            value = evaluate(candidate)
            if value < best_value:
                best, best_value, current = candidate, value, candidate
        added = [r for r in best if r not in on_board]
        if not added:
            return None  # the plan departs with what is on board
        return next(a.index for a in loadable if _slot(env, a) == added[0].request_id)
