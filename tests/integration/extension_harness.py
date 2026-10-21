"""Shared builders, rollouts and invariant checks for the component tests (plan 8.5)."""

from __future__ import annotations

import copy
import math
import os

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.baselines.heuristics.canonical_baselines import (
    GreedyHeuristicPolicy,
    RandomFeasiblePolicy,
)
from EVRoutingEnv.models.core.hos_audit import (
    AuditSettings,
    hos_audit,
    intervals_from_event_log,
)
from EVRoutingEnv.models.core.requests import RequestStatus
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.feasibility import FeasibilityReason
from EVRoutingEnv.utils.utils import load_config


MAPS = {
    "texas_longhaul": "EVRoutingEnv/config_files/config_joint_texas_longhaul.yaml",
    "california": "EVRoutingEnv/config_files/config_joint.yaml",
    "seattle": "EVRoutingEnv/config_files/config_joint_seattle.yaml",
}

PUBLIC = {"enabled": True, "exogenous_traffic": {"utilization": 0.9, "service_preset": "keskin_truck"},
          "outages": {"enabled": True}}
HOS_US = {"enabled": True, "regime": "us_fmcsa_property"}
HOS_EU = {"enabled": True, "regime": "eu_561_2006"}
SAME_DAY = {
    "mode": "same_day", "max_requests_per_epoch": 3, "max_requests": 150,
    "static_context_size": 40,
    "time_windows": {"enabled": True, "earliest_min": 1.0, "earliest_max": 14.0,
                     "window_width_min": 4.0, "window_width_max": 8.0},
}
# Exogenous streams whose draws a fleet action must never shift.
EXOGENOUS_STREAMS = {
    "station_arrivals", "station_service", "station_initial", "port_state", "iid_wait",
    "request_static_context", "request_epoch", "request_poisson",
    "traffic", "energy", "delivery",
}


def build(map_name="seattle", *, trucks=2, stops=4, max_time=None, **blocks) -> EventDrivenTruckEnv:
    config = copy.deepcopy(load_config(MAPS[map_name]))
    config["environment"].update(num_trucks=trucks, num_stops=stops)
    if max_time is not None:
        config["environment"]["max_time"] = max_time
    if "hours_of_service" in blocks:
        config["environment"]["max_time"] = min(config["environment"]["max_time"], 96.0)
    if "requests" in blocks:
        blocks.setdefault("depot_charging", {"enabled": True})
        config["environment"]["max_time"] = max_time or 24.0
    for name, block in blocks.items():
        if name == "problem":
            config["problem"].update(block)
        elif name == "observation":
            config["observation"] = copy.deepcopy(block)
        else:
            config[name] = copy.deepcopy(block)
    return EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)


def policy_for(name, seed):
    if name == "greedy":
        from EVRoutingEnv.baselines.heuristics.component_baselines import (
            GreedyDispatchPolicy,
        )

        return None, GreedyHeuristicPolicy(), GreedyDispatchPolicy()
    return None, RandomFeasiblePolicy(seed), RandomFeasiblePolicy(seed)


def choose(env, name, seed):
    _, static_policy, dynamic_policy = policy_for(name, seed)
    return dynamic_policy if env.request_engine is not None else static_policy


def rollout(env, seed, policy, *, check=None, max_steps=5000, record=None):
    observation, info = env.reset(seed=seed)
    done, steps = False, 0
    while not done and steps < max_steps:
        if record is not None:
            record.append((env.global_clock, observation.copy(), env.mask_fn().copy()))
        action = policy(env, observation, info)
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        if check is not None:
            check(env)
    return info, steps


class InvariantChecker:
    """Invariants 1-7 and 11 of plan section 8.5, checked after every step."""

    def __init__(self, env):
        self.env = env
        self.last_clock = 0.0
        self.queued_at: dict[int, int] = {}

    def __call__(self, env) -> None:
        # 1. The clock never decreases; queued events are not in the past.
        assert env.global_clock >= self.last_clock - 1e-9
        self.last_clock = env.global_clock
        for event in env.event_queue:
            assert event.time >= env.global_clock - 1e-9 or env._episode_ended
        # 2. Battery within [0, capacity].
        for truck in env.trucks:
            assert -1e-9 <= truck.current_battery <= truck.battery_capacity + 1e-9
        # The station-hop counter (read by the feasibility engine and Kullman's
        # policy) matches the arrival log: station arrivals since the last
        # arrival anywhere else (customer, pickup or depot).
        public = set(map(int, env.charging_nodes))
        for truck in env.trucks:
            count = 0
            for event in reversed(truck.event_history):
                if event["event_type"] != "ROUTING_END":
                    continue
                if int(event["location"]) not in public:
                    break
                count += 1
            assert truck.stations_since_stop == count
        # 3. Port accounting per station; one session per truck.
        stations = env.public_stations
        if stations is not None and hasattr(stations, "queues"):
            sessions = {}
            for node, station in stations.queues.items():
                assert station.fleet_busy() + station.outside_busy() + station.reserved() <= station.up_ports()
                for port in station.ports:
                    if port.occupant is not None and port.occupant.kind == "fleet":
                        sessions.setdefault(port.occupant.key, []).append(node)
                # 4. FCFS: the queue is in arrival order.
                arrivals = [entry.arrival for entry in station.queue]
                assert arrivals == sorted(arrivals)
            assert all(len(nodes) == 1 for nodes in sessions.values())
            # The truck -> station index agrees with a full scan.
            for truck in env.trucks:
                scanned = [
                    node for node, station in stations.queues.items()
                    if station.position(truck.truck_id) is not None
                    or station.reserved_for(truck.truck_id)
                    or station.charging(truck.truck_id)
                ]
                assert stations._station_of(truck.truck_id) == (scanned[0] if scanned else None)
        # 5./6. Requests: single owner, never served twice, payload bounds.
        engine = env.request_engine
        if engine is not None:
            for truck in env.trucks:
                load = engine.registry.on_board_demand(truck.truck_id)
                assert -1e-9 <= load <= truck.payload_capacity + 1e-9
                if engine.is_drayage:
                    containers = [
                        r for r in engine.registry.requests()
                        if r.owner == truck.truck_id and r.status in {
                            RequestStatus.LOADING, RequestStatus.ON_BOARD,
                            RequestStatus.CLAIMED_DELIVERY, RequestStatus.IN_SERVICE}
                    ]
                    assert len(containers) <= 1
            for request in engine.registry.requests():
                if request.status is RequestStatus.HIDDEN:
                    assert request.reveal_time > env.global_clock - 1e-9
        # 7. Hours of service: the independent audit is clean.
        if env.hos is not None and env._episode_ended:
            assert audit_violations(env) == []
        # 11. Kullman commitment: a queued truck charges before routing.
        if stations is not None and hasattr(stations, "queues"):
            for truck in env.trucks:
                position = stations.queue_position(truck.truck_id)
                if position is not None:
                    self.queued_at[truck.truck_id] = int(truck.current_node)
                elif truck.truck_id in self.queued_at and env.truck_states[truck.truck_id] == "routing":
                    node = self.queued_at.pop(truck.truck_id)
                    charged = any(
                        event["event_type"] in {"CHARGING_START", "PORT_RELEASED_UNUSED"}
                        and event["location"] == node
                        for event in truck.event_history
                    ) or any(
                        event["event_type"] == "PORT_RELEASED_UNUSED"
                        for event in truck.event_history
                    )
                    assert charged or not stations.kullman


def audit_violations(env):
    settings = AuditSettings(
        env.hos.regime.name,
        charging_counts_as_break=env.hos.config.charging_counts_as_break,
        queue_wait_counts_as_break=env.hos.config.queue_wait_counts_as_break,
        eu_split_break=env.hos.config.eu_split_break,
        eu_extended_daily_driving=env.hos.config.eu_extended_daily_driving,
        eu_reduced_daily_rest=env.hos.config.eu_reduced_daily_rest,
    )
    violations = []
    for truck in env.trucks:
        intervals = intervals_from_event_log(
            truck.export_event_log(),
            handling_overhead_hours=env.hos.config.handling_overhead_hours,
        )
        violations.extend(hos_audit(intervals, settings))
    allowed = env.hos.unplanned_violations
    return violations if len(violations) > allowed else []


def mask_matches_execution(env, sample: int = 6, rng=None) -> None:
    """Invariant 10 on copies: feasible actions execute, infeasible ones are
    rejected with the reason code the engine reported."""
    rng = rng or np.random.default_rng(0)
    from EVRoutingEnv.state.feasibility import joint_action_feasibility

    decisions = joint_action_feasibility(env)
    indices = [i for i, d in enumerate(decisions) if d.reason is not FeasibilityReason.EMPTY_ACTION_SLOT
               and d.reason is not FeasibilityReason.REQUEST_HIDDEN]
    for index in rng.choice(indices, size=min(sample, len(indices)), replace=False):
        clone = copy.deepcopy(env)
        clone.invalid_action_mode = "terminate"
        before = clone.invalid_action_count
        clone.step(int(index))
        if decisions[index].feasible:
            assert clone.invalid_action_count == before, (index, decisions[index])
        else:
            assert clone.invalid_action_count == before + 1
            assert clone.termination_reason == f"invalid_action:{decisions[index].reason.value}"


def perturb_after(env, cutoff: float, salt: int = 99991) -> dict:
    """Resample every exogenous draw made after ``cutoff`` (non-anticipativity).

    Returns a counter of the draws actually resampled, so a test can check the
    perturbation was not vacuous.
    """
    streams = env.scenario_random_streams
    counter = {"late_draws": 0}
    originals = {
        "uniforms": streams.uniforms,
        "exponential": streams.exponential,
        "keyed_generator": streams.keyed_generator,
        "standard_normal": streams.standard_normal,
    }

    def late() -> bool:
        is_late = env.global_clock > cutoff + 1e-12
        counter["late_draws"] += is_late
        return is_late

    def salted(key):
        return ("perturbed", salt, key)

    streams.uniforms = lambda stream, key, size=1: originals["uniforms"](
        stream, salted(key) if late() else key, size)
    streams.exponential = lambda stream, key, mean: originals["exponential"](
        stream, salted(key) if late() else key, mean)
    streams.keyed_generator = lambda stream, key: originals["keyed_generator"](
        stream, salted(key) if late() else key)
    streams.standard_normal = lambda stream, key: originals["standard_normal"](
        stream, salted(key) if late() else key)
    return counter


def perturb_hidden_requests(env, cutoff: float) -> int:
    """Change every request revealed after ``cutoff`` (same ids and times)."""
    engine = env.request_engine
    if engine is None:
        return 0
    changed = 0
    customers = [int(node) for node in engine.context.customers]
    for request in engine.registry.requests():
        if request.reveal_time > cutoff + 1e-12:
            other = customers[(customers.index(int(request.delivery.node)) + 7) % len(customers)] \
                if int(request.delivery.node) in customers else customers[0]
            request.delivery = type(request.delivery)(other, request.delivery.earliest,
                                                      request.delivery.latest,
                                                      request.delivery.service_hours)
            request.demand = max(0.5, request.demand * 0.5)
            changed += 1
    return changed


def nonfinite(values) -> bool:
    return not np.isfinite(np.asarray(values)).all()


def same(a, b) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b)) and not math.isnan(float(np.nansum(a)))
