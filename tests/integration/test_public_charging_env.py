"""Public stations in the environment: Kullman's queue rules and cases A1-A19."""

import copy

import numpy as np
import pytest
from extension_harness import InvariantChecker, build, choose, rollout

from EVRoutingEnv.models.simulation import queueing
from EVRoutingEnv.models.simulation.public_station_queue import QueueEntry
from EVRoutingEnv.state.feasibility import (
    ActionKind,
    FeasibilityReason,
    joint_action_feasibility,
)
from EVRoutingEnv.state.features import CHARGER_FEATURES


pytestmark = pytest.mark.integration

KULLMAN = {"enabled": True, "exogenous_traffic": {"utilization": 0.65}}


def _single(public=None, **extra):
    env = build("seattle", trucks=1, stops=3, public_charging=public or KULLMAN, **extra)
    env.reset(seed=2)
    return env


def _nearest_feasible_charger(env):
    decisions = joint_action_feasibility(env)
    truck = env.trucks[env.active_truck_id]
    options = [
        (env.transport_graph.get_time_distance(int(truck.current_node), d.target_node), i, d.target_node)
        for i, d in enumerate(decisions[: env.num_charging_nodes]) if d.feasible
    ]
    _, index, node = min(options)
    return index, node


def _quiet(station, *, busy_until=None, up=True, queue=0, recover_at=None):
    """Script a station: no outside arrivals, ports busy or down, a queue."""
    station._next_arrival = None
    station.queue = []
    for port in station.ports:
        port.occupant, port.busy_until = None, None
        port.timeline_up = up
        port.next_change = recover_at if (not up and recover_at is not None) else float("inf")
        if busy_until is not None:
            port.occupant = QueueEntry("outside", 10_000 + port.index, -1.0, busy_until)
            port.busy_until = busy_until
    for i in range(queue):
        station.queue.append(QueueEntry("outside", 20_000 + i, -0.5 + i * 1e-3, 1.0))


def _arrive(env, action):
    env.step(action)
    truck = env.trucks[0]
    assert env.active_truck_id == 0 and int(truck.current_node) in env.public_stations.queues
    return truck


def _kind_index(env, kind):
    return env.extra_action_offset + env.extra_action_kinds.index(kind)


def _charge_indices(env):
    return range(env.num_navigation_actions, env.extra_action_offset)


def _charger_row(env, node):
    features = env.get_canonical_features()
    row = sorted(env.charging_nodes).index(node)
    return dict(zip(CHARGER_FEATURES, features.charger_features[row], strict=True))


def test_A3_all_ports_down_gives_a_queue_or_leave_epoch() -> None:
    env = _single()
    action, node = _nearest_feasible_charger(env)
    station = env.public_stations.queues[node]
    _quiet(station, up=False, recover_at=env.global_clock + 20.0)
    _arrive(env, action)
    decisions = joint_action_feasibility(env)
    assert decisions[_kind_index(env, ActionKind.QUEUE)].feasible
    charges = [decisions[i] for i in _charge_indices(env)]
    assert not any(d.feasible for d in charges)
    assert any(d.reason is FeasibilityReason.PORTS_BUSY for d in charges)
    assert any(d.feasible and d.action_kind is ActionKind.CHARGER for d in decisions)  # balk
    assert _charger_row(env, node)["observed_available_ports"] == 0.0
    recovery = min(port.next_change for port in station.ports)
    env.step(_kind_index(env, ActionKind.QUEUE))
    # FCFS resumes at the first recovery: the queued truck gets that port.
    assert env.active_truck_id == 0 and env.public_stations.committed(0)
    assert env.global_clock == pytest.approx(recovery)


def test_A4_a_port_failing_mid_session_finishes_the_session() -> None:
    env = _single()
    action, node = _nearest_feasible_charger(env)
    station = env.public_stations.queues[node]
    _quiet(station)
    truck = _arrive(env, action)
    index = next(i for i in _charge_indices(env) if joint_action_feasibility(env)[i].feasible)
    env._execute_charge_action(truck, env.charge_action_values[index - env.num_navigation_actions], node)
    port = next(p for p in station.ports if p.occupant is not None and p.occupant.kind == "fleet")
    end = port.busy_until
    port.next_change = 0.5 * (env.global_clock + end)  # fails while busy
    env._advance_to_next_decision()
    assert env.global_clock == pytest.approx(end)  # the session ran to its end
    assert any(e["event_type"] == "CHARGING_END" and e["timestamp"] == pytest.approx(end)
               for e in truck.event_history)
    assert not port.timeline_up and port.occupant is None


def test_A5_head_without_a_feasible_charge_releases_the_port() -> None:
    env = _single(charging={**copy.deepcopy(build("seattle").config["charging"]), "target_soc_levels": [0.5]})
    action, node = _nearest_feasible_charger(env)
    station = env.public_stations.queues[node]
    _quiet(station, busy_until=env.global_clock + 1.5)
    _arrive(env, action)
    env.step(_kind_index(env, ActionKind.QUEUE))
    truck = env.trucks[0]
    assert any(e["event_type"] == "PORT_RELEASED_UNUSED" for e in truck.event_history)
    assert station.reserved() == 0 and not env.public_stations.committed(0)
    decisions = joint_action_feasibility(env)
    assert any(d.feasible and d.action_kind is ActionKind.CHARGER for d in decisions)


def test_A6_a_queued_truck_waits_without_epochs_and_must_charge_at_the_head() -> None:
    env = _single()
    action, node = _nearest_feasible_charger(env)
    station = env.public_stations.queues[node]
    arrival_time_guess = env.global_clock
    _quiet(station, busy_until=arrival_time_guess + 30.0, queue=2)
    _arrive(env, action)
    queued_at = env.global_clock
    env.step(_kind_index(env, ActionKind.QUEUE))
    # No epoch in between: the next decision is the reservation itself.
    assert env.public_stations.committed(0)
    assert env.global_clock > queued_at
    decisions = joint_action_feasibility(env)
    navigation = decisions[: env.num_navigation_actions]
    assert all(not d.feasible for d in navigation)
    assert any(d.reason is FeasibilityReason.COMMITTED_TO_CHARGE for d in navigation)
    assert any(decisions[i].feasible for i in _charge_indices(env))
    assert not decisions[_kind_index(env, ActionKind.QUEUE)].feasible


def test_A7_legacy_head_that_leaves_passes_the_port_at_once() -> None:
    env = _single({**KULLMAN, "queue_policy": "legacy"})
    action, node = _nearest_feasible_charger(env)
    station = env.public_stations.queues[node]
    _quiet(station, busy_until=env.global_clock + 3.0)
    env.step(action)  # joins the queue automatically
    station.queue.append(QueueEntry("outside", 30_000, env.global_clock + 1e-6, 2.0))
    reserved_at = None
    while reserved_at is None:
        if station.reserved_for(0):
            reserved_at = env.global_clock
            break
        env.step(env.action_space.n - 1)  # unreachable in practice
    decisions = joint_action_feasibility(env)
    leave = next(i for i, d in enumerate(decisions[: env.num_charging_nodes]) if d.feasible)
    env.step(leave)
    starts = [entry for entry in station.log.starts if entry[1] == 30_000]
    assert starts and starts[0][2] == pytest.approx(reserved_at)


def test_A8_free_port_and_empty_queue_needs_no_queue_action() -> None:
    env = _single()
    action, node = _nearest_feasible_charger(env)
    _quiet(env.public_stations.queues[node])
    _arrive(env, action)
    decisions = joint_action_feasibility(env)
    assert decisions[_kind_index(env, ActionKind.QUEUE)].reason is FeasibilityReason.QUEUE_NOT_NEEDED
    assert any(decisions[i].feasible for i in _charge_indices(env))


@pytest.mark.parametrize("policy", ["greedy", "random"])
def test_A9_vanishing_outside_traffic_matches_the_fleet_only_simulator(policy) -> None:
    public = {"enabled": True, "queue_policy": "legacy",
              "exogenous_traffic": {"utilization": 1e-12, "initial_state": "empty"}}
    for seed in range(3):
        traces = []
        for blocks in ({}, {"public_charging": public}):
            env = build("california", trucks=3, stops=8, **blocks)
            actions, clocks = [], []
            observation, info = env.reset(seed=seed)
            done = False
            pol = choose(env, policy, seed)
            while not done:
                action = pol(env, observation, info)
                observation, _, terminated, truncated, info = env.step(action)
                actions.append(action)
                clocks.append(env.global_clock)
                done = terminated or truncated
            traces.append((actions, clocks, [t.export_event_log() for t in env.trucks],
                           info["termination_reason"]))
        assert traces[0] == traces[1]


def test_A10_utilization_099_keeps_the_invariants() -> None:
    env = build("seattle", public_charging={"enabled": True, "exogenous_traffic": {"utilization": 0.99}})
    rollout(env, 0, choose(env, "random", 0), check=InvariantChecker(env))


def test_A12_station_closure_releases_queued_trucks_and_outside_vehicles() -> None:
    env = _single()
    action, node = _nearest_feasible_charger(env)
    station = env.public_stations.queues[node]
    _quiet(station, busy_until=env.global_clock + 30.0, queue=2)
    truck = _arrive(env, action)
    env._execute_extra_action(truck, ActionKind.QUEUE)  # queued, not yet advanced
    assert station.position(0) == 2
    env.set_charger_available(node, False)
    assert station.queue == [] and station.log.balked_on_closure == 2
    assert any(e.data.get("reason") == "charger_closed" for e in env.event_queue if e.truck_id == 0)
    env._advance_to_next_decision()
    assert env.active_truck_id == 0 and env.truck_states[0] == "ready"


def test_A13_horizon_with_queued_outside_vehicles_does_not_change_termination() -> None:
    env = build("seattle", public_charging={"enabled": True, "exogenous_traffic": {"utilization": 0.95}},
                max_time=6.0)
    info, _ = rollout(env, 1, choose(env, "greedy", 1))
    assert info["termination_reason"] in {"time_limit_exhausted", "success",
                                          "fleet_stopped_with_unserved_customers"}
    assert all(s.clock <= env.max_time + 1e-9 for s in env.public_stations.queues.values())


def test_A14_a_never_visited_station_shows_only_the_prior() -> None:
    env = build("seattle", public_charging=KULLMAN)
    env.reset(seed=0)
    node = sorted(env.charging_nodes)[0]
    row = _charger_row(env, node)
    params = env.public_stations.station_parameters(node)
    assert [row[c] for c in ("observed_busy_ports", "observed_queue_length",
                             "observation_age_hours", "observed_available_ports")] == [-1.0] * 4
    assert row["expected_wait_given_memory"] == pytest.approx(
        queueing.mean_wait(params.ports, params.mean_arrival_rate, params.service_rate), rel=1e-6)
    assert row["known_utilization_prior"] == pytest.approx(0.65)


@pytest.mark.parametrize(("memory", "visible"), [("shared", True), ("per_truck", False)])
def test_A15_memory_is_shared_by_the_fleet_or_kept_per_truck(memory, visible) -> None:
    env = build("seattle", trucks=2, public_charging={**KULLMAN, "memory": memory})
    env.reset(seed=0)
    node = sorted(env.charging_nodes)[3]
    station = env.public_stations.queue_at(node, env.global_clock)
    env.public_stations.observe(0, node, env.global_clock, station)
    env.active_truck_id = 1
    row = _charger_row(env, node)
    assert (row["observed_busy_ports"] >= 0) == visible


def test_A16_realtime_information_equals_the_true_state() -> None:
    env = build("seattle", public_charging={**KULLMAN, "information": "realtime_queue"})
    observation, info = env.reset(seed=3)
    policy = choose(env, "greedy", 3)
    for _ in range(8):
        for node in sorted(env.charging_nodes)[:10]:
            station = env.public_stations.queue_at(node, env.global_clock)
            row = _charger_row(env, node)
            assert row["observed_busy_ports"] == station.busy_ports()
            assert row["observed_queue_length"] == station.queue_length()
            assert row["observation_age_hours"] == 0.0
        if env._episode_ended:
            break
        observation, _, terminated, truncated, info = env.step(policy(env, observation, info))


def test_mask_never_reads_hidden_queue_state() -> None:
    env = build("seattle", trucks=2, public_charging=KULLMAN)
    env.reset(seed=4)
    before = env.mask_fn().copy()
    for station in env.public_stations.queues.values():
        _quiet(station, busy_until=50.0, queue=5)
    assert np.array_equal(before, env.mask_fn())


EQUIVALENCE = [
    ("kullman", KULLMAN),
    ("legacy", {**KULLMAN, "queue_policy": "legacy"}),
    ("outages_realtime", {**KULLMAN, "information": "realtime_queue",
                          "exogenous_traffic": {"utilization": 0.9}, "outages": {"enabled": True}}),
]


def _equivalence(name, public, seed, policy):
    traces = []
    for eager in (False, True):
        env = build("seattle", trucks=2, stops=4, max_time=24.0,
                    public_charging={**public, "debug_eager_public_queues": eager})
        actions, clocks = [], []
        observation, info = env.reset(seed=seed)
        pol = choose(env, policy, seed)
        done = False
        while not done:
            action = pol(env, observation, info)
            observation, _, terminated, truncated, info = env.step(action)
            actions.append(action)
            clocks.append(env.global_clock)
            done = terminated or truncated
        traces.append((actions, clocks, [t.export_event_log() for t in env.trucks]))
    assert traces[0] == traces[1], (name, seed, policy)


@pytest.mark.parametrize(("name", "public"), EQUIVALENCE)
@pytest.mark.parametrize("policy", ["greedy", "random"])
def test_lazy_queues_equal_the_eager_reference(name, public, policy) -> None:
    for seed in range(3):
        _equivalence(name, public, seed, policy)


@pytest.mark.statistical
@pytest.mark.parametrize(("name", "public"), EQUIVALENCE)
@pytest.mark.parametrize("policy", ["greedy", "random"])
def test_lazy_queues_equal_the_eager_reference_fifty_seeds(name, public, policy) -> None:
    for seed in range(50):
        _equivalence(name, public, seed, policy)


def test_iid_stationary_waits_draw_once_per_visit() -> None:
    env = build("seattle", trucks=2, public_charging={
        "enabled": True, "queue_model": "iid_stationary_wait", "queue_policy": "legacy",
        "exogenous_traffic": {"utilization": 0.9}})
    rollout(env, 2, choose(env, "random", 2), check=InvariantChecker(env))
    draws = env.public_stations.drawn_waits
    visits = sum(env.public_stations.visits.values())
    assert len(draws) == visits


# --- regressions from the local code review ------------------------------------


def _facade(trucks=4):
    env = build("seattle", trucks=trucks, stops=4, public_charging=KULLMAN)
    env.reset(seed=0)
    node = sorted(env.public_stations.queues)[0]
    station = env.public_stations.queues[node]
    _quiet(station)
    station.ports = station.ports[:2]
    return env, env.public_stations, node, station


def _pushes(facade, clock):
    pushed = []
    facade.drain(clock, lambda truck, time, data: pushed.append((truck, time, data["token"])),
                 lambda *args, **kwargs: None)
    return pushed


def test_two_ports_freeing_together_wake_both_queued_trucks() -> None:
    env, facade, node, station = _facade()
    now = env.global_clock
    facade.start_charging(0, node, 1.0, now)
    facade.start_charging(1, node, 1.0, now)
    facade.join(2, node, now)
    facade.join(3, node, now)
    first = _pushes(facade, now)
    assert [(t, time) for t, time, _ in first] == [(2, pytest.approx(now + 1.0))]
    facade.queue_at(node, now + 1.0)  # truck 2's wake-up fires
    second = _pushes(facade, now + 1.0)
    assert [(t, time) for t, time, _ in second] == [(3, pytest.approx(now + 1.0))]
    assert station.reserved_for(2) and station.reserved_for(3)


def test_a_released_reservation_passes_to_the_next_queued_truck() -> None:
    env, facade, node, station = _facade()
    station.ports = station.ports[:1]
    now = env.global_clock
    facade.start_charging(0, node, 1.0, now)
    facade.join(1, node, now)
    facade.join(2, node, now)
    _pushes(facade, now)
    facade.queue_at(node, now + 1.0)
    assert station.reserved_for(1)
    facade.release_reservation(1, node, now + 1.0)  # case A5
    woken = _pushes(facade, now + 1.0)
    assert [(t, time) for t, time, _ in woken] == [(2, pytest.approx(now + 1.0))]
    assert station.reserved_for(2)


def test_closure_invalidates_pending_wake_ups() -> None:
    env, facade, node, station = _facade()
    station.ports = station.ports[:1]
    now = env.global_clock
    facade.start_charging(0, node, 2.0, now)
    facade.join(1, node, now)
    [(truck, time, token)] = _pushes(facade, now)
    facade.set_station_available(node, False, clock=now + 0.5)
    assert facade.confirm_reservation(1, node, token, time) is False


def test_closure_drops_a_reservation_not_yet_scheduled() -> None:
    env, facade, node, station = _facade()
    station.ports = station.ports[:1]
    now = env.global_clock
    facade.start_charging(0, node, 1.0, now)
    facade.join(1, node, now)
    # No drain in between: truck 1's port is reserved but its wake-up is not
    # yet scheduled when the station closes and sends it away.
    facade.queue_at(node, now + 1.0)
    assert station.reserved_for(1)
    assert 1 in facade.set_station_available(node, False, clock=now + 1.0)
    assert _pushes(facade, now + 1.0) == []
    assert 1 not in facade.scheduled


def test_queries_keep_the_fleet_first_tie_rule() -> None:
    env = _single()
    action, node = _nearest_feasible_charger(env)
    station = env.public_stations.queues[node]
    _quiet(station, busy_until=env.global_clock + 10.0)
    truck = _arrive(env, action)
    station._next_arrival = (env.global_clock, station._arrivals_k)  # an outside car, same instant
    joint_action_feasibility(env)  # mask query
    env.get_canonical_features()  # feature query
    assert station.queue == []
    env._execute_extra_action(truck, ActionKind.QUEUE)
    station.advance(env.global_clock)
    assert [entry.kind for entry in station.queue] == ["fleet", "outside"]


def test_closing_the_private_depot_charger_works() -> None:
    from extension_harness import SAME_DAY

    env = build("seattle", trucks=1, requests=SAME_DAY, public_charging=KULLMAN,
                depot_charging={"enabled": True})
    env.reset(seed=0)
    depot = env.request_engine.depot
    env.set_charger_available(depot, False)
    assert env.charging_station.station_available[depot] is False


def test_closure_logs_the_departure_of_a_released_reservation() -> None:
    env, facade, node, station = _facade()
    station.ports = station.ports[:1]
    now = env.global_clock
    facade.join(7, node, now)  # free port: reserved at once
    assert station.reserved_for(7)
    station.close(now)
    assert ("fleet", 7, now, 0) in station.log.departures
