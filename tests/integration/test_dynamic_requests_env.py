"""Dynamic requests in the environment: same-day freight, drayage, cases C1-C22."""

import copy
import dataclasses
import itertools

import pytest
from extension_harness import SAME_DAY, InvariantChecker, build, choose, rollout

from EVRoutingEnv.models.core.requests import RequestStatus
from EVRoutingEnv.models.environment.request_engine import (
    hindsight_problem,
    kancharla_power_ratio,
)
from EVRoutingEnv.state.feasibility import (
    ActionKind,
    FeasibilityReason,
    _leg_energy_check,
    joint_action_feasibility,
)


pytestmark = pytest.mark.integration

INITIAL = {**SAME_DAY, "initial_requests": 3}
TERMINAL = {"name": "Port of Seattle, Terminal 18", "lat": 47.578, "lon": -122.353}
DRAYAGE = {**SAME_DAY, "mode": "drayage", "initial_requests": 3}


def _env(requests=INITIAL, trucks=1, seed=0, **blocks):
    env = build("seattle", trucks=trucks, requests=requests, **blocks)
    env.reset(seed=seed)
    return env


def _slot(env, request_id):
    return env.num_charging_nodes + request_id


def _depot(env):
    return env.num_charging_nodes + env.fixed_num_stops


def _kind(env, kind):
    return env.extra_action_offset + env.extra_action_kinds.index(kind)


def _released_open(env):
    return [r for r in env.request_engine.registry.revealed()
            if r.status is RequestStatus.OPEN and r.release_time <= env.global_clock + 1e-9]


def _load_one(env):
    request = _released_open(env)[0]
    env.step(_slot(env, request.request_id))
    assert request.status is RequestStatus.ON_BOARD and request.owner == 0
    return request


def _nearest_charger(env):
    decisions = joint_action_feasibility(env)
    node = int(env.trucks[env.active_truck_id].current_node)
    return min(
        (env.transport_graph.get_time_distance(node, d.target_node), i)
        for i, d in enumerate(decisions[: env.num_charging_nodes]) if d.feasible
    )[1]


def test_C1_requests_released_at_a_ready_instant_are_visible() -> None:
    env = _env()
    assert env.global_clock == 0.0 and env.active_truck_id == 0
    revealed = env.request_engine.registry.revealed()
    assert revealed and all(r.reveal_time <= 0.0 for r in revealed)
    decisions = joint_action_feasibility(env)
    assert any(decisions[_slot(env, r.request_id)].feasible for r in revealed)


def test_C2_two_live_requests_at_one_node_have_distinct_slots() -> None:
    requests = {**SAME_DAY, "static_context_size": 4, "max_requests_per_epoch": 6, "initial_requests": 8}
    for seed in range(20):
        env = _env(requests, seed=seed)
        by_node = {}
        for request in env.request_engine.registry.revealed():
            by_node.setdefault(int(request.delivery.node), []).append(request)
        shared = [group for group in by_node.values() if len(group) > 1]
        if not shared:
            continue
        first, second = shared[0][:2]
        assert first.request_id != second.request_id
        assert [r.request_id for r in env.request_engine.registry.live_at_node(first.delivery.node)][:2] == [
            first.request_id, second.request_id]
        with pytest.raises(RuntimeError, match="use request ids"):
            env.task_registry.task_for_node(first.delivery.node)
        rows = env.get_canonical_features().customer_features
        assert rows[first.request_id, 0] == first.request_id and rows[second.request_id, 0] == second.request_id
        return
    pytest.fail("no shared node in 20 seeds")


def test_C3_a_request_lost_en_route_releases_the_claim_on_arrival() -> None:
    env = _env()
    env.step(_nearest_charger(env))  # away from the depot, nothing on board
    truck = env.trucks[0]
    request = next(r for r in env.request_engine.registry.revealed() if r.status is RequestStatus.OPEN)
    env._execute_request_slot(truck, request.request_id)
    assert request.status is RequestStatus.CLAIMED_PICKUP
    expiry = env.global_clock + 1e-3
    request.pickup = dataclasses.replace(request.pickup, latest=expiry)
    env._push_system_event(expiry, "request_expire", request_id=request.request_id)
    env._advance_to_next_decision()
    assert request.status is RequestStatus.LOST and request.owner is None
    assert env.active_truck_id == 0 and env.truck_states[0] == "ready"
    assert any(e["event_type"] == "REQUEST_LOST_ON_ARRIVAL" for e in truck.event_history)


def test_C4_a_second_load_over_capacity_is_masked() -> None:
    env = _env()
    first, second = _released_open(env)[:2]
    first.demand, second.demand = 15.0, 10.0
    env.step(_slot(env, first.request_id))
    decision = joint_action_feasibility(env)[_slot(env, second.request_id)]
    assert decision.reason is FeasibilityReason.PAYLOAD_EXCEEDED


@pytest.mark.parametrize("deadline", ["hard", "soft"])
def test_C5_C6_late_deliveries(deadline) -> None:
    env = _env({**INITIAL, "deadline_type": deadline})
    request = _load_one(env)
    request.delivery = dataclasses.replace(request.delivery, earliest=0.0, latest=env.global_clock)
    _, reward, *_ = env.step(_slot(env, request.request_id))
    assert request.status is RequestStatus.SERVED_LATE and request.delay_hours > 0.0
    metrics = env._get_info()["request_metrics"]
    assert metrics["requests_served_late"] == 1 and metrics["requests_served_on_time"] == 0
    assert not env.trucks[0].failed  # the truck carries on
    if deadline == "soft":
        assert reward <= -request.delay_hours + 1e-9


def test_C7_static_mode_keeps_the_legacy_late_arrival_failure() -> None:
    from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
    from EVRoutingEnv.utils.utils import load_config

    config = copy.deepcopy(load_config("EVRoutingEnv/config_files/config_joint_time_windows.yaml"))
    config["environment"]["num_trucks"] = 1
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    env.reset(seed=0)
    decisions = joint_action_feasibility(env)
    index = next(i for i, d in enumerate(decisions) if d.feasible and d.action_kind is ActionKind.CUSTOMER)
    task = env.task_registry.task_for_node(decisions[index].target_node)
    truck = env.trucks[env.active_truck_id]
    env._execute_navigation_action(truck, task.node_id)
    task.latest_service = env.global_clock
    task.earliest_service = 0.0
    env._advance_to_next_decision()
    assert truck.failed and truck.failure_reason == "time_window_violation_after_realization"


def test_C8_wait_is_infeasible_when_no_release_can_follow() -> None:
    env = _env()
    assert joint_action_feasibility(env)[_kind(env, ActionKind.WAIT)].feasible
    env.request_engine._known_schedule = []
    for request in env.request_engine.registry.revealed():
        request.release_time = 0.0
    assert joint_action_feasibility(env)[_kind(env, ActionKind.WAIT)].reason is FeasibilityReason.NO_FUTURE_RELEASE


def test_C9_wait_wakes_at_a_release_just_ahead_without_looping() -> None:
    env = _env()
    now = env.global_clock
    request = env.request_engine.registry.revealed()[0]
    request.release_time = now + 1e-12
    env.request_engine._information_times = sorted({*env.request_engine._information_times, now + 1e-12})
    env.step(_kind(env, ActionKind.WAIT))
    assert env.active_truck_id == 0
    assert now < env.global_clock <= now + 1e-12 + 1e-15


def test_C10_end_shift_with_cargo_is_infeasible() -> None:
    env = _env()
    _load_one(env)
    assert joint_action_feasibility(env)[_kind(env, ActionKind.END_SHIFT)].reason is \
        FeasibilityReason.REQUESTS_ON_BOARD


@pytest.mark.parametrize("preemptive", [False, True])
def test_C11_depot_returns_with_cargo(preemptive) -> None:
    env = _env({**INITIAL, "allow_preemptive_returns": preemptive})
    carried = _load_one(env)
    env.step(_nearest_charger(env))
    decision = joint_action_feasibility(env)[_depot(env)]
    if not preemptive:
        assert decision.reason is FeasibilityReason.REQUESTS_ON_BOARD
        return
    assert decision.feasible
    env.step(_depot(env))
    assert int(env.trucks[0].current_node) == env.request_engine.depot
    assert carried.status is RequestStatus.ON_BOARD  # kept
    others = [r for r in _released_open(env)]
    if others:
        assert joint_action_feasibility(env)[_slot(env, others[0].request_id)].feasible


def test_C12_requests_after_the_fleet_retired_are_lost() -> None:
    env = _env(SAME_DAY, trucks=2)
    while not env._episode_ended:
        env.step(_kind(env, ActionKind.END_SHIFT))
    assert env.termination_reason == "requests_lost"
    assert env.global_clock < env.max_time
    assert all(r.status is RequestStatus.LOST for r in env.request_engine.registry.requests())


def test_C13_no_feasible_requests_at_all() -> None:
    impossible = {**SAME_DAY, "time_windows": {"enabled": True, "earliest_min": 0.0, "earliest_max": 0.0,
                                               "window_width_min": 0.01, "window_width_max": 0.01}}
    env = _env(impossible, trucks=2)
    assert len(env.request_engine.registry) == 0
    info, _ = rollout(env, 0, choose(env, "greedy", 0), check=InvariantChecker(env))
    assert info["termination_reason"] == "success"


def test_C14_overflowing_the_envelope_raises_at_reset() -> None:
    env = build("seattle", trucks=1, requests={**SAME_DAY, "max_requests": 2})
    with pytest.raises(ValueError, match="envelope"):
        env.reset(seed=0)


def _drayage(seed=0, **drayage):
    return _env(DRAYAGE, seed=seed, drayage={"terminals": [TERMINAL], **drayage})


def _load_container(env):
    for request in env.request_engine.registry.revealed():
        if request.status is RequestStatus.OPEN and joint_action_feasibility(env)[_slot(env, request.request_id)].feasible:
            request.pickup = dataclasses.replace(request.pickup, earliest=0.0)
            env.step(_slot(env, request.request_id))
            assert env.request_engine.container(0) is request
            return request
    pytest.fail("no reachable container")


def test_C17_the_first_stop_after_the_depot_is_an_origin() -> None:
    env = _drayage()
    decisions = joint_action_feasibility(env)
    assert all(d.reason is FeasibilityReason.DRAYAGE_ORIGIN_FIRST for d in decisions[: env.num_charging_nodes]
               if not d.feasible and d.reason is not FeasibilityReason.INSUFFICIENT_ENERGY)
    assert not any(d.feasible for d in decisions[: env.num_charging_nodes])
    assert any(decisions[_slot(env, r.request_id)].feasible for r in env.request_engine.registry.revealed())


def test_C15_C16_a_loaded_truck_goes_only_to_stations_or_its_destination() -> None:
    env = _drayage()
    container = _load_container(env)
    decisions = joint_action_feasibility(env)
    for request in env.request_engine.registry.revealed():
        decision = decisions[_slot(env, request.request_id)]
        if request is container:
            assert decision.feasible or decision.reason is FeasibilityReason.INSUFFICIENT_ENERGY
        elif request.status is RequestStatus.OPEN:
            assert decision.reason is FeasibilityReason.CONTAINER_ON_BOARD
    assert decisions[_depot(env)].reason is FeasibilityReason.CONTAINER_ON_BOARD
    assert any(d.feasible for d in decisions[: env.num_charging_nodes])


def test_C18_early_origin_waits_and_late_destination_is_served_late() -> None:
    env = _drayage()
    request = next(r for r in env.request_engine.registry.revealed()
                   if joint_action_feasibility(env)[_slot(env, r.request_id)].feasible)
    opening = env.global_clock + 5.0
    request.pickup = dataclasses.replace(request.pickup, earliest=opening)
    env.step(_slot(env, request.request_id))
    loading = next(e for e in env.trucks[0].event_history if e["event_type"] == "LOADING_START")
    assert loading["timestamp"] == pytest.approx(opening)
    request.delivery = dataclasses.replace(request.delivery, latest=env.global_clock)
    env.step(_slot(env, request.request_id))
    assert request.status is RequestStatus.SERVED_LATE and request.delay_hours > 0.0


def test_C19_load_dependent_energy_scales_only_loaded_legs() -> None:
    plain = _drayage()
    kancharla = _drayage(energy_model="kancharla_ramadurai_2020")
    truck_plain, truck_k = plain.trucks[0], kancharla.trucks[0]
    target = next(r.pickup.node for r in kancharla.request_engine.registry.revealed())
    empty_plain = _leg_energy_check(plain, truck_plain, target, ActionKind.CUSTOMER).required_energy
    empty_k = _leg_energy_check(kancharla, truck_k, target, ActionKind.CUSTOMER).required_energy
    assert empty_k == pytest.approx(empty_plain)
    container = _load_container(kancharla)
    destination = int(container.delivery.node)
    ratio = kancharla_power_ratio(container.extra["container_kg"], kancharla.request_engine.base_speed_kmh)
    loaded = _leg_energy_check(kancharla, truck_k, destination, ActionKind.CUSTOMER).required_energy
    nominal = kancharla.transport_graph.get_path_energy(int(truck_k.current_node), destination) * max(
        1.0, kancharla.traffic_config["max_energy_multiplier"])
    assert ratio > 1.0 and loaded == pytest.approx(nominal * ratio)


def test_C19_a_loaded_leg_to_a_charger_is_scaled_too() -> None:
    from EVRoutingEnv.baselines.optimization.perfect_information import _clone

    env = _drayage(energy_model="kancharla_ramadurai_2020")
    container = _load_container(env)
    ratio = kancharla_power_ratio(container.extra["container_kg"], env.request_engine.base_speed_kmh)
    empty = _clone(env)
    empty.request_engine.energy_scale = lambda truck_id: 1.0  # the same truck, priced empty
    index = _nearest_charger(env)
    node = env.charging_nodes[index]
    needed = _leg_energy_check(empty, empty.trucks[0], node, ActionKind.CHARGER).required_energy
    # Between the empty and the loaded requirement: only the empty truck may go.
    for state in (env, empty):
        state.trucks[0].current_battery = needed * (1.0 + ratio) / 2.0
    assert joint_action_feasibility(empty)[index].feasible
    assert joint_action_feasibility(env)[index].reason is FeasibilityReason.INSUFFICIENT_ENERGY
    # Executed: the loaded truck draws ratio times the energy of the empty one.
    for state in (env, empty):
        state.trucks[0].current_battery = state.trucks[0].battery_capacity
        state.step(index)
    drawn = [s.trucks[0].battery_capacity - s.trucks[0].current_battery for s in (env, empty)]
    assert ratio > 1.0 and drawn[1] > 0.0 and drawn[0] == pytest.approx(ratio * drawn[1])


def test_C20_a_terminal_off_the_map_is_rejected_at_init() -> None:
    with pytest.raises(ValueError, match="km from the nearest"):
        build("seattle", requests=DRAYAGE, drayage={"terminals": [{"name": "Nowhere", "lat": 0.0, "lon": 0.0}]})


TACOMA = {"name": "Port of Tacoma, Husky Terminal", "lat": 47.273, "lon": -122.41}


def test_drayage_customers_pair_only_with_terminals_within_the_radius() -> None:
    radius = 0.5
    env = build("seattle", trucks=1, requests={**DRAYAGE, "static_context_size": 10},
                drayage={"terminals": [TERMINAL, TACOMA], "service_radius_hours": radius})
    env.reset(seed=0)
    hours = env.transport_graph.get_time_distance
    requests = env.request_engine.registry.requests()
    assert requests
    for r in requests:
        importing = r.kind.value == "drayage_import"
        terminal, customer = (r.pickup.node, r.delivery.node) if importing else (r.delivery.node, r.pickup.node)
        assert terminal in env.request_engine.terminals
        assert max(hours(terminal, customer), hours(customer, terminal)) <= radius + 1e-9


def test_a_radius_too_small_for_the_static_context_is_rejected_at_init() -> None:
    # 0.3 h around the two ports leaves 11 nodes: a context of 10 plus the depot.
    blocks = {"drayage": {"terminals": [TERMINAL, TACOMA], "service_radius_hours": 0.3}}
    build("seattle", trucks=1, requests={**DRAYAGE, "static_context_size": 10}, **blocks)
    with pytest.raises(ValueError, match="static context"):
        build("seattle", trucks=1, requests={**DRAYAGE, "static_context_size": 11}, **blocks)


def test_hidden_requests_do_not_count_as_unserved() -> None:
    env = _env()
    registry = env.request_engine.registry
    assert any(r.status is RequestStatus.HIDDEN for r in registry.requests())
    for request in registry.revealed():
        request.status = RequestStatus.SERVED
    assert env.task_registry.all_served()


def test_metric_snapshots_count_elapsed_loading_and_window_waits() -> None:
    from EVRoutingEnv.evaluation.metrics import extract_operational_metrics

    seen = set()
    for seed in (1, 2):
        env = _env(trucks=2, seed=seed)
        slots = range(env.num_charging_nodes, env.num_charging_nodes + env.fixed_num_stops)
        for _ in range(8):
            decisions = joint_action_feasibility(env)
            env.step(next((i for i in slots if decisions[i].feasible),
                          next(i for i, d in enumerate(decisions) if d.feasible)))
            if env._episode_ended:
                break
            clock, service, window = env.global_clock, 0.0, 0.0
            for truck in env.trucks:
                state = env.truck_states[truck.truck_id]
                last = {e["event_type"]: e["timestamp"] for e in truck.event_history}
                if state == "loading":
                    elapsed = clock - last["LOADING_START"]
                    service += elapsed
                elif state == "unloading":
                    service += clock - truck.unloading_start_time
                elif state == "waiting_for_service":
                    elapsed = clock - last["ROUTING_END"]
                    window += elapsed
                if state in {"loading", "waiting_for_service"} and elapsed > 0.0:
                    seen.add(state)
            metrics = extract_operational_metrics(env)
            assert metrics.total_service_time == pytest.approx(
                sum(t.total_unloading_time for t in env.trucks) + service)
            assert metrics.total_time_window_waiting == pytest.approx(
                sum(t.time_window_waiting_time for t in env.trucks) + window)
    assert seen == {"loading", "waiting_for_service"}


def test_day_ahead_planners_refuse_unrevealed_requests() -> None:
    from EVRoutingEnv.baselines.heuristics.alns import solve_alns_plan
    from EVRoutingEnv.baselines.heuristics.canonical_baselines import (
        RollingHorizonMPCPolicy,
    )
    from EVRoutingEnv.baselines.optimization.branch_and_price.export import (
        export_nominal_instance,
    )
    from EVRoutingEnv.baselines.optimization.exact_optimization import (
        solve_nominal_plan,
    )

    env = _env()
    for call in (lambda: solve_alns_plan(env), lambda: solve_nominal_plan(env),
                 lambda: export_nominal_instance(env), lambda: RollingHorizonMPCPolicy()(env)):
        with pytest.raises(NotImplementedError, match="requests.mode=same_day"):
            call()


@pytest.mark.parametrize("policy", ["greedy_dispatch", "wait_at_depot", "rolling_alns"])
def test_dynamic_baselines_run_clean(policy) -> None:
    from EVRoutingEnv.baselines.heuristics import component_baselines as cb

    make = {"greedy_dispatch": cb.GreedyDispatchPolicy, "wait_at_depot": cb.WaitAtDepotPolicy,
            "rolling_alns": cb.RollingHorizonALNSPolicy}[policy]
    for mode, extra in (("same_day", {}), ("drayage", {"drayage": {"terminals": [TERMINAL]}})):
        env = build("seattle", trucks=2, requests={**SAME_DAY, "mode": mode}, **extra)
        info, _ = rollout(env, 1, make(), check=InvariantChecker(env))
        assert info["request_metrics"]["requests_served_on_time"] > 0


def test_static_instance_with_depot_loading() -> None:
    from EVRoutingEnv.baselines.heuristics.component_baselines import (
        GreedyDispatchPolicy,
    )

    env = build("california", trucks=2, stops=6, problem={"loading": "at_depot"},
                depot_charging={"enabled": False})
    info, _ = rollout(env, 0, GreedyDispatchPolicy(), check=InvariantChecker(env))
    assert len(env.request_engine.registry) == 6
    assert info["request_metrics"]["requests_served_on_time"] >= 1


def test_hindsight_problem_holds_every_request_with_its_times() -> None:
    env = _env(SAME_DAY, trucks=2)
    problem = hindsight_problem(env)
    registry = env.request_engine.registry
    assert len(problem["requests"]) == len(registry)
    assert any(r.status is RequestStatus.HIDDEN for r in registry.requests())  # includes the future
    for row, request in zip(problem["requests"], registry.requests(), strict=True):
        assert row["release_time"] == request.release_time and row["reveal_time"] == request.reveal_time
        assert row["delivery"]["latest"] == request.delivery.latest
    assert problem["scenario"] == env.scenario_random_streams.metadata()


# --- Li, Archetti & Ljubic (2024) formulation (1) as an oracle -----------------


def _li_oracle(env) -> int:
    """Enumerate trips with release dates for one truck (deliveries by T_E)."""
    engine = env.request_engine
    depot, horizon = engine.depot, env.max_time
    capacity = env.trucks[0].payload_capacity
    requests = engine.registry.requests()

    def hours(a, b):
        return 0.0 if a == b else env.transport_graph.get_time_distance(a, b)

    best = 0
    for size in range(len(requests), 0, -1):
        if size <= best:
            break
        for subset in itertools.permutations(requests, size):
            for cuts in itertools.product((False, True), repeat=size - 1):
                trips, trip = [], [subset[0]]
                for request, cut in zip(subset[1:], cuts, strict=True):
                    if cut:
                        trips.append(trip)
                        trip = []
                    trip.append(request)
                trips.append(trip)
                time, served = 0.0, 0
                for trip in trips:
                    if sum(r.demand for r in trip) > capacity + 1e-9:
                        served = -1
                        break
                    time = max(time, max(r.release_time for r in trip))
                    node = depot
                    for request in trip:
                        time += hours(node, int(request.delivery.node))
                        time += request.delivery.service_hours
                        served += time <= horizon + 1e-9
                        node = int(request.delivery.node)
                    time += hours(node, depot)
                best = max(best, served)
    return best


def test_li_2022_orienteering_with_release_dates_matches_the_hindsight_search() -> None:
    from EVRoutingEnv.baselines.optimization.perfect_information import (
        solve_request_hindsight,
    )

    checked = 0
    for seed in range(40):
        config_blocks = dict(
            requests={**SAME_DAY, "arrival_process": "poisson", "poisson_rate_per_hour": 0.8,
                      "request_cutoff_hours": 4.0, "dispatch_margin_hours": 0.0,
                      "loading_service_hours": 0.0, "time_windows": {"enabled": False},
                      "static_context_size": 30, "max_requests_per_epoch": 1},
            depot_charging={"enabled": False},
        )
        env = build("seattle", trucks=1, max_time=6.0, **config_blocks)
        env.config["truck"]["battery_capacity"] = 1e6
        env.config["traffic"].update(enable_traffic=False, enable_energy_uncertainty=False)
        env.config["delivery"]["enable_stochastic_unloading"] = False
        rebuilt = type(env)(copy.deepcopy(env.config), verbose=False, enable_plotting=False)
        rebuilt.reset(seed=seed)
        count = len(rebuilt.request_engine.registry)
        if not 2 <= count <= 6:
            continue
        oracle = _li_oracle(rebuilt)
        solution = solve_request_hindsight(
            rebuilt, routing_actions=False, canonical_loading=True, wait_only_at_depot=True,
        )
        assert solution.proven_optimal
        assert solution.served_on_time == oracle, (seed, count)
        checked += 1
        if checked == 6:
            return
    assert checked >= 3


def test_waiting_only_at_the_depot_is_refused_with_time_windows() -> None:
    from EVRoutingEnv.baselines.optimization.perfect_information import (
        solve_request_hindsight,
    )

    env = _env(SAME_DAY)  # windows enabled
    with pytest.raises(ValueError, match="time windows"):
        solve_request_hindsight(env, wait_only_at_depot=True)


def test_A17_depot_charging_with_fewer_ports_than_trucks_is_fleet_fcfs() -> None:
    env = build("seattle", trucks=3, requests=INITIAL, depot_charging={"enabled": True, "ports": 1})
    env.config["truck"]["initial_battery"] = 50.0
    env.reset(seed=0)
    depot = env.request_engine.depot
    assert env.charging_station.private_node == depot
    assert depot not in env.public_stations.queues if env.public_stations is not None else True
    full = env.num_navigation_actions + env.charge_action_values.index(max(env.charge_action_values))
    while not env._episode_ended and sum(
        any(e["event_type"] == "CHARGING_END" for e in t.event_history) for t in env.trucks
    ) < 3:
        truck = env.trucks[env.active_truck_id]
        assert len(env.charging_station.charger_occupancy[depot]) <= 1
        if not any(e["event_type"] == "CHARGING_START" for e in truck.event_history) and \
                joint_action_feasibility(env)[full].feasible:
            env.step(full)
        else:
            env.step(_kind(env, ActionKind.END_SHIFT))
    sessions = sorted(
        (next(e["timestamp"] for e in t.event_history if e["event_type"] == "CHARGING_START"),
         next(e["timestamp"] for e in t.event_history if e["event_type"] == "CHARGING_END"))
        for t in env.trucks
    )
    for (_, end), (start, _) in zip(sessions, sessions[1:]):
        assert start == pytest.approx(end)  # one port, first come first served


def test_perfect_information_mask_uses_realized_scaled_energy_and_time() -> None:
    env = _drayage(energy_model="kancharla_ramadurai_2020")
    container = _load_container(env)
    truck = env.trucks[0]
    destination = int(container.delivery.node)
    env._perfect_information = True
    from EVRoutingEnv.state.feasibility import realized_route_costs

    realized = realized_route_costs(env, truck, destination)
    required = _leg_energy_check(env, truck, destination, ActionKind.CUSTOMER).required_energy
    assert required == pytest.approx(realized["energy_kwh"] * env.request_engine.energy_scale(0))


def test_perfect_information_pickup_deadline_uses_the_realized_leg() -> None:
    from EVRoutingEnv.state.feasibility import (
        evaluate_request_slot,
        realized_route_costs,
    )

    env = _env()
    env.step(_nearest_charger(env))
    truck = env.trucks[0]
    request = next(r for r in env.request_engine.registry.revealed() if r.status is RequestStatus.OPEN)
    depot = int(request.pickup.node)
    nominal = env.transport_graph.get_time_distance(int(truck.current_node), depot)
    env._perfect_information = True
    realized = realized_route_costs(env, truck, depot)["travel_time_hours"]
    midpoint = env.global_clock + 0.5 * (nominal + realized)
    request.pickup = dataclasses.replace(request.pickup, latest=midpoint)
    feasible = evaluate_request_slot(env, truck, request.request_id).feasible
    assert feasible == (env.global_clock + realized <= midpoint)


def test_C23_the_last_hard_pickup_still_meets_the_window_nominally() -> None:
    """Loading takes time: a load at the latest pickup still delivers on time."""
    for seed in range(5):
        env = _env(SAME_DAY, seed=seed)
        depot = int(env.joint_instance.depot_node)
        requests = env.request_engine.registry.requests()
        assert requests
        for r in requests:
            drive = env.transport_graph.get_time_distance(depot, int(r.delivery.node))
            assert r.pickup.service_hours > 0.0
            assert r.pickup.latest + r.pickup.service_hours + drive <= r.delivery.latest + 1e-9


def test_request_legs_report_the_nominal_distance() -> None:
    """Traffic delays a leg; it does not lengthen the road (as in static mode)."""
    env = build("seattle", trucks=2, requests=SAME_DAY)
    rollout(env, 1, choose(env, "greedy", 1))
    legs = [e for t in env.trucks for e in t.event_history if e["event_type"] == "ROUTING_END"]
    assert legs
    for event in legs:
        origin, destination = int(event["details"]["origin"]), int(event["location"])
        nominal = env.transport_graph.get_time_distance(origin, destination)
        assert event["details"]["distance_km"] == pytest.approx(nominal * env.trucks[0].base_speed)
