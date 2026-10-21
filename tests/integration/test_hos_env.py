"""Hours of service in the environment: cases B6-B15 and audited rollouts."""

import copy
import dataclasses

import pytest
from extension_harness import (
    SAME_DAY,
    InvariantChecker,
    audit_violations,
    build,
    choose,
    rollout,
)

from EVRoutingEnv.models.core import hours_of_service as h
from EVRoutingEnv.models.environment.hos_tracker import HosTracker
from EVRoutingEnv.state.feasibility import (
    ActionKind,
    FeasibilityReason,
    joint_action_feasibility,
)


pytestmark = pytest.mark.integration

US = {"enabled": True, "regime": "us_fmcsa_property"}
US_STOPS = {**US, "rest_locations": "stops_only"}


def _kind_index(env, kind):
    return env.extra_action_offset + env.extra_action_kinds.index(kind)


def _set_clock(env, truck_id, **fields):
    env.hos._clocks[truck_id] = dataclasses.replace(
        env.hos._clocks[truck_id], time=env.global_clock, **fields
    )


@pytest.mark.parametrize(
    ("regime", "rest_locations"),
    [("us_fmcsa_property", "anywhere"), ("eu_561_2006", "anywhere"),
     ("us_fmcsa_property", "stops_only"), ("eu_561_2006", "stops_only"),
     ("bai2023_simplified", "stops_only")],
)
@pytest.mark.parametrize("policy", ["greedy", "random"])
def test_rollouts_pass_the_independent_audit(regime, rest_locations, policy) -> None:
    env = build("seattle", trucks=2, stops=5,
                hours_of_service={"enabled": True, "regime": regime, "rest_locations": rest_locations})
    for seed in range(3):
        info, _ = rollout(env, seed, choose(env, policy, seed), check=InvariantChecker(env))
        assert audit_violations(env) == []
        assert info["hours_of_service"]["regime"] == regime


def test_certified_instances_stay_certified_under_hours_of_service() -> None:
    env = build("texas_longhaul", trucks=3, stops=6, hours_of_service=US)
    _, info = env.reset(seed=0)
    assert info["instance_feasibility"]["certified"]


def test_B6_window_expired_allows_charging_but_offers_a_daily_rest() -> None:
    env = build("seattle", trucks=1, stops=3, hours_of_service=US)
    env.reset(seed=1)
    decisions = joint_action_feasibility(env)
    charger = next(i for i, d in enumerate(decisions[: env.num_charging_nodes]) if d.feasible)
    env.step(charger)
    truck = env.trucks[0]
    _set_clock(env, 0, shift_start=env.global_clock - 14.0)
    decisions = joint_action_feasibility(env)
    navigation = decisions[: env.num_navigation_actions]
    assert not any(d.feasible for d in navigation)
    assert any(d.reason is FeasibilityReason.HOS_SHIFT_WINDOW for d in navigation)
    assert decisions[_kind_index(env, ActionKind.REST_DAILY)].feasible
    assert any(d.feasible for d in decisions[env.num_navigation_actions: env.extra_action_offset]) or (
        truck.current_battery >= truck.battery_capacity - 1e-9
    )


def test_B7_long_legs_get_their_rests_inside_and_arrive_exactly(monkeypatch) -> None:
    recorded = []
    original = HosTracker.plan_leg

    def spy(self, truck, departure, driving_hours):
        # Planning starts from the clock brought to the departure time.
        clock = h.advance_to(self.regime, self.clock(truck.truck_id), departure, h.HosActivity.IDLE)
        segments = original(self, truck, departure, driving_hours)
        recorded.append((clock, driving_hours, segments))
        return segments

    monkeypatch.setattr(HosTracker, "plan_leg", spy)
    env = build("california", trucks=2, stops=6, hours_of_service=US)
    rollout(env, 0, choose(env, "greedy", 0))
    assert any(sum(a != "drive" for a, _ in segments) >= 2 for _, _, segments in recorded)
    regime = env.hos.regime
    for clock, driving, segments in recorded:
        expected, _ = h.insert_rests(regime, clock, driving)
        assert [(a.value, hours) for a, hours in expected] == segments
    for truck in env.trucks:
        events = truck.export_event_log()["events"]
        starts = [e for e in events if e["event_type"] == "ROUTING_START"]
        ends = [e for e in events if e["event_type"] == "ROUTING_END"]
        rests = [e for e in events if e["event_type"] == "HOS_REST" and e["details"]["roadside"]]
        for start, end in zip(starts, ends, strict=False):
            inside = sum(r["details"]["end"] - r["details"]["start"] for r in rests
                         if start["timestamp"] - 1e-9 <= r["details"]["start"] <= end["timestamp"] + 1e-9)
            assert end["timestamp"] == pytest.approx(
                start["timestamp"] + end["details"]["travel_time_hours"] + inside, abs=1e-9
            )
    assert audit_violations(env) == []


def test_B8_stops_only_masks_a_leg_over_the_allowance() -> None:
    env = build("seattle", trucks=1, stops=3, hours_of_service=US_STOPS)
    env.reset(seed=0)
    _set_clock(env, 0, driving_since_break=8.0 - 0.05)
    decisions = joint_action_feasibility(env)
    navigation = [d for d in decisions[: env.num_navigation_actions] if d.target_node is not None]
    assert any(d.reason is FeasibilityReason.HOS_DRIVING_LIMIT for d in navigation)
    assert decisions[_kind_index(env, ActionKind.REST_BREAK)].feasible


def test_B8_generation_rejects_maps_whose_legs_exceed_the_allowance() -> None:
    with pytest.raises(ValueError, match="hours of"):
        env = build("california", trucks=2, stops=4,
                    hours_of_service={"enabled": True, "regime": "eu_561_2006", "rest_locations": "stops_only"})
        env.config["environment"]["generation_attempts"] = 4
        env.generation_attempts = 4
        env.reset(seed=0)


def test_B9_realized_overrun_gets_one_counted_roadside_break() -> None:
    for seed in range(20):
        env = build("seattle", trucks=1, stops=3, hours_of_service=US_STOPS)
        env.reset(seed=seed)
        decisions = joint_action_feasibility(env)
        truck = env.trucks[0]
        options = sorted(
            (env.transport_graph.get_time_distance(int(truck.current_node), d.target_node), i)
            for i, d in enumerate(decisions[: env.num_navigation_actions]) if d.feasible
        )
        nominal, action = options[-1]
        _set_clock(env, 0, driving_since_break=8.0 - nominal - 1e-6)
        env.step(action)
        if env.hos.unplanned_breaks:
            rollout_tail = choose(env, "greedy", seed)
            observation, info = env._get_observation(), env._get_info()
            while not env._episode_ended:
                observation, _, terminated, truncated, info = env.step(rollout_tail(env, observation, info))
            assert info["hours_of_service"]["hos_unplanned_breaks"] == env.hos.unplanned_breaks >= 1
            assert audit_violations(env) == []
            return
    pytest.fail("no realized overrun in 20 seeds")


def test_B10_a_rest_straddling_the_horizon_is_not_credited() -> None:
    env = build("seattle", trucks=1, stops=3, max_time=3.0, hours_of_service=US)
    env.reset(seed=0)
    env.step(_kind_index(env, ActionKind.REST_DAILY))
    assert env._episode_ended and env.termination_reason == "time_limit_exhausted"
    assert env.trucks[0].rest_time == 0.0
    assert not any(e["event_type"] == "HOS_REST" for e in env.trucks[0].event_history)


def test_B11_handling_longer_than_the_session_gives_no_credit_and_no_negative_time() -> None:
    config = h.HosRegime.build("eu_561_2006")
    tracker = HosTracker(dataclasses.replace(
        build("seattle", hours_of_service={"enabled": True, "regime": "eu_561_2006"}).extensions.hours_of_service,
        handling_overhead_hours=1.0), num_trucks=1)
    truck = type("T", (), {"truck_id": 0})()
    tracker.record(truck, h.HosActivity.DRIVE, 0.0, 4.5)
    tracker.record_charge(truck, 4.5, 5.25)
    clock = tracker.clock(0)
    assert clock.time == pytest.approx(5.25)
    assert h.driving_allowance(config, clock) == 0.0


def test_B12_resting_after_charging_frees_the_port_for_the_next_truck() -> None:
    charging = copy.deepcopy(build("seattle").config["charging"])
    charging["port_capacity_scale"] = 1e-9  # one port everywhere
    for seed in range(20):
        if _b12_case(seed, charging):
            return
    pytest.fail("no seed where the second truck queued behind the first")


def _b12_case(seed, charging) -> bool:
    env = build("seattle", trucks=2, stops=3, hours_of_service=US, charging=charging)
    env.reset(seed=seed)
    decisions = joint_action_feasibility(env)
    charger = min(
        (env.transport_graph.get_time_distance(int(env.trucks[0].current_node), d.target_node), i)
        for i, d in enumerate(decisions[: env.num_charging_nodes]) if d.feasible
    )[1]
    env.step(charger)  # truck 0
    env.step(charger)  # truck 1, same station
    charged = 0
    while not env._episode_ended and charged < 2:
        truck = env.trucks[env.active_truck_id]
        decisions = joint_action_feasibility(env)
        charges = [i for i in range(env.num_navigation_actions, env.extra_action_offset) if decisions[i].feasible]
        if charges and not truck.must_leave_charger:
            env.step(charges[-1])
            charged += 1
        elif decisions[_kind_index(env, ActionKind.REST_BREAK)].feasible:
            env.step(_kind_index(env, ActionKind.REST_BREAK))
        else:
            env.step(next(i for i, d in enumerate(decisions) if d.feasible))
    sessions = sorted(
        (
            next(e["timestamp"] for e in truck.event_history if e["event_type"] == "CHARGING_START"),
            next(e["timestamp"] for e in truck.event_history if e["event_type"] == "CHARGING_END"),
        )
        for truck in env.trucks
    )
    (_, first_end), (second_start, _) = sessions
    second = min(env.trucks, key=lambda t: -next(
        e["timestamp"] for e in t.event_history if e["event_type"] == "CHARGING_START"))
    if not any(e["event_type"] == "WAITING_END" for e in second.event_history):
        return False  # it arrived after the port was free: not this case
    assert second_start == pytest.approx(first_end)
    return True


def test_B13_over_limits_at_the_depot_end_shift_and_rests_remain() -> None:
    env = build("seattle", trucks=1, hours_of_service=US, requests=SAME_DAY)
    env.reset(seed=0)
    _set_clock(env, 0, driving_since_break=8.0, driving_in_shift=11.0)
    decisions = joint_action_feasibility(env)
    assert decisions[_kind_index(env, ActionKind.END_SHIFT)].feasible
    assert decisions[_kind_index(env, ActionKind.REST_DAILY)].feasible
    assert not any(d.feasible for d in decisions[: env.num_charging_nodes])


def test_B14_window_wait_completes_a_us_break() -> None:
    tracker = HosTracker(build("seattle", hours_of_service=US).extensions.hours_of_service, num_trucks=1)
    truck = type("T", (), {"truck_id": 0})()
    tracker.record(truck, h.HosActivity.DRIVE, 0.0, 8.0)
    tracker.record(truck, h.HosActivity.WINDOW_WAIT, 8.0, 8.5)
    assert h.driving_allowance(tracker.regime, tracker.clock(0)) == pytest.approx(3.0)


def test_B15_direct_construction_rejects_long_horizons() -> None:
    from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
    from EVRoutingEnv.utils.utils import load_config

    config = copy.deepcopy(load_config("EVRoutingEnv/config_files/config_joint_seattle.yaml"))
    config["environment"]["max_time"] = 120.0
    config["hours_of_service"] = US
    with pytest.raises(ValueError, match="weekly limits"):
        EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
