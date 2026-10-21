"""SYSTEM events: validation, same-time priorities and isolation from trucks."""

import copy
import heapq
import os

import pytest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.models.environment.event_handlers import SYSTEM_TRUCK_ID, Event, EventType
from EVRoutingEnv.utils.utils import load_config


def test_system_events_require_the_system_id_and_a_known_kind() -> None:
    Event(time=1.0, truck_id=SYSTEM_TRUCK_ID, event_type=EventType.SYSTEM, data={"kind": "epoch_boundary"})
    with pytest.raises(ValueError, match="truck_id -1"):
        Event(time=1.0, truck_id=0, event_type=EventType.SYSTEM, data={"kind": "epoch_boundary"})
    with pytest.raises(ValueError, match="kind"):
        Event(time=1.0, truck_id=-1, event_type=EventType.SYSTEM, data={"kind": "teleport"})
    with pytest.raises(ValueError, match="non-negative"):
        Event(time=1.0, truck_id=-1, event_type=EventType.TRUCK_READY)


def test_same_time_order_is_system_routing_completion_other() -> None:
    events = [
        Event(time=2.0, truck_id=0, event_type=EventType.TRUCK_READY, data={"reason": "initial"}),
        Event(time=2.0, truck_id=0, event_type=EventType.TRUCK_READY, data={"reason": "charge_complete"}),
        Event(time=2.0, truck_id=0, event_type=EventType.TRUCK_ROUTING),
        Event(time=2.0, truck_id=SYSTEM_TRUCK_ID, event_type=EventType.SYSTEM,
              data={"kind": "request_release"}, sequence=2),
        Event(time=2.0, truck_id=SYSTEM_TRUCK_ID, event_type=EventType.SYSTEM,
              data={"kind": "request_release"}, sequence=1),
    ]
    order = []
    heap = list(events)
    heapq.heapify(heap)
    while heap:
        event = heapq.heappop(heap)
        order.append((event.event_type, event.data.get("reason"), event.sequence))
    assert order == [
        (EventType.SYSTEM, None, 1),
        (EventType.SYSTEM, None, 2),
        (EventType.TRUCK_ROUTING, None, 0),
        (EventType.TRUCK_READY, "charge_complete", 0),
        (EventType.TRUCK_READY, "initial", 0),
    ]


def _env(trucks=3):
    config = copy.deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["environment"]["num_trucks"] = trucks
    return EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)


def test_system_event_at_the_head_never_reads_or_changes_the_last_truck() -> None:
    env = _env(3)
    env.reset(seed=1)
    last = env.trucks[-1]
    last.is_complete = True  # the old loop would discard the event as this truck's
    before = copy.deepcopy(last.__dict__)
    seen = []
    env._system_event_handlers["epoch_boundary"] = "_probe"
    env._probe = lambda event: seen.append(event.time)
    env._push_system_event(env.global_clock, "epoch_boundary")
    env._advance_to_next_decision()
    assert seen == [env.global_clock]
    assert last.__dict__ == before


def test_system_events_at_or_past_the_horizon_are_dropped_and_never_truncate() -> None:
    from EVRoutingEnv.baselines.heuristics.canonical_baselines import GreedyHeuristicPolicy

    env = _env(2)
    observation, info = env.reset(seed=0)
    seen = []
    env._system_event_handlers["epoch_boundary"] = "_probe"
    env._probe = lambda event: seen.append(event.time)
    env._push_system_event(env.max_time, "epoch_boundary")
    env._push_system_event(env.max_time + 5.0, "epoch_boundary")
    policy = GreedyHeuristicPolicy()
    done = False
    while not done:
        observation, _, terminated, truncated, info = env.step(policy(env, observation, info))
        done = terminated or truncated
    assert seen == []
    assert info["termination_reason"] == "success"


def test_unregistered_system_event_kind_is_an_error() -> None:
    env = _env(2)
    env.reset(seed=0)
    env._push_system_event(env.global_clock, "request_expire", request_id=0)
    with pytest.raises(RuntimeError, match="no handler"):
        env._advance_to_next_decision()
