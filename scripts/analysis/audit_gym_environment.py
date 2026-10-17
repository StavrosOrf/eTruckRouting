"""Reproduce the Gym audit findings without changing simulator code.

Run from the repository root:
    PYTHONPATH=. python scripts/analysis/audit_gym_environment.py --output audit.json

A reproduced defect is evidence, not a passing correctness test. Exit status is
nonzero for unexpected probe errors, or remaining defects with --check-fixed. Some probes deliberately construct
boundary states through private APIs, as the existing queue tests do; see the
report for scope and reachability qualifications. Environments are isolated and
closed after each probe. No training or published-results files are changed.
"""

from __future__ import annotations

import heapq
import json
import traceback
from contextlib import contextmanager
from copy import deepcopy

import numpy as np

from EVRoutingEnv.models.core.customer import CustomerTask
from EVRoutingEnv.models.environment.curriculum_env import (
    CurriculumEnvWrapper,
    UniformRandomStrategy,
)
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.models.environment.event_handlers import Event, EventType
from EVRoutingEnv.models.environment.joint_instance import (
    _capacity_feasible_demands,
    _preassign_to_trucks,
)
from EVRoutingEnv.utils.utils import load_config


PROBES = []


def probe(name):
    def register(function):
        PROBES.append((name, function))
        return function

    return register


def config(trucks=1, stops=2):
    c = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    c["environment"].update(num_trucks=trucks, num_stops=stops)
    c["truck"]["battery_capacity"] = 10000.0
    c["traffic"].update(enable_traffic=False, enable_energy_uncertainty=False)
    c["delivery"]["enable_stochastic_unloading"] = False
    return c


@contextmanager
def env_for(c):
    e = EventDrivenTruckEnv(c, verbose=False, enable_plotting=False)
    try:
        e.reset(seed=7)
        yield e
    finally:
        e.close()


@probe("discrete_depot_alias")
def depot():
    with env_for(config()) as e:
        a = e.num_charging_nodes + e.fixed_num_stops
        reason = str(e.last_action_feasibility[a].reason)
        node = e.trucks[0].current_node
        _, r, _t, _tr, i = e.step(a)
        return {
            "mask_reason": reason,
            "reward": r,
            "invalid_actions": i["invalid_action_count"],
            "origin": node,
            "position": e.trucks[0].current_node,
            "served": i["task_counts"]["served"],
        }


@probe("penalize_strands_single_truck")
def refused():
    c = config()
    c["environment"].update(invalid_action_mode="penalize", invalid_action_budget=64)
    with env_for(c) as e:
        _, r, t, tr, i = e.step(e.num_navigation_actions)
        return {
            "reward": r,
            "terminated": t,
            "truncated": tr,
            "invalid_actions": i["invalid_action_count"],
            "reason": i["termination_reason"],
            "truck_state": e.truck_states[0],
        }


@probe("task_base_service_time_ignored")
def service():
    c = config()
    c["problem"]["base_service_time"] = 3.0
    c["delivery"]["base_unloading_time"] = 0.2
    with env_for(c) as e:
        node = e.task_registry.tasks()[0].node_id
        e.step((node, 0.0, False))
        task = e.task_registry.task_for_node(node)
        return {
            "task_base": task.base_service_time,
            "served_duration": task.served_at - task.service_started_at,
            "measured_service": e.trucks[0].total_unloading_time,
        }


@probe("exact_energy_arrival_rejected_by_dynamics")
def exact():
    with env_for(config()) as e:
        node = e.charging_nodes[0]
        e.trucks[0].current_battery = e.transport_graph.get_path_energy(
            e.trucks[0].current_node, node
        )
        allowed = bool(e.mask_fn()[0])
        _, r, t, _tr, _i = e.step(0)
        return {
            "mask_allows": allowed,
            "reward": r,
            "terminated": t,
            "battery": e.trucks[0].current_battery,
            "reason": e.trucks[0].failure_reason,
            "location": e.trucks[0].current_node,
        }


@probe("positive_reward_on_feasibility_dead_end")
def failure():
    with env_for(config()) as e:
        node = e.task_registry.tasks()[0].node_id
        en = e.transport_graph.get_path_energy(e.trucks[0].current_node, node)
        e.trucks[0].current_battery = en + 0.0001
        allowed = bool(e.mask_fn()[e.num_charging_nodes])
        _, r, t, _tr, i = e.step(e.num_charging_nodes)
        return {
            "mask_allows": allowed,
            "reward": r,
            "terminated": t,
            "reason": i["termination_reason"],
            "episode_reward": i["episode_reward"],
            "failure_penalty": e.reward_config["failure_penalty"],
        }


@probe("time_limit_overshoot")
def deadline():
    c = config(stops=1)
    c["environment"]["max_time"] = 0.01
    with env_for(c) as e:
        _, _r, t, tr, i = e.step(e.num_charging_nodes)
        return {
            "clock": e.global_clock,
            "limit": e.max_time,
            "served": i["task_counts"]["served"],
            "terminated": t,
            "truncated": tr,
            "reason": i["termination_reason"],
        }


@probe("charge_reward_missing_multiplier")
def charging_reward():
    c = config()
    c["rewards"]["time_multiplier"] = 5.0
    with env_for(c) as e:
        truck = e.trucks[0]
        truck.current_node = e.charging_nodes[0]
        truck.current_battery = 4000.0
        _, r, _t, _tr, _i = e.step((truck.current_node, 0.5, True))
        return {
            "reward": r,
            "duration": truck.total_charging_time,
            "configured_multiplier": 5.0,
            "expected_reward": -5.0 * truck.total_charging_time,
        }


@probe("preassignment_overloads_feasible_demands")
def preassign():
    tasks = tuple(
        CustomerTask(i, i + 1, d, 0.2) for i, d in enumerate([5.0, 5.0, 4.0, 3.0, 3.0])
    )
    assigned = _preassign_to_trucks(tasks, num_trucks=2, payload_capacity=10.0)
    loads = [sum(t.demand for t in assigned if t.preassigned_to == i) for i in range(2)]
    generated = None
    for seed in range(10000):
        demands = _capacity_feasible_demands(
            np.random.default_rng(seed),
            count=5,
            num_trucks=2,
            payload_capacity=10.0,
            min_demand=1.0,
            max_demand=6.0,
        )
        ts = tuple(CustomerTask(i, i + 1, d, 0.2) for i, d in enumerate(demands))
        assigned = _preassign_to_trucks(ts, num_trucks=2, payload_capacity=10.0)
        ls = [
            sum(t.demand for t in assigned if t.preassigned_to == i) for i in range(2)
        ]
        if max(ls) > 10.0 + 1e-9:
            generated = {"seed": seed, "demands": demands, "loads": ls}
            break
    return {"capacity": 10.0, "explicit_loads": loads, "generated": generated}


@probe("curriculum_constructor")
def curriculum():
    try:
        e = CurriculumEnvWrapper(config(), UniformRandomStrategy((1, 2), (1, 2)))
    except Exception as exc:  # noqa: BLE001 -- Record each isolated probe error.
        return {"exception": type(exc).__name__, "message": str(exc)}
    e.close()
    return {"constructed": True}


@probe("routing_eta_not_in_observation")
def eta():
    with env_for(config(trucks=2)) as e:
        e.step(e.num_charging_nodes)
        truck = e.trucks[0]
        before = e._get_observation().copy()
        old = truck.route_arrival_time
        truck.route_arrival_time += 10.0
        for event in e.event_queue:
            if event.truck_id == 0 and event.event_type is EventType.TRUCK_ROUTING:
                event.time += 10.0
        heapq.heapify(e.event_queue)
        after = e._get_observation()
        return {
            "state": e.truck_states[0],
            "old_eta": old,
            "new_eta": truck.route_arrival_time,
            "same_observation": bool(np.array_equal(before, after)),
        }


@probe("waitlist_leaks_on_charger_to_charger_departure")
def waitlist():
    with env_for(config(trucks=2)) as e:
        station = e.charging_station
        src, dst = e.charging_nodes[:2]
        e.event_queue.clear()
        e.active_truck_id = None
        for truck in e.trucks:
            truck.current_node = src
            truck.current_battery = 5000.0
            e.truck_states[truck.truck_id] = "waiting_to_charge"
        station.charger_capacity[src] = 1
        station.charger_waitlist[src] = [
            {"truck_id": 0, "sequence": 1},
            {"truck_id": 1, "sequence": 2},
        ]
        station.wake_waiting_trucks(
            src, e.global_clock, e.event_queue, EventType, Event, e.truck_states
        )
        e._advance_to_next_decision()
        e.step((dst, 0.0, False))
        return {
            "active": e.active_truck_id,
            "other_truck_state": e.truck_states[1],
            "free_ports": 1 - len(station.charger_occupancy[src]),
            "old_waitlist": station.charger_waitlist[src],
            "events_pending": len(e.event_queue),
        }


@probe("failure_reason_on_legacy_success")
def legacy():
    c = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    c["environment"].update(num_trucks=1, num_stops=1, verbose=False)
    c["truck"]["battery_capacity"] = 10000.0
    c["traffic"].update(enable_traffic=False, enable_energy_uncertainty=False)
    with env_for(c) as e:
        for _ in range(3):
            target = e.trucks[0].get_remaining_deliveries()[0]
            _, _r, t, tr, i = e.step((target, 0.0, False))
            if t or tr:
                break
        return {
            "terminated": t,
            "successful": i["successful"],
            "reason": i["termination_reason"],
            "customers": e._get_info()["operational_metrics"]["customers_served"],
        }


@probe("stale_wake_reason_not_filtered")
def stale():
    with env_for(config()) as e:
        truck = e.trucks[0]
        target = e.task_registry.tasks()[0].node_id
        e._execute_navigation_action(truck, target)
        original_eta = truck.route_arrival_time
        heapq.heappush(
            e.event_queue,
            Event(
                time=0.0,
                truck_id=0,
                event_type=EventType.TRUCK_READY,
                data={"reason": "charger_port_available"},
            ),
        )
        e._advance_to_next_decision()
        result = {
            "truck_state": e.truck_states[0],
            "route_still_pending": truck.route_destination is not None,
            "clock": e.global_clock,
            "arrival_time": original_eta,
        }
        try:
            e.step((e.task_registry.tasks()[1].node_id, 0.0, False))
        except Exception as exc:  # noqa: BLE001 -- Record each isolated probe error.
            result["next_action_exception"] = repr(exc)
        return result


@probe("success_beyond_horizon")
def success_after_deadline():
    c = config(stops=1)
    c["environment"]["max_time"] = 30.0
    with env_for(c) as e:
        e.step(e.num_charging_nodes)
        _, _, terminated, truncated, info = e.step(
            e.num_charging_nodes + e.fixed_num_stops
        )
        return {
            "clock": e.global_clock,
            "limit": e.max_time,
            "terminated": terminated,
            "truncated": truncated,
            "successful": info["successful"],
            "reason": info["termination_reason"],
        }


@probe("legacy_default_constructor")
def legacy_defaults():
    outcomes = {}
    for path in ("config.yaml", "config_joint.yaml"):
        try:
            e = EventDrivenTruckEnv("EVRoutingEnv/config_files/" + path)
        except Exception as exc:  # noqa: BLE001 -- Record each isolated probe error.
            outcomes[path] = {"exception": type(exc).__name__, "message": str(exc)}
        else:
            e.close()
            outcomes[path] = {"constructed": True}
    return outcomes


@probe("foreign_graph_cache")
def foreign_cache():
    import networkx as nx

    from EVRoutingEnv.models.core.transportation_graph import TransportationGraph

    graph = nx.DiGraph()
    graph.add_edge(0, 1, distance=999.0, time=999.0)
    transport = TransportationGraph(graph)
    return {"edge_energy": 999.0, "observed_energy": transport.get_path_energy(0, 1)}


@probe("time_cache_after_scale")
def time_cache():
    import networkx as nx

    from EVRoutingEnv.models.core.transportation_graph import TransportationGraph

    graph = nx.DiGraph()
    graph.add_edge(0, 1, distance=1.0, time=2.0)
    transport = TransportationGraph(graph, precompute_distances=False)
    before = transport.get_time_distance(0, 1)
    transport.scale_network(time_scale=3.0)
    return {
        "before": before,
        "edge_time": transport.graph[0][1]["time"],
        "observed_after": transport.get_time_distance(0, 1),
    }


@probe("legacy_fleet_delivery_slots")
def legacy_slots():
    c = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    c["environment"].update(num_trucks=2, num_stops=3)
    with env_for(c) as e:
        unique = set().union(*(set(t.delivery_sequence[1:]) for t in e.trucks))
        return {
            "unique_customers": len(unique),
            "flat_delivery_slots": e.state_space_manager.num_stops,
        }


@probe("undeclared_cpsat_dependency")
def cpsat_dependency():
    import tomllib
    from pathlib import Path

    project = tomllib.loads(Path("pyproject.toml").read_text())
    declared = list(project["project"]["dependencies"])
    for dependencies in project["project"].get("optional-dependencies", {}).values():
        declared.extend(dependencies)
    return {
        "declared": any(
            dependency.lower().startswith("ortools") for dependency in declared
        ),
        "present_in_lock": 'name = "ortools"' in Path("uv.lock").read_text(),
    }


@probe("legacy_last_customer_skips_service")
def legacy_final_service():
    c = deepcopy(load_config("EVRoutingEnv/config_files/config.yaml"))
    c["environment"].update(num_trucks=1, num_stops=1)
    c["truck"]["battery_capacity"] = 10000.0
    c["traffic"].update(enable_traffic=False, enable_energy_uncertainty=False)
    with env_for(c) as e:
        _, reward, terminated, _, _ = e.step(
            (e.trucks[0].get_next_delivery_target(), 0.0, False)
        )
        return {
            "terminated": terminated,
            "service_time": e.trucks[0].total_unloading_time,
            "configured_service_time": c["delivery"]["base_unloading_time"],
            "reward": reward,
        }


@probe("legacy_depot_collects_delivery_bonus")
def legacy_depot_bonus():
    c = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    c["environment"].update(num_trucks=1, num_stops=1)
    c["truck"]["battery_capacity"] = 10000.0
    c["traffic"].update(enable_traffic=False, enable_energy_uncertainty=False)
    with env_for(c) as e:
        e.step((e.trucks[0].get_remaining_deliveries()[0], 0.0, False))
        before = e.trucks[0].total_routing_time
        _, reward, _, _, _ = e.step((e.trucks[0].delivery_sequence[0], 0.0, False))
        travel = e.trucks[0].total_routing_time - before
        return {
            "depot_reward": reward,
            "return_travel_time": travel,
            "bonus_paid_at_depot": reward + travel * e.reward_config["time_multiplier"],
        }


CHECKS = {
    "legacy_last_customer_skips_service": (
        "G18",
        lambda o: (
            o["terminated"]
            and o["service_time"] == 0.0
            and o["configured_service_time"] > 0.0
        ),
    ),
    "legacy_depot_collects_delivery_bonus": (
        "G19",
        lambda o: o["bonus_paid_at_depot"] > 0.0,
    ),
    "undeclared_cpsat_dependency": (
        "G17",
        lambda o: not o["declared"] and not o["present_in_lock"],
    ),
    "discrete_depot_alias": (
        "G01",
        lambda o: o["invalid_actions"] == 0 and o["served"] > 0,
    ),
    "penalize_strands_single_truck": (
        "G02",
        lambda o: o["terminated"] and o["invalid_actions"] == 1,
    ),
    "preassignment_overloads_feasible_demands": (
        "G03",
        lambda o: max(o["explicit_loads"]) > o["capacity"],
    ),
    "time_limit_overshoot": (
        "G04",
        lambda o: o["clock"] > o["limit"] and o["served"] > 0,
    ),
    "success_beyond_horizon": (
        "G04",
        lambda o: o["clock"] > o["limit"] and o["successful"] and o["truncated"],
    ),
    "waitlist_leaks_on_charger_to_charger_departure": (
        "G05",
        lambda o: (
            o["free_ports"] > 0
            and o["other_truck_state"] == "waiting_to_charge"
            and o["events_pending"] == 0
        ),
    ),
    "stale_wake_reason_not_filtered": (
        "G06",
        lambda o: (
            o.get("next_action_exception") == "RuntimeError('truck is already routing')"
        ),
    ),
    "task_base_service_time_ignored": (
        "G07",
        lambda o: abs(o["task_base"] - o["served_duration"]) > 1e-6,
    ),
    "positive_reward_on_feasibility_dead_end": (
        "G08",
        lambda o: (
            o["terminated"] and o["reward"] > 0 and o["reason"] == "no_feasible_action"
        ),
    ),
    "charge_reward_missing_multiplier": (
        "G09",
        lambda o: abs(o["reward"] - o["expected_reward"]) > 1e-6,
    ),
    "exact_energy_arrival_rejected_by_dynamics": (
        "G10",
        lambda o: o["mask_allows"] and o["reason"] == "battery_depleted",
    ),
    "curriculum_constructor": ("G11", lambda o: o.get("exception") == "ValueError"),
    "legacy_default_constructor": (
        "G12",
        lambda o: all(v.get("exception") == "KeyError" for v in o.values()),
    ),
    "failure_reason_on_legacy_success": (
        "G13",
        lambda o: o["successful"] and o["reason"] != "success",
    ),
    "foreign_graph_cache": ("G14", lambda o: o["observed_energy"] != o["edge_energy"]),
    "time_cache_after_scale": ("G15", lambda o: o["observed_after"] != o["edge_time"]),
    "legacy_fleet_delivery_slots": (
        "G16",
        lambda o: o["unique_customers"] > o["flat_delivery_slots"],
    ),
    "routing_eta_not_in_observation": ("L01", lambda o: o["same_observation"]),
}


def main():
    import argparse
    import importlib.metadata
    import subprocess
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check-fixed",
        action="store_true",
        help="Fail if a G01-G19 defect still reproduces; L01 is a documented modeling limitation.",
    )
    parser.add_argument(
        "--probe", action="append", choices=[name for name, _ in PROBES]
    )
    args = parser.parse_args()
    results = []
    for name, function in PROBES:
        if args.probe and name not in args.probe:
            continue
        finding, reproduced = CHECKS[name]
        print(f"Running {finding}: {name}", file=sys.stderr, flush=True)
        try:
            observed = function()
            results.append(
                {
                    "finding": finding,
                    "probe": name,
                    "reproduced": bool(reproduced(observed)),
                    "observed": observed,
                }
            )
        except Exception as exc:  # noqa: BLE001 -- Record each isolated probe error.
            results.append(
                {
                    "finding": finding,
                    "probe": name,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    payload = {
        "audited_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "baseline_commit": "be712ca6eb8c1b7071f1e2f248b4a10ff5881762",
        "python": sys.version.split()[0],
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "gymnasium", "torch")
        },
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, default=str) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return int(
        any(
            "error" in result
            or (
                args.check_fixed
                and result["finding"].startswith("G")
                and result.get("reproduced", False)
            )
            for result in results
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
