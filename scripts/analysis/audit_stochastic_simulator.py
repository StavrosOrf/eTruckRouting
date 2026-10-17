"""Reproduce second-pass simulator findings without modifying runtime code."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.models.simulation.charging_curve import ChargingCurveModel
from EVRoutingEnv.models.simulation.delivery_simulator import DeliverySimulator
from EVRoutingEnv.utils.utils import load_config


@contextmanager
def environment(trucks=1, customers=2, **sections):
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["environment"].update(num_trucks=trucks, num_stops=customers)
    assert config["truck"]["battery_capacity"] == 400.0
    for section, values in sections.items():
        config.setdefault(section, {}).update(values)
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        env.reset(seed=7)
        # Small controlled leg costs isolate event semantics, at the unchanged
        # default battery. Actual-map scaling is measured by the benchmark.
        env.transport_graph.get_path_energy = lambda a, b: float(a != b)
        yield env
    finally:
        env.close()


def unearned_reward():
    with environment(customers=1, environment={"max_time": 0.01}) as env:
        _, reward, terminated, truncated, info = env.step(env.num_charging_nodes)
        return {
            "reward": reward,
            "served": info["task_counts"]["served"],
            "clock": env.global_clock,
            "terminated": terminated,
            "truncated": truncated,
            "operating_hours": info["operational_metrics"]["total_operating_time"],
        }


def preassigned_wait():
    with environment(
        trucks=2,
        customers=4,
        problem={"assignment": "preassigned"},
        environment={"max_time": 1000.0},
    ) as env:
        # Every leg is affordable. Make truck 1's first service slower than
        # truck 0's two services, so foreign work remains unclaimed.
        original_time = env.transport_graph.get_time_distance
        owner = {
            task.node_id: task.preassigned_to for task in env.task_registry.tasks()
        }

        def leg_time(origin, target):
            if origin == target:
                return 0.0
            return 1.0 if owner.get(target) == 0 else 100.0

        env.transport_graph.get_time_distance = leg_time
        for node in env.charging_nodes:
            env.set_charger_available(node, False)
        rows = []
        for _ in range(8):
            active = env.active_truck_id
            if active is None:
                break
            available = [
                t
                for t in env.task_registry.tasks()
                if t.is_available and t.preassigned_to == active
            ]
            if not available:
                break
            _, _, terminated, truncated, _ = env.step(
                (available[0].node_id, 0.0, False)
            )
            rows.append(
                {
                    "acting": active,
                    "clock": env.global_clock,
                    "failed": [t.truck_id for t in env.trucks if t.failed],
                }
            )
            if env.trucks[0].failed or terminated or truncated:
                break
        result = {
            "transitions": rows,
            "truck0_failed": env.trucks[0].failed,
            "reason": env.trucks[0].failure_reason,
            "truck0_served": env.trucks[0].served_task_ids,
            "foreign_unassigned": [
                t.task_id
                for t in env.task_registry.tasks()
                if t.is_available and t.preassigned_to == 1
            ],
            "truck1_state": env.truck_states[1],
        }
        env.transport_graph.get_time_distance = original_time
        return result


def negative_action():
    with environment() as env:
        before = env.trucks[0].current_node
        try:
            _, reward, terminated, truncated, info = env.step(-1)
        except ValueError:
            return {
                "rejected": True,
                "in_action_space": env.action_space.contains(-1),
                "before": before,
                "after": env.trucks[0].current_node,
            }
        return {
            "in_action_space": env.action_space.contains(-1),
            "before": before,
            "after": env.trucks[0].current_node,
            "last_charger": env.charging_nodes[-1],
            "invalid_actions": info["invalid_action_count"],
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }


def step_after_truncation():
    with environment(environment={"max_episode_steps": 1}) as env:
        _, _, terminated, truncated, _ = env.step(env.num_charging_nodes)
        before = {
            "clock": env.global_clock,
            "steps": env.episode_steps,
            "served": len(env.trucks[0].served_task_ids),
        }
        candidates = np.flatnonzero(env.mask_fn())
        if candidates.size:
            env.step(int(candidates[0]))
        return {
            "first_terminated": terminated,
            "first_truncated": truncated,
            "before": before,
            "after": {
                "clock": env.global_clock,
                "steps": env.episode_steps,
                "served": len(env.trucks[0].served_task_ids),
            },
        }


def tiny_service():
    simulator = DeliverySimulator(
        enable_stochastic_unloading=True, base_unloading_time=0.02, seed=7
    )
    samples = np.asarray([simulator.apply_unloading_time(i, 12.0) for i in range(1000)])
    return {
        "base": 0.02,
        "mean": float(samples.mean()),
        "std": float(samples.std()),
        "minimum": float(samples.min()),
        "maximum": float(samples.max()),
        "unique": len(np.unique(samples)),
    }


def charging_stagnation_child(output):
    initial = 0.5
    power = 1e-12
    delta = power * 0.9 * 0.001 / 400.0
    output.put(
        {
            "initial_soc": initial,
            "step_delta": delta,
            "changes_soc": initial + delta > initial,
        }
    )
    ChargingCurveModel().calculate_charge_to_target(
        initial_soc=initial,
        target_soc=0.6,
        battery_capacity=400.0,
        charger_type="DCFast",
        charger_config={
            "charge_rate": power,
            "efficiency": 0.9,
            "use_realistic_curve": True,
            "taper_start_soc": 0.8,
            "taper_power_min": power,
        },
    )
    output.put({"returned": True})


def charging_stagnation():
    context = mp.get_context("fork")
    output = context.Queue()
    process = context.Process(target=charging_stagnation_child, args=(output,))
    process.start()
    info = output.get(timeout=2)
    process.join(timeout=1)
    stalled = process.is_alive()
    if stalled:
        process.terminate()
        process.join()
    output.close()
    return info | {"timed_out": stalled, "timeout_seconds": 1.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-fixed", action="store_true")
    args = parser.parse_args()
    cases = [
        ("N01", "delivery_bonus_before_service", unearned_reward),
        ("N02", "ownership_ignored_by_idling", preassigned_wait),
        ("N03", "negative_discrete_action_accepted", negative_action),
        ("N04", "step_limit_continues_on_second_step", step_after_truncation),
        ("N05", "small_service_distribution_collapses", tiny_service),
        ("N06", "charging_integrator_no_progress", charging_stagnation),
    ]
    rows = []
    for finding, name, function in cases:
        started = time.perf_counter()
        try:
            row = {"finding": finding, "probe": name, "observed": function()}
        except Exception as error:  # noqa: BLE001 -- Persist isolated probe failures.
            row = {
                "finding": finding,
                "probe": name,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        row["wall_seconds"] = time.perf_counter() - started
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.check_fixed:
        expected = {
            "N01": lambda value: (
                value["reward"] < 0
                and value["served"] == 0
                and value["operating_hours"] == 0.01
            ),
            "N02": lambda value: not value["truck0_failed"],
            "N03": lambda value: (
                value.get("rejected", False) and value["before"] == value["after"]
            ),
            "N04": lambda value: value["before"] == value["after"],
            "N05": lambda value: value["unique"] > 1 and value["std"] > 0,
            "N06": lambda value: not value["timed_out"],
        }
        for row in rows:
            row["fixed"] = "error" not in row and expected[row["finding"]](
                row["observed"]
            )
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    return int(any("error" in row or row.get("fixed") is False for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
