"""Exercise legal stochastic transitions and post-terminal stability at 400 kWh."""

from __future__ import annotations

import argparse
import json
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config
from scripts.analysis.benchmark_stochastic_simulator import invariants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    args = parser.parse_args()
    rows = []
    for case in range(args.episodes):
        env = None
        steps = 0
        try:
            config = deepcopy(
                load_config("EVRoutingEnv/config_files/config_joint.yaml")
            )
            trucks = (1, 2, 4, 8)[case % 4]
            config["environment"].update(
                num_trucks=trucks, num_stops=4 * trucks, max_episode_steps=120
            )
            if case % 2:
                config["problem"]["assignment"] = "preassigned"
            assert config["truck"]["battery_capacity"] == 400.0
            env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
            observation, info = env.reset(seed=2_000_000_000 + case)
            env.truck_initial_batteries = {
                t.truck_id: t.current_battery for t in env.trucks
            }
            rng = np.random.default_rng(case)
            served = 0
            while True:
                mask = env.mask_fn()
                choices = np.flatnonzero(mask)
                assert choices.size, "active decision has no feasible action"
                clock = env.global_clock
                observation, _, terminated, truncated, info = env.step(
                    int(rng.choice(choices))
                )
                steps += 1
                assert env.global_clock >= clock
                invariants(env)
                metrics = info["operational_metrics"]
                assert metrics["customers_served"] >= served
                served = metrics["customers_served"]
                accounted = sum(
                    metrics[name]
                    for name in [
                        "total_travel_time",
                        "total_charging_time",
                        "total_queue_time",
                        "total_time_window_waiting",
                        "total_service_time",
                    ]
                )
                assert abs(metrics["total_operating_time"] - accounted) < 1e-6
                if terminated or truncated:
                    again, reward, term, cutoff, after = env.step(0)
                    assert (term, cutoff) == (terminated, truncated)
                    assert reward == 0.0
                    assert np.array_equal(again, observation)
                    assert after["global_clock"] == info["global_clock"]
                    break
            rows.append(
                {
                    "case": case,
                    "seed": env.scenario_seed,
                    "trucks": trucks,
                    "customers": 4 * trucks,
                    "assignment": config["problem"].get("assignment", "joint"),
                    "steps": steps,
                    "success": info["successful"],
                    "termination_reason": info["termination_reason"],
                    "invariants_passed": True,
                }
            )
        except Exception as error:  # noqa: BLE001 -- Preserve each independent failure.
            rows.append(
                {
                    "case": case,
                    "steps": steps,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            if env is not None:
                env.close()
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(
        json.dumps(
            {
                "episodes": len(rows),
                "steps": sum(row["steps"] for row in rows),
                "errors": sum("error" in row for row in rows),
            }
        )
    )
    return int(any("error" in row for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
