"""Test the per-truck MILP's *formulation* without needing a Gurobi licence.

R1.4 objects to the inherited per-truck model being presented as an optimality
reference. Validating it the way the fleet planner was validated -- solve, then
compare against exhaustive enumeration -- needs Gurobi, which may not be
available. But the fleet planner's defect was not in its solver: it was that the
model *could not express* the optimal plan, because its circuit forced every
truck to serve a customer. That class of defect is testable without solving
anything, by enumerating what the model can express and what it cannot.

The per-truck model documents this restriction:

    "At most one charger visit between consecutive deliveries"

So this script enumerates, on tiny deterministic instances, the best plan
reachable under that restriction and the best plan reachable when up to ``k``
charging stops per segment are allowed. If the two differ, the restriction is
binding and the model's "optimal" label is unsound in the same way the fleet
planner's was -- regardless of how well Gurobi solves it.

Charging is priced with the simulator's own integrator, so a plan this script
calls feasible is one the simulator would actually execute.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from copy import deepcopy
from pathlib import Path

from EVRoutingEnv.baselines.canonical_baselines import _energy, _travel_hours
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.models.simulation.charging_curve import ChargingCurveModel
from EVRoutingEnv.utils.utils import load_config
from scripts.evaluation.canonical_harness import split_seeds


def _charge_hours(model, env, node, capacity, start_kwh, target_kwh) -> float:
    """Hours to raise the battery from ``start_kwh`` to ``target_kwh`` at ``node``."""
    if target_kwh <= start_kwh + 1e-9:
        return 0.0
    station = env.charging_station
    charger_type = station.charger_type.get(int(node), "DCFast")
    section = "dcfast" if charger_type == "DCFast" else "level2"
    config = dict(env.charging_config.get(section, {}))
    config["use_realistic_curve"] = env.charging_config.get("use_realistic_curve", False)
    config["charge_rate"] = float(station.charger_power_kw.get(int(node), 150.0))
    config["efficiency"] = float(
        station.charger_efficiency.get(int(node), config.get("efficiency", 0.90))
    )
    if "taper_power_min" in config:
        config["taper_power_min"] = min(
            float(config["taper_power_min"]), config["charge_rate"]
        )
    _, details = model.calculate_charge_to_target(
        initial_soc=max(0.0, min(1.0, start_kwh / capacity)),
        target_soc=max(1e-6, min(1.0, target_kwh / capacity)),
        battery_capacity=capacity,
        charger_config=config,
        charger_type=charger_type,
    )
    return float(details["actual_charge_hours"])


def _segment_cost(
    env, curve, capacity, battery, origin, destination, stops, safety
) -> tuple[float, float] | None:
    """Time and remaining battery for one segment via ``stops`` chargers, or None.

    At each charging stop the truck charges the minimum needed to reach the next
    node with the safety margin, which is time-optimal for a monotone charging
    curve: charging more can never shorten the remaining route.
    """
    total_time = 0.0
    here = origin
    for index, node in enumerate(list(stops) + [destination]):
        leg_energy = _energy(env, here, node) * safety
        leg_time = _travel_hours(env, here, node)
        if not math.isfinite(leg_energy) or not math.isfinite(leg_time):
            return None
        if battery < leg_energy - 1e-9:
            return None
        battery -= leg_energy
        total_time += leg_time
        here = node
        if index < len(stops):
            following = stops[index + 1] if index + 1 < len(stops) else destination
            need = _energy(env, node, following) * safety
            if not math.isfinite(need):
                return None
            target = min(capacity, max(battery, need))
            total_time += _charge_hours(curve, env, node, capacity, battery, target)
            battery = target
    return total_time, battery


def _best_plan(env, curve, capacity, sequence, chargers, max_stops, safety):
    """Exhaustive best plan when at most ``max_stops`` chargers may be used per leg."""
    start_battery = capacity
    best = math.inf

    def recurse(index: int, battery: float, elapsed: float) -> None:
        nonlocal best
        if elapsed >= best:
            return
        if index >= len(sequence) - 1:
            best = min(best, elapsed)
            return
        origin, destination = sequence[index], sequence[index + 1]
        for count in range(max_stops + 1):
            for stops in itertools.permutations(chargers, count):
                outcome = _segment_cost(
                    env, curve, capacity, battery, origin, destination, stops, safety
                )
                if outcome is None:
                    continue
                recurse(index + 1, outcome[1], elapsed + outcome[0])

    recurse(0, start_battery, 0.0)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="EVRoutingEnv/config_files/config_joint_preassigned.yaml"
    )
    parser.add_argument("--customers", type=int, default=3)
    parser.add_argument("--trucks", type=int, default=2)
    parser.add_argument("--chargers", type=int, default=4,
                        help="Nearest chargers considered, to keep enumeration finite.")
    parser.add_argument("--max-stops", type=int, default=2)
    parser.add_argument("--scenarios", type=int, default=12)
    parser.add_argument("--safety", type=float, default=1.20)
    parser.add_argument(
        "--output",
        default="results/canonical/exact_validation/per_truck_model.json",
    )
    arguments = parser.parse_args()

    config = deepcopy(load_config(arguments.config))
    config["environment"].update(
        {
            "num_stops": arguments.customers,
            "num_trucks": arguments.trucks,
            "allow_variable_num_stops": False,
        }
    )
    config["traffic"].update(
        {"enable_traffic": False, "enable_energy_uncertainty": False,
         "std_dev_factor": 0.0}
    )
    config["delivery"]["enable_stochastic_unloading"] = False

    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    curve = ChargingCurveModel()
    records = []
    try:
        for seed in split_seeds("validation", arguments.scenarios):
            env.reset(seed=seed)
            capacity = float(env.trucks[0].battery_capacity)
            depot = int(env.joint_instance.depot_node)
            for truck in env.trucks:
                owned = [
                    int(task.node_id)
                    for task in env.task_registry.tasks()
                    if task.preassigned_to in (None, truck.truck_id)
                ]
                if not owned:
                    continue
                sequence = [depot, *owned, depot]
                nearest = sorted(
                    env.charging_nodes,
                    key=lambda node: min(
                        _travel_hours(env, stop, int(node)) for stop in sequence
                    ),
                )[: arguments.chargers]
                restricted = _best_plan(
                    env, curve, capacity, sequence, nearest, 1, arguments.safety
                )
                general = _best_plan(
                    env, curve, capacity, sequence, nearest,
                    arguments.max_stops, arguments.safety,
                )
                records.append(
                    {
                        "scenario_seed": int(seed),
                        "truck": int(truck.truck_id),
                        "stops": len(owned),
                        "best_with_one_charger_per_leg": restricted,
                        "best_with_up_to_k_per_leg": general,
                        "restriction_binds": bool(
                            general < restricted - 1e-6
                            or (math.isinf(restricted) and math.isfinite(general))
                        ),
                        "restriction_makes_infeasible": bool(
                            math.isinf(restricted) and math.isfinite(general)
                        ),
                    }
                )
    finally:
        env.close()

    binding = [r for r in records if r["restriction_binds"]]
    infeasible = [r for r in records if r["restriction_makes_infeasible"]]
    summary = {
        "plans": len(records),
        "restriction_binds": len(binding),
        "restriction_makes_infeasible": len(infeasible),
        "max_stops_allowed_in_general_model": arguments.max_stops,
        "records": records,
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"plans enumerated: {len(records)}")
    print(f"  one-charger-per-leg restriction is binding: {len(binding)}")
    print(f"  of which it makes an otherwise feasible route infeasible: {len(infeasible)}")
    for record in binding[:5]:
        print(
            f"    seed {record['scenario_seed']} truck {record['truck']}: "
            f"restricted={record['best_with_one_charger_per_leg']:.3f} "
            f"general={record['best_with_up_to_k_per_leg']:.3f}"
        )
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
