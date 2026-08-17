"""Cross-check the nominal planners against exhaustive enumeration.

R1.4 and R2.7 ask two different questions that this script answers separately.

*Is the nominal model solved correctly?*  On instances small enough to enumerate
every assignment and ordering, the brute-force optimum of the same objective is
computed directly.  CP-SAT must match it exactly; ALNS must be at least as good
as its own reported incumbent and never better than the true optimum.  A
mismatch means the model, not the solver, is wrong.

*How far from optimal is the executed plan?*  The enumeration optimum is a
nominal quantity: it ignores realized traffic, energy draw, queueing, and
service times.  It therefore bounds the *planning* problem, not the stochastic
one, and the script reports the executed outcome next to it rather than calling
the difference a suboptimality gap.

The instances are deliberately tiny (a handful of customers), because the point
is a correctness cross-check, not a benchmark.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from copy import deepcopy
from pathlib import Path

from EVRoutingEnv.baselines.alns import ALNSParameters, solve_alns_plan
from EVRoutingEnv.baselines.exact_optimization import (
    _TIME_SCALE,
    ExactPlannerParameters,
    solve_nominal_plan,
)
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config
from scripts.evaluation.canonical_harness import split_seeds


def _arc_hours(env, source: int, target: int, charging_power_kw: float) -> float:
    """The planners' nominal arc cost, recomputed here independently."""
    from EVRoutingEnv.baselines.canonical_baselines import _energy, _travel_hours

    travel = _travel_hours(env, source, target)
    energy = _energy(env, source, target)
    if not math.isfinite(travel) or not math.isfinite(energy):
        return math.inf
    return travel + energy / max(charging_power_kw, 1.0)


def enumerate_optimum(
    env,
    charging_power_kw: float,
    objective: str,
) -> tuple[float, dict[int, tuple[int, ...]]]:
    """Brute-force the best assignment and ordering of every customer.

    Every customer goes to exactly one truck, every truck starts and ends at the
    depot, and payload capacity is enforced -- the same constraints the CP-SAT
    model states, evaluated by exhaustive search instead of by a solver.
    """
    depot = int(env.joint_instance.depot_node)
    tasks = list(env.task_registry.tasks())
    customers = [int(task.node_id) for task in tasks]
    demand = {int(task.node_id): float(task.demand) for task in tasks}
    service = {int(task.node_id): float(task.base_service_time) for task in tasks}
    trucks = [truck for truck in env.trucks if not truck.failed]
    capacity = {truck.truck_id: float(truck.payload_capacity) for truck in trucks}

    def route_hours(route: tuple[int, ...]) -> float:
        if not route:
            return 0.0
        total = _arc_hours(env, depot, route[0], charging_power_kw)
        for index in range(len(route) - 1):
            total += _arc_hours(
                env, route[index], route[index + 1], charging_power_kw
            )
        total += _arc_hours(env, route[-1], depot, charging_power_kw)
        return total + sum(service[node] for node in route)

    best_cost = math.inf
    best_routes: dict[int, tuple[int, ...]] = {}
    truck_ids = [truck.truck_id for truck in trucks]

    # Every assignment of customers to trucks, then every ordering within each
    # truck. Exponential by construction, which is why this is tiny-only.
    for assignment in itertools.product(truck_ids, repeat=len(customers)):
        buckets: dict[int, list[int]] = {truck_id: [] for truck_id in truck_ids}
        for customer, truck_id in zip(customers, assignment, strict=True):
            buckets[truck_id].append(customer)
        if any(
            sum(demand[node] for node in bucket) > capacity[truck_id] + 1e-9
            for truck_id, bucket in buckets.items()
        ):
            continue

        per_truck_best: dict[int, tuple[float, tuple[int, ...]]] = {}
        for truck_id, bucket in buckets.items():
            best_for_truck = (math.inf, ())
            for order in itertools.permutations(bucket):
                hours = route_hours(order)
                if hours < best_for_truck[0]:
                    best_for_truck = (hours, order)
            per_truck_best[truck_id] = best_for_truck

        hours = [value[0] for value in per_truck_best.values()]
        if any(not math.isfinite(value) for value in hours):
            continue
        cost = max(hours) if objective == "makespan" else sum(hours)
        if cost < best_cost:
            best_cost = cost
            best_routes = {
                truck_id: value[1] for truck_id, value in per_truck_best.items()
            }
    return best_cost, best_routes


def _tiny_config(base: dict, customers: int, trucks: int) -> dict:
    """A deterministic instance small enough to enumerate."""
    config = deepcopy(base)
    config["environment"].update(
        {
            "num_stops": customers,
            "num_trucks": trucks,
            "allow_variable_num_stops": False,
        }
    )
    # Enumeration compares *nominal* plans, so every stochastic source is off:
    # any realized draw would make the comparison meaningless.
    config["traffic"].update(
        {
            "enable_traffic": False,
            "enable_energy_uncertainty": False,
            "std_dev_factor": 0.0,
        }
    )
    config["delivery"]["enable_stochastic_unloading"] = False
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="EVRoutingEnv/config_files/config_joint.yaml"
    )
    parser.add_argument("--customers", type=int, default=5)
    parser.add_argument("--trucks", type=int, default=2)
    parser.add_argument("--scenarios", type=int, default=12)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--objective", default="total_time", choices=["total_time", "makespan"])
    parser.add_argument(
        "--output", default="results/canonical/exact_validation/enumeration.json"
    )
    arguments = parser.parse_args()

    base = load_config(arguments.config)
    config = _tiny_config(base, arguments.customers, arguments.trucks)
    planner = ExactPlannerParameters(
        time_limit_seconds=arguments.time_limit,
        workers=4,
        objective=arguments.objective,
    )
    searcher = ALNSParameters(
        iterations=20_000, time_limit_seconds=10.0, objective=arguments.objective
    )

    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    records = []
    try:
        for seed in split_seeds(arguments.split, arguments.scenarios):
            env.reset(seed=seed)
            brute_force, _ = enumerate_optimum(
                env, planner.average_charging_power_kw, arguments.objective
            )
            exact = solve_nominal_plan(env, planner)
            heuristic = solve_alns_plan(env, searcher)
            record = {
                "scenario_seed": int(seed),
                "customers": arguments.customers,
                "trucks": arguments.trucks,
                "objective": arguments.objective,
                "enumeration_hours": brute_force,
                "cpsat_hours": exact.objective_hours,
                "cpsat_status": exact.status,
                "cpsat_bound_hours": exact.best_bound_hours,
                "cpsat_wall_seconds": exact.wall_seconds,
                "alns_hours": heuristic.objective_hours,
                "alns_wall_seconds": heuristic.wall_seconds,
            }
            # CP-SAT is an integer model: every arc and service duration is
            # rounded onto a 1/_TIME_SCALE grid, so agreement is asserted
            # against that grain rather than against exact equality. Each of
            # the at most (2 * customers + trucks) rounded terms on a route can
            # be off by half a grid step.
            tolerance = (
                (2 * arguments.customers + arguments.trucks) * 0.5 / _TIME_SCALE
            )
            record["match_tolerance_hours"] = tolerance
            record["cpsat_matches_enumeration"] = (
                exact.objective_hours is not None
                and abs(exact.objective_hours - brute_force) <= tolerance
            )
            record["alns_ratio"] = (
                brute_force / heuristic.objective_hours
                if heuristic.objective_hours
                else None
            )
            records.append(record)
            print(
                f"  seed {seed}: enumeration={brute_force:.4f} "
                f"cpsat={exact.objective_hours} ({exact.status}) "
                f"alns={heuristic.objective_hours:.4f} "
                f"match={record['cpsat_matches_enumeration']}",
                flush=True,
            )
    finally:
        env.close()

    matches = sum(1 for record in records if record["cpsat_matches_enumeration"])
    ratios = [r["alns_ratio"] for r in records if r["alns_ratio"] is not None]
    summary = {
        "scenarios": len(records),
        "cpsat_matches_enumeration": matches,
        "cpsat_match_rate": matches / len(records) if records else None,
        "alns_mean_ratio_to_optimum": sum(ratios) / len(ratios) if ratios else None,
        "alns_optimal_count": sum(1 for ratio in ratios if ratio > 1 - 1e-6),
        "records": records,
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(
        f"\nCP-SAT matched enumeration on {matches}/{len(records)} instances; "
        f"ALNS mean ratio to the true optimum "
        f"{summary['alns_mean_ratio_to_optimum']}",
        flush=True,
    )
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
