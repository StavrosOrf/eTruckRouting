"""Validate the per-truck Gurobi MILP against exhaustive enumeration.

`validate_per_truck_model.py` tests what the per-truck model can *express*, and
needs no solver. This script tests what its Gurobi *encoding* actually returns,
which is the narrower gap that doc 11 §11.4 left open pending a licence.

The comparison is made on the simulator's own cost model, not the MILP's. The
MILP prices charging linearly and pads energy with a safety factor; enumeration
prices it with the nonlinear integrator the simulator actually executes. So a
gap here is not a coding bug -- it is the operational cost of the MILP's
internal approximations, which is exactly what a reviewer asking "is this
really the optimum?" wants quantified.

Method, per truck and scenario:

1. Solve with `OptimalVRPSingleTruckPolicy` and read the delivery order and the
   charger chosen for each leg out of the returned plan.
2. Re-price that structure with the simulator's integrator, charging the
   minimum needed at each stop (time-optimal for a monotone curve).
3. Enumerate every delivery order and every choice of at most one charger per
   leg -- the MILP's own documented restriction -- and price each the same way.
4. Report the ratio of the MILP's plan to the enumerated optimum.

A correct encoding solving its own model well should land at ratio 1.0 whenever
its approximations do not bite, and above 1.0 by the amount they cost. A ratio
below 1.0 would mean the MILP found a plan enumeration could not reach, which
under the same restriction would indicate the enumeration is wrong.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from copy import deepcopy
from pathlib import Path

from EVRoutingEnv.baselines.canonical_baselines import _travel_hours
from EVRoutingEnv.baselines.optimal_vrp_single_truck import OptimalVRPSingleTruckPolicy
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.models.simulation.charging_curve import ChargingCurveModel
from EVRoutingEnv.utils.utils import load_config
from scripts.evaluation.canonical_harness import split_seeds
from scripts.evaluation.validate_per_truck_model import _segment_cost


def _price_structure(env, curve, capacity, order, stops_per_leg, depot, safety):
    """Cost a delivery order with a given charger choice per leg, or None."""
    sequence = [depot, *order, depot]
    battery = capacity
    total = 0.0
    for index in range(len(sequence) - 1):
        stops = stops_per_leg[index]
        outcome = _segment_cost(
            env, curve, capacity, battery,
            sequence[index], sequence[index + 1], stops, safety,
        )
        if outcome is None:
            return None
        leg_time, battery = outcome
        total += leg_time
    return total


def _enumerate_best(env, curve, capacity, owned, chargers, depot, safety):
    """Best plan over all delivery orders with at most one charger per leg."""
    best = math.inf
    best_order = None
    for order in itertools.permutations(owned):
        legs = len(order) + 1
        # Per leg: no charger, or exactly one of the candidates.
        choices = [[()] + [(int(c),) for c in chargers] for _ in range(legs)]
        for combo in itertools.product(*choices):
            cost = _price_structure(env, curve, capacity, order, combo, depot, safety)
            if cost is not None and cost < best:
                best, best_order = cost, (order, combo)
    return best, best_order


def _structure_from_plan(plan, depot, owned):
    """Read (delivery order, chargers per leg) out of a solver plan."""
    order = []
    legs = [[]]
    for step in plan:
        if step.kind == "nav_charger" and step.target is not None:
            legs[-1].append(int(step.target))
        elif step.kind == "nav_delivery" and step.target is not None:
            target = int(step.target)
            if target == depot:
                continue
            order.append(target)
            legs.append([])
    # The solver may plan only a prefix; only a full tour is comparable.
    if sorted(order) != sorted(int(o) for o in owned):
        return None
    legs = legs[: len(order) + 1]
    while len(legs) < len(order) + 1:
        legs.append([])
    return order, [tuple(leg) for leg in legs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="EVRoutingEnv/config_files/config_joint_preassigned.yaml"
    )
    parser.add_argument("--customers", type=int, default=3)
    parser.add_argument("--trucks", type=int, default=2)
    parser.add_argument("--chargers", type=int, default=3)
    parser.add_argument("--scenarios", type=int, default=8)
    parser.add_argument("--safety", type=float, default=1.10)
    parser.add_argument(
        "--output",
        default="results/canonical/exact_validation/per_truck_gurobi.json",
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

            # The per-truck model is single-vehicle: it plans over the whole
            # delivery sequence, so trucks sharing a sequence share a plan and
            # only distinct sequences are worth solving.
            seen_sequences = set()
            for truck in env.trucks:
                sequence_key = tuple(int(n) for n in truck.delivery_sequence)
                if not sequence_key or sequence_key in seen_sequences:
                    continue
                seen_sequences.add(sequence_key)

                # delivery_sequence is [depot, customer, ...]; the tour returns
                # to the depot, which the model adds itself.
                start = int(sequence_key[0])
                owned = [int(n) for n in sequence_key[1:]]
                if not owned:
                    continue

                policy = OptimalVRPSingleTruckPolicy(verbose=False)
                policy.energy_safety_factor = arguments.safety
                env.active_truck_id = truck.truck_id
                try:
                    plan = policy._solve_truck(truck=truck, env=env)
                except Exception as exc:  # solver refused this instance
                    records.append(
                        {"scenario_seed": int(seed), "truck": int(truck.truck_id),
                         "status": f"solver_error: {type(exc).__name__}",
                         "detail": str(exc)[:200]}
                    )
                    continue

                structure = _structure_from_plan(plan, start, owned)

                # Enumeration must be able to express the MILP's own plan,
                # otherwise a ratio below 1 would only mean the candidate set
                # was too small. Take the nearest chargers plus every charger
                # the MILP actually used.
                tour = [start, *owned, start]
                nearest = sorted(
                    env.charging_nodes,
                    key=lambda node: min(
                        _travel_hours(env, stop, int(node)) for stop in tour
                    ),
                )[: arguments.chargers]
                used = [c for leg in (structure[1] if structure else []) for c in leg]
                candidates = sorted({int(c) for c in nearest} | {int(c) for c in used})

                enumerated, _ = _enumerate_best(
                    env, curve, capacity, owned, candidates, start, arguments.safety
                )

                if structure is None:
                    records.append(
                        {"scenario_seed": int(seed), "truck": int(truck.truck_id),
                         "status": "partial_plan", "enumerated": enumerated,
                         "planned": [s.target for s in plan if s.kind != "charge"]}
                    )
                    continue

                order, legs = structure
                solved = _price_structure(
                    env, curve, capacity, order, legs, start, arguments.safety
                )
                record = {
                    "scenario_seed": int(seed),
                    "truck": int(truck.truck_id),
                    "stops": len(owned),
                    "milp_plan_cost": solved,
                    "enumerated_best": enumerated,
                    "status": "ok",
                }
                if solved is not None and math.isfinite(enumerated) and enumerated > 0:
                    record["ratio"] = solved / enumerated
                    record["matches_optimum"] = bool(solved <= enumerated + 1e-6)
                elif solved is None:
                    record["status"] = "milp_plan_infeasible_under_simulator"
                records.append(record)
    finally:
        env.close()

    scored = [r for r in records if r.get("ratio") is not None]
    summary = {
        "records": len(records),
        "scored": len(scored),
        "matching_optimum": sum(1 for r in scored if r.get("matches_optimum")),
        "max_ratio": max((r["ratio"] for r in scored), default=None),
        "mean_ratio": (
            sum(r["ratio"] for r in scored) / len(scored) if scored else None
        ),
        "below_one": sum(1 for r in scored if r["ratio"] < 1 - 1e-6),
        "safety_factor": arguments.safety,
    }

    out = Path(arguments.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
