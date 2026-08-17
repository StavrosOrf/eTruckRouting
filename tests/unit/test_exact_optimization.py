"""Properties of the nominal CP-SAT planner used as the optimization baseline."""

import math
import os

import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from copy import deepcopy

from EVRoutingEnv.baselines.exact_optimization import (
    ExactPlannerParameters,
    MathematicalProgrammingPolicy,
    solve_nominal_plan,
)
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


def _deterministic_config(customers: int = 5, trucks: int = 2) -> dict:
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["environment"].update(
        {"num_stops": customers, "num_trucks": trucks, "allow_variable_num_stops": False}
    )
    config["traffic"].update(
        {
            "enable_traffic": False,
            "enable_energy_uncertainty": False,
            "std_dev_factor": 0.0,
        }
    )
    config["delivery"]["enable_stochastic_unloading"] = False
    return config


def test_a_truck_may_be_left_idle_when_that_is_cheaper() -> None:
    """The circuit must admit an empty route.

    Forcing every truck through the depot circuit forces it to serve at least
    one customer.  Under the total-time objective the optimum frequently uses
    fewer vehicles than are available, and the planner reported those strictly
    worse plans as OPTIMAL until the depot gained a self-loop.
    """
    env = EventDrivenTruckEnv(
        _deterministic_config(customers=4, trucks=3),
        verbose=False,
        enable_plotting=False,
    )
    try:
        idle_seen = False
        for seed in (1_000_000_000, 1_000_000_001, 1_000_000_002, 1_000_000_003):
            env.reset(seed=seed)
            plan = solve_nominal_plan(
                env,
                ExactPlannerParameters(
                    time_limit_seconds=20.0, workers=2, objective="total_time"
                ),
            )
            assert plan.status in ("OPTIMAL", "FEASIBLE")
            served = [node for route in plan.routes.values() for node in route]
            assert len(served) == len(set(served))
            if any(not route for route in plan.routes.values()):
                idle_seen = True
        assert idle_seen, "no plan ever left a truck idle; the circuit is over-constrained"
    finally:
        env.close()


def test_every_customer_is_planned_exactly_once() -> None:
    env = EventDrivenTruckEnv(
        _deterministic_config(), verbose=False, enable_plotting=False
    )
    try:
        env.reset(seed=1_000_000_005)
        plan = solve_nominal_plan(
            env, ExactPlannerParameters(time_limit_seconds=20.0, workers=2)
        )
        planned = sorted(node for route in plan.routes.values() for node in route)
        expected = sorted(int(task.node_id) for task in env.task_registry.tasks())
        assert planned == expected
    finally:
        env.close()


def test_diagnostics_report_a_gap_only_when_a_bound_exists() -> None:
    env = EventDrivenTruckEnv(
        _deterministic_config(), verbose=False, enable_plotting=False
    )
    try:
        policy = MathematicalProgrammingPolicy(
            ExactPlannerParameters(time_limit_seconds=20.0, workers=2)
        )
        assert policy.diagnostics()["status"] == "NOT_RUN"

        observation, info = env.reset(seed=1_000_000_006)
        policy(env, observation, info)
        diagnostics = policy.diagnostics()

        assert diagnostics["solver"] == "cpsat"
        assert diagnostics["solves"] == 1
        assert diagnostics["decisions"] >= 1
        if diagnostics["best_bound_hours"] is None:
            assert diagnostics["relative_gap"] is None
        else:
            assert diagnostics["relative_gap"] >= -1e-9
        if diagnostics["proven_optimal"]:
            assert math.isclose(
                diagnostics["objective_hours"],
                diagnostics["best_bound_hours"],
                rel_tol=1e-6,
                abs_tol=1e-3,
            )
    finally:
        env.close()


@pytest.mark.integration
def test_planner_matches_exhaustive_enumeration_on_a_tiny_instance() -> None:
    """The model itself is validated, not just the solver's status string."""
    from scripts.evaluation.validate_exact_objective import (
        _TIME_SCALE,
        enumerate_optimum,
    )

    customers, trucks = 4, 2
    env = EventDrivenTruckEnv(
        _deterministic_config(customers, trucks), verbose=False, enable_plotting=False
    )
    try:
        parameters = ExactPlannerParameters(
            time_limit_seconds=30.0, workers=2, objective="total_time"
        )
        tolerance = (2 * customers + trucks) * 0.5 / _TIME_SCALE
        for seed in (1_000_000_000, 1_000_000_001, 1_000_000_002):
            env.reset(seed=seed)
            optimum, _ = enumerate_optimum(
                env, parameters.average_charging_power_kw, "total_time"
            )
            plan = solve_nominal_plan(env, parameters)
            assert plan.status == "OPTIMAL"
            assert abs(plan.objective_hours - optimum) <= tolerance
    finally:
        env.close()
