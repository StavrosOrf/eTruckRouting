"""End-to-end checks of the exact branch-and-price baseline (needs Julia + Gurobi)."""

import os
import shutil

import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from copy import deepcopy

from EVRoutingEnv.baselines.optimization.branch_and_price.brute_force import exact_optimum
from EVRoutingEnv.baselines.optimization.branch_and_price.export import export_nominal_instance
from EVRoutingEnv.baselines.optimization.branch_and_price.julia_bridge import BPParameters, solve_bp_plan
from EVRoutingEnv.baselines.optimization.branch_and_price.policy import BranchAndPricePolicy
from EVRoutingEnv.baselines.optimization.branch_and_price.validate import nominal_replay
from EVRoutingEnv.evaluation.metrics import extract_operational_metrics
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


def _toolchain_available() -> bool:
    if shutil.which("julia") is None and not os.path.exists("/etc/profile.d/modules.sh"):
        return False
    return bool(os.environ.get("GRB_LICENSE_FILE")) or os.path.exists(os.path.expanduser("~/gurobi.lic"))


pytestmark = pytest.mark.skipif(not _toolchain_available(), reason="Julia/Gurobi toolchain unavailable")


def _env(customers=5, trucks=2, nominal=False):
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["environment"].update(
        {"num_stops": customers, "num_trucks": trucks, "allow_variable_num_stops": False}
    )
    if nominal:
        config["traffic"].update({"std_dev_factor": 0.0, "energy_uncertainty_factor": 0.0})
        config["delivery"]["enable_stochastic_unloading"] = False
    return EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)


def _run_episode(env, policy):
    policy.reset()
    for _ in range(env.max_episode_steps + 1):
        if env._episode_ended:
            break
        env.step(policy(env))
    return extract_operational_metrics(env)


@pytest.fixture(scope="module")
def solved():
    env = _env()
    env.reset(seed=4)
    solution = solve_bp_plan(env, BPParameters(time_limit_seconds=300))
    return env, solution


def test_solver_proves_the_brute_force_optimum(solved) -> None:
    env, solution = solved
    assert solution.proven_optimal
    assert solution.objective_hours == pytest.approx(exact_optimum(export_nominal_instance(env)), abs=1e-6)
    assert solution.best_bound_hours == pytest.approx(solution.objective_hours, abs=1e-9)


def test_plan_replays_on_the_environment(solved) -> None:
    env, solution = solved
    replay = nominal_replay(env, solution.routes)
    assert replay["feasible"], replay["reason"]
    assert replay["total_travel_hours"] == pytest.approx(solution.objective_hours, abs=1e-6)


def test_policy_executes_the_plan_without_fallback_when_nominal(solved) -> None:
    _, solution = solved
    env = _env(nominal=True)
    env.reset(seed=4)
    policy = BranchAndPricePolicy(BPParameters(time_limit_seconds=300))
    metrics = _run_episode(env, policy)
    assert metrics.success
    assert policy.diagnostics()["plan_fallbacks"] == 0
    assert metrics.total_travel_time == pytest.approx(solution.objective_hours, abs=1e-6)


def test_policy_completes_a_stochastic_episode() -> None:
    env = _env()
    env.reset(seed=4)
    policy = BranchAndPricePolicy(BPParameters(time_limit_seconds=300))
    _run_episode(env, policy)
    assert env._episode_ended
    assert policy.diagnostics()["proven_optimal"]
