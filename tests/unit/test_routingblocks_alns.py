"""The RoutingBlocks baseline: exact route pricing, optimality on small cases, and replay.

The solver runs in its own interpreter (``scripts/setup_rb_env.sh``); these tests
skip when it is not installed.
"""

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.baselines.optimization.branch_and_price.brute_force import exact_optimum
from EVRoutingEnv.baselines.optimization.branch_and_price.export import export_nominal_instance
from EVRoutingEnv.baselines.optimization.branch_and_price.validate import nominal_replay
from EVRoutingEnv.baselines.optimization.routingblocks_alns import bridge
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config

ROOT = Path(__file__).resolve().parents[2]

try:
    SOLVER = bridge.solver_python()
except FileNotFoundError:
    SOLVER = None

pytestmark = pytest.mark.skipif(SOLVER is None, reason="solver interpreter (.venv-rb) not installed")


def make_env(config_path: str, trucks: int, customers: int) -> EventDrivenTruckEnv:
    config = deepcopy(load_config(config_path))
    config["environment"].update(
        {"num_stops": customers, "num_trucks": trucks, "allow_variable_num_stops": False})
    return EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)


@pytest.mark.parametrize("config_path,seed", [
    ("EVRoutingEnv/config_files/config_joint.yaml", 1),
    ("EVRoutingEnv/config_files/config_joint_time_windows.yaml", 2),
])
def test_route_dp_matches_brute_force_on_every_subset(tmp_path, config_path, seed) -> None:
    env = make_env(config_path, 2, 5)
    env.reset(seed=seed)
    path = tmp_path / "instance.json"
    path.write_text(json.dumps(export_nominal_instance(env)))
    environment = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1")
    done = subprocess.run([str(SOLVER), str(ROOT / "tests/unit/rb_dp_check.py"), str(path), "5"],
                          capture_output=True, text=True, env=environment, check=True)
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["subsets"] == 31
    assert result["mismatches"] == []


@pytest.mark.parametrize("config_path,trucks,customers,seed", [
    ("EVRoutingEnv/config_files/config_joint.yaml", 3, 6, 1),
    ("EVRoutingEnv/config_files/config_joint_time_windows.yaml", 2, 6, 0),
])
def test_plan_reaches_optimum_and_replays_in_the_environment(config_path, trucks, customers, seed) -> None:
    env = make_env(config_path, trucks, customers)
    env.reset(seed=seed)
    optimum = exact_optimum(export_nominal_instance(env))
    plan = bridge.solve_rb_plan(env, bridge.RBParameters(time_limit_seconds=20.0))
    assert plan.status == "HEURISTIC" and not plan.proven_optimal
    assert plan.objective_hours == pytest.approx(optimum, abs=1e-6)

    env.reset(seed=seed)
    replay = nominal_replay(env, plan.routes)
    assert replay["feasible"], replay["reason"]
    assert replay["total_travel_hours"] == pytest.approx(plan.objective_hours, abs=1e-6)


def test_parallel_workers_return_a_valid_plan() -> None:
    env = make_env("EVRoutingEnv/config_files/config_joint.yaml", 3, 6)
    env.reset(seed=1)
    plan = bridge.solve_rb_plan(env, bridge.RBParameters(time_limit_seconds=15.0, workers=2,
                                                         extra={"pool_interval_seconds": 5}))
    env.reset(seed=1)
    replay = nominal_replay(env, plan.routes)
    assert replay["feasible"], replay["reason"]
    assert replay["total_travel_hours"] == pytest.approx(plan.objective_hours, abs=1e-6)
    assert plan.certificate["search"]["workers"] == 2
