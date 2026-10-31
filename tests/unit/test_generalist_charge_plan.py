"""The charging plan (suite_v4): the planner's moves to the active truck's next stop in the plan input."""

from types import SimpleNamespace

import numpy as np
import pytest
from generalist_harness import small_config

from algo import generalist_plans as plans
from algo.generalist_observation import (
    CHARGE_PLAN_FEATURE_NAMES,
    FEATURE_DIMS,
    PLAN_FEATURE_NAMES,
    GeneralistEnv,
    ObservationSpec,
    feature_dims,
)
from algo.generalist_policy import GeneralistPolicyConfig
from EVRoutingEnv.state.feasibility import ActionKind, joint_action_feasibility

VALIDATION_SEED = 1_000_000_003
PLANNED = FEATURE_DIMS["action"] + len(PLAN_FEATURE_NAMES["action"])  # first charging-plan action column


@pytest.fixture(autouse=True)
def clean_plan_settings(monkeypatch):
    for name in ("EVRP_PLAN_CACHE", "EVRP_PLAN_DROPOUT", "EVRP_PLAN_MODE"):
        monkeypatch.delenv(name, raising=False)
    plans._CACHES.clear()
    yield
    plans._CACHES.clear()


def _env(component: str, charge_plan: bool = True, **sizes) -> GeneralistEnv:
    spec = ObservationSpec(charger_budget=4, plan_conditioned=True, charge_plan=charge_plan)
    return GeneralistEnv(small_config(component, **{"trucks": 2, "stops": 4, **sizes}), spec, problem="small")


def test_the_charging_plan_columns_follow_the_plan_columns_and_need_the_plan() -> None:
    dims = feature_dims(ObservationSpec(plan_conditioned=True, charge_plan=True))
    plan_only = feature_dims(ObservationSpec(plan_conditioned=True))
    for key in dims:
        assert dims[key] == plan_only[key] + len(CHARGE_PLAN_FEATURE_NAMES.get(key, ()))
    with pytest.raises(ValueError, match="plan_conditioned"):
        ObservationSpec(charge_plan=True)
    config = GeneralistPolicyConfig(charger_budget=4, plan_conditioned=True, charge_plan=True)
    assert config.observation_spec.charge_plan
    assert GeneralistPolicyConfig(plan_conditioned=True).observation_spec.charge_plan is False


def test_the_retirement_guard_hides_the_depot_from_a_truck_with_planned_customers(monkeypatch) -> None:
    monkeypatch.setenv("EVRP_SAFETY_MASK", "on")
    monkeypatch.setenv("EVRP_RETIRE_GUARD", "on")
    env = _env("static")
    observation, _ = env.reset(seed=VALIDATION_SEED)
    hidden = 0
    while not env._episode_ended and env.active_truck_id is not None:
        decisions = env.last_action_feasibility
        depots = {i for i, decision in enumerate(decisions) if decision.feasible and decision.action_kind is ActionKind.DEPOT}
        offered = {int(action) for action in observation.action_index}
        if env.plan_routes().get(int(env.active_truck_id)) and offered - depots:
            assert not depots & offered
            hidden += len(depots)
        observation, _, terminated, truncated, _ = env.step(env.plan_moves()[0][0])
        if terminated or truncated:
            break
    assert hidden > 0
    assert env.termination_reason == "success"


def test_following_the_planned_moves_plays_the_planners_episode() -> None:
    """Each decision flags at most one candidate, the planner's next move; playing it serves everyone.

    The view keeps every planned station (here beyond a 4-charger budget), so
    it offers what the plan-only view offers and possibly more.
    """
    plain, charged = _env("static", charge_plan=False), _env("static")
    a, _ = plain.reset(seed=VALIDATION_SEED)
    b, _ = charged.reset(seed=VALIDATION_SEED)
    flagged = steps = 0
    while True:
        assert set(a.action_index.tolist()) <= set(b.action_index.tolist())
        assert b.context[-1] == 1.0 and b.actions[:, PLANNED].sum() <= 1.0
        moves = charged.plan_moves()
        planned_stations = {action for action, _ in moves if action < charged.num_charging_nodes}
        feasible = joint_action_feasibility(charged)
        assert {s for s in planned_stations if feasible[s].feasible} <= set(b.action_index.tolist())
        hits = np.flatnonzero(b.actions[:, PLANNED] > 0.5)
        if hits.size:
            flagged += 1
            assert int(b.action_index[hits[0]]) == moves[0][0]
            action = int(b.action_index[hits[0]])
        else:
            action = int(b.action_index[0])
        stations = b.actions[:, PLANNED + 1] > 0.5
        assert np.all(b.actions[~stations, PLANNED + 2] == 0.0)
        assert {int(x) for x in b.action_index[stations]} <= planned_stations
        steps += 1
        a, _, terminated, truncated, _ = plain.step(action)
        b, _, _, _, _ = charged.step(action)
        if terminated or truncated:
            break
    assert flagged >= 0.9 * steps
    assert charged.termination_reason == "success"


def test_the_first_planned_move_is_the_planners_own_choice() -> None:
    from EVRoutingEnv.baselines.optimization.longhaul.policy import LonghaulPlanPolicy

    env = _env("static")
    observation, _ = env.reset(seed=VALIDATION_SEED)
    planner = LonghaulPlanPolicy(plans.static_settings())
    steps = 0
    while env.active_truck_id is not None and not env._episode_ended:
        moves = env.plan_moves()
        assert moves and planner(env) == moves[0][0]
        assert env.plan_moves() == moves  # solved once per decision
        observation, _, terminated, truncated, _ = env.step(moves[0][0])
        steps += 1
        if terminated or truncated:
            break
    assert steps > 5 and env.termination_reason == "success"


def test_dynamic_problems_have_no_charging_plan() -> None:
    env = _env("same_day")
    observation, _ = env.reset(seed=VALIDATION_SEED)
    assert env.plan_moves() is None
    assert observation.context[-1] == 0.0 and not observation.actions[:, PLANNED:].any()


def test_a_withheld_plan_withholds_the_charging_plan(monkeypatch) -> None:
    monkeypatch.setenv("EVRP_PLAN_MODE", "off")
    env = _env("static")
    observation, _ = env.reset(seed=VALIDATION_SEED)
    assert observation.context[-2:].tolist() == [0.0, 0.0] and not observation.actions[:, FEATURE_DIMS["action"]:].any()


def test_a_search_copy_keeps_the_planned_moves() -> None:
    from algo import generalist_search as search

    env = _env("static")
    observation, _ = env.reset(seed=VALIDATION_SEED)
    for _ in range(3):
        observation, *_ = env.step(env.plan_moves()[0][0])
    shadow = search.fresh_clone(env, 7)
    assert shadow.plan_moves() == env.plan_moves()
    assert np.array_equal(shadow._get_observation().actions, observation.actions)


def test_the_charge_plan_arm_is_full_dagger_plan_plus_the_charging_plan(tmp_path) -> None:
    from scripts.campaigns import generalist_graphppo as campaign
    from scripts.training.train_generalist_ppo import build_argument_parser

    assert campaign.RECIPES["full_dagger_plan_charge"] == [*campaign.RECIPES["full_dagger_plan"], "--charge-plan"]
    bks = tmp_path / "bks"
    for name in campaign.PROBLEMS:
        (bks / name).mkdir(parents=True)
        (bks / name / "validation.json").write_text("{}")
        (bks / name / "train_demonstrations.json").write_text("[]")
    for name in campaign.STATIC:
        for file in campaign.PLAN_FILES:
            (bks / name / file).write_text("{}")
    destination = tmp_path / "v4"
    campaign.prepare(destination, "named", "full", 1, bks_from=bks, arms=["full_dagger_plan_charge"])
    (run,) = campaign.read_manifest(destination)["runs"]
    parser = build_argument_parser()
    parsed = parser.parse_args(["--references", "r", *campaign.read_manifest(destination)["common_flags"], *run["flags"]])
    assert parsed.charge_plan and parsed.plan_conditioned
