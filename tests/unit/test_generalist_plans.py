"""The plan-conditioned view (suite_v2): plan columns, withholding, the static plan cache, and the plan arms."""

import json

import numpy as np
import pytest
from generalist_harness import small_config

from algo import generalist_plans as plans
from algo.generalist_observation import (
    FEATURE_DIMS,
    PLAN_FEATURE_NAMES,
    GeneralistEnv,
    ObservationSpec,
    collate,
    feature_dims,
)
from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig
from EVRoutingEnv.baselines.heuristics.canonical_baselines import RandomFeasiblePolicy

VALIDATION_SEED = 1_000_000_003  # validation namespace: never dropped
TRAIN_SEED = 3


@pytest.fixture(autouse=True)
def clean_plan_settings(monkeypatch):
    for name in ("EVRP_PLAN_CACHE", "EVRP_PLAN_DROPOUT", "EVRP_PLAN_MODE"):
        monkeypatch.delenv(name, raising=False)
    plans._CACHES.clear()
    yield
    plans._CACHES.clear()


def _envs(component: str):
    base = GeneralistEnv(small_config(component, trucks=2, stops=4), ObservationSpec(charger_budget=4))
    planned = GeneralistEnv(small_config(component, trucks=2, stops=4), ObservationSpec(charger_budget=4, plan_conditioned=True))
    return base, planned


def test_feature_widths_grow_only_when_plan_conditioned() -> None:
    assert feature_dims() == FEATURE_DIMS == feature_dims(ObservationSpec())
    dims = feature_dims(ObservationSpec(plan_conditioned=True))
    for key, names in PLAN_FEATURE_NAMES.items():
        assert dims[key] == FEATURE_DIMS[key] + len(names)
    assert dims["truck"] == FEATURE_DIMS["truck"] and dims["charger"] == FEATURE_DIMS["charger"]


@pytest.mark.parametrize("component", ["static", "same_day"])
def test_plan_columns_extend_the_base_view_along_a_rollout(component) -> None:
    base, planned = _envs(component)
    a, _ = base.reset(seed=VALIDATION_SEED)
    b, _ = planned.reset(seed=VALIDATION_SEED)
    policy = RandomFeasiblePolicy(VALIDATION_SEED)
    on_plan = steps = 0
    for _ in range(25):
        assert np.array_equal(b.tasks[:, : FEATURE_DIMS["task"]], a.tasks)
        assert np.array_equal(b.actions[:, : FEATURE_DIMS["action"]], a.actions)
        assert np.array_equal(b.edges[..., : FEATURE_DIMS["edge"]], a.edges)
        assert np.array_equal(b.context[: FEATURE_DIMS["context"]], a.context) and b.context[-1] == 1.0
        assert np.array_equal(a.action_index, b.action_index)
        planned_tasks = b.tasks[:, FEATURE_DIMS["task"]]
        # A planned task has a planned owner edge; every on-plan candidate targets the active truck's next task.
        owners = b.edges[: len(b.trucks), len(b.trucks): len(b.trucks) + len(b.tasks), FEATURE_DIMS["edge"]]
        assert np.array_equal(planned_tasks > 0, owners.sum(axis=0) > 0)
        assert b.actions[:, FEATURE_DIMS["action"]].sum() <= 1.0
        on_plan += int(b.actions[:, FEATURE_DIMS["action"]].sum() > 0)
        steps += 1
        action = policy(base)
        a, _, terminated, truncated, _ = base.step(action)
        b, _, _, _, _ = planned.step(action)
        if terminated or truncated:
            break
    assert on_plan >= steps // 2


def test_the_plan_is_withheld_by_mode_and_by_dropout_on_train_seeds_only(monkeypatch) -> None:
    _, planned = _envs("same_day")
    monkeypatch.setenv("EVRP_PLAN_MODE", "off")
    observation, _ = planned.reset(seed=VALIDATION_SEED)
    assert observation.context[-1] == 0.0
    assert not observation.tasks[:, FEATURE_DIMS["task"]:].any() and not observation.edges[..., FEATURE_DIMS["edge"]:].any()
    monkeypatch.setenv("EVRP_PLAN_MODE", "on")
    monkeypatch.setenv("EVRP_PLAN_DROPOUT", "1.0")
    assert planned.reset(seed=TRAIN_SEED)[0].context[-1] == 0.0
    assert planned.reset(seed=VALIDATION_SEED)[0].context[-1] == 1.0  # validation always keeps its plan
    monkeypatch.setenv("EVRP_PLAN_DROPOUT", "0.3")
    draws = [plans.plan_withheld(seed, 1) for seed in range(400)]
    assert 0.2 < np.mean(draws) < 0.4 and draws == [plans.plan_withheld(seed, 1) for seed in range(400)]
    monkeypatch.setenv("EVRP_PLAN_DROPOUT", "2")
    with pytest.raises(ValueError):
        plans.plan_dropout()


def test_static_plans_come_from_the_cache_when_the_seed_is_there(tmp_path, monkeypatch) -> None:
    from EVRoutingEnv.baselines.optimization.longhaul.policy import LonghaulPlanPolicy

    config = small_config("static", trucks=2, stops=4)
    env = GeneralistEnv(config, ObservationSpec(charger_budget=4, plan_conditioned=True), problem="small")
    env.reset(seed=VALIDATION_SEED)
    computed = LonghaulPlanPolicy(plans.static_settings())
    routes = computed.planned_routes(env)
    record = computed.plan_record()
    (tmp_path / "small").mkdir()
    (tmp_path / "small" / "plans_validation.json").write_text(json.dumps({str(VALIDATION_SEED): record}))
    monkeypatch.setenv("EVRP_PLAN_CACHE", str(tmp_path))
    env.reset(seed=VALIDATION_SEED)
    env.plan_routes()
    provider = env._plan_provider
    assert provider.planner.stats["plan_cached"] is True
    slot_of_node = {int(t.node_id): slot for slot, t in enumerate(env.joint_instance.tasks)}
    assert env.plan_routes() == {tid: [slot_of_node[n] for n in nodes] for tid, nodes in routes.items()}
    assert sorted(s for route in env.plan_routes().values() for s in route) == list(range(len(env.joint_instance.tasks)))


def test_the_rolling_planner_reads_flat_request_routes() -> None:
    from algo.generalist_experts import rolling_request

    env = GeneralistEnv(small_config("same_day", trucks=2, stops=4), ObservationSpec(charger_budget=4))
    env.reset(seed=VALIDATION_SEED)
    planner = rolling_request("same_day", env)
    for _ in range(5):
        routes = planner.planned_routes(env)
        assert set(routes) == {int(t.truck_id) for t in env.trucks}
        assert all(isinstance(r, int) for route in routes.values() for r in route)
        env.step(RandomFeasiblePolicy(1)(env))


def test_a_plan_conditioned_policy_runs_and_old_configs_still_load(tmp_path) -> None:
    _, planned = _envs("static")
    observation, _ = planned.reset(seed=VALIDATION_SEED)
    config = GeneralistPolicyConfig(hidden_dim=16, encoder_layers=1, action_head_layers=1, charger_budget=4, plan_conditioned=True)
    policy = GeneralistActorCritic(config)
    positions, _, values = policy.act(collate([observation, observation]), True)
    assert positions.shape == (2,) and values.shape[0] == 2
    old = {"hidden_dim": 16, "encoder_layers": 1, "action_head_layers": 1, "nonlinear_edge_messages": True,
           "charger_budget": 32, "fleet_assignment": False, "feature_schema_version": "generalist-v1"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(old))
    loaded = GeneralistPolicyConfig.load(path)
    assert loaded.plan_conditioned is False and loaded.observation_spec == ObservationSpec(32, False, False)


def test_the_plan_arms_carry_the_plan_flags_and_need_the_plan_cache(tmp_path) -> None:
    from scripts.campaigns import generalist_graphppo as campaign
    from scripts.training.train_generalist_ppo import build_argument_parser

    assert [run["arm"] for run in campaign.arms_for("screen", "full", 1, bks="/b")] == list(campaign.SCREEN)
    bks = tmp_path / "bks"
    for name in campaign.PROBLEMS:
        (bks / name).mkdir(parents=True)
        (bks / name / "validation.json").write_text("{}")
        (bks / name / "train_demonstrations.json").write_text("[]")
    arms = ["full_bks_plan", "full_bks_retained_plan", "full_dagger_plan"]
    with pytest.raises(SystemExit, match="plan cache"):
        campaign.prepare(tmp_path / "v2", "named", "full", 1, bks_from=bks, arms=arms)
    for name in campaign.STATIC:
        for file in campaign.PLAN_FILES:
            (bks / name / file).write_text("{}")
    destination = tmp_path / "v2"
    campaign.prepare(destination, "named", "full", 1, bks_from=bks, arms=arms)
    manifest = campaign.read_manifest(destination)
    assert [run["arm"] for run in manifest["runs"]] == arms
    for name in campaign.STATIC:
        assert (destination / "bks" / name / "plans_train.json").exists()
    parser = build_argument_parser()
    for run in manifest["runs"]:
        parsed = parser.parse_args(["--references", "r", *manifest["common_flags"], *run["flags"]])
        assert parsed.plan_conditioned and parsed.plan_cache == str(destination / "bks") and parsed.plan_dropout == 0.3


def test_the_deep_plan_arm_holds_imitation_on_and_runs_on_its_own_schedule(tmp_path, capsys) -> None:
    from scripts.campaigns import generalist_graphppo as campaign
    from scripts.training.train_generalist_ppo import build_argument_parser

    bks = tmp_path / "bks"
    for name in campaign.PROBLEMS:
        (bks / name).mkdir(parents=True)
        (bks / name / "validation.json").write_text("{}")
        (bks / name / "train_demonstrations.json").write_text("[]")
    for name in campaign.STATIC:
        for file in campaign.PLAN_FILES:
            (bks / name / file).write_text("{}")
    destination = tmp_path / "v3"
    campaign.prepare(destination, "named", "full", 1, bks_from=bks, arms=["full_dagger_plan_deep"], schedule=4_000_000)
    manifest = campaign.read_manifest(destination)
    assert manifest["schedule_timesteps"] == 4_000_000
    (run,) = manifest["runs"]
    parsed = build_argument_parser().parse_args(["--references", "r", *manifest["common_flags"], *run["flags"]])
    assert parsed.bc_coefficient == parsed.bc_coefficient_final == 0.1
    assert parsed.dagger_beta == 0.5 and parsed.dagger_beta_final == 0.2 and parsed.dagger_beta_anneal_timesteps == 4_000_000
    assert len(parsed.dagger_expert) == 5 and parsed.dagger_pretrain_rounds == 2
    assert parsed.encoder_layers == 4 and parsed.validation_scenarios == 64 and parsed.plan_conditioned
    (destination / "references").mkdir(exist_ok=True)
    (destination / "references" / "summary.json").write_text("{}")
    campaign.submit(destination, "train", 4_000_000, dry_run=True, time_limit=36)
    assert "--time=36:00:00" in capsys.readouterr().out
    campaign.submit(destination, "train", 4_000_000, dry_run=True)
    assert "--time=24:00:00" in capsys.readouterr().out
