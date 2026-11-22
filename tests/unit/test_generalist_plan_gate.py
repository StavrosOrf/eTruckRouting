"""The decision-time plan gate: the planner's own action marked in the observation, and the margin rule."""

import json

import pytest
import torch
from generalist_harness import small_config

from algo import generalist_plans as plans
from algo.generalist_observation import GeneralistEnv, ObservationSpec, collate
from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig
from EVRoutingEnv.baselines.heuristics.canonical_baselines import RandomFeasiblePolicy
from scripts.campaigns import generalist_graphppo as campaign

VALIDATION_SEED = 1_000_000_003  # validation namespace: the plan is never dropped


@pytest.fixture(autouse=True)
def clean_plan_settings(monkeypatch):
    for name in ("EVRP_PLAN_CACHE", "EVRP_PLAN_DROPOUT", "EVRP_PLAN_MODE", "EVRP_PLAN_GATE"):
        monkeypatch.delenv(name, raising=False)
    plans._CACHES.clear()
    yield
    plans._CACHES.clear()


def _env(component: str, charge_plan: bool = False) -> GeneralistEnv:
    spec = ObservationSpec(charger_budget=4, plan_conditioned=True, charge_plan=charge_plan)
    return GeneralistEnv(small_config(component, trucks=2, stops=4), spec)


def test_the_gate_is_off_unless_a_margin_is_set(monkeypatch) -> None:
    assert plans.plan_gate() is None
    monkeypatch.setenv("EVRP_PLAN_GATE", "off")
    assert plans.plan_gate() is None
    monkeypatch.setenv("EVRP_PLAN_GATE", "1.5")
    assert plans.plan_gate() == 1.5
    monkeypatch.setenv("EVRP_PLAN_GATE", "inf")
    assert plans.plan_gate() == float("inf")
    for bad in ("-1", "nan"):
        monkeypatch.setenv("EVRP_PLAN_GATE", bad)
        with pytest.raises(ValueError):
            plans.plan_gate()


@pytest.mark.parametrize("component,charge_plan", [("static", True), ("same_day", False)])
def test_the_planners_action_is_marked_only_with_the_gate_on(component, charge_plan, monkeypatch) -> None:
    env = _env(component, charge_plan)
    observation, _ = env.reset(seed=VALIDATION_SEED)
    assert observation.plan_position == -1  # gate off: nothing is marked and the planner is not asked for an action
    monkeypatch.setenv("EVRP_PLAN_GATE", "inf")
    observation, _ = env.reset(seed=VALIDATION_SEED)
    fallback = RandomFeasiblePolicy(VALIDATION_SEED)
    marked = steps = 0
    for _ in range(40):
        steps += 1
        target = env.plan_action()
        if observation.plan_position >= 0:
            marked += 1
            assert int(observation.action_index[observation.plan_position]) == target
            action = target
        else:
            assert target is None  # the planner's action is always a candidate when it has one
            action = fallback(env)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert marked >= steps // 2


def test_following_the_marked_action_replays_the_rolling_planner(monkeypatch) -> None:
    """No public stations here, so nothing is forced and the marked action is the planner's own step every time."""
    from algo.generalist_experts import rolling_request

    monkeypatch.setenv("EVRP_PLAN_GATE", "inf")
    env = _env("same_day")
    observation, _ = env.reset(seed=VALIDATION_SEED)
    twin = GeneralistEnv(small_config("same_day", trucks=2, stops=4), ObservationSpec(charger_budget=4))
    twin.reset(seed=VALIDATION_SEED)
    planner = rolling_request("same_day", twin)
    steps = 0
    for _ in range(60):
        steps += 1
        expected = int(planner(twin))
        assert observation.plan_position >= 0
        assert int(observation.action_index[observation.plan_position]) == expected
        observation, _, terminated, truncated, _ = env.step(expected)
        twin.step(expected)
        if terminated or truncated:
            break
    assert steps > 5


def test_the_planners_action_needs_the_plan_of_the_same_decision() -> None:
    from algo.generalist_experts import rolling_request

    env = GeneralistEnv(small_config("same_day", trucks=2, stops=4), ObservationSpec(charger_budget=4))
    env.reset(seed=VALIDATION_SEED)
    planner = rolling_request("same_day", env)
    with pytest.raises(RuntimeError):
        planner.planned_action(env)
    planner.planned_routes(env)
    action = planner.planned_action(env)
    assert planner.planned_action(env) == action  # asking again plans nothing and changes nothing
    env.step(action)
    with pytest.raises(RuntimeError):
        planner.planned_action(env)


def test_the_gate_follows_the_planner_unless_the_policy_is_sure_enough(monkeypatch) -> None:
    monkeypatch.setenv("EVRP_PLAN_GATE", "inf")
    env = _env("static", charge_plan=True)
    observation, _ = env.reset(seed=VALIDATION_SEED)
    assert observation.plan_position >= 0
    config = GeneralistPolicyConfig(
        hidden_dim=16, encoder_layers=1, action_head_layers=1, charger_budget=4, plan_conditioned=True, charge_plan=True,
    )
    torch.manual_seed(0)
    policy = GeneralistActorCritic(config)
    batch = collate([observation, observation, observation])
    own, _, values = policy.act(batch, True)
    with torch.no_grad():
        logp = torch.log_softmax(policy(batch).logits, dim=-1)
    # A plan position that differs from the policy's own choice, one row without a plan, one outside the candidates.
    other = next(p for p in torch.nonzero(policy(batch).feasible_mask[0]).view(-1).tolist() if p != int(own[0]))
    plan = torch.tensor([other, -1, 10_000])
    gap = float(logp[0, own[0]] - logp[0, other])
    assert gap >= 0.0

    follow, logprob, gated_values, mine = policy.act_gated(batch, plan, float("inf"))
    assert torch.equal(mine, own) and torch.equal(gated_values, values)
    assert follow.tolist() == [other, int(own[1]), int(own[2])]
    assert torch.allclose(logprob, logp.gather(1, follow.view(-1, 1)).squeeze(1))
    alone = policy.act_gated(batch, plan, 0.0)[0]
    assert torch.equal(alone, own)
    assert policy.act_gated(batch, plan, gap + 1e-3)[0].tolist()[0] == other  # not sure enough: the planner's action
    if gap > 1e-3:
        assert policy.act_gated(batch, plan, gap - 1e-3)[0].tolist()[0] == int(own[0])  # sure enough: its own


def test_gated_chunks_report_the_ungated_choice(monkeypatch) -> None:
    from algo.generalist_ppo import PairBudget, act_in_chunks

    monkeypatch.setenv("EVRP_PLAN_GATE", "inf")
    env = _env("static", charge_plan=True)
    observation, _ = env.reset(seed=VALIDATION_SEED)
    config = GeneralistPolicyConfig(
        hidden_dim=16, encoder_layers=1, action_head_layers=1, charger_budget=4, plan_conditioned=True, charge_plan=True,
    )
    torch.manual_seed(1)
    policy = GeneralistActorCritic(config)
    observations = [observation] * 5
    plain = act_in_chunks(policy, observations, "cpu", True, PairBudget(None))[0]
    own: list = []
    gated = act_in_chunks(policy, observations, "cpu", True, PairBudget(None), plan_margin=float("inf"), own=own)[0]
    assert own == plain.tolist()
    assert gated.tolist() == [observation.plan_position] * 5
    untouched: list = []
    assert torch.equal(act_in_chunks(policy, observations, "cpu", True, PairBudget(None), own=untouched)[0], plain)
    assert untouched == []  # no gate, nothing reported


def test_the_gate_margin_is_frozen_with_the_tag(tmp_path, monkeypatch) -> None:
    assert campaign.decision_from(False, None) is None
    settings = campaign.decision_from(True, 0.0, retire_guard=True, public_wake=True, count=64, plan_gate=2.0)
    assert settings["plan_gate"] == 2.0
    assert campaign.decision_from(False, None, plan_gate=float("inf")) == {"plan_gate": float("inf")}
    with pytest.raises(SystemExit):
        campaign.decision_from(False, None, plan_gate=-0.5)
    campaign.write_decision(tmp_path, settings)
    assert json.loads((tmp_path / campaign.DECISION).read_text())["plan_gate"] == 2.0
    campaign.apply_decision(tmp_path)
    assert plans.plan_gate() == 2.0
    other = tmp_path / "plain"
    campaign.write_decision(other, campaign.decision_from(True, 0.0))
    campaign.apply_decision(other)
    assert plans.plan_gate() is None  # a tag without a gate switches it off again


def test_the_safety_mask_and_the_charger_budget_never_hide_the_planners_action(monkeypatch) -> None:
    monkeypatch.setenv("EVRP_PLAN_GATE", "inf")
    monkeypatch.setenv("EVRP_SAFETY_MASK", "on")
    monkeypatch.setenv("EVRP_SAFETY_MARGIN", "0")
    spec = ObservationSpec(charger_budget=1, plan_conditioned=True)
    for component in ("static", "same_day"):
        env = GeneralistEnv(small_config(component, trucks=2, stops=4), spec)
        observation, _ = env.reset(seed=VALIDATION_SEED)
        for _ in range(40):
            target = env.plan_action()
            if target is None:
                break
            assert observation.plan_position >= 0
            assert int(observation.action_index[observation.plan_position]) == target
            observation, _, terminated, truncated, _ = env.step(target)
            if terminated or truncated:
                break
