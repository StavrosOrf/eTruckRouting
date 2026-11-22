"""A policy with a learned plan margin, and the staged training around it."""

import json

import numpy as np
import pytest
import torch
import yaml
from generalist_harness import small_config

from algo import generalist_plans as plans
from algo.canonical_ppo import PPOConfig
from algo.generalist_observation import GeneralistEnv, ObservationSpec, collate
from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig
from algo.generalist_ppo import (
    GeneralistPPO,
    ReferenceTable,
    SuiteProblem,
    SuiteReward,
    evaluate_policy_suite,
    planner_baseline,
    problem_category,
    run_reference_episodes,
    summarize_suite,
)

VALIDATION_SEED = 1_000_000_003  # validation namespace: the plan is never dropped
SMALL = dict(hidden_dim=16, encoder_layers=1, action_head_layers=1, charger_budget=4, plan_conditioned=True, charge_plan=True)


@pytest.fixture(autouse=True)
def plan_settings(monkeypatch):
    for name in ("EVRP_PLAN_CACHE", "EVRP_PLAN_DROPOUT", "EVRP_PLAN_GATE", "EVRP_PLAN_MARK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EVRP_PLAN_MODE", "on")
    plans._CACHES.clear()
    yield
    plans._CACHES.clear()


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    folder = tmp_path_factory.mktemp("plan_default")
    problems = []
    for component in ("static", "same_day"):
        path = folder / f"{component}.yaml"
        path.write_text(yaml.safe_dump(small_config(component, trucks=2, stops=4)))
        problems.append(SuiteProblem(component, str(path)))
    rows = {problem.name: {row["seed"]: row for row in run_reference_episodes(problem, range(4))} for problem in problems}
    return problems, rows


def _observation(component: str = "static"):
    env = GeneralistEnv(small_config(component, trucks=2, stops=4), ObservationSpec(4, False, True, True))
    observation, _ = env.reset(seed=VALIDATION_SEED)
    return observation


def test_marking_is_on_for_a_learned_margin_and_under_the_gate(monkeypatch) -> None:
    assert not plans.plan_marked()
    assert _observation().plan_position == -1
    monkeypatch.setenv("EVRP_PLAN_MARK", "on")
    assert plans.plan_marked() and plans.plan_gate() is None
    assert _observation().plan_position >= 0
    monkeypatch.setenv("EVRP_PLAN_MARK", "off")
    monkeypatch.setenv("EVRP_PLAN_GATE", "2")
    assert plans.plan_marked()


def test_the_margin_raises_only_the_planners_candidate(monkeypatch) -> None:
    monkeypatch.setenv("EVRP_PLAN_MARK", "on")
    observation = _observation()
    plan = observation.plan_position
    torch.manual_seed(0)
    plain = GeneralistActorCritic(GeneralistPolicyConfig(**SMALL))
    marginal = GeneralistActorCritic(GeneralistPolicyConfig(**SMALL, plan_default=True, plan_margin_init=3.0))
    marginal.adopt(plain.state_dict())  # a checkpoint without a margin: the head keeps its start
    unmarked = collate([observation])
    unmarked = type(unmarked)(**{**unmarked.__dict__, "plan_position": torch.tensor([-1])})
    batch = collate([observation])
    assert int(batch.plan_position[0]) == plan
    with torch.no_grad():
        base = plain(batch).logits
        raised = marginal(batch).logits
        same = marginal(unmarked).logits
    offset = torch.zeros_like(base)
    offset[0, plan] = 3.0
    assert torch.allclose(raised, base + offset, atol=1e-5)
    assert torch.allclose(same, base, atol=1e-5)  # nothing marked, nothing raised
    margin, marked = marginal.plan_margins(batch)
    assert marked.tolist() == [True] and margin.tolist() == pytest.approx([3.0])
    assert marginal.plan_margins(unmarked)[1].tolist() == [False]
    with pytest.raises(RuntimeError):
        plain.plan_margins(batch)


def test_a_large_margin_is_the_planner_and_the_margin_is_trainable(monkeypatch) -> None:
    monkeypatch.setenv("EVRP_PLAN_MARK", "on")
    observation = _observation()
    torch.manual_seed(1)
    policy = GeneralistActorCritic(GeneralistPolicyConfig(**SMALL, plan_default=True, plan_margin_init=30.0))
    batch = collate([observation, observation])
    assert policy.act(batch, deterministic=True)[0].tolist() == [observation.plan_position] * 2
    # The gradient of a deviation's log probability reaches the margin head and lowers the margin.
    other = next(p for p in range(observation.num_actions) if p != observation.plan_position)
    logprob, _, _ = policy.evaluate_actions(batch, torch.tensor([other, other]))
    gradient = torch.autograd.grad(logprob.sum(), policy.margin_head[-1].bias)[0]
    assert float(gradient) < 0.0
    with pytest.raises(ValueError):
        GeneralistPolicyConfig(plan_default=True)  # needs the plan in the observation
    saved = GeneralistPolicyConfig(**SMALL, plan_default=True)
    assert GeneralistPolicyConfig(**json.loads(json.dumps(saved.__dict__))) == saved


def _trainer(suite, monkeypatch, **overrides):
    monkeypatch.setenv("EVRP_PLAN_MARK", "on")
    problems, rows = suite
    settings = dict(total_timesteps=4 * 4 * 8, num_envs=4, rollout_steps=8, minibatch_size=16, epochs=2, seed=5,
                    entropy_coefficient=0.0, normalize_reward=False, gate_learning_rate=1e-2, critic_learning_rate=1e-3,
                    learning_rate=1e-4)
    settings.update(overrides)
    baseline = settings.pop("baseline", None)
    policy_config = GeneralistPolicyConfig(**SMALL, plan_default=True, plan_margin_init=2.0, critic_heads="family")
    reward = SuiteReward(reference="scenario")
    return GeneralistPPO(
        policy_config, PPOConfig(**settings), problems, ReferenceTable(rows), reward, [list(range(4))] * len(problems),
        baseline=baseline,
    )


def test_learning_rates_are_set_per_group_and_annealed_together(suite, monkeypatch) -> None:
    trainer = _trainer(suite, monkeypatch)
    try:
        rates = sorted(group["base_lr"] for group in trainer.optimizer.param_groups)
        assert rates == [1e-4, 1e-3, 1e-2]
        held = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        assert held == {id(p) for p in trainer.policy.parameters()}
        trainer.learn(lambda record: record["update"] == 2)
        for group in trainer.optimizer.param_groups:  # the second update ran at 3/4 of each base rate
            assert group["lr"] == pytest.approx(0.75 * group["base_lr"])
    finally:
        trainer.close()


def test_the_critic_warms_up_before_any_other_weight_moves(suite, monkeypatch) -> None:
    trainer = _trainer(suite, monkeypatch, critic_warmup_updates=1)
    try:
        before = {name: value.clone() for name, value in trainer.policy.state_dict().items()}
        critic = lambda name: name.startswith(("value_head.", "value_heads."))  # noqa: E731
        statistics = lambda name: "normalizer" in name  # noqa: E731 - running input statistics, not weights
        trainer.learn(lambda record: record["update"] == 1)
        after = trainer.policy.state_dict()
        moved = {name for name in before if not torch.equal(before[name], after[name]) and not statistics(name)}
        assert moved and all(critic(name) for name in moved)
        assert trainer.history[0]["critic_warmup"] == 1.0 and trainer.history[0]["approx_kl"] == pytest.approx(0.0, abs=1e-6)
        trainer.learn(lambda record: record["update"] == 2)
        final = trainer.policy.state_dict()
        assert not torch.equal(final["margin_head.2.bias"], before["margin_head.2.bias"])
        assert trainer.history[1]["critic_warmup"] == 0.0
        assert {"deviation_rate", "plan_marked"} <= set(trainer.history[1])
        assert any(key.startswith("margin/") for key in trainer.history[1])
    finally:
        trainer.close()


def test_deviations_are_charged_and_the_planners_score_is_taken_off(suite, monkeypatch) -> None:
    problems, rows = suite
    reward = SuiteReward(reference="scenario")
    table = ReferenceTable(rows)
    baseline = {name: {seed: reward.score(row, table.scale(name, seed, "scenario")[0]) for seed, row in values.items()}
                for name, values in rows.items()}
    plain = _trainer(suite, monkeypatch, rollout_steps=64, total_timesteps=4 * 64)
    charged = _trainer(suite, monkeypatch, rollout_steps=64, total_timesteps=4 * 64, deviation_penalty=0.25, baseline=baseline)
    try:
        charged.policy.load_state_dict(plain.policy.state_dict())
        for trainer in (plain, charged):
            trainer._observations = trainer.envs.reset()
            trainer._observe(trainer._observations)
        first, second = plain._collect_rollout(), charged._collect_rollout()
        assert torch.equal(first["actions"], second["actions"])  # same policy, same noise, same decisions
        plan = torch.tensor([observation.plan_position for observation in first["observations"]])
        deviated = (plan >= 0) & (first["actions"].cpu() != plan)
        assert deviated.any() and not deviated.all()
        assert second["statistics"]["deviation_rate"] == pytest.approx(float(deviated.sum() / (plan >= 0).sum()))
        assert second["statistics"]["baseline_misses"] == 0.0
        assert "deviation_rate" in first["statistics"] and "baseline_misses" not in first["statistics"]
        # The rewards differ by the charge on every deviation and, where an episode ended, the planner's score.
        finished = [record for window in charged.episodes.values() for record in window]
        assert finished
        expected = 0.25 * float(deviated.sum()) + sum(baseline[record["problem"]][record["seed"]] for record in finished)
        assert float(first["rewards"].sum() - second["rewards"].sum()) == pytest.approx(expected, rel=1e-4, abs=1e-3)
    finally:
        plain.close()
        charged.close()


def test_the_planner_baseline_scores_stored_episodes_in_the_rewards_units(suite, tmp_path) -> None:
    problems, rows = suite
    reward = SuiteReward(reference="scenario")
    table = ReferenceTable(rows)
    for name, values in rows.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(list(values.values())))
    scores = planner_baseline(tmp_path, problems, table, reward)
    for name, values in rows.items():
        assert set(scores[name]) == set(values)
        for seed, row in values.items():
            # A reference scored against itself: the hours cancel and the bonus terms remain.
            assert scores[name][seed] == pytest.approx(reward.score(row, table.scale(name, seed, "scenario")[0]))
    with pytest.raises(FileNotFoundError):
        planner_baseline(tmp_path / "missing", problems, table, reward)


def test_evaluation_reports_how_often_a_learned_margin_left_the_plan(suite, monkeypatch) -> None:
    monkeypatch.setenv("EVRP_PLAN_MARK", "on")
    problems, rows = suite
    torch.manual_seed(2)
    follower = GeneralistActorCritic(GeneralistPolicyConfig(**SMALL, plan_default=True, plan_margin_init=40.0))
    found = evaluate_policy_suite(follower, problems, [[0, 1]] * len(problems), ReferenceTable(rows), SuiteReward(), envs_per_problem=2)
    assert len(found) == 2 * len(problems)
    for row in found:
        counts = row["plan_default"]
        assert counts["decisions"] > 0 and counts["deviated"] == 0 and counts["marked"] <= counts["decisions"]
    summary = summarize_suite(found, [problem.name for problem in problems])
    assert all(summary[problem.name]["deviated"] == 0.0 for problem in problems)
    plain = GeneralistActorCritic(GeneralistPolicyConfig(**SMALL))
    assert all("plan_default" not in row for row in evaluate_policy_suite(
        plain, problems, [[0]] * len(problems), ReferenceTable(rows), SuiteReward(), envs_per_problem=1))


def test_problem_categories_follow_the_names() -> None:
    assert [problem_category(name) for name in (
        "tx_longhaul_t8", "tx_longhaul_congested", "pnw_longhaul_hos_t13", "ca_sameday_t16", "ca_drayage_combined",
    )] == ["lh_plain", "lh_congested", "lh_hos", "sameday", "drayage"]
    assert np.isfinite(PPOConfig(deviation_penalty=0.001, critic_warmup_updates=3).deviation_penalty)
    for bad in (dict(deviation_penalty=-1.0), dict(critic_warmup_updates=-1), dict(gate_learning_rate=0.0)):
        with pytest.raises(ValueError):
            PPOConfig(**bad)
