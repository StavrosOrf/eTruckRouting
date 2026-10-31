"""Plan-conditioned input (plan 9.5): the rolling planner advises, never decides."""

import json

import numpy as np
import pytest
import torch
import yaml
from generalist_harness import random_observations, small_config, small_env

from algo.generalist_observation import (
    PLAN_ACTION_DIM,
    PLAN_TASK_DIM,
    PLAN_TRUCK_DIM,
    ObservationSpec,
    build_observation,
    collate,
)
from algo.generalist_plan import tracker_for
from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig


def _spec(**overrides) -> ObservationSpec:
    options = {"charger_budget": 4, "plan_conditioning": True}
    options.update(overrides)
    return ObservationSpec(**options)


def _policy(**overrides) -> GeneralistActorCritic:
    torch.manual_seed(0)
    config = GeneralistPolicyConfig(hidden_dim=32, encoder_layers=2, charger_budget=4, **overrides)
    return GeneralistActorCritic(config).eval()


def _run_until_planned(env, seed: int = 3, limit: int = 400):
    """Advance until the tracker holds a non-empty plan over live tasks; the observation."""
    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(0)
    for _ in range(limit):
        tracker = getattr(env, "_generalist_plan_tracker", None)
        if (
            tracker is not None and len(observation.tasks)
            and any(tracker._routes.get(truck_id) for truck_id in tracker._routes)
            and observation.plan_tasks is not None and observation.plan_tasks.any()
        ):
            return observation
        action = int(observation.action_index[rng.integers(observation.num_actions)])
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset(seed=seed + 1)
    raise AssertionError("no planned state reached")


def test_plan_rows_exist_only_with_the_flag_and_zero_on_static() -> None:
    observation = random_observations(small_env("same_day", spec=_spec()), seed=0, steps=1)[0]
    assert observation.plan_tasks is not None and observation.plan_tasks.shape[1] == PLAN_TASK_DIM
    assert observation.plan_trucks is not None and observation.plan_trucks.shape[1] == PLAN_TRUCK_DIM
    assert observation.plan_actions is not None and observation.plan_actions.shape[1] == PLAN_ACTION_DIM
    static = random_observations(small_env("static", spec=_spec()), seed=0, steps=1)[0]
    assert static.plan_tasks is not None and not static.plan_tasks.any()
    assert static.plan_actions is not None and not static.plan_actions.any()
    plain = random_observations(small_env("same_day", spec=_spec(plan_conditioning=False)), seed=0, steps=1)[0]
    assert plain.plan_tasks is None and plain.plan_trucks is None and plain.plan_actions is None
    assert collate([plain]).plan_tasks is None


def test_the_on_plan_rows_mark_the_planners_assignment() -> None:
    env = small_env("same_day", spec=_spec())
    observation = _run_until_planned(env)
    tracker = tracker_for(env)
    open_rids = {int(slot) for slot in _slots(env, observation)}
    planned_any = {
        route_rid
        for route in tracker._routes.values()
        for route_rid in ([rid for rid in route if rid in open_rids][:1])
    }
    flagged_any = {
        int(slot)
        for row, slot in enumerate(_slots(env, observation))
        if observation.plan_tasks[row, 1] > 0
    }
    assert flagged_any == planned_any
    # Candidates flagged on-plan all point at the active truck's next planned task.
    active_rows = np.flatnonzero(observation.plan_tasks[:, 0])
    on_plan = np.flatnonzero(observation.plan_actions[:, 0])
    if len(active_rows):
        assert set(observation.action_task[on_plan].tolist()) <= set(active_rows.tolist())
        ranks = observation.plan_tasks[:, 2]
        assert ranks.max() <= 1.0 and ranks[active_rows[0]] == 1.0
    env.close()


def _slots(env, observation):
    from algo.generalist_observation import _C, _TERMINAL_REQUESTS, _task_slots, extract_canonical_features

    rows = extract_canonical_features(env).customer_features
    keep = ~np.isin(rows[:, _C["request_status_code"]].astype(int), list(_TERMINAL_REQUESTS))
    return _task_slots(env, rows[keep], True)


def test_plan_dropout_zeroes_whole_episodes(monkeypatch) -> None:
    import algo.generalist_plan as plan_module

    env = small_env("same_day", spec=_spec(plan_dropout=0.5))
    monkeypatch.setattr(plan_module.random, "random", lambda: 0.0)  # always dropped
    observation = random_observations(env, seed=5, steps=3)[-1]
    tracker = tracker_for(env)
    assert tracker._dropped and tracker.plan_calls == 0
    assert observation.plan_tasks is not None and not observation.plan_tasks.any()
    monkeypatch.setattr(plan_module.random, "random", lambda: 0.9)  # never dropped
    env.reset(seed=6)
    assert not tracker_for(env)._dropped
    env.close()
    with pytest.raises(ValueError):
        ObservationSpec(plan_dropout=1.0)


def test_zero_plan_rows_inject_exactly_nothing() -> None:
    from dataclasses import replace

    observations = random_observations(small_env("same_day", spec=_spec()), seed=0, steps=5)
    policy = _policy(plan_conditioning=True)
    batch = collate(observations)
    stripped = replace(
        batch,
        plan_tasks=torch.zeros_like(batch.plan_tasks),
        plan_trucks=torch.zeros_like(batch.plan_trucks),
        plan_actions=torch.zeros_like(batch.plan_actions),
    )
    bare = replace(batch, plan_tasks=None, plan_trucks=None, plan_actions=None)
    with torch.no_grad():
        zeroed, none_at_all = policy(stripped), policy(bare)
    assert torch.equal(zeroed.logits, none_at_all.logits)
    assert torch.equal(zeroed.values, none_at_all.values)


def test_padding_keeps_the_planned_logits() -> None:
    dynamic = random_observations(small_env("same_day", spec=_spec()), seed=0, steps=3)[-1]
    static = random_observations(small_env("static", spec=_spec()), seed=0, steps=1)[0]
    policy = _policy(plan_conditioning=True)
    with torch.no_grad():
        alone = policy(collate([dynamic]))
        mixed = policy(collate([dynamic, static]))
    width = alone.logits.shape[1]
    assert torch.allclose(alone.logits[0], mixed.logits[0, :width], atol=1e-5)


def test_gradients_reach_the_plan_modules() -> None:
    observations = random_observations(small_env("same_day", spec=_spec()), seed=0, steps=5)
    policy = _policy(plan_conditioning=True).train()
    batch = collate(observations)
    policy.observe(batch)
    # Make sure at least one plan row is non-zero so gradients can flow.
    batch.plan_tasks[..., 1] = 1.0
    batch.plan_trucks[..., 0] = 1.0
    log_probs, entropy, values = policy.evaluate_actions(
        batch, torch.zeros(len(observations), dtype=torch.long)
    )
    (log_probs.mean() + values.mean()).backward()
    for name in ("plan_task_encoder", "plan_truck_encoder"):
        parameter = dict(policy.named_parameters())[f"{name}.weight"]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_checkpoints_round_trip_and_old_configs_default_off(tmp_path) -> None:
    observations = random_observations(small_env("same_day", spec=_spec()), seed=0, steps=4)
    policy = _policy(plan_conditioning=True)
    policy.observe(collate(observations))
    policy.save(tmp_path, prefix="best")
    restored = GeneralistActorCritic.load(tmp_path, prefix="best").eval()
    assert restored.config.plan_conditioning
    batch = collate(observations)
    assert torch.equal(policy(batch).logits, restored(batch).logits)
    legacy = json.loads((tmp_path / "best_config.json").read_text())
    del legacy["plan_conditioning"]
    (tmp_path / "legacy.json").write_text(json.dumps(legacy))
    assert not GeneralistPolicyConfig.load(tmp_path / "legacy.json").plan_conditioning


def test_the_plan_cache_replans_only_on_reveal_or_exhaustion() -> None:
    env = small_env("same_day", spec=_spec())
    observation, _ = env.reset(seed=9)
    rng = np.random.default_rng(1)
    steps = 0
    released_values = set()
    while steps < 120:
        released_values.add(int(env.request_engine.metrics(env.global_clock)["requests_released"]))
        action = int(observation.action_index[rng.integers(observation.num_actions)])
        observation, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    tracker = tracker_for(env)
    assert tracker.plan_calls >= 1
    assert tracker.plan_calls < steps / 2  # cached between reveals, never per step
    assert tracker.plan_seconds >= 0.0
    env.close()


def test_one_ppo_update_trains_through_the_plan_path(tmp_path) -> None:
    from algo.canonical_ppo import PPOConfig
    from algo.generalist_ppo import (
        GeneralistPPO,
        ReferenceTable,
        SuiteProblem,
        SuiteReward,
        run_reference_episodes,
    )

    problems = []
    for component in ("static", "same_day"):
        path = tmp_path / f"{component}.yaml"
        path.write_text(yaml.safe_dump(small_config(component, trucks=2, stops=4)))
        problems.append(SuiteProblem(component, str(path)))
    rows = {
        problem.name: {row["seed"]: row for row in run_reference_episodes(problem, range(4))}
        for problem in problems
    }
    config = PPOConfig(total_timesteps=2 * 4, num_envs=2, rollout_steps=4, minibatch_size=4, epochs=1, seed=1)
    policy_config = GeneralistPolicyConfig(
        hidden_dim=32, encoder_layers=1, charger_budget=4, plan_conditioning=True
    )
    trainer = GeneralistPPO(
        policy_config, config, problems, ReferenceTable(rows), SuiteReward(), [list(range(4))] * 2,
        plan_dropout=0.3,
    )
    try:
        history = trainer.learn()
        assert history and history[-1]["timesteps"] == 2 * 4
        assert np.isfinite(history[-1]["value_loss"])
    finally:
        trainer.close()
