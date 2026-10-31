"""Inference-time wrappers: decoding, lookahead cloning, repair, gate-ready rows."""

import math

import pytest
import torch

from generalist_harness import small_env
from algo.generalist_inference import (
    EnsemblePolicy,
    LookaheadController,
    PlannerRepairController,
    TemperaturePolicy,
    clone_environment,
    evaluate_wrapped_problem,
    parse_wrapper,
    run_wrapped_episode,
)
from algo.generalist_observation import ObservationSpec, collate
from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig
from algo.generalist_ppo import ReferenceTable, SuiteProblem, SuiteReward
from scripts.campaigns.generalist_near_optimal_gate import gate_problem


SPEC = ObservationSpec(charger_budget=4)


def tiny_policy(seed: int = 0) -> GeneralistActorCritic:
    torch.manual_seed(seed)
    config = GeneralistPolicyConfig(
        hidden_dim=32, encoder_layers=1, action_head_layers=1, charger_budget=4
    )
    return GeneralistActorCritic(config).eval()


def observations_from(env, seed: int, steps: int, policy) -> list:
    observation, _ = env.reset(seed=seed)
    kept = []
    with torch.no_grad():
        for _ in range(steps):
            kept.append(observation)
            position = int(policy(collate([observation])).logits[0].argmax().item())
            observation, _, terminated, truncated, _ = env.step(int(observation.action_index[position]))
            if terminated or truncated:
                break
    return kept


def scenario_references(name: str, seeds, *, dynamic: bool) -> ReferenceTable:
    rows = {
        int(seed): {"dynamic": dynamic, "hours": 12.0, "served": 3, "tasks": 4}
        for seed in seeds
    }
    return ReferenceTable({name: rows})


def test_parse_wrapper_specs():
    spec = parse_wrapper("lookahead:k=4,horizon=12,bootstrap=1")
    assert spec.name == "lookahead" and spec.options == {"k": 4, "horizon": 12, "bootstrap": 1}
    assert not spec.batched
    spec = parse_wrapper("temperature:t=0.7")
    assert spec.batched and spec.options == {"temperature": 0.7}
    assert parse_wrapper("ensemble:runs=a|b|c").options == {"runs": ["a", "b", "c"]}
    assert parse_wrapper("repair").options == {}
    with pytest.raises(ValueError):
        parse_wrapper("beam:k=3")
    with pytest.raises(ValueError):
        parse_wrapper("temperature:0.7")


def test_temperature_zero_is_greedy_and_sampling_stays_feasible():
    env = small_env("same_day", spec=SPEC)
    policy = tiny_policy()
    observations = observations_from(env, 11, 6, policy)
    batch = collate(observations)
    with torch.no_grad():
        base = policy(batch)
    greedy = TemperaturePolicy(policy, 0.0)
    actions, log_probs, values = greedy.act(batch, deterministic=False)
    assert torch.equal(actions, base.logits.argmax(dim=-1))
    assert torch.allclose(values, base.values)
    sampled = TemperaturePolicy(policy, 1.3)
    torch.manual_seed(3)
    actions, log_probs, _ = sampled.act(batch)
    assert bool(base.feasible_mask.gather(1, actions.view(-1, 1)).all())
    assert bool(torch.isfinite(log_probs).all())
    assert greedy.decisions == len(observations) and greedy.latency()["mean_seconds"] >= 0.0
    with pytest.raises(ValueError):
        TemperaturePolicy(policy, -0.1)
    env.close()


def test_ensemble_single_member_is_the_base_policy():
    env = small_env("same_day", spec=SPEC)
    policy = tiny_policy()
    observations = observations_from(env, 7, 4, policy)
    batch = collate(observations)
    with torch.no_grad():
        base = policy(batch)
    ensemble = EnsemblePolicy([policy])
    actions, _, values = ensemble.act(batch)
    assert torch.equal(actions, base.logits.argmax(dim=-1))
    assert torch.allclose(values, base.values)
    pair = EnsemblePolicy([policy, tiny_policy(seed=5)])
    actions, _, _ = pair.act(batch)
    assert bool(base.feasible_mask.gather(1, actions.view(-1, 1)).all())
    other = tiny_policy()
    other.config = GeneralistPolicyConfig(hidden_dim=32, encoder_layers=1, action_head_layers=1, charger_budget=8)
    with pytest.raises(ValueError):
        EnsemblePolicy([policy, other])
    env.close()


def test_clone_environment_isolates_the_live_episode():
    policy = tiny_policy()
    env = small_env("same_day", spec=SPEC)
    twin = small_env("same_day", spec=SPEC)
    observation, _ = env.reset(seed=19)
    twin_observation, _ = twin.reset(seed=19)
    # March both to the same mid-episode state.
    with torch.no_grad():
        for _ in range(3):
            position = int(policy(collate([observation])).logits[0].argmax().item())
            action = int(observation.action_index[position])
            observation, *_ = env.step(action)
            twin_observation, *_ = twin.step(action)
    clock, steps = float(env.global_clock), env.episode_steps
    shadow = clone_environment(env)
    shadow_observation = shadow._get_observation()
    with torch.no_grad():
        for _ in range(6):
            position = int(policy(collate([shadow_observation])).logits[0].argmax().item())
            shadow_observation, _, terminated, truncated, _ = shadow.step(
                int(shadow_observation.action_index[position])
            )
            if terminated or truncated:
                break
    assert float(env.global_clock) == clock and env.episode_steps == steps
    # The original episode continues exactly like its never-cloned twin.
    with torch.no_grad():
        for _ in range(4):
            position = int(policy(collate([observation])).logits[0].argmax().item())
            action = int(observation.action_index[position])
            twin_position = int(policy(collate([twin_observation])).logits[0].argmax().item())
            assert int(twin_observation.action_index[twin_position]) == action
            observation, _, done_a, trunc_a, _ = env.step(action)
            twin_observation, _, done_b, trunc_b, _ = twin.step(action)
            assert (done_a, trunc_a) == (done_b, trunc_b)
            assert math.isclose(float(env.global_clock), float(twin.global_clock))
            if done_a or trunc_a:
                break
    env.close()
    twin.close()
    shadow.close()


def test_lookahead_k1_matches_the_base_action():
    env = small_env("same_day", spec=SPEC)
    policy = tiny_policy()
    references = scenario_references("p", [23], dynamic=True)
    reward = SuiteReward(reference="scenario")
    scale, _ = references.scale("p", 23, "scenario")
    controller = LookaheadController(policy, reward, scale, k=1, horizon=4)
    observation, _ = env.reset(seed=23)
    with torch.no_grad():
        for _ in range(4):
            base = int(policy(collate([observation])).logits[0].argmax().item())
            chosen = controller(env, observation)
            assert chosen == int(observation.action_index[base])
            observation, _, terminated, truncated, _ = env.step(chosen)
            if terminated or truncated:
                break
    with pytest.raises(ValueError):
        LookaheadController(policy, reward, scale, k=0, horizon=1)
    env.close()


def test_lookahead_rows_satisfy_the_gate_reader():
    policy = tiny_policy()
    reward = SuiteReward(reference="scenario")
    problem = SuiteProblem(name="tiny_static", config="unused", teacher="auto")
    seeds = [31, 37]
    references = scenario_references("tiny_static", seeds, dynamic=False)
    env = small_env("static", spec=SPEC)
    rows = []
    for seed in seeds:
        scale, _ = references.scale("tiny_static", seed, "scenario")
        controller = LookaheadController(policy, reward, scale, k=2, horizon=2)
        rows.append(run_wrapped_episode(env, seed, controller, problem, reward, scale))
    env.close()
    for row in rows:
        assert {"problem", "seed", "score", "success", "failed", "served", "tasks", "hours",
                "decisions", "decision_seconds_mean", "decision_seconds_p95"} <= set(row)
        assert row["decisions"] > 0 and row["decision_seconds_mean"] > 0.0
    bks = {
        seed: {"best": {"failed": False, "success": True, "served": 4, "hours": 10.0},
               "bound": {"kind": "nominal_lower_bound_hours", "value": 8.0, "proven": False}}
        for seed in seeds
    }
    verdict = gate_problem("tiny_static", rows, bks, resamples=50)
    assert verdict["status"] in ("PASS", "FAIL")
    assert set(verdict["criteria"]) == {"failure", "success", "hours"}
    assert verdict["seeds"] == len(seeds)


def _k1_factory(problem, scale, *, policy, reward):
    return LookaheadController(policy, reward, scale, k=1, horizon=0)


def test_evaluate_wrapped_problem_covers_every_seed_once():
    import functools

    from generalist_harness import small_config

    policy = tiny_policy()
    reward = SuiteReward(reference="scenario")
    seeds = [41, 43]
    references = scenario_references("tiny_static", seeds, dynamic=False)
    problem = SuiteProblem(name="tiny_static", config=small_config("static"), teacher="auto")
    make = functools.partial(_k1_factory, policy=policy, reward=reward)
    rows = evaluate_wrapped_problem(problem, seeds, SPEC, reward, references, make, workers=1)
    assert [row["seed"] for row in rows] == seeds
    assert all(row["problem"] == "tiny_static" for row in rows)
    assert all("score" in row and "decision_seconds_mean" in row for row in rows)


def test_planner_repair_warm_start_runs_a_static_episode():
    policy = tiny_policy()
    env = small_env("static", spec=SPEC)
    controller = PlannerRepairController(
        policy, iterations=60, time_limit=2.0, restarts=1, max_steps=200
    )
    reward = SuiteReward(reference="scenario")
    references = scenario_references("tiny_static", [53], dynamic=False)
    scale, _ = references.scale("tiny_static", 53, "scenario")
    problem = SuiteProblem(name="tiny_static", config="unused", teacher="auto")
    row = run_wrapped_episode(env, 53, controller, problem, reward, scale)
    assert row["problem"] == "tiny_static" and row["decisions"] > 0
    assert controller._executor is not None
    assert "warm_start" in controller._executor.stats or controller._executor.stats.get("plan")
    dynamic_env = small_env("same_day", spec=SPEC)
    dynamic_env.reset(seed=3)
    with pytest.raises(ValueError):
        PlannerRepairController(policy)(dynamic_env, None)
    env.close()
    dynamic_env.close()


def test_scheduled_temperature_anneals_by_progress():
    from algo.generalist_inference import CONTEXT_PROGRESS_INDEX

    env = small_env("same_day", spec=SPEC)
    policy = tiny_policy()
    observations = observations_from(env, 11, 6, policy)
    batch = collate(observations)
    # The schedule reads episode progress from the context row: now / horizon,
    # zero at reset and nondecreasing along the episode.
    progress = batch.context[:, CONTEXT_PROGRESS_INDEX]
    assert float(progress[0]) == 0.0
    assert bool((progress[1:] >= progress[:-1] - 1e-9).all())
    assert bool((progress >= 0).all()) and bool((progress <= 1.0 + 1e-9).all())
    with torch.no_grad():
        base = policy(batch)
    # t annealed to zero: rows at progress 1 must take the argmax exactly.
    batch.context[:, CONTEXT_PROGRESS_INDEX] = 1.0
    cold = TemperaturePolicy(policy, 1.3, t_end=0.0)
    actions, _, _ = cold.act(batch)
    assert torch.equal(actions, base.logits.argmax(dim=-1))
    # Constant schedule (t_end omitted) still samples feasible actions.
    batch.context[:, CONTEXT_PROGRESS_INDEX] = 0.5
    warm = TemperaturePolicy(policy, 1.3)
    torch.manual_seed(5)
    actions, log_probs, _ = warm.act(batch)
    assert bool(base.feasible_mask.gather(1, actions.view(-1, 1)).all())
    assert bool(torch.isfinite(log_probs).all())
    with pytest.raises(ValueError):
        TemperaturePolicy(policy, 1.0, t_end=-0.5)
    env.close()


def test_top_p_nucleus_keeps_the_greedy_action():
    env = small_env("same_day", spec=SPEC)
    policy = tiny_policy()
    batch = collate(observations_from(env, 11, 6, policy))
    with torch.no_grad():
        base = policy(batch)
    # A vanishing nucleus leaves only the argmax to sample.
    tight = TemperaturePolicy(policy, 1.3, top_p=1e-6)
    torch.manual_seed(7)
    actions, log_probs, _ = tight.act(batch)
    assert torch.equal(actions, base.logits.argmax(dim=-1))
    assert bool(torch.isfinite(log_probs).all())
    # top_p = 1 must reproduce plain tempered sampling exactly.
    loose = TemperaturePolicy(policy, 1.3, top_p=1.0)
    torch.manual_seed(9)
    expected, _, _ = TemperaturePolicy(policy, 1.3).act(batch)
    torch.manual_seed(9)
    actions, _, _ = loose.act(batch)
    assert torch.equal(actions, expected)
    with pytest.raises(ValueError):
        TemperaturePolicy(policy, 1.0, top_p=0.0)
    env.close()
