"""The privileged critic (plan 9.4) sees the hidden stream; the actor never does."""

import json
from dataclasses import replace

import numpy as np
import torch
import yaml
from generalist_harness import random_observations, small_config, small_env

from algo.generalist_observation import (
    PRIVILEGED_DIM,
    ObservationSpec,
    build_observation,
    collate,
)
from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig
from EVRoutingEnv.models.core.requests import RequestStatus


def _spec(privileged: bool = True) -> ObservationSpec:
    return ObservationSpec(charger_budget=4, privileged=privileged)


def _policy(**overrides) -> GeneralistActorCritic:
    torch.manual_seed(0)
    config = GeneralistPolicyConfig(hidden_dim=32, encoder_layers=2, charger_budget=4, **overrides)
    return GeneralistActorCritic(config).eval()


def _observations(privileged: bool = True) -> list:
    return random_observations(small_env("same_day", spec=_spec(privileged)), seed=0, steps=5)


def test_the_hidden_stream_summary_exists_only_for_dynamic_problems() -> None:
    env = small_env("same_day", spec=_spec())
    observation, _ = env.reset(seed=0)
    assert observation.privileged is not None
    assert observation.privileged.shape == (PRIVILEGED_DIM,)
    static = random_observations(small_env("static", spec=_spec()), seed=0, steps=1)[0]
    assert static.privileged is not None and not static.privileged.any()
    assert _observations(privileged=False)[0].privileged is None


def test_the_summary_aggregates_the_hidden_stream() -> None:
    env = small_env("same_day", spec=_spec())
    env.reset(seed=0)
    requests = env.request_engine.registry.requests()
    # The tiny stream reveals everything at the first epoch; hide one request
    # again so the summary has something to see, half the horizon away.
    hidden = requests[0]
    hidden.status = RequestStatus.HIDDEN
    hidden.reveal_time = hidden.release_time = env.max_time / 2
    observation = build_observation(env, _spec())
    values = observation.privileged
    assert values is not None and values.any()
    assert np.isfinite(values).all()
    assert values[0] == 1.0 / len(requests)        # hidden_fraction
    assert np.isclose(values[1], 1.0 / 64.0)       # hidden_count
    assert np.isclose(values[3], 0.5)              # next_reveal_in
    bucket = 7 + 5 * 2                             # reveal at half horizon: third bucket
    assert values[bucket] > 0.0
    assert values[7] == 0.0 and values[7 + 5] == 0.0 and values[7 + 15] == 0.0


def test_collate_zero_fills_observations_without_privileged_rows() -> None:
    with_rows, without = _observations()[0], _observations(privileged=False)[0]
    batch = collate([with_rows, without])
    assert batch.privileged.shape == (2, PRIVILEGED_DIM)
    assert torch.equal(batch.privileged[0], torch.as_tensor(with_rows.privileged))
    assert not batch.privileged[1].any()


def test_the_actor_never_reads_the_privileged_rows() -> None:
    observations = _observations()
    policy = _policy(privileged_critic=True)
    policy.observe(collate(observations))
    batch = collate(observations)
    perturbed = replace(batch, privileged=batch.privileged + 3.0)
    original, shifted = policy(batch), policy(perturbed)
    assert torch.equal(original.logits, shifted.logits)
    assert not torch.allclose(original.values, shifted.values)


def test_the_privileged_flag_never_changes_the_actor() -> None:
    observations = _observations()
    off, on = _policy(), _policy(privileged_critic=True)
    shared = {
        name: value for name, value in off.state_dict().items() if not name.startswith("value_head")
    }
    missing, unexpected = on.load_state_dict(shared, strict=False)
    assert not unexpected
    assert all(name.startswith(("value_head", "privileged_")) for name in missing)
    batch = collate(observations)
    assert torch.equal(off(batch).logits, on(batch).logits)


def test_gradients_reach_the_privileged_modules() -> None:
    observations = _observations()
    policy = _policy(privileged_critic=True).train()
    policy.observe(collate(observations))
    _, _, values = policy.evaluate_actions(
        collate(observations), torch.zeros(len(observations), dtype=torch.long)
    )
    values.pow(2).mean().backward()
    for name, parameter in policy.named_parameters():
        if name.startswith("privileged_encoder"):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_checkpoints_round_trip_and_old_configs_default_off(tmp_path) -> None:
    observations = _observations()
    policy = _policy(privileged_critic=True)
    policy.observe(collate(observations))
    policy.save(tmp_path, prefix="best")
    restored = GeneralistActorCritic.load(tmp_path, prefix="best").eval()
    assert restored.config.privileged_critic
    batch = collate(observations)
    assert torch.equal(policy(batch).logits, restored(batch).logits)
    assert torch.equal(policy(batch).values, restored(batch).values)
    legacy = json.loads((tmp_path / "best_config.json").read_text())
    del legacy["privileged_critic"]
    (tmp_path / "legacy.json").write_text(json.dumps(legacy))
    assert not GeneralistPolicyConfig.load(tmp_path / "legacy.json").privileged_critic


def test_one_ppo_update_trains_through_the_privileged_path(tmp_path) -> None:
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
        hidden_dim=32, encoder_layers=1, charger_budget=4, privileged_critic=True
    )
    trainer = GeneralistPPO(
        policy_config, config, problems, ReferenceTable(rows), SuiteReward(), [list(range(4))] * 2
    )
    try:
        history = trainer.learn()
        assert history and history[-1]["timesteps"] == 2 * 4
        assert np.isfinite(history[-1]["value_loss"])
    finally:
        trainer.close()
