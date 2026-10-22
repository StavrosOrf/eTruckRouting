"""One generalist policy reads any problem size, and padding or row order never matter."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from generalist_harness import random_observations, small_env

from algo.generalist_observation import ObservationSpec, collate
from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig


def _policy(**overrides) -> GeneralistActorCritic:
    torch.manual_seed(0)
    config = GeneralistPolicyConfig(hidden_dim=32, encoder_layers=2, charger_budget=4, fleet_assignment=True, **overrides)
    return GeneralistActorCritic(config).eval()


def _mixed_observations() -> list:
    spec = ObservationSpec(charger_budget=4, fleet_assignment=True)
    observations = random_observations(small_env("static", spec=spec), seed=0, steps=5)
    observations += random_observations(small_env("hos", spec=spec, trucks=3, stops=7), seed=1, steps=5)
    observations += random_observations(small_env("same_day", spec=spec), seed=2, steps=5)
    return observations


def _permuted(observation, generator):
    """The same state with every row type and the action list reordered."""
    trucks, tasks, chargers = len(observation.trucks), len(observation.tasks), len(observation.chargers)
    order = {key: generator.permutation(size) for key, size in (("t", trucks), ("c", tasks), ("k", chargers))}
    actions = generator.permutation(observation.num_actions)
    inverse = {key: np.argsort(value) for key, value in order.items()}
    nodes = np.concatenate([order["t"], trucks + order["c"], trucks + tasks + order["k"]])

    def relink(links, key):
        return np.where(links >= 0, inverse[key][np.maximum(links, 0)], -1)

    return replace(
        observation,
        trucks=observation.trucks[order["t"]],
        tasks=observation.tasks[order["c"]],
        chargers=observation.chargers[order["k"]],
        edges=observation.edges[nodes][:, nodes],
        actions=observation.actions[actions],
        action_index=observation.action_index[actions],
        action_task=relink(observation.action_task[actions], "c"),
        action_charger=relink(observation.action_charger[actions], "k"),
        active_truck=int(inverse["t"][observation.active_truck]) if observation.active_truck >= 0 else -1,
    ), actions


def test_padding_inside_a_batch_never_changes_logits_or_values() -> None:
    observations = _mixed_observations()
    policy = _policy()
    policy.observe(collate(observations))
    together = policy(collate(observations))
    for position, observation in enumerate(observations):
        alone = policy(collate([observation]))
        width = observation.num_actions
        assert torch.allclose(alone.logits[0, :width], together.logits[position, :width], atol=1e-5)
        assert torch.allclose(alone.values[0], together.values[position], atol=1e-5)
        assert not together.feasible_mask[position, width:].any()


def test_reordering_trucks_tasks_chargers_and_actions_permutes_the_policy() -> None:
    observations = _mixed_observations()
    policy = _policy()
    policy.observe(collate(observations))
    generator = np.random.default_rng(7)
    for observation in observations:
        permuted, actions = _permuted(observation, generator)
        original = policy(collate([observation]))
        shuffled = policy(collate([permuted]))
        width = observation.num_actions
        assert torch.allclose(shuffled.logits[0, :width], original.logits[0, :width][actions], atol=1e-4)
        assert torch.allclose(shuffled.values, original.values, atol=1e-4)


def test_gradients_reach_every_parameter_and_stay_finite() -> None:
    observations = _mixed_observations()
    policy = _policy().train()
    policy.observe(collate(observations))
    batch = collate(observations)
    logprob, entropy, values = policy.evaluate_actions(batch, torch.zeros(len(observations), dtype=torch.long))
    (logprob.mean() + entropy.mean() + values.pow(2).mean()).backward()
    for name, parameter in policy.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_evaluate_refuses_actions_outside_the_candidate_set() -> None:
    observations = sorted(_mixed_observations(), key=lambda observation: observation.num_actions)
    pair = [observations[0], observations[-1]]
    assert pair[0].num_actions < pair[1].num_actions
    # The last slot is real for the wider state and padding for the narrower one.
    last = torch.full((2,), pair[1].num_actions - 1, dtype=torch.long)
    with pytest.raises(ValueError, match="outside the candidate set"):
        _policy().evaluate_actions(collate(pair), last)


def test_checkpoints_round_trip_and_refuse_another_schema(tmp_path) -> None:
    observations = _mixed_observations()
    policy = _policy()
    policy.observe(collate(observations))
    policy.save(tmp_path, prefix="best")
    restored = GeneralistActorCritic.load(tmp_path, prefix="best").eval()
    assert torch.equal(policy(collate(observations)).logits, restored(collate(observations)).logits)
    with pytest.raises(ValueError, match="incompatible"):
        GeneralistPolicyConfig(feature_schema_version="generalist-v0")
