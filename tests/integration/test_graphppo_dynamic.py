"""GraphPPO smoke tests on the dynamic request modes (plan section 8.7)."""

import numpy as np
import pytest
import torch
from extension_harness import SAME_DAY, build

from algo.canonical_policy import CanonicalActorCritic, CanonicalPolicyConfig
from algo.canonical_ppo import CanonicalPPO, PPOConfig
from EVRoutingEnv.models.core.requests import RequestStatus
from EVRoutingEnv.state.feasibility import FeasibilityReason, joint_action_feasibility


pytestmark = pytest.mark.integration

SMALL = {**SAME_DAY, "max_requests": 60, "max_requests_per_epoch": 2, "static_context_size": 20}


def _env(**requests):
    return build("seattle", trucks=3, stops=4, requests={**SMALL, **requests})


@pytest.mark.parametrize("revealed", [0, 1, "max"])
def test_canonical_graph_builds_with_any_number_of_revealed_requests(revealed) -> None:
    probe = _env(initial_requests=0)
    probe.reset(seed=0)
    # "max": the envelope is exactly full.
    envelope = len(probe.request_engine.registry) if revealed == "max" else SMALL["max_requests"]
    env = _env(initial_requests=0, max_requests=envelope)
    env.reset(seed=0)
    registry = env.request_engine.registry
    for request in registry.requests():
        request.status = RequestStatus.HIDDEN
    count = env.fixed_num_stops if revealed == "max" else revealed
    assert len(registry) >= count
    for request in registry.requests()[:count]:
        request.status = RequestStatus.OPEN
    sets = env.get_canonical_sets()
    graph = env.get_canonical_graph()
    assert int(sets.customer_mask.sum()) == count
    assert graph.node_features["customer"].shape[0] == count
    decisions = joint_action_feasibility(env)
    hidden = [env.num_charging_nodes + i for i in range(count, env.fixed_num_stops)]
    assert all(decisions[i].reason is FeasibilityReason.REQUEST_HIDDEN for i in hidden)
    policy = CanonicalActorCritic(CanonicalPolicyConfig.from_env(env, hidden_dim=32, encoder_output_dim=32))
    observation = torch.as_tensor(env._get_observation()[None, :], dtype=torch.float32)
    output = policy(observation, torch.as_tensor(env.mask_fn()[None, :]))
    # Never scored: outside the feasible set, so zero probability.
    assert not output.feasible_mask[0, hidden].any()
    probabilities = output.distribution().probs[0].detach().numpy()
    assert np.all(probabilities[hidden] == 0.0)


def _ppo(total):
    env = _env()
    config = CanonicalPolicyConfig.from_env(env, hidden_dim=32, encoder_output_dim=32)
    seen = []

    class Checked:
        """Every rollout must satisfy invariant 10: no executed action is refused."""

        def __init__(self):
            self.env = _env()
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space

        def reset(self, seed=None, options=None):
            return self.env.reset(seed=seed, options=options)

        def step(self, action):
            result = self.env.step(action)
            assert self.env.invalid_action_count == 0
            seen.append(action)
            return result

        def mask_fn(self):
            return self.env.mask_fn()

        def policy_mask_fn(self):
            return self.env.policy_mask_fn()

        def close(self):
            self.env.close()

    trainer = CanonicalPPO(
        config,
        PPOConfig(total_timesteps=total, num_envs=2, rollout_steps=64, epochs=1,
                  minibatch_size=64, seed=0),
        Checked,
    )
    try:
        trainer.learn()
    finally:
        trainer.close()
    assert len(seen) >= total - 2 * 64  # whole rollouts of 2 x 64 steps


def test_short_ppo_run_on_a_dynamic_config() -> None:
    _ppo(512)


@pytest.mark.statistical
def test_twenty_thousand_step_ppo_run_on_a_three_truck_dynamic_config() -> None:
    _ppo(20_000)
