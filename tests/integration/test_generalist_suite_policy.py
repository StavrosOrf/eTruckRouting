"""One generalist GraphPPO policy acts on all five suite problems (plan 4.1 gates).

The five problems span 25 to 432 chargers, 30 to 200 request slots and 234 to
470 environment actions; the canonical flat observation of the largest is
3.6 MB. The same weights must score all of them, batched together, and one
PPO update over the mixed rollout must run end to end.
"""

import pytest
import torch

from algo.canonical_ppo import PPOConfig
from algo.generalist_observation import GeneralistEnv, collate
from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig
from algo.generalist_ppo import (
    GeneralistPPO,
    ReferenceTable,
    SuiteReward,
    evaluate_policy_suite,
    run_reference_episodes,
)
from EVRoutingEnv.state.action_mask import get_action_mask
from scripts.training.train_generalist_ppo import load_problems


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def suite():
    problems = load_problems(None)
    rows = {problem.name: {row["seed"]: row for row in run_reference_episodes(problem, [0, 1])} for problem in problems}
    return problems, ReferenceTable(rows)


def test_one_policy_scores_every_problem_in_one_batch(suite) -> None:
    problems, _ = suite
    torch.manual_seed(0)
    policy = GeneralistActorCritic(GeneralistPolicyConfig(hidden_dim=32, fleet_assignment=True)).eval()
    observations, envelopes = [], set()
    for problem in problems:
        env = GeneralistEnv(problem.config, policy.config.observation_spec)
        observation, _ = env.reset(seed=0)
        envelopes.add((len(env.charging_nodes), env.fixed_num_stops, env.action_space.n))
        assert get_action_mask(env)[observation.action_index].all()
        assert len(observation.chargers) <= 32 + 2 * len(env.trucks)
        observations.append(observation)
        env.close()
    assert len(envelopes) == len(problems)
    policy.observe(collate(observations))
    output = policy(collate(observations))
    for position, observation in enumerate(observations):
        alone = policy(collate([observation]))
        assert torch.allclose(alone.logits[0, : observation.num_actions], output.logits[position, : observation.num_actions], atol=1e-5)


def test_a_mixed_rollout_trains_and_evaluates(suite) -> None:
    problems, references = suite
    config = PPOConfig(total_timesteps=5 * 6, num_envs=5, rollout_steps=6, minibatch_size=15, epochs=1, seed=0)
    trainer = GeneralistPPO(
        GeneralistPolicyConfig(hidden_dim=32, encoder_layers=1, fleet_assignment=True),
        config, problems, references, SuiteReward(reference="scenario"), [[0, 1]] * len(problems),
        rollout_workers=5,
    )
    try:
        history = trainer.learn()
    finally:
        trainer.close()
    assert history[-1]["timesteps"] == 30
    rows = evaluate_policy_suite(
        trainer.policy, problems[:2], [[0], [1]], references, SuiteReward(), workers=2,
    )
    assert sorted(row["problem"] for row in rows) == sorted(problem.name for problem in problems[:2])
