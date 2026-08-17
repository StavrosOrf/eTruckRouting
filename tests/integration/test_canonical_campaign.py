"""End-to-end checks for the canonical policy inside the immutable runner.

These guard the exact path the headline campaign uses: build a policy, save it,
reload it from disk, and score it through ``run_evaluation_campaign`` so that a
broken checkpoint or artifact contract fails here rather than mid-campaign.
"""

import json
import os
from copy import deepcopy

import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from algo.behavior_cloning import collect_demonstrations, pretrain_policy
from algo.canonical_policy import CanonicalActorCritic, CanonicalPolicyConfig
from EVRoutingEnv.evaluation.artifacts import CampaignSeedPlan, collect_run_manifest
from EVRoutingEnv.evaluation.runner import run_evaluation_campaign
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config
from scripts.evaluation.run_canonical_campaign import (
    CanonicalPolicyRunner,
    build_policy,
)


def _config() -> dict:
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["environment"]["num_stops"] = 5
    return config


@pytest.fixture(scope="module")
def config() -> dict:
    return _config()


def _policy_config(env, **overrides) -> CanonicalPolicyConfig:
    return CanonicalPolicyConfig.from_env(
        env,
        state_encoder="flat",
        action_head="independent",
        hidden_dim=32,
        encoder_output_dim=32,
        **overrides,
    )


@pytest.mark.integration
def test_saved_policy_round_trips_and_scores_through_the_runner(config, tmp_path):
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        env.reset(seed=7)
        policy = CanonicalActorCritic(_policy_config(env))
    finally:
        env.close()

    checkpoint = tmp_path / "checkpoint"
    policy.save(checkpoint, prefix="best")
    reloaded = CanonicalActorCritic.load(checkpoint, prefix="best")
    assert reloaded.config == policy.config

    seeds = [CampaignSeedPlan().seed("validation", index) for index in range(2)]
    manifest = collect_run_manifest(
        run_id="rl__test",
        algorithm="rl",
        split="validation",
        command=("pytest",),
        resolved_config=config,
        scenario_seeds=seeds,
        repository_root=os.getcwd(),
        checkpoint=str(checkpoint),
    )
    artifacts = run_evaluation_campaign(
        environment_factory=lambda: EventDrivenTruckEnv(
            config, verbose=False, enable_plotting=False
        ),
        policy=CanonicalPolicyRunner(checkpoint, prefix="best"),
        manifest=manifest,
        output_directory=tmp_path / "campaign",
        max_policy_steps=400,
    )

    assert artifacts.episode_count == len(seeds)
    assert artifacts.summary_path.exists()
    rows = [
        json.loads(line)
        for line in artifacts.episode_rows_path.read_text().splitlines()
        if line.strip()
    ]
    assert [row["scenario_seed"] for row in rows] == seeds
    for row in rows:
        # Failures must survive into the artifacts rather than being dropped.
        assert "success" in row
        assert row["policy_calls"] > 0
        assert row["inference_seconds"] >= 0.0

    summary = json.loads(artifacts.summary_path.read_text())
    assert summary["aggregate"]["episode_count"] == len(seeds)


@pytest.mark.integration
def test_untrained_policy_still_only_emits_feasible_actions(config):
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        observation, _info = env.reset(seed=11)
        runner_policy = CanonicalActorCritic(_policy_config(env))
        runner_policy.eval()

        import numpy as np
        import torch

        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 60:
            mask = env.mask_fn()
            batch = torch.as_tensor(np.expand_dims(observation, 0), dtype=torch.float32)
            mask_tensor = torch.as_tensor(np.expand_dims(mask, 0), dtype=torch.bool)
            actions, _, _ = runner_policy.act(batch, mask_tensor, deterministic=True)
            action = int(actions[0].item())
            assert mask[action], "policy selected an action outside the hard mask"
            observation, _, terminated, truncated, _info = env.step(action)
            steps += 1
    finally:
        env.close()


@pytest.mark.integration
def test_behaviour_cloning_moves_the_policy_toward_the_demonstrator(config):
    from EVRoutingEnv.baselines.canonical_baselines import (
        GreedyHeuristicPolicy,
        HeuristicParameters,
    )

    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        env.reset(seed=3)
        policy = CanonicalActorCritic(_policy_config(env))
        dataset = collect_demonstrations(
            env,
            GreedyHeuristicPolicy(HeuristicParameters(energy_safety_factor=1.15)),
            [CampaignSeedPlan().seed("train", index) for index in range(25)],
            successful_only=False,
        )
    finally:
        env.close()

    assert len(dataset) > 0
    history = pretrain_policy(policy, dataset, epochs=6, batch_size=64)
    assert history[-1]["train_loss"] < history[0]["train_loss"]


@pytest.mark.integration
def test_build_policy_rejects_unknown_methods():
    with pytest.raises(ValueError, match="unknown method"):
        build_policy("oracle", {})
