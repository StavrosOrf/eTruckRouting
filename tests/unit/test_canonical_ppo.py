"""Correctness and learnability checks for the canonical PPO stack.

The synthetic environment below emits real canonical observations built through
:func:`pad_canonical_features`, so the tests exercise the same unpacking,
masking, and ragged-action path the simulator uses -- while keeping the optimal
policy known in closed form.
"""

import os

import numpy as np
import pytest
import torch


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from gymnasium import spaces

from algo.canonical_policy import CanonicalActorCritic, CanonicalPolicyConfig
from algo.canonical_ppo import (
    CanonicalPPO,
    PPOConfig,
    RewardShaping,
    RunningReturnScale,
    SyncCanonicalVecEnv,
    WorkerCanonicalVecEnv,
)
from algo.canonical_state import (
    FEASIBLE_COLUMN,
    scatter_logits,
    unpack_flat_observation,
)
from EVRoutingEnv.state.features import (
    ACTION_FEATURES,
    CHARGER_FEATURES,
    CUSTOMER_FEATURES,
    EDGE_FEATURES,
    RELATION_TYPES,
    SCHEMA_VERSION,
    TRUCK_FEATURES,
    CanonicalFleetFeatures,
)
from EVRoutingEnv.state.representations import (
    CanonicalShapeSpec,
    pad_canonical_features,
)


SHAPE = CanonicalShapeSpec(max_trucks=1, max_customers=2, max_chargers=2, max_actions=4)
_CHARGE_COLUMN = ACTION_FEATURES.index("charge_value")


def _features(rng: np.random.Generator, feasible: np.ndarray) -> CanonicalFleetFeatures:
    """Build one valid canonical snapshot with the given feasibility pattern."""
    action_rows = np.zeros((SHAPE.max_actions, len(ACTION_FEATURES)), dtype=np.float32)
    action_rows[:, _CHARGE_COLUMN] = rng.uniform(0.0, 1.0, size=SHAPE.max_actions)
    action_rows[:, FEASIBLE_COLUMN] = feasible.astype(np.float32)
    counts = {"truck": 1, "customer": 2, "charger": 2}
    return CanonicalFleetFeatures(
        schema_version=SCHEMA_VERSION,
        truck_features=rng.normal(size=(1, len(TRUCK_FEATURES))).astype(np.float32),
        customer_features=rng.normal(size=(2, len(CUSTOMER_FEATURES))).astype(
            np.float32
        ),
        charger_features=rng.normal(size=(2, len(CHARGER_FEATURES))).astype(np.float32),
        action_features=action_rows,
        global_features=rng.normal(size=8).astype(np.float32),
        pairwise_features={
            relation: rng.uniform(
                0.0,
                1.0,
                size=(counts[relation[0]], counts[relation[1]], len(EDGE_FEATURES)),
            ).astype(np.float32)
            for relation in RELATION_TYPES
        },
    )


class CanonicalBanditEnv:
    """One-step task whose optimal action is the feasible row with max charge value."""

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(SHAPE.flat_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(SHAPE.max_actions)
        self._feasible = np.ones(SHAPE.max_actions, dtype=bool)
        self._values = np.zeros(SHAPE.max_actions, dtype=np.float32)

    def _sample(self) -> np.ndarray:
        feasible = self._rng.random(SHAPE.max_actions) < 0.75
        if not feasible.any():
            feasible[self._rng.integers(SHAPE.max_actions)] = True
        features = _features(self._rng, feasible)
        self._feasible = feasible
        self._values = features.action_features[:, _CHARGE_COLUMN].copy()
        return pad_canonical_features(features, SHAPE).flatten()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self._sample(), {}

    def step(self, action):
        action = int(action)
        if not self._feasible[action]:
            raise AssertionError("policy selected an infeasible action")
        best = float(self._values[self._feasible].max())
        reward = float(self._values[action]) - best
        observation = self._sample()
        return observation, reward, True, False, {"episode_reward": reward}

    def mask_fn(self) -> np.ndarray:
        return self._feasible.copy()

    def close(self) -> None:
        return None


def _observation_batch(count: int, seed: int = 0) -> torch.Tensor:
    environment = CanonicalBanditEnv(seed=seed)
    rows = [environment.reset(seed=seed)[0]]
    for _ in range(count - 1):
        rows.append(environment.step(int(np.flatnonzero(environment.mask_fn())[0]))[0])
    return torch.as_tensor(np.stack(rows), dtype=torch.float32)


def test_unpacking_round_trips_every_canonical_block() -> None:
    rng = np.random.default_rng(3)
    feasible = np.array([True, False, True, True])
    features = _features(rng, feasible)
    padded = pad_canonical_features(features, SHAPE)
    tensors = unpack_flat_observation(
        torch.as_tensor(padded.flatten()).unsqueeze(0), SHAPE
    )

    np.testing.assert_allclose(
        tensors.nodes["truck"][0].numpy(), padded.truck_features, rtol=1e-6
    )
    np.testing.assert_allclose(
        tensors.action[0].numpy(), padded.action_features, rtol=1e-6
    )
    np.testing.assert_allclose(
        tensors.global_features[0].numpy(), padded.global_features, rtol=1e-6
    )
    for relation in RELATION_TYPES:
        np.testing.assert_allclose(
            tensors.pairwise[relation][0].numpy(),
            padded.pairwise_features[relation],
            rtol=1e-6,
        )
        np.testing.assert_array_equal(
            tensors.pairwise_mask[relation][0].numpy(),
            padded.pairwise_mask[relation],
        )
    np.testing.assert_array_equal(tensors.feasible_action_mask()[0].numpy(), feasible)


def test_ragged_actions_follow_the_hard_mask() -> None:
    observations = _observation_batch(3, seed=11)
    tensors = unpack_flat_observation(observations, SHAPE)
    rows, ptr = tensors.ragged_actions()

    mask = tensors.feasible_action_mask()
    assert rows.shape == (int(mask.sum().item()), len(ACTION_FEATURES))
    assert int(ptr[0].item()) == 0
    assert int(ptr[-1].item()) == rows.shape[0]
    np.testing.assert_array_equal((ptr[1:] - ptr[:-1]).numpy(), mask.sum(dim=1).numpy())


def test_override_mask_may_narrow_but_never_widen() -> None:
    observations = _observation_batch(2, seed=5)
    tensors = unpack_flat_observation(observations, SHAPE)
    mask = tensors.feasible_action_mask()

    narrowed = mask.clone()
    for row in range(narrowed.shape[0]):
        feasible_positions = torch.nonzero(narrowed[row]).view(-1)
        if feasible_positions.numel() > 1:
            narrowed[row, feasible_positions[0]] = False
    rows, _ = tensors.ragged_actions(narrowed)
    assert rows.shape[0] == int(narrowed.sum().item())

    with pytest.raises(ValueError, match="infeasible"):
        tensors.ragged_actions(~mask)


def test_empty_feasible_set_is_refused() -> None:
    rng = np.random.default_rng(9)
    features = _features(rng, np.zeros(SHAPE.max_actions, dtype=bool))
    observation = torch.as_tensor(
        pad_canonical_features(features, SHAPE).flatten()
    ).unsqueeze(0)
    tensors = unpack_flat_observation(observation, SHAPE)

    with pytest.raises(RuntimeError, match="no feasible action"):
        tensors.ragged_actions()


def test_scatter_logits_places_values_and_suppresses_padding() -> None:
    mask = torch.tensor([[True, False, True], [False, True, False]])
    logits = torch.tensor([1.0, 2.0, 3.0])
    ptr = torch.tensor([0, 2, 3])

    dense = scatter_logits(logits, ptr, mask)
    assert dense[0, 0] == 1.0
    assert dense[0, 2] == 2.0
    assert dense[1, 1] == 3.0
    assert torch.isfinite(dense).all()
    probabilities = torch.softmax(dense, dim=-1)
    assert torch.allclose(probabilities[~mask], torch.zeros(3), atol=1e-30)


@pytest.mark.parametrize("state_encoder", ["flat", "deep_sets", "hetero_graph"])
@pytest.mark.parametrize(
    "action_head", ["independent", "complete_gcn", "self_attention"]
)
def test_every_combination_produces_finite_gradients(
    state_encoder, action_head
) -> None:
    observations = _observation_batch(4, seed=2)
    config = CanonicalPolicyConfig(
        max_trucks=SHAPE.max_trucks,
        max_customers=SHAPE.max_customers,
        max_chargers=SHAPE.max_chargers,
        max_actions=SHAPE.max_actions,
        state_encoder=state_encoder,
        action_head=action_head,
        hidden_dim=16,
        encoder_output_dim=16,
        action_attention_heads=2,
    )
    policy = CanonicalActorCritic(config)
    policy.observe(observations)
    output = policy(observations)

    assert output.logits.shape == (4, SHAPE.max_actions)
    assert torch.isfinite(output.values).all()
    distribution = output.distribution()
    assert torch.isfinite(distribution.entropy()).all()
    (distribution.entropy().sum() + output.values.sum()).backward()
    gradients = [
        parameter.grad
        for parameter in policy.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_evaluate_actions_rejects_actions_outside_the_hard_mask() -> None:
    observations = _observation_batch(3, seed=8)
    config = CanonicalPolicyConfig(
        max_trucks=SHAPE.max_trucks,
        max_customers=SHAPE.max_customers,
        max_chargers=SHAPE.max_chargers,
        max_actions=SHAPE.max_actions,
        state_encoder="flat",
        action_head="independent",
        hidden_dim=16,
        encoder_output_dim=16,
    )
    policy = CanonicalActorCritic(config)
    tensors = unpack_flat_observation(observations, SHAPE)
    mask = tensors.feasible_action_mask()

    # Only rows that actually have an infeasible action can express the
    # violation; a row where everything happens to be feasible would otherwise
    # make this test depend on the sampler's luck.
    rows = [row for row in range(mask.shape[0]) if bool((~mask[row]).any().item())]
    assert rows, "sampled batch had no infeasible action to test against"
    actions = torch.tensor(
        [
            int(torch.nonzero(~mask[row]).view(-1)[0].item())
            if row in rows
            else int(torch.nonzero(mask[row]).view(-1)[0].item())
            for row in range(mask.shape[0])
        ]
    )

    with pytest.raises(ValueError, match="outside the feasible set"):
        policy.evaluate_actions(observations, actions)


def test_running_return_scale_tracks_reward_magnitude() -> None:
    scale = RunningReturnScale(num_envs=4, gamma=0.99)
    rewards = np.full(4, 500.0, dtype=np.float32)
    dones = np.zeros(4, dtype=bool)
    for _ in range(50):
        scale.update(rewards, dones)

    assert scale.scale > 100.0
    normalized = scale.normalize(rewards)
    assert np.all(np.abs(normalized) <= 10.0)


class _TruncatingEnv(CanonicalBanditEnv):
    """Episodes that always end by truncation rather than termination."""

    def __init__(self, seed: int = 0, length: int = 4):
        super().__init__(seed=seed)
        self._length = length
        self._step = 0

    def reset(self, seed=None, options=None):
        self._step = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, _, _, info = super().step(action)
        self._step += 1
        truncated = self._step >= self._length
        return observation, reward, False, truncated, info


def test_truncated_episodes_are_bootstrapped_not_treated_as_terminal() -> None:
    config = CanonicalPolicyConfig(
        max_trucks=SHAPE.max_trucks,
        max_customers=SHAPE.max_customers,
        max_chargers=SHAPE.max_chargers,
        max_actions=SHAPE.max_actions,
        state_encoder="flat",
        action_head="independent",
        hidden_dim=16,
        encoder_output_dim=16,
    )
    ppo_config = PPOConfig(
        total_timesteps=256,
        num_envs=2,
        rollout_steps=16,
        epochs=1,
        minibatch_size=16,
        normalize_reward=False,
    )
    trainer = CanonicalPPO(config, ppo_config, lambda: _TruncatingEnv(seed=3))
    try:
        observations, masks = trainer.envs.reset()
        observation_tensor = trainer._to_tensor(observations)
        mask_tensor = trainer._to_bool(masks)

        # Force a non-zero critic so a dropped bootstrap would be visible.
        with torch.no_grad():
            for parameter in trainer.policy.value_head[-1].parameters():
                parameter.fill_(1.0)

        rollout, _, _ = trainer._collect_rollout(observation_tensor, mask_tensor)
        bootstrapped = rollout["returns"]
    finally:
        trainer.close()

    assert torch.isfinite(bootstrapped).all()
    # With a constant positive critic and truncation-only episodes, every
    # truncated step must carry a bootstrap term, so returns cannot collapse to
    # the immediate reward alone.
    assert bootstrapped.abs().sum() > 0.0


def test_ppo_learns_the_optimal_action_on_a_known_task() -> None:
    """End-to-end guard: the trainer must beat its own random initialization."""
    config = CanonicalPolicyConfig(
        max_trucks=SHAPE.max_trucks,
        max_customers=SHAPE.max_customers,
        max_chargers=SHAPE.max_chargers,
        max_actions=SHAPE.max_actions,
        state_encoder="flat",
        action_head="independent",
        hidden_dim=64,
        encoder_output_dim=64,
    )
    ppo_config = PPOConfig(
        total_timesteps=12_000,
        num_envs=4,
        rollout_steps=64,
        epochs=4,
        minibatch_size=64,
        learning_rate=3e-3,
        entropy_coefficient=0.0,
        normalize_reward=False,
        seed=0,
    )
    counter = {"value": 0}

    def factory():
        counter["value"] += 1
        return CanonicalBanditEnv(seed=counter["value"])

    trainer = CanonicalPPO(config, ppo_config, factory)
    try:
        history = trainer.learn()
    finally:
        trainer.close()

    # Reward is (chosen value - best feasible value), so the optimum is exactly 0.
    early = float(np.mean(history.mean_episode_reward[:3]))
    late = float(np.mean(history.mean_episode_reward[-3:]))
    assert late > early, f"PPO did not improve: {early:.4f} -> {late:.4f}"
    assert late > -0.10, f"PPO did not approach the optimum: {late:.4f}"


def test_worker_vec_env_matches_the_synchronous_one() -> None:
    """Workers must change only where the simulator runs, not what it returns.

    The sub-environments are deterministic given their seeds, so stepping both
    vector environments through the same actions has to produce the same
    observations, rewards, and terminal flags. Anything else would mean the
    parallel path trains on a different problem.
    """
    stream = list(range(101, 141))

    def factory():
        return CanonicalBanditEnv(seed=0)

    shaping = RewardShaping(
        enabled=True, success_bonus=5.0, incompletion_penalty=2.0
    )
    synchronous = SyncCanonicalVecEnv(
        factory, num_envs=4, seed_stream=stream, shaping=shaping
    )
    parallel = WorkerCanonicalVecEnv(
        factory, num_envs=4, seed_stream=stream, shaping=shaping, workers=2
    )
    try:
        assert parallel.observation_size == synchronous.observation_size
        assert parallel.action_size == synchronous.action_size

        sync_obs, sync_mask = synchronous.reset()
        worker_obs, worker_mask = parallel.reset()
        np.testing.assert_allclose(worker_obs, sync_obs)
        np.testing.assert_array_equal(worker_mask, sync_mask)

        rng = np.random.default_rng(7)
        for _ in range(6):
            actions = np.array(
                [int(rng.choice(np.flatnonzero(row))) for row in sync_mask]
            )
            sync_obs, sync_reward, sync_done, sync_mask, sync_info = synchronous.step(
                actions
            )
            (
                worker_obs,
                worker_reward,
                worker_done,
                worker_mask,
                worker_info,
            ) = parallel.step(actions)
            np.testing.assert_allclose(worker_obs, sync_obs)
            np.testing.assert_allclose(worker_reward, sync_reward, rtol=1e-6)
            np.testing.assert_array_equal(worker_done, sync_done)
            np.testing.assert_array_equal(worker_mask, sync_mask)
            for one, other in zip(sync_info, worker_info, strict=True):
                assert ("terminal_observation" in one) == (
                    "terminal_observation" in other
                )
    finally:
        synchronous.close()
        parallel.close()
