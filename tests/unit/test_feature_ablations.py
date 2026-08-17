"""Component ablations must remove content without changing shapes."""

import os
from copy import deepcopy

import numpy as np
import pytest
import torch


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from algo.canonical_policy import CanonicalActorCritic, CanonicalPolicyConfig
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.features import (
    ABLATION_BLOCKS,
    CHARGER_FEATURES,
    TRUCK_FEATURES,
)
from EVRoutingEnv.utils.utils import load_config


def _env(**environment_overrides) -> EventDrivenTruckEnv:
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["environment"].update(environment_overrides)
    return EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)


def test_unknown_ablation_block_is_rejected() -> None:
    env = _env(ablate_features=["not_a_block"])
    try:
        with pytest.raises(ValueError, match="unknown ablation blocks"):
            env.reset(seed=1)
    finally:
        env.close()


def test_queue_ablation_zeroes_only_the_queue_columns() -> None:
    control, ablated = _env(), _env(ablate_features=["queue"])
    try:
        control.reset(seed=808)
        ablated.reset(seed=808)
        before = control.get_canonical_features()
        after = ablated.get_canonical_features()

        queue_columns = [
            CHARGER_FEATURES.index(name) for name in ABLATION_BLOCKS["queue"][1]
        ]
        assert np.allclose(after.charger_features[:, queue_columns], 0.0)
        # Everything else on the charger rows is untouched.
        other = [
            index
            for index in range(len(CHARGER_FEATURES))
            if index not in queue_columns
        ]
        assert np.allclose(
            after.charger_features[:, other], before.charger_features[:, other]
        )
        assert np.allclose(after.truck_features, before.truck_features)
    finally:
        control.close()
        ablated.close()


def test_active_truck_ablation_hides_which_truck_is_deciding() -> None:
    control, ablated = _env(), _env(ablate_features=["active_truck"])
    try:
        control.reset(seed=99)
        ablated.reset(seed=99)
        column = TRUCK_FEATURES.index("is_active")
        assert control.get_canonical_features().truck_features[:, column].sum() > 0
        assert np.allclose(
            ablated.get_canonical_features().truck_features[:, column], 0.0
        )
    finally:
        control.close()
        ablated.close()


def test_edge_ablation_blanks_every_pairwise_relation() -> None:
    control, ablated = _env(), _env(ablate_features=["edges"])
    try:
        control.reset(seed=77)
        ablated.reset(seed=77)
        before = control.get_canonical_features().pairwise_features
        after = ablated.get_canonical_features().pairwise_features
        assert any(np.abs(value).sum() > 0 for value in before.values())
        assert all(np.abs(value).sum() == 0 for value in after.values())
    finally:
        control.close()
        ablated.close()


def test_ablations_do_not_change_the_observation_width() -> None:
    control = _env()
    ablated = _env(ablate_features=["queue", "active_truck", "edges", "depot"])
    try:
        baseline, _ = control.reset(seed=4)
        other, _ = ablated.reset(seed=4)
        assert baseline.shape == other.shape
        assert not np.allclose(baseline, other), "ablation changed nothing"
    finally:
        control.close()
        ablated.close()


def test_pooling_ablation_keeps_the_critic_but_changes_the_actor() -> None:
    config = CanonicalPolicyConfig(
        max_trucks=2, max_customers=4, max_chargers=3, max_actions=10
    )
    ablated_config = CanonicalPolicyConfig(
        max_trucks=2,
        max_customers=4,
        max_chargers=3,
        max_actions=10,
        ablate_state_pooling=True,
    )
    full = CanonicalActorCritic(config).eval()
    ablated = CanonicalActorCritic(ablated_config).eval()
    ablated.load_state_dict(full.state_dict())

    generator = torch.Generator().manual_seed(0)
    observation = torch.rand(
        (3, config.shape.flat_size), generator=generator, dtype=torch.float32
    )
    # Make sure at least one action per row is feasible and unpadded.
    slices = ablated.unpack(observation)
    assert slices.action_padding_mask.any()

    with torch.no_grad():
        baseline = full(observation)
        without_pooling = ablated(observation)

    torch.testing.assert_close(baseline.values, without_pooling.values)
    assert not torch.allclose(baseline.logits, without_pooling.logits)
