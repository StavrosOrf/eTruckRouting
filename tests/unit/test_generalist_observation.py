"""The generalist view keeps the hard mask, drops identifiers and bounds chargers."""

import numpy as np
import pytest
from generalist_harness import BLOCKS, random_observations, small_env

from algo.generalist_observation import (
    ACTION_FEATURE_NAMES,
    CHARGER_FEATURE_NAMES,
    FEATURE_DIMS,
    TASK_FEATURE_NAMES,
    TRUCK_FEATURE_NAMES,
    NODE_TYPES,
    ObservationSpec,
    collate,
)
from EVRoutingEnv.state.action_mask import get_action_mask
from EVRoutingEnv.state.feasibility import ActionKind, joint_action_feasibility


@pytest.mark.parametrize("component", sorted(BLOCKS))
def test_candidates_are_the_hard_mask_inside_the_charger_subgraph(component) -> None:
    env = small_env(component, spec=ObservationSpec(charger_budget=4, fleet_assignment=True))
    observation, _ = env.reset(seed=3)
    for _ in range(25):
        decisions = joint_action_feasibility(env)
        mask = get_action_mask(env)
        assert observation.trucks.shape[1] == FEATURE_DIMS["truck"]
        assert observation.tasks.shape[1] == FEATURE_DIMS["task"]
        assert observation.chargers.shape[1] == FEATURE_DIMS["charger"]
        assert observation.actions.shape == (observation.num_actions, FEATURE_DIMS["action"])
        assert len(set(observation.action_index.tolist())) == observation.num_actions
        assert mask[observation.action_index].all()
        offered = set(observation.action_index.tolist())
        for action in np.flatnonzero(mask):
            if decisions[action].action_kind is not ActionKind.CHARGER:
                assert action in offered, decisions[action]
        for position, action in enumerate(observation.action_index):
            kind = decisions[action].action_kind
            if kind is ActionKind.CHARGER:
                assert 0 <= observation.action_charger[position] < len(observation.chargers)
            if kind is ActionKind.CUSTOMER:
                assert 0 <= observation.action_task[position] < len(observation.tasks)
        assert len(observation.chargers) <= 4 + 2 * len(env.trucks)
        size = len(observation.trucks) + len(observation.tasks) + len(observation.chargers)
        assert observation.edges.shape == (size, size, FEATURE_DIMS["edge"])
        for values in (observation.trucks, observation.tasks, observation.chargers, observation.context, observation.actions):
            assert np.isfinite(values).all()
        observation, _, terminated, truncated, _ = env.step(int(observation.action_index[0]))
        if terminated or truncated:
            break


def test_no_identifier_or_absolute_clock_is_a_feature() -> None:
    names = [*TRUCK_FEATURE_NAMES, *TASK_FEATURE_NAMES, *CHARGER_FEATURE_NAMES, *ACTION_FEATURE_NAMES]
    # Identity enters only as relations to the active truck or the depot.
    relational = [name for name in names if name.endswith("_active") or "depot" in name]
    for forbidden in ("node", "_id", "claimed_by", "owner", "_at", "started", "release_time"):
        assert not [name for name in names if forbidden in name and name not in relational], forbidden


def test_the_subgraph_reports_what_it_left_out() -> None:
    env = small_env("static", spec=ObservationSpec(charger_budget=2))
    observation, _ = env.reset(seed=0)
    from algo.generalist_observation import CONTEXT_FEATURE_NAMES

    excluded = observation.context[CONTEXT_FEATURE_NAMES.index("chargers_excluded_fraction")]
    assert len(env.charging_nodes) > 2 + 2 * len(env.trucks)
    assert excluded == pytest.approx(1 - len(observation.chargers) / len(env.charging_nodes))


def test_a_larger_budget_offers_every_feasible_charger() -> None:
    env = small_env("static", spec=ObservationSpec(charger_budget=10_000))
    observation, _ = env.reset(seed=0)
    assert set(observation.action_index.tolist()) == set(np.flatnonzero(get_action_mask(env)).tolist())


def test_collate_pads_each_type_and_keeps_the_edge_blocks() -> None:
    observations = random_observations(small_env("same_day"), seed=0, steps=6)
    observations += random_observations(small_env("static", trucks=3, stops=6), seed=1, steps=4)
    batch = collate(observations)
    for position, observation in enumerate(observations):
        counts = {"truck": len(observation.trucks), "task": len(observation.tasks), "charger": len(observation.chargers)}
        for node_type in NODE_TYPES:
            assert int(batch.node_masks[node_type][position].sum()) == counts[node_type]
        bounds = np.cumsum([0, counts["truck"], counts["task"], counts["charger"]])
        block = observation.edges[bounds[0]:bounds[1], bounds[2]:bounds[3]]
        stored = batch.pairwise[("truck", "charger")][position, : counts["truck"], : counts["charger"]]
        assert np.array_equal(stored.numpy(), block)
        assert int(batch.action_mask[position].sum()) == observation.num_actions
