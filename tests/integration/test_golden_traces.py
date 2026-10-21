"""Legacy behaviour is frozen: replayed golden actions reproduce every hash.

The traces were recorded by ``scripts/analysis/record_golden_traces.py`` before
any Phase 1 change. A failure here means a change altered legacy trajectories;
that is only acceptable as its own reviewed commit that re-records the traces.
"""

import json

import pytest

from scripts.analysis.record_golden_traces import (
    CONFIGS,
    FEATURE_INDEX,
    INDEX,
    POLICIES,
    SEEDS,
    STREAMS,
    make_env,
    replay_feature_hashes,
    run_episode,
    stream_hashes,
)


pytestmark = pytest.mark.integration

GOLDEN = json.loads(INDEX.read_text())


def test_golden_traces_cover_every_frozen_config_policy_and_seed() -> None:
    recorded = {(row["config"], row["policy"], row["seed"]) for row in GOLDEN["episodes"]}
    assert recorded == {
        (config, policy, seed) for config in CONFIGS for policy in POLICIES for seed in SEEDS
    }
    assert GOLDEN["streams"] == list(STREAMS)


@pytest.fixture(scope="module")
def environments():
    cache = {}
    yield lambda name: cache.setdefault(name, make_env(name))
    for env in cache.values():
        env.close()


@pytest.mark.parametrize(
    "episode",
    GOLDEN["episodes"],
    ids=lambda row: f"{row['config'].removesuffix('.yaml')}-{row['policy']}-{row['seed']}",
)
def test_replay_reproduces_golden_hashes(environments, episode) -> None:
    trace = run_episode(environments(episode["config"]), episode["seed"], actions=episode["actions"])
    hashes = stream_hashes(trace)
    diverged = [stream for stream in STREAMS if hashes[stream] != episode["sha256"][stream]]
    assert not diverged, f"streams diverged from the golden trace: {diverged}"


FEATURES = json.loads(FEATURE_INDEX.read_text())


@pytest.mark.parametrize(
    "episode",
    FEATURES["episodes"],
    ids=lambda row: f"{row['config'].removesuffix('.yaml')}-{row['policy']}-{row['seed']}",
)
def test_v6_columns_of_current_schema_match_frozen_extraction(environments, episode) -> None:
    """With every new block disabled, the v6 columns of v7 are unchanged."""
    golden = next(
        row for row in GOLDEN["episodes"]
        if (row["config"], row["policy"], row["seed"])
        == (episode["config"], episode["policy"], episode["seed"])
    )
    replayed = replay_feature_hashes(
        environments(episode["config"]), golden, FEATURES["columns"]
    )
    assert len(replayed) == len(episode["steps"])
    for step, (actual, expected) in enumerate(zip(replayed, episode["steps"], strict=True)):
        assert actual == expected, f"feature blocks diverged at decision {step}"
