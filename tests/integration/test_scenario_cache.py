"""A cached certified scenario must be indistinguishable from one generated on the spot."""

import os

import numpy as np
import pytest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv


def _episode(env, seed):
    obs, info = env.reset(seed=seed)
    instance = env.joint_instance
    return (instance.depot_node, [t.node_id for t in instance.tasks], instance.witness_routes,
            tuple(env.feasibility_witness), info["scenario"]["seed"]), obs


@pytest.mark.integration
def test_scenario_cache_round_trip(tmp_path, monkeypatch):
    config = "EVRoutingEnv/config_files/config_joint.yaml"
    monkeypatch.delenv("EVRP_SCENARIO_CACHE", raising=False)
    reference = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    expected = {seed: _episode(reference, seed) for seed in (31, 32)}
    reference.close()

    monkeypatch.setenv("EVRP_SCENARIO_CACHE", str(tmp_path))
    writer = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    first = {seed: _episode(writer, seed) for seed in (31, 32)}
    assert len(list(tmp_path.rglob("*.pkl"))) == 2
    reader = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    replay = {seed: _episode(reader, seed) for seed in (31, 32)}
    for seed in (31, 32):
        assert first[seed][0] == expected[seed][0] == replay[seed][0]
        np.testing.assert_array_equal(replay[seed][1], expected[seed][1])
    # a different configuration must not reuse the stored instance
    other = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    other.config["truck"]["battery_capacity"] = 450.0
    assert other._scenario_cache_path([450.0, 450.0]) != writer._scenario_cache_path([450.0, 450.0])
