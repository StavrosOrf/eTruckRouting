"""The opt-in depot-energy ring keeps every customer at the configured distance."""

import os
from copy import deepcopy

import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


pytestmark = pytest.mark.integration


def _env(**environment) -> EventDrivenTruckEnv:
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["environment"].update(num_trucks=4, num_stops=8, **environment)
    return EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)


def test_unset_ring_keeps_original_instances() -> None:
    plain, explicit = _env(), _env(min_depot_energy_kwh=0.0, max_depot_energy_kwh=None)
    try:
        for seed in (3, 11):
            plain.reset(seed=seed)
            explicit.reset(seed=seed)
            assert plain.joint_instance == explicit.joint_instance
    finally:
        plain.close()
        explicit.close()


@pytest.mark.parametrize(("low", "high"), [(150.0, None), (100.0, 250.0)])
def test_ring_bounds_hold_in_both_directions(low, high) -> None:
    env = _env(min_depot_energy_kwh=low, max_depot_energy_kwh=high)
    try:
        for seed in (3, 11, 29):
            env.reset(seed=seed)
            depot = env.joint_instance.depot_node
            graph = env.transport_graph
            for task in env.joint_instance.tasks:
                distance = min(
                    graph.get_path_energy(depot, task.node_id),
                    graph.get_path_energy(task.node_id, depot),
                )
                assert distance >= low - 1e-6
                if high is not None:
                    assert distance <= high + 1e-6
    finally:
        env.close()
