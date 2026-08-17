"""Scaling the road network has to move every consumer of it at once."""

import os
from copy import deepcopy

import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


def _env(**network_overrides) -> EventDrivenTruckEnv:
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config.setdefault("network", {}).update(network_overrides)
    return EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)


def test_travel_time_scale_moves_every_leg() -> None:
    base, scaled = _env(), _env(travel_time_scale=1.4)
    try:
        base.reset(seed=2_000_000_000)
        scaled.reset(seed=2_000_000_000)
        origin = int(base.trucks[0].current_node)
        targets = sorted(base.charging_nodes)[:5]
        for target in targets:
            reference = base.transport_graph.get_time_distance(origin, int(target))
            moved = scaled.transport_graph.get_time_distance(origin, int(target))
            assert moved == pytest.approx(reference * 1.4, rel=1e-6)
    finally:
        base.close()
        scaled.close()


def test_energy_scale_moves_shortest_path_energies() -> None:
    """Energies come from a precomputed cache, not from the live edges.

    Scaling the edges alone would leave every energy lookup unchanged, so the
    cache is scaled with them; this is the check that keeps the two in step.
    """
    base, scaled = _env(), _env(energy_scale=1.3)
    try:
        base.reset(seed=2_000_000_000)
        scaled.reset(seed=2_000_000_000)
        origin = int(base.trucks[0].current_node)
        for target in sorted(base.charging_nodes)[:5]:
            reference = base.transport_graph.get_path_energy(origin, int(target))
            moved = scaled.transport_graph.get_path_energy(origin, int(target))
            assert moved == pytest.approx(reference * 1.3, rel=1e-6)
    finally:
        base.close()
        scaled.close()


def test_scaling_is_rejected_when_not_positive() -> None:
    for override in ({"travel_time_scale": 0.0}, {"energy_scale": -1.0}):
        with pytest.raises(ValueError, match="must be positive"):
            _env(**override).close()


def test_default_configuration_is_untouched() -> None:
    """The scale hooks must be inert unless a campaign asks for them."""
    plain, explicit = _env(), _env(travel_time_scale=1.0, energy_scale=1.0)
    try:
        plain.reset(seed=7)
        explicit.reset(seed=7)
        origin = int(plain.trucks[0].current_node)
        target = int(sorted(plain.charging_nodes)[0])
        assert plain.transport_graph.get_path_energy(
            origin, target
        ) == pytest.approx(explicit.transport_graph.get_path_energy(origin, target))
        assert plain.transport_graph.get_time_distance(
            origin, target
        ) == pytest.approx(explicit.transport_graph.get_time_distance(origin, target))
    finally:
        plain.close()
        explicit.close()
