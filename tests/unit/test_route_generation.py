"""Failed route searches must never become runnable benchmark instances."""

import networkx as nx
import numpy as np
import pytest

from EVRoutingEnv.models.environment.loaders import create_truck


class TwoNodeNetwork:
    def __init__(self, energy):
        self.graph = nx.DiGraph([(0, 1), (1, 0)])
        self.energy = energy

    def get_all_nodes(self):
        return list(self.graph)

    def generate_delivery_sequence(self, *, start_node, **kwargs):
        return [start_node, 1 - start_node]

    def get_path_energy(self, origin, destination):
        return 0.0 if origin == destination else self.energy


def generate(energy, *, initial="full", flexible=False):
    return create_truck(
        truck_id=0,
        transport_graph=TwoNodeNetwork(energy),
        config={
            "truck": {
                "battery_capacity": 100.0,
                "base_speed": 40.0,
                "initial_battery": initial,
            }
        },
        num_stops=1,
        min_hop_distance=0.1,
        max_hop_distance=1.0,
        charging_nodes=[],
        enable_flexible_delivery_order=flexible,
        rng=np.random.default_rng(7),
    )


@pytest.mark.parametrize("energy", [200.0, -1.0, float("inf"), float("nan")])
def test_failed_route_search_raises_instead_of_returning_last_candidate(energy):
    with pytest.raises(ValueError, match="could not generate a battery-feasible"):
        generate(energy)


def test_generation_uses_actual_initial_charge():
    with pytest.raises(ValueError, match="could not generate a battery-feasible"):
        generate(60.0, initial=20.0)
    truck, sequence, start = generate(60.0)
    assert truck.current_battery == 100.0
    assert sequence == [start, 1 - start]


def test_flexible_route_requires_enough_energy_for_depot_return():
    with pytest.raises(ValueError, match="could not generate a battery-feasible"):
        generate(60.0, flexible=True)
    truck, _, _ = generate(40.0, flexible=True)
    assert truck.enable_flexible_delivery_order
