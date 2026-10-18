"""Energy and time must describe the same executable route."""

import math
import random
from itertools import pairwise

import networkx as nx

from EVRoutingEnv.models.core.transportation_graph import TransportationGraph
from EVRoutingEnv.models.core.customer import CustomerTask, FleetTaskRegistry
from EVRoutingEnv.models.core.truck import Truck
from EVRoutingEnv.state.feasibility import FeasibilityReason, evaluate_joint_route
from EVRoutingEnv.utils.utils import get_graph, load_config


def test_energy_and_time_follow_one_path(tmp_path):
    graph = nx.DiGraph()
    graph.add_edge(0, 1, distance=100.0, time=1.0)
    graph.add_edge(0, 2, distance=1.0, time=50.0)
    graph.add_edge(2, 1, distance=1.0, time=50.0)
    transport = TransportationGraph(graph, precompute_distances=False)
    transport._cache_file = str(tmp_path / "paired-costs.pkl")

    assert transport.get_leg_costs(0, 1) == (2.0, 100.0)
    assert transport.get_path_energy(0, 1) == 2.0
    assert transport.get_time_distance(0, 1) == 100.0
    assert transport.get_shortest_path(0, 1) == [0, 2, 1]

    truck = Truck(0, "electric", [0, 1], 2.0, 2.0, 40.0, payload_capacity=1.0)
    tasks = FleetTaskRegistry([CustomerTask(0, 1, 1.0, 0.0, latest_service=10.0)])
    decision = evaluate_joint_route(
        truck=truck,
        truck_state="ready",
        target_node=1,
        transport_graph=transport,
        charging_nodes=[],
        task_registry=tasks,
        depot_node=0,
    )
    assert not decision.feasible
    assert decision.reason is FeasibilityReason.TIME_WINDOW_EXPIRED

    assert transport.save_distance_cache()
    restored = TransportationGraph(graph.copy(), precompute_distances=False)
    restored._cache_file = transport._cache_file
    assert restored.load_distance_cache()
    assert restored.get_leg_costs(0, 1) == (2.0, 100.0)
    assert restored.get_shortest_path(0, 1) == [0, 2, 1]

    restored.scale_network(time_scale=2.0, energy_scale=3.0)
    assert restored.get_leg_costs(0, 1) == (6.0, 200.0)
    assert restored.get_shortest_path(0, 1) == [0, 2, 1]


def test_negative_energy_edge_is_not_an_executable_shortcut():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, distance=-2.0, time=1.0)
    graph.add_edge(0, 2, distance=1.0, time=3.0)
    graph.add_edge(2, 1, distance=1.0, time=4.0)
    transport = TransportationGraph(graph, precompute_distances=False)

    assert transport.get_leg_costs(0, 1) == (2.0, 7.0)
    assert transport.get_shortest_path(0, 1) == [0, 2, 1]

    graph = nx.DiGraph()
    graph.add_edge(0, 1, distance=-2.0, time=1.0)
    transport = TransportationGraph(graph, precompute_distances=False)
    energy, travel_time = transport.get_leg_costs(0, 1)
    assert math.isinf(energy) and math.isinf(travel_time)
    assert transport.get_shortest_path(0, 1) == []
    assert transport.get_leg_costs(99, 99) == (math.inf, math.inf)
    assert transport.get_shortest_path(99, 99) == []


def test_equal_energy_routes_choose_shorter_time_independent_of_edge_order(tmp_path):
    edges = [
        (0, 1, {"distance": 2.0, "time": 20.0}),
        (0, 3, {"distance": 1.0, "time": 3.0}),
        (3, 1, {"distance": 1.0, "time": 4.0}),
        (0, 2, {"distance": 1.0, "time": 3.0}),
        (2, 1, {"distance": 1.0, "time": 4.0}),
    ]
    graph_a = nx.DiGraph()
    graph_a.add_edges_from(edges)
    graph_b = nx.DiGraph()
    graph_b.add_edges_from(reversed(edges))
    first = TransportationGraph(graph_a, precompute_distances=False)
    second = TransportationGraph(graph_b, precompute_distances=False)
    assert first._cache_fingerprint == second._cache_fingerprint
    assert first.get_leg_costs(0, 1) == second.get_leg_costs(0, 1) == (2.0, 7.0)
    assert first.get_shortest_path(0, 1) == second.get_shortest_path(0, 1) == [0, 2, 1]

    first._cache_file = str(tmp_path / "tie.pkl")
    second._cache_file = first._cache_file
    assert first.save_distance_cache()
    second.clear_distance_cache()
    assert second.load_distance_cache()
    assert second.get_shortest_path(0, 1) == [0, 2, 1]


def test_shipped_negative_energy_od_uses_an_executable_alternate_route():
    graph = get_graph(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    original_ids = {data["original_id"]: node for node, data in graph.nodes(data=True)}
    origin = original_ids[2923697738]
    target = original_ids[91030007]
    assert graph[origin][target]["distance"] < 0.0

    transport = TransportationGraph(graph, precompute_distances=False)
    energy, travel_time = transport.get_leg_costs(origin, target)
    path = transport.get_shortest_path(origin, target)
    assert math.isfinite(energy) and energy >= 0.0
    assert math.isfinite(travel_time) and travel_time >= 0.0
    assert path[0] == origin and path[-1] == target
    assert all(graph[u][v]["distance"] >= 0.0 for u, v in zip(path, path[1:]))


def test_prevalidated_edges_preserve_exhaustive_route_ordering():
    """Check energy/time/hop/lexicographic ties against independent enumeration."""
    for seed in range(4):
        rng = random.Random(seed)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(7))
        for a in graph:
            for b in graph:
                if a != b and rng.random() < 0.7:
                    graph.add_edge(a, b, distance=float(rng.randrange(4)), time=float(rng.randrange(4)))
        graph.add_edge(0, 1, distance=-1.0, time=0.0)
        usable = graph.copy()
        usable.remove_edge(0, 1)
        transport = TransportationGraph(graph, precompute_distances=False)
        for a in graph:
            for b in graph:
                candidates = [(0.0, 0.0, 0, (a,))] if a == b else []
                if a != b:
                    for path in nx.all_simple_paths(usable, a, b):
                        legs = [usable[u][v] for u, v in pairwise(path)]
                        candidates.append((sum(e["distance"] for e in legs),
                                           sum(e["time"] for e in legs), len(legs), tuple(path)))
                expected = min(candidates) if candidates else (math.inf, math.inf, 0, ())
                assert transport.get_leg_costs(a, b) == expected[:2]
                assert tuple(transport.get_shortest_path(a, b)) == expected[3]


def test_prevalidated_edges_and_matrices_refresh_after_scaling_and_mutation():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, distance=2.0, time=3.0)
    graph.add_edge(1, 0, distance=4.0, time=5.0)
    transport = TransportationGraph(graph, precompute_distances=False)
    assert transport.get_leg_costs(0, 1) == (2.0, 3.0)
    transport.scale_network(energy_scale=2.0, time_scale=3.0)
    # Source 1 was not populated before scaling: it must use refreshed edges.
    assert transport.get_leg_costs(1, 0) == (8.0, 15.0)
    old_matrix = transport.cost_matrices()
    graph[1][0]["distance"] = 7.0
    transport.clear_distance_cache()
    assert transport.get_leg_costs(1, 0) == (7.0, 15.0)
    assert transport.cost_matrices() is not old_matrix
