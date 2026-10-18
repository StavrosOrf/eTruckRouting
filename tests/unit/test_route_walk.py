"""Independent checks of route-first witnesses and reusable map preparation."""

from itertools import permutations, product

import networkx as nx
import numpy as np
import pytest

from EVRoutingEnv.models.environment.joint_instance import (
    generate_joint_routing_instance,
)
from EVRoutingEnv.models.environment.route_walk import RouteWalkTopology
from scripts.analysis.analyze_scenario_capacity import feasible_pairs


class MatrixRoad:
    def __init__(self, customers=150, chargers=8):
        count = customers + chargers + 1
        self.graph = nx.star_graph(count - 1).to_directed()
        self.chargers = list(range(1, chargers + 1))
        nodes = np.arange(count)
        # Directed, nonnegative costs; the hub's incident legs are cheaper.
        energy = np.where((nodes[:, None] == 0) | (nodes[None, :] == 0), 15.0, 30.0)
        energy += (nodes[:, None] % 5) * 0.5
        np.fill_diagonal(energy, 0.0)
        self.matrices = ({i: i for i in nodes.tolist()}, energy, energy / 100.0)
        self.lookups = 0

    def get_all_nodes(self):
        return list(self.graph)

    def cost_matrices(self):
        return self.matrices

    def get_path_energy(self, a, b):
        self.lookups += 1
        return float(self.matrices[1][a, b])


def generate(graph, seed=17, trucks=10, customers=100, **kwargs):
    return generate_joint_routing_instance(
        graph,
        graph.chargers,
        np.random.default_rng(seed),
        num_customers=customers,
        num_trucks=trucks,
        battery_capacity=120.0,
        payload_capacity=max(20.0, 4.0 * customers / trucks),
        min_customer_demand=1.0,
        max_customer_demand=4.0,
        base_service_time=0.2,
        energy_multiplier=1.2,
        **kwargs,
    )


def assert_witness(graph, instance, trucks, initial=None):
    """Account for each action independently of the sampler's reserve tables."""
    tasks = {task.node_id: task for task in instance.tasks}
    seen = set()
    for tid, route in enumerate(instance.witness_routes):
        node = instance.depot_node
        battery = 120.0 if initial is None else initial[tid]
        payload = 0.0
        sequence = []
        stations_since_stop = 0
        for target, value, charging in route:
            if charging:
                assert target == node and target in graph.chargers
                assert value == 1.0
                assert battery < 120.0
                battery = 120.0
            else:
                required = graph.get_path_energy(node, target) * 1.2
                assert np.isfinite(required) and 0.0 <= required <= battery + 1e-9
                battery -= required
                node = target
                if target in graph.chargers:
                    stations_since_stop += 1
                    assert stations_since_stop <= 1
                else:
                    stations_since_stop = 0
                if target in tasks:
                    assert target not in seen
                    seen.add(target)
                    payload += tasks[target].demand
                    sequence.append(target)
                    assert instance.witness_owners[tasks[target].task_id] == tid
        assert node == instance.depot_node
        assert payload <= max(20.0, 4.0 * len(tasks) / trucks) + 1e-9
        assigned = [task for task in tasks.values() if task.preassigned_to == tid]
        if assigned and assigned[0].sequence_position is not None:
            assert sequence == [
                task.node_id
                for task in sorted(assigned, key=lambda t: t.sequence_position)
            ]
    assert seen == set(tasks)


@pytest.mark.parametrize(
    "trucks,customers", [(1, 20), (10, 100), (100, 2000), (128, 3)]
)
@pytest.mark.parametrize("mode", ["fleet", "preassigned", "sequential"])
def test_closed_walks_cover_unique_customers_with_feasible_energy_and_payload(
    trucks, customers, mode
):
    graph = MatrixRoad(customers + 10)
    instance = generate(
        graph,
        trucks=trucks,
        customers=customers,
        preassign=mode != "fleet",
        sequential=mode == "sequential",
    )
    assert_witness(graph, instance, trucks)
    assert len(instance.tasks) == customers
    assert instance.generation_attempts_used == 1
    if mode == "sequential":
        for tid in range(trucks):
            positions = [
                task.sequence_position
                for task in instance.tasks
                if task.preassigned_to == tid
            ]
            assert positions == list(range(len(positions)))


def test_sampling_uses_cached_matrices_and_never_searches_for_customer_routes(
    monkeypatch,
):
    from EVRoutingEnv.models.environment import route_certificate

    monkeypatch.setattr(
        route_certificate,
        "construct_energy_routes",
        lambda *args, **kwargs: pytest.fail("sample-first charger search was called"),
    )
    graph = MatrixRoad()
    first = generate(graph)
    topology = graph._route_walk_topology[2]
    generate(graph, seed=99)
    replay = generate(graph)
    assert first == replay
    assert graph._route_walk_topology[2] is topology
    assert graph.lookups == 0
    assert generate(MatrixRoad()) == first


def test_preparation_invalidates_when_network_costs_change():
    graph = MatrixRoad()
    generate(graph)
    old = graph._route_walk_topology[2]
    index, energy, hours = graph.matrices
    graph.matrices = (index, energy * 20.0, hours)
    with pytest.raises(ValueError, match="could not certify"):
        generate(graph, generation_attempts=2)
    assert graph._route_walk_topology[2] is not old


def test_negative_legs_cannot_supply_energy_or_be_used_as_exits():
    graph = MatrixRoad(customers=12, chargers=2)
    graph.matrices[1][3, 4] = -50.0
    graph.matrices[1][5, 1] = -50.0
    for seed in range(8):
        instance = generate(graph, seed=seed, trucks=2, customers=8)
        assert_witness(graph, instance, 2)


def test_insufficient_nodes_fail_before_any_retry():
    graph = MatrixRoad(customers=4)
    with pytest.raises(ValueError, match="too few eligible"):
        generate(graph, trucks=2, customers=5)


def test_return_cache_is_bounded_and_cannot_change_seeded_sampling():
    graph = MatrixRoad(customers=80)
    expected = generate(graph, trucks=2, customers=8)
    for seed in range(40):
        generate(graph, seed=seed, trucks=2, customers=8)
    assert generate(graph, trucks=2, customers=8) == expected
    assert (
        len(graph._route_walk_topology[2]._returns)
        <= RouteWalkTopology.DEPOT_CACHE_SIZE
    )


def test_return_anchors_exclude_stations_that_need_another_station():
    graph = MatrixRoad(customers=2, chargers=2)
    energy = graph.matrices[1]
    # Station 1 reaches the depot only via station 2 on this directed map.
    energy[1, 0], energy[1, 2], energy[2, 0] = 150.0, 30.0, 60.0
    topology = RouteWalkTopology.for_graph(graph, graph.chargers, 120.0, 1.0)
    home = topology._return_paths(0, np.array([120.0, 120.0]))
    assert home.chargers.tolist() == [2]
    # Customer 3 cannot reserve only its cheap hop to station 1.
    energy[3, 1], energy[3, 2], energy[3, 0] = 1.0, 50.0, 100.0
    graph.matrices = (graph.matrices[0], energy.copy(), graph.matrices[2])
    topology = RouteWalkTopology.for_graph(graph, graph.chargers, 120.0, 1.0)
    assert topology._return_paths(0, np.array([120.0, 120.0])).escape[3] == 50.0


def test_routes_requiring_two_stations_are_rejected_in_both_directions():
    graph = MatrixRoad(customers=1, chargers=2)
    graph.graph = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0)])
    nx.set_edge_attributes(
        graph.graph,
        {(0, 1): 80.0, (1, 2): 80.0, (2, 3): 80.0, (3, 0): 20.0},
        "energy",
    )
    energy = np.array(
        [
            [
                nx.shortest_path_length(graph.graph, a, b, weight="energy")
                for b in range(4)
            ]
            for a in range(4)
        ]
    )
    graph.matrices = ({n: n for n in range(4)}, energy, energy / 100.0)
    # Either eligible depot forces a two-station outbound or final return leg.
    for seed in range(4):
        with pytest.raises(ValueError, match="one.charg"):
            generate(graph, seed=seed, trucks=1, customers=1, generation_attempts=2)


def test_far_delivery_can_return_via_another_delivery_without_chaining_stations():
    graph = MatrixRoad(customers=2, chargers=3)
    graph.chargers = [1, 3, 5]
    edges = [
        (0, 1, 80.0),
        (1, 2, 40.0),
        (2, 3, 40.0),
        (3, 4, 40.0),
        (4, 5, 40.0),
        (5, 0, 40.0),
    ]
    graph.graph = nx.DiGraph()
    graph.graph.add_weighted_edges_from(edges, weight="energy")
    energy = np.array(
        [
            [
                nx.shortest_path_length(graph.graph, a, b, weight="energy")
                for b in range(6)
            ]
            for a in range(6)
        ]
    )
    graph.matrices = ({n: n for n in range(6)}, energy, energy / 100.0)
    instance = generate(graph, seed=11, trucks=1, customers=2, generation_attempts=1)
    assert instance.depot_node == 0
    # Far customer 2 cannot return via a single station, but customer 4 allows
    # a feasible journey home with at most one station between service stops.
    assert [
        node for node, _, charging in instance.witness_routes[0] if not charging
    ] == [1, 2, 3, 4, 5, 0]
    assert_witness(graph, instance, 1)
    topology = graph._route_walk_topology[2]
    assert topology._two_delivery_customer_bound(topology.index[0]) == 2


def test_two_delivery_bound_rejects_before_building_partial_routes():
    graph = MatrixRoad(customers=2, chargers=0)
    index, _, hours = graph.matrices
    # Each leg is affordable, but a three-leg round with two distinct
    # customers cannot fit, and no charger exists.
    energy = np.full((3, 3), 40.0)
    np.fill_diagonal(energy, 0.0)
    graph.matrices = (index, energy, hours)
    topology = RouteWalkTopology.for_graph(graph, [], 100.0, 1.0)
    assert topology._two_delivery_customer_bound(0) == 0
    assert topology._two_delivery_bounds == {0: 0}
    for seed in range(3):
        with pytest.raises(ValueError, match="two.delivery"):
            generate(graph, seed=seed, trucks=1, customers=2, generation_attempts=2)


@pytest.mark.parametrize("seed", range(4))
def test_two_delivery_bound_matches_independent_segment_enumeration(seed):
    graph = MatrixRoad(customers=4, chargers=2)
    index, _, hours = graph.matrices
    energy = np.random.default_rng(seed).integers(5, 90, size=(7, 7)).astype(float)
    np.fill_diagonal(energy, 0.0)
    graph.matrices = (index, energy, hours)
    topology = RouteWalkTopology.for_graph(graph, graph.chargers, 100.0, 1.0)
    for depot in (0, 3):
        customers = [n for n in topology.customers if n != depot]
        possible = set()
        possible_pairs = set()
        for first, second in permutations(customers, 2):
            for stations in product([None, *graph.chargers], repeat=3):
                node, battery = depot, 100.0
                feasible = True
                for stop, station in zip((first, second, depot), stations, strict=True):
                    targets = (stop,) if station is None else (station, stop)
                    for target in targets:
                        battery -= energy[node, target]
                        if battery < -1e-9:
                            feasible = False
                            break
                        if target in graph.chargers:
                            battery = 100.0
                        node = target
                    if not feasible:
                        break
                if feasible:
                    possible.update((first, second))
                    possible_pairs.add((first, second))
                    break
        assert topology._two_delivery_customer_bound(depot) == len(possible)
        ordered, single, pairs = feasible_pairs(topology, depot)
        assert ordered.tolist() == customers
        assert {
            (int(ordered[a]), int(ordered[b]))
            for a, b in zip(*np.nonzero(pairs), strict=True)
        } == possible_pairs
        for index, customer in enumerate(ordered):
            possible_single = False
            for stations in product([None, *graph.chargers], repeat=2):
                node, battery = depot, 100.0
                feasible = True
                for stop, station in zip((customer, depot), stations, strict=True):
                    targets = (stop,) if station is None else (station, stop)
                    for target in targets:
                        battery -= energy[node, target]
                        if battery < -1e-9:
                            feasible = False
                            break
                        if target in graph.chargers:
                            battery = 100.0
                        node = target
                    if not feasible:
                        break
                if feasible:
                    possible_single = True
                    break
            assert bool(single[index]) == possible_single
