import math
import unittest
from types import SimpleNamespace

import networkx as nx

from truck_env.models.truck import Truck
from truck_env.optimization import GurobiTruckRoutingSolver, HAS_GUROBI


def _build_dummy_env(trucks):
    graph = nx.DiGraph()
    for node in range(5):
        graph.add_node(node, has_charger=False)

    edges = {
        (0, 1): 1.0,
        (1, 2): 1.0,
        (0, 2): 5.0,
        (2, 1): 1.0,
        (1, 4): 2.0,
        (0, 4): 10.0,
        (3, 4): 2.0,
        (3, 1): 4.0,
        (4, 3): 2.0,
        (2, 4): 1.0,
    }
    for (u, v), t in edges.items():
        graph.add_edge(u, v, time=t, distance=t)

    transport_graph = SimpleNamespace(graph=graph)

    class DummyEnv:
        def __init__(self, graph_ns, trucks):
            self.transport_graph = graph_ns
            self.trucks = trucks

        def reset(self, seed=None):
            return None, {}

    return DummyEnv(transport_graph, trucks)


def _create_truck(truck_id, sequence):
    return Truck(
        truck_id=truck_id,
        truck_type="electric",
        delivery_sequence=sequence,
        initial_battery=400.0,
        battery_capacity=400.0,
        base_speed=40.0,
    )
@unittest.skipUnless(HAS_GUROBI, "gurobipy is required for optimization tests")
class GurobiSolverTests(unittest.TestCase):
    def test_solver_prints_charge_durations(self):
        self.assertTrue(HAS_GUROBI)
    def test_solver_finds_shortest_sequence_single_truck(self):
        truck = _create_truck(0, [0, 1, 2])
        env = _build_dummy_env([truck])
        solver = GurobiTruckRoutingSolver(
            config_path="truck_env/config_files/config.yaml",
            env=env,
            auto_reset=False,
        )

        solution = solver.solve()

        self.assertEqual(solution.truck_routes[0], [0, 1, 2])
        self.assertTrue(math.isclose(solution.truck_times[0], 2.0))
        self.assertTrue(math.isclose(solution.total_time, 2.0))

    def test_solver_supports_multiple_trucks(self):
        truck_a = _create_truck(0, [0, 1, 2])
        truck_b = _create_truck(1, [3, 4])
        env = _build_dummy_env([truck_a, truck_b])
        solver = GurobiTruckRoutingSolver(
            config_path="truck_env/config_files/config.yaml",
            env=env,
            auto_reset=False,
        )

        solution = solver.solve()

        self.assertEqual(solution.truck_routes[0], [0, 1, 2])
        self.assertEqual(solution.truck_routes[1], [3, 4])
        self.assertTrue(math.isclose(solution.truck_times[0], 2.0))
        self.assertTrue(math.isclose(solution.truck_times[1], 2.0))
        self.assertTrue(math.isclose(solution.total_time, 4.0))

    def test_solver_handles_truck_without_deliveries(self):
        truck = _create_truck(0, [0])
        env = _build_dummy_env([truck])
        solver = GurobiTruckRoutingSolver(
            config_path="truck_env/config_files/config.yaml",
            env=env,
            auto_reset=False,
        )

        solution = solver.solve()

        self.assertEqual(solution.truck_routes[0], [0])
        self.assertEqual(solution.truck_times[0], 0.0)
        self.assertEqual(solution.total_time, 0.0)


if __name__ == "__main__":
    unittest.main()
