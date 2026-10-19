"""Matrix-backed transportation graph used by the 10^4-node case studies."""

import json
import math

import numpy as np
import pytest

from EVRoutingEnv.models.core.matrix_transportation_graph import MatrixTransportationGraph, is_matrix_network
from EVRoutingEnv.utils.utils import get_graph, load_config


@pytest.fixture()
def folder(tmp_path):
    rng = np.random.default_rng(0)
    n = 6
    energy = rng.uniform(5, 50, (n, n)).astype(np.float32)
    hours = rng.uniform(0.1, 1.5, (n, n)).astype(np.float32)
    np.fill_diagonal(energy, 0); np.fill_diagonal(hours, 0)
    np.save(tmp_path / "od_energy_kwh.npy", energy)
    np.save(tmp_path / "od_time_h.npy", hours)
    ids = np.array([101, 205, 333, 4000, 5123, 90000], np.int64)
    np.save(tmp_path / "od_node_ids.npy", ids)
    json.dump({"205": {"station_type": "DCFast", "total_capacity": 8.0},
               "5123": {"station_type": "DCFast", "total_capacity": 3.0}}, open(tmp_path / "station_info_dict.json", "w"))
    network = {"shortest_path_energy_file": "od_energy_kwh.npy", "shortest_path_time_file": "od_time_h.npy",
               "station_info_file": "station_info_dict.json"}
    return tmp_path, network, energy, hours, ids


def test_costs_chargers_and_paths_match_the_files(folder):
    path, network, energy, hours, ids = folder
    assert is_matrix_network(network) and not is_matrix_network({"shortest_path_energy_file": "x.json"})
    g = MatrixTransportationGraph.from_config(network, str(path))
    assert g.num_nodes == 6 and g.get_all_nodes() == list(range(6))
    assert g.get_charging_nodes() == [1, 4]            # internal index = rank of the OSM id
    assert g.get_charger_details()[1] == {"original_id": 205, "types": {"DCFast": 8}}
    assert g.get_charger_capacity(4) == 3 and g.get_charger_type(1) == "DCFast" and g.get_charger_info(0) == {}
    for a, b in ((0, 3), (5, 1), (2, 2)):
        want_e, want_t = (0.0, 0.0) if a == b else (float(energy[a, b]), float(hours[a, b]))
        assert g.get_leg_costs(a, b) == (want_e, want_t)
        assert g.get_path_energy(a, b) == want_e and g.get_time_distance(a, b) == want_t
    assert g.get_shortest_path(0, 3) == [0, 3] and g.get_shortest_path(2, 2) == []
    assert g.get_leg_costs(0, 99) == (math.inf, math.inf) and g.get_shortest_path(0, 99) == []
    nearest, e = g.get_nearest_charging_node(0)
    assert nearest == min(g.get_charging_nodes(), key=lambda c: energy[0, c]) and e == float(energy[0, nearest])


def test_graph_view_answers_what_the_environment_asks(folder):
    path, network, *_ = folder
    g = MatrixTransportationGraph.from_config(network, str(path))
    assert 3 in g.graph and 6 not in g.graph and len(g.graph) == g.graph.number_of_nodes() == 6
    assert g.graph.in_degree(2) == g.graph.out_degree(2) == 5
    assert list(g.graph.successors(2)) == [0, 1, 3, 4, 5]
    assert g.graph.nodes[4]["original_id"] == 5123 and g.graph.nodes[4]["has_charger"] and not g.graph.nodes[0]["has_charger"]


def test_cost_matrices_equal_per_pair_lookups_and_scale_network_rescales_both(folder):
    path, network, energy, hours, ids = folder
    g = MatrixTransportationGraph.from_config(network, str(path))
    index, E, T = g.cost_matrices()
    assert index == {i: i for i in range(6)}
    for a in range(6):
        for b in range(6):
            assert g.get_leg_costs(a, b) == (float(E[a, b]), float(T[a, b]))
    dense_index, dense = g.dense_transport_matrix()
    assert dense.shape == (6, 6, 3) and dense[..., 2].all()
    g.scale_network(time_scale=2.0, energy_scale=0.5)
    _, E2, T2 = g.cost_matrices()
    np.testing.assert_allclose(E2, energy * 0.5, rtol=1e-6)
    np.testing.assert_allclose(T2, hours * 2.0, rtol=1e-6)
    assert g.get_leg_costs(0, 3) == (float(E2[0, 3]), float(T2[0, 3]))
    assert np.load(path / "od_energy_kwh.npy").dtype == np.float32 and np.array_equal(np.load(path / "od_energy_kwh.npy"), energy)


def test_unusable_entries_become_unreachable(folder):
    path, network, energy, hours, ids = folder
    bad = energy.copy(); bad[0, 1] = -1.0; bad[2, 3] = np.nan
    g = MatrixTransportationGraph(bad, hours, ids, {})
    assert g.get_leg_costs(0, 1) == (math.inf, math.inf) and g.get_leg_costs(2, 3) == (math.inf, math.inf)
    _, E, _ = g.cost_matrices()
    assert math.isinf(E[0, 1]) and math.isinf(E[2, 3]) and np.isfinite(E[1, 0])


def test_networkx_loader_refuses_matrix_configs():
    config = load_config("EVRoutingEnv/config_files/config_joint.yaml")
    config["network"] = {**config["network"], "shortest_path_energy_file": "od_energy_kwh.npy"}
    with pytest.raises(ValueError, match="matrix"):
        get_graph(config)
