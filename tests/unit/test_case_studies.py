"""Data contract of the generated case studies under EVRoutingEnv/data/case_studies."""

import json
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from EVRoutingEnv.utils.utils import get_graph, load_config

ROOT = Path(__file__).resolve().parents[2]
CASES = sorted(p for p in (ROOT / "EVRoutingEnv" / "data" / "case_studies").glob("*") if (p / "metadata.json").exists())
pytestmark = pytest.mark.skipif(not CASES, reason="no generated case studies")


def _load(path: Path) -> dict[tuple[int, int], float]:
    out = {}
    for key, value in json.load(open(path)).items():
        u, v = map(int, re.findall(r"-?\d+", key))
        out[(u, v)] = value
    return out


@pytest.fixture(params=CASES, ids=lambda p: p.name)
def case(request):
    path = request.param
    meta = json.load(open(path / "metadata.json"))
    return path, meta, load_config(str(ROOT / meta["config_file"]))


def _generated_only(meta):
    """The imported California data keeps its own quirks (missing pairs, Level 2 stations, negative legs)."""
    if not meta.get("generated", True):
        pytest.skip("applies to generated case studies only")


def test_files_follow_the_california_format(case):
    path, meta, _ = case
    if not meta.get("generated", True):
        energy = _load(path / "shortest_path_energy_dict.json")
        nodes = {u for u, _ in energy} | {v for _, v in energy}
        assert len(nodes) == meta["counts"]["nodes"]
        assert len(energy) == meta["counts"]["nodes"] * (meta["counts"]["nodes"] - 1) - meta["missing_pairs"]
        for name in ("shortest_path_time_dict.json", "station_info_dict.json", "waiting_time_lookup.json"):
            assert (path / name).exists()
        return
    for name in ("shortest_path_energy_dict.json", "shortest_path_time_dict.json",
                 "station_info_dict.json", "waiting_time_lookup.json", "metadata.json"):
        assert (path / name).exists(), name
    energy, time = _load(path / "shortest_path_energy_dict.json"), _load(path / "shortest_path_time_dict.json")
    nodes = {u for u, _ in energy} | {v for _, v in energy}
    n = meta["counts"]["nodes"]
    assert len(nodes) == n
    assert set(energy) == set(time) == {(u, v) for u in nodes for v in nodes if u != v}
    assert min(energy.values()) >= 0.0 and all(np.isfinite(list(energy.values())))
    assert min(time.values()) > 0.0 and all(np.isfinite(list(time.values())))


def test_chargers_are_dc_fast_only_with_port_counts(case):
    path, meta, _ = case
    _generated_only(meta)
    stations = json.load(open(path / "station_info_dict.json"))
    nodes = {n["node"]: n for n in json.load(open(path / "vis_data" / "nodes.json"))}
    assert len(stations) == meta["counts"]["chargers"] > 0
    for node, info in stations.items():
        assert info["station_type"] == "DCFast"
        assert info["total_capacity"] >= 1 and float(info["total_capacity"]).is_integer()
        assert nodes[int(node)]["role"] == "charger"
    ports = {s["node"]: s["ports"] for s in json.load(open(path / "vis_data" / "stations.json"))}
    assert {str(k): v for k, v in ports.items()} == {k: int(v["total_capacity"]) for k, v in stations.items()}
    assert sum(ports.values()) == meta["counts"]["dcfast_ports_in_nodes"]


def test_nearby_stations_were_merged(case):
    path, meta, _ = case
    _generated_only(meta)
    sites = json.load(open(path / "vis_data" / "stations.json"))
    assert meta["preset"]["charger_merge_m"] > 0
    # merged nodes keep every member station's ports and ids
    for s in sites:
        assert len(s["afdc_ids"]) >= s["n_stations"] >= 1
    assert any(s["n_stations"] > 1 for s in sites)


def test_network_is_asymmetric_in_energy_and_time(case):
    path, meta, _ = case
    od = np.load(path / "vis_data" / "od_matrices.npz")
    for key, floor, minimum in (("energy_kwh", 1.0, 0.30), ("time_h", 0.05, 0.05)):
        m = od[key].astype(float)  # NaN where the original data has no pair; comparisons below ignore them
        iu = np.triu_indices(len(m), 1)
        f, r = m[iu], m.T[iu]
        ok = (f > floor) & (r > floor)
        assert np.mean(np.abs(f[ok] - r[ok]) / np.maximum(f[ok], r[ok]) > 0.05) > minimum, key


def test_environment_graph_loads_and_routes_every_pair(case):
    path, meta, config = case
    graph = get_graph(config)
    assert graph.number_of_nodes() == meta["counts"]["nodes"]
    if meta.get("generated", True):
        assert nx.is_strongly_connected(graph)
    else:  # the original data has destination-only nodes (San Diego): the environment never uses them as depots
        outside = meta["nodes_outside_largest_strong_component"]
        core = max(nx.strongly_connected_components(graph), key=len)
        assert graph.number_of_nodes() - len(core) == len(outside) >= 0
    assert sum(1 for _, d in graph.nodes(data=True) if d["has_charger"]) == meta["counts"]["chargers"]
    chargers = [d["charger_type"] for _, d in graph.nodes(data=True) if d["has_charger"]]
    if meta.get("generated", True):
        assert all(set(c) == {"DCFast"} for c in chargers)
        assert sum(sum(c.values()) for c in chargers) == meta["counts"]["dcfast_ports_in_nodes"]
    else:  # California mixes Level 2 and DC-fast stations
        assert all(set(c) <= {"DCFast", "Level2"} for c in chargers)
        assert sum(sum(c.values()) for c in chargers) == meta["counts"]["charger_ports_total"]
    assert meta["battery_kwh"] == config["truck"]["battery_capacity"]
    california = load_config(str(ROOT / "EVRoutingEnv" / "config_files" / "config_joint.yaml"))
    assert config["truck"] == california["truck"] or meta["preset"].get("battery_kwh")
    assert config["charging"] == california["charging"] or meta["preset"].get("station_power_classes_kw")
    assert config["network"]["data_path"].rstrip("/").endswith(f"case_studies/{meta['name']}")


def test_every_dcfast_site_is_kept_unless_the_preset_limits_it(case):
    path, meta, _ = case
    _generated_only(meta)
    if meta["preset"]["max_chargers"] == 0:
        # sites that snap onto one road node are combined, so nodes <= sites; ports are never lost
        assert meta["counts"]["chargers"] <= meta["counts"]["dcfast_sites_available"]
        assert meta["counts"]["chargers"] >= 0.9 * meta["counts"]["dcfast_sites_available"]
        sites = json.load(open(path / "vis_data" / "all_dcfast_sites.json"))
        # a site is only left out if no road of the study network is within max_snap_m (e.g. Puget Sound islands)
        assert np.mean([site["selected"] for site in sites]) >= 0.9


def test_energy_scale_matches_the_declared_vehicle(case):
    _, meta, _ = case
    _generated_only(meta)
    per_km = meta["ranges"]["kwh_per_km_median"]
    if meta["vehicle_profile"] == "heavy_truck":
        assert 1.0 <= per_km <= 2.0  # California data: median 1.57 kWh/km


def test_recorded_scenario_check_is_clean(case):
    """check_scenarios.py output (when present): many distinct certified scenarios, replayable by seed."""
    path, meta, _ = case
    check = path / "scenario_check.json"
    if not check.exists():
        pytest.skip("scenario check not run for this case")
    chk = json.load(open(check))
    assert chk["scenario_failures"] == []
    assert chk["same_seed_replays_same_scenario"] is True
    assert chk["episodes_played"] == chk["episodes_requested"] >= 10
    assert chk["distinct_scenarios"] == chk["episodes_played"]
    assert chk["distinct_depots"] >= 0.5 * chk["episodes_played"]
    assert chk["nodes"] == meta["counts"]["nodes"] and chk["charger_nodes"] == meta["counts"]["chargers"]


def test_california_is_the_default_scenario_and_old_data_is_gone():
    data = ROOT / "EVRoutingEnv" / "data"
    for name in ("shortest_path_energy_dict.json", "shortest_path_time_dict.json", "station_info_dict.json",
                 "waiting_time_lookup.json", "distance_matrix_cache.pkl"):
        assert not (data / name).exists(), f"{name} should only exist inside case_studies/california"
    assert not (data / "vis_data").exists()
    for config in (ROOT / "EVRoutingEnv" / "config_files").glob("*.yaml"):
        path = load_config(str(config))["network"]["data_path"].rstrip("/")
        case = config.stem.replace("config_joint_", "")
        expected = case if (data / "case_studies" / case / "metadata.json").exists() and config.stem != "config_joint" else "california"
        assert path.endswith(f"case_studies/{expected}"), (config.name, path)


def test_generated_cases_use_dc_fast_sites_merged_within_their_radius_complete_linkage(case):
    """5 km for the US cases; the 1000 km-scale European case merges within 10 km to keep the node count down."""
    path, meta, _ = case
    _generated_only(meta)
    radius = meta["preset"]["charger_merge_m"]
    assert radius in (5000, 10000, 15000) and meta["preset"]["charger_merge_linkage"] == "complete"
    for site in json.load(open(path / "vis_data" / "stations.json")):
        assert site["ports"] >= 1
        assert site["diameter_m"] <= radius + 1e-6  # complete linkage: every pair of member stations within the radius
    assert {st["station_type"] for st in json.load(open(path / "station_info_dict.json")).values()} == {"DCFast"}


def test_california_chargers_are_not_merged(case):
    path, meta, _ = case
    if meta.get("generated", True):
        pytest.skip("California only")
    sites = json.load(open(path / "vis_data" / "stations.json"))
    assert len(sites) == meta["counts"]["chargers"] == 25
    assert all(site["n_stations"] == 1 and site["diameter_m"] == 0.0 for site in sites)
