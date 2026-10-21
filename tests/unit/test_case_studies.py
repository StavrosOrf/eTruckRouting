"""Data contract of the generated case studies under EVRoutingEnv/data/case_studies."""

import json
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from EVRoutingEnv.models.core.matrix_transportation_graph import (
    MatrixTransportationGraph,
)
from EVRoutingEnv.utils.utils import get_graph, load_config, resolve_network_data_dir


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


MATRICES = ("od_energy_kwh.npy", "od_time_h.npy", "od_node_ids.npy")


def _matrices_built(path: Path) -> None:
    """The all-pairs matrices of the 10^4-node cases (600 MB each) are gitignored."""
    missing = [name for name in MATRICES if not (path / name).exists()]
    if missing:
        pytest.skip(f"{path.name}: {', '.join(missing)} not built in this checkout "
                    "(rebuild with scripts/casestudies/submit_npy_build.sh)")


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
    if meta.get("format") == "npy":
        for name in ("station_info_dict.json", "waiting_time_lookup.json", "metadata.json"):
            assert (path / name).exists(), name
        _matrices_built(path)
        n = meta["counts"]["nodes"]
        ids = np.load(path / "od_node_ids.npy")
        energy, time = np.load(path / "od_energy_kwh.npy", mmap_mode="r"), np.load(path / "od_time_h.npy", mmap_mode="r")
        assert ids.shape == (n,) and np.all(np.diff(ids) > 0), "node ids must be unique and ascending"
        assert energy.shape == time.shape == (n, n) and energy.dtype == time.dtype == np.float32
        assert np.all(np.diag(energy) == 0) and np.all(np.diag(time) == 0)
        for lo in range(0, n, 2000):  # row blocks keep the memory small
            e, t = np.asarray(energy[lo:lo + 2000]), np.asarray(time[lo:lo + 2000])
            assert np.isfinite(e).all() and np.isfinite(t).all() and e.min() >= 0.0 and t.min() >= 0.0
            off = np.ones(e.shape, bool)
            off[np.arange(len(e)), np.arange(lo, lo + len(e))] = False
            assert e[off].min() >= meta["leg_energy_floor_kwh"] - 1e-6 and t[off].min() > 0.0
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
    if meta.get("format") == "npy":
        _matrices_built(path)
        rng = np.random.default_rng(0)
        n = meta["counts"]["nodes"]
        i, j = rng.integers(0, n, 400_000), rng.integers(0, n, 400_000)
        i, j = i[i < j], j[i < j]
        od = {key: np.load(path / file, mmap_mode="r") for key, file in (("energy_kwh", "od_energy_kwh.npy"), ("time_h", "od_time_h.npy"))}
        pairs = {key: (np.asarray(m[i, j], float), np.asarray(m[j, i], float)) for key, m in od.items()}
    else:
        od = np.load(path / "vis_data" / "od_matrices.npz")
        pairs = {}
        for key in ("energy_kwh", "time_h"):
            m = od[key].astype(float)  # NaN where the original data has no pair; comparisons below ignore them
            iu = np.triu_indices(len(m), 1)
            pairs[key] = (m[iu], m.T[iu])
    # a floor, not a per-case target: every case so far clears it by a wide margin (25-79 %), china_jinghu's
    # flatter, un-governed corridor the lowest measured so far at ~30 %
    for key, floor, minimum in (("energy_kwh", 1.0, 0.25), ("time_h", 0.05, 0.05)):
        f, r = pairs[key]
        ok = (f > floor) & (r > floor)
        assert np.mean(np.abs(f[ok] - r[ok]) / np.maximum(f[ok], r[ok]) > 0.05) > minimum, key


def test_matrix_case_loads_into_the_environment_graph(case):
    path, meta, config = case
    if meta.get("format") != "npy":
        pytest.skip("matrix case studies only")
    _matrices_built(path)
    graph = MatrixTransportationGraph.from_config(config["network"], resolve_network_data_dir(config["network"]))
    assert graph.num_nodes == meta["counts"]["nodes"]
    assert len(graph.get_charging_nodes()) == meta["counts"]["chargers"]
    assert sum(sum(graph.get_charger_details()[c]["types"].values()) for c in graph.get_charging_nodes()) == meta["counts"]["dcfast_ports_in_nodes"]
    ids = np.load(path / "od_node_ids.npy")
    stations = json.load(open(path / "station_info_dict.json"))
    assert {int(ids[c]) for c in graph.get_charging_nodes()} == {int(k) for k in stations}
    energy = np.load(path / "od_energy_kwh.npy", mmap_mode="r")
    a, b = 3, graph.num_nodes - 2
    assert graph.get_path_energy(a, b) == float(energy[a, b]) and graph.get_path_energy(a, a) == 0.0
    assert graph.get_shortest_path(a, b) == [a, b]
    assert meta["battery_kwh"] == config["truck"]["battery_capacity"]


def test_environment_graph_loads_and_routes_every_pair(case):
    path, meta, config = case
    if meta.get("format") == "npy":
        pytest.skip("matrix case: see test_matrix_case_loads_into_the_environment_graph")
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
    cases = {c.name for c in (data / "case_studies").iterdir() if (c / "metadata.json").exists()}
    aliases = {"pnw": "pacific_northwest"}
    for config in (ROOT / "EVRoutingEnv" / "config_files").glob("*.yaml"):
        path = load_config(str(config))["network"]["data_path"].rstrip("/")
        # A variant suffix (e.g. _longhaul, _drayage) reuses its base case study.
        name = config.stem.removeprefix("config_joint_").removeprefix("config_dynamic_")
        for alias, case in aliases.items():
            if name.startswith(alias + "_"):
                name = case + name[len(alias):]
        matches = [case for case in cases if name == case or name.startswith(case + "_")]
        expected = max(matches, key=len) if matches and config.stem != "config_joint" else "california"
        assert path.endswith(f"case_studies/{expected}"), (config.name, path)


def test_generated_cases_use_dc_fast_sites_merged_within_the_preset_radius(case):
    path, meta, _ = case
    _generated_only(meta)
    merge_m, linkage = meta["preset"]["charger_merge_m"], meta["preset"]["charger_merge_linkage"]
    # every preset uses complete linkage (every pair of member stations within the radius), whatever radius
    # it picks for its own scale: 5 km for the dict-format US/metro cases, 10-20 km for the 10^4-node cases
    assert linkage == "complete"
    for site in json.load(open(path / "vis_data" / "stations.json")):
        assert site["ports"] >= 1
        assert site["diameter_m"] <= merge_m + 1e-6  # every pair of member stations within the merge radius
    assert {st["station_type"] for st in json.load(open(path / "station_info_dict.json")).values()} == {"DCFast"}


def test_california_chargers_are_not_merged(case):
    path, meta, _ = case
    if meta.get("generated", True):
        pytest.skip("California only")
    sites = json.load(open(path / "vis_data" / "stations.json"))
    assert len(sites) == meta["counts"]["chargers"] == 25
    assert all(site["n_stations"] == 1 and site["diameter_m"] == 0.0 for site in sites)
