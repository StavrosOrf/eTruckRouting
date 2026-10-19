"""Build a case-study data folder in the same format as EVRoutingEnv/data (California).

    python scripts/casestudies/build_case_study.py scripts/casestudies/presets/seattle.yaml \
        --raw-dir .casestudy_raw

Inputs (see fetch_data.py): Amazon Last Mile stops or OSM places (delivery/depot locations), OpenStreetMap
road network, terrain tiles (slope), public DC-fast chargers (AFDC in the US, OpenStreetMap elsewhere).

Output: EVRoutingEnv/data/case_studies/<name>/
    shortest_path_energy_dict.json   '(u, v)' -> kWh of the minimum-energy route u->v   (asymmetric)
    shortest_path_time_dict.json     '(u, v)' -> hours on that same route               (asymmetric)
    station_info_dict.json           node -> {station_type: DCFast, total_capacity: ports}
    waiting_time_lookup.json         charger queue model (copied from the California data)
    vis_data/                        nodes, stations, roads, terrain and OD matrices for plotting
    metadata.json                    provenance, model assumptions, measured statistics

Presets with ``output_format: npy`` (europe_10k, 10^4 nodes) store the same route costs as float32 matrices
instead, because their 10^8 ordered pairs do not fit a JSON dictionary or a NetworkX graph:
    od_energy_kwh.npy, od_time_h.npy   (N, N) matrices, entry [i, j] = route i -> j; row i is OSM node od_node_ids[i]
    od_node_ids.npy                    node ids in ascending order (the JSON loader numbers nodes the same way)
and run in stages so the routing can be spread over Slurm array tasks (see scripts/casestudies/submit_npy_build.sh):
    --stage prepare   road network, terminals, routing graph (degree-2 road nodes folded away, exact)
    --stage od        one shard of the all-pairs routing:   --shard i --n-shards K
    --stage assemble  matrices, visualisation data, metadata
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chargers as chg  # noqa: E402
import contract as ctr  # noqa: E402
import fetch_data  # noqa: E402
import road_graph as rg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CA_DATA = REPO / "EVRoutingEnv" / "data" / "case_studies" / "california"  # default scenario; holds the shared charger queue model
MIN_LEG_KWH = 0.02  # legs never return energy: a net-downhill leg costs at least this


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ------------------------------------------------------------------ inputs
def load_amazon(path: Path, stations: list[str]):
    R = json.load(open(path))
    stops, depots = [], {}
    for rid, r in R.items():
        c = r["station_code"]
        if c not in stations:
            continue
        for key, v in r["stops"].items():
            if v["type"] == "Station":
                depots[c] = (v["lat"], v["lng"])
            else:
                z = v.get("zone_id")
                stops.append((v["lat"], v["lng"], z if isinstance(z, str) else "", c))
    missing = set(stations) - set(depots)
    if missing:
        raise SystemExit(f"Amazon stations not found in data: {sorted(missing)}")
    return stops, depots


def study_bbox(stops, depots, pct, margin):
    la = np.array([s[0] for s in stops] + [d[0] for d in depots.values()])
    lo = np.array([s[1] for s in stops] + [d[1] for d in depots.values()])
    (la0, la1), (lo0, lo1) = np.percentile(la, pct), np.percentile(lo, pct)
    return (la0 - margin, lo0 - margin, la1 + margin, lo1 + margin)


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def extract_region(raw: Path, pbf: Path, bbox, max_class: int) -> rg.RoadNetwork:
    """Road network of one Geofabrik extract, cached per extract so large multi-country cases can extract
    their regions as separate parallel jobs (``--extract-only``) and reuse them across presets."""
    key = f"{pbf.name}{pbf.stat().st_size}{bbox}{rg.CLASSES}{max_class}v1"
    cache = raw / "cache" / f"extract_{pbf.name.split('-latest')[0]}_{hashlib.sha1(key.encode()).hexdigest()[:10]}.pkl"
    if cache.exists():
        log("using cached extraction", cache.name)
        return pickle.load(open(cache, "rb"))
    log("extracting road network from", pbf.name)
    net = rg.extract(pbf, bbox, max_class)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(net, open(cache, "wb"), protocol=4)
    return net


def load_network(raw: Path, preset: dict, bbox):
    """Extract + merge regions + SCC + elevation, cached because it takes minutes to hours."""
    pbfs = [fetch_data.ensure_osm(raw, r) for r in _as_list(preset["geofabrik"])]
    max_class = rg.CLASS_ID[preset.get("max_road_class", "living_street")]
    zoom = int(preset.get("terrain_zoom", fetch_data.TERRAIN_ZOOM))
    smooth = float(preset.get("smooth_length_m", 60.0))
    key = f"{[(p.name, p.stat().st_size) for p in pbfs]}{bbox}{rg.CLASSES}{max_class}{zoom}{smooth}v4"
    cache = raw / "cache" / f"{preset['name']}_{hashlib.sha1(key.encode()).hexdigest()[:10]}.pkl"
    if cache.exists():
        log("using cached road network", cache.name)
        return pickle.load(open(cache, "rb"))
    stage = cache.with_suffix(".stage1.pkl")
    if stage.exists():
        log("using cached extraction", stage.name)
        net = pickle.load(open(stage, "rb"))
        return _finish_network(net, raw, smooth, zoom, cache)
    nets = []
    for pbf in pbfs:
        nets.append(extract_region(raw, pbf, bbox, max_class))
        log(f"  {pbf.name}: {nets[-1].n} nodes, {len(nets[-1].eu)} edges")
    net = rg.merge_networks(nets)
    del nets
    log(f"  merged: {net.n} nodes, {len(net.eu)} edges")
    net = rg.largest_scc(net)
    log(f"  largest SCC: {net.n} nodes, {len(net.eu)} edges")
    stage.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(net, open(stage, "wb"), protocol=4)
    return _finish_network(net, raw, smooth, zoom, cache)


def _finish_network(net, raw, smooth, zoom, cache):
    info = rg.attach_elevation(net, raw, smooth, zoom)
    info.pop("mosaic")  # large; the vis grid is rebuilt from the cached tiles when needed
    log(f"  elevation {info['elevation_range_m']} m, smoothing rms change {info['dem_rms_change_m']:.2f} m")
    out = (net, info)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(cache, "wb"), protocol=4)
    return out


def scan_region(raw: Path, pbf: Path, preset: dict, bbox) -> dict:
    """Charging stations and populated places of one extract, cached per extract (``--scan-only``).

    Reading a 5 GB extract for its amenity and place nodes takes minutes, so large multi-country cases scan
    their regions as parallel jobs and every later build reuses the result.
    """
    min_kw, tesla = float(preset.get("min_charger_kw", 50.0)), bool(preset.get("exclude_tesla_only", False))
    min_pop = preset.get("min_population", 500)
    assume_pop = bool(preset.get("assume_population", False))
    charger_source = preset.get("chargers_source", "afdc")
    want_ch = charger_source in ("osm", "osm_services")
    want_svc = charger_source == "osm_services"
    svc_ports = int(preset.get("service_area_ports", 4))
    want_pl = preset.get("delivery_source", "amazon") == "places"
    key = f"{pbf.name}{pbf.stat().st_size}{bbox}{min_kw}{tesla}{min_pop}{assume_pop}{want_ch}{want_pl}{want_svc}{svc_ports}v3"
    cache = raw / "cache" / f"scan_{pbf.name.split('-latest')[0]}_{hashlib.sha1(key.encode()).hexdigest()[:10]}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    log("scanning", pbf.name, "for", "chargers" if want_ch else "", "service areas" if want_svc else "",
        "places" if want_pl else "")
    out = {"chargers": chg.scan_osm_chargers(pbf, bbox, min_kw, tesla) if want_ch else [],
           "service_areas": chg.scan_osm_service_areas(pbf, bbox, svc_ports) if want_svc else [],
           "places": rg.load_places(pbf, min_pop=min_pop, assume_population=assume_pop) if want_pl else []}
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(cache, "wb"), protocol=4)
    log(f"  {pbf.name}: {len(out['chargers'])} charging stations, {len(out['service_areas'])} service areas, "
        f"{len(out['places'])} places")
    return out


def load_place_candidates(raw: Path, preset: dict, bbox=None):
    places = []
    for r in _as_list(preset["geofabrik"]):
        pbf = fetch_data.ensure_osm(raw, r)
        places += scan_region(raw, pbf, preset, bbox)["places"] if bbox is not None else rg.load_places(
            pbf, min_pop=preset.get("min_population", 500), assume_population=bool(preset.get("assume_population", False)))
    return places


def sample_places(places: list[dict], n: int, rng) -> list[dict]:
    """Population-weighted (sqrt) draw of distinct towns/cities: big cities are likelier, small towns appear."""
    w = np.sqrt(np.array([p["population"] for p in places], float))
    idx = rng.choice(len(places), size=min(n * 3, len(places)), replace=False, p=w / w.sum())
    return [places[i] for i in idx]  # over-draw: some will drop on snapping/dedup


def pick_places(places: list[dict], n: int, rng, snap, used: dict, max_snap: float, min_spacing_m: float) -> list[dict]:
    """Up to n places, drawn by sqrt-population weight without replacement (big cities first, villages fill the gaps).

    A place is kept when it snaps to an unused road node within max_snap and lies at least min_spacing_m from
    every kept place; the spacing test hashes the kept places into a 3-D grid, so it is linear in the number of
    candidates (the pairwise check of ``sample_places`` is quadratic and too slow for 10^4 places).
    """
    w = np.sqrt(np.array([p["population"] for p in places], float))
    order = np.argsort(-(rng.random(len(places)) ** (1.0 / w)))   # Efraimidis-Spirakis weighted order
    cand = [places[i] for i in order]
    lat = np.array([p["lat"] for p in cand]); lon = np.array([p["lon"] for p in cand])
    nodes_i, dist = snap(lat, lon)
    xyz = chg.sphere_xyz(lat, lon)
    cell, r2 = {}, min_spacing_m ** 2
    picked = []
    for k, pl in enumerate(cand):
        i = int(nodes_i[k])
        if dist[k] > max_snap or i in used:
            continue
        c = tuple((xyz[k] // min_spacing_m).astype(np.int64))
        if any(((xyz[k] - q) ** 2).sum() < r2
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               for q in cell.get((c[0] + dx, c[1] + dy, c[2] + dz), ())):
            continue
        cell.setdefault(c, []).append(xyz[k])
        used[i] = "customer"
        picked.append({"node_idx": i, "lat": pl["lat"], "lon": pl["lon"], "zone_id": pl["name"],
                       "population": pl["population"], "place": pl["place"], "snap_m": round(float(dist[k]), 1),
                       "population_is_assumed": pl.get("population_is_assumed", False)})
        if len(picked) >= n:
            break
    return picked


# ------------------------------------------------------------------ snapping
class Snapper:
    """Nearest drivable node, excluding freeway-class nodes by default (stops are never on a freeway).

    An expressway service area is the one terminal type that really does sit beside the motorway/trunk
    mainline (reached by a short ramp that is usually not even in the graph, since ``service`` roads are
    excluded): pass ``min_class="motorway"`` to let it snap to the mainline itself instead of being pushed
    to the nearest ordinary road, which can be many km away on a rural expressway.
    """

    def __init__(self, net: rg.RoadNetwork, min_class: str = "primary"):
        ok = np.zeros(net.n, bool)
        low = net.hwclass >= rg.CLASS_ID[min_class]
        ok[net.eu[low]] = True
        ok[net.ev[low]] = True
        self.idx = np.nonzero(ok)[0]
        self.tree = cKDTree(chg.sphere_xyz(net.lat[self.idx], net.lon[self.idx]))

    def __call__(self, lat, lon):
        d, i = self.tree.query(chg.sphere_xyz(lat, lon))
        if np.ndim(lat) == 0:
            return int(self.idx[i[0]]), float(d[0])
        return self.idx[i], d


# --------------------------------------------------------------- OD matrices
_OD: dict = {}


def _tree_sums(pk, s, terminals, keys, t_u, e_u, l_u, n):
    """Time, energy and length summed along the predecessor-tree path s -> each terminal (NaN if unreachable).

    Walking back from every terminal costs (terminals x path length) lookups per source, which made the
    routing quadratic in the number of terminals. Instead collect the union of the paths (they share
    trunks, so it is far smaller), give every union node the cost of its parent edge, and add the costs up
    the tree by pointer doubling: O(|union| log depth), independent of how many terminals share a trunk.
    """
    scratch = _OD.get("scratch")
    if scratch is None:
        scratch = _OD["scratch"] = (np.zeros(n, bool), np.empty(n, np.int32))
    visited, loc = scratch
    tn = terminals.astype(np.int64)
    front = np.unique(tn)
    parts = [front]
    visited[front] = True
    while True:
        p = pk[front]
        p = np.unique(p[p >= 0].astype(np.int64))
        p = p[~visited[p]]
        if p.size == 0:
            break
        visited[p] = True
        parts.append(p)
        front = p
    U = np.concatenate(parts)
    visited[U] = False
    loc[U] = np.arange(len(U), dtype=np.int32)
    par = pk[U].astype(np.int64)
    has = par >= 0
    anc = np.arange(len(U), dtype=np.int64)
    anc[has] = loc[par[has]]
    wt, we, wl = np.zeros(len(U)), np.zeros(len(U)), np.zeros(len(U))
    ei = np.searchsorted(keys, par[has] * n + U[has])
    wt[has], we[has], wl[has] = t_u[ei], e_u[ei], l_u[ei]
    while True:
        nxt = anc[anc]
        if np.array_equal(nxt, anc):
            break
        wt = wt + wt[anc]; we = we + we[anc]; wl = wl + wl[anc]
        anc = nxt
    li = loc[tn]
    ok = U[anc[li]] == s            # the tree path ends at the source: reachable
    nan = np.where(ok, 0.0, np.nan)
    return wt[li] + nan, we[li] + nan, wl[li] + nan


def _od_rows(args):
    """Dijkstra from a few sources, then sum time/energy/length along the predecessor tree to every terminal."""
    c0, src = args
    csr, keys, e_u, t_u, l_u = _OD["csr"], _OD["keys"], _OD["e"], _OD["t"], _OD["l"]
    terminals, kappa, elev, n = _OD["terminals"], _OD["kappa"], _OD["elev"], _OD["n"]
    D, P = dijkstra(csr, directed=True, indices=src, return_predecessors=True)
    rows = []
    for k, s in enumerate(src):
        at, ae, al = _tree_sums(P[k], int(s), terminals, keys, t_u, e_u, l_u, n)
        e_red = D[k, terminals] + kappa * (elev[terminals] - elev[s])
        rows.append((e_red, at, ae, al))
    return c0, rows


def od_graph(net: rg.RoadNetwork, terminals: np.ndarray, t_h, e_kwh, vehicle, contract: bool = False, seed: int = 0) -> dict:
    """Search graph for the all-pairs routing: reduced (non-negative) edge costs, optionally on the junction graph.

    Edge energies can be negative (regeneration). Reweighting by the elevation potential
    c' = c - kappa*(h_v - h_u) >= 0 makes Dijkstra exact; the true net energy of a route is
    D' + kappa*(h_t - h_s). With ``contract`` the degree-2 road nodes are first folded into their edges
    (see contract.py), which leaves every terminal-to-terminal route cost exactly unchanged.
    """
    kappa = rg.potential_slope(vehicle)
    eu, ev, costs = net.eu, net.ev, [t_h, e_kwh, net.length]
    elev, term, stats = net.elev, np.asarray(terminals, np.int64), {}
    if contract:
        protect = np.zeros(net.n, bool)
        protect[terminals] = True
        alive, eu, ev, costs = ctr.contract_chains(net.n, eu, ev, protect, costs, seed=seed, log=log)
        remap = -np.ones(net.n, np.int64)
        remap[alive] = np.arange(int(alive.sum()))
        eu, ev, elev, term = remap[eu], remap[ev], net.elev[alive], remap[term]
        stats.update(road_nodes_before_contraction=int(net.n), road_edges_before_contraction=int(len(net.eu)),
                     junction_nodes=int(alive.sum()), junction_edges=int(len(eu)))
    t_u, e_u, l_u = costs
    n = len(elev)
    red = e_u - kappa * (elev[ev] - elev[eu])
    stats.update(reduced_cost_min=float(red.min()), n_negative_reduced=int((red < -1e-9).sum()))
    red = np.maximum(red, 0.0) + 1e-9
    # parallel edges: keep the cheapest
    order = np.lexsort((red, ev, eu))
    eu, ev = eu[order], ev[order]
    first = np.r_[True, (eu[1:] != eu[:-1]) | (ev[1:] != ev[:-1])]
    sel = order[first]
    eu, ev = eu[first].astype(np.int64), ev[first].astype(np.int64)
    return dict(csr=csr_matrix((red[sel], (eu, ev)), shape=(n, n)), keys=eu * n + ev, e=e_u[sel], t=t_u[sel],
                l=l_u[sel], terminals=term, kappa=kappa, elev=elev, n=n, stats=stats)


def od_block(g: dict, r0: int, r1: int, chunk: int = 4, workers=None, label: str = ""):
    """Rows r0..r1 of the terminal-to-terminal matrices: (E_net, time, km, accumulated-energy check, stats)."""
    import multiprocessing as mp

    _OD.update(csr=g["csr"], keys=g["keys"], e=g["e"], t=g["t"], l=g["l"], terminals=g["terminals"],
               kappa=g["kappa"], elev=g["elev"], n=g["n"])
    terminals = g["terminals"]
    R, T = r1 - r0, len(terminals)
    E = np.zeros((R, T)); TT = np.zeros((R, T)); LL = np.zeros((R, T)); AE = np.zeros((R, T))
    jobs = [(c0, terminals[r0 + c0:r0 + min(c0 + chunk, R)]) for c0 in range(0, R, chunk)]
    workers = workers or min(16, max(1, (mp.cpu_count() or 2) // 4))
    done = 0
    with mp.get_context("fork").Pool(workers) as pool:
        for c0, rows in pool.imap_unordered(_od_rows, jobs):
            for k, (e_red, at, ae, al) in enumerate(rows):
                E[c0 + k], TT[c0 + k], LL[c0 + k], AE[c0 + k] = e_red, at, al, ae
            done += len(rows)
            if (done // chunk) % 10 == 0 or done == R:
                log(f"  OD rows {label}{done}/{R}")
    stats = {"max_energy_consistency_error_kwh": float(np.nanmax(np.abs(np.where(np.isfinite(E), E - AE, 0)))),
             "unreachable_pairs": int((~np.isfinite(E)).sum())}
    _OD.clear()
    return E, TT, LL / 1000.0, stats


def od_matrices(net: rg.RoadNetwork, terminals: np.ndarray, t_h, e_kwh, vehicle, chunk=4, workers=None):
    """Minimum-energy routes between terminals, with time/length summed on the same route.

    Energy is also re-accumulated along each route as a check.
    """
    g = od_graph(net, terminals, t_h, e_kwh, vehicle)
    E, TT, KM, stats = od_block(g, 0, len(terminals), chunk, workers)
    return E, TT, KM, {**g["stats"], **stats}


def asymmetry_stats(E, TT, min_e=1.0):
    iu = np.triu_indices(len(E), 1)
    out = {}
    for name, M, floor in (("energy", E, min_e), ("time", TT, 0.05)):
        f, r = M[iu], M.T[iu]
        ok = (f > floor) & (r > floor)
        rel = np.abs(f[ok] - r[ok]) / np.maximum(f[ok], r[ok])
        out[name] = {
            "pairs": int(ok.sum()),
            "median_rel_diff": round(float(np.median(rel)), 4),
            "p90_rel_diff": round(float(np.percentile(rel, 90)), 4),
            "frac_over_5pct": round(float((rel > 0.05).mean()), 4),
            "frac_over_20pct": round(float((rel > 0.20).mean()), 4),
        }
    return out


# ------------------------------------------------------------------ outputs
def write_dict(path: Path, ids, M, decimals):
    items = {}
    for i, u in enumerate(ids):
        row = np.round(M[i], decimals)
        for j, v in enumerate(ids):
            if i != j:
                items[f"({u}, {v})"] = float(row[j])
    json.dump(items, open(path, "w"))


def write_roads(path: Path, net: rg.RoadNetwork, max_class: int):
    """Undirected road segments of the main network (<= tertiary) for plotting."""
    m = net.hwclass <= max_class
    a = np.minimum(net.eu[m], net.ev[m]).astype(np.int64); b = np.maximum(net.eu[m], net.ev[m]).astype(np.int64)
    key = a * net.n + b
    _, first = np.unique(key, return_index=True)
    a, b = a[first], b[first]
    cls = net.hwclass[m][first]
    grade = (net.elev[b] - net.elev[a]) / np.maximum(net.length[m][first], 1.0)
    q = lambda x: np.round(x * 1e5).astype(np.int32)  # 1 m
    np.savez_compressed(
        path, lat0=q(net.lat[a]), lon0=q(net.lon[a]), lat1=q(net.lat[b]), lon1=q(net.lon[b]),
        cls=cls.astype(np.int8), grade=grade.astype(np.float16),
    )
    return int(len(a))


def write_terrain_grid(path: Path, raw: Path, bbox, zoom, step_m=250.0):
    mos = rg.TerrainMosaic(raw, bbox, zoom)
    lat0 = 0.5 * (bbox[0] + bbox[2])
    dlat = step_m / 111320.0
    dlon = step_m / (111320.0 * np.cos(np.radians(lat0)))
    lats = np.arange(bbox[0], bbox[2], dlat); lons = np.arange(bbox[1], bbox[3], dlon)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    z = mos.sample(LA.ravel(), LO.ravel()).reshape(LA.shape)
    np.savez_compressed(path, elev=np.round(z).astype(np.int16), lat0=lats[0], lon0=lons[0], dlat=dlat, dlon=dlon)


def make_config(name: str, title: str, overrides: dict) -> Path:
    """config_joint_<name>.yaml: the primary joint-fleet config pointed at this case study.

    Only network.data_path changes by default, so results stay comparable with California
    (400 kWh battery, 150/350/750 kW charger classes, same fleet, payload and traffic settings).
    ``overrides`` may set battery_kwh and station_power_classes_kw from the preset.
    """
    import re

    src = (REPO / "EVRoutingEnv" / "config_files" / "config_joint.yaml").read_text()
    sub = [(r"(?m)^(  data_path:).*$", rf"\1 EVRoutingEnv/data/case_studies/{name}/")]
    if overrides.get("output_format") == "npy":  # the environment loads matrices when the file name ends in .npy
        sub.append((r"(?m)^(  shortest_path_energy_file:).*$", r"\1 od_energy_kwh.npy"))
        sub.append((r"(?m)^(  shortest_path_time_file:).*$", r"\1 od_time_h.npy"))
    if overrides.get("battery_kwh"):
        sub.append((r"(?m)^(  battery_capacity:).*$", rf"\1 {float(overrides['battery_kwh']):.1f}"))
    if overrides.get("station_power_classes_kw"):
        sub.append((r"(?m)^(  station_power_classes_kw:).*$", rf"\1 {list(map(float, overrides['station_power_classes_kw']))}"))
    for pat, rep in sub:
        src, n = re.subn(pat, rep, src, count=1)
        if n != 1:
            raise SystemExit(f"config_joint.yaml no longer matches pattern {pat!r}")
    head = (f"# Joint fleet-routing on the {title} case study (generated by scripts/casestudies/build_case_study.py).\n"
            "# Identical to config_joint.yaml except network.data_path"
            + (" and the preset overrides." if len(sub) > 1 else ".") + "\n\n")
    out = REPO / "EVRoutingEnv" / "config_files" / f"config_joint_{name}.yaml"
    out.write_text(head + src.split("\n", 1)[1].lstrip("\n"))
    return out


def od_cache_dir(raw: Path, name: str) -> Path:
    return raw / "cache" / f"{name}_od"


def run_od_shard(preset: dict, raw: Path, shard: int, n_shards: int) -> Path:
    """Rows of one shard of the all-pairs routing (a Slurm array task), from the graph saved by ``--stage prepare``."""
    cdir = od_cache_dir(raw, preset["name"])
    g = pickle.load(open(cdir / "graph.pkl", "rb"))
    bounds = np.linspace(0, len(g["terminals"]), n_shards + 1).astype(int)
    r0, r1 = int(bounds[shard]), int(bounds[shard + 1])
    log(f"  shard {shard}/{n_shards}: rows {r0}..{r1} of {len(g['terminals'])} on {g['n']} search nodes")
    E, TT, KM, st = od_block(g, r0, r1, workers=preset.get("workers"), label=f"[shard {shard}] ")
    f = cdir / f"shard_{shard:04d}_of_{n_shards:04d}.npz"
    tmp = cdir / f"tmp_{f.name}"
    np.savez(tmp, ids=g["ids"], r0=r0, r1=r1, E=E.astype(np.float32), T=TT.astype(np.float32),
             KM=KM.astype(np.float32), **st)
    tmp.replace(f)
    log("  wrote", f)
    return f


def write_npy_matrices(out: Path, cdir: Path, ids: list[int], sample: int = 4_000_000, seed: int = 0):
    """Assemble the shards into ``od_energy_kwh.npy`` / ``od_time_h.npy`` (float32, row i = node ids[i]).

    Writes row by row into memory-mapped files, so the 10^4 x 10^4 matrices never sit in memory twice. Summary
    statistics (percentiles, asymmetry) are taken on a random sample of pairs.
    """
    from numpy.lib.format import open_memmap

    T = len(ids)
    files = sorted(cdir.glob("shard_*_of_*.npz"))
    if not files:
        raise SystemExit(f"no OD shards in {cdir}")
    nsh = {int(f.stem.split("_of_")[1]) for f in files}
    if len(nsh) != 1 or len(files) != nsh.pop():
        raise SystemExit(f"incomplete shard set in {cdir}: {[f.name for f in files]}")
    Em = open_memmap(out / "od_energy_kwh.npy", mode="w+", dtype=np.float32, shape=(T, T))
    Tm = open_memmap(out / "od_time_h.npy", mode="w+", dtype=np.float32, shape=(T, T))
    Km = open_memmap(cdir / "od_dist_km.npy", mode="w+", dtype=np.float32, shape=(T, T))
    stats = {"max_energy_consistency_error_kwh": 0.0, "unreachable_pairs": 0}
    neg, covered = 0, 0
    for f in files:
        z = np.load(f)
        if not np.array_equal(z["ids"], np.array(ids, np.int64)):
            raise SystemExit(f"{f.name} was computed for a different terminal set; rerun --stage prepare and od")
        r0, r1 = int(z["r0"]), int(z["r1"])
        if r0 != covered:
            raise SystemExit(f"shards do not tile the rows: gap before {f.name}")
        covered = r1
        diag = (np.arange(r1 - r0), np.arange(r0, r1))
        E_net, tt, km = z["E"].astype(np.float64), z["T"].copy(), z["KM"].copy()
        E_net[diag] = 0.0; tt[diag] = 0.0; km[diag] = 0.0
        neg += int((E_net < MIN_LEG_KWH).sum()) - (r1 - r0)
        E = np.maximum(E_net, MIN_LEG_KWH); E[diag] = 0.0
        Em[r0:r1], Tm[r0:r1], Km[r0:r1] = E, tt, km
        stats["max_energy_consistency_error_kwh"] = max(stats["max_energy_consistency_error_kwh"],
                                                        float(z["max_energy_consistency_error_kwh"]))
        stats["unreachable_pairs"] += int(z["unreachable_pairs"])
    if covered != T:
        raise SystemExit(f"shards cover {covered} of {T} rows")
    for m in (Em, Tm, Km):
        m.flush()
    rng = np.random.default_rng(seed)
    i = rng.integers(0, T, sample); j = rng.integers(0, T, sample)
    off = i != j
    i, j = i[off], j[off]
    e_f, e_r = Em[i, j].astype(float), Em[j, i].astype(float)
    t_f, t_r, k_f = Tm[i, j].astype(float), Tm[j, i].astype(float), Km[i, j].astype(float)
    asym = {}
    for key, f_, r_, floor in (("energy", e_f, e_r, 1.0), ("time", t_f, t_r, 0.05)):
        ok = (f_ > floor) & (r_ > floor)
        rel = np.abs(f_[ok] - r_[ok]) / np.maximum(f_[ok], r_[ok])
        asym[key] = {"pairs_sampled": int(ok.sum()), "median_rel_diff": round(float(np.median(rel)), 4),
                     "p90_rel_diff": round(float(np.percentile(rel, 90)), 4),
                     "frac_over_5pct": round(float((rel > 0.05).mean()), 4),
                     "frac_over_20pct": round(float((rel > 0.20).mean()), 4)}
    ranges = {
        "energy_kwh_p5_p50_p95_max": [round(float(x), 2) for x in np.percentile(e_f, [5, 50, 95, 100])],
        "time_h_p5_p50_p95_max": [round(float(x), 3) for x in np.percentile(t_f, [5, 50, 95, 100])],
        "dist_km_p5_p50_p95_max": [round(float(x), 1) for x in np.percentile(k_f, [5, 50, 95, 100])],
        "kwh_per_km_median": round(float(np.median(e_f / np.maximum(k_f, 0.1))), 3),
        "sampled_pairs": int(len(e_f)),
    }
    np.save(out / "od_node_ids.npy", np.array(ids, np.int64))
    return stats, neg, asym, ranges


def build(preset: dict, raw: Path, out_root: Path, overrides: dict, stage: str = "all", shard: int = 0, n_shards: int = 1):
    preset = {**preset, **{k: v for k, v in overrides.items() if v is not None}}
    name = preset["name"]
    npy = preset.get("output_format", "json") == "npy"
    if stage == "od":
        return run_od_shard(preset, raw, shard, n_shards), None
    if stage != "all" and not npy:
        raise SystemExit("--stage needs output_format: npy in the preset")
    out = out_root / name
    (out / "vis_data").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(preset.get("seed", 0))
    profile = preset.get("vehicle_profile", "heavy_truck")
    vehicle = {**rg.VEHICLES[profile], **preset.get("vehicle", {})}
    source = preset.get("delivery_source", "amazon")
    zoom = int(preset.get("terrain_zoom", fetch_data.TERRAIN_ZOOM))

    log(f"=== {name}")
    if source == "amazon":
        stops, depots = load_amazon(fetch_data.ensure_amazon(raw), preset["amazon_stations"])
        bbox = study_bbox(stops, depots, preset["stop_percentiles"], preset["bbox_margin_deg"])
        log(f"  {len(stops)} Amazon stops, {len(depots)} depots, study bbox {np.round(bbox, 3).tolist()}")
    else:
        bbox = tuple(preset["bbox"])
        log(f"  region bbox {bbox}")

    net, terrain_info = load_network(raw, preset, bbox)
    snap = Snapper(net)
    max_snap = float(preset["max_snap_m"])

    # --- chargers: every public DC-fast site, merged within charger_merge_m (preset: 5 km, complete linkage)
    sites = []
    charger_source = preset.get("chargers_source", "afdc")
    if charger_source == "afdc":
        for st in _as_list(preset["state"]):
            sites += chg.load_dcfast(fetch_data.ensure_afdc(raw, st), bbox, preset.get("exclude_tesla_only", False))
        charger_src = "NREL/NLR Alternative Fuels Data Center station locator, public DC-fast only"
    elif charger_source == "osm":
        # outside the US there is no AFDC: DC sockets of amenity=charging_station in the same Geofabrik extracts
        sites = chg.dedupe_sites([c for r in _as_list(preset["geofabrik"])
                                  for c in scan_region(raw, fetch_data.ensure_osm(raw, r), preset, bbox)["chargers"]])
        charger_src = (f"OpenStreetMap amenity=charging_station (CCS/CHAdeMO/NACS sockets, tagged output >= "
                       f"{float(preset.get('min_charger_kw', 50.0)):g} kW), ODbL")
    elif charger_source == "osm_services":
        # real amenity=charging_station tagging plus every mapped expressway service area (highway=services/
        # rest_area), assumed to host a DC-fast bank where it is not itself tagged with a real charger: OSM's
        # mapping of service areas is far more complete than its mapping of the chargers inside them in regions
        # like China, where the expressway service-area EV rollout outpaces OSM tagging of individual stations.
        scans = [scan_region(raw, fetch_data.ensure_osm(raw, r), preset, bbox) for r in _as_list(preset["geofabrik"])]
        real = chg.dedupe_sites([c for s in scans for c in s["chargers"]])
        # dedupe_sites also catches the same OSM way re-extracted from two overlapping region files (e.g.
        # Hebei's Geofabrik extract already contains Beijing and Tianjin), which would otherwise double its
        # assumed ports when it is merged below
        svc = chg.dedupe_sites([c for s in scans for c in s["service_areas"]])
        svc = chg.drop_near(svc, real, float(preset.get("service_area_dedup_m", 300.0)))
        sites = real + svc
        svc_ports = int(preset.get("service_area_ports", 4))
        charger_src = (f"OpenStreetMap amenity=charging_station (real stations, >= "
                       f"{float(preset.get('min_charger_kw', 50.0)):g} kW) plus every mapped expressway service "
                       f"area (highway=services/rest_area) not already carrying one, assumed to host {svc_ports} "
                       f"DC-fast ports each (modelled, not measured: see 'charger_clusters.modelled_sites' below), ODbL")
    else:
        raise SystemExit(f"unknown chargers_source {charger_source!r} (afdc, osm or osm_services)")
    merged = chg.merge_sites(sites, preset["charger_merge_m"], preset.get("charger_merge_linkage", "complete"))
    # sites that snap to the same road node are one charger node; sites far from any road are dropped.
    # osm_services sites genuinely sit beside the motorway/trunk mainline (reached by a ramp that is usually
    # not in the graph at all, since `service` roads are excluded), so let them snap to it directly instead
    # of being pushed out to the nearest ordinary road, which can be many km away on a rural expressway.
    snap_charger = Snapper(net, "motorway") if charger_source == "osm_services" else snap
    by_node: dict[int, list[dict]] = {}
    for c in merged:
        i, d = snap_charger(c["lat"], c["lon"])
        if d <= max_snap:
            c["node_idx"], c["snap_m"] = i, round(d, 1)
            by_node.setdefault(i, []).append(c)
    placeable = [chg.combine(group) | {"node_idx": i, "snap_m": min(g["snap_m"] for g in group)}
                 for i, group in by_node.items()]
    placeable.sort(key=lambda c: (-c["ports"], c["lat"], c["lon"]))
    chosen = chg.spread_subset(placeable, int(preset["max_chargers"]))
    log(f"  DC-fast: {len(sites)} stations ({sum(s['ports'] for s in sites)} ports) -> {len(merged)} merged sites "
        f"-> {len(placeable)} road-snapped nodes -> {len(chosen)} charger nodes ({sum(c['ports'] for c in chosen)} ports)")
    used: dict[int, str] = {}
    charger_nodes = []
    for c in chosen:
        used[c["node_idx"]] = "charger"
        charger_nodes.append(c)

    depot_nodes, cust_nodes, n_assumed_pop = [], [], 0
    if source == "amazon":
        # depots: the real Amazon stations; customers: random real stops, one per road node
        for code, (la, lo) in depots.items():
            i, d = snap(la, lo)
            if int(i) in used:
                continue
            used[int(i)] = "depot"
            depot_nodes.append({"code": code, "node_idx": int(i), "lat": la, "lon": lo, "snap_m": round(float(d), 1)})
        order = rng.permutation(len(stops))
        s_lat = np.array([s[0] for s in stops]); s_lon = np.array([s[1] for s in stops])
        nodes_i, dist = snap(s_lat[order], s_lon[order])
        for k, o in enumerate(order):
            if len(cust_nodes) >= preset["n_customers"]:
                break
            i = int(nodes_i[k])
            if dist[k] > max_snap or i in used:
                continue
            used[i] = "customer"
            cust_nodes.append({"node_idx": i, "lat": float(s_lat[o]), "lon": float(s_lon[o]), "zone_id": stops[o][2],
                               "amazon_station": stops[o][3], "snap_m": round(float(dist[k]), 1)})
        delivery_src = "Amazon Last Mile Routing Research Challenge 2021 (training routes), CC BY-NC 4.0"
    else:
        # towns and cities with a population tag in OpenStreetMap (like California's city-to-city lanes)
        if npy:
            places = load_place_candidates(raw, preset, bbox)
            log(f"  {len(places)} populated places with a population tag")
            cust_nodes = pick_places(places, int(preset["n_customers"]) + int(preset.get("n_depots", 4)), rng, snap,
                                     used, max_snap, float(preset.get("min_place_spacing_m", 4000.0)))
        else:
            cand = sample_places(load_place_candidates(raw, preset), int(preset["n_customers"]), rng)
            lat = np.array([p["lat"] for p in cand]); lon = np.array([p["lon"] for p in cand])
            nodes_i, dist = snap(lat, lon)
            picked = []
            for k, pl in enumerate(cand):
                i = int(nodes_i[k])
                if dist[k] > max_snap or i in used:
                    continue
                xy = chg.local_xy([pl["lat"]], [pl["lon"]], float(lat.mean()))[0]
                if any(np.hypot(*(xy - q)) < preset.get("min_place_spacing_m", 4000.0) for q in (pp["xy"] for pp in picked)):
                    continue
                used[i] = "customer"
                picked.append({"node_idx": i, "lat": pl["lat"], "lon": pl["lon"], "zone_id": pl["name"],
                               "population": pl["population"], "place": pl["place"], "snap_m": round(float(dist[k]), 1), "xy": xy})
                if len(picked) >= preset["n_customers"] + int(preset.get("n_depots", 4)):
                    break
            for pp in picked:
                pp.pop("xy")
            cust_nodes = picked
        # depots: the largest places become regional hubs (plain nodes in the environment)
        big = sorted(range(len(cust_nodes)), key=lambda j: -cust_nodes[j]["population"])[: int(preset.get("n_depots", 4))]
        for j in sorted(big, reverse=True):
            c = cust_nodes.pop(j)
            depot_nodes.append({"code": c["zone_id"], "node_idx": c["node_idx"], "lat": c["lat"], "lon": c["lon"],
                                "snap_m": c["snap_m"], "population": c["population"],
                                "population_is_assumed": c.get("population_is_assumed", False)})
        depot_nodes.reverse()
        n_assumed_pop = sum(1 for lst in (depot_nodes, cust_nodes) for m in lst if m.get("population_is_assumed"))
        delivery_src = "OpenStreetMap place nodes (city/town/village)" + (
            f", {n_assumed_pop} of {len(depot_nodes) + len(cust_nodes)} with no population tag "
            f"(population assumed by place kind: see road_graph.PLACE_KIND_POPULATION_FALLBACK)"
            if preset.get("assume_population") else " with population tag") + ", ODbL"
    if len(cust_nodes) + len(depot_nodes) < preset["n_customers"]:
        log(f"  WARNING only {len(cust_nodes) + len(depot_nodes)} delivery nodes placed")

    terminals_meta = ([("charger", c) for c in charger_nodes] + [("depot", d) for d in depot_nodes]
                      + [("customer", c) for c in cust_nodes])
    if npy:  # matrix row i is the i-th smallest OSM node id, the same node numbering the JSON loader derives
        terminals_meta.sort(key=lambda rm: int(net.osm_id[rm[1]["node_idx"]]))
    term_idx = np.array([m["node_idx"] for _, m in terminals_meta])
    ids = [int(net.osm_id[i]) for i in term_idx]
    log(f"  terminals: {len(charger_nodes)} chargers, {len(depot_nodes)} depots, {len(cust_nodes)} customers")

    if npy:
        cdir = od_cache_dir(raw, name)
        if stage in ("all", "prepare"):
            cdir.mkdir(parents=True, exist_ok=True)
            for stale in cdir.glob("shard_*"):
                stale.unlink()
            t_h, e_kwh, _ = rg.edge_costs(net, vehicle)
            g = od_graph(net, term_idx, t_h, e_kwh, vehicle, contract=bool(preset.get("contract_chains", False)))
            g["ids"] = np.array(ids, np.int64)
            del t_h, e_kwh
            pickle.dump(g, open(cdir / "graph.pkl", "wb"), protocol=4)
            log(f"  routing graph saved ({g['n']} search nodes, {len(g['e'])} edges, {len(ids)} terminals) -> {cdir}")
            if stage == "prepare":
                return cdir, None
            od_stats = dict(g["stats"])
            del g
            for sh in range(n_shards):
                run_od_shard(preset, raw, sh, n_shards)
        else:
            saved = pickle.load(open(cdir / "graph.pkl", "rb"))
            od_stats = dict(saved["stats"])
            if not np.array_equal(saved["ids"], np.array(ids, np.int64)):
                raise SystemExit("terminal set differs from the one the OD shards were computed for; rerun --stage prepare")
            del saved
        log("  assembling OD matrices")
        shard_stats, neg, asym, ranges = write_npy_matrices(out, cdir, ids)
        od_stats.update(shard_stats)
        if od_stats["unreachable_pairs"] > 0:
            raise SystemExit("unreachable terminal pairs in a strongly connected graph: " + str(od_stats))
    else:
        t_h, e_kwh, _ = rg.edge_costs(net, vehicle)
        log("  computing all-pairs minimum-energy routes between terminals")
        E_net, TT, KM, od_stats = od_matrices(net, term_idx, t_h, e_kwh, vehicle, workers=preset.get("workers"))
        if od_stats["unreachable_pairs"] - len(ids) > 0:
            raise SystemExit("unreachable terminal pairs in a strongly connected graph: " + str(od_stats))
        np.fill_diagonal(E_net, 0); np.fill_diagonal(TT, 0); np.fill_diagonal(KM, 0)
        neg = int((E_net[~np.eye(len(ids), dtype=bool)] < MIN_LEG_KWH).sum())
        E = np.maximum(E_net, MIN_LEG_KWH); np.fill_diagonal(E, 0)
        asym = asymmetry_stats(E_net, TT)
        off_diag = ~np.eye(len(ids), dtype=bool)
        ranges = {
            "energy_kwh_p5_p50_p95_max": [round(float(x), 2) for x in np.percentile(E[off_diag], [5, 50, 95, 100])],
            "time_h_p5_p50_p95_max": [round(float(x), 3) for x in np.percentile(TT[off_diag], [5, 50, 95, 100])],
            "dist_km_p5_p50_p95_max": [round(float(x), 1) for x in np.percentile(KM[off_diag], [5, 50, 95, 100])],
            "kwh_per_km_median": round(float(np.median(E[off_diag] / np.maximum(KM[off_diag], 0.1))), 3),
        }

        # --- write the California-format files
        write_dict(out / "shortest_path_energy_dict.json", ids, E, 4)
        write_dict(out / "shortest_path_time_dict.json", ids, TT, 5)
    station_info = {str(int(net.osm_id[c["node_idx"]])): {"station_type": "DCFast", "total_capacity": float(c["ports"])}
                    for c in charger_nodes}
    json.dump(station_info, open(out / "station_info_dict.json", "w"), indent=1)
    shutil.copy(CA_DATA / "waiting_time_lookup.json", out / "waiting_time_lookup.json")

    # --- visualisation data
    vis = out / "vis_data"
    node_rows = []
    for (role, m), nid in zip(terminals_meta, ids):
        i = m["node_idx"]
        node_rows.append({
            "node": nid, "role": role, "lat": round(float(net.lat[i]), 6), "lon": round(float(net.lon[i]), 6),
            "elev_m": round(float(net.elev[i]), 1), "snap_m": m["snap_m"],
            **({"depot_code": m["code"]} if role == "depot" else {}),
            **({"zone_id": m["zone_id"]} if role == "customer" else {}),
            **({"amazon_station": m["amazon_station"]} if role == "customer" and "amazon_station" in m else {}),
            **({"population": m["population"]} if "population" in m else {}),
            **({"population_is_assumed": True} if m.get("population_is_assumed") else {}),
        })
    json.dump(node_rows, open(vis / "nodes.json", "w"))
    json.dump([{
        "node": int(net.osm_id[c["node_idx"]]), "lat": round(float(net.lat[c["node_idx"]]), 6),
        "lon": round(float(net.lon[c["node_idx"]]), 6),
        "site_lat": round(c["lat"], 6), "site_lon": round(c["lon"], 6), "ports": c["ports"],
        "n_stations": c["n_stations"], "diameter_m": c["diameter_m"], "snap_m": c["snap_m"],
        "names": c["names"], "networks": c["networks"], "connectors": c["connectors"], "afdc_ids": c["afdc_ids"],
        "modelled": c.get("modelled", False),
    } for c in charger_nodes], open(vis / "stations.json", "w"), indent=1)
    chosen_ids = {i for x in chosen for i in x["afdc_ids"]}
    json.dump([{"cluster_rank": i, "lat": round(c["lat"], 6), "lon": round(c["lon"], 6), "ports": c["ports"],
                "n_stations": c["n_stations"], "networks": c["networks"], "selected": c["afdc_ids"][0] in chosen_ids,
                "modelled": c.get("modelled", False)}
               for i, c in enumerate(merged)], open(vis / "all_dcfast_sites.json", "w"))
    n_road = write_roads(vis / "roads.npz", net, rg.CLASS_ID[preset.get("vis_max_class", "tertiary_link")])
    ext_km = max((bbox[2] - bbox[0]) * 111.0, (bbox[3] - bbox[1]) * 111.0 * np.cos(np.radians(0.5 * (bbox[0] + bbox[2]))))
    write_terrain_grid(vis / "terrain.npz", raw, bbox, zoom, step_m=max(250.0, ext_km * 1000.0 / 700.0))
    if not npy:
        np.savez_compressed(vis / "od_matrices.npz", node_ids=np.array(ids, np.int64), energy_net_kwh=E_net.astype(np.float32),
                            energy_kwh=E.astype(np.float32), time_h=TT.astype(np.float32), dist_km=KM.astype(np.float32))

    meta = {
        "name": name, "title": preset["title"], "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preset": preset, "units": {"energy": "kWh (battery)", "time": "hours", "distance": "km"},
        "scale_km": round(float(ext_km), 0),
        "counts": {"nodes": len(ids), "chargers": len(charger_nodes), "depots": len(depot_nodes), "customers": len(cust_nodes),
                   "delivery_nodes_with_assumed_population": n_assumed_pop,
                   "dcfast_ports_in_nodes": int(sum(c["ports"] for c in charger_nodes)),
                   "dcfast_sites_available": len(merged), "dcfast_stations_raw": len(sites),
                   "dcfast_stations_modelled": sum(1 for s in sites if s.get("modelled", False))},
        "study_bbox_min_lat_lon_max_lat_lon": [round(x, 5) for x in bbox],
        "road_network": {"nodes": int(net.n), "directed_edges": int(len(net.eu)), "plotted_segments": n_road,
                         "excluded": "service roads, private/no-motor-vehicle ways"
                                     + (f", road classes below {preset['max_road_class']}" if "max_road_class" in preset else ""),
                         **{k: v for k, v in terrain_info.items()}},
        "vehicle_profile": profile, "vehicle_model": vehicle,
        "route_policy": "minimum battery energy (regeneration allowed on the road graph); time summed on the same route",
        "leg_energy_floor_kwh": MIN_LEG_KWH, "legs_floored": neg,
        "od_stats": od_stats,
        "ranges": ranges,
        "asymmetry": asym,
        "charger_clusters": {"max_diameter_m": max(c["diameter_m"] for c in charger_nodes),
                             "merged_nodes_with_multiple_stations": sum(c["n_stations"] > 1 for c in charger_nodes),
                             "modelled_sites": sum(1 for c in charger_nodes if c.get("modelled", False))},
        "sources": {
            "delivery_and_depot_locations": delivery_src,
            "road_network": "OpenStreetMap contributors via Geofabrik, ODbL",
            "terrain": preset.get("terrain_source", "AWS Terrain Tiles (Terrarium; USGS 3DEP/NED over the US)"),
            "chargers": charger_src,
        },
    }
    if npy:
        meta["format"] = "npy"
        meta["files"] = {"energy": "od_energy_kwh.npy", "time": "od_time_h.npy", "node_ids": "od_node_ids.npy",
                         "layout": "float32 (nodes x nodes); row/column i is OSM node od_node_ids[i] (ascending); "
                                   "diagonal 0; entry [i, j] is the minimum-energy route i -> j"}
    cfg = make_config(name, preset["title"], preset)
    meta["battery_kwh"] = float(yaml.safe_load(cfg.read_text())["truck"]["battery_capacity"])
    meta["config_file"] = str(cfg.relative_to(REPO))
    json.dump(meta, open(out / "metadata.json", "w"), indent=1)
    log(f"  done -> {out}")
    log(f"  asymmetry: {json.dumps(asym)}")
    return out, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preset")
    ap.add_argument("--raw-dir", type=Path, default=fetch_data.default_raw_dir())
    ap.add_argument("--out-root", type=Path, default=REPO / "EVRoutingEnv" / "data" / "case_studies")
    ap.add_argument("--n-customers", type=int)
    ap.add_argument("--max-chargers", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--workers", type=int, help="processes for the all-pairs route computation")
    ap.add_argument("--extract-only", metavar="REGION", help="only extract and cache the road network of one Geofabrik "
                    "region of the preset (e.g. europe/france); run one job per region, then build normally")
    ap.add_argument("--scan-only", metavar="REGION", help="only scan one region's charging stations and places into the cache")
    ap.add_argument("--stage", choices=["all", "prepare", "od", "assemble"], default="all",
                    help="npy presets only. prepare: network, terminals and routing graph; od: one shard of the routing "
                         "(--shard/--n-shards, one Slurm array task each); assemble: matrices, visualisation data, metadata")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args()
    if a.scan_only:
        pre = yaml.safe_load(open(a.preset))
        scan_region(a.raw_dir, fetch_data.ensure_osm(a.raw_dir, a.scan_only), pre, tuple(pre["bbox"]))
        raise SystemExit(0)
    if a.extract_only:
        pre = yaml.safe_load(open(a.preset))
        extract_region(a.raw_dir, fetch_data.ensure_osm(a.raw_dir, a.extract_only), tuple(pre["bbox"]),
                       rg.CLASS_ID[pre.get("max_road_class", "living_street")])
        raise SystemExit(0)
    build(yaml.safe_load(open(a.preset)), a.raw_dir, a.out_root,
          {"n_customers": a.n_customers, "max_chargers": a.max_chargers, "seed": a.seed, "workers": a.workers},
          stage=a.stage, shard=a.shard, n_shards=a.n_shards)
