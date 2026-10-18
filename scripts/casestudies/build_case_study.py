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
        log("extracting road network from", pbf.name)
        nets.append(rg.extract(pbf, bbox, max_class))
        log(f"  {nets[-1].n} nodes, {len(nets[-1].eu)} edges")
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


def load_place_candidates(raw: Path, preset: dict):
    places = []
    for r in _as_list(preset["geofabrik"]):
        places += rg.load_places(fetch_data.ensure_osm(raw, r), min_pop=preset.get("min_population", 500))
    return places


def sample_places(places: list[dict], n: int, rng) -> list[dict]:
    """Population-weighted (sqrt) draw of distinct towns/cities: big cities are likelier, small towns appear."""
    w = np.sqrt(np.array([p["population"] for p in places], float))
    idx = rng.choice(len(places), size=min(n * 3, len(places)), replace=False, p=w / w.sum())
    return [places[i] for i in idx]  # over-draw: some will drop on snapping/dedup


# ------------------------------------------------------------------ snapping
class Snapper:
    """Nearest drivable node, excluding freeway-class nodes (stops are never on a freeway)."""

    def __init__(self, net: rg.RoadNetwork):
        ok = np.zeros(net.n, bool)
        low = net.hwclass >= rg.CLASS_ID["primary"]
        ok[net.eu[low]] = True
        ok[net.ev[low]] = True
        self.idx = np.nonzero(ok)[0]
        self.lat0 = float(net.lat.mean())
        self.tree = cKDTree(chg.local_xy(net.lat[self.idx], net.lon[self.idx], self.lat0))

    def __call__(self, lat, lon):
        d, i = self.tree.query(chg.local_xy(lat, lon, self.lat0))
        if np.ndim(lat) == 0:
            return int(self.idx[i[0]]), float(d[0])
        return self.idx[i], d


# --------------------------------------------------------------- OD matrices
_OD: dict = {}


def _od_rows(args):
    """Dijkstra from a few sources, then walk the predecessor tree back from every terminal."""
    c0, src = args
    csr, keys, e_u, t_u, l_u = _OD["csr"], _OD["keys"], _OD["e"], _OD["t"], _OD["l"]
    terminals, kappa, elev, n = _OD["terminals"], _OD["kappa"], _OD["elev"], _OD["n"]
    D, P = dijkstra(csr, directed=True, indices=src, return_predecessors=True)
    rows = []
    for k, s in enumerate(src):
        cur = terminals.astype(np.int64).copy()
        at = np.zeros(len(cur)); ae = np.zeros(len(cur)); al = np.zeros(len(cur))
        pk = P[k]
        while True:
            act = np.nonzero(cur != s)[0]
            if act.size == 0:
                break
            c = cur[act]
            p = pk[c].astype(np.int64)
            if (p < 0).any():            # unreachable target
                bad = act[p < 0]
                at[bad] = ae[bad] = al[bad] = np.nan
                cur[bad] = s
                keep = p >= 0
                act, c, p = act[keep], c[keep], p[keep]
                if act.size == 0:
                    continue
            ei = np.searchsorted(keys, p * n + c)
            at[act] += t_u[ei]; ae[act] += e_u[ei]; al[act] += l_u[ei]
            cur[act] = p
        e_red = D[k, terminals] + kappa * (elev[terminals] - elev[s])
        rows.append((e_red, at, ae, al))
    return c0, rows


def od_matrices(net: rg.RoadNetwork, terminals: np.ndarray, t_h, e_kwh, vehicle, chunk=4, workers=None):
    """Minimum-energy routes between terminals, with time/length summed on the same route.

    Edge energies can be negative (regeneration). Reweighting by the elevation potential
    c' = c - kappa*(h_v - h_u) >= 0 makes Dijkstra exact; the true net energy of a route is
    D' + kappa*(h_t - h_s). Energy is also re-accumulated along each route as a check.
    """
    import multiprocessing as mp

    n = net.n
    kappa = rg.potential_slope(vehicle)
    dh = net.elev[net.ev] - net.elev[net.eu]
    red = e_kwh - kappa * dh
    stats = {"reduced_cost_min": float(red.min()), "n_negative_reduced": int((red < -1e-9).sum())}
    red = np.maximum(red, 0.0) + 1e-9
    # parallel edges: keep the cheapest
    order = np.lexsort((red, net.ev, net.eu))
    eu, ev = net.eu[order], net.ev[order]
    first = np.r_[True, (eu[1:] != eu[:-1]) | (ev[1:] != ev[:-1])]
    sel = order[first]
    eu, ev = net.eu[sel].astype(np.int64), net.ev[sel].astype(np.int64)
    _OD.update(csr=csr_matrix((red[sel], (eu, ev)), shape=(n, n)), keys=eu * n + ev, e=e_kwh[sel], t=t_h[sel],
               l=net.length[sel], terminals=terminals, kappa=kappa, elev=net.elev, n=n)
    T = len(terminals)
    E = np.zeros((T, T)); TT = np.zeros((T, T)); LL = np.zeros((T, T)); AE = np.zeros((T, T))
    jobs = [(c0, terminals[c0:c0 + chunk]) for c0 in range(0, T, chunk)]
    workers = workers or min(16, max(1, (mp.cpu_count() or 2) // 4))
    done = 0
    with mp.get_context("fork").Pool(workers) as pool:
        for c0, rows in pool.imap_unordered(_od_rows, jobs):
            for k, (e_red, at, ae, al) in enumerate(rows):
                E[c0 + k], TT[c0 + k], LL[c0 + k], AE[c0 + k] = e_red, at, al, ae
            done += len(rows)
            if (done // chunk) % 10 == 0 or done == T:
                log(f"  OD rows {done}/{T}")
    stats["max_energy_consistency_error_kwh"] = float(np.nanmax(np.abs(np.where(np.isfinite(E), E - AE, 0))))
    stats["unreachable_pairs"] = int((~np.isfinite(E)).sum())
    _OD.clear()
    return E, TT, LL / 1000.0, stats


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


def build(preset: dict, raw: Path, out_root: Path, overrides: dict):
    preset = {**preset, **{k: v for k, v in overrides.items() if v is not None}}
    name = preset["name"]
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
        sites = chg.load_dcfast_osm([fetch_data.ensure_osm(raw, r) for r in _as_list(preset["geofabrik"])], bbox,
                                    float(preset.get("min_charger_kw", 50.0)), preset.get("exclude_tesla_only", False))
        charger_src = (f"OpenStreetMap amenity=charging_station (CCS/CHAdeMO/NACS sockets, tagged output >= "
                       f"{float(preset.get('min_charger_kw', 50.0)):g} kW), ODbL")
    else:
        raise SystemExit(f"unknown chargers_source {charger_source!r} (afdc or osm)")
    merged = chg.merge_sites(sites, preset["charger_merge_m"], preset.get("charger_merge_linkage", "complete"))
    # sites that snap to the same road node are one charger node; sites far from any road are dropped
    by_node: dict[int, list[dict]] = {}
    for c in merged:
        i, d = snap(c["lat"], c["lon"])
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

    depot_nodes, cust_nodes = [], []
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
                                "snap_m": c["snap_m"], "population": c["population"]})
        depot_nodes.reverse()
        delivery_src = "OpenStreetMap place nodes (city/town/village with population tag), ODbL"
    if len(cust_nodes) + len(depot_nodes) < preset["n_customers"]:
        log(f"  WARNING only {len(cust_nodes) + len(depot_nodes)} delivery nodes placed")

    terminals_meta = ([("charger", c) for c in charger_nodes] + [("depot", d) for d in depot_nodes]
                      + [("customer", c) for c in cust_nodes])
    term_idx = np.array([m["node_idx"] for _, m in terminals_meta])
    ids = [int(net.osm_id[i]) for i in term_idx]
    log(f"  terminals: {len(charger_nodes)} chargers, {len(depot_nodes)} depots, {len(cust_nodes)} customers")

    t_h, e_kwh, _ = rg.edge_costs(net, vehicle)
    log("  computing all-pairs minimum-energy routes between terminals")
    E_net, TT, KM, od_stats = od_matrices(net, term_idx, t_h, e_kwh, vehicle, workers=preset.get("workers"))
    if od_stats["unreachable_pairs"] - len(ids) > 0:
        raise SystemExit("unreachable terminal pairs in a strongly connected graph: " + str(od_stats))
    np.fill_diagonal(E_net, 0); np.fill_diagonal(TT, 0); np.fill_diagonal(KM, 0)
    neg = int((E_net[~np.eye(len(ids), dtype=bool)] < MIN_LEG_KWH).sum())
    E = np.maximum(E_net, MIN_LEG_KWH); np.fill_diagonal(E, 0)
    asym = asymmetry_stats(E_net, TT)

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
        })
    json.dump(node_rows, open(vis / "nodes.json", "w"))
    json.dump([{
        "node": int(net.osm_id[c["node_idx"]]), "lat": round(float(net.lat[c["node_idx"]]), 6),
        "lon": round(float(net.lon[c["node_idx"]]), 6),
        "site_lat": round(c["lat"], 6), "site_lon": round(c["lon"], 6), "ports": c["ports"],
        "n_stations": c["n_stations"], "diameter_m": c["diameter_m"], "snap_m": c["snap_m"],
        "names": c["names"], "networks": c["networks"], "connectors": c["connectors"], "afdc_ids": c["afdc_ids"],
    } for c in charger_nodes], open(vis / "stations.json", "w"), indent=1)
    chosen_ids = {i for x in chosen for i in x["afdc_ids"]}
    json.dump([{"cluster_rank": i, "lat": round(c["lat"], 6), "lon": round(c["lon"], 6), "ports": c["ports"],
                "n_stations": c["n_stations"], "networks": c["networks"], "selected": c["afdc_ids"][0] in chosen_ids}
               for i, c in enumerate(merged)], open(vis / "all_dcfast_sites.json", "w"))
    n_road = write_roads(vis / "roads.npz", net, rg.CLASS_ID[preset.get("vis_max_class", "tertiary_link")])
    ext_km = max((bbox[2] - bbox[0]) * 111.0, (bbox[3] - bbox[1]) * 111.0 * np.cos(np.radians(0.5 * (bbox[0] + bbox[2]))))
    write_terrain_grid(vis / "terrain.npz", raw, bbox, zoom, step_m=max(250.0, ext_km * 1000.0 / 700.0))
    np.savez_compressed(vis / "od_matrices.npz", node_ids=np.array(ids, np.int64), energy_net_kwh=E_net.astype(np.float32),
                        energy_kwh=E.astype(np.float32), time_h=TT.astype(np.float32), dist_km=KM.astype(np.float32))

    off_diag = ~np.eye(len(ids), dtype=bool)
    meta = {
        "name": name, "title": preset["title"], "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preset": preset, "units": {"energy": "kWh (battery)", "time": "hours", "distance": "km"},
        "scale_km": round(float(ext_km), 0),
        "counts": {"nodes": len(ids), "chargers": len(charger_nodes), "depots": len(depot_nodes), "customers": len(cust_nodes),
                   "dcfast_ports_in_nodes": int(sum(c["ports"] for c in charger_nodes)),
                   "dcfast_sites_available": len(merged), "dcfast_stations_raw": len(sites)},
        "study_bbox_min_lat_lon_max_lat_lon": [round(x, 5) for x in bbox],
        "road_network": {"nodes": int(net.n), "directed_edges": int(len(net.eu)), "plotted_segments": n_road,
                         "excluded": "service roads, private/no-motor-vehicle ways"
                                     + (f", road classes below {preset['max_road_class']}" if "max_road_class" in preset else ""),
                         **{k: v for k, v in terrain_info.items()}},
        "vehicle_profile": profile, "vehicle_model": vehicle,
        "route_policy": "minimum battery energy (regeneration allowed on the road graph); time summed on the same route",
        "leg_energy_floor_kwh": MIN_LEG_KWH, "legs_floored": neg,
        "od_stats": od_stats,
        "ranges": {
            "energy_kwh_p5_p50_p95_max": [round(float(x), 2) for x in np.percentile(E[off_diag], [5, 50, 95, 100])],
            "time_h_p5_p50_p95_max": [round(float(x), 3) for x in np.percentile(TT[off_diag], [5, 50, 95, 100])],
            "dist_km_p5_p50_p95_max": [round(float(x), 1) for x in np.percentile(KM[off_diag], [5, 50, 95, 100])],
            "kwh_per_km_median": round(float(np.median(E[off_diag] / np.maximum(KM[off_diag], 0.1))), 3),
        },
        "asymmetry": asym,
        "charger_clusters": {"max_diameter_m": max(c["diameter_m"] for c in charger_nodes),
                             "merged_nodes_with_multiple_stations": sum(c["n_stations"] > 1 for c in charger_nodes)},
        "sources": {
            "delivery_and_depot_locations": delivery_src,
            "road_network": "OpenStreetMap contributors via Geofabrik, ODbL",
            "terrain": preset.get("terrain_source", "AWS Terrain Tiles (Terrarium; USGS 3DEP/NED over the US)"),
            "chargers": charger_src,
        },
    }
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
    a = ap.parse_args()
    build(yaml.safe_load(open(a.preset)), a.raw_dir, a.out_root,
          {"n_customers": a.n_customers, "max_chargers": a.max_chargers, "seed": a.seed, "workers": a.workers})
