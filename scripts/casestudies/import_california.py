"""One-time import of the original California data into the layout of the generated case studies.

    python scripts/casestudies/import_california.py --source <dir with the original files> --raw-dir .casestudy_raw

This was run once to create EVRoutingEnv/data/case_studies/california/ (the default scenario). The
original files (shortest_path_*_dict.json, station_info_dict.json, waiting_time_lookup.json,
vis_data/node_CA_df.json, station_CA_df.json) were then removed from EVRoutingEnv/data, so California is
loaded from the case-study folder only. To rerun this script, restore those files from the repository
history into --source. It writes the same files as the other cases (dictionaries, station info, waiting
time lookup, metadata, vis_data; figures with visualize_case_study.py). The default config
config_files/config_joint.yaml points at the folder.

Node elevation, the plotted road backdrop and its grades are derived here with the same pipeline as the other
cases (OSM motorway..primary roads + terrain tiles), because the shipped vis_data/california_hgv_mw-tr-pr-se.osm
is a GraphML file that is cut off mid-edge. The energy and time dictionaries, the nodes and the stations are
exactly the original data.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_case_study as bcs  # noqa: E402
import fetch_data  # noqa: E402
import road_graph as rg  # noqa: E402

OUT = bcs.CA_DATA  # EVRoutingEnv/data/case_studies/california


def load_dict(path: Path) -> dict[tuple[int, int], float]:
    out = {}
    for key, value in json.load(open(path)).items():
        u, v = map(int, re.findall(r"-?\d+", key))
        out[(u, v)] = value
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742.0 * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=fetch_data.default_raw_dir())
    ap.add_argument("--zoom", type=int, default=10)
    ap.add_argument("--source", type=Path, default=bcs.REPO / "EVRoutingEnv" / "data", help="folder with the original California files")
    a = ap.parse_args()
    DATA = a.source
    (OUT / "vis_data").mkdir(parents=True, exist_ok=True)
    bcs.log("California: reading the original files")

    energy = load_dict(DATA / "shortest_path_energy_dict.json")
    time_h = load_dict(DATA / "shortest_path_time_dict.json")
    ids = sorted({u for u, _ in energy} | {v for _, v in energy})
    index = {n: i for i, n in enumerate(ids)}
    lanes = json.load(open(DATA / "vis_data" / "node_CA_df.json"))
    stations = json.load(open(DATA / "vis_data" / "station_CA_df.json"))
    station_info = json.load(open(DATA / "station_info_dict.json"))

    # --- nodes: delivery nodes from the freight lanes, charger nodes from the station table
    coords = {}
    for r in lanes:
        coords.setdefault(int(r["origin_node"]), (r["Start_Latitude"], r["Start_Longitude"], r["origin_city"]))
        coords.setdefault(int(r["destination_node"]), (r["End_Latitude"], r["End_Longitude"], r["destination_city"]))
    by_node = {int(s["node"]): s for s in stations}
    assert set(ids) <= set(coords) | set(by_node), "nodes without coordinates"
    lat = np.array([by_node[n]["Latitude"] if n in by_node else coords[n][0] for n in ids], float)
    lon = np.array([by_node[n]["Longitude"] if n in by_node else coords[n][1] for n in ids], float)
    bbox = (lat.min() - 0.3, lon.min() - 0.3, lat.max() + 0.3, lon.max() + 0.3)

    # --- roads and terrain, through the same pipeline as the other cases. The shipped
    # vis_data/california_hgv_mw-tr-pr-se.osm is a GraphML file that is cut off mid-edge (about 60 % of the
    # edges survive), so the plotted network is re-extracted from OpenStreetMap: motorway to primary roads.
    pbf = fetch_data.ensure_osm(a.raw_dir, "north-america/us/california")
    bcs.log("extracting motorway..primary roads from the OSM extract")
    net = rg.extract(pbf, (32.0, -125.0, 42.5, -114.0), rg.CLASS_ID["primary_link"])
    bcs.log(f"  {net.n} nodes, {len(net.eu)} edges")
    info = rg.attach_elevation(net, a.raw_dir, 150.0, a.zoom, min_elevation_m=-100.0)
    mos = info.pop("mosaic")
    elev = np.maximum(mos.sample(lat, lon), -100.0)
    n_road = bcs.write_roads(OUT / "vis_data" / "roads.npz", net, rg.CLASS_ID["primary_link"])
    ext_km = max((bbox[2] - bbox[0]) * 111.0, (bbox[3] - bbox[1]) * 111.0 * np.cos(np.radians(0.5 * (bbox[0] + bbox[2]))))
    bcs.write_terrain_grid(OUT / "vis_data" / "terrain.npz", a.raw_dir, bbox, a.zoom, step_m=max(250.0, ext_km * 1000.0 / 700.0))

    node_rows = []
    for i, n in enumerate(ids):
        charger = n in by_node
        node_rows.append({
            "node": n, "role": "charger" if charger else "customer", "lat": round(float(lat[i]), 6), "lon": round(float(lon[i]), 6),
            "elev_m": round(float(elev[i]), 1), "snap_m": 0.0,
            **({"station_type": station_info[str(n)]["station_type"], "zone_id": by_node[n]["Station Name"]} if charger else {"zone_id": coords[n][2]}),
        })
    json.dump(node_rows, open(OUT / "vis_data" / "nodes.json", "w"))

    def conn(s):
        return [c for c in str(s.get("EV Connector Types") or "").split() if c]

    st_rows = []
    for s in stations:
        n = int(s["node"]); kind = station_info[str(n)]["station_type"]
        st_rows.append({
            "node": n, "lat": float(s["Latitude"]), "lon": float(s["Longitude"]), "site_lat": float(s["Latitude"]), "site_lon": float(s["Longitude"]),
            "ports": int(station_info[str(n)]["total_capacity"]), "station_type": kind, "n_stations": 1, "diameter_m": 0.0, "snap_m": 0.0,
            "names": [s["Station Name"]], "networks": [s["EV Network"]] if s.get("EV Network") else [], "connectors": conn(s), "afdc_ids": [s["ID"]],
        })
    json.dump(st_rows, open(OUT / "vis_data" / "stations.json", "w"), indent=1)
    json.dump([{"cluster_rank": i, "lat": r["lat"], "lon": r["lon"], "ports": r["ports"], "n_stations": 1, "networks": r["networks"],
                "selected": True, "station_type": r["station_type"]} for i, r in enumerate(st_rows)], open(OUT / "vis_data" / "all_dcfast_sites.json", "w"))

    # --- matrices (NaN where the original dictionary has no pair, 257 of 66306)
    n = len(ids)
    E = np.full((n, n), np.nan); T = np.full((n, n), np.nan); K = np.full((n, n), np.nan)
    for (u, v), x in energy.items():
        E[index[u], index[v]] = x
    for (u, v), x in time_h.items():
        T[index[u], index[v]] = x
    lane_pairs = [(index[int(r["origin_node"])], index[int(r["destination_node"])], r["road_only_length"]) for r in lanes
                  if int(r["origin_node"]) in index and int(r["destination_node"]) in index and r["road_only_length"] is not None]
    for i, j, length in lane_pairs:  # road length is only known for the freight lanes
        K[i, j] = length
    np.fill_diagonal(E, 0); np.fill_diagonal(T, 0); np.fill_diagonal(K, 0)
    np.savez_compressed(OUT / "vis_data" / "od_matrices.npz", node_ids=np.array(ids, np.int64), energy_net_kwh=E.astype(np.float32),
                        energy_kwh=E.astype(np.float32), time_h=T.astype(np.float32), dist_km=K.astype(np.float32))

    # --- same file names as the other cases; the original dictionaries are copied unchanged
    for name in ("shortest_path_energy_dict.json", "shortest_path_time_dict.json", "station_info_dict.json", "waiting_time_lookup.json"):
        if (DATA / name).resolve() != (OUT / name).resolve():
            shutil.copy(DATA / name, OUT / name)

    off = ~np.eye(n, dtype=bool) & np.isfinite(E) & np.isfinite(T)
    km = np.array([E[i, j] / length for i, j, length in lane_pairs if length > 20 and np.isfinite(E[i, j]) and E[i, j] > 0])
    Ef = np.where(np.isfinite(E), E, 0.0)
    asym = bcs.asymmetry_stats(np.where(np.isfinite(E), E, 0.0), np.where(np.isfinite(T), T, 0.0))
    kinds = {k: sum(v["station_type"] == k for v in station_info.values()) for k in {v["station_type"] for v in station_info.values()}}
    cfg = bcs.REPO / "EVRoutingEnv" / "config_files" / "config_joint.yaml"  # the default config points at this folder
    meta = {
        "name": "california", "title": "California (existing data)", "generated": False,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "units": {"energy": "kWh (battery)", "time": "hours", "distance": "km"}, "scale_km": round(float(ext_km), 0),
        "preset": {"note": "not generated by build_case_study.py: this is the original data of the paper"},
        "counts": {"nodes": n, "chargers": len(station_info), "depots": 0, "customers": n - len(station_info),
                   "charger_ports_total": int(sum(v["total_capacity"] for v in station_info.values())), "chargers_by_type": kinds,
                   "dcfast_ports_in_nodes": int(sum(v["total_capacity"] for v in station_info.values() if v["station_type"] == "DCFast"))},
        "study_bbox_min_lat_lon_max_lat_lon": [round(x, 5) for x in bbox],
        "road_network": {"source": "OpenStreetMap (Geofabrik California extract), motorway..primary, for plotting only; "
                                   "the shipped california_hgv_mw-tr-pr-se.osm is truncated GraphML",
                         "nodes": int(net.n), "directed_edges": int(len(net.eu)), "plotted_segments": n_road, **info},
        "vehicle_profile": "not stated (Li et al. 2025 data)", "vehicle_model": {"note": "energy model of the original data is not documented in the repository",
                                                                                 "implied_kwh_per_km_median_on_freight_lanes": round(float(np.median(km)), 3)},
        "route_policy": "minimum battery energy on the original network; negative-energy pairs are excluded by the environment",
        "nodes_outside_largest_strong_component": [  # destination-only nodes: nothing in the original data leaves them
            {"node": int(x), "in_degree": int(np.isfinite(E[:, index[x]]).sum() - 1), "out_degree": int(np.isfinite(E[index[x]]).sum() - 1)}
            for x in ids if np.isfinite(E[index[x]]).sum() - 1 == 0],
        "legs_negative": int((E[~np.eye(n, dtype=bool) & np.isfinite(E)] < 0).sum()),
        "missing_pairs": int((~np.isfinite(E) & ~np.eye(n, dtype=bool)).sum()),
        "ranges": {
            "energy_kwh_p5_p50_p95_max": [round(float(x), 2) for x in np.percentile(E[off], [5, 50, 95, 100])],
            "time_h_p5_p50_p95_max": [round(float(x), 3) for x in np.percentile(T[off], [5, 50, 95, 100])],
            "dist_km_p5_p50_p95_max": [round(float(x), 1) for x in np.nanpercentile(K[np.isfinite(K) & (K > 0)], [5, 50, 95, 100])],
            "kwh_per_km_median": round(float(np.median(km)), 3),
        },
        "asymmetry": asym,
        "sources": {"energy_time_network_stations": "Li et al. (HPEC 2025) California logistics network as shipped in EVRoutingEnv/data",
                    "terrain": "AWS Terrain Tiles (Terrarium; USGS 3DEP/NED), only for node elevation and plotted road grade"},
    }
    meta["battery_kwh"] = float(__import__("yaml").safe_load(cfg.read_text())["truck"]["battery_capacity"])
    meta["config_file"] = str(cfg.relative_to(bcs.REPO))
    json.dump(meta, open(OUT / "metadata.json", "w"), indent=1)
    bcs.log(f"done -> {OUT}")
    bcs.log(f"  {n} nodes, {len(station_info)} stations {kinds}, {meta['counts']['charger_ports_total']} ports, asymmetry {json.dumps(asym)}")


if __name__ == "__main__":
    main()
