"""DC-fast charging sites from the AFDC station list, merged into charger nodes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def local_xy(lat, lon, lat0):
    """Equirectangular projection to metres around lat0 (good to <0.1 % over 100 km)."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    return np.c_[np.radians(lon) * 6371008.8 * np.cos(np.radians(lat0)), np.radians(lat) * 6371008.8]


def load_dcfast(afdc_json: Path, bbox, exclude_tesla_only: bool = False) -> list[dict]:
    """Open, public stations with at least one DC-fast port inside bbox=(min_lat,min_lon,max_lat,max_lon)."""
    mnla, mnlo, mxla, mxlo = bbox
    out = []
    for s in json.load(open(afdc_json))["fuel_stations"]:
        ports = s.get("ev_dc_fast_num") or 0
        if ports <= 0 or s.get("status_code") != "E" or s.get("access_code") != "public":
            continue
        lat, lon = s.get("latitude"), s.get("longitude")
        if lat is None or lon is None or not (mnla <= lat <= mxla and mnlo <= lon <= mxlo):
            continue
        conns = s.get("ev_connector_types") or []
        if exclude_tesla_only and conns and set(conns) == {"TESLA"}:
            continue
        out.append({
            "afdc_id": s["id"], "name": s.get("station_name"), "lat": float(lat), "lon": float(lon),
            "ports": int(ports), "network": s.get("ev_network"), "connectors": conns,
            "city": s.get("city"),
        })
    return out


def merge_sites(sites: list[dict], merge_m: float) -> list[dict]:
    """Single-linkage merge of stations closer than merge_m; ports are summed.

    The merged location is the port-weighted centroid. ``diameter_m`` records the largest
    distance between member stations so chained merges are visible in the metadata.
    """
    if not sites:
        return []
    lat0 = float(np.mean([s["lat"] for s in sites]))
    xy = local_xy([s["lat"] for s in sites], [s["lon"] for s in sites], lat0)
    lab = fcluster(linkage(xy, "single"), t=merge_m, criterion="distance") if len(sites) > 1 else np.array([1])
    clusters = []
    for c in np.unique(lab):
        idx = np.nonzero(lab == c)[0]
        mem = [sites[i] for i in idx]
        w = np.array([m["ports"] for m in mem], float)
        diam = float(pdist(xy[idx]).max()) if len(idx) > 1 else 0.0
        clusters.append({
            "lat": float(np.average([m["lat"] for m in mem], weights=w)),
            "lon": float(np.average([m["lon"] for m in mem], weights=w)),
            "ports": int(w.sum()),
            "n_stations": len(mem),
            "diameter_m": round(diam, 1),
            "names": sorted({m["name"] for m in mem if m["name"]}),
            "networks": sorted({m["network"] for m in mem if m["network"]}),
            "connectors": sorted({c for m in mem for c in m["connectors"]}),
            "afdc_ids": [m["afdc_id"] for m in mem],
        })
    return sorted(clusters, key=lambda c: (-c["ports"], c["lat"], c["lon"]))


def spread_subset(clusters: list[dict], k: int) -> list[dict]:
    """Farthest-point (k-center) subset, seeded with the largest site, to keep coverage."""
    if k <= 0 or len(clusters) <= k:
        return clusters
    lat0 = float(np.mean([c["lat"] for c in clusters]))
    xy = local_xy([c["lat"] for c in clusters], [c["lon"] for c in clusters], lat0)
    chosen = [0]  # clusters are sorted by ports, descending
    d = np.linalg.norm(xy - xy[0], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(xy - xy[nxt], axis=1))
    return [clusters[i] for i in sorted(chosen)]


def combine(group: list[dict]) -> dict:
    """Merge already-merged sites that land on one road node (ports add up)."""
    if len(group) == 1:
        return dict(group[0])
    w = np.array([g["ports"] for g in group], float)
    return {
        "lat": float(np.average([g["lat"] for g in group], weights=w)),
        "lon": float(np.average([g["lon"] for g in group], weights=w)),
        "ports": int(w.sum()),
        "n_stations": sum(g["n_stations"] for g in group),
        "diameter_m": max(g["diameter_m"] for g in group),
        "names": sorted({n for g in group for n in g["names"]}),
        "networks": sorted({n for g in group for n in g["networks"]}),
        "connectors": sorted({n for g in group for n in g["connectors"]}),
        "afdc_ids": [i for g in group for i in g["afdc_ids"]],
    }
