"""DC-fast charging sites from the AFDC station list, merged into charger nodes."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist


def local_xy(lat, lon, lat0):
    """Equirectangular projection to metres around lat0 (good to <0.1 % over 100 km)."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    return np.c_[np.radians(lon) * 6371008.8 * np.cos(np.radians(lat0)), np.radians(lat) * 6371008.8]


def sphere_xyz(lat, lon):
    """Points on the Earth sphere in metres (chord distance = great-circle distance to <1e-6 relative at 20 km).

    ``local_xy`` is one equirectangular map around a single latitude: east-west distances are off by 15 % at the
    edges of a 17-degree-tall region, which matters when merging stations within a fixed radius at that scale.
    """
    la = np.radians(np.asarray(lat, float)); lo = np.radians(np.asarray(lon, float))
    r = 6371008.8
    return np.c_[r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)]


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


# OpenStreetMap socket keys that are DC fast charging (CCS, CHAdeMO, NACS/Tesla Supercharger)
_OSM_DC_SOCKETS = ("type2_combo", "ccs", "chademo", "tesla_supercharger", "tesla_supercharger_ccs", "nacs")
_OSM_CONNECTOR = {"type2_combo": "CCS", "ccs": "CCS", "chademo": "CHADEMO", "tesla_supercharger": "TESLA",
                  "tesla_supercharger_ccs": "TESLA", "nacs": "TESLA"}


def _osm_count(v):
    try:
        return int(float(str(v).split(";")[0]))
    except ValueError:
        return 1 if str(v).lower() in ("yes", "true") else 0


def _osm_kw(v):
    """kW from tags like '150 kW', '150000' (W), '50;22'. The largest value wins; None when absent or unparsable."""
    best = None
    for part in str(v or "").replace(",", ".").split(";"):
        m = re.match(r"\s*([0-9]*\.?[0-9]+)\s*(kw|w|mw)?", part.strip().lower())
        if not m:
            continue
        x = float(m.group(1))
        unit = m.group(2)
        x = x / 1000.0 if unit == "w" or (unit is None and x >= 1000) else x * 1000.0 if unit == "mw" else x
        best = x if best is None else max(best, x)
    return best


def osm_dcfast_ports(tags, min_kw: float = 50.0):
    """(ports, connectors) of the DC-fast sockets of one OSM charging station, from its tags.

    A socket type counts when it is a DC type (CCS, CHAdeMO, Tesla/NACS) and its tagged output is at
    least min_kw. A socket without a tagged output is kept: CCS and CHAdeMO are DC-only types, and the
    per-socket output is often missing in OSM. Plug types are not summed: one dispenser usually offers
    a CCS and a CHAdeMO cable (and Tesla sites repeat their stalls under three keys) but charges one car,
    so ports = the largest per-type count, capped by the `capacity` tag when it is smaller.
    """
    ports, conns = 0, set()
    for sock in _OSM_DC_SOCKETS:
        n = _osm_count(tags.get(f"socket:{sock}", "0"))
        if n <= 0:
            continue
        kw = _osm_kw(tags.get(f"socket:{sock}:output"))
        if kw is not None and kw < min_kw:
            continue
        ports = max(ports, n)
        conns.add(_OSM_CONNECTOR[sock])
    cap = _osm_count(tags.get("capacity", "0"))
    if ports and cap > 0:
        ports = min(ports, cap)
    return ports, sorted(conns)


def scan_osm_chargers(pbf, bbox, min_kw: float = 50.0, exclude_tesla_only: bool = False) -> list[dict]:
    """DC-fast ``amenity=charging_station`` nodes and ways of one extract, before de-duplication."""
    import osmium

    mnla, mnlo, mxla, mxlo = bbox
    out = []
    fp = osmium.FileProcessor(str(pbf)).with_locations().with_filter(osmium.filter.KeyFilter("amenity"))
    for o in fp:
        if o.is_relation() or o.tags.get("amenity") != "charging_station":
            continue
        t = o.tags
        if t.get("access") in ("private", "no", "permit") or t.get("motorcar") == "no":
            continue
        if o.is_node():
            lat, lon, kind = o.location.lat, o.location.lon, "node"
        else:
            try:
                pts = [(n.lat, n.lon) for n in o.nodes]
            except osmium.InvalidLocationError:
                continue
            if not pts:
                continue
            lat, lon = float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
            kind = "way"
        if not (mnla <= lat <= mxla and mnlo <= lon <= mxlo):
            continue
        ports, conns = osm_dcfast_ports(t, min_kw)
        if ports <= 0 or (exclude_tesla_only and set(conns) == {"TESLA"}):
            continue
        out.append({"afdc_id": f"{kind}/{o.id}", "name": t.get("name") or t.get("brand"), "lat": float(lat),
                    "lon": float(lon), "ports": int(ports), "network": t.get("operator") or t.get("network"),
                    "connectors": conns, "city": t.get("addr:city")})
    return out


def scan_osm_service_areas(pbf, bbox, assume_ports: int = 4) -> list[dict]:
    """Expressway service areas (``highway=services``/``rest_area``) as modelled DC-fast charger sites.

    Used where explicit ``amenity=charging_station`` tagging badly under-counts real chargers (e.g. China):
    OSM's mapping of the service areas themselves is far more complete than its mapping of the charging bays
    inside them, and the great majority of Chinese expressway service areas (服务区) now host a DC-fast bank
    (Ministry of Transport rollout along the national expressway network). Each service area found this way
    is given ``assume_ports`` DC-fast ports; this is a modelling assumption, not a measurement, flagged
    ``modelled: True`` per site and surfaced in the case's metadata.json so it stays distinguishable from
    real AFDC/OSM station data. Call ``drop_near`` against the real stations first so a service area that
    already carries its own ``amenity=charging_station`` tag is not double-counted.
    """
    import osmium

    mnla, mnlo, mxla, mxlo = bbox
    out = []
    fp = osmium.FileProcessor(str(pbf)).with_locations().with_filter(osmium.filter.KeyFilter("highway"))
    for o in fp:
        if o.is_relation() or o.tags.get("highway") not in ("services", "rest_area"):
            continue
        t = o.tags
        if o.is_node():
            lat, lon, kind = o.location.lat, o.location.lon, "node"
        else:
            try:
                pts = [(n.lat, n.lon) for n in o.nodes]
            except osmium.InvalidLocationError:
                continue
            if not pts:
                continue
            lat, lon = float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
            kind = "way"
        if not (mnla <= lat <= mxla and mnlo <= lon <= mxlo):
            continue
        out.append({"afdc_id": f"service/{kind}/{o.id}", "name": t.get("name"), "lat": float(lat),
                    "lon": float(lon), "ports": int(assume_ports), "network": t.get("operator") or "expressway service area",
                    "connectors": ["CCS"], "city": None, "modelled": True})
    return out


def drop_near(points: list[dict], others: list[dict], radius_m: float = 300.0) -> list[dict]:
    """Drop entries of ``points`` within ``radius_m`` of any entry of ``others`` (e.g. a service area that
    already carries its own real ``amenity=charging_station`` tag, so it is not double-counted)."""
    if not points or not others:
        return points
    tree = cKDTree(sphere_xyz([o["lat"] for o in others], [o["lon"] for o in others]))
    xyz = sphere_xyz([p["lat"] for p in points], [p["lon"] for p in points])
    near = tree.query_ball_point(xyz, r=radius_m)
    return [p for p, hits in zip(points, near) if len(hits) == 0]


def dedupe_sites(out: list[dict], dedup_m: float = 15.0) -> list[dict]:
    """Drop the same station mapped as a node and as an area, or listed in two overlapping extracts."""
    if not out or dedup_m <= 0:
        return out
    tree = cKDTree(sphere_xyz([s["lat"] for s in out], [s["lon"] for s in out]))
    drop = set()
    for i, j in sorted(tree.query_pairs(dedup_m)):
        if i in drop or j in drop:
            continue
        if out[i]["ports"] == out[j]["ports"] and out[i]["network"] == out[j]["network"]:
            drop.add(j)
    return [s for k, s in enumerate(out) if k not in drop]


def load_dcfast_osm(pbfs, bbox, min_kw: float = 50.0, exclude_tesla_only: bool = False, dedup_m: float = 15.0) -> list[dict]:
    """Public DC-fast sites from OpenStreetMap ``amenity=charging_station`` nodes and ways.

    Same record format as ``load_dcfast`` (the id list is called ``afdc_id`` for compatibility; here it is
    ``node/<id>`` or ``way/<id>``). Stations tagged private/no access or not for cars are skipped. Two
    entries within ``dedup_m`` with the same operator and port count are one station mapped twice
    (a node inside a mapped area), so only the first is kept.
    """
    out = []
    for pbf in pbfs:
        out += scan_osm_chargers(pbf, bbox, min_kw, exclude_tesla_only)
    return dedupe_sites(out, dedup_m)


def _leader_labels(xyz: np.ndarray, ports: np.ndarray, merge_m: float) -> np.ndarray:
    """Greedy leader clustering: largest unassigned site first, absorbing every unassigned site within merge_m/2.

    Any two members are then within merge_m of each other (triangle inequality), the same guarantee as complete
    linkage, but it needs a KD-tree instead of the O(n^2) distance matrix, so it scales to 10^5 stations.
    """
    tree = cKDTree(xyz)
    lab = -np.ones(len(xyz), np.int64)
    for i in np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0], -ports)):
        if lab[i] >= 0:
            continue
        near = np.asarray(tree.query_ball_point(xyz[i], r=merge_m / 2.0), np.int64)
        lab[near[lab[near] < 0]] = i
    return lab


def merge_sites(sites: list[dict], merge_m: float, linkage_method: str = "single") -> list[dict]:
    """Merge stations closer than merge_m (single linkage by default); ports are summed.

    ``linkage_method="complete"`` instead guarantees every pair inside a node is within merge_m,
    which keeps large radii from chaining neighbouring towns into one node. ``"leader"`` gives the same
    guarantee with a greedy radius-merge_m/2 clustering that needs no distance matrix (for 10^5 stations), but
    it leaves clearly more nodes: 2405 against 1584 for the 12k stations of europe_10k at 20 km. Regions
    taller than 5 degrees of latitude are measured on the sphere rather than on a single-latitude map.

    The merged location is the port-weighted centroid. ``diameter_m`` records the largest
    distance between member stations so chained merges are visible in the metadata.
    """
    if not sites:
        return []
    lat0 = float(np.mean([s["lat"] for s in sites]))
    lat_span = max(s["lat"] for s in sites) - min(s["lat"] for s in sites)
    # a single-latitude map is good to <0.1 % over 100 km but 15 % off at the edges of a 17-degree region
    xy = (sphere_xyz if lat_span > 5.0 else lambda la, lo: local_xy(la, lo, lat0))(
        [s["lat"] for s in sites], [s["lon"] for s in sites])
    if linkage_method == "leader":
        lab = _leader_labels(xy, np.array([s["ports"] for s in sites], float), merge_m)
    else:
        lab = fcluster(linkage(xy, linkage_method), t=merge_m, criterion="distance") if len(sites) > 1 else np.array([1])
    order = np.argsort(lab, kind="stable")
    cuts = np.r_[0, np.nonzero(np.diff(lab[order]))[0] + 1, len(lab)]
    clusters = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        mem = [sites[i] for i in idx]
        w = np.array([m["ports"] for m in mem], float)
        diam = float(pdist(xy[idx]).max()) if len(idx) > 1 else 0.0
        clusters.append({
            "lat": float(np.average([m["lat"] for m in mem], weights=w)),
            "lon": float(np.average([m["lon"] for m in mem], weights=w)),
            "ports": int(w.sum()),
            "n_stations": len(mem),
            "diameter_m": round(diam, 1),
            "displacement_m": round(float(np.linalg.norm(xy[idx] - np.average(xy[idx], axis=0, weights=w), axis=1).max()), 1),
            "names": sorted({m["name"] for m in mem if m["name"]}),
            "networks": sorted({m["network"] for m in mem if m["network"]}),
            "connectors": sorted({c for m in mem for c in m["connectors"]}),
            "afdc_ids": [m["afdc_id"] for m in mem],
            "modelled": all(m.get("modelled", False) for m in mem),
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
        "modelled": all(g.get("modelled", False) for g in group),
    }
