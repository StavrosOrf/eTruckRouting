"""Directed drivable road graph from OpenStreetMap, with terrain and an EV energy model.

Pipeline: PBF -> directed segment graph -> largest strongly connected component ->
node elevation from terrain tiles (graph-smoothed, bridges/tunnels interpolated) ->
per-edge time (kh) and battery energy (kWh) that depend on direction.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg

import fetch_data

G0 = 9.81
CLASSES = [
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street",
]  # `service` roads (parking lots, driveways) are deliberately excluded
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
DEFAULT_KPH = np.array([105, 60, 90, 50, 65, 40, 55, 35, 45, 30, 40, 40, 15], float)
# realised mean speed as a fraction of the posted/default limit (urban congestion, turns)
SPEED_FACTOR = np.array([0.85, 0.70, 0.80, 0.65, 0.70, 0.60, 0.70, 0.60, 0.70, 0.60, 0.75, 0.75, 0.80])

# All values are modelling assumptions recorded in each case's metadata.json; override them in a
# preset under `vehicle:` or pick another profile with `vehicle_profile:`.
_COMMON = {
    "air_density": 1.2,
    "signal_delay_s": 20.0,     # mean wait at a traffic signal
    "signal_stop_prob": 0.5,
    "stop_delay_s": 8.0,
    "stop_stop_prob": 1.0,
}
VEHICLES = {
    # Class-8 battery-electric truck, 36 t gross (US 80,000 lb limit). Calibrated so that the median
    # consumption over routes (about 1.3-1.6 kWh/km) matches the California case study data
    # (median 1.57 kWh/km, 10th-90th percentile 1.13-1.97), which pairs with a 400 kWh battery.
    "heavy_truck": {**_COMMON, "mass_kg": 36000.0, "crr": 0.007, "cda_m2": 6.5,
                    "eta_drive": 0.90, "eta_regen": 0.65, "aux_kw": 6.0},
    # Medium electric delivery van, loaded (E-Transit / EDV class), about 0.3 kWh/km.
    "van": {**_COMMON, "mass_kg": 4200.0, "crr": 0.010, "cda_m2": 2.8,
            "eta_drive": 0.80, "eta_regen": 0.60, "aux_kw": 2.0},
}
DEFAULT_VEHICLE = VEHICLES["heavy_truck"]


def _hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742000.0 * np.arcsin(np.sqrt(a))


def _speed(tag):
    if not tag:
        return None
    s = tag.split(";")[0].strip().lower()
    try:
        if s.endswith("mph"):
            return float(s[:-3]) * 1.609344
        return float(s.split()[0])
    except ValueError:
        return None


@dataclass
class RoadNetwork:
    osm_id: np.ndarray       # (n,) int64
    lat: np.ndarray
    lon: np.ndarray
    ctrl: np.ndarray         # (n,) int8: 0 none, 1 traffic signal, 2 stop sign
    eu: np.ndarray           # (m,) int32 edge tail
    ev: np.ndarray           # (m,) int32 edge head
    length: np.ndarray       # (m,) horizontal metres
    kph: np.ndarray
    hwclass: np.ndarray      # (m,) int8 index into CLASSES
    span: np.ndarray         # (m,) bool, bridge or tunnel
    elev_dem: np.ndarray | None = None
    elev: np.ndarray | None = None   # smoothed elevation used by the energy model

    @property
    def n(self):
        return len(self.osm_id)


def extract(pbf: Path, bbox: tuple[float, float, float, float], max_class: int = len(CLASSES) - 1) -> RoadNetwork:
    """Read drivable ways (road class index <= max_class) that touch bbox=(min_lat, min_lon, max_lat, max_lon)."""
    import osmium

    mnla, mnlo, mxla, mxlo = bbox
    fp = osmium.FileProcessor(str(pbf)).with_locations().with_filter(osmium.filter.KeyFilter("highway"))
    ctrl_nodes: dict[int, int] = {}
    seg_u, seg_v, seg_len, seg_kph, seg_cls, seg_span = [], [], [], [], [], []
    coords: dict[int, tuple[float, float]] = {}
    for o in fp:
        if o.is_node():
            hw = o.tags["highway"]
            if hw == "traffic_signals":
                ctrl_nodes[o.id] = 1
            elif hw == "stop" and o.id not in ctrl_nodes:
                ctrl_nodes[o.id] = 2
            continue
        if not o.is_way():
            continue
        t = o.tags
        cls = CLASS_ID.get(t.get("highway"))
        if cls is None or cls > max_class:
            continue
        if (t.get("access") in ("no", "private") and t.get("motor_vehicle") != "yes") \
                or t.get("motor_vehicle") in ("no", "private") or t.get("area") == "yes":
            continue
        ids, la, lo = [], [], []
        for n in o.nodes:
            ids.append(n.ref); la.append(n.lat); lo.append(n.lon)
        if len(ids) < 2:
            continue
        la = np.asarray(la); lo = np.asarray(lo)
        if not np.any((la >= mnla) & (la <= mxla) & (lo >= mnlo) & (lo <= mxlo)):
            continue
        L = _hav(la[:-1], lo[:-1], la[1:], lo[1:])
        keep = L > 0.5
        if not keep.any():
            continue
        kph = _speed(t.get("maxspeed")) or DEFAULT_KPH[cls]
        ow = t.get("oneway")
        if ow in ("yes", "true", "1") or t.get("junction") == "roundabout" or (cls <= 1 and ow != "no"):
            fwd, bwd = True, False
        elif ow == "-1":
            fwd, bwd = False, True
        else:
            fwd, bwd = True, True
        span = (t.get("bridge", "no") not in ("no", "")) or (t.get("tunnel", "no") not in ("no", ""))
        a = np.asarray(ids[:-1])[keep]; b = np.asarray(ids[1:])[keep]; L = L[keep]
        coords.update(zip(ids, zip(la.tolist(), lo.tolist())))
        for fwd_dir, (x, y) in ((fwd, (a, b)), (bwd, (b, a))):
            if fwd_dir:
                seg_u.append(x); seg_v.append(y); seg_len.append(L)
                seg_kph.append(np.full(len(L), kph)); seg_cls.append(np.full(len(L), cls, np.int8))
                seg_span.append(np.full(len(L), span))
    u = np.concatenate(seg_u); v = np.concatenate(seg_v)
    ids_all = np.unique(np.concatenate([u, v]))
    ll = np.array([coords[int(i)] for i in ids_all])
    ctrl = np.array([ctrl_nodes.get(int(i), 0) for i in ids_all], np.int8)
    eu = np.searchsorted(ids_all, u).astype(np.int32)
    ev = np.searchsorted(ids_all, v).astype(np.int32)
    return RoadNetwork(
        ids_all, ll[:, 0], ll[:, 1], ctrl, eu, ev, np.concatenate(seg_len),
        np.concatenate(seg_kph), np.concatenate(seg_cls), np.concatenate(seg_span),
    )


def merge_networks(nets: list[RoadNetwork]) -> RoadNetwork:
    """Union of networks extracted from adjacent PBFs (shared OSM node ids, duplicate ways dropped)."""
    if len(nets) == 1:
        return nets[0]
    ids = np.unique(np.concatenate([n.osm_id for n in nets]))
    lat = np.zeros(len(ids)); lon = np.zeros(len(ids)); ctrl = np.zeros(len(ids), np.int8)
    cols = []
    for n in nets:
        ix = np.searchsorted(ids, n.osm_id)
        lat[ix], lon[ix] = n.lat, n.lon
        ctrl[ix] = np.maximum(ctrl[ix], n.ctrl)
        cols.append((ix[n.eu], ix[n.ev], n.length, n.kph, n.hwclass, n.span))
    u, v, L, kph, cls, span = (np.concatenate([c[i] for c in cols]) for i in range(6))
    order = np.lexsort((span, cls, kph, np.round(L * 10), v, u))
    u, v, L, kph, cls, span = u[order], v[order], L[order], kph[order], cls[order], span[order]
    first = np.ones(len(u), bool)
    first[1:] = ~((u[1:] == u[:-1]) & (v[1:] == v[:-1]) & (np.round(L[1:] * 10) == np.round(L[:-1] * 10))
                  & (kph[1:] == kph[:-1]) & (cls[1:] == cls[:-1]) & (span[1:] == span[:-1]))
    return RoadNetwork(ids, lat, lon, ctrl, u[first].astype(np.int32), v[first].astype(np.int32),
                       L[first], kph[first], cls[first].astype(np.int8), span[first])


# Population estimate by `place` kind, used only when a node has no numeric `population` tag and
# ``assume_population=True``. This is a modelling input, not a measurement: it matters where OSM names and
# geolocates places far more completely than it records their size, e.g. China has about 80,000 named
# `place=village` nodes in the Beijing-Shanghai corridor alone but a population tag on well under 1% of
# them. Ratios follow the kind hierarchy (city > town > suburb/quarter > village > hamlet); only their
# relative size for the sqrt-population sampling weight matters, not the absolute figure.
PLACE_KIND_POPULATION_FALLBACK = {
    "city": 200000, "town": 20000, "borough": 15000, "suburb": 8000, "quarter": 5000,
    "neighbourhood": 4000, "village": 1500, "hamlet": 300, "isolated_dwelling": 50, "locality": 100,
}


def load_places(pbf: Path, kinds=("city", "town", "village"), min_pop: int = 500,
                assume_population: bool = False) -> list[dict]:
    """Named places from the OSM `place` nodes of one extract.

    By default a place is kept only when it carries a numeric `population` tag (as in Europe and the US).
    With ``assume_population=True``, a place without one is kept anyway with its population estimated from
    ``PLACE_KIND_POPULATION_FALLBACK[kind]`` (flagged ``population_is_assumed: True``) instead of being
    dropped; use this where population tagging on place nodes is sparse (see above).
    """
    import osmium

    out = []
    fp = osmium.FileProcessor(str(pbf)).with_filter(osmium.filter.KeyFilter("place"))
    for o in fp:
        kind = o.tags.get("place")
        if not o.is_node() or kind not in kinds:
            continue
        assumed = False
        try:
            pop = int(str(o.tags.get("population", "")).replace(",", "").split()[0])
        except (ValueError, IndexError):
            if not assume_population:
                continue
            pop = PLACE_KIND_POPULATION_FALLBACK.get(kind, 100)
            assumed = True
        if pop >= min_pop and o.tags.get("name"):
            out.append({"name": o.tags["name"], "lat": o.location.lat, "lon": o.location.lon,
                        "population": pop, "place": kind, "population_is_assumed": assumed})
    return out


def largest_scc(net: RoadNetwork) -> RoadNetwork:
    """Keep the largest strongly connected component so every pair is reachable."""
    A = csr_matrix((np.ones(len(net.eu)), (net.eu, net.ev)), shape=(net.n, net.n))
    _, lab = connected_components(A, directed=True, connection="strong")
    keep = lab == np.argmax(np.bincount(lab))
    em = keep[net.eu] & keep[net.ev]
    remap = -np.ones(net.n, np.int64)
    remap[keep] = np.arange(keep.sum())
    return RoadNetwork(
        net.osm_id[keep], net.lat[keep], net.lon[keep], net.ctrl[keep],
        remap[net.eu[em]].astype(np.int32), remap[net.ev[em]].astype(np.int32),
        net.length[em], net.kph[em], net.hwclass[em], net.span[em],
    )


# ---------------------------------------------------------------- terrain
class TerrainMosaic:
    """Bilinear-sampled elevation from Terrarium PNG tiles."""

    def __init__(self, raw: Path, bbox, z: int = fetch_data.TERRAIN_ZOOM):
        from PIL import Image

        self.z = z
        self.x0, self.x1, self.y0, self.y1 = fetch_data.ensure_terrain(raw, bbox, z)
        nx, ny = self.x1 - self.x0 + 1, self.y1 - self.y0 + 1
        self.grid = np.zeros((ny * 256, nx * 256), np.float32)
        for x in range(self.x0, self.x1 + 1):
            for y in range(self.y0, self.y1 + 1):
                p = raw / "terrain" / str(z) / str(x) / f"{y}.png"
                a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
                self.grid[(y - self.y0) * 256:(y - self.y0 + 1) * 256,
                          (x - self.x0) * 256:(x - self.x0 + 1) * 256] = a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768

    def sample(self, lat, lon):
        n = 2.0 ** self.z
        X = ((np.asarray(lon) + 180.0) / 360.0 * n - self.x0) * 256 - 0.5
        Y = ((1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n - self.y0) * 256 - 0.5
        X = np.clip(X, 0, self.grid.shape[1] - 1.001); Y = np.clip(Y, 0, self.grid.shape[0] - 1.001)
        x0 = X.astype(int); y0 = Y.astype(int); fx = X - x0; fy = Y - y0
        g = self.grid
        return (g[y0, x0] * (1 - fx) * (1 - fy) + g[y0, x0 + 1] * fx * (1 - fy)
                + g[y0 + 1, x0] * (1 - fx) * fy + g[y0 + 1, x0 + 1] * fx * fy)


def attach_elevation(net: RoadNetwork, raw: Path, smooth_length_m: float = 60.0,
                     zoom: int = fetch_data.TERRAIN_ZOOM, min_elevation_m: float = -15.0) -> dict:
    """Sample terrain at every node and regularise it along the road graph.

    Minimises  sum_v mu_v (h_v - h_dem_v)^2 + l^2 * sum_e (h_u - h_v)^2 / L_e  (l = smoothing length, mu_v = road
    length the node represents) so that DEM noise
    does not turn flat streets into 5 % grades. Nodes whose edges are all bridges/tunnels get
    mu ~ 0 and are interpolated between their land-side ends (a bare-earth DEM sees the valley
    under a bridge, not the deck). The telescoping gravity term makes trip energy depend on the
    end points' elevation, so this only changes how it splits between uphill and downhill legs.
    """
    mos = TerrainMosaic(raw, (net.lat.min(), net.lon.min(), net.lat.max(), net.lon.max()), zoom)
    h_dem = mos.sample(net.lat, net.lon).astype(np.float64)
    # offshore bathymetry in the tiles (down to -4800 m) is not road elevation: coastal nodes that
    # fall on water pixels would create absurd grades. Roads are above -15 m in the study regions.
    h_dem = np.where(h_dem < min_elevation_m, 0.0, h_dem)
    # nodes where every incident edge is a bridge/tunnel are free
    non_span = np.zeros(net.n, bool)
    np.logical_or.at(non_span, net.eu[~net.span], True)
    np.logical_or.at(non_span, net.ev[~net.span], True)
    a = np.minimum(net.eu, net.ev).astype(np.int64); b = np.maximum(net.eu, net.ev).astype(np.int64)
    key = a * net.n + b
    order = np.argsort(key, kind="stable")
    uk, first = np.unique(key[order], return_index=True)
    L = np.minimum.reduceat(net.length[order], first)
    ua, ub = uk // net.n, uk % net.n
    # data weight = road length each node represents, so the smoothing scale is in metres
    mu = np.zeros(net.n)
    np.add.at(mu, ua, L / 2); np.add.at(mu, ub, L / 2)
    mu = np.where(non_span, mu, 1e-6)
    w = smooth_length_m ** 2 / np.maximum(L, 5.0)
    W = coo_matrix((np.r_[w, w], (np.r_[ua, ub], np.r_[ub, ua])), shape=(net.n, net.n)).tocsr()
    A = diags(mu + np.asarray(W.sum(1)).ravel()) - W
    rhs = mu * h_dem
    x, info = cg(A, rhs, x0=h_dem, rtol=1e-8, maxiter=4000, M=diags(1.0 / A.diagonal()))
    if info != 0:
        raise RuntimeError(f"elevation smoothing did not converge (cg info={info})")
    net.elev_dem, net.elev = h_dem, x
    return {
        "terrain_zoom": mos.z,
        "dem_rms_change_m": float(np.sqrt(np.mean((x - h_dem)[non_span] ** 2))),
        "n_span_nodes_interpolated": int((~non_span).sum()),
        "elevation_range_m": [float(x.min()), float(x.max())],
        "dem_nodes_reset_to_sea_level": int((mos.sample(net.lat, net.lon) < min_elevation_m).sum()),
        "mosaic": mos,
    }


# --------------------------------------------------------------- energy / time
def edge_costs(net: RoadNetwork, vehicle: dict | None = None):
    """Per-edge travel time [h], battery energy [kWh, may be negative], horizontal length [m]."""
    p = {**DEFAULT_VEHICLE, **(vehicle or {})}
    m, g = p["mass_kg"], G0
    L = net.length
    dh = net.elev[net.ev] - net.elev[net.eu]
    Ls = np.sqrt(L * L + dh * dh)
    kph = np.minimum(net.kph, p["max_speed_kph"]) if p.get("max_speed_kph") else net.kph  # speed-limited trucks
    v = kph * SPEED_FACTOR[net.hwclass] / 3.6                          # m/s
    ctrl = net.ctrl[net.ev]
    delay = np.where(ctrl == 1, p["signal_delay_s"], np.where(ctrl == 2, p["stop_delay_s"], 0.0))
    pstop = np.where(ctrl == 1, p["signal_stop_prob"], np.where(ctrl == 2, p["stop_stop_prob"], 0.0))
    t_s = Ls / v + delay
    resist = (p["crr"] * m * g * (L / Ls) + 0.5 * p["air_density"] * p["cda_m2"] * v * v) * Ls  # >= 0
    wheel = m * g * dh + resist                                         # J at the wheels
    batt = np.where(wheel >= 0, wheel / p["eta_drive"], wheel * p["eta_regen"])
    batt += p["aux_kw"] * 1000.0 * t_s
    batt += pstop * 0.5 * m * v * v * (1.0 / p["eta_drive"] - p["eta_regen"])
    return t_s / 3600.0, batt / 3.6e6, L


def potential_slope(vehicle: dict | None = None) -> float:
    """kWh per metre of climb; c_e - kappa*(h_v-h_u) >= 0 on every edge (no negative cycles)."""
    p = {**DEFAULT_VEHICLE, **(vehicle or {})}
    return p["eta_regen"] * p["mass_kg"] * G0 / 3.6e6
