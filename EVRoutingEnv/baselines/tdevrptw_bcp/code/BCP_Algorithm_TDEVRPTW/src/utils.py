"""
Utilities: JSON parser, PWL helpers, time-dependent travel & consumption core,
           beta_min & MBR preprocessing.

Goals:
- Consistency with the TD-EVRPTW paper (branch-cut-and-price).
- Compatibility with existing modules (bcp_solver, labeling, pricing, notebook).
"""

from __future__ import annotations
import json
import math
import heapq
from typing import List, Tuple, Iterable, Dict, Any

import numpy as np

EPS = 1e-9


# =============================== PWL ===================================

class PiecewiseLinearFunction:
    """
    Simple PWL:
      - keeps (x, y) sorted with unique x
      - evaluate(x): linear interpolation, clamped at the domain boundaries
      - evaluate_inverse(y): inverse over monotone segments, clamped at boundaries; for flat segments, returns the left endpoint
    """
    def __init__(self, points: List[Tuple[float, float]]):
        uniq: Dict[float, float] = {}
        for x, y in points:
            uniq[float(x)] = float(y)
        xs = sorted(uniq.keys())
        self.points: List[Tuple[float, float]] = [(x, uniq[x]) for x in xs]

    def evaluate(self, x: float) -> float:
        pts = self.points
        if not pts:
            return float("nan")
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        # locate segment with binary search
        lo, hi = 0, len(pts) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if pts[mid][0] <= x:
                lo = mid
            else:
                hi = mid
        x1, y1 = pts[lo]; x2, y2 = pts[hi]
        if abs(x2 - x1) <= EPS:
            return y1  # vertical/degenerate segment: return left endpoint
        t = (x - x1) / (x2 - x1)
        return y1 + t * (y2 - y1)

    def evaluate_inverse(self, y: float) -> float:
        """
        Return x such that f(x) = y. If y is outside the range, clamp to the closest boundary.
        If the segment is flat (y1 == y2), return the left endpoint of the segment containing y.
        """
        pts = self.points
        if not pts:
            return float("nan")
        ys = [p[1] for p in pts]
        ymin, ymax = min(ys), max(ys)
        if y <= ymin + EPS:
            # first x attaining the minimum value
            for i in range(len(pts)-1):
                if pts[i][1] <= ymin + EPS <= pts[i+1][1] + EPS or pts[i][1] >= ymin - EPS >= pts[i+1][1] - EPS:
                    return pts[i][0]
            return pts[0][0]
        if y >= ymax - EPS:
            for i in range(len(pts)-1, 0, -1):
                if pts[i-1][1] <= ymax + EPS <= pts[i][1] + EPS or pts[i-1][1] >= ymax - EPS >= pts[i][1] - EPS:
                    return pts[i][0]
            return pts[-1][0]

        # scan for the segment containing y (handles both non-decreasing and non-increasing)
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]; x2, y2 = pts[i + 1]
            if (y1 <= y <= y2) or (y2 <= y <= y1):
                if abs(y2 - y1) <= EPS:
                    return x1  # flat segment
                # linear inverse on the segment
                t = (y - y1) / (y2 - y1)
                return x1 + t * (x2 - x1)
        # if not found due to numerical issues, clamp to the right boundary
        return pts[-1][0]

    def domain(self) -> Tuple[float, float]:
        if not self.points:
            return (float("nan"), float("nan"))
        return self.points[0][0], self.points[-1][0]


# ============================ Instance loading ================================

def _calculate_distance_matrix(nodes: List[dict]) -> np.ndarray:
    coords = np.array([[n["coords"]["x"], n["coords"]["y"]] for n in nodes], dtype=float)
    dif = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(dif, axis=2)


def _ensure_defaults(instance: dict) -> None:
    """
    Insert minimal defaults to keep consistency with the solver/notebook when missing.
    (Does not override values already present in the JSON.)
    """
    params = instance.setdefault("parameters", {})
    # planning horizon
    if "planning_horizon_T" not in params:
        params["planning_horizon_T"] = 24.0 * 60.0  # minutes
    # physical energy consumption model (if not declared, keep linear fallback)
    params.setdefault("energy_consumption_rate", 1.0)  # linear fallback (kWh/km, arbitrary units)
    # if available, these activate the physical model
    params.setdefault("base_speed", 25.0)  # km/h (units consistent with coordinates and times)
    params.setdefault("consumption_h1", 1.54)
    params.setdefault("consumption_h2", 52.97)
    params.setdefault("consumption_normalizing_factor", 1015.0)

    # minimal TD definitions
    td = instance.setdefault("time_dependent_definitions", {})
    sp = td.setdefault("speed_profiles", {})
    sp.setdefault("time_intervals_factors", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sp.setdefault("profiles", {"normal": [1.0] * (len(sp["time_intervals_factors"]) - 1)})

    # minimal TD assignments
    tda = instance.setdefault("time_dependent_assignments", {})
    # station recharge modes: optional
    tda.setdefault("station_recharge_modes", {})

    # optional waiting function
    td.setdefault("waiting_function", {"pattern": "none", "points": []})


def load_instance(file_path: str) -> dict:
    with open(file_path, "r") as f:
        instance = json.load(f)

    _ensure_defaults(instance)

    nodes = instance["nodes"]
    instance["num_nodes"] = len(nodes)
    instance["nodes_by_id"] = {n["id"]: n for n in nodes}
    instance["depot_id"] = next(n["id"] for n in nodes if n["type"] == 2)
    instance["station_ids"] = {n["id"] for n in nodes if n["type"] == 1}
    instance["customer_ids"] = {n["id"] for n in nodes if n["type"] == 0}

    # distance matrix
    instance["distance_matrix"] = _calculate_distance_matrix(nodes)

    # precompute absolute breakpoints tb = factors * T
    T = float(instance["parameters"]["planning_horizon_T"])
    sp = instance["time_dependent_definitions"]["speed_profiles"]
    factors = np.array(sp["time_intervals_factors"], dtype=float)
    if factors.ndim != 1 or factors.size < 2:
        raise ValueError("speed_profiles.time_intervals_factors must have at least 2 points.")
    tb = factors * T
    instance["_abs_time_breakpoints"] = tb  # for fast reuse

    # validate profile lengths (must be len(tb)-1)
    for name, prof in sp.get("profiles", {}).items():
        if len(prof) != len(tb) - 1:
            raise ValueError(
                f"Profile '{name}' has {len(prof)} intervals, expected {len(tb)-1} "
                f"(len(time_intervals_factors)={len(tb)})."
            )

    return instance


# ================== Time-dependent travel & consumption =======================

def _calc_physical_segment_consumption(v: float, dt: float, h1: float, h2: float) -> float:
    """
    Energy consumption on a segment of duration dt (in hours) at speed v (km/h):
    E = (h1 * v^3 + h2 * v) * dt  -> then optionally normalized elsewhere.
    """
    return (h1 * (v ** 3) + h2 * v) * dt


def _travel_td_core(
    departure_time: float,
    total_distance: float,
    tb: np.ndarray,
    multipliers: np.ndarray,
    base_speed: float,
    h1: float,
    h2: float,
    normalize_factor: float | None = None,
) -> Tuple[float, float]:
    """
    Advance through time intervals [tb[k], tb[k+1]) using the fixed multiplier
    on each interval (as in the paper). Returns (travel_time, energy_consumption).

    Note: times are in MINUTES if tb is in minutes; speeds are in km/h ⇒ we convert dt to HOURS.
    """
    if total_distance <= 1e-12:
        return 0.0, 0.0

    cur_t = float(departure_time)
    dist_left = float(total_distance)
    travel_time = 0.0
    energy = 0.0

    K = len(tb) - 1
    if K <= 0:
        return float("inf"), float("inf")

    # helper to locate the interval of cur_t
    # idx = max k s.t. tb[k] <= cur_t
    idx = int(np.searchsorted(tb, cur_t, side="right") - 1)
    idx = max(0, min(idx, K - 1))

    safety_counter = 0
    while dist_left > 1e-9:
        safety_counter += 1
        if safety_counter > 20000:
            # protection against pathological loops
            return float("inf"), float("inf")

        mult = float(multipliers[idx])
        if mult <= EPS:
            return float("inf"), float("inf")  # zero speed -> arc not traversable in this setup

        v = base_speed * mult  # km/h
        # end of the current interval in "minutes"
        t_end = tb[idx + 1] if idx + 1 < len(tb) else float("inf")
        dt_min = t_end - cur_t
        if dt_min < 0:
            dt_min = 0.0

        # distance that can be covered within the current interval
        # note: v is km/h, dt is minutes ⇒ dt_hours = dt_min / 60
        dt_hours = dt_min / 60.0 if math.isfinite(dt_min) else float("inf")
        dist_cap = v * dt_hours  # km

        if dist_cap >= dist_left or not math.isfinite(dist_cap):
            # we finish the arc within this interval
            dt_needed_hours = dist_left / v
            dt_needed_min = dt_needed_hours * 60.0
            travel_time += dt_needed_min
            if h1 is None or h2 is None:
                # no physical; should not happen here
                pass
            else:
                energy += _calc_physical_segment_consumption(v, dt_needed_hours, h1, h2)
            dist_left = 0.0
            cur_t += dt_needed_min
            break
        else:
            # consume the entire interval
            travel_time += dt_min
            if h1 is not None and h2 is not None:
                energy += _calc_physical_segment_consumption(v, dt_hours, h1, h2)
            dist_left -= dist_cap
            cur_t = t_end
            if idx + 1 < K:
                idx += 1
            else:
                # beyond breakpoints: keep last multiplier
                # (technically should not happen because tb covers [0, T])
                pass

    if normalize_factor is not None and normalize_factor > 0:
        energy = energy / float(normalize_factor)
    return travel_time, energy


def get_travel_time_and_consumption(
    departure_time: float,
    total_distance: float,
    time_breakpoints: np.ndarray,
    multipliers: np.ndarray,
    params: dict,
) -> Tuple[float, float]:
    """
    Compute (travel_time, energy_consumption) for a time-dependent arc.
    - If base_speed, h1, h2 are present: use the physical model (consistent with the paper).
    - Otherwise: linear fallback (consumption = distance * energy_consumption_rate), but the time is
      still computed with multipliers (assuming fallback base_speed = 1 km/h).
    """
    base_speed = params.get("base_speed", None)
    h1 = params.get("consumption_h1", None)
    h2 = params.get("consumption_h2", None)
    norm = params.get("consumption_normalizing_factor", None)

    if base_speed is not None and h1 is not None and h2 is not None and norm is not None:
        return _travel_td_core(
            departure_time, total_distance, time_breakpoints, multipliers, base_speed, h1, h2, norm
        )
    else:
        # TD travel time with base_speed=1 km/h + linear consumption
        travel_time, _ = _travel_td_core(
            departure_time, total_distance, time_breakpoints, multipliers,
            base_speed=1.0, h1=0.0, h2=0.0, normalize_factor=None
        )
        if not math.isfinite(travel_time):
            return float("inf"), float("inf")
        energy_rate = params.get("energy_consumption_rate", 1.0)
        return travel_time, total_distance * float(energy_rate)

# ============================ beta_min & MBR ==================================

# ============================ beta_min & MBR ==================================

def _beta_min_over_arc(instance: dict, i: int, j: int) -> float:
    """
    Lower bound on β_ij(t) (consumption on arc (i,j)) by sampling:
      - all breakpoints tb
      - mid-points of each interval
    This captures interior minima within intervals, improving the bound.
    """
    if i == j:
        return 0.0

    params = instance["parameters"]
    tb = instance["_abs_time_breakpoints"]  # in minutes
    sp = instance["time_dependent_definitions"]["speed_profiles"]
    profile_name = instance["time_dependent_assignments"]["arc_speed_profiles"][i][j]
    multipliers = np.array(sp["profiles"].get(profile_name, [1.0]), dtype=float)
    dist = float(instance["distance_matrix"][i, j])

    if dist <= 1e-12:
        return 0.0

    samples: List[float] = list(tb)
    # midpoints
    mids = (tb[:-1] + tb[1:]) * 0.5
    samples.extend(mids.tolist())

    best = float("inf")
    for t0 in samples:
        tt, cons = get_travel_time_and_consumption(t0, dist, tb, multipliers, params)
        if math.isfinite(tt) and math.isfinite(cons):
            if cons < best:
                best = cons

    if not math.isfinite(best):
        # if we cannot estimate, return a high bound to avoid false positives
        return float("inf")
    return float(best)


def compute_beta_min_matrix(instance: dict) -> np.ndarray:
    n = instance["num_nodes"]
    mat = np.full((n, n), float("inf"))
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 0.0
            else:
                mat[i, j] = _beta_min_over_arc(instance, i, j)
    return mat


def compute_mbr(instance: dict) -> np.ndarray:
    """
    MBR(i): minimum energy required to depart from i and reach
    at least one STATION or the DEPOT, without depleting the battery (Rule 1 in the paper).
    Computed with Dijkstra on weights = β_min.
    """
    n = instance["num_nodes"]
    depot = instance["depot_id"]
    stations = set(instance["station_ids"])
    targets = stations.union({depot})

    if "beta_min_matrix" not in instance:
        instance["beta_min_matrix"] = compute_beta_min_matrix(instance)
    beta_min = instance["beta_min_matrix"]

    mbr = np.full(n, float("inf"))
    for s in range(n):
        # Dijkstra
        dist = np.full(n, float("inf"))
        used = np.zeros(n, dtype=bool)
        dist[s] = 0.0

        for _ in range(n):
            u = int(np.argmin(np.where(used, float("inf"), dist)))
            if used[u] or not math.isfinite(dist[u]):
                break
            used[u] = True
            if u in targets:
                mbr[s] = dist[u]
                break
            # relaxations
            du = dist[u]
            row = beta_min[u]
            for v in range(n):
                w = row[v]
                if not math.isfinite(w):
                    continue
                nd = du + w
                if nd < dist[v]:
                    dist[v] = nd

    # depot and stations have MBR=0 (terminals)
    mbr[depot] = 0.0
    if stations:
        mbr[list(stations)] = 0.0

    # any inf -> B_cap to prune useless labels
    B = float(instance["parameters"]["battery_capacity"])
    mbr[~np.isfinite(mbr)] = B
    return mbr

