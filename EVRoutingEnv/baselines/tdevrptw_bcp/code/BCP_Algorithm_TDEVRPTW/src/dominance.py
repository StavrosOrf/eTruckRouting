"""Dominance rules (scalar + partial dominance) for labels."""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Tuple, List
import numpy as np

if TYPE_CHECKING:
    from labeling import Label  # noqa

EPS = 1e-6


def _is_subset(a: set, b: set) -> bool:
    """Return True if a ⊆ b, handling empty sets and None safely."""
    if a is None or len(a) == 0:
        return True
    if b is None:
        return False
    return a.issubset(b)


def check_dominance(
    label_M: "Label",
    label_L: "Label",
    instance: dict | None = None,
) -> bool:
    """
    Return True if label M dominates label L.

    Sufficient (safe) pruning conditions:
      1) same current node and same path anchor (last_customer_id) ⇒ same topological state.
      2) cost(M) ≤ cost(L)            (reduced cost so far)
      3) served_demand(M) ≤ served_demand(L)   (no worse load)
      4) time(M) ≤ time(L)            (no worse arrival/leave time)
      5) battery(M) ≥ battery(L)      (no worse energy resource)
      6) visited_customers(M) ⊆ visited_customers(L)  (more flexibility ahead)
      7) (optional) MBR slack: [battery(M) - MBR(node)] ≥ [battery(L) - MBR(node)]
      8) at least one strict improvement to avoid duplicates.

    Notes:
      - If `instance` contains `mbr` (array), we apply (7).
      - Requiring identical last_customer_id is conservative for sequence/TD constraints:
        it avoids risky pruning when recharge/wait policies depend on the last customer.
    """
    # 1) Matching topological state
    if label_M.current_node_id != label_L.current_node_id:
        return False
    if label_M.last_customer_id != label_L.last_customer_id:
        return False

    # 2..5) Resource/cost/time comparisons
    if label_M.cost > label_L.cost + EPS:
        return False
    if label_M.served_demand > label_L.served_demand + EPS:
        return False
    if label_M.time > label_L.time + EPS:
        return False
    if label_M.battery < label_L.battery - EPS:
        return False

    # 6) Flexibility on visited set
    if not _is_subset(label_M.visited_customers, label_L.visited_customers):
        return False

    # 7) Slack vs. MBR (if available): larger slack ⇒ safer extendability
    if instance is not None and "mbr" in instance:
        node = label_M.current_node_id
        mbr_val = float(instance["mbr"][node])
        slack_M = label_M.battery - mbr_val
        slack_L = label_L.battery - mbr_val
        if slack_M + EPS < slack_L:
            return False

    # 8) Avoid perfect duplicates (all equal within EPS)
    same_cost = abs(label_M.cost - label_L.cost) < EPS
    same_demand = abs(label_M.served_demand - label_L.served_demand) < EPS
    same_time = abs(label_M.time - label_L.time) < EPS
    same_batt = abs(label_M.battery - label_L.battery) < EPS
    same_visits = (label_M.visited_customers == label_L.visited_customers)

    if same_cost and same_demand and same_time and same_batt and same_visits:
        # Not “dominance”: it is a duplicate; let the caller handle deduping elsewhere
        return False

    return True

# =============================== PARTIAL DOMINANCE ON LAMBDA SERIES ===================================

def _normalize_timeseries(
    series: Iterable[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Convert a series [(t, v), ...] into two sorted arrays (t, v) and also return the domain [t_min, t_max].
    Deduplicate on t by keeping the last value for numerical stability.
    """
    if not series:
        return np.array([], dtype=float), np.array([], dtype=float), (np.inf, -np.inf)

    # Aggregate by t (keep last value for numerical stability)
    agg = {}
    for t, v in series:
        agg[float(t)] = float(v)
    ts = np.array(sorted(agg.keys()), dtype=float)
    vs = np.array([agg[t] for t in ts], dtype=float)
    return ts, vs, (float(ts[0]), float(ts[-1]))


def partial_dominance_timeseries(lambda_M: list, lambda_L: list) -> list:
    """
    Simple “partial dominance” over piecewise-linear resource-time functions.
    Given λ_M and λ_L as lists of points [(t, val), ...] (e.g., remaining battery / departure time),
    return the list of times T where L is DOMINATED by M, i.e., λ_L(t) ≤ λ_M(t) + EPS.

    Improvements vs the previous version:
      - unify both series' breakpoints
      - include domain endpoints
      - handle misaligned domains (constant clamped extrapolation at the ends)
      - use a consistent EPS
    """
    tM, vM, domM = _normalize_timeseries(lambda_M)
    tL, vL, domL = _normalize_timeseries(lambda_L)

    if tM.size == 0 or tL.size == 0:
        return []

    # Union of breakpoints + both domains' endpoints
    t_all = sorted(set(tM.tolist() + tL.tolist() + [domM[0], domM[1], domL[0], domL[1]]))
    t_all = np.array(t_all, dtype=float)

    # Linear interpolation inside, constant clamp outside the domain
    def interp_clamped(tq: np.ndarray, tx: np.ndarray, vx: np.ndarray) -> np.ndarray:
        if tx.size == 1:
            return np.full_like(tq, vx[0])
        # numpy.interp does linear extrapolation; avoid it by clamping edges
        v = np.interp(tq, tx, vx)
        v[tq < tx[0]] = vx[0]
        v[tq > tx[-1]] = vx[-1]
        return v

    m_vals = interp_clamped(t_all, tM, vM)
    l_vals = interp_clamped(t_all, tL, vL)

    dominated_times = [float(t) for t, lv, mv in zip(t_all, l_vals, m_vals) if lv <= mv + EPS]
    return dominated_times
