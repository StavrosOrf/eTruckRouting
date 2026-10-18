"""Branching rules:
- Primary: Customer-Successor Branching (CBR)
- Fallback: Arc Branching (ABR) on physical arcs (immediate i->j)

This module exposes:
- select_branching_variable(solution, instance, tolerance=1e-6) -> (u, v) | None
  (prefers CBR; falls back to ABR)
- create_branches(branch_var, parent_constraints, instance) -> [branch0, branch1]

Constraints dictionary supports:
- 'forbidden_arcs': set[tuple[int,int]]              # for ABR and safety
- 'required_next_customer': dict[int,int]            # CBR: enforce next customer after u is v
- 'forbidden_next_customer': dict[int,set[int]]      # CBR: forbid that the next customer after u is some set
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any


def _compute_customer_successor_flows(solution: List[dict], instance: dict) -> Dict[Tuple[int, int], float]:
    """Flow z_uv for consecutive customers (ignoring stations) across a fractional solution."""
    flows = defaultdict(float)
    customer_ids = instance["customer_ids"]
    for item in solution:
        val = float(item.get('value', 0.0))
        if val <= 0.0:
            continue
        path = item.get('route', {}).get('path', [])
        # extract customers only, in visiting order
        seq = [nid for nid in path if nid in customer_ids]
        for i in range(len(seq) - 1):
            flows[(seq[i], seq[i + 1])] += val
    return flows


def _compute_arc_flows(solution: List[dict]) -> Dict[Tuple[int, int], float]:
    """Flow on physical arcs (i->j) as they appear in the paths (includes stations)."""
    flows = defaultdict(float)
    for item in solution:
        val = float(item.get('value', 0.0))
        if val <= 0.0:
            continue
        path = item.get('route', {}).get('path', [])
        for i in range(len(path) - 1):
            flows[(path[i], path[i + 1])] += val
    return flows


def _arg_most_fractional(flows: Dict[Tuple[int, int], float], tol: float) -> Tuple[int, int] | None:
    """Pick (u,v) with flow in (tol, 1-tol) and minimal |flow-0.5|; tie-break lexicographically."""
    best = None
    best_gap = float("inf")
    for (u, v), f in flows.items():
        if tol < f < 1.0 - tol:
            gap = abs(f - 0.5)
            if gap < best_gap or (abs(gap - best_gap) <= 1e-12 and (u, v) < (best if best else (u, v))):
                best = (u, v)
                best_gap = gap
    return best


def select_branching_variable(solution: List[dict], instance: dict, tolerance: float = 1e-6):
    """
    Prefer CBR (customer->customer consecutive flow), fallback to ABR (arc flow).
    Returns (u,v) or None. The type is inferred later (both customers -> CBR, else ABR).
    """
    if not solution:
        return None

    # 1) CBR
    cs_flows = _compute_customer_successor_flows(solution, instance)
    cs_pick = _arg_most_fractional(cs_flows, tolerance)
    if cs_pick is not None:
        u, v = cs_pick
        print(f"  [CBR] Branching on consecutive customers ({u}, {v}) with flow {cs_flows[(u,v)]:.3f}")
        return (u, v)

    # 2) ABR
    arc_flows = _compute_arc_flows(solution)
    abr_pick = _arg_most_fractional(arc_flows, tolerance)
    if abr_pick is not None:
        u, v = abr_pick
        print(f"  [ABR] Branching on arc ({u}, {v}) with flow {arc_flows[(u,v)]:.3f}")
        return (u, v)

    # 3) nothing fractional
    return None


def _clone_constraints(parent: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-ish clone for known keys."""
    child = {}
    for k, v in (parent or {}).items():
        if k == 'forbidden_arcs':
            child[k] = set(v)
        elif k == 'required_next_customer':
            child[k] = dict(v)
        elif k == 'forbidden_next_customer':
            child[k] = {kk: set(vv) for kk, vv in v.items()}
        else:
            # generic shallow copy
            try:
                child[k] = v.copy()  # type: ignore
            except Exception:
                child[k] = v
    # ensure keys exist
    child.setdefault('forbidden_arcs', set())
    child.setdefault('required_next_customer', {})
    child.setdefault('forbidden_next_customer', {})
    return child


def create_branches(branch_var: tuple, parent_constraints: dict, instance: dict) -> list:
    """
    Create two branches for branch_var (u,v):

    If both u and v are CUSTOMERS  -> CBR:
      - branch_0 (left):  forbid that the next CUSTOMER after u is v      (forbidden_next_customer[u] ⊇ {v})
                          (safety) also forbid physical arc (u,v)
      - branch_1 (right): require that the next CUSTOMER after u is v     (required_next_customer[u] = v)
                          (optional) forbid_next_customer[u] all others

    Else -> ABR (physical arc branching):
      - branch_0: forbid physical arc (u,v)
      - branch_1: require arc (u,v) immediately  => forbid all other successors of u
                   and all other predecessors of v (applies to ALL nodes)
    """
    u, v = branch_var
    customer_ids = set(instance["customer_ids"])
    num_nodes = int(instance["num_nodes"])

    left = _clone_constraints(parent_constraints)
    right = _clone_constraints(parent_constraints)

    if u in customer_ids and v in customer_ids:
        # =============================== CBR (customer-successor) ===================================
        # LEFT: forbid u -> v as next customer
        left['forbidden_next_customer'].setdefault(u, set()).add(v)
        # safety: also forbid direct arc (not necessary for CBR logic, but harmless)
        left['forbidden_arcs'].add((u, v))

        # RIGHT: require that next customer after u is v
        # set requirement
        if u in right['required_next_customer'] and right['required_next_customer'][u] != v:
            # infeasible branching (conflict): let the master handle potential infeasibility
            pass
        right['required_next_customer'][u] = v

        # optional/robustness: forbid all other next-customers different from v
        others = (customer_ids - {v})
        if others:
            right['forbidden_next_customer'].setdefault(u, set()).update(others)

        return [left, right]

    else:
        # =============================== ABR (physical arc) ===================================
        # LEFT: forbid arc (u, v)
        left['forbidden_arcs'].add((u, v))

        # RIGHT: require that u->v is taken immediately
        # Emulate "require" by forbidding all other out-arcs from u and all other in-arcs to v
        # This enforces immediate adjacency in the compact graph (no stations allowed between u and v).
        for k in range(num_nodes):
            if k != v:
                right['forbidden_arcs'].add((u, k))
            if k != u:
                right['forbidden_arcs'].add((k, v))

        return [left, right]
