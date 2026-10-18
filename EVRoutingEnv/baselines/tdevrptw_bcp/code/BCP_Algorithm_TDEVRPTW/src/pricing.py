"""Pricing: exact labeling-based solver + simple heuristics (warm start).

- Exact pricing uses forward labeling with dominance and LB pruning.
- Heuristics provide quick negative columns (single-customer 0-c-0, small chains).

This file is compatible with:
- bcp_solver.BCPSolver (calls via pricing_worker.solve_pricing_wrapper)
- model.RMP_Model (expects route dicts with 'path', 'cost', 'customers_visited')
- labeling.Label / labeling.extend_label
- dominance.check_dominance
- utils.get_travel_time_and_consumption
"""

from __future__ import annotations

import math
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np

from labeling import Label, extend_label
from dominance import check_dominance
from utils import get_travel_time_and_consumption, PiecewiseLinearFunction


# =============================== helpers / LB ===================================

def _min_return_to_depot_lb(instance: dict) -> np.ndarray:
    """Per-node lower bound of the cost to return to the depot, using straight-line distance as a safe LB."""
    n = instance['num_nodes']
    depot = instance['depot_id']
    D = instance['distance_matrix']
    lb = D[:, depot].astype(float).copy()
    # ensure depot itself is 0
    lb[depot] = 0.0
    return lb


def _objective_cost_of_arc(instance: dict, i: int, j: int, depart_time: float) -> Tuple[float, float, float]:
    """
    Return (time_ij, energy_ij, distance_ij) for arc (i, j) departing at depart_time,
    computed with the same time-dependent functions used by the solver.
    """
    params = instance['parameters']
    T = float(params['planning_horizon_T'])
    speed_defs = instance['time_dependent_definitions']['speed_profiles']
    tb = np.array(speed_defs['time_intervals_factors'], dtype=float) * T
    profile = instance['time_dependent_assignments']['arc_speed_profiles'][i][j]
    mult = np.array(speed_defs['profiles'].get(profile, [1.0]), dtype=float)
    dist = float(instance['distance_matrix'][i, j])
    tt, cons = get_travel_time_and_consumption(depart_time, dist, tb, mult, params)
    return tt, cons, dist


def _accumulate_route_cost(instance: dict, path: List[int], depart_times: List[float]) -> float:
    """
    Compute c_r consistently with the selected objective:
    - parameters['objective'] in {'distance', 'time', 'energy'} (default: 'distance').
    Requires depart_times aligned with path[:-1].
    """
    metric = str(instance['parameters'].get('objective', 'distance')).lower()
    acc_time = 0.0
    acc_energy = 0.0
    acc_dist = 0.0
    for k in range(len(path)-1):
        i, j = path[k], path[k+1]
        tt, cons, dist = _objective_cost_of_arc(instance, i, j, depart_times[k])
        if math.isfinite(tt):
            acc_time += tt
        if math.isfinite(cons):
            acc_energy += cons
        acc_dist += dist
    if metric == 'time':
        return float(acc_time)
    elif metric == 'energy':
        return float(acc_energy)
    else:
        return float(acc_dist)


def _reconstruct_path_and_departure_times(final_label: Label, instance: dict) -> Tuple[List[int], List[float]]:
    """
    Reconstruct the node sequence and per-arc departure times from the depot.
    Assumes Label.time stores the departure time at its current node (as in extend_label).
    Returns (path, depart_times_for_each_arc).
    """
    seq = []
    times = []
    L = final_label
    while L is not None:
        seq.append(L.current_node_id)
        times.append(L.time)
        L = L.parent
    seq.reverse()
    times.reverse()
    depart_times = times[:-1]
    return seq, depart_times


def _build_route_dict(final_label: Label, instance: dict) -> dict:
    """Build the route dict required by the RMP: path, cost (c_r), customers_visited."""
    path, dep_times = _reconstruct_path_and_departure_times(final_label, instance)
    cost = _accumulate_route_cost(instance, path, dep_times)
    customers_visited: Set[int] = set(nid for nid in path if instance['nodes_by_id'][nid]['type'] == 0)
    return {'path': path, 'cost': cost, 'customers_visited': customers_visited}

# =============================== Exact pricing ===================================

def solve_pricing_problem(instance: dict, duals: dict, branching_constraints: dict,
                          max_neg_routes: int = 50, max_labels: int = 20000,
                          graph_mask: np.ndarray | None = None) -> List[dict]:
    """
    Exact pricing via forward labeling with dominance and bounding.

    Args:
        instance: Problem data (nodes, distances, TD profiles, parameters).
        duals: Dual values from the current RMP (used inside extend_label to update reduced costs).
        branching_constraints: Branch-and-bound constraints; supports {'forbidden_arcs'} as a set of (i, j).
        max_neg_routes: Maximum number of negative reduced-cost routes to return.
        max_labels: Safety cap on the total number of generated/processed labels.
        graph_mask: Optional mask of allowed arcs (currently unused).

    Returns:
        A list (up to max_neg_routes) of route dicts with keys:
            - 'path': sequence of node IDs (starting and ending at depot),
            - 'cost': route reduced cost c_r (negative if improving),
            - 'customers_visited': set of customer IDs served by the route.

    Notes:
        - Labels are expanded from the depot using a priority heap ordered by
          (reduced_cost, time, -battery, num_visited_customers, node_id).
        - Dominance is enforced per bucket keyed by (node, last_customer_id).
        - Pruning: elementarity on customers, avoid station→station chains,
          respect forbidden_arcs, and apply a simple LB using the per-node
          lower bound to return to the depot.
        - A route is finalized only when a label at the depot has visited at least one customer.
    """
    depot = instance['depot_id']
    num_nodes = instance['num_nodes']
    customer_ids: Set[int] = set(instance['customer_ids'])
    station_ids: Set[int] = set(instance['station_ids'])

    forbidden_arcs: Set[Tuple[int,int]] = set(branching_constraints.get('forbidden_arcs', set()))

    # Lower bound to prune: reduced-cost so far + min return to depot (distance LB) >= 0
    min_ret = _min_return_to_depot_lb(instance)

    # Buckets for dominance: (node, last_customer) -> list of nondominated labels
    buckets: Dict[Tuple[int,int], List[Label]] = defaultdict(list)

    # Init label at depot
    B = instance['parameters']['battery_capacity']
    root = Label(cost=0.0, time=0.0, battery=B, served_demand=0.0,
                 current_node_id=depot, num_visited_customers=0,
                 visited_customers=set(), last_customer_id=-1, parent=None)

    # key: (reduced cost, time, -battery, num_visited, node_id, label)
    open_heap: List[Tuple[float,float,float,int,int,Label]] = []
    heapq.heappush(open_heap, (root.cost, root.time, -root.battery, root.num_visited_customers, root.current_node_id, root))

    neg_routes: List[dict] = []
    labels_generated = 0

    while open_heap:
        _, _, _, _, _, lab = heapq.heappop(open_heap)

        labels_generated += 1
        if labels_generated > max_labels:
            break

        node = lab.current_node_id

        # If at depot and visited some customers, we can finalize a route
        if node == depot and lab.parent is not None and lab.num_visited_customers > 0:
            route = _build_route_dict(lab, instance)
            # Reduced cost of complete route equals lab.cost (duals subtracted at customer expansions)
            if lab.cost < -1e-9:
                neg_routes.append(route)
                if len(neg_routes) >= max_neg_routes:
                    break
            continue

        # Dominance insertion into bucket
        key = (lab.current_node_id, lab.last_customer_id)
        bucket = buckets[key]
        dominated = False
        to_keep = []
        for M in bucket:
            if check_dominance(M, lab, instance):
                dominated = True
                break
            if not check_dominance(lab, M, instance):
                to_keep.append(M)
        if dominated:
            continue
        bucket[:] = to_keep + [lab]

        # Expand to all nodes subject to constraints
        for j in range(num_nodes):
            if j == node:
                continue
            if (node, j) in forbidden_arcs:
                continue
            # elementarity: do not revisit a customer
            if j in customer_ids and j in lab.visited_customers:
                continue
            # avoid station->station chains to reduce useless columns
            if node in station_ids and j in station_ids:
                continue
            # Always allow closing to depot only if we visited at least one customer
            if j == depot and lab.num_visited_customers == 0:
                continue

            new_lab = extend_label(lab, j, instance, duals)
            if new_lab is None:
                continue

            # Simple LB: if even closing directly from j to depot cannot make cost negative, prune
            if j != depot and (new_lab.cost + float(min_ret[j]) >= -1e-9):
                continue

            heapq.heappush(open_heap, (new_lab.cost, new_lab.time, -new_lab.battery,
                                       new_lab.num_visited_customers, new_lab.current_node_id, new_lab))

    return neg_routes

# =============================== Heuristics ===================================

def _single_customer_candidates(instance: dict, duals: dict) -> List[Tuple[float, List[int]]]:
    """Build simple 0-c-0 candidates with reduced costs (distance-based RC for speed)."""
    depot = instance['depot_id']
    D = instance['distance_matrix']
    cand: List[Tuple[float, List[int]]] = []
    for c in sorted(instance['customer_ids']):
        rc = D[depot, c] + D[c, depot] - float(duals.get(c, 0.0))
        cand.append((rc, [depot, c, depot]))
    cand.sort(key=lambda x: x[0])
    return cand


def solve_pricing_heuristic(instance: dict, duals: dict, method: str, k: int,
                            branching_constraints: dict) -> List[dict]:
    """
    Very fast warm starts:
      - 'k-shrink': take up to k best 0-c-0 columns with negative reduced cost
      - 'relax-s' / 'relax-b': fallback to exact but with small limits (acts as light search)
    """
    method = (method or '').lower()
    depot = instance['depot_id']

    if method == 'k-shrink':
        routes: List[dict] = []
        # Precompute TD departure times for each 0-c-0 candidate
        for rc, path in _single_customer_candidates(instance, duals)[:max(1, k)]:
            if rc < -1e-9:
                c = path[1]
                # arc depot->c
                t_dep0 = 0.0
                tt01, _, _ = _objective_cost_of_arc(instance, depot, c, t_dep0)
                if not math.isfinite(tt01):
                    continue
                arr_c = t_dep0 + tt01
                # passive wait to respect customer's TW
                tw = instance['nodes_by_id'][c]['time_window']
                eff_arr_c = max(arr_c, tw['start'])
                # service at customer
                service = instance['nodes_by_id'][c].get('service_time', 0.0)
                t_dep_c = eff_arr_c + service
                depart_times = [t_dep0, t_dep_c]
                cost = _accumulate_route_cost(instance, path, depart_times[:len(path)-1])
                routes.append({'path': path, 'cost': cost,
                               'customers_visited': set([c])})
        return routes

    elif method in ('relax-s', 'relax-b'):
        # To keep interface intact, we simply run a shallow exact search with tight caps
        # (acts as a bounded heuristic search).
        return solve_pricing_problem(instance, duals, branching_constraints,
                                     max_neg_routes=10, max_labels=3000)

    return []