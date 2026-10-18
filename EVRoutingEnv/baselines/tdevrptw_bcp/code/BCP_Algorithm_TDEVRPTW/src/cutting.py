"""Subset Row Inequalities (SRC) separation (k=2) with small sets S (|S|=3..S_max).

Separation is heuristic and fast:
- build fractional coverages and co-occurrences among customers,
- enumerate small subsets S (3..S_max) among the most fractional customers with high co-occurrence,
- evaluate violation of: sum_r floor(|S∩r|/2) x_r <= floor(|S|/2),
- add the most violated cuts (avoiding duplicates).

Expected interface:
- rmp_model.model          -> gurobipy.Model
- rmp_model.route_vars     -> list[gp.Var] aligned with rmp_model.routes
- rmp_model.routes[i]      -> dict with at least 'customers_visited': set[int]

LP-side solution:
- 'solution' is a list of dicts {'route': rmp_model.routes[i], 'value': x_i} for positive vars.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple, Set

import gurobipy as gp
from gurobipy import GRB


def _get_or_init_cache(rmp_model) -> dict:
    """Store persistent info on the model (to avoid re-adding the same cuts)."""
    if not hasattr(rmp_model, "_cut_cache"):
        rmp_model._cut_cache = {
            "added_src_sets": set(),  # set of frozenset(customer_ids)
        }
    return rmp_model._cut_cache


def _fractional_coverage(solution: List[dict], customers: List[int]) -> Dict[int, float]:
    """coverage[i] = sum of x_r over routes r that visit i (fractional coverage)."""
    cov = {i: 0.0 for i in customers}
    for item in solution:
        val = float(item.get("value", 0.0))
        if val <= 0.0:
            continue
        visited = item["route"]["customers_visited"]
        for i in visited:
            if i in cov:
                cov[i] += val
    return cov


def _fractional_cooccurrence(solution: List[dict], customers: List[int]) -> Dict[Tuple[int, int], float]:
    """co[(i,j)] = sum of x_r over routes r that visit both i and j (i<j)."""
    co = {}
    cust_set = set(customers)
    for item in solution:
        val = float(item.get("value", 0.0))
        if val <= 0.0:
            continue
        visited = [c for c in item["route"]["customers_visited"] if c in cust_set]
        if len(visited) < 2:
            continue
        vs = sorted(visited)
        # all pairs within the route
        for a_idx in range(len(vs)):
            for b_idx in range(a_idx + 1, len(vs)):
                key = (vs[a_idx], vs[b_idx])
                co[key] = co.get(key, 0.0) + val
    return co


def _candidate_customers(coverage: Dict[int, float], top_k: int) -> List[int]:
    """
    Select “most fractional” customers: coverage closest to 0.5.
    (Very common in SRC heuristics.)
    """
    ranked = sorted(coverage.items(), key=lambda kv: abs(kv[1] - 0.5))
    return [i for i, _ in ranked[:max(3, top_k)]]


def _enumerate_subsets(customers: List[int], max_subset_size: int, cap: int) -> List[Set[int]]:
    """
    Enumerate subsets S of size 3..max_subset_size from 'customers'.
    Limit the total number to 'cap' (lexicographic order).
    """
    cand_sets: List[Set[int]] = []
    for s in range(3, max(3, max_subset_size) + 1):
        for comb in combinations(customers, s):
            cand_sets.append(set(comb))
            if len(cand_sets) >= cap:
                return cand_sets
    return cand_sets


def _src_violation_for_set(S: Set[int], solution: List[dict]) -> Tuple[float, float, float]:
    """
    Compute (LHS, RHS, violation) for SRC k=2 on a subset S.
    LHS = sum_r floor(|S∩r|/2) * x_r
    RHS = floor(|S|/2)
    """
    rhs = len(S) // 2
    lhs = 0.0
    for item in solution:
        x = float(item.get("value", 0.0))
        if x <= 1e-12:
            continue
        rS = len(S.intersection(item["route"]["customers_visited"]))
        coeff = rS // 2
        if coeff > 0:
            lhs += coeff * x
    violation = lhs - rhs
    return lhs, rhs, violation


def _add_src_cut(rmp_model, S: Set[int], rhs: int):
    """Add the SRC cut for S: sum coeff_r * x_r <= rhs."""
    cut_expr = gp.LinExpr()
    for i, var in enumerate(rmp_model.route_vars):
        rS = len(S.intersection(rmp_model.routes[i]['customers_visited']))
        coeff = rS // 2
        if coeff > 0:
            cut_expr += coeff * var
    name = f"SRC_{'_'.join(map(str, sorted(S)))}"
    rmp_model.model.addConstr(cut_expr <= rhs, name=name)


def find_and_add_violated_cuts(
    rmp_model,
    solution: List[dict],
    instance: dict,
    max_cuts: int = 10,
    max_subset_size: int = 5,
    top_k_customers: int = 12,
    subset_cap: int = 400,
    tol: float = 1e-6,
) -> int:
    """
    Search and add violated SRC k=2 with heuristic separation:
      - take customers with most fractional coverage (closest to 0.5),
      - generate subsets S with |S|=3..max_subset_size,
      - evaluate the violation of: sum floor(|S∩r|/2) x_r <= floor(|S|/2),
      - add the top 'max_cuts' most violated that are not already in the model.

    Parameters:
      max_subset_size: maximum |S| (recommended 4 or 5).
      top_k_customers: how many customers to consider when enumerating S.
      subset_cap: hard limit on the number of S tested (runtime guard).
    """
    if not solution or max_cuts <= 0:
        return 0

    customers = sorted(list(instance['customer_ids']))
    if len(customers) < 3:
        return 0

    cache = _get_or_init_cache(rmp_model)
    already = cache["added_src_sets"]

    # coverages and co-occurrences
    coverage = _fractional_coverage(solution, customers)
    co = _fractional_cooccurrence(solution, customers)

    # pick candidate customers (most fractional)
    cand_cust = _candidate_customers(coverage, top_k=top_k_customers)

    # strengthen the pool by adding neighbors with high co-occurrence
    # to focus S where the LP “mixes” the most
    extra = set()
    for (i, j), w in sorted(co.items(), key=lambda kv: kv[1], reverse=True)[:2 * top_k_customers]:
        if i in cand_cust or j in cand_cust:
            extra.add(i); extra.add(j)
    cand_cust = sorted(set(cand_cust).union(extra))

    # enumerate subsets S (3..max_subset_size) with cap
    candidate_sets = _enumerate_subsets(cand_cust, max_subset_size=max_subset_size, cap=subset_cap)
    if not candidate_sets:
        return 0

    # evaluate violations and select top ones
    violated: List[Tuple[float, Set[int], int]] = []  # (violation, S, rhs)
    for S in candidate_sets:
        S_key = frozenset(S)
        if S_key in already:
            continue
        lhs, rhs, viol = _src_violation_for_set(S, solution)
        if viol > tol:
            violated.append((viol, S, rhs))

    if not violated:
        return 0

    violated.sort(key=lambda x: x[0], reverse=True)
    to_add = violated[:max_cuts]

    # add selected cuts
    added = 0
    for viol, S, rhs in to_add:
        _add_src_cut(rmp_model, S, rhs)
        cache["added_src_sets"].add(frozenset(S))
        added += 1

    if added > 0:
        print(f"  Cuts: added {added} SRC (k=2) with |S|<= {max_subset_size}.")
        rmp_model.model.update()
    return added
