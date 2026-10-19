"""Set partitioning over every exactly priced route the search has seen.

The ALNS cache holds each route it ever priced, with its exact cost. Keeping the
cheapest ordering per customer set and solving
``min sum c_r x_r  s.t.  every customer in exactly one route, at most K routes``
recombines good routes found in different parts of the search. HiGHS solves it,
warm-started with the incumbent, under a time limit.
"""

from __future__ import annotations

import math

import highspy
import numpy as np


def best_per_set(cache: dict[tuple, float]) -> dict[frozenset, tuple[float, tuple]]:
    best: dict[frozenset, tuple[float, tuple]] = {}
    for route, cost in cache.items():
        if not route or not math.isfinite(cost):
            continue
        key = frozenset(route)
        held = best.get(key)
        if held is None or cost < held[0]:
            best[key] = (cost, route)
    return best


def solve_partition(pool: dict[frozenset, tuple[float, tuple]], customers: range, fleet: int,
                    incumbent: list[tuple], time_limit: float, max_columns: int = 400_000, threads: int = 1):
    """Return (routes, cost) of the best partition of ``pool`` found, or None.

    ``pool`` maps a customer set to (cost, cheapest known ordering).
    """
    pool = dict(pool)
    for route in incumbent:  # the incumbent is always a column, so the start is feasible
        key = frozenset(route)
        if key not in pool:
            return None
    incumbent_sets = [frozenset(r) for r in incumbent]
    entries = sorted(pool.values(), key=lambda e: e[0])
    if len(entries) > max_columns:
        keep = {s for s in incumbent_sets}
        head = entries[:max_columns]
        have = {frozenset(r) for _, r in head}
        entries = head + [pool[s] for s in keep if s not in have]
    index = {c: k for k, c in enumerate(customers)}
    n_rows = len(index)
    n_cols = len(entries)
    if n_cols == 0:
        return None
    starts, rows, costs = [0], [], []
    for cost, route in entries:
        rows.extend(sorted(index[c] for c in route))
        rows.append(n_rows)  # fleet row
        starts.append(len(rows))
        costs.append(cost)
    lp = highspy.HighsLp()
    lp.num_col_ = n_cols
    lp.num_row_ = n_rows + 1
    lp.col_cost_ = np.array(costs, dtype=float)
    lp.col_lower_ = np.zeros(n_cols)
    lp.col_upper_ = np.ones(n_cols)
    lp.row_lower_ = np.append(np.ones(n_rows), 0.0)
    lp.row_upper_ = np.append(np.ones(n_rows), float(fleet))
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.array(starts, dtype=np.int32)
    lp.a_matrix_.index_ = np.array(rows, dtype=np.int32)
    lp.a_matrix_.value_ = np.ones(len(rows))
    lp.integrality_ = [highspy.HighsVarType.kInteger] * n_cols

    model = highspy.Highs()
    model.setOptionValue("output_flag", False)
    model.setOptionValue("time_limit", float(time_limit))
    model.setOptionValue("threads", int(threads))
    model.setOptionValue("mip_rel_gap", 1e-6)
    model.passModel(lp)

    position = {frozenset(r): k for k, (_, r) in enumerate(entries)}
    start_value = np.zeros(n_cols)
    for s in incumbent_sets:
        if s in position:
            start_value[position[s]] = 1.0
    warm = highspy.HighsSolution()
    warm.col_value = list(start_value)
    model.setSolution(warm)
    model.run()

    info = model.getInfo()
    if info.primal_solution_status != highspy.SolutionStatus.kSolutionStatusFeasible:
        return None
    values = np.array(model.getSolution().col_value)
    chosen = [entries[k] for k in np.flatnonzero(values > 0.5)]
    if sum(len(r) for _, r in chosen) != n_rows:
        return None
    return [r for _, r in chosen], float(sum(c for c, _ in chosen))
