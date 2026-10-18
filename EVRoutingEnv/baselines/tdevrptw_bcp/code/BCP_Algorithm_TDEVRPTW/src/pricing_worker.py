"""
Worker wrapper for multiprocessing: unpacks arguments and calls pricing functions.
This module must import numpy because child processes need it.
"""

import numpy as np
import copy

def solve_pricing_wrapper(args):
    from pricing import solve_pricing_problem, solve_pricing_heuristic

    (
        instance,
        duals,
        method,
        k,
        branching_constraints,
        is_heuristic
    ) = args

    # Avoid shared mutations across processes
    local_instance = copy.deepcopy(instance)
    local_duals = dict(duals or {})
    local_branch = copy.deepcopy(branching_constraints or {})

    try:
        if is_heuristic:
            return solve_pricing_heuristic(local_instance, local_duals, method, k, local_branch)
        else:
            # Use pricing’s internal defaults; caps can be added via args if needed
            return solve_pricing_problem(local_instance, local_duals, local_branch, graph_mask=None)
    except Exception as e:
        # Do not propagate worker errors: the master will handle missing columns
        print(f"[pricing_worker] Exception in worker: {e}")
        return []
