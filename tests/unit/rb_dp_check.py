"""Run inside the solver interpreter: the native route DP must equal brute force on every subset.

    python tests/unit/rb_dp_check.py INSTANCE.json MAX_SUBSET_SIZE

Prints one JSON line {"subsets": n, "mismatches": [...]}. Used by
``test_routingblocks_alns.py``; not collected by pytest on its own.
"""

import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

from EVRoutingEnv.baselines.optimization.routingblocks_alns.problem import Problem

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "brute", ROOT / "EVRoutingEnv/baselines/optimization/branch_and_price/brute_force.py")
brute = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brute)

instance = json.loads(Path(sys.argv[1]).read_text())
limit = int(sys.argv[2])
problem = Problem(instance)
best = brute.best_route_per_subset(instance)
mismatches, checked = [], 0
for mask in range(1, 1 << problem.n_customers):
    members = [c + 1 for c in range(problem.n_customers) if mask >> c & 1]
    if len(members) > limit:
        continue
    checked += 1
    values = [problem.price(p) for p in itertools.permutations(members)]
    values = [v for v in values if math.isfinite(v)]
    dp = min(values) if values else math.inf
    reference = best.get(mask, math.inf)
    if math.isfinite(dp) != math.isfinite(reference) or (math.isfinite(dp) and abs(dp - reference) > 1e-6):
        mismatches.append({"customers": members, "dp": None if math.isinf(dp) else dp,
                           "brute": None if math.isinf(reference) else reference})
print(json.dumps({"subsets": checked, "mismatches": mismatches}))
