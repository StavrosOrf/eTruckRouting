"""Run inside the solver interpreter: a ``--warm-from`` plan must seed its own
candidate solution, separate from the environment's own witness.

    python tests/unit/rb_seed_sequences_check.py INSTANCE.json WARM_FROM_CERT.json

Prints one JSON line {"candidates": [{"routes": n, "cost": float}, ...]}. Used
by ``test_routingblocks_alns.py``; not collected by pytest on its own.

Regression test for a bug where ``seed_sequences`` concatenated the
environment's witness and every ``--warm-from`` plan into one combined
"solution": with two sources that each cover every customer, the merge
double-covers some customers into a single malformed seed (observed: a clean
5-route, 144 h plan merged with the environment's own 10-route witness into
one 15-route, 428 h non-solution).
"""

import json
import sys

from EVRoutingEnv.baselines.optimization.routingblocks_alns.problem import Problem
from EVRoutingEnv.baselines.optimization.routingblocks_alns.run_solver import seed_sequences

instance = json.loads(open(sys.argv[1]).read())
certificate = json.loads(open(sys.argv[2]).read())
instance.setdefault("warm_start", {})["sequences"] = [
    [stop["node"] for stop in route["stops"] if stop["kind"] == "customer"]
    for route in certificate["routes"]
]
problem = Problem(instance)
candidates = seed_sequences(problem)
print(json.dumps({
    "n_customers": problem.n_customers,
    "candidates": [{"routes": len(c), "customers": sum(len(seq) for seq in c),
                   "cost": sum(problem.price(seq) for seq in c)} for c in candidates],
}))
