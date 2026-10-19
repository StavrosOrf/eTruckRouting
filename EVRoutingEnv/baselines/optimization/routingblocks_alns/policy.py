"""Execute the RoutingBlocks plan in the (stochastic) environment.

Same execution layer as ``BranchAndPricePolicy`` (planned step when the mask
allows it and it is worst-case energy safe, otherwise energy-safe navigation
toward the next planned customer, every fallback counted); only the planner
differs. A plan is solved per scenario, or loaded from a certificate written
offline by ``scripts/evaluation/solve_routingblocks.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from EVRoutingEnv.baselines.heuristics.canonical_baselines import HeuristicParameters
from EVRoutingEnv.baselines.optimization.branch_and_price.julia_bridge import BPSolution, solution_from_certificate
from EVRoutingEnv.baselines.optimization.branch_and_price.policy import BranchAndPricePolicy, certificate_path
from EVRoutingEnv.baselines.optimization.routingblocks_alns.bridge import RBParameters, solve_rb_plan


class RoutingBlocksPolicy(BranchAndPricePolicy):
    name = "rb_plan"

    def __init__(self, parameters: RBParameters | None = None, *, certificate_dir: str | Path | None = None,
                 heuristic_parameters: HeuristicParameters | None = None):
        super().__init__(None, certificate_dir=certificate_dir, heuristic_parameters=heuristic_parameters)
        self.parameters = parameters or RBParameters()

    def _load_or_solve(self, env) -> BPSolution:
        if self.certificate_dir is not None:
            path = certificate_path(self.certificate_dir, env)
            if path.exists():
                certificate = json.loads(path.read_text())
                return solution_from_certificate(certificate, certificate.get("wall_seconds", 0.0))
        return solve_rb_plan(env, self.parameters)

    def diagnostics(self) -> dict:
        plan = self.last_plan
        if plan is None:
            return {"solver": "routingblocks_alns", "status": "NOT_RUN"}
        certificate = plan.certificate
        return {
            "solver": "routingblocks_alns",
            "status": plan.status,
            "proven_optimal": False,
            "objective_hours": plan.objective_hours,
            "best_bound_hours": plan.best_bound_hours,
            "relative_gap": plan.gap,
            "solver_wall_seconds": certificate.get("wall_seconds"),
            "decisions": self._decisions,
            "plan_fallbacks": self._fallbacks,
        }
