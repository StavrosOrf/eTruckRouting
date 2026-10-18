"""Execute the certified branch-and-price plan in the (stochastic) environment.

The plan is solved once per scenario on the nominal problem, or loaded from a
certificate written offline by ``scripts/evaluation/solve_branch_and_price.py``.
Each decision takes the truck's next planned step when the hard mask allows
it. Under stochastic legs a planned step can become infeasible (less energy
left than planned, a window lost to delay); the truck then advances toward
its next planned customer through the same energy-safe navigation the greedy
heuristic and the CP-SAT baseline use, and the policy counts the fallback so
that "executed the optimal plan" stays an auditable claim.
"""

from __future__ import annotations

import json
from pathlib import Path

from EVRoutingEnv.baselines.heuristics.canonical_baselines import (
    ActionKind,
    CandidateAction,
    GreedyHeuristicPolicy,
    HeuristicParameters,
    decode_feasible_actions,
)
from EVRoutingEnv.baselines.optimization.branch_and_price.julia_bridge import (
    BPParameters,
    BPSolution,
    solution_from_certificate,
    solve_bp_plan,
)
from EVRoutingEnv.models.core.customer import TaskStatus


def certificate_path(directory: str | Path, env) -> Path:
    """Where the offline driver stores the certificate of ``env``'s scenario."""
    return Path(directory) / f"seed_{env.scenario_seed}.json"


class BranchAndPricePolicy:
    name = "bp_plan"

    def __init__(self, parameters: BPParameters | None = None, *, certificate_dir: str | Path | None = None,
                 heuristic_parameters: HeuristicParameters | None = None):
        self.parameters = parameters or BPParameters()
        self.certificate_dir = None if certificate_dir is None else Path(certificate_dir)
        self.navigator = GreedyHeuristicPolicy(
            heuristic_parameters or HeuristicParameters(target_soc=1.0, demand_weight=2.0)
        )
        self.last_plan: BPSolution | None = None
        self._key: object = None
        self._routes: dict[int, list[tuple[int, float, bool]]] = {}
        self._position: dict[int, int] = {}
        self._decisions = 0
        self._fallbacks = 0

    def reset(self) -> None:
        self._key = None

    def _load_or_solve(self, env) -> BPSolution:
        if self.certificate_dir is not None:
            path = certificate_path(self.certificate_dir, env)
            if path.exists():
                certificate = json.loads(path.read_text())
                return solution_from_certificate(certificate, certificate.get("wall_seconds", 0.0))
        return solve_bp_plan(env, self.parameters)

    def _ensure_plan(self, env) -> None:
        key = (id(env), env.scenario_seed)
        if self._key == key:
            return
        plan = self._load_or_solve(env)
        self.last_plan = plan
        self._key = key
        self._routes = {tid: list(route) for tid, route in enumerate(plan.routes)}
        self._position = {tid: 0 for tid in range(env.num_trucks)}
        self._decisions = 0
        self._fallbacks = 0

    def diagnostics(self) -> dict:
        plan = self.last_plan
        if plan is None:
            return {"solver": "branch_and_price", "status": "NOT_RUN"}
        certificate = plan.certificate
        return {
            "solver": "branch_and_price",
            "status": plan.status,
            "proven_optimal": plan.proven_optimal,
            "objective_hours": plan.objective_hours,
            "best_bound_hours": plan.best_bound_hours,
            "relative_gap": plan.gap,
            "solver_wall_seconds": certificate.get("wall_seconds"),
            "phases": certificate.get("phases"),
            "nodes_explored": certificate.get("nodes_explored"),
            "decisions": self._decisions,
            "plan_fallbacks": self._fallbacks,
        }

    # -- execution -----------------------------------------------------

    @staticmethod
    def _action_for(env, step) -> int:
        node, value, charging = step
        if charging:
            return env.num_navigation_actions + list(env.charge_action_values).index(value)
        if node == env.joint_instance.depot_node:
            return env.num_charging_nodes + env.fixed_num_stops
        if node in env.charging_nodes:
            return list(env.charging_nodes).index(node)
        return env.num_charging_nodes + env.task_registry.task_for_node(node).task_id

    def _next_customer(self, env, tid: int) -> int | None:
        for node, _, charging in self._routes.get(tid, [])[self._position[tid]:]:
            if charging or node in env.charging_nodes or node == env.joint_instance.depot_node:
                continue
            task = env.task_registry.task_for_node(node)
            if task.status in (TaskStatus.UNASSIGNED, TaskStatus.CLAIMED):
                return int(node)
        return None

    def _resync(self, env, tid: int, truck) -> None:
        """After a detour, continue the plan right after the truck's last served stop."""
        route = self._routes.get(tid, [])
        here = int(truck.current_node)
        if here in env.charging_nodes or here == env.joint_instance.depot_node:
            return  # only a served customer pins the truck's place in the plan
        for k in range(len(route) - 1, self._position[tid] - 1, -1):
            node, _, charging = route[k]
            if not charging and node == here:
                self._position[tid] = k + 1
                return

    def __call__(self, env, observation=None, info=None) -> int:
        self._ensure_plan(env)
        tid = env.active_truck_id
        truck = env.trucks[tid]
        candidates = decode_feasible_actions(env)
        feasible = {c.index for c in candidates}
        self._decisions += 1
        self._resync(env, tid, truck)
        route = self._routes.get(tid, [])
        while self._position[tid] < len(route):
            step = route[self._position[tid]]
            node, value, charging = step
            if charging and truck.current_battery >= value * truck.battery_capacity - 1e-9:
                self._position[tid] += 1  # already at or above the planned level
                continue
            action = self._action_for(env, step)
            if action in feasible:
                self._position[tid] += 1
                return action
            break
        if not route and self._position[tid] == 0:
            # An idle truck in the plan retires at once.
            depot = self._action_for(env, (env.joint_instance.depot_node, 0.0, False))
            if depot in feasible:
                return depot
        self._fallbacks += 1
        by_kind: dict[ActionKind, list[CandidateAction]] = {}
        for candidate in candidates:
            by_kind.setdefault(candidate.kind, []).append(candidate)
        goal = self._next_customer(env, tid)
        if goal is None:
            goal = self.navigator._select_goal(env, truck, by_kind.get(ActionKind.CUSTOMER, []),
                                               by_kind.get(ActionKind.DEPOT, []))
        return self.navigator.navigate_toward(env, truck, goal, candidates, by_kind)
