"""Execute the certified branch-and-price plan in the (stochastic) environment.

The plan is solved once per scenario on the nominal problem, or loaded from a
certificate written offline by ``scripts/evaluation/solve_branch_and_price.py``.
Each decision takes the truck's next planned step when the hard mask allows
it and the step is safe under worst-case energy: after the leg, at the
largest realized energy the simulator can draw, the truck must still be able
to leave for some charger or the depot. The optimal nominal plan uses exactly
the mask's per-leg margin, so without this check a run of high energy draws
can strand a truck at a customer. Otherwise (unsafe or masked step, a window
lost to delay) the truck advances toward its next planned customer through the
same energy-safe navigation the greedy heuristic and the CP-SAT baseline use.
The policy counts every such fallback, so that "executed the optimal plan"
stays an auditable claim.
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
            heuristic_parameters
            or HeuristicParameters(energy_safety_factor=1.15, target_soc=1.0, demand_weight=2.0)
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

    @staticmethod
    def _worst_energy_multiplier(env) -> float:
        """Largest realized/nominal energy ratio the simulator can produce."""
        traffic = env.traffic_config
        if not (traffic["enable_traffic"] and traffic["enable_energy_uncertainty"]):
            return 1.0
        if float(traffic["std_dev_factor"]) == 0.0 and float(traffic["energy_uncertainty_factor"]) == 0.0:
            return 1.0  # no noise at all: every leg draws exactly its nominal energy
        return max(1.0, float(traffic["max_energy_multiplier"]))

    def _leg_is_safe(self, env, truck, node: int) -> bool:
        """After the leg at worst-case energy, a charger or the depot must stay reachable."""
        depot = int(env.joint_instance.depot_node)
        if node == depot or node in env.charging_nodes:
            return True  # the route ends there, or the truck can recharge
        graph = env.transport_graph
        worst = self._worst_energy_multiplier(env)
        left = float(truck.current_battery) - worst * float(graph.get_path_energy(int(truck.current_node), node))
        return any(
            worst * float(graph.get_path_energy(node, int(exit_node))) <= left + 1e-9
            for exit_node in [*env.charging_nodes, depot]
            if int(exit_node) != node
        )

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
            if action in feasible and (charging or self._leg_is_safe(env, truck, int(node))):
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
