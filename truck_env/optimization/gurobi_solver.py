"""Gurobi-based solver for the truck routing problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:  # pragma: no cover
    from truck_env.models.event_driven_env import EventDrivenTruckEnv


try:  # pragma: no cover - optional dependency
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    gp = None
    GRB = None
    _GUROBI_IMPORT_ERROR = exc
else:  # pragma: no cover - ensures variable defined
    _GUROBI_IMPORT_ERROR = None
HAS_GUROBI = gp is not None and GRB is not None


@dataclass
class TruckRouteSolution:
    """Container for an optimal routing solution."""

    total_time: float
    truck_routes: Dict[int, List[int]]
    truck_times: Dict[int, float]


class GurobiTruckRoutingSolver:
    """
    Solve the deterministic truck routing problem optimally with Gurobi.

    The solver first instantiates the event-driven environment to sample
    truck start locations and their delivery sequences. It then builds an
    optimization model that allows each truck to reorder its assigned
    deliveries to minimize the total completion time across all trucks.
    """

    def __init__(
        self,
        config_path: str,
        seed: int = 0,
        env: Optional["EventDrivenTruckEnv"] = None,
        auto_reset: bool = True,
    ):
        self.config_path = config_path
        self.seed = seed

        if gp is None or GRB is None:
            raise ImportError(
                "gurobipy is required to use GurobiTruckRoutingSolver"
            ) from _GUROBI_IMPORT_ERROR

        if env is None:
            from truck_env.models.event_driven_env import EventDrivenTruckEnv

            self.env = EventDrivenTruckEnv(
                config=config_path, verbose=False, enable_plotting=False
            )
        else:
            self.env = env

        if auto_reset and hasattr(self.env, "reset"):
            self.env.reset(seed=seed)

        if not hasattr(self.env, "trucks"):
            raise ValueError("Environment must expose a `trucks` attribute.")

        if not hasattr(self.env, "transport_graph"):
            raise ValueError("Environment must expose a `transport_graph` attribute.")

        self._graph: nx.DiGraph = self.env.transport_graph.graph
        self._model: Optional[gp.Model] = None
        self._route_vars: Dict[int, Dict[Tuple[int, int], gp.Var]] = {}
        self._order_vars: Dict[int, Dict[int, gp.Var]] = {}
        self._truck_info: Dict[int, Dict[str, object]] = {}
        self._arc_time: Dict[Tuple[int, int, int], float] = {}
        self._planned_routes: Dict[int, List[int]] = {}
        self._route_progress: Dict[int, int] = {}
        self._plan_ready: bool = False
        self._tracked_truck_ids: Optional[Tuple[int, ...]] = None
        self._last_solution: Optional[TruckRouteSolution] = None
        self._policy_env: Optional["EventDrivenTruckEnv"] = env
        self._charge_buffer = 0.05  # 5% buffer

    def _extract_truck_data(self) -> List[Dict[str, object]]:
        """Collect start and delivery nodes for each truck."""
        truck_data: List[Dict[str, object]] = []
        for truck in self.env.trucks:
            sequence = [int(node) for node in truck.delivery_sequence]
            if not sequence:
                raise ValueError(f"Truck {truck.truck_id} has empty delivery sequence.")
            truck_data.append(
                {
                    "truck_id": int(truck.truck_id),
                    "start": sequence[0],
                    "deliveries": sequence[1:],
                }
            )
        return truck_data

    def _compute_time_matrix(self, nodes: List[int]) -> Dict[int, Dict[int, float]]:
        """Compute travel times between all node pairs."""
        matrix: Dict[int, Dict[int, float]] = {}
        for src in nodes:
            lengths = nx.single_source_dijkstra_path_length(
                self._graph, src, weight="time"
            )
            matrix[src] = {}
            for dst in nodes:
                if src == dst:
                    continue
                matrix[src][dst] = lengths.get(dst, float("inf"))
        return matrix

    def _add_truck_constraints(
        self, model: gp.Model, truck: Dict[str, object]
    ) -> gp.LinExpr:
        """Build routing constraints for a single truck and return its time expr."""
        truck_id = truck["truck_id"]
        start_node = truck["start"]
        deliveries: List[int] = list(truck["deliveries"])

        real_nodes = [start_node] + deliveries
        time_matrix = self._compute_time_matrix(real_nodes)

        end_node = f"end_{truck_id}"
        arc_vars: Dict[Tuple[int, int], gp.Var] = {}

        # Build route variables between real nodes
        for src in real_nodes:
            for dst in real_nodes:
                if src == dst or dst == start_node:
                    continue
                travel_time = time_matrix.get(src, {}).get(dst, float("inf"))
                if not float(travel_time) < float("inf"):
                    continue
                arc_vars[(src, dst)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"x_{truck_id}_{src}_{dst}",
                )
                self._arc_time[(truck_id, src, dst)] = float(travel_time)

        # Arcs to terminal node (zero-time)
        for src in real_nodes:
            arc_vars[(src, end_node)] = model.addVar(
                vtype=GRB.BINARY,
                name=f"x_{truck_id}_{src}_{end_node}",
            )
            self._arc_time[(truck_id, src, end_node)] = 0.0

        # Allow skipping deliveries when none exist
        if not deliveries:
            start_to_end = arc_vars[(start_node, end_node)]
            model.addConstr(start_to_end == 1.0, name=f"skip_{truck_id}")
            self._route_vars[truck_id] = arc_vars
            self._order_vars[truck_id] = {}
            self._truck_info[truck_id] = {
                "start": start_node,
                "end": end_node,
                "deliveries": deliveries,
            }
            return gp.LinExpr(0.0)

        # Flow constraints
        outgoing_start = gp.quicksum(
            var for (src, dst), var in arc_vars.items() if src == start_node
        )
        model.addConstr(outgoing_start == 1.0, name=f"start_out_{truck_id}")

        incoming_end = gp.quicksum(
            var for (src, dst), var in arc_vars.items() if dst == end_node
        )
        model.addConstr(incoming_end == 1.0, name=f"end_in_{truck_id}")

        for delivery in deliveries:
            incoming = gp.quicksum(
                var for (src, dst), var in arc_vars.items() if dst == delivery
            )
            outgoing = gp.quicksum(
                var for (src, dst), var in arc_vars.items() if src == delivery
            )
            model.addConstr(incoming == 1.0, name=f"in_{truck_id}_{delivery}")
            model.addConstr(outgoing == 1.0, name=f"out_{truck_id}_{delivery}")

        # MTZ subtour elimination
        order_vars: Dict[int, gp.Var] = {}
        for idx, node in enumerate(deliveries, start=1):
            order_vars[node] = model.addVar(
                lb=1.0, ub=len(deliveries), vtype=GRB.CONTINUOUS, name=f"u_{truck_id}_{node}"
            )

        for i in deliveries:
            for j in deliveries:
                if i == j:
                    continue
                var = arc_vars.get((i, j))
                if var is None:
                    continue
                model.addConstr(
                    order_vars[i] - order_vars[j] + len(deliveries) * var
                    <= len(deliveries) - 1,
                    name=f"mtz_{truck_id}_{i}_{j}",
                )

        self._route_vars[truck_id] = arc_vars
        self._order_vars[truck_id] = order_vars
        self._truck_info[truck_id] = {
            "start": start_node,
            "end": end_node,
            "deliveries": deliveries,
        }

        time_expr = gp.quicksum(
            self._arc_time[(truck_id, src, dst)] * var
            for (src, dst), var in arc_vars.items()
        )
        return time_expr

    def build_model(self) -> gp.Model:
        """Construct the full Gurobi model."""
        truck_data = self._extract_truck_data()
        model = gp.Model("truck_routing")
        model.Params.OutputFlag = 0
        self._route_vars = {}
        self._order_vars = {}
        self._truck_info = {}
        self._arc_time = {}

        total_time_expr = gp.LinExpr(0.0)
        for truck in truck_data:
            truck_time = self._add_truck_constraints(model, truck)
            total_time_expr += truck_time

        model.setObjective(total_time_expr, GRB.MINIMIZE)
        self._model = model
        return model

    def solve(self) -> TruckRouteSolution:
        """Solve the routing problem and return structured results."""
        model = self._model or self.build_model()
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            raise RuntimeError("Gurobi did not find an optimal solution.")

        truck_routes: Dict[int, List[int]] = {}
        truck_times: Dict[int, float] = {}
        total_time = model.ObjVal

        for truck_id, info in self._truck_info.items():
            start = info["start"]
            end_node = info["end"]
            route_vars = self._route_vars[truck_id]
            successors = {}
            for (src, dst), var in route_vars.items():
                if var.X > 0.5:
                    successors[src] = dst

            current = start
            ordered_nodes = [current]
            time_spent = 0.0

            while successors.get(current) is not None:
                nxt = successors[current]
                if nxt == end_node:
                    break
                ordered_nodes.append(nxt)
                time_spent += self._arc_time[(truck_id, current, nxt)]
                current = nxt

            truck_routes[truck_id] = ordered_nodes
            truck_times[truck_id] = time_spent

        solution = TruckRouteSolution(
            total_time=total_time, truck_routes=truck_routes, truck_times=truck_times
        )
        self._last_solution = solution
        return solution

    # ------------------------------------------------------------------
    # Policy-style helpers to interact with the environment
    # ------------------------------------------------------------------
    def reset_policy(self):
        """Invalidate any cached plan so the next call recomputes routes."""
        self._planned_routes = {}
        self._route_progress = {}
        self._plan_ready = False
        self._tracked_truck_ids = None

    def _prepare_routes_for_policy(self, env: "EventDrivenTruckEnv"):
        """Solve for the current environment and align truck sequences."""
        self._policy_env = env
        self.env = env
        self._graph = env.transport_graph.graph
        self._model = None
        self.reset_policy()
        solution = self.solve()

        for truck_id, nodes in solution.truck_routes.items():
            deliveries = [int(n) for n in nodes[1:]]  # drop start node
            self._planned_routes[truck_id] = deliveries
            self._route_progress[truck_id] = 0

            if truck_id < len(env.trucks):
                truck = env.trucks[truck_id]
                start_node = int(truck.delivery_sequence[0])
                if nodes:
                    start_node = int(nodes[0])
                truck.delivery_sequence = [start_node] + deliveries
                truck.current_sequence_index = min(
                    truck.current_sequence_index, len(truck.delivery_sequence) - 1
                )

        self._plan_ready = True
        self._tracked_truck_ids = tuple(id(t) for t in env.trucks)

    def _sync_route_progress(self, truck_id: int, truck):
        route = self._planned_routes.get(truck_id)
        if not route:
            return
        idx = self._route_progress.get(truck_id, 0)
        current_node = int(truck.current_node)
        while idx < len(route) and current_node == int(route[idx]):
            idx += 1
        self._route_progress[truck_id] = idx

    def _next_planned_delivery(self, truck_id: int) -> Optional[int]:
        route = self._planned_routes.get(truck_id)
        if not route:
            return None
        idx = self._route_progress.get(truck_id, 0)
        if idx >= len(route):
            return None
        return int(route[idx])

    def _can_reach_with_current_battery(self, truck, target_node: int) -> bool:
        energy_needed = self.env.transport_graph.get_path_energy(
            int(truck.current_node), int(target_node)
        )
        if energy_needed == float("inf"):
            return False
        return energy_needed <= truck.current_battery + 1e-6

    def _select_supporting_charger(
        self, truck, target_node: int
    ) -> Optional[int]:
        current_node = int(truck.current_node)
        best_node = None
        best_cost = float("inf")
        for charger in self.env.charging_nodes:
            energy_to_charger = self.env.transport_graph.get_path_energy(
                current_node, int(charger)
            )
            if (
                energy_to_charger == float("inf")
                or energy_to_charger > truck.current_battery + 1e-6
            ):
                continue
            energy_from_charger = self.env.transport_graph.get_path_energy(
                int(charger), int(target_node)
            )
            if energy_from_charger == float("inf"):
                continue
            score = energy_to_charger + energy_from_charger
            if score < best_cost:
                best_cost = score
                best_node = int(charger)
        return best_node

    def _charger_power(self, env: "EventDrivenTruckEnv", charger_node: int) -> Tuple[float, float]:
        charger_type = env.charging_station.charger_type.get(charger_node, "level2")
        if charger_type == "DCFast":
            cfg = env.charging_config["dcfast"]
        else:
            cfg = env.charging_config["level2"]
        return float(cfg["charge_rate"]), float(cfg["efficiency"])

    def _required_energy_from_charger(
        self, env: "EventDrivenTruckEnv", charger_node: int, target_node: int
    ) -> float:
        energy_to_target = env.transport_graph.get_path_energy(
            int(charger_node), int(target_node)
        )
        if energy_to_target == float("inf"):
            return float("inf")

        nearest_after, energy_after = env.transport_graph.get_nearest_charging_node(
            int(target_node)
        )
        post_delivery = 0.0
        if nearest_after is not None and energy_after != float("inf"):
            post_delivery = energy_after

        return (energy_to_target + post_delivery) * (1.0 + self._charge_buffer)

    def _build_charge_action(self, env, truck, target_node: int) -> Optional[Tuple[int, float, bool]]:
        charger_node = int(truck.current_node)
        required_energy = self._required_energy_from_charger(
            env, charger_node, int(target_node)
        )
        if required_energy == float("inf"):
            return None
        if truck.current_battery >= required_energy - 1e-6:
            return None
        rate, efficiency = self._charger_power(env, charger_node)
        effective_rate = max(rate * efficiency, 1e-6)
        deficit = required_energy - truck.current_battery
        hours = deficit / effective_rate
        return (charger_node, hours, True)

    def get_action(self, env: "EventDrivenTruckEnv"):
        """Return the next action tuple to follow the optimized schedule."""
        self.env = env
        if self._plan_ready:
            current_ids = tuple(id(t) for t in env.trucks)
            if self._tracked_truck_ids != current_ids:
                self.reset_policy()

        if not self._plan_ready:
            self._prepare_routes_for_policy(env)

        truck_id = env.active_truck_id
        if truck_id is None:
            return (0, 0.0, False)

        truck = env.trucks[truck_id]
        self._sync_route_progress(truck_id, truck)
        next_target = self._next_planned_delivery(truck_id)

        if next_target is None:
            return (int(truck.current_node), 0.0, False)

        current_node = int(truck.current_node)
        # If we are at a charger and need energy, initiate charging
        if current_node in env.charging_nodes:
            charge_action = self._build_charge_action(env, truck, next_target)
            if charge_action is not None:
                return charge_action

        if self._can_reach_with_current_battery(truck, next_target):
            return (int(next_target), 0.0, False)

        charger_node = self._select_supporting_charger(truck, next_target)
        if charger_node is None:
            # No reachable charger; fall back to attempting delivery
            return (int(next_target), 0.0, False)

        if charger_node == current_node:
            charge_action = self._build_charge_action(env, truck, next_target)
            if charge_action is not None:
                return charge_action

        return (int(charger_node), 0.0, False)


class GurobiOptimalPolicy:
    """Lightweight adapter exposing the solver via a policy-like interface."""

    def __init__(self, config, seed: int = 0):
        self._solver = GurobiTruckRoutingSolver(
            config_path=config, seed=seed, auto_reset=False
        )
        self._active_env_id: Optional[int] = None

    def start_episode(self, env: "EventDrivenTruckEnv"):
        """Prepare to run on a freshly reset environment."""
        self._active_env_id = id(env)
        self._solver.reset_policy()

    def get_action(self, env: "EventDrivenTruckEnv"):
        """Return the next optimal action for the environment state."""
        if self._active_env_id != id(env):
            self.start_episode(env)
        return self._solver.get_action(env)
