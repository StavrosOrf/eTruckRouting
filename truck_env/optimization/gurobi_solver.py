"""Gurobi-based solver for the truck routing problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import networkx as nx

from truck_env.utils.utils import check_navigation_feasibility

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

        # Force deliveries to follow the provided sequence order
        if not deliveries:
            # No deliveries, just connect start to end
            arc_vars[(start_node, end_node)] = model.addVar(
                vtype=GRB.BINARY, name=f"x_{truck_id}_{start_node}_{end_node}"
            )
            model.addConstr(
                arc_vars[(start_node, end_node)] == 1.0, name=f"skip_{truck_id}"
            )
            self._arc_time[(truck_id, start_node, end_node)] = 0.0
        else:
            for idx in range(len(real_nodes) - 1):
                src = real_nodes[idx]
                dst = real_nodes[idx + 1]
                travel_time = time_matrix.get(src, {}).get(dst, float("inf"))
                if not float(travel_time) < float("inf"):
                    raise ValueError(
                        f"No valid path between sequential deliveries {src}->{dst}"
                    )
                var = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"x_{truck_id}_{src}_{dst}",
                )
                model.addConstr(var == 1.0, name=f"order_{truck_id}_{idx}")
                arc_vars[(src, dst)] = var
                self._arc_time[(truck_id, src, dst)] = float(travel_time)

            # Terminal arc to end node (zero-time)
            last_node = real_nodes[-1]
            arc_vars[(last_node, end_node)] = model.addVar(
                vtype=GRB.BINARY, name=f"x_{truck_id}_{last_node}_{end_node}"
            )
            model.addConstr(
                arc_vars[(last_node, end_node)] == 1.0,
                name=f"end_{truck_id}",
            )
            self._arc_time[(truck_id, last_node, end_node)] = 0.0

        self._route_vars[truck_id] = arc_vars
        self._order_vars[truck_id] = {}
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

    def _build_charge_action(
        self, env, truck, hop_target: int, ensure_delivery: bool
    ) -> Optional[Tuple[int, float, bool]]:
        charger_node = int(truck.current_node)
        hop_target = int(hop_target)
        if hop_target in env.charging_nodes and not ensure_delivery:
            required_energy = env.transport_graph.get_path_energy(
                charger_node, hop_target
            )
        else:
            required_energy = self._required_energy_from_charger(
                env, charger_node, hop_target
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

    def _delivery_feasible_after_arrival(self, env, truck, target_node: int, node_for_leg: Optional[int] = None, assume_full: bool = False) -> bool:
        node = int(node_for_leg) if node_for_leg is not None else int(truck.current_node)
        energy_needed = env.transport_graph.get_path_energy(node, int(target_node))
        if energy_needed == float("inf"):
            return False
        nearest_after, energy_after = env.transport_graph.get_nearest_charging_node(
            int(target_node)
        )
        post_delivery = 0.0
        if nearest_after is not None and energy_after != float("inf"):
            post_delivery = energy_after
        available = (
            truck.battery_capacity if assume_full or node in env.charging_nodes else truck.current_battery
        )
        required = (energy_needed + post_delivery) * (1.0 + self._charge_buffer)
        return available >= required - 1e-6

    def _plan_multi_charger_path(
        self, env: "EventDrivenTruckEnv", truck, target_node: int
    ) -> Optional[List[int]]:
        start = int(truck.current_node)
        target = int(target_node)
        nodes = set(env.charging_nodes)
        nodes.add(target)
        nodes.add(start)
        capacity = truck.battery_capacity + 1e-6
        adjacency: Dict[int, List[Tuple[int, float]]] = {
            node: [] for node in nodes
        }
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                energy = env.transport_graph.get_path_energy(i, j)
                if energy == float("inf"):
                    continue
                if i == start and i not in env.charging_nodes:
                    available = truck.current_battery + 1e-6
                else:
                    available = capacity
                if energy > available:
                    continue
                if j == target:
                    assume_full = i in env.charging_nodes
                    if not self._delivery_feasible_after_arrival(
                        env, truck, target, node_for_leg=i, assume_full=assume_full
                    ):
                        continue
                adjacency[i].append((j, energy))

        import heapq

        heap: List[Tuple[float, int, List[int]]] = [(0.0, start, [start])]
        best_cost: Dict[int, float] = {start: 0.0}

        while heap:
            cost, node, path = heapq.heappop(heap)
            if node == target:
                return path
            for nxt, e in adjacency.get(node, []):
                new_cost = cost + e
                if new_cost + 1e-6 < best_cost.get(nxt, float("inf")):
                    best_cost[nxt] = new_cost
                    heapq.heappush(heap, (new_cost, nxt, path + [nxt]))
        return None

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
        path = self._plan_multi_charger_path(env, truck, next_target)
        if path is None or len(path) < 2:
            return (int(next_target), 0.0, False)
        next_hop = path[1]

        # If we are at a charger, ensure enough energy for next hop
        if current_node in env.charging_nodes:
            ensure_delivery = next_hop == next_target
            charge_action = self._build_charge_action(
                env, truck, next_hop if not ensure_delivery else next_target, ensure_delivery
            )
            if charge_action is not None:
                return charge_action

        energy_hop = env.transport_graph.get_path_energy(current_node, int(next_hop))
        if energy_hop == float("inf"):
            return (int(next_target), 0.0, False)
        if energy_hop > truck.current_battery + 1e-6:
            # Should not happen if charging logic works, fallback to staying put
            return (int(current_node), 0.0, False)

        return (int(next_hop), 0.0, False)


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
