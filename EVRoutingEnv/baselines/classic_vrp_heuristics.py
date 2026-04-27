"""
Classic single-truck VRP heuristics adapted for EV routing.

Two baselines are provided:
- ClarkeWrightEVPolicy: Clarke-Wright savings construction.
- NearestNeighbor2OptEVPolicy: nearest neighbor tour with light 2-opt cleanup.

Both policies operate in flexible-delivery mode and fall back to the existing
heuristic policy when flexible ordering is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from EVRoutingEnv.baselines.heuristic_policy import HeuristicPolicy
from EVRoutingEnv.state.action_mask import get_action_mask
from EVRoutingEnv.state.gnn_utils import create_default_gnn_space, extract_action_graph


@dataclass
class _PlanCache:
    remaining_set: frozenset
    order: List[int]


class _BaseClassicEVPolicy:
    """Base utilities for fast VRP heuristics in flexible mode."""

    def __init__(self, verbose: bool = False, buffer_frac: float = 0.05):
        self.verbose = verbose
        self.buffer_frac = max(0.0, float(buffer_frac))
        self._plans: Dict[int, _PlanCache] = {}
        self._fallback = HeuristicPolicy(verbose=verbose, buffer_frac=buffer_frac)

    def get_action(self, env) -> int:
        if env.active_truck_id is None:
            return env.action_space.sample()

        truck = env.trucks[env.active_truck_id]
        if truck.is_complete or truck.failed:
            return env.action_space.sample()

        if not getattr(env, "enable_flexible_delivery_order", False):
            return self._fallback.get_action(env)

        remaining = [
            int(node)
            for node in truck.get_remaining_deliveries()
            if int(node) not in env.charging_nodes
        ]
        if not remaining:
            return self._fallback.get_action(env)

        remaining_set = frozenset(remaining)
        plan = self._plans.get(truck.truck_id)
        if plan is None or plan.remaining_set != remaining_set:
            plan_order = self._build_plan(env, truck, remaining)
            self._plans[truck.truck_id] = _PlanCache(remaining_set, plan_order)
            plan = self._plans[truck.truck_id]

        action_graph = self._get_action_graph(env)
        action_idx = self._choose_next_action(env, truck, plan.order, action_graph)
        if action_idx is not None:
            return action_idx

        return self._safe_fallback_action(env)

    def _build_plan(self, env, truck, remaining: Sequence[int]) -> List[int]:
        raise NotImplementedError

    def _get_action_graph(self, env):
        cached_space = getattr(env, "_default_gnn_state_space", None)
        if cached_space is None:
            mode = "vrp" if getattr(env, "enable_flexible_delivery_order", False) else "nonflex"
            cached_space = create_default_gnn_space(env, mode=mode, use_detour=False)
            env._default_gnn_state_space = cached_space
        return extract_action_graph(cached_space, env)

    def _choose_next_action(self, env, truck, plan_order: Sequence[int], action_graph):
        mask = action_graph.feasible_action_mask
        if mask is None or len(mask) == 0:
            return None

        remaining = set(
            int(node)
            for node in truck.get_remaining_deliveries()
            if int(node) not in env.charging_nodes
        )

        feasible_delivery_nodes = []
        for idx, (node_id, is_charging) in enumerate(action_graph.action_to_node_map):
            if idx >= len(mask) or not mask[idx]:
                continue
            if is_charging:
                continue
            if node_id in remaining and node_id not in env.charging_nodes:
                feasible_delivery_nodes.append(node_id)

        target_node = None
        for node in plan_order:
            if node in feasible_delivery_nodes:
                target_node = node
                break

        if target_node is None and feasible_delivery_nodes:
            target_node = self._nearest_node(env, truck.current_node, feasible_delivery_nodes)

        if target_node is None:
            return self._select_charge_action(env, action_graph, mask)

        delivery_action = self._find_action_for_node(
            action_graph.action_to_node_map, target_node, is_charging=False
        )
        if delivery_action is not None and delivery_action < len(mask) and mask[delivery_action]:
            return delivery_action

        charge_action = self._select_charge_action(env, action_graph, mask)
        if charge_action is not None:
            return charge_action

        return self._select_best_charger_action(env, action_graph, mask, target_node, len(remaining))

    def _nearest_node(self, env, current_node: int, nodes: Iterable[int]) -> Optional[int]:
        graph = env.transport_graph
        best_node = None
        best_time = float("inf")
        for node in nodes:
            travel_time = graph.get_time_distance(int(current_node), int(node))
            if travel_time < best_time:
                best_time = travel_time
                best_node = int(node)
        return best_node

    def _find_action_for_node(self, action_to_node_map, node_id: int, is_charging: bool) -> Optional[int]:
        for idx, (candidate, is_charge) in enumerate(action_to_node_map):
            if int(candidate) == int(node_id) and bool(is_charge) == bool(is_charging):
                return idx
        return None

    def _select_charge_action(self, env, action_graph, mask: np.ndarray) -> Optional[int]:
        durations = getattr(action_graph.data, "action_charge_durations", [])
        best_idx = None
        best_duration = float("inf")
        for idx, is_feasible in enumerate(mask):
            if not is_feasible:
                continue
            if idx >= len(action_graph.action_to_node_map):
                continue
            _, is_charging = action_graph.action_to_node_map[idx]
            if not is_charging:
                continue
            duration = durations[idx] if idx < len(durations) else float("inf")
            if duration < best_duration:
                best_duration = duration
                best_idx = idx
        return best_idx

    def _select_best_charger_action(
        self,
        env,
        action_graph,
        mask: np.ndarray,
        target_node: Optional[int],
        remaining_count: int,
    ) -> Optional[int]:
        graph = env.transport_graph
        truck = env.trucks[env.active_truck_id]
        current_node = int(truck.current_node)
        current_battery = float(truck.current_battery)
        battery_capacity = float(truck.battery_capacity)

        safety_factor = self._energy_safety_factor(env)
        best_idx = None
        best_score = float("inf")

        for idx, (node_id, is_charging) in enumerate(action_graph.action_to_node_map):
            if idx >= len(mask) or not mask[idx]:
                continue
            if is_charging:
                continue
            if node_id not in env.charging_nodes:
                continue

            energy_to_charger = graph.get_path_energy(current_node, int(node_id))
            if energy_to_charger == float("inf"):
                continue
            energy_to_charger *= safety_factor
            if energy_to_charger > current_battery:
                continue

            if target_node is None:
                score = energy_to_charger
            else:
                energy_to_target = graph.get_path_energy(int(node_id), int(target_node))
                if energy_to_target == float("inf"):
                    continue
                energy_to_target *= safety_factor

                required_from_charger = energy_to_target
                if remaining_count > 1:
                    _, energy_after = graph.get_nearest_charging_node(int(target_node))
                    if energy_after is None or energy_after == float("inf"):
                        continue
                    required_from_charger += energy_after * safety_factor

                required_from_charger *= 1.0 + self.buffer_frac
                if required_from_charger > battery_capacity + 1e-6:
                    continue

                battery_on_arrival = max(0.0, current_battery - energy_to_charger)
                charge_rate = self._charger_rate(env, int(node_id))
                deficit = max(0.0, min(required_from_charger, battery_capacity) - battery_on_arrival)
                charge_time = deficit / max(charge_rate, 1e-6)
                score = charge_time + 1e-6 * energy_to_charger

            if score < best_score - 1e-9:
                best_score = score
                best_idx = idx

        return best_idx

    def _charger_rate(self, env, node_id: int) -> float:
        if hasattr(env, "charger_type"):
            ctype = env.charger_type.get(node_id, "level2")
        else:
            ctype = env.transport_graph.get_charger_type(node_id) or "level2"
        key = "dcfast" if str(ctype).lower() == "dcfast" else "level2"
        cfg = env.charging_config[key]
        return float(cfg["charge_rate"]) * float(cfg["efficiency"])

    def _energy_safety_factor(self, env) -> float:
        traffic_cfg = getattr(env, "traffic_config", {})
        if traffic_cfg.get("enable_traffic") and traffic_cfg.get("enable_energy_uncertainty"):
            return float(traffic_cfg.get("max_energy_multiplier", 1.0))
        return 1.0

    def _safe_fallback_action(self, env) -> int:
        try:
            mask = get_action_mask(env)
        except Exception:
            return env.action_space.sample()

        if mask is None or not np.any(mask):
            return env.action_space.sample()

        for idx, is_feasible in enumerate(mask):
            if is_feasible:
                return int(idx)

        return env.action_space.sample()


class ClarkeWrightEVPolicy(_BaseClassicEVPolicy):
    """Clarke-Wright savings route construction with EV-aware action selection."""

    def __init__(self, verbose: bool = False, buffer_frac: float = 0.05):
        super().__init__(verbose=verbose, buffer_frac=buffer_frac)

    def _build_plan(self, env, truck, remaining: Sequence[int]) -> List[int]:
        depot = int(truck.delivery_sequence[0])
        deliveries = [int(node) for node in remaining if int(node) != depot]
        if len(deliveries) <= 1:
            return deliveries

        graph = env.transport_graph
        routes = {node: [node] for node in deliveries}
        route_of = {node: node for node in deliveries}
        savings: List[Tuple[float, int, int]] = []

        for i in deliveries:
            for j in deliveries:
                if i >= j:
                    continue
                c_di = graph.get_time_distance(depot, i)
                c_dj = graph.get_time_distance(depot, j)
                c_ij = graph.get_time_distance(i, j)
                if any(val == float("inf") for val in (c_di, c_dj, c_ij)):
                    continue
                savings.append((c_di + c_dj - c_ij, i, j))

        savings.sort(key=lambda x: x[0], reverse=True)

        for _, i, j in savings:
            ri = route_of.get(i)
            rj = route_of.get(j)
            if ri is None or rj is None or ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]

            merged = None
            if route_i[-1] == i and route_j[0] == j:
                merged = route_i + route_j
            elif route_i[0] == i and route_j[-1] == j:
                merged = route_j + route_i
            elif route_i[0] == i and route_j[0] == j:
                merged = list(reversed(route_i)) + route_j
            elif route_i[-1] == i and route_j[-1] == j:
                merged = route_i + list(reversed(route_j))

            if merged is None:
                continue

            new_key = merged[0]
            routes[new_key] = merged
            for node in merged:
                route_of[node] = new_key
            if ri in routes and ri != new_key:
                routes.pop(ri, None)
            if rj in routes and rj != new_key:
                routes.pop(rj, None)

            if len(routes) == 1:
                break

        if len(routes) > 1:
            merged_route = []
            for route in routes.values():
                merged_route.extend(route)
            return merged_route

        if not routes:
            return []
        return list(next(iter(routes.values())))


class NearestNeighbor2OptEVPolicy(_BaseClassicEVPolicy):
    """Nearest-neighbor tour with light 2-opt improvement."""

    def __init__(self, verbose: bool = False, buffer_frac: float = 0.05, max_2opt_iters: int = 50):
        super().__init__(verbose=verbose, buffer_frac=buffer_frac)
        self.max_2opt_iters = max(1, int(max_2opt_iters))

    def _build_plan(self, env, truck, remaining: Sequence[int]) -> List[int]:
        depot = int(truck.delivery_sequence[0])
        deliveries = [int(node) for node in remaining if int(node) != depot]
        if len(deliveries) <= 1:
            return deliveries

        graph = env.transport_graph
        unvisited = set(deliveries)
        current = int(truck.current_node) if truck.current_node in unvisited else depot
        route = []

        while unvisited:
            next_node = min(
                unvisited,
                key=lambda n: graph.get_time_distance(current, n),
            )
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        route = self._two_opt(route, depot, graph)
        return route

    def _two_opt(self, route: List[int], depot: int, graph) -> List[int]:
        if len(route) < 4:
            return route

        best = route[:]
        best_cost = self._route_cost([depot] + best + [depot], graph)
        improved = True
        iters = 0

        while improved and iters < self.max_2opt_iters:
            improved = False
            iters += 1
            for i in range(len(best) - 2):
                for k in range(i + 1, len(best) - 1):
                    new_route = best[:i] + list(reversed(best[i : k + 1])) + best[k + 1 :]
                    new_cost = self._route_cost([depot] + new_route + [depot], graph)
                    if new_cost + 1e-6 < best_cost:
                        best = new_route
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break

        return best

    def _route_cost(self, route: List[int], graph) -> float:
        cost = 0.0
        for i in range(len(route) - 1):
            seg = graph.get_time_distance(int(route[i]), int(route[i + 1]))
            if seg == float("inf"):
                return float("inf")
            cost += seg
        return cost
