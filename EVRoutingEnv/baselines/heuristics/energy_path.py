"""Reference dispatcher using only nominal costs and the public hard mask."""

from __future__ import annotations

import math

import networkx as nx

from EVRoutingEnv.state.feasibility import joint_action_feasibility


class EnergyPathHeuristicPolicy:
    """Route through chargers with full-battery approach AND customer exit.

    Goals are locked until claimed. Directed charger components and scarce
    staging options matter: a locally nearest customer need not be reachable
    after the truck has left its depot. No simulator state is modified here.
    """

    name = "energy_path_heuristic"

    def __init__(self):
        self._episode = None
        self._goals = {}

    def __call__(self, env, observation=None, info=None):
        if self._episode is not env.task_registry:
            self._episode = env.task_registry
            self._goals = {}
        truck = env.trucks[env.active_truck_id]
        depot = int(env.joint_instance.depot_node)
        origin = int(truck.current_node)
        graph = env.transport_graph
        stations = [
            int(node)
            for node in env.charging_nodes
            if env.charging_station.station_available[node]
        ]
        safety = (
            max(1.0, float(env.traffic_config["max_energy_multiplier"]))
            if env.traffic_config["enable_traffic"]
            and env.traffic_config["enable_energy_uncertainty"]
            else 1.0
        )

        def energy(source, target):
            value = graph.get_path_energy(source, target)
            return value if math.isfinite(value) and value >= 0.0 else math.inf

        decisions = joint_action_feasibility(env)
        allowed = {i for i, item in enumerate(decisions) if item.feasible}
        charges = [i for i in allowed if i >= env.num_navigation_actions]
        recharge = max(
            charges,
            key=lambda i: env.charge_action_values[i - env.num_navigation_actions],
            default=None,
        )
        available_energy = (
            truck.battery_capacity if recharge is not None else truck.current_battery
        )
        routes = nx.DiGraph()
        routes.add_nodes_from([origin, *stations])
        for source in routes.nodes:
            budget = available_energy if source == origin else truck.battery_capacity
            for target in stations:
                needed = safety * energy(source, target)
                duration = graph.get_time_distance(source, target)
                if (
                    source != target
                    and needed <= budget + 1e-9
                    and math.isfinite(duration)
                ):
                    routes.add_edge(source, target, weight=duration)
        lengths, paths = nx.single_source_dijkstra(routes, origin, weight="weight")
        route_complete = env.task_registry.route_complete_for(truck.truck_id)
        available = env.task_registry.available_tasks(
            truck.remaining_payload, truck_id=truck.truck_id
        )
        pending = env.task_registry.pending_tasks()
        if env.sequential_routing:
            pending = [task for task in pending if task.preassigned_to == truck.truck_id]
        exits = stations + ([depot] if len(pending) == 1 else [])

        def approach(goal, reserve):
            options = []
            for source, duration in lengths.items():
                if source == goal:
                    continue
                budget = (
                    available_energy if source == origin else truck.battery_capacity
                )
                if safety * (energy(source, goal) + reserve) <= budget + 1e-9:
                    options.append(
                        (duration + graph.get_time_distance(source, goal), source)
                    )
            if not options:
                return None
            cost, source = min(options)
            return cost, [*paths[source], goal]

        options = {}
        if route_complete:
            result = approach(depot, 0.0)
            if result is not None:
                options[depot] = (0, *result)
        else:
            for task in available:
                goal = task.node_id
                reserve = min((energy(goal, node) for node in exits), default=math.inf)
                result = approach(goal, reserve)
                if result is not None:
                    scarcity = sum(
                        safety * (energy(station, goal) + reserve)
                        <= truck.battery_capacity + 1e-9
                        for station in stations
                    )
                    options[goal] = (scarcity if origin == depot else 0, *result)
        goal = self._goals.get(truck.truck_id)
        if goal not in options:
            unreserved = [node for node in options if node not in self._goals.values()]
            goal = min(
                unreserved or options,
                key=lambda node: (*options[node][:2], node),
                default=None,
            )
        if goal is not None:
            self._goals[truck.truck_id] = goal
            path = options[goal][2]
            target = path[1]
            reserve = (
                0.0
                if target != goal or goal == depot
                else min((energy(goal, node) for node in exits), default=math.inf)
            )
        else:
            # Other trucks may still be claiming work in another directed
            # charger component. Remain in this component until the registry
            # causes automatic fleet waiting or depot return to become legal.
            next_stations = [
                node for node in lengths if node != origin and node in stations
            ]
            if not next_stations:
                raise RuntimeError(
                    "reference dispatcher has no reachable goal or charger"
                )
            station = min(
                next_stations,
                key=lambda node: (
                    graph.get_time_distance(node, depot),
                    lengths[node],
                    node,
                ),
            )
            target = paths[station][1]
            reserve = 0.0
        needed = safety * (energy(origin, target) + reserve)
        if truck.current_battery + 1e-9 < needed:
            if recharge is None:
                raise RuntimeError("reference route needs an unavailable recharge")
            return recharge
        if target in env.charging_nodes:
            action = env.charging_nodes.index(target)
        elif target == depot:
            action = env.num_charging_nodes + env.fixed_num_stops
        else:
            action = env.num_charging_nodes + next(
                i
                for i, task in enumerate(env.task_registry.tasks())
                if task.node_id == target
            )
        if action not in allowed:
            raise RuntimeError(
                f"reference route violates hard mask: {decisions[action].reason}"
            )
        return action
