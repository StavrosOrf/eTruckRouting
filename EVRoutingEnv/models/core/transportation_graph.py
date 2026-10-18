"""
Transportation Graph class for managing the road network.
"""

import hashlib
import heapq
import json
import math
import os
import pickle
import tempfile

import networkx as nx
import numpy as np


ROUTING_POLICY_VERSION = "min_energy_then_time_v1"


class TransportationGraph:
    """Manages the transportation network graph and routing operations."""

    def __init__(self, graph: nx.DiGraph, precompute_distances: bool = True):
        """
        Initialize the transportation graph.

        Args:
            graph: NetworkX directed graph with nodes and edges representing the road network
            precompute_distances: If True, precompute and cache all-pairs shortest paths
        """
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.charging_nodes = self._extract_charging_nodes()
        self._ordered_successors = {
            node: tuple(sorted(graph.successors(node))) for node in graph
        }
        self._route_successors = None

        # Distance cache
        self._distance_cache = {}  # (source, target) -> route energy
        self._time_cache = {}  # (source, target, "time") -> time on that route
        self._path_cache = {}  # (source, target) -> energy-optimal route
        self._populated_sources = set()
        # A cache belongs to one graph, including its node mapping and costs.
        # The old package-wide pickle had no identity and leaked costs between
        # unrelated networks. Keep versioned caches outside the source tree.
        identity = {
            "routing_policy": ROUTING_POLICY_VERSION,
            "nodes": sorted(
                (node, data.get("original_id", node))
                for node, data in graph.nodes(data=True)
            ),
            "edges": sorted(
                (u, v, data.get("distance", 1.0), data.get("time"))
                for u, v, data in graph.edges(data=True)
            ),
        }
        self._cache_fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self._cache_file = os.path.join(
            tempfile.gettempdir(),
            "evrp_transport_cache_v3",
            self._cache_fingerprint + ".pkl",
        )

        if precompute_distances:
            self._initialize_distance_cache()

        # Lazily built dense all-pairs view used by the canonical state encoders.
        self._dense_node_index: dict[int, int] | None = None
        self._dense_transport: np.ndarray | None = None
        # Full-precision energy/time matrices for the instance certificate (see below).
        self._cost_matrices: tuple[dict[int, int], np.ndarray, np.ndarray] | None = None

    def scale_network(self, time_scale: float = 1.0, energy_scale: float = 1.0) -> None:
        """Scale every travel time and energy on this network.

        Used by the generalization campaign to move the road network itself.
        Edge attributes alone are not enough: shortest-path energies are served
        from a precomputed cache that is loaded from disk and never consults the
        edges again, so scaling one without the other would leave the simulator
        and its planners disagreeing about the same leg. Everything derived is
        scaled or invalidated here, in one place.
        """
        if time_scale == 1.0 and energy_scale == 1.0:
            return
        for _, _, attributes in self.graph.edges(data=True):
            if "time" in attributes:
                attributes["time"] = float(attributes["time"]) * time_scale
            if "distance" in attributes:
                attributes["distance"] = float(attributes["distance"]) * energy_scale

        self._distance_cache = {
            key: value * energy_scale for key, value in self._distance_cache.items()
        }
        self._time_cache = {
            key: value * time_scale for key, value in self._time_cache.items()
        }
        # The dense view is derived from the above, so it has to be rebuilt.
        self._dense_node_index = None
        self._dense_transport = None
        self._cost_matrices = None
        self._route_successors = None
        # A scaled network must never be written back over the shared cache file.
        self._cache_file = None

    def dense_transport_matrix(self) -> tuple[dict[int, int], np.ndarray]:
        """Return an all-pairs (energy, travel hours, reachable) lookup table.

        The road network is fixed for the lifetime of the environment, so this
        table is built once and then reused by every canonical state extraction.
        Unreachable pairs are stored as zeros with a reachability flag of zero
        so that downstream tensors stay finite.
        """
        if self._dense_transport is not None and self._dense_node_index is not None:
            return self._dense_node_index, self._dense_transport

        index, energy, hours = self.cost_matrices()
        usable = np.isfinite(energy) & (energy >= 0.0) & np.isfinite(hours) & (hours >= 0.0)
        values = np.zeros((*energy.shape, 3), dtype=np.float32)
        values[:, :, 0] = np.where(usable, energy, 0.0)
        values[:, :, 1] = np.where(usable, hours, 0.0)
        values[:, :, 2] = usable
        self._dense_node_index = index
        self._dense_transport = values
        return index, values

    def cost_matrices(self) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
        """Return ``(node index, energy, hours)`` as float64 all-pairs matrices.

        Entries are exactly what :meth:`get_leg_costs` returns (``inf`` when a pair is
        unreachable), so vectorised planners make the same decisions as per-pair lookups.
        Built once per graph; ``dense_transport_matrix`` is float32 and is not used for this.
        """
        if self._cost_matrices is None:
            nodes = sorted(int(n) for n in self.graph.nodes())
            index = {node: position for position, node in enumerate(nodes)}
            energy = np.full((len(nodes), len(nodes)), np.inf)
            hours = np.full((len(nodes), len(nodes)), np.inf)
            for i, source in enumerate(nodes):
                for j, target in enumerate(nodes):
                    energy[i, j], hours[i, j] = self.get_leg_costs(source, target)
            self._cost_matrices = (index, energy, hours)
        return self._cost_matrices

    def _extract_charging_nodes(self) -> list[int]:
        """Extract all nodes that have charging stations."""
        return [
            node
            for node, data in self.graph.nodes(data=True)
            if data.get("has_charger", False)
        ]

    def _initialize_distance_cache(self):
        """
        Initialize the distance cache by either loading from file or computing it.
        This is called during __init__ if precompute_distances=True.
        """
        if self.load_distance_cache():
            return

        # Compute energy-optimal paths and the travel time on each same path.
        # Negative OD energy values cannot be executed by Truck.move_to_node,
        # so those edges are excluded rather than used as regenerative shortcuts.
        print("[TransportationGraph] Computing all-pairs route costs...")
        print(
            f"  - Graph has {self.num_nodes} nodes, {self.graph.number_of_edges()} edges"
        )

        # Compute deterministic minimum-energy paths, breaking ties by travel
        # time and then by hop count and node sequence.
        import time

        start_time = time.time()
        
        # For each source node, compute paired costs to all destinations.
        for source_node in self.graph.nodes():
            self._populate_source_routes(source_node)
        
        compute_time = time.time() - start_time

        num_entries = len(self._distance_cache)
        print(
            f"  - Computed {num_entries} distance pairs in {compute_time:.2f}s"
        )

        self.save_distance_cache()

    def get_charging_nodes(self) -> list[int]:
        """Return list of all charging station nodes."""
        return self.charging_nodes.copy()

    def get_charger_details(self) -> dict[int, dict[str, object]]:
        """Return details for each charger node.

        Returns a dict keyed by internal node id with values:
        { 'original_id': <original node id>, 'types': {type: count, ...} }
        """
        details: dict[int, dict[str, object]] = {}
        for node in self.charging_nodes:
            data = self.graph.nodes[node]
            details[node] = {
                "original_id": data.get("original_id", node),
                "types": data.get("charger_type", {}) or {},
            }
        return details

    def get_all_nodes(self) -> list[int]:
        """Return list of all nodes in the graph."""
        return list(self.graph.nodes())

    def get_path_energy(self, from_node: int, to_node: int) -> float:
        """Energy consumed along the energy-optimal executable route."""
        return self.get_leg_costs(from_node, to_node)[0]

    def get_shortest_path(self, from_node: int, to_node: int) -> list[int]:
        """Return the route used by both energy and travel-time lookups."""
        self.get_leg_costs(from_node, to_node)
        return list(self._path_cache.get((int(from_node), int(to_node)), ()))

    def get_time_distance(self, from_node: int, to_node: int) -> float:
        """Travel hours on the route returned by :meth:`get_shortest_path`."""
        return self.get_leg_costs(from_node, to_node)[1]

    def get_leg_costs(self, from_node: int, to_node: int) -> tuple[float, float]:
        """Return energy and time for one energy-optimal executable route.

        Negative or nonfinite edge costs cannot be used by the truck's movement
        and event models. Such edges are excluded from route search; an alternate
        valid path is used when one exists. Both costs are therefore drawn from
        the same explicit path, including for data supplied as OD matrices.
        """
        from_node, to_node = int(from_node), int(to_node)
        pair = (from_node, to_node)
        if from_node not in self.graph or to_node not in self.graph:
            return math.inf, math.inf
        if from_node == to_node:
            self._distance_cache[pair] = 0.0
            self._time_cache[(*pair, "time")] = 0.0
            self._path_cache[pair] = (from_node,)
            return 0.0, 0.0

        if from_node not in self._populated_sources:
            self._populate_source_routes(from_node)
        return (
            self._distance_cache.get(pair, math.inf),
            self._time_cache.get((*pair, "time"), math.inf),
        )

    @staticmethod
    def _usable_route_costs(data: dict) -> tuple[float, float] | None:
        """Hide malformed or negative-consumption edges from route search."""
        try:
            energy = float(data["distance"])
            travel_time = float(data["time"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (math.isfinite(energy) and energy >= 0.0):
            return None
        if not (math.isfinite(travel_time) and travel_time >= 0.0):
            return None
        return energy, travel_time

    def _populate_source_routes(self, source_node: int) -> None:
        """Cache all executable energy-optimal paths from one source."""
        if source_node in self._populated_sources:
            return
        if source_node not in self.graph:
            self._populated_sources.add(source_node)
            return
        if self._route_successors is None:
            # The fixed graph is scanned once per source. Validate/read its
            # edge costs once rather than repeating NetworkX dictionary access
            # and float validation for every source/edge combination.
            self._route_successors = {}
            for node, neighbors in self._ordered_successors.items():
                usable = []
                for neighbor in neighbors:
                    costs = self._usable_route_costs(self.graph[node][neighbor])
                    if costs is not None:
                        usable.append((neighbor, *costs))
                self._route_successors[node] = tuple(usable)
        # Every edge has non-negative energy and time. Lexicographic Dijkstra
        # gives a stable path even when graph insertion order differs, which is
        # important because disk caches are keyed by edge values, not order.
        origin_path = (source_node,)
        best = {source_node: (0.0, 0.0, 0, origin_path)}
        pending = [(0.0, 0.0, 0, origin_path, source_node)]
        while pending:
            energy, travel_time, hops, path, node = heapq.heappop(pending)
            if (energy, travel_time, hops, path) != best[node]:
                continue
            for neighbor, leg_energy, leg_time in self._route_successors[node]:
                candidate = (
                    energy + leg_energy,
                    travel_time + leg_time,
                    hops + 1,
                    (*path, neighbor),
                )
                if not (math.isfinite(candidate[0]) and math.isfinite(candidate[1])):
                    continue
                if neighbor not in best or candidate < best[neighbor]:
                    best[neighbor] = candidate
                    heapq.heappush(pending, (*candidate, neighbor))

        for target_node, (energy, route_time, _, path) in best.items():
            pair = (source_node, target_node)
            self._distance_cache[pair] = energy
            self._time_cache[(*pair, "time")] = route_time
            self._path_cache[pair] = path
        self._populated_sources.add(source_node)

    def get_nearest_charging_node(self, from_node: int) -> tuple[int, float]:
        """
        Find the nearest charging station from a given node.

        Args:
            from_node: Starting node

        Returns:
            Tuple of (nearest_charging_node, distance) or (None, float('inf')) if none found
        """
        min_distance = float("inf")
        nearest_node = None

        for charging_node in self.charging_nodes:
            if charging_node == from_node:
                return charging_node, 0.0

            distance = self.get_path_energy(from_node, charging_node)
            if distance < min_distance:
                min_distance = distance
                nearest_node = charging_node

        return nearest_node, min_distance

    def get_charger_info(self, node: int) -> dict:
        """
        Get charging station information for a node.

        Args:
            node: Node to check

        Returns:
            Dictionary with 'station_type' and 'total_capacity', or empty dict if no charger
        """
        if node in self.charging_nodes:
            node_data = self.graph.nodes[node]
            # Prefer derived info from charger_type dict if present
            types_dict = node_data.get("charger_type") or {}
            if isinstance(types_dict, dict) and types_dict:
                try:
                    # Sum capacities across types
                    total_capacity = int(sum(int(v) for v in types_dict.values()))
                except Exception:
                    total_capacity = sum(
                        int(v) if isinstance(v, (int, float, str)) and str(v).isdigit() else 0
                        for v in types_dict.values()
                    )
                # Choose dominant type (largest count) for waiting-time lookup compatibility
                dominant_type = None
                if types_dict:
                    dominant_type = max(types_dict.items(), key=lambda kv: int(kv[1]))[0]
                return {
                    "station_type": dominant_type or node_data.get("station_type", "Level2"),
                    "total_capacity": max(0, int(total_capacity)),
                }
            # Fallback to any pre-stored fields
            return {
                "station_type": node_data.get("station_type", "Level2"),
                "total_capacity": int(node_data.get("total_capacity", 1)),
            }
        return {}

    def get_charger_type(self, node: int) -> str:
        """
        Get the charger type for a charging station node.

        Args:
            node: Charging station node

        Returns:
            'Level2' or 'DCFast', or None if not a charging station
        """
        info = self.get_charger_info(node)
        return info.get("station_type", None)

    def get_charger_capacity(self, node: int) -> float:
        """
        Get the total capacity (number of charging slots) for a charging station.

        Args:
            node: Charging station node

        Returns:
            Number of charging slots, or 0 if not a charging station
        """
        info = self.get_charger_info(node)
        return info.get("total_capacity", 0.0)

    def clear_distance_cache(self):
        """Clear all paired route costs and paths from memory."""
        self._distance_cache.clear()
        self._time_cache.clear()
        self._path_cache.clear()
        self._route_successors = None
        self._populated_sources.clear()
        self._ordered_successors = {
            node: tuple(sorted(self.graph.successors(node))) for node in self.graph
        }
        self._dense_node_index = None
        self._dense_transport = None
        self._cost_matrices = None
        print("[TransportationGraph] Distance cache cleared from memory")

    def save_distance_cache(self) -> bool:
        """
        Save the distance cache to file.

        Returns:
            True if successful, False otherwise
        """
        if self._cache_file is None:
            return False
        temporary_path = None
        try:
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(self._cache_file), delete=False
            ) as file:
                temporary_path = file.name
                pickle.dump(
                    {
                        "version": 3,
                        "routing_policy": ROUTING_POLICY_VERSION,
                        "fingerprint": self._cache_fingerprint,
                        "distances": self._distance_cache,
                        "times": self._time_cache,
                        "paths": self._path_cache,
                        "populated_sources": self._populated_sources,
                    },
                    file,
                )
            os.replace(temporary_path, self._cache_file)
            return True
        except (OSError, pickle.PickleError):
            return False
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def load_distance_cache(self) -> bool:
        """Load only a versioned cache that matches this exact graph."""
        if self._cache_file is None or not os.path.exists(self._cache_file):
            return False
        try:
            with open(self._cache_file, "rb") as file:
                payload = pickle.load(file)
            if (
                not isinstance(payload, dict)
                or payload.get("version") != 3
                or payload.get("routing_policy") != ROUTING_POLICY_VERSION
                or payload.get("fingerprint") != self._cache_fingerprint
            ):
                return False
            distances = payload.get("distances")
            times = payload.get("times")
            paths = payload.get("paths")
            populated_sources = payload.get("populated_sources")
            if (
                not isinstance(distances, dict)
                or not isinstance(times, dict)
                or not isinstance(paths, dict)
                or not isinstance(populated_sources, set)
            ):
                return False
            self._distance_cache = distances
            self._time_cache = times
            self._path_cache = paths
            self._populated_sources = populated_sources
            self._dense_node_index = None
            self._dense_transport = None
            self._cost_matrices = None
            return True
        except (OSError, EOFError, pickle.PickleError):
            return False

    def get_cache_stats(self) -> dict:
        """
        Get statistics about the distance cache.

        Returns:
            Dictionary with cache statistics
        """
        cache_exists = self._cache_file is not None and os.path.exists(self._cache_file)
        cache_size_mb = 0
        if cache_exists:
            cache_size_mb = os.path.getsize(self._cache_file) / (1024 * 1024)

        return {
            "num_cached_distances": len(self._distance_cache),
            "cache_file_path": self._cache_file,
            "cache_file_exists": cache_exists,
            "cache_file_size_mb": cache_size_mb,
        }
