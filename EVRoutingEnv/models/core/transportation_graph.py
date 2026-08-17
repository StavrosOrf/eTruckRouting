"""
Transportation Graph class for managing the road network.
"""

import os
import pickle

import networkx as nx
import numpy as np


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

        # Distance cache
        self._distance_cache = {}  # Dict[Tuple[int, int], float]
        # Cache file path: EVRoutingEnv/data/distance_matrix_cache.pkl
        # __file__ is at EVRoutingEnv/models/transportation_graph.py
        # So we go up to EVRoutingEnv, then into data
        self._cache_file = os.path.join(
            os.path.dirname(__file__),  # EVRoutingEnv/models
            "..",  # EVRoutingEnv
            "data",
            "distance_matrix_cache.pkl",
        )
        self._cache_file = os.path.normpath(self._cache_file)

        if precompute_distances:
            self._initialize_distance_cache()

        # Lazily built dense all-pairs view used by the canonical state encoders.
        self._dense_node_index: dict[int, int] | None = None
        self._dense_transport: np.ndarray | None = None

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

        rescaled = {}
        for key, value in self._distance_cache.items():
            # Energy is keyed (source, target); time carries a third element.
            scale = time_scale if len(key) > 2 and key[2] == "time" else energy_scale
            rescaled[key] = value * scale
        self._distance_cache = rescaled
        # The dense view is derived from the above, so it has to be rebuilt.
        self._dense_node_index = None
        self._dense_transport = None
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

        nodes = sorted(self.graph.nodes())
        index = {int(node): position for position, node in enumerate(nodes)}
        values = np.zeros((len(nodes), len(nodes), 3), dtype=np.float32)
        for source_position, source in enumerate(nodes):
            for target_position, target in enumerate(nodes):
                energy = self._finite_path_value(self.get_path_energy, source, target)
                travel_hours = self._finite_path_value(
                    self.get_time_distance, source, target
                )
                if energy is not None and travel_hours is not None:
                    values[source_position, target_position] = (
                        energy,
                        travel_hours,
                        1.0,
                    )
        self._dense_node_index = index
        self._dense_transport = values
        return index, values

    @staticmethod
    def _finite_path_value(function, source: int, target: int) -> float | None:
        try:
            value = float(function(int(source), int(target)))
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(value) or value < 0.0:
            return None
        return value

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
        # Ensure cache directory exists first
        cache_dir = os.path.dirname(self._cache_file)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Try to load from cache file
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, "rb") as f:
                    self._distance_cache = pickle.load(f)
                cache_entries = len(self._distance_cache)
                cache_size_mb = os.path.getsize(self._cache_file) / (1024 * 1024)
                # print(f"[TransportationGraph] ✓ Loaded distance cache from disk")
                # print(
                #     f"  - Cache contains {cache_entries} distance entries ({cache_size_mb:.2f} MB)"
                # )
                return
            except Exception as e:
                print(
                    f"[TransportationGraph] ✗ Error loading distance cache: {e}"
                )
                print("[TransportationGraph] Computing distance matrix from scratch...")
        else:
            print(f"[TransportationGraph] Cache file not found at {self._cache_file}")
            print("[TransportationGraph] Computing distance matrix from scratch...")

        # Compute all-pairs shortest paths
        print("[TransportationGraph] Computing all-pairs shortest path distances...")
        print(
            f"  - Graph has {self.num_nodes} nodes, {self.graph.number_of_edges()} edges"
        )

        # Compute all-pairs shortest paths using Bellman-Ford (handles negative weights)
        import time

        start_time = time.time()
        
        # For each source node, compute shortest paths to all destinations
        for source_node in self.graph.nodes():
            try:
                # Use Bellman-Ford which handles negative weights
                lengths = nx.single_source_bellman_ford_path_length(
                    self.graph, source_node, weight="distance"
                )
                for target_node, distance in lengths.items():
                    self._distance_cache[(source_node, target_node)] = distance
            except nx.NetworkXError:
                # If Bellman-Ford fails (e.g., negative cycle), skip this source
                pass
        
        compute_time = time.time() - start_time

        num_entries = len(self._distance_cache)
        print(
            f"  - Computed {num_entries} distance pairs in {compute_time:.2f}s"
        )

        # Save to cache file
        try:
            with open(self._cache_file, "wb") as f:
                pickle.dump(self._distance_cache, f)
            cache_size_mb = os.path.getsize(self._cache_file) / (1024 * 1024)
            print(
                f"  - ✓ Saved cache to disk: {self._cache_file} ({cache_size_mb:.2f} MB)"
            )
        except Exception as e:
            print(f"  - ✗ Error saving cache file: {e}")

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
        """
        Get the distance between two nodes.

        Uses cached distance matrix for O(1) lookup if available,
        otherwise falls back to Bellman-Ford computation.

        Args:
            from_node: Starting node
            to_node: Destination node

        Returns:
            Distance in kWh of energy used for the trip, or float('inf') if no path exists
        """
        # Check cache first
        cache_key = (from_node, to_node)
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        # If same node, distance is 0
        if from_node == to_node:
            self._distance_cache[cache_key] = 0.0
            return 0.0

        # Compute using Bellman-Ford and cache result
        try:
            lengths = nx.single_source_bellman_ford_path_length(
                self.graph, from_node, weight="distance"
            )
            distance = lengths.get(to_node, float('inf'))
            self._distance_cache[cache_key] = distance
            return distance
        except nx.NetworkXError:

            raise ValueError(f"No valid path from {from_node} to {to_node}")
            # If path doesn't exist or negative cycle detected
            self._distance_cache[cache_key] = float('inf')
            return float('inf')

    def get_shortest_path(self, from_node: int, to_node: int) -> list[int]:
        """
        Get the shortest path between two nodes.

        Args:
            from_node: Starting node
            to_node: Destination node

        Returns:
            List of nodes in the shortest path, or empty list if no path exists
        """
        try:
            return nx.shortest_path(self.graph, from_node, to_node, weight="distance")
        except nx.NetworkXNoPath:
            return []

    def get_time_distance(self, from_node: int, to_node: int) -> float:
        """
        Get the travel time between two nodes.
        
        Uses shortest path computation similar to get_path_energy for consistency.

        Args:
            from_node: Starting node
            to_node: Destination node

        Returns:
            Travel time in hours, or float('inf') if no path exists
        """
        # Check cache first
        cache_key = (from_node, to_node, 'time')
        if hasattr(self, '_time_cache') and cache_key in self._time_cache:
            return self._time_cache[cache_key]
        
        # Initialize time cache if needed
        if not hasattr(self, '_time_cache'):
            self._time_cache = {}
        
        # If same node, time distance is 0
        if from_node == to_node:
            self._time_cache[cache_key] = 0.0
            return 0.0
        
        # Check if direct edge exists
        if self.graph.has_edge(from_node, to_node):
            time_val = self.graph[from_node][to_node]["time"]
            self._time_cache[cache_key] = time_val
            return time_val
        
        # Compute shortest path using Bellman-Ford (multi-hop)
        try:
            lengths = nx.single_source_bellman_ford_path_length(
                self.graph, from_node, weight="time"
            )
            time_val = lengths.get(to_node, float('inf'))
            self._time_cache[cache_key] = time_val
            return time_val
        except nx.NetworkXError:
            # If path doesn't exist or negative cycle detected
            self._time_cache[cache_key] = float('inf')
            return float('inf')

    def get_edge_data(self, from_node: int, to_node: int) -> dict:
        """
        Get edge data between two nodes.

        Args:
            from_node: Starting node
            to_node: Destination node

        Returns:
            Dictionary containing edge attributes (distance, time, terrain_factor)
        """
        if self.graph.has_edge(from_node, to_node):
            return self.graph[from_node][to_node]
        return {}

    def get_neighbors(self, node: int) -> list[int]:
        """
        Get all neighboring nodes (direct connections).

        Args:
            node: Node to get neighbors for

        Returns:
            List of neighboring node IDs
        """
        return list(self.graph.neighbors(node))

    def generate_delivery_sequence(
        self,
        start_node: int,
        num_stops: int,
        min_hop_distance: float = 10.0,
        max_hop_distance: float = 100.0,
        exclude_charging_nodes: bool = False,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        """
        Generate a random delivery sequence with constrained hop distances.

        Args:
            start_node: Starting node for the delivery sequence
            num_stops: Number of delivery stops (excluding start)
            min_hop_distance: Minimum distance between consecutive stops (km)
            max_hop_distance: Maximum distance between consecutive stops (km)
            exclude_charging_nodes: If True, avoid charging nodes as delivery stops
            rng: Optional episode-scoped generator for reproducible instances

        Returns:
            List of nodes representing the delivery sequence [start, stop1, stop2, ...]
        """
        sequence = [start_node]
        current_node = start_node

        # Get candidate nodes (excluding charging nodes if requested) and avoid sinks
        graph_ref = self.graph
        def _is_sink(node_id: int) -> bool:
            return graph_ref.out_degree(node_id) == 0

        if exclude_charging_nodes:
            candidate_nodes = [
                n for n in self.get_all_nodes() if n not in self.charging_nodes and not _is_sink(n)
            ]
        else:
            candidate_nodes = [n for n in self.get_all_nodes() if not _is_sink(n)]

        attempts = 0
        max_attempts = 1000

        while len(sequence) < num_stops + 1 and attempts < max_attempts:
            attempts += 1

            # Find nodes within distance range from current node
            valid_next_nodes = []
            for node in candidate_nodes:
                if node in sequence:  # Skip already visited nodes
                    continue

                distance = self.get_path_energy(current_node, node)
                # Skip unreachable nodes (distance == inf)
                if distance == float('inf'):
                    continue
                if min_hop_distance <= distance <= max_hop_distance:
                    valid_next_nodes.append(node)

            if not valid_next_nodes:
                # Try relaxing distance constraints
                for node in candidate_nodes:
                    if node not in sequence:
                        distance = self.get_path_energy(current_node, node)
                        # Skip unreachable nodes
                        if distance != float('inf'):
                            valid_next_nodes.append(node)

                if not valid_next_nodes:
                    # Try including charging nodes as last resort
                    for node in self.get_all_nodes():
                        if node not in sequence and not _is_sink(node):
                            distance = self.get_path_energy(current_node, node)
                            if distance != float('inf'):
                                valid_next_nodes.append(node)

                if not valid_next_nodes:
                    break

            # Randomly select next node
            if rng is None:
                next_node = np.random.choice(valid_next_nodes)
            else:
                next_node = rng.choice(valid_next_nodes)
            sequence.append(next_node)
            current_node = next_node

        # If we couldn't generate enough stops, fill with random reachable nodes
        while len(sequence) < num_stops + 1:
            # Try to find reachable nodes (excluding charging if originally requested)
            if exclude_charging_nodes:
                remaining_nodes = [
                    n for n in candidate_nodes 
                    if n not in sequence and self.get_path_energy(current_node, n) != float('inf')
                ]
            else:
                remaining_nodes = [
                    n for n in self.get_all_nodes()
                    if n not in sequence and not _is_sink(n) and self.get_path_energy(current_node, n) != float('inf')
                ]
            
            if not remaining_nodes:
                break
            if rng is None:
                next_node = np.random.choice(remaining_nodes)
            else:
                next_node = rng.choice(remaining_nodes)
            sequence.append(next_node)
            current_node = next_node

        # make from numpy array to list of ints
        sequence = [int(n) for n in sequence]

        return sequence

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

    def has_charger(self, node: int) -> bool:
        """Check if a node has a charging station."""
        return node in self.charging_nodes

    def clear_distance_cache(self):
        """Clear the distance cache from memory."""
        self._distance_cache.clear()
        print("[TransportationGraph] Distance cache cleared from memory")

    def save_distance_cache(self) -> bool:
        """
        Save the distance cache to file.

        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            with open(self._cache_file, "wb") as f:
                pickle.dump(self._distance_cache, f)
            cache_size_mb = os.path.getsize(self._cache_file) / (1024 * 1024)
            print(f"[TransportationGraph] Distance cache saved: {cache_size_mb:.2f} MB")
            return True
        except Exception as e:
            print(f"[TransportationGraph] Error saving distance cache: {e}")
            return False

    def load_distance_cache(self) -> bool:
        """
        Load the distance cache from file.

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self._cache_file):
            print("[TransportationGraph] Cache file not found")
            return False

        try:
            with open(self._cache_file, "rb") as f:
                self._distance_cache = pickle.load(f)
            print("[TransportationGraph] Distance cache loaded from disk")
            print(f"  - Contains {len(self._distance_cache)} distance entries")
            return True
        except Exception as e:
            print(f"[TransportationGraph] Error loading distance cache: {e}")
            return False

    def get_cache_stats(self) -> dict:
        """
        Get statistics about the distance cache.

        Returns:
            Dictionary with cache statistics
        """
        cache_exists = os.path.exists(self._cache_file)
        cache_size_mb = 0
        if cache_exists:
            cache_size_mb = os.path.getsize(self._cache_file) / (1024 * 1024)

        return {
            "num_cached_distances": len(self._distance_cache),
            "cache_file_path": self._cache_file,
            "cache_file_exists": cache_exists,
            "cache_file_size_mb": cache_size_mb,
        }
