"""Transportation graph backed by all-pairs cost matrices instead of a NetworkX edge graph.

The California-format JSON case studies list every ordered node pair as a graph edge. That is fine for a few
hundred nodes and impossible for the 10^4 nodes of ``europe_10k`` (10^8 edges). A matrix case study stores
``od_energy_kwh.npy`` and ``od_time_h.npy`` instead: float32 ``N x N`` matrices whose entry ``[i, j]`` is the
minimum-energy route ``i -> j`` and the hours on that same route. Row ``i`` is the ``i``-th smallest OSM node id,
the numbering the JSON loader assigns, so internal node indices mean the same in both formats.

The matrices already are the all-pairs route costs, so lookups are array reads and no route search runs. A leg
is the direct entry, ``get_shortest_path`` is ``[from, to]``.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from EVRoutingEnv.models.core.transportation_graph import ROUTING_POLICY_VERSION, TransportationGraph


def is_matrix_network(network_config: Dict[str, Any]) -> bool:
    """True when ``network.shortest_path_energy_file`` names a ``.npy`` matrix rather than a JSON dictionary."""
    return str(network_config.get("shortest_path_energy_file", "")).endswith(".npy")


class _NodeView:
    """The slice of ``networkx.DiGraph.nodes`` the environment uses: ``nodes[i]`` -> property dict."""

    def __init__(self, props: List[Dict[str, Any]]):
        self._props = props

    def __getitem__(self, node: int) -> Dict[str, Any]:
        return self._props[int(node)]

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self._props)))

    def __len__(self) -> int:
        return len(self._props)

    def __call__(self, data: bool = False):
        return ((i, p) for i, p in enumerate(self._props)) if data else iter(self)


class MatrixGraph:
    """A complete directed graph over ``n`` nodes, answering the read-only queries of ``nx.DiGraph``."""

    def __init__(self, props: List[Dict[str, Any]]):
        self.nodes = _NodeView(props)
        self._n = len(props)

    def __contains__(self, node) -> bool:
        try:
            return 0 <= int(node) < self._n
        except (TypeError, ValueError):
            return False

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._n))

    def __len__(self) -> int:
        return self._n

    def number_of_nodes(self) -> int:
        return self._n

    def number_of_edges(self) -> int:
        return self._n * (self._n - 1)

    def in_degree(self, node: int) -> int:
        return self._n - 1

    def out_degree(self, node: int) -> int:
        return self._n - 1

    def successors(self, node: int) -> Iterator[int]:
        return (j for j in range(self._n) if j != int(node))


class MatrixTransportationGraph(TransportationGraph):
    """``TransportationGraph`` whose route costs are read from ``N x N`` energy and time matrices."""

    def __init__(self, energy: np.ndarray, hours: np.ndarray, original_ids, chargers: Dict[int, Dict[str, int]]):
        """
        Args:
            energy: ``(N, N)`` kWh of the minimum-energy route ``i -> j`` (may be a read-only memmap).
            hours: ``(N, N)`` travel hours on that route.
            original_ids: ``N`` OSM node ids; position ``i`` is the internal index of node ``i``.
            chargers: internal index -> ``{charger type: ports}``.
        """
        n = len(original_ids)
        if energy.shape != (n, n) or hours.shape != (n, n):
            raise ValueError(f"cost matrices {energy.shape}/{hours.shape} do not match {n} node ids")
        self._energy, self._hours = energy, hours
        props = [
            {"has_charger": i in chargers, "charger_type": chargers.get(i), "original_id": int(original_ids[i])}
            for i in range(n)
        ]
        self.graph = MatrixGraph(props)
        self.num_nodes = n
        self.charging_nodes = self._extract_charging_nodes()
        self._cache_file = None
        self._cache_fingerprint = f"matrix-{n}"
        self._dense_node_index: Optional[dict[int, int]] = None
        self._dense_transport: Optional[np.ndarray] = None
        self._cost_matrices: Optional[Tuple[dict[int, int], np.ndarray, np.ndarray]] = None
        self._owned = False  # matrices are shared with the page cache until scale_network copies them
        self._distance_cache: dict = {}
        self._time_cache: dict = {}
        self._path_cache: dict = {}
        self._populated_sources: set = set()

    @classmethod
    def from_config(cls, network_config: Dict[str, Any], data_dir: str) -> "MatrixTransportationGraph":
        """Load the matrices, node ids and charger list of a matrix case study folder."""
        energy_file = os.path.join(data_dir, network_config["shortest_path_energy_file"])
        time_file = os.path.join(data_dir, network_config["shortest_path_time_file"])
        ids_file = os.path.join(data_dir, network_config.get("node_ids_file", "od_node_ids.npy"))
        # memory-mapped: a training run starts many environments, and they share one copy in the page cache
        energy = np.load(energy_file, mmap_mode="r")
        hours = np.load(time_file, mmap_mode="r")
        ids = np.load(ids_file)
        with open(os.path.join(data_dir, network_config["station_info_file"])) as f:
            raw = json.load(f)
        index = {int(node): i for i, node in enumerate(ids)}
        chargers: Dict[int, Dict[str, int]] = {}
        from EVRoutingEnv.utils.utils import map_charger_type

        for node, info in raw.items():
            if int(node) not in index:
                raise ValueError(f"charger node {node} is not in {os.path.basename(ids_file)}")
            entries = info["chargers"] if "chargers" in info else [info]
            for entry in entries:
                kind = map_charger_type(entry["station_type"])
                chargers.setdefault(index[int(node)], {})
                chargers[index[int(node)]][kind] = chargers[index[int(node)]].get(kind, 0) + int(entry["total_capacity"])
        return cls(energy, hours, ids, chargers)

    # ------------------------------------------------------------------ costs
    def get_leg_costs(self, from_node: int, to_node: int) -> tuple[float, float]:
        i, j = int(from_node), int(to_node)
        if not (0 <= i < self.num_nodes and 0 <= j < self.num_nodes):
            return math.inf, math.inf
        if i == j:
            return 0.0, 0.0
        e, t = float(self._energy[i, j]), float(self._hours[i, j])
        if not (math.isfinite(e) and e >= 0.0 and math.isfinite(t) and t >= 0.0):
            return math.inf, math.inf
        return e, t

    def get_shortest_path(self, from_node: int, to_node: int) -> list[int]:
        """The matrix stores route costs only, so a route is the direct leg."""
        i, j = int(from_node), int(to_node)
        if i == j or math.isinf(self.get_leg_costs(i, j)[0]):
            return []
        return [i, j]

    def cost_matrices(self) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
        """``(node index, energy, hours)``; the stored matrices, with unusable entries set to ``inf``."""
        if self._cost_matrices is None:
            index = {node: node for node in range(self.num_nodes)}
            energy, hours = self._energy, self._hours
            bad = ~(np.isfinite(energy) & (energy >= 0.0) & np.isfinite(hours) & (hours >= 0.0))
            if bad.any():
                energy, hours = np.where(bad, np.inf, energy), np.where(bad, np.inf, hours)
            self._cost_matrices = (index, energy, hours)
        return self._cost_matrices

    def scale_network(self, time_scale: float = 1.0, energy_scale: float = 1.0) -> None:
        """Scale every leg's time and energy (copies the matrices into memory once)."""
        if time_scale == 1.0 and energy_scale == 1.0:
            return
        self._energy = np.asarray(self._energy, dtype=np.float32) * np.float32(energy_scale)
        self._hours = np.asarray(self._hours, dtype=np.float32) * np.float32(time_scale)
        self._dense_node_index = None
        self._dense_transport = None
        self._cost_matrices = None

    # --------------------------------------------------------------- no search cache
    def get_all_nodes(self) -> list[int]:
        return list(range(self.num_nodes))

    def clear_distance_cache(self):
        """Nothing is cached: every cost is an array read."""
        self._dense_node_index = None
        self._dense_transport = None
        self._cost_matrices = None

    def save_distance_cache(self) -> bool:
        return False

    def load_distance_cache(self) -> bool:
        return False

    def get_cache_stats(self) -> dict:
        return {
            "num_cached_distances": 0,
            "cache_file_path": None,
            "cache_file_exists": False,
            "cache_file_size_mb": 0,
            "routing_policy": ROUTING_POLICY_VERSION,
        }
