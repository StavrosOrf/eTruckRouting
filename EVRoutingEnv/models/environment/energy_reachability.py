"""Static energy certificates for depot, charging and customer reachability."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnergyReachability:
    """Reusable small network matrices; independent of fleet size and RNG."""

    nodes: tuple[int, ...]
    chargers: tuple[int, ...]
    energy: np.ndarray
    index: dict[int, int]
    budget: float

    @classmethod
    def build(cls, graph, charging_nodes, battery_capacity, energy_multiplier=1.0):
        if not math.isfinite(energy_multiplier) or energy_multiplier < 1.0:
            raise ValueError("energy_multiplier must be finite and at least 1")
        nodes = tuple(int(node) for node in graph.get_all_nodes())
        energy = np.asarray(
            [[graph.get_path_energy(a, b) for b in nodes] for a in nodes]
        )
        # The execution model rejects net-negative energy legs; they cannot
        # be counted as free energy or as usable charging exits by generation.
        energy = np.where(np.isfinite(energy) & (energy >= 0.0), energy, np.inf)
        return cls(
            nodes,
            tuple(map(int, charging_nodes)),
            energy,
            {node: i for i, node in enumerate(nodes)},
            battery_capacity / energy_multiplier,
        )

    def anchors(self, depot: int) -> list[int]:
        """Chargers reachable from the depot AND able to return to it."""
        nodes = [*self.chargers, int(depot)]
        indices = [self.index[node] for node in nodes]
        reachable = self.energy[np.ix_(indices, indices)] <= self.budget + 1e-9
        for k in range(len(nodes)):
            reachable |= reachable[:, k, None] & reachable[None, k, :]
        return [
            node
            for i, node in enumerate(nodes)
            if reachable[-1, i] and reachable[i, -1]
        ]

    def customers(self, depot: int, candidates) -> list[int]:
        """Customers on a full-battery anchor->customer->anchor segment.

        Customers have no charging port: separate affordable approach/exit legs
        do not suffice. This certifies individual energy reachability only,
        not a full fleet assignment, windows, queues or the episode horizon.
        """
        anchors = [self.index[node] for node in self.anchors(depot)]
        candidates = [int(node) for node in candidates if int(node) != int(depot)]
        indices = [self.index[node] for node in candidates]
        inbound = self.energy[np.ix_(anchors, indices)].min(axis=0)
        outbound = self.energy[np.ix_(indices, anchors)].min(axis=1)
        return [
            node
            for node, required in zip(candidates, inbound + outbound, strict=True)
            if required <= self.budget + 1e-9
        ]
