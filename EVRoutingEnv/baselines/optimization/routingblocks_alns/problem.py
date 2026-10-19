"""Load an ``evrp-bp-instance/v1`` export into the native pricing core.

Runs in the solver's own interpreter (see ``build.py``); it needs only numpy and
the compiled ``evrp_core`` module, never the environment package.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

from .build import build, library_path

_core = None


def load_core_module():
    """Import (building first if needed) the native ``evrp_core`` extension."""
    global _core
    if _core is None:
        build()
        spec = importlib.util.spec_from_file_location("evrp_core", library_path())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _core = module
    return _core


class Problem:
    """Nominal instance with local indices: 0 depot, 1..n customers, then chargers."""

    def __init__(self, instance: dict, objective: str | None = None, max_labels: int = 512):
        if instance.get("schema") != "evrp-bp-instance/v1":
            raise ValueError(f"unsupported schema {instance.get('schema')!r}")
        self.instance = instance
        self.objective = objective or instance.get("objective", "travel")
        if self.objective not in ("travel", "operating"):
            raise ValueError(f"objective must be 'travel' or 'operating', got {self.objective!r}")
        nodes = instance["nodes"]
        chargers = {c["node"]: c for c in instance["chargers"]}
        keep = [k for k, x in enumerate(nodes)
                if x["kind"] != "charger" or chargers[x["id"]]["available"]]
        self.nodes = [nodes[k] for k in keep]
        kinds = [x["kind"] for x in self.nodes]
        if kinds[0] != "depot":
            raise ValueError("first node must be the depot")
        self.n_customers = kinds.count("customer")
        if kinds != ["depot"] + ["customer"] * self.n_customers + ["charger"] * (len(kinds) - 1 - self.n_customers):
            raise ValueError("nodes must be ordered depot, customers, chargers")
        rows = np.array(keep)

        def matrix(key):
            raw = instance[key]
            full = np.array([[math.inf if v is None else v for v in row] for row in raw], dtype=float)
            return np.ascontiguousarray(full[np.ix_(rows, rows)])

        self.energy = matrix("energy_kwh")
        self.hours = matrix("time_h")
        vehicle = instance["vehicle"]
        self.battery = float(vehicle["battery_kwh"])
        self.payload = float(vehicle["payload"])
        self.trucks = int(vehicle["count"])
        starts = {round(float(s), 9) for s in vehicle["start_soc"]}
        if len(starts) != 1:
            raise NotImplementedError("trucks must start with the same SoC")
        self.start_soc = float(vehicle["start_soc"][0])
        self.margin = float(instance["energy_margin"])
        self.horizon = float(instance["horizon_h"])
        self.levels = sorted(float(v) for v in instance["charge_levels"])
        n = self.n_customers
        cust = self.nodes[1:1 + n]
        self.demand = [0.0] + [float(c["demand"]) for c in cust]
        self.service = [0.0] + [float(c["service_h"]) for c in cust]
        self.tw_open = [0.0] + [float(c["tw"][0]) for c in cust]
        self.tw_close = [math.inf] + [math.inf if c["tw"][1] is None else float(c["tw"][1]) for c in cust]
        curves = instance["charge_curves"]
        station_nodes = self.nodes[1 + n:]
        curve_of = [int(chargers[s["id"]]["curve"]) for s in station_nodes]
        weight = 0.0 if self.objective == "travel" else 1.0
        module = load_core_module()
        self.core = module.Core(
            self.energy, self.hours, n, self.demand, self.service, self.tw_open, self.tw_close,
            self.battery, self.start_soc * self.battery, self.margin, self.horizon, self.payload,
            self.levels, curves["cumulative_hours"], curve_of, float(curves["soc_step"]), weight,
            max_labels,
        )

    # -- identifiers ---------------------------------------------------------------
    def env_node(self, local: int) -> int:
        return int(self.nodes[local]["id"])

    @property
    def customers(self) -> range:
        return range(1, self.n_customers + 1)

    def price(self, sequence) -> float:
        return float(self.core.price_routes([list(sequence)])[0])

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "Problem":
        return cls(json.loads(Path(path).read_text()), **kwargs)
