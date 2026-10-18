"""
environment/evrptw_env.py
=========================
EVRPTW Gymnasium-style environment for PPO rollouts.

Wraps instance generation, reward computation (Eq. 15), feasibility
checking, and state transitions into a standard step/reset interface.
The environment supports curriculum learning by gradually increasing
n_customers from 10 → 100 over 35,000 training episodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class EVRPTWConfig:
    n_customers_min: int   = 10
    n_customers_max: int   = 100
    n_chargers: int        = 12
    battery_capacity: float = 150.0   # kWh
    vehicle_payload: float  = 3000.0  # kg
    time_horizon_h: float   = 8.0
    service_time_min: float = 15.0
    energy_cv_min: float    = 0.30
    energy_cv_max: float    = 0.40
    seed: Optional[int]     = None


class EVRPTWEnv:
    """
    Electric Vehicle Routing Problem with Time Windows environment.

    Supports:
    - Curriculum learning: n_customers grows with episode count
    - Stochastic energy consumption (Gaussian, Eq. 2)
    - Chance-constrained battery feasibility (Eq. 3–4)
    - Reward shaping (Eq. 15)

    Observation (returned by reset / step)
    ----------------------------------------
    {
      "node_features"  : ndarray (N, 7)   — Eq. 5
      "edge_index"     : ndarray (2, E)   — COO format
      "edge_features"  : ndarray (E, 4)   — Eq. 6 (uses LSTM predictions)
      "vehicle_state"  : ndarray (3,)     — [b/B, w/Q, t/T_max]
      "feasible_mask"  : ndarray (N,) bool
    }
    """

    # Reward coefficients (Table — paper §4.4, default config)
    LAMBDA_INFEAS   = 1000.0   # λ₁ — battery infeasibility penalty
    LAMBDA_TARDY    = 50.0     # λ₂ — time window tardiness penalty
    COST_ROUTING    = 0.85     # c_ij per km
    COST_CHARGING   = 0.18     # c_c per kWh
    COST_TARDINESS  = 5.0      # c_t per minute

    def __init__(self, config: EVRPTWConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._episode = 0
        self._instance: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, episode: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate a new random EVRPTW instance and return initial observation.

        Parameters
        ----------
        episode : int, optional
            Current episode count for curriculum learning.
        """
        if episode is not None:
            self._episode = episode
        else:
            self._episode += 1

        n_cust = self._curriculum_n_customers()
        self._instance = self._generate_instance(n_cust)
        self._state = self._init_state()
        return self._make_obs()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute routing action (visit node `action`) and return transition.

        Returns
        -------
        obs   : next observation dict
        reward: float — shaped reward (Eq. 15)
        done  : bool  — True when all customers served and vehicle at depot
        info  : dict  — auxiliary metrics
        """
        assert self._instance is not None, "Call reset() before step()"
        reward, info = self._execute_action(action)
        done = self._is_done()
        obs = self._make_obs() if not done else {}
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Curriculum learning
    # ------------------------------------------------------------------

    def _curriculum_n_customers(self) -> int:
        """
        Linearly anneal n_customers from min → max over 35,000 episodes,
        matching the curriculum learning schedule described in §4.1.
        """
        total_episodes = 35_000
        frac = min(self._episode / total_episodes, 1.0)
        n = int(
            self.config.n_customers_min
            + frac * (self.config.n_customers_max - self.config.n_customers_min)
        )
        return max(n, self.config.n_customers_min)

    # ------------------------------------------------------------------
    # Instance generation
    # ------------------------------------------------------------------

    def _generate_instance(self, n_customers: int) -> Dict:
        """
        Generate a random Solomon-style EVRPTW instance.

        Node layout: index 0 = depot, 1..n_cust = customers,
                     n_cust+1..n_cust+n_chg = charging stations.
        """
        cfg = self.config
        n_chg = cfg.n_chargers
        N = 1 + n_customers + n_chg

        # Random node positions in [0, 100] km²
        coords = self.rng.uniform(0, 100, size=(N, 2))
        coords[0] = [50.0, 50.0]   # depot at centre

        # Customer demands (kg), service times (min), time windows
        demands = np.zeros(N)
        demands[1:1+n_customers] = self.rng.uniform(50, 500, size=n_customers)

        service_times = np.zeros(N)
        service_times[1:1+n_customers] = cfg.service_time_min

        horizon_min = cfg.time_horizon_h * 60
        e = self.rng.uniform(0, horizon_min * 0.5, size=N)
        l = e + self.rng.uniform(30, horizon_min * 0.5, size=N)
        l = np.clip(l, e + 15, horizon_min)
        time_windows = np.stack([e, l], axis=1)

        # Distance and travel time matrices
        diff = coords[:, None, :] - coords[None, :, :]   # (N, N, 2)
        dist = np.sqrt((diff ** 2).sum(-1))               # (N, N) km
        speed_kmh = 40.0
        travel_times = dist / speed_kmh * 60.0            # minutes

        # Stochastic energy model: μ_ij from physics, σ_ij from CV
        # μ_ij = k_e × d_ij (simplified linear model — replace with LSTM)
        energy_per_km = 0.25   # kWh/km (rough EV delivery van value)
        mu_energy = dist * energy_per_km
        cv = self.rng.uniform(cfg.energy_cv_min, cfg.energy_cv_max)
        sigma_energy = mu_energy * cv
        var_energy = sigma_energy ** 2

        # Charging rates (kW)
        charging_rates = np.zeros(N)
        charger_idx = list(range(1 + n_customers, N))
        charging_rates[charger_idx] = self.rng.choice(
            [50.0, 100.0, 150.0], size=n_chg
        )

        return {
            "n_customers": n_customers,
            "n_chargers": n_chg,
            "N": N,
            "coords": coords,
            "demands": demands,
            "service_times": service_times,
            "time_windows": time_windows,
            "dist": dist,
            "travel_times": travel_times,
            "mu_energy": mu_energy,
            "var_energy": var_energy,
            "charging_rates": charging_rates,
            "charger_idx": charger_idx,
            "customer_idx": list(range(1, 1 + n_customers)),
        }

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _init_state(self) -> Dict:
        cfg = self.config
        inst = self._instance
        return {
            "current_node": 0,
            "battery_kwh": cfg.battery_capacity,
            "load_kg": 0.0,
            "time_min": 0.0,
            "cumvar": 0.0,
            "unvisited": set(inst["customer_idx"]),
            "route": [0],
            "total_cost": 0.0,
            "n_vehicles": 0,
        }

    def _make_obs(self) -> Dict[str, np.ndarray]:
        """Build observation dict matching GATEncoder input spec."""
        inst = self._instance
        st   = self._state
        cfg  = self.config
        N    = inst["N"]

        # Node features (Eq. 5) — 7-dimensional
        node_feat = np.zeros((N, 7), dtype=np.float32)
        # Depot: [0,0,0,0,1,0,0]
        node_feat[0, 4] = 1.0
        # Customers
        T_max = cfg.time_horizon_h * 60
        for i in inst["customer_idx"]:
            node_feat[i] = [
                inst["demands"][i] / cfg.vehicle_payload,
                inst["time_windows"][i, 0] / T_max,
                inst["time_windows"][i, 1] / T_max,
                inst["service_times"][i] / T_max,
                0.0,
                1.0,
                float(i not in st["unvisited"]),   # visited flag
            ]
        # Charging stations
        R_max = 150.0
        for j in inst["charger_idx"]:
            node_feat[j] = [0, 0, 0, inst["charging_rates"][j] / R_max, 0, 0, 1]

        # Fully-connected edge index
        src, dst = zip(*[(i, j) for i in range(N) for j in range(N) if i != j])
        edge_index = np.array([src, dst], dtype=np.int64)

        # Edge features (Eq. 6) — 4-dimensional: [d/D, t/T, μ/B, σ/√B]
        E = edge_index.shape[1]
        edge_feat = np.zeros((E, 4), dtype=np.float32)
        D_max = 141.4  # diagonal of 100×100 grid
        for e_idx, (i, j) in enumerate(zip(src, dst)):
            mu  = inst["mu_energy"][i, j]
            sig = math.sqrt(inst["var_energy"][i, j])
            edge_feat[e_idx] = [
                inst["dist"][i, j] / D_max,
                inst["travel_times"][i, j] / T_max,
                mu / cfg.battery_capacity,
                sig / math.sqrt(cfg.battery_capacity),
            ]

        # Vehicle state
        vehicle_state = np.array([
            st["battery_kwh"] / cfg.battery_capacity,
            st["load_kg"] / cfg.vehicle_payload,
            st["time_min"] / T_max,
        ], dtype=np.float32)

        # Feasibility mask — time windows + capacity
        feasible = np.zeros(N, dtype=bool)
        for j in range(N):
            if j == st["current_node"]:
                continue
            tt = inst["travel_times"][st["current_node"], j]
            arr = st["time_min"] + tt
            tw_ok = arr <= inst["time_windows"][j, 1]
            cap_ok = (j not in inst["customer_idx"]
                      or st["load_kg"] + inst["demands"][j] <= cfg.vehicle_payload)
            feasible[j] = tw_ok and cap_ok

        return {
            "node_features": node_feat,
            "edge_index": edge_index,
            "edge_features": edge_feat,
            "vehicle_state": vehicle_state,
            "feasible_mask": feasible,
        }

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    def _execute_action(self, action: int) -> Tuple[float, Dict]:
        inst = self._instance
        st   = self._state
        cfg  = self.config
        cur  = st["current_node"]

        mu_e  = inst["mu_energy"][cur, action]
        var_e = inst["var_energy"][cur, action]
        tt    = inst["travel_times"][cur, action]

        # Battery check (simplified — full chance constraint in ChanceConstraintChecker)
        energy_used = self.rng.normal(mu_e, math.sqrt(var_e))   # realised consumption
        battery_after = st["battery_kwh"] - energy_used

        infeasible = battery_after < 0
        reward = -(cfg.COST_ROUTING * inst["dist"][cur, action])

        if infeasible:
            reward -= cfg.LAMBDA_INFEAS
            battery_after = max(battery_after, 0.0)

        # Tardiness penalty
        arr_time = st["time_min"] + tt
        l_j = inst["time_windows"][action, 1]
        if arr_time > l_j:
            tardiness_min = arr_time - l_j
            reward -= cfg.LAMBDA_TARDY * tardiness_min

        # Charging
        charge_kwh = 0.0
        if action in inst["charger_idx"]:
            charge_kwh = cfg.battery_capacity - battery_after
            battery_after = cfg.battery_capacity
            reward -= cfg.COST_CHARGING * charge_kwh

        # Update state
        st["battery_kwh"] = battery_after
        st["time_min"] = arr_time + inst["service_times"][action]
        st["cumvar"] += var_e
        st["current_node"] = action
        st["unvisited"].discard(action)
        st["route"].append(action)
        st["total_cost"] -= reward

        info = {
            "action": action,
            "energy_used_kwh": energy_used,
            "battery_kwh": battery_after,
            "time_min": st["time_min"],
            "infeasible": infeasible,
            "charge_kwh": charge_kwh,
        }
        return reward, info

    def _is_done(self) -> bool:
        st = self._state
        return (
            len(st["unvisited"]) == 0
            and st["current_node"] == 0
        ) or st["time_min"] > self.config.time_horizon_h * 60
