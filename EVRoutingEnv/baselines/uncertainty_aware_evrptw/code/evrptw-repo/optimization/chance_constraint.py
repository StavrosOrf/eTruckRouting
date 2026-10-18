"""
optimization/chance_constraint.py
==================================
Analytical chance-constrained battery feasibility checker.

Implements Equations (3)–(4) from the paper:
    P(b^k_j ≥ β) ≥ α

Under Gaussian energy consumption, the deterministic equivalent is:
    b^k_j = b^k_i - μ_ij + g_ik  -  Φ⁻¹(α) · √(σ²_ij + σ²_cum,i)   (Eq. 4)

The checker is used at every PPO action selection step (Algorithm 3, lines 6–10)
to decide whether a route arc is battery-feasible or whether the vehicle must
first visit a charging station.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class BatteryState:
    """
    Mutable battery state carried forward along a partial route.

    Attributes
    ----------
    current_kwh : float
        Current battery level in kWh.
    cumulative_variance : float
        Accumulated σ²_cum,i — sum of σ²_E over arcs visited so far.
        This propagates uncertainty across multi-segment routes.
    capacity_kwh : float
        Maximum battery capacity B (kWh).
    min_reserve_kwh : float
        Minimum required battery reserve β (kWh).
    """

    current_kwh: float
    cumulative_variance: float = 0.0
    capacity_kwh: float = 150.0
    min_reserve_kwh: float = 7.5     # 5% of 150 kWh default

    @property
    def soc(self) -> float:
        """State of charge as fraction of capacity."""
        return self.current_kwh / self.capacity_kwh

    def copy(self) -> "BatteryState":
        return BatteryState(
            current_kwh=self.current_kwh,
            cumulative_variance=self.cumulative_variance,
            capacity_kwh=self.capacity_kwh,
            min_reserve_kwh=self.min_reserve_kwh,
        )


class ChanceConstraintChecker:
    """
    Checks and enforces battery feasibility at a given confidence level α.

    For a candidate arc (i → j) with predicted energy distribution
    (μ_E, σ²_E), the required battery at departure is:

        b_required = μ_E + Φ⁻¹(α) · √(σ²_E + σ²_cum)   (Eq. 4)

    If b_current < b_required, the arc is infeasible and the vehicle must
    detour to the nearest available charging station.

    Parameters
    ----------
    confidence : float
        Coverage level α (default 0.95 → Φ⁻¹(0.95) ≈ 1.6449).
    min_battery_fraction : float
        Minimum reserved battery β as a fraction of total capacity B.
    """

    def __init__(
        self,
        confidence: float = 0.95,
        min_battery_fraction: float = 0.05,
    ) -> None:
        self.confidence = confidence
        # Φ⁻¹(α): inverse normal CDF at confidence level
        self.z_alpha = self._phi_inverse(confidence)
        self.min_battery_fraction = min_battery_fraction

    @staticmethod
    def _phi_inverse(p: float) -> float:
        """
        Rational approximation to Φ⁻¹(p) (Abramowitz & Stegun 26.2.17).
        Accurate to ±4.5 × 10⁻⁴ for 0 < p < 1.
        """
        import math
        # For p = 0.95: returns ≈ 1.6449
        # For p = 0.99: returns ≈ 2.3263
        assert 0 < p < 1, "Confidence level must be in (0, 1)"
        if p < 0.5:
            sign = -1.0
            p_adj = p
        else:
            sign = 1.0
            p_adj = 1.0 - p
        t = math.sqrt(-2.0 * math.log(p_adj))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        num = c0 + c1 * t + c2 * t ** 2
        den = 1 + d1 * t + d2 * t ** 2 + d3 * t ** 3
        return sign * (t - num / den)

    def required_battery(
        self,
        mu_E: float,
        sigma2_E: float,
        battery_state: BatteryState,
    ) -> float:
        """
        Compute battery level required to traverse arc (i → j) safely.

            b_required = μ_E + Φ⁻¹(α) · √(σ²_E + σ²_cum,i)

        Parameters
        ----------
        mu_E : float
            Predicted mean energy consumption (kWh).
        sigma2_E : float
            Predicted energy variance (kWh²).
        battery_state : BatteryState
            Current battery state including cumulative variance.

        Returns
        -------
        float : minimum battery level needed at departure (kWh).
        """
        combined_std = math.sqrt(sigma2_E + battery_state.cumulative_variance)
        b_req = mu_E + self.z_alpha * combined_std + battery_state.min_reserve_kwh
        return b_req

    def is_feasible(
        self,
        mu_E: float,
        sigma2_E: float,
        battery_state: BatteryState,
    ) -> bool:
        """
        Return True if the current battery level satisfies the chance constraint.
        """
        b_req = self.required_battery(mu_E, sigma2_E, battery_state)
        return battery_state.current_kwh >= b_req

    def update_battery(
        self,
        mu_E: float,
        sigma2_E: float,
        battery_state: BatteryState,
        charge_kwh: float = 0.0,
    ) -> BatteryState:
        """
        Update battery state after executing arc (i → j).

        Parameters
        ----------
        mu_E : float
            Mean energy consumed on the arc.
        sigma2_E : float
            Energy variance of the arc.
        charge_kwh : float
            Energy added at node i (if a charging station was visited).

        Returns
        -------
        BatteryState : updated battery state at node j.
        """
        new_state = battery_state.copy()
        new_state.current_kwh = min(
            battery_state.current_kwh - mu_E + charge_kwh,
            battery_state.capacity_kwh,
        )
        # Accumulate uncertainty (independence assumption — see paper §5.5)
        new_state.cumulative_variance += sigma2_E
        return new_state

    def find_nearest_charger(
        self,
        current_node: int,
        charging_stations: list[int],
        distance_matrix: "array-like",
    ) -> int:
        """
        Return the index of the nearest charging station.

        Parameters
        ----------
        current_node : int
            Current vehicle position (node index).
        charging_stations : list[int]
            Indices of all charging station nodes.
        distance_matrix : 2-D array
            Symmetric distance matrix in km.

        Returns
        -------
        int : index of nearest charging station.
        """
        import numpy as np
        dists = np.array(distance_matrix)[current_node, charging_stations]
        return charging_stations[int(np.argmin(dists))]

    def compute_charging_amount(
        self,
        battery_state: BatteryState,
        charging_rate_kw: float,
        max_charge_time_min: float,
        min_charge_time_min: float = 5.0,
    ) -> tuple[float, float]:
        """
        Compute energy added and time spent at a charging station.

        Charges to full capacity unless time budget is exceeded.

        Returns
        -------
        (charge_kwh, charge_time_min) : tuple
        """
        energy_needed = battery_state.capacity_kwh - battery_state.current_kwh
        time_needed_min = (energy_needed / charging_rate_kw) * 60.0
        charge_time_min = min(max(time_needed_min, min_charge_time_min),
                              max_charge_time_min)
        charge_kwh = charging_rate_kw * (charge_time_min / 60.0)
        return charge_kwh, charge_time_min
