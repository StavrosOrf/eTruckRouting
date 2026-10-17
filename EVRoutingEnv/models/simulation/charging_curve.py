"""
Charging Curve Models for EV Battery Charging Simulation.

Implements realistic charging behavior including:
- Linear (constant-rate) charging - existing simple model
- CCCV (Constant Current - Constant Voltage) - realistic DC fast charging with SOC-based tapering
"""

import math
from itertools import pairwise

import numpy as np

from scipy.integrate import quad


class ChargingCurveModel:
    """
    Models battery charging curves for different charger types.

    Supports multiple charging models:
    1. Linear: Constant power until full (existing behavior)
    2. CCCV: Realistic DC fast charging with power tapering at high SOC
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize charging curve model.

        Args:
            verbose: Enable detailed logging of charging calculations
        """
        self.verbose = verbose

    def calculate_charge(
        self,
        initial_soc: float,
        charge_hours: float,
        battery_capacity: float,
        charger_config: dict,
        charger_type: str = "DCFast",
    ) -> tuple[float, dict]:
        """
        Calculate actual charge delivered over time period using configured model.

        Args:
            initial_soc: Initial state of charge (0.0 to 1.0)
            charge_hours: Requested charging duration (hours)
            battery_capacity: Battery capacity (kWh)
            charger_config: Charger configuration dict with keys:
                - charge_rate: Peak power (kW)
                - efficiency: Charging efficiency (0.0 to 1.0)
                - use_realistic_curve: Whether to use realistic curve (optional, default False)
                - taper_start_soc: SOC when tapering begins (optional, default 0.8)
                - taper_power_min: Minimum power at 100% SOC (optional, default 30.0 kW)
            charger_type: "DCFast" or "Level2"

        Returns:
            Tuple of (charge_amount_kwh, details_dict) where details_dict contains:
                - actual_charge_hours: Actual time to deliver charge (may differ if hit 100%)
                - final_soc: Final SOC after charging
                - average_power: Average power during session (kW)
                - taper_factor: Ratio of avg power to peak power (1.0 = no taper)
                - model_used: "linear" or "cccv"
                - power_curve: List of (time, power, soc) samples for logging
        """
        # Validate inputs
        if not math.isfinite(initial_soc) or not 0.0 <= initial_soc <= 1.0:
            raise ValueError(
                f"Invalid initial_soc: {initial_soc}, must be in [0.0, 1.0]"
            )
        if not math.isfinite(charge_hours) or charge_hours <= 0:
            raise ValueError(f"Invalid charge_hours: {charge_hours}, must be > 0")
        if not math.isfinite(battery_capacity) or battery_capacity <= 0:
            raise ValueError(
                f"Invalid battery_capacity: {battery_capacity}, must be > 0"
            )

        # Extract config parameters
        peak_power = float(charger_config["charge_rate"])  # kW
        efficiency = float(charger_config["efficiency"])
        use_realistic = charger_config["use_realistic_curve"]
        if not isinstance(use_realistic, bool):
            raise TypeError("use_realistic_curve must be boolean")
        if charger_type not in {"DCFast", "Level2"}:
            raise ValueError("charger_type must be 'DCFast' or 'Level2'")
        if not math.isfinite(peak_power) or peak_power <= 0.0:
            raise ValueError("charge_rate must be positive")
        if not math.isfinite(efficiency) or not 0.0 < efficiency <= 1.0:
            raise ValueError("efficiency must be in (0, 1]")

        # Only apply realistic curve to DC Fast chargers
        if use_realistic and charger_type == "DCFast":
            return self._cccv_charge(
                initial_soc=initial_soc,
                charge_hours=charge_hours,
                battery_capacity=battery_capacity,
                peak_power=peak_power,
                efficiency=efficiency,
                taper_start_soc=charger_config["taper_start_soc"],
                taper_power_min=charger_config["taper_power_min"],
            )
        else:
            return self._linear_charge(
                initial_soc=initial_soc,
                charge_hours=charge_hours,
                battery_capacity=battery_capacity,
                peak_power=peak_power,
                efficiency=efficiency,
            )

    def calculate_charge_to_target(
        self,
        initial_soc: float,
        target_soc: float,
        battery_capacity: float,
        charger_config: dict,
        charger_type: str = "DCFast",
    ) -> tuple[float, dict]:
        """Integrate the charging curve until an exact target SoC is reached."""
        if not math.isfinite(initial_soc) or not 0.0 <= initial_soc <= 1.0:
            raise ValueError("initial_soc must be in [0, 1]")
        if not math.isfinite(target_soc) or not 0.0 < target_soc <= 1.0:
            raise ValueError("target_soc must be in (0, 1]")
        if target_soc <= initial_soc + 1e-12:
            raise ValueError("target_soc must be above initial_soc")
        if not math.isfinite(battery_capacity) or battery_capacity <= 0.0:
            raise ValueError("battery_capacity must be positive")

        peak_power = float(charger_config["charge_rate"])
        efficiency = float(charger_config["efficiency"])
        use_realistic = charger_config["use_realistic_curve"]
        if not isinstance(use_realistic, bool):
            raise TypeError("use_realistic_curve must be boolean")
        if charger_type not in {"DCFast", "Level2"}:
            raise ValueError("charger_type must be 'DCFast' or 'Level2'")
        if not math.isfinite(peak_power) or peak_power <= 0.0:
            raise ValueError("charge_rate must be positive")
        if not math.isfinite(efficiency) or not 0.0 < efficiency <= 1.0:
            raise ValueError("efficiency must be in (0, 1]")

        target_energy = (target_soc - initial_soc) * battery_capacity
        if not use_realistic or charger_type != "DCFast":
            duration = target_energy / (peak_power * efficiency)
            charge, details = self._linear_charge(
                initial_soc=initial_soc,
                charge_hours=duration,
                battery_capacity=battery_capacity,
                peak_power=peak_power,
                efficiency=efficiency,
            )
            details["target_soc"] = target_soc
            return charge, details

        taper_start_soc = float(charger_config["taper_start_soc"])
        taper_power_min = float(charger_config["taper_power_min"])
        if not math.isfinite(taper_start_soc) or not 0.0 < taper_start_soc < 1.0:
            raise ValueError("taper_start_soc must be in (0, 1)")
        if (
            not math.isfinite(taper_power_min)
            or taper_power_min <= 0.0
            or taper_power_min > peak_power
        ):
            raise ValueError("taper_power_min must be in (0, charge_rate]")

        # Integrate in SOC space: runtime is bounded independently of charger
        # power. Fixed time steps can stagnate below the float SOC resolution.
        elapsed = 0.0
        delivered = target_energy
        power_curve = [
            (
                0.0,
                self.cccv_power_at_soc(
                    initial_soc, peak_power, taper_start_soc, taper_power_min
                ),
                initial_soc,
            )
        ]
        boundaries = sorted(
            {
                initial_soc,
                target_soc,
                *(
                    value
                    for value in (0.1, 0.5, taper_start_soc)
                    if initial_soc < value < target_soc
                ),
                *(
                    initial_soc + (target_soc - initial_soc) * i / 20
                    for i in range(1, 20)
                ),
            }
        )
        for lower, upper in pairwise(boundaries):
            integral, _ = quad(
                lambda soc: (
                    peak_power
                    / self.cccv_power_at_soc(
                        soc, peak_power, taper_start_soc, taper_power_min
                    )
                ),
                lower,
                upper,
                epsabs=1e-11,
                epsrel=1e-9,
                limit=100,
            )
            duration = (battery_capacity / peak_power) * integral / efficiency
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError("charging duration is not numerically representable")
            elapsed += duration
            if not math.isfinite(elapsed):
                raise ValueError("charging duration is not numerically representable")
            power_curve.append(
                (
                    elapsed,
                    self.cccv_power_at_soc(
                        upper, peak_power, taper_start_soc, taper_power_min
                    ),
                    upper,
                )
            )

        current_soc = target_soc
        final_power = self.cccv_power_at_soc(
            current_soc,
            peak_power,
            taper_start_soc,
            taper_power_min,
        )
        if not power_curve or power_curve[-1][2] < target_soc - 1e-12:
            power_curve.append((elapsed, final_power, target_soc))
        average_power = delivered / elapsed if elapsed > 0.0 else 0.0
        details = {
            "actual_charge_hours": elapsed,
            "final_soc": target_soc,
            "target_soc": target_soc,
            "average_power": average_power,
            "taper_factor": average_power / peak_power,
            "model_used": "cccv",
            "power_curve": power_curve,
        }
        return delivered, details

    def charge_hours_to_target_fast(
        self,
        initial_soc: float,
        target_soc: float,
        battery_capacity: float,
        peak_power: float,
        efficiency: float,
        taper_start_soc: float,
        taper_power_min: float,
    ) -> float:
        """Hours of the realistic CCCV curve from ``initial_soc`` to ``target_soc``.

        Same integral as :meth:`calculate_charge_to_target` (the time is
        capacity / (peak * efficiency) * integral of peak / P(soc) d soc), evaluated with a fixed
        Gauss-Legendre rule on the smooth pieces of the curve instead of repeated adaptive
        quadrature, so it costs microseconds. It agrees with the adaptive integral to about 1e-9
        relative; the simulator keeps using the adaptive one. Used where only the duration is
        needed in bulk, e.g. instance certification.
        """
        if target_soc <= initial_soc + 1e-12:
            raise ValueError("target_soc must be above initial_soc")
        edges, cumulative = self._curve_table(peak_power, taper_start_soc, taper_power_min)
        integral = (
            self._cumulative_integral(target_soc, edges, cumulative, peak_power, taper_start_soc, taper_power_min)
            - self._cumulative_integral(initial_soc, edges, cumulative, peak_power, taper_start_soc, taper_power_min)
        )
        return (battery_capacity / peak_power) * integral / efficiency

    def _curve_table(self, peak_power, taper_start_soc, taper_power_min):
        """Integral of peak/P(soc) from 0 to each smooth-piece boundary (cached per charger class)."""
        key = (peak_power, taper_start_soc, taper_power_min)
        table = self.__dict__.setdefault("_curve_tables", {}).get(key)
        if table is None:
            kinks = [0.0, 0.1, 0.5, taper_start_soc, 1.0]
            ratio = taper_power_min / peak_power
            if ratio < 1.0:  # the taper floor is reached where 1 - 0.6 * progress**1.5 = ratio
                progress = ((1.0 - ratio) / 0.6) ** (1.0 / 1.5)
                if progress < 1.0:
                    kinks.append(taper_start_soc + progress * (1.0 - taper_start_soc))
            edges = np.array(sorted(set(kinks)))
            cumulative = [0.0]
            for lower, upper in pairwise(edges):
                cumulative.append(
                    cumulative[-1]
                    + self._piece_integral(lower, upper, peak_power, taper_start_soc, taper_power_min)
                )
            table = (edges, np.array(cumulative))
            self._curve_tables[key] = table
        return table

    def _piece_integral(self, lower, upper, peak_power, taper_start_soc, taper_power_min):
        """Integral of peak/P over [lower, upper] inside one smooth piece of the curve (Gauss-Legendre)."""
        if upper <= lower:
            return 0.0
        mid = 0.5 * (upper + lower)
        if mid >= 0.5 and mid < taper_start_soc:
            return (upper - lower)  # plateau: P = peak, so peak/P = 1
        # the taper pieces have an x**1.5 start, so they get a higher order than the ramps
        nodes, weights = self._gauss_legendre(48 if mid >= taper_start_soc else 16)
        half = 0.5 * (upper - lower)
        soc = half * nodes + mid
        if mid < 0.1:
            power = 0.6 + 0.3 * (soc / 0.1)
        elif mid < 0.5:
            power = 0.9 + 0.1 * ((soc - 0.1) / 0.4)
        else:
            progress = np.maximum(soc - taper_start_soc, 0.0) / (1.0 - taper_start_soc)
            power = np.maximum(1.0 - 0.6 * progress ** 1.5, taper_power_min / peak_power)
        return half * float(np.dot(weights, 1.0 / power))

    def _cumulative_integral(self, soc, edges, cumulative, peak_power, taper_start_soc, taper_power_min):
        piece = max(0, int(np.searchsorted(edges, soc, side="right")) - 1)
        if piece >= len(edges) - 1 or soc == edges[piece]:
            return float(cumulative[min(piece, len(edges) - 1)])
        return float(cumulative[piece]) + self._piece_integral(
            float(edges[piece]), soc, peak_power, taper_start_soc, taper_power_min
        )

    @staticmethod
    def _gauss_legendre(order: int):
        cache = ChargingCurveModel.__dict__.get("_gl_cache")
        if cache is None:
            cache = {}
            ChargingCurveModel._gl_cache = cache
        if order not in cache:
            cache[order] = np.polynomial.legendre.leggauss(order)
        return cache[order]

    @staticmethod
    def _cccv_power_array(soc, peak_power, taper_start_soc, taper_power_min):
        """Vectorised twin of :meth:`cccv_power_at_soc` (same piecewise curve)."""
        ramp1 = peak_power * (0.6 + 0.3 * (soc / 0.1))
        ramp2 = peak_power * (0.9 + 0.1 * ((soc - 0.1) / 0.4))
        progress = np.clip((soc - taper_start_soc) / (1.0 - taper_start_soc), 0.0, None)
        taper = np.maximum(peak_power * (1.0 - (1.0 - 0.4) * progress ** 1.5), taper_power_min)
        return np.where(soc < 0.1, ramp1, np.where(soc < 0.5, ramp2, np.where(soc < taper_start_soc, peak_power, taper)))

    def _linear_charge(
        self,
        initial_soc: float,
        charge_hours: float,
        battery_capacity: float,
        peak_power: float,
        efficiency: float,
    ) -> tuple[float, dict]:
        """
        Linear (constant-rate) charging model - existing behavior.

        Power remains constant at peak_power until battery is full.
        """
        # Calculate maximum possible charge
        max_charge_kwh = (1.0 - initial_soc) * battery_capacity

        # Calculate requested charge
        requested_charge_kwh = charge_hours * peak_power * efficiency

        # Actual charge is minimum of requested and max possible
        actual_charge_kwh = min(requested_charge_kwh, max_charge_kwh)

        # Additional safety: Ensure charge doesn't exceed capacity (handle floating point errors)
        actual_charge_kwh = min(actual_charge_kwh, max_charge_kwh)

        # Calculate actual time taken
        actual_charge_hours = actual_charge_kwh / (peak_power * efficiency)

        # Calculate final SOC
        final_soc = min(1.0, initial_soc + actual_charge_kwh / battery_capacity)

        # Average power is constant (no tapering)
        average_power = peak_power
        taper_factor = 1.0

        # Generate simple power curve (constant)
        num_samples = min(
            10, max(2, int(actual_charge_hours * 4))
        )  # 4 samples per hour
        power_curve = []
        for i in range(num_samples):
            t = (i / (num_samples - 1)) * actual_charge_hours if num_samples > 1 else 0
            soc = (
                initial_soc + (final_soc - initial_soc) * (i / (num_samples - 1))
                if num_samples > 1
                else final_soc
            )
            power_curve.append((t, peak_power, soc))

        details = {
            "actual_charge_hours": actual_charge_hours,
            "final_soc": final_soc,
            "average_power": average_power,
            "taper_factor": taper_factor,
            "model_used": "linear",
            "power_curve": power_curve,
        }

        if self.verbose:
            print(
                f"    [Linear Charging] {initial_soc * 100:.1f}% → {final_soc * 100:.1f}%"
            )
            print(
                f"      Charge: {actual_charge_kwh:.2f} kWh in {actual_charge_hours:.2f}h"
            )
            print(f"      Power: {average_power:.1f} kW (constant)")

        return actual_charge_kwh, details

    @staticmethod
    def cccv_power_at_soc(
        soc: float,
        peak_power: float,
        taper_start_soc: float,
        taper_power_min: float,
    ) -> float:
        """
        Instantaneous CCCV charging power at a given SOC.

        This is the same piecewise curve used by `_cccv_charge`.
        """
        for label, value in (
            ("soc", soc),
            ("peak_power", peak_power),
            ("taper_start_soc", taper_start_soc),
            ("taper_power_min", taper_power_min),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if not 0.0 <= soc <= 1.0:
            raise ValueError("soc must be in [0, 1]")
        if peak_power <= 0.0:
            raise ValueError("peak_power must be positive")
        if not 0.0 < taper_start_soc < 1.0:
            raise ValueError("taper_start_soc must be in (0, 1)")
        if not 0.0 < taper_power_min <= peak_power:
            raise ValueError("taper_power_min must be in (0, peak_power]")

        if soc < 0.1:
            # Initial ramp-up (0-10%): start at ~60% power.
            ramp_progress = soc / 0.1
            return peak_power * (0.6 + 0.3 * ramp_progress)

        if soc < 0.5:
            # Continue ramp (10-50%): 90% to 100% power.
            ramp_progress = (soc - 0.1) / 0.4
            return peak_power * (0.9 + 0.1 * ramp_progress)

        if soc < taper_start_soc:
            # Plateau phase (50-80%): maintain peak power.
            return peak_power

        # Taper phase (80-100%): gradual decline matching the curve.
        soc_progress = (soc - taper_start_soc) / (1.0 - taper_start_soc)
        taper_ratio = 0.4  # End at 40% of peak power.
        power_fraction = 1.0 - (1.0 - taper_ratio) * (soc_progress**1.5)
        return max(peak_power * power_fraction, taper_power_min)

    def montoya_breakpoints(
        self,
        battery_capacity: float,
        peak_power: float,
        efficiency: float,
        taper_start_soc: float,
        taper_power_min: float,
        boundaries: tuple[float, ...] = (0.5, 0.8, 1.0),
    ) -> list[tuple[float, float]]:
        """Breakpoints of a Montoya-style piecewise-linear charging function.

        Montoya et al. (2017) model the nonlinear charging function as a concave
        piecewise-linear map from charging time to state of charge, and solve
        the routing problem over that approximation.  The breakpoints here are
        not invented: each one is the (time, SoC) pair the simulator's own CCCV
        integrator reaches, so the piecewise-linear function interpolates the
        exact curve at the segment boundaries and the comparison isolates the
        approximation rather than a different physical assumption.

        The default boundaries are the phase changes of the underlying curve --
        end of ramp, start of taper, full -- which is the three-segment form the
        literature uses.
        """
        if not 0.0 < taper_start_soc < 1.0:
            raise ValueError("taper_start_soc must be in (0, 1)")
        if sorted(boundaries) != list(boundaries) or boundaries[-1] > 1.0:
            raise ValueError("boundaries must be increasing and end at or below 1.0")

        points = [(0.0, 0.0)]
        for soc in boundaries:
            _, details = self.calculate_charge_to_target(
                initial_soc=0.0,
                target_soc=float(soc),
                battery_capacity=battery_capacity,
                charger_config={
                    "charge_rate": peak_power,
                    "efficiency": efficiency,
                    "use_realistic_curve": True,
                    "taper_start_soc": taper_start_soc,
                    "taper_power_min": taper_power_min,
                },
                charger_type="DCFast",
            )
            points.append((float(details["actual_charge_hours"]), float(soc)))
        return points

    @staticmethod
    def montoya_time_to_soc(
        breakpoints: list[tuple[float, float]],
        initial_soc: float,
        target_soc: float,
    ) -> float:
        """Charging hours from ``initial_soc`` to ``target_soc`` under the PWL model."""
        if target_soc <= initial_soc:
            return 0.0

        def time_at(soc: float) -> float:
            if soc <= breakpoints[0][1]:
                return breakpoints[0][0]
            for (time_low, soc_low), (time_high, soc_high) in pairwise(breakpoints):
                if soc <= soc_high:
                    span = soc_high - soc_low
                    if span <= 0.0:
                        return time_high
                    weight = (soc - soc_low) / span
                    return time_low + weight * (time_high - time_low)
            return breakpoints[-1][0]

        return time_at(target_soc) - time_at(initial_soc)

    def _cccv_charge(
        self,
        initial_soc: float,
        charge_hours: float,
        battery_capacity: float,
        peak_power: float,
        efficiency: float,
        taper_start_soc: float,
        taper_power_min: float,
    ) -> tuple[float, dict]:
        """
        CCCV (Constant Current - Constant Voltage) charging model.

        Realistic DC fast charging behavior based on Chevrolet Silverado EV curve:
        - Phase 1 (0-50% SOC): Ramp up from ~50% to 100% peak power
        - Phase 2 (50-80% SOC): Constant power at peak (plateau)
        - Phase 3 (80-100% SOC): Gradual taper down to minimum power

        Uses numerical integration with small time steps to simulate charging.
        """
        # Simulation parameters
        dt = 0.01  # Time step (hours) - 0.6 minutes for accuracy
        time_elapsed = 0.0
        current_soc = initial_soc
        total_energy = 0.0
        current_power = 0.0  # Initialize to prevent UnboundLocalError

        # Track power curve for logging
        power_curve = []
        sample_interval = max(0.1, charge_hours / 20)  # ~20 samples
        next_sample_time = 0.0

        # Simulate charging process
        while time_elapsed < charge_hours and current_soc < 1.0:
            # Determine current power based on SOC - matches real Silverado EV curve.
            current_power = self.cccv_power_at_soc(
                current_soc,
                peak_power,
                taper_start_soc,
                taper_power_min,
            )

            # Calculate energy added in this time step (accounting for efficiency)
            energy_step = current_power * efficiency * dt

            # Check if this would overfill the battery
            remaining_capacity = (1.0 - current_soc) * battery_capacity
            if energy_step > remaining_capacity:
                # Partial step to reach exactly 100%
                partial_dt = remaining_capacity / (current_power * efficiency)
                energy_step = remaining_capacity
                total_energy += energy_step
                time_elapsed += partial_dt
                current_soc = 1.0

                # Add final sample
                power_curve.append((time_elapsed, current_power, current_soc))
                break

            # Apply energy step
            total_energy += energy_step
            current_soc += energy_step / battery_capacity

            # Check if we reached full capacity (including floating point overshoot)
            if current_soc >= 1.0:
                current_soc = 1.0
                # Add final sample and break
                power_curve.append((time_elapsed + dt, current_power, current_soc))
                time_elapsed += dt
                break

            time_elapsed += dt

            # Sample power curve for logging
            if time_elapsed >= next_sample_time:
                power_curve.append((time_elapsed, current_power, current_soc))
                next_sample_time += sample_interval

        # Ensure we have final sample
        if len(power_curve) == 0 or power_curve[-1][0] < time_elapsed:
            power_curve.append((time_elapsed, current_power, current_soc))

        # Calculate metrics
        actual_charge_kwh = total_energy
        actual_charge_hours = time_elapsed
        final_soc = min(1.0, current_soc)  # Clamp final SOC to [0, 1]

        # Additional safety: ensure charge doesn't exceed battery capacity
        headroom_kwh = max(0.0, (1.0 - initial_soc) * battery_capacity)
        if actual_charge_kwh > headroom_kwh:
            actual_charge_kwh = headroom_kwh
            details_clamped = True
        else:
            details_clamped = False

        # Recompute final SOC and average power after any clamping
        final_soc = min(
            1.0,
            initial_soc
            + (actual_charge_kwh / battery_capacity if battery_capacity > 0 else 0.0),
        )
        average_power = (
            actual_charge_kwh / (actual_charge_hours * efficiency)
            if actual_charge_hours > 0
            else 0
        )
        taper_factor = average_power / peak_power if peak_power > 0 else 1.0

        # Calculate average power and taper factor
        # (average_power/taper_factor recomputed above to reflect any clamp)

        details = {
            "actual_charge_hours": actual_charge_hours,
            "final_soc": final_soc,
            "average_power": average_power,
            "taper_factor": taper_factor,
            "model_used": "cccv",
            "power_curve": power_curve,
            "taper_start_soc": taper_start_soc,
            "peak_power": peak_power,
            "taper_power_min": taper_power_min,
            "clamped_to_capacity": details_clamped,
        }

        if self.verbose:
            print(
                f"    [CCCV Charging] {initial_soc * 100:.1f}% → {final_soc * 100:.1f}%"
            )
            print(
                f"      Charge: {actual_charge_kwh:.2f} kWh in {actual_charge_hours:.2f}h"
            )
            print(
                f"      Power: {average_power:.1f} kW avg (peak: {peak_power:.1f} kW)"
            )
            print(
                f"      Taper: {taper_factor * 100:.1f}% efficiency (started at {taper_start_soc * 100:.0f}% SOC)"
            )

        return actual_charge_kwh, details

    def estimate_charge_time(
        self,
        initial_soc: float,
        target_soc: float,
        battery_capacity: float,
        charger_config: dict,
        charger_type: str = "DCFast",
    ) -> float:
        """
        Estimate time required to charge from initial_soc to target_soc.

        Useful for action masking and policy decisions.

        Args:
            initial_soc: Starting SOC (0.0 to 1.0)
            target_soc: Target SOC (0.0 to 1.0)
            battery_capacity: Battery capacity (kWh)
            charger_config: Charger configuration dict
            charger_type: "DCFast" or "Level2"

        Returns:
            Estimated charge time (hours)
        """
        if target_soc <= initial_soc:
            return 0.0

        # This used to bisect on charge duration. Bisection cannot converge at a
        # target of 1.0, because the integrator stops exactly at full and the
        # search never observes an overshoot: it exhausted its iterations and
        # silently returned the midpoint of its 0-20 h range, i.e. 10 hours for
        # a charge that actually takes about half an hour. Integrating the curve
        # directly to the target is both exact and cheaper, and it is the same
        # routine the simulator and every baseline already use, so masks and
        # planners cannot disagree with what execution will do.
        _, details = self.calculate_charge_to_target(
            initial_soc=initial_soc,
            target_soc=target_soc,
            battery_capacity=battery_capacity,
            charger_config=charger_config,
            charger_type=charger_type,
        )
        return float(details["actual_charge_hours"])
