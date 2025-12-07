"""
Charging Curve Models for EV Battery Charging Simulation.

Implements realistic charging behavior including:
- Linear (constant-rate) charging - existing simple model
- CCCV (Constant Current - Constant Voltage) - realistic DC fast charging with SOC-based tapering
"""

import numpy as np
from typing import Dict, Tuple, Optional


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
        charger_config: Dict,
        charger_type: str = "DCFast"
    ) -> Tuple[float, Dict]:
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
        if not 0.0 <= initial_soc <= 1.0:
            raise ValueError(f"Invalid initial_soc: {initial_soc}, must be in [0.0, 1.0]")
        if charge_hours <= 0:
            raise ValueError(f"Invalid charge_hours: {charge_hours}, must be > 0")
        if battery_capacity <= 0:
            raise ValueError(f"Invalid battery_capacity: {battery_capacity}, must be > 0")
            
        # Extract config parameters
        peak_power = charger_config["charge_rate"]  # kW
        efficiency = charger_config["efficiency"]
        use_realistic = charger_config["use_realistic_curve"]
        
        # Only apply realistic curve to DC Fast chargers
        if use_realistic and charger_type == "DCFast":
            return self._cccv_charge(
                initial_soc=initial_soc,
                charge_hours=charge_hours,
                battery_capacity=battery_capacity,
                peak_power=peak_power,
                efficiency=efficiency,
                taper_start_soc=charger_config["taper_start_soc"],
                taper_power_min=charger_config["taper_power_min"]
            )
        else:
            return self._linear_charge(
                initial_soc=initial_soc,
                charge_hours=charge_hours,
                battery_capacity=battery_capacity,
                peak_power=peak_power,
                efficiency=efficiency
            )
    
    def _linear_charge(
        self,
        initial_soc: float,
        charge_hours: float,
        battery_capacity: float,
        peak_power: float,
        efficiency: float
    ) -> Tuple[float, Dict]:
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
        num_samples = min(10, max(2, int(actual_charge_hours * 4)))  # 4 samples per hour
        power_curve = []
        for i in range(num_samples):
            t = (i / (num_samples - 1)) * actual_charge_hours if num_samples > 1 else 0
            soc = initial_soc + (final_soc - initial_soc) * (i / (num_samples - 1)) if num_samples > 1 else final_soc
            power_curve.append((t, peak_power, soc))
        
        details = {
            "actual_charge_hours": actual_charge_hours,
            "final_soc": final_soc,
            "average_power": average_power,
            "taper_factor": taper_factor,
            "model_used": "linear",
            "power_curve": power_curve
        }
        
        if self.verbose:
            print(f"    [Linear Charging] {initial_soc*100:.1f}% → {final_soc*100:.1f}%")
            print(f"      Charge: {actual_charge_kwh:.2f} kWh in {actual_charge_hours:.2f}h")
            print(f"      Power: {average_power:.1f} kW (constant)")
        
        return actual_charge_kwh, details
    
    def _cccv_charge(
        self,
        initial_soc: float,
        charge_hours: float,
        battery_capacity: float,
        peak_power: float,
        efficiency: float,
        taper_start_soc: float,
        taper_power_min: float
    ) -> Tuple[float, Dict]:
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
            # Determine current power based on SOC - matches real Silverado EV curve
            if current_soc < 0.1:
                # Initial ramp-up (0-10%): start at ~60% power
                ramp_progress = current_soc / 0.1
                current_power = peak_power * (0.6 + 0.3 * ramp_progress)  # 60% to 90%
            elif current_soc < 0.5:
                # Continue ramp (10-50%): 90% to 100% power
                ramp_progress = (current_soc - 0.1) / 0.4
                current_power = peak_power * (0.9 + 0.1 * ramp_progress)  # 90% to 100%
            elif current_soc < taper_start_soc:
                # Plateau phase (50-80%): maintain peak power
                current_power = peak_power
            else:
                # Taper phase (80-100%): gradual decline matching the curve
                # The curve shows a more gradual taper than exponential
                soc_progress = (current_soc - taper_start_soc) / (1.0 - taper_start_soc)
                
                # Use polynomial taper for smoother decline (matches image better)
                # Power declines from peak to ~40% at 100% SOC (image shows ~50kW from 150kW)
                taper_ratio = 0.4  # End at 40% of peak power
                power_fraction = 1.0 - (1.0 - taper_ratio) * (soc_progress ** 1.5)  # Gentler curve
                current_power = peak_power * power_fraction
                current_power = max(current_power, taper_power_min)  # Don't go below minimum
            
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
        max_possible_charge = (1.0 - initial_soc) * battery_capacity
        actual_charge_kwh = min(actual_charge_kwh, max_possible_charge)
        
        # Calculate average power and taper factor
        average_power = actual_charge_kwh / (actual_charge_hours * efficiency) if actual_charge_hours > 0 else 0
        taper_factor = average_power / peak_power if peak_power > 0 else 1.0
        
        details = {
            "actual_charge_hours": actual_charge_hours,
            "final_soc": final_soc,
            "average_power": average_power,
            "taper_factor": taper_factor,
            "model_used": "cccv",
            "power_curve": power_curve,
            "taper_start_soc": taper_start_soc,
            "peak_power": peak_power,
            "taper_power_min": taper_power_min
        }
        
        if self.verbose:
            print(f"    [CCCV Charging] {initial_soc*100:.1f}% → {final_soc*100:.1f}%")
            print(f"      Charge: {actual_charge_kwh:.2f} kWh in {actual_charge_hours:.2f}h")
            print(f"      Power: {average_power:.1f} kW avg (peak: {peak_power:.1f} kW)")
            print(f"      Taper: {taper_factor*100:.1f}% efficiency (started at {taper_start_soc*100:.0f}% SOC)")
        
        return actual_charge_kwh, details
    
    def estimate_charge_time(
        self,
        initial_soc: float,
        target_soc: float,
        battery_capacity: float,
        charger_config: Dict,
        charger_type: str = "DCFast"
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
        
        # Calculate required energy
        energy_needed = (target_soc - initial_soc) * battery_capacity
        
        # Use binary search to find time that delivers required energy
        # (more accurate than simple division for tapered curves)
        low, high = 0.0, 20.0  # Search range (0 to 20 hours)
        tolerance = 0.01  # 0.01 hour = 36 seconds tolerance
        
        for _ in range(20):  # Max 20 iterations
            mid = (low + high) / 2
            charge_kwh, _ = self.calculate_charge(
                initial_soc=initial_soc,
                charge_hours=mid,
                battery_capacity=battery_capacity,
                charger_config=charger_config,
                charger_type=charger_type
            )
            
            final_soc = initial_soc + charge_kwh / battery_capacity
            
            if abs(final_soc - target_soc) < 0.001:  # Close enough (0.1% SOC)
                return mid
            elif final_soc < target_soc:
                low = mid
            else:
                high = mid
        
        return (low + high) / 2
