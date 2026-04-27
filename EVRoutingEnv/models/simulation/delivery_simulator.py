"""
Delivery Time Simulator Module
==============================

This module simulates stochastic unloading time at delivery locations with 
time-of-day dependent variation. The implementation follows the same reproducible 
pattern as the TrafficSimulator to ensure consistent behavior across different 
RL algorithm runs.

Key Features:
- Time-of-day dependent variation (business hours vs. off-hours)
- Reproducible randomness via seeded RNG and caching
- Bounded unloading times to prevent extreme outliers
- Configurable parameters via config.yaml
"""

import numpy as np
from typing import Dict, Tuple, Optional


class DeliverySimulator:
    """
    Simulates stochastic unloading time at delivery locations.
    
    The simulator applies Gaussian noise to a base unloading time, with variance
    that depends on the time of day. Business hours (9am-5pm) have higher variance
    to represent variable conditions like dock availability, traffic at the facility,
    and staffing levels.
    
    Attributes:
        enable_stochastic_unloading (bool): Whether to apply stochastic variation
        base_unloading_time (float): Base unloading duration in hours
        std_dev_factor (float): Std dev as fraction of base time (e.g., 0.20 = 20%)
        max_std_dev_hours (float): Maximum allowed std dev in hours
        business_hours_multiplier (float): Variance multiplier during business hours
        min_unloading_multiplier (float): Minimum unloading time bound (e.g., 0.75 = 75%)
        max_unloading_multiplier (float): Maximum unloading time bound (e.g., 1.5 = 150%)
        verbose (bool): Whether to print debug information
        seed (int): Random seed for reproducibility
    """
    
    def __init__(
        self,
        enable_stochastic_unloading: bool = False,
        base_unloading_time: float = 0.5,
        std_dev_factor: float = 0.20,
        max_std_dev_hours: float = 0.25,
        business_hours_multiplier: float = 1.5,
        min_unloading_multiplier: float = 0.75,
        max_unloading_multiplier: float = 1.5,
        verbose: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize the DeliverySimulator.
        
        Args:
            enable_stochastic_unloading: Enable/disable stochastic unloading time
            base_unloading_time: Base unloading duration in hours (default: 0.5h = 30min)
            std_dev_factor: Std dev as fraction of base time (default: 0.20)
            max_std_dev_hours: Cap on std dev in hours (default: 0.25h = 15min)
            business_hours_multiplier: Variance multiplier for 9am-5pm (default: 1.5)
            min_unloading_multiplier: Lower bound multiplier (default: 0.75)
            max_unloading_multiplier: Upper bound multiplier (default: 1.5)
            verbose: Print debug information
            seed: Random seed for reproducibility
        """
        self.enable_stochastic_unloading = enable_stochastic_unloading
        self.base_unloading_time = base_unloading_time
        self.std_dev_factor = std_dev_factor
        self.max_std_dev_hours = max_std_dev_hours
        self.business_hours_multiplier = business_hours_multiplier
        self.min_unloading_multiplier = min_unloading_multiplier
        self.max_unloading_multiplier = max_unloading_multiplier
        self.verbose = verbose
        
        # Reproducible uncertainty system (same pattern as TrafficSimulator)
        self.seed = seed
        self._uncertainty_cache: Dict[Tuple[int, int, int], float] = {}
        self._delivery_counters: Dict[int, int] = {}
        
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        else:
            self._rng = np.random
    
    def apply_unloading_time(
        self,
        delivery_node: int,
        current_time: float
    ) -> float:
        """
        Apply stochastic unloading time for a delivery at a specific node and time.
        
        The unloading time is sampled from a Gaussian distribution:
            actual_time ~ N(base_time, std_dev^2)
        
        Where std_dev depends on whether it's business hours (9am-5pm).
        
        Args:
            delivery_node: Node ID where delivery occurs
            current_time: Current simulation time in hours
        
        Returns:
            Actual unloading time in hours (bounded to realistic range)
        """
        if not self.enable_stochastic_unloading:
            return self.base_unloading_time
        
        # Calculate time-of-day effects (business hours: 9am-5pm)
        hour_of_day = current_time % 24
        is_business_hours = (9 <= hour_of_day < 17)
        business_multiplier = self.business_hours_multiplier if is_business_hours else 1.0
        
        # Calculate standard deviation with business hours effect
        base_std_dev = self.base_unloading_time * self.std_dev_factor
        std_dev = base_std_dev * business_multiplier
        
        # Cap standard deviation to prevent extreme outliers
        if self.max_std_dev_hours > 0:
            std_dev = min(std_dev, self.max_std_dev_hours)
        
        # Get reproducible random value for this delivery
        unloading_random = self._get_uncertainty_value(delivery_node, current_time)
        
        # Sample from normal distribution
        actual_unloading_time = self.base_unloading_time + std_dev * unloading_random
        
        # Apply bounds to keep unloading time realistic
        min_time = self.base_unloading_time * self.min_unloading_multiplier
        max_time = self.base_unloading_time * self.max_unloading_multiplier
        
        # Ensure minimum time is always positive
        min_time = max(min_time, 0.1)  # At least 0.1 hours (6 minutes)
        
        actual_unloading_time = max(actual_unloading_time, min_time)
        actual_unloading_time = min(actual_unloading_time, max_time)
        
        if self.verbose:
            variation_percent = (
                (actual_unloading_time - self.base_unloading_time) 
                / self.base_unloading_time
            ) * 100
            hours_label = " [BUSINESS HOURS]" if is_business_hours else " [OFF-HOURS]"
            print(
                f"    Unloading time{hours_label}: "
                f"{self.base_unloading_time:.2f}h → {actual_unloading_time:.2f}h "
                f"({variation_percent:+.1f}%)"
            )
        
        return actual_unloading_time
    
    def _get_uncertainty_value(
        self, 
        delivery_node: int, 
        current_time: float
    ) -> float:
        """
        Generate reproducible random value for a delivery.
        
        Uses a deterministic seeding approach based on:
        - delivery_node: Location of delivery
        - time_bucket: Time discretized to 0.5-hour buckets
        - delivery_idx: Counter for multiple deliveries at same node/time
        
        This ensures that the same delivery (node, time, occurrence) always gets
        the same random value across different algorithm runs with the same seed.
        
        Args:
            delivery_node: Node ID where delivery occurs
            current_time: Current simulation time in hours
        
        Returns:
            Standard normal random value for this delivery
        """
        # Time bucket (0.5-hour granularity)
        time_bucket = int(current_time / 0.5)
        
        # Track delivery count at this node (for multiple deliveries)
        if delivery_node not in self._delivery_counters:
            self._delivery_counters[delivery_node] = 0
        
        delivery_idx = self._delivery_counters[delivery_node]
        cache_key = (delivery_node, time_bucket, delivery_idx)
        self._delivery_counters[delivery_node] += 1
        
        # Check cache first (for efficiency)
        if cache_key in self._uncertainty_cache:
            return self._uncertainty_cache[cache_key]
        
        # Generate deterministic value using hash of cache_key
        deterministic_seed = hash(cache_key) % (2**31)
        temp_rng = np.random.RandomState(deterministic_seed)
        unloading_random = temp_rng.randn()
        
        # Cache for future lookups
        self._uncertainty_cache[cache_key] = unloading_random
        return unloading_random
    
    def reset_delivery_counters(self):
        """
        Reset delivery counters for a new episode.
        
        This should be called at the start of each episode to ensure
        consistent delivery counting across episodes.
        """
        self._delivery_counters.clear()
    
    def clear_cache(self):
        """
        Clear all cached uncertainty values and counters.
        
        Useful for testing or when changing simulator parameters.
        """
        self._uncertainty_cache.clear()
        self._delivery_counters.clear()
    
    def __repr__(self) -> str:
        """String representation of the simulator configuration."""
        status = "ENABLED" if self.enable_stochastic_unloading else "DISABLED"
        return (
            f"DeliverySimulator(status={status}, "
            f"base_time={self.base_unloading_time:.2f}h, "
            f"std_dev={self.std_dev_factor:.1%}, "
            f"business_hours_mult={self.business_hours_multiplier:.1f}x)"
        )
