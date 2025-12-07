"""
Traffic simulation module for electric vehicle routing environment.

Provides time-of-day dependent traffic modeling with rush hour effects.
"""

import numpy as np


class TrafficSimulator:
    """
    Simulates traffic uncertainty with time-of-day dependent variance.
    
    Uses Gaussian distribution with higher variance during rush hours (7-9am, 5-7pm).
    Rush hour effects are applied based on how much of the journey occurs during rush hours.
    """
    
    def __init__(
        self,
        enable_traffic: bool,
        std_dev_factor: float,
        max_std_dev_hours: float,
        rush_hour_multiplier: float,
        verbose: bool = False
    ):
        """
        Initialize the traffic simulator.
        
        Args:
            enable_traffic: Whether traffic simulation is enabled
            std_dev_factor: Standard deviation as fraction of mean travel time
            max_std_dev_hours: Maximum standard deviation cap (hours)
            rush_hour_multiplier: Multiplier for variance during rush hours
            verbose: Whether to print detailed traffic information
        """
        self.enable_traffic = enable_traffic
        self.std_dev_factor = std_dev_factor
        self.max_std_dev_hours = max_std_dev_hours
        self.rush_hour_multiplier = rush_hour_multiplier
        self.verbose = verbose
    
    def apply_traffic(
        self,
        travel_time: float,
        current_time: float,
        from_node: int = None,
        to_node: int = None
    ) -> float:
        """
        Apply traffic simulation to travel time using time-of-day dependent variance.
        
        Args:
            travel_time: Base travel time from the graph (hours)
            current_time: Current simulation time (hours)
            from_node: Source node (unused, kept for API compatibility)
            to_node: Destination node (unused, kept for API compatibility)

        Returns:
            Travel time with traffic variation applied (hours)
        """
        if not self.enable_traffic or travel_time <= 0:
            return travel_time
        
        # Calculate what fraction of the journey occurs during rush hours
        departure_time = current_time
        arrival_time = departure_time + travel_time
        rush_hour_fraction = self._calculate_rush_hour_fraction(departure_time, arrival_time)
        
        # Calculate standard deviation based on rush hour exposure
        # Interpolate between base std_dev and rush_hour std_dev
        base_std_dev = travel_time * self.std_dev_factor
        rush_std_dev = base_std_dev * self.rush_hour_multiplier
        std_dev = base_std_dev + rush_hour_fraction * (rush_std_dev - base_std_dev)

        # Cap the std_dev if max is specified
        if self.max_std_dev_hours > 0:
            std_dev = min(std_dev, self.max_std_dev_hours)

        # Sample from normal distribution N(mean=travel_time, std=std_dev)
        actual_travel_time = np.random.normal(loc=travel_time, scale=std_dev)

        # Ensure travel time is bounded (at least 85% of original, at most 250%)
        actual_travel_time = max(actual_travel_time, travel_time * 0.85)
        actual_travel_time = min(actual_travel_time, travel_time * 2.5)

        if self.verbose:
            variation_percent = ((actual_travel_time - travel_time) / travel_time) * 100
            if rush_hour_fraction > 0.5:
                rush_label = f" [RUSH HOUR {rush_hour_fraction*100:.0f}%]"
            elif rush_hour_fraction > 0:
                rush_label = f" [PARTIAL RUSH {rush_hour_fraction*100:.0f}%]"
            else:
                rush_label = ""
            print(
                f"    Traffic simulation{rush_label}: {travel_time:.2f}h → {actual_travel_time:.2f}h ({variation_percent:+.1f}%)"
            )

        return actual_travel_time
    
    def _calculate_rush_hour_fraction(self, start_time: float, end_time: float) -> float:
        """
        Calculate what fraction of a time interval overlaps with rush hours.
        
        Rush hours are 7-9am and 5-7pm (hours 7-9 and 17-19 in 24h format).
        
        Args:
            start_time: Journey start time (hours since simulation start)
            end_time: Journey end time (hours since simulation start)
            
        Returns:
            Fraction of journey during rush hours (0.0 to 1.0)
        """
        if end_time <= start_time:
            return 0.0
            
        journey_duration = end_time - start_time
        rush_hour_duration = 0.0
        
        # Sample the journey at regular intervals to check rush hour overlap
        # Use fine-grained sampling for accuracy (every 0.1 hours = 6 minutes)
        sample_interval = 0.1
        num_samples = max(1, int(journey_duration / sample_interval))
        
        for i in range(num_samples):
            sample_time = start_time + (i + 0.5) * (journey_duration / num_samples)
            hour_of_day = sample_time % 24
            
            # Check if this sample point is during rush hour
            if (7 <= hour_of_day <= 9) or (17 <= hour_of_day <= 19):
                rush_hour_duration += journey_duration / num_samples
        
        return rush_hour_duration / journey_duration
