"""
Traffic simulation module for electric vehicle routing environment.

Provides time-of-day dependent traffic modeling with rush hour effects.
"""

import math
from collections import defaultdict, deque

import numpy as np

from EVRoutingEnv.models.simulation.scenario import ScenarioRandomStreams


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
        enable_energy_uncertainty: bool = False,
        energy_uncertainty_factor: float = 0.10,
        min_energy_multiplier: float = 0.90,
        max_energy_multiplier: float = 1.20,
        verbose: bool = False,
        seed: int | None = None,
        random_streams: ScenarioRandomStreams | None = None,
    ):
        """
        Initialize the traffic simulator.

        Args:
            enable_traffic: Whether traffic simulation is enabled
            std_dev_factor: Standard deviation as fraction of mean travel time
            max_std_dev_hours: Maximum standard deviation cap (hours)
            rush_hour_multiplier: Multiplier for variance during rush hours
            enable_energy_uncertainty: Whether to apply energy consumption uncertainty
            energy_uncertainty_factor: Std dev as fraction of base energy
            min_energy_multiplier: Minimum energy multiplier (best case)
            max_energy_multiplier: Maximum energy multiplier (worst case)
            verbose: Whether to print detailed traffic information
            seed: Random seed for reproducible uncertainty (if None, uses system random)
        """
        if not math.isfinite(std_dev_factor) or std_dev_factor < 0.0:
            raise ValueError("std_dev_factor must be finite and non-negative")
        if not math.isfinite(max_std_dev_hours):
            raise ValueError("max_std_dev_hours must be finite")
        if not math.isfinite(rush_hour_multiplier) or rush_hour_multiplier < 1.0:
            raise ValueError("rush_hour_multiplier must be finite and at least 1")
        if (
            not math.isfinite(energy_uncertainty_factor)
            or energy_uncertainty_factor < 0.0
        ):
            raise ValueError(
                "energy_uncertainty_factor must be finite and non-negative"
            )
        if (
            not math.isfinite(min_energy_multiplier)
            or not math.isfinite(max_energy_multiplier)
            or not 0.0 < min_energy_multiplier <= 1.0 <= max_energy_multiplier
        ):
            raise ValueError(
                "energy multipliers must satisfy 0 < min <= 1 <= max"
            )

        self.enable_traffic = enable_traffic
        self.std_dev_factor = std_dev_factor
        self.max_std_dev_hours = max_std_dev_hours
        self.rush_hour_multiplier = rush_hour_multiplier
        self.enable_energy_uncertainty = enable_energy_uncertainty
        self.energy_uncertainty_factor = energy_uncertainty_factor
        self.min_energy_multiplier = min_energy_multiplier
        self.max_energy_multiplier = max_energy_multiplier
        self.verbose = verbose

        # Reproducible uncertainty system. Keyed samples are independent of
        # policy/algorithm RNG consumption and stable across Python processes.
        self.seed = (
            int(seed) if seed is not None else int(np.random.SeedSequence().entropy)
        )
        self.random_streams = random_streams or ScenarioRandomStreams(self.seed)
        self._uncertainty_cache: dict[
            tuple[int, int, int, int], tuple[float, float]
        ] = {}
        self._journey_counters: dict[tuple[int, int], int] = {}
        self._anonymous_counter = 0
        self._pending_energy: dict[tuple[int, int, int], deque[float]] = defaultdict(
            deque
        )

    def apply_traffic(
        self,
        travel_time: float,
        current_time: float,
        from_node: int | None = None,
        to_node: int | None = None,
    ) -> tuple[float, float]:
        """
        Apply traffic simulation to travel time using time-of-day dependent variance.

        Args:
            travel_time: Base travel time from the graph (hours)
            current_time: Current simulation time (hours)
            from_node: Source node (unused, kept for API compatibility)
            to_node: Destination node (unused, kept for API compatibility)

        Returns:
            Tuple of (actual_travel_time, traffic_multiplier) where traffic_multiplier
            can be used to correlate energy uncertainty
        """
        if not math.isfinite(travel_time) or travel_time < 0.0:
            raise ValueError("travel_time must be finite and non-negative")
        if not math.isfinite(current_time) or current_time < 0.0:
            raise ValueError("current_time must be finite and non-negative")
        if from_node is not None and int(from_node) < 0:
            raise ValueError("from_node must be non-negative")
        if to_node is not None and int(to_node) < 0:
            raise ValueError("to_node must be non-negative")
        if not self.enable_traffic or travel_time <= 0:
            return travel_time, 1.0

        # Calculate what fraction of the journey occurs during rush hours
        departure_time = current_time
        arrival_time = departure_time + travel_time
        rush_hour_fraction = self._calculate_rush_hour_fraction(
            departure_time, arrival_time
        )

        # Calculate standard deviation based on rush hour exposure
        # Interpolate between base std_dev and rush_hour std_dev
        base_std_dev = travel_time * self.std_dev_factor
        rush_std_dev = base_std_dev * self.rush_hour_multiplier
        std_dev = base_std_dev + rush_hour_fraction * (rush_std_dev - base_std_dev)

        # Cap the std_dev if max is specified
        if self.max_std_dev_hours > 0:
            std_dev = min(std_dev, self.max_std_dev_hours)

        # Get reproducible random values for this journey
        traffic_random, energy_random = self._get_uncertainty_values(
            from_node, to_node, current_time
        )
        if from_node is not None and to_node is not None:
            pending_key = self._base_event_key(from_node, to_node, current_time)
            self._pending_energy[pending_key].append(energy_random)

        # Sample from normal distribution N(mean=travel_time, std=std_dev)
        # Use pre-generated random value for reproducibility
        actual_travel_time = travel_time + std_dev * traffic_random

        # Ensure travel time is bounded (at least 85% of original, at most 250%)
        actual_travel_time = max(actual_travel_time, travel_time * 0.85)
        actual_travel_time = min(actual_travel_time, travel_time * 2.5)

        # Calculate traffic multiplier (how much worse than base)
        traffic_multiplier = actual_travel_time / travel_time

        if self.verbose:
            variation_percent = ((actual_travel_time - travel_time) / travel_time) * 100
            if rush_hour_fraction > 0.5:
                rush_label = f" [RUSH HOUR {rush_hour_fraction * 100:.0f}%]"
            elif rush_hour_fraction > 0:
                rush_label = f" [PARTIAL RUSH {rush_hour_fraction * 100:.0f}%]"
            else:
                rush_label = ""
            print(
                f"    Traffic simulation{rush_label}: {travel_time:.2f}h → {actual_travel_time:.2f}h ({variation_percent:+.1f}%) [multiplier: {traffic_multiplier:.3f}]"
            )

        return actual_travel_time, traffic_multiplier

    def apply_energy_uncertainty(
        self,
        base_energy: float,
        traffic_multiplier: float = 1.0,
        current_time: float | None = None,
        from_node: int | None = None,
        to_node: int | None = None,
    ) -> float:
        """
        Apply energy consumption uncertainty correlated with traffic conditions.

        Traffic conditions (stop-and-go, speed variations) affect energy efficiency.
        Worse traffic (higher traffic_multiplier) = more inefficient driving = higher energy consumption.

        Args:
            base_energy: Base energy consumption from the graph (kWh)
            traffic_multiplier: Traffic delay multiplier from apply_traffic (>=1.0)
            current_time: Current simulation time (hours, optional)
            from_node: Source node (unused, kept for API compatibility)
            to_node: Destination node (unused, kept for API compatibility)

        Returns:
            Energy consumption with traffic-induced variation applied (kWh)
        """
        if not math.isfinite(base_energy) or base_energy < 0.0:
            raise ValueError("base_energy must be finite and non-negative")
        if not math.isfinite(traffic_multiplier) or traffic_multiplier <= 0.0:
            raise ValueError("traffic_multiplier must be finite and positive")
        if current_time is not None and (
            not math.isfinite(current_time) or current_time < 0.0
        ):
            raise ValueError("current_time must be finite and non-negative")
        if from_node is not None and int(from_node) < 0:
            raise ValueError("from_node must be non-negative")
        if to_node is not None and int(to_node) < 0:
            raise ValueError("to_node must be non-negative")
        if (
            not self.enable_traffic
            or not self.enable_energy_uncertainty
            or base_energy <= 0
        ):
            return base_energy

        # Correlation: Worse traffic (traffic_multiplier > 1) increases energy consumption
        # Map traffic_multiplier [0.85, 2.5] to energy_bias
        # traffic_multiplier = 1.0 (no delay) -> bias = 0 (neutral)
        # traffic_multiplier > 1.0 (delay) -> bias > 0 (higher energy)
        # traffic_multiplier < 1.0 (faster) -> bias < 0 (lower energy)
        # Use stronger correlation: bias is proportional to traffic deviation
        traffic_deviation = (
            traffic_multiplier - 1.0
        )  # How much worse/better than normal

        # Calculate standard deviation for energy
        std_dev = base_energy * self.energy_uncertainty_factor

        # Sample from normal distribution with traffic-correlated mean shift
        # Mean shifts proportionally with traffic (50% of traffic delay affects energy)
        biased_mean = base_energy * (1.0 + traffic_deviation * 0.5)

        # Consume the energy draw paired with the immediately preceding traffic
        # sample for this traversal. If traffic was disabled/not called, create a
        # new keyed traversal draw.
        if from_node is not None and to_node is not None:
            pending_key = self._base_event_key(from_node, to_node, current_time)
            pending_values = self._pending_energy.get(pending_key)
            if pending_values:
                energy_random = pending_values.popleft()
                if not pending_values:
                    self._pending_energy.pop(pending_key, None)
            else:
                _, energy_random = self._get_uncertainty_values(
                    from_node, to_node, current_time
                )
        else:
            anonymous_key = ("anonymous_energy", self._anonymous_counter)
            self._anonymous_counter += 1
            energy_random = self.random_streams.standard_normal("energy", anonymous_key)

        actual_energy = biased_mean + std_dev * energy_random

        # Apply bounds to keep realistic
        actual_energy = max(actual_energy, base_energy * self.min_energy_multiplier)
        actual_energy = min(actual_energy, base_energy * self.max_energy_multiplier)

        if self.verbose:
            variation_percent = ((actual_energy - base_energy) / base_energy) * 100
            print(
                f"    Energy uncertainty (correlated w/ traffic {traffic_multiplier:.3f}): {base_energy:.2f} kWh → {actual_energy:.2f} kWh ({variation_percent:+.1f}%)"
            )

        return actual_energy

    def _calculate_rush_hour_fraction(
        self, start_time: float, end_time: float
    ) -> float:
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

    @staticmethod
    def _base_event_key(
        from_node: int,
        to_node: int,
        current_time: float | None,
    ) -> tuple[int, int, int]:
        time_value = 0.0 if current_time is None else float(current_time)
        return (int(from_node), int(to_node), int(time_value / 0.5))

    def _get_uncertainty_values(
        self,
        from_node: int | None,
        to_node: int | None,
        current_time: float | None,
    ) -> tuple[float, float]:
        """
        Get reproducible random values for traffic and energy uncertainty.

        Uses a seeded RNG and caching to ensure the same journey at the same time
        with the same seed produces identical uncertainty values across different algorithms.

        Args:
            from_node: Source node ID
            to_node: Destination node ID
            current_time: Current simulation time (hours)

        Returns:
            Tuple of (traffic_random, energy_random) - standard normal samples
        """
        if from_node is None or to_node is None:
            event_key = ("anonymous_traversal", self._anonymous_counter)
            self._anonymous_counter += 1
            return (
                self.random_streams.standard_normal("travel_time", event_key),
                self.random_streams.standard_normal("energy", event_key),
            )

        # Create cache key: (from, to, time_bucket)
        # Use 0.5-hour buckets for time to balance granularity vs cache size
        from_node, to_node, time_bucket = self._base_event_key(
            from_node, to_node, current_time
        )

        # Track journey count for this edge to handle multiple journeys at same time
        edge_key = (from_node, to_node)
        if edge_key not in self._journey_counters:
            self._journey_counters[edge_key] = 0

        journey_idx = self._journey_counters[edge_key]
        cache_key = (from_node, to_node, time_bucket, journey_idx)

        # Increment counter for next journey on this edge
        self._journey_counters[edge_key] += 1

        # Check cache
        if cache_key in self._uncertainty_cache:
            return self._uncertainty_cache[cache_key]

        traffic_random = self.random_streams.standard_normal("travel_time", cache_key)
        energy_random = self.random_streams.standard_normal("energy", cache_key)

        # Cache for future use
        self._uncertainty_cache[cache_key] = (traffic_random, energy_random)

        return traffic_random, energy_random

    def reset_scenario(
        self,
        seed: int,
        random_streams: ScenarioRandomStreams | None = None,
    ) -> None:
        """Reset all traversal state for a new episode scenario."""
        self.seed = int(seed)
        self.random_streams = random_streams or ScenarioRandomStreams(self.seed)
        self.clear_cache()

    def reset_journey_counters(self):
        """
        Reset journey counters for a new episode.

        Call this at the start of each episode to ensure consistent
        uncertainty across episodes with the same seed.
        """
        self._journey_counters.clear()
        self._pending_energy.clear()
        self._anonymous_counter = 0

    def clear_cache(self):
        """
        Clear the uncertainty cache.

        Useful for freeing memory if running many episodes.
        """
        self._uncertainty_cache.clear()
        self._journey_counters.clear()
        self._pending_energy.clear()
        self._anonymous_counter = 0
