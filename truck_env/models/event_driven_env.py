"""
Event-Driven Truck Routing Environment - Single-agent controlling active truck.

Uses a global clock and event queue. Each truck generates events (route_complete, charge_complete)
and the environment steps forward when events finish. Only one truck is active at a time.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import heapq
import sys
import os
import json
import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truck_env.utils.utils import get_graph, load_config
from truck_env.models.transportation_graph import TransportationGraph
from truck_env.models.truck import Truck
from truck_env.models.event_handlers import EventType, Event, EventHandler
from truck_env.models.loaders import create_truck
from truck_env.models.observations import get_observation, action_to_string
from truck_env.utils.plotter import EnvironmentPlotter
from truck_env.utils.statistics import EnvironmentStatistics


class EventDrivenTruckEnv(gym.Env):
    """
    Event-driven truck routing environment with global clock.

    Single-agent paradigm: controls whichever truck is currently active.
    Time advances to the next event, and step() is called when a truck needs a decision.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Union[str, Dict],
        verbose: Optional[bool] = None,
        enable_plotting: Optional[bool] = None,
        run_id: Optional[str] = None,
    ):
        """
        Initialize the event-driven environment.

        Args:
            config: Path to config.yaml file or config dictionary (required)
            verbose: Print detailed information (overrides config if provided)
            enable_plotting: Enable plotting and statistics (overrides config if provided)
        """
        super().__init__()

        self.config = load_config(config)

        # Extract parameters from config (all required, no defaults)
        env_config = self.config["environment"]
        advanced_config = self.config["advanced"]

        # Load all parameters from config (no overrides, no defaults)
        self.num_trucks = advanced_config["num_trucks"]
        self.num_stops = env_config["num_stops"]
        self.min_hop_distance = env_config["min_hop_distance"]
        self.max_hop_distance = env_config["max_hop_distance"]
        self.max_time = env_config["max_time"]
        self.verbose = verbose if verbose is not None else env_config["verbose"]

        # Visualization and output settings
        self.enable_plotting = enable_plotting

        # Generate unique run_id based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}" if run_id is None else run_id

        # Create output directory and helpers if plotting is enabled
        if self.enable_plotting:
            self.output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "../results", self.run_id
            )
            os.makedirs(self.output_dir, exist_ok=True)

            # Initialize plotter and statistics collector
            self.plotter = EnvironmentPlotter(
                self.output_dir, self.verbose, use_osm=False
            )
            self.stats_collector = EnvironmentStatistics(self.output_dir, self.verbose)

            if self.verbose:
                print(f"Plotting enabled. Output directory: {self.output_dir}")
        else:
            self.plotter = None
            self.stats_collector = None

        # Initialize event handler
        self.event_handler = EventHandler(self.verbose)

        # Extract reward and charging config (require them to exist)
        self.reward_config = self.config["rewards"]
        self.charging_config = self.config["charging"]

        # Load graph and initialize transportation network
        graph = get_graph()
        self.transport_graph = TransportationGraph(graph)

        # Load waiting time lookup table for queue simulation
        waiting_time_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "waiting_time_lookup.json",
        )
        with open(waiting_time_path, "r") as f:
            self.waiting_time_lookup = json.load(f)

        # Get charging nodes
        self.charging_nodes = self.transport_graph.get_charging_nodes()
        self.num_charging_nodes = len(self.charging_nodes)

        if self.verbose:
            print(f"Event-Driven Environment initialized:")
            print(f"  - Total nodes: {self.transport_graph.num_nodes}")
            print(f"  - Charging nodes: {self.num_charging_nodes}")
            print(f"  - Number of trucks: {self.num_trucks}")
            print(f"  - Max simulation time: {self.max_time} hours")

        # Define action space - Discrete for single active truck
        # Actions: [chargers (0 to num_charging_nodes-1), next_delivery (num_charging_nodes),
        #           charge_1h, charge_2h, charge_3h, charge_4h]
        charge_durations = self.charging_config["charge_durations"]
        self.num_navigation_actions = (
            self.num_charging_nodes + 1
        )  # Chargers + next delivery
        self.num_charge_actions = len(charge_durations)  # Charge for 1-4 hours

        # Discrete action space (single agent)
        self.action_space = spaces.Discrete(
            self.num_navigation_actions + self.num_charge_actions
        )

        # Define observation space - Box for current active truck + global state
        # Observation: [truck_state (10), global_time (1), active_trucks (1), events_pending (1)]
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    0.0,  # current_node (normalized)
                    0.0,  # next_delivery_node (normalized)
                    0.0,  # battery_level
                    0.0,  # battery_percentage
                    0.0,  # is_charging
                    0.0,  # deliveries_remaining
                    0.0,  # nearest_charger_distance
                    0.0,  # can_reach_next_delivery
                    0.0,  # time_elapsed (truck)
                    0.0,  # distance_traveled
                    0.0,  # global_time
                    0.0,  # active_trucks
                    0.0,  # events_pending
                ]
            ),
            high=np.array(
                [
                    1.0,  # current_node (normalized)
                    1.0,  # next_delivery_node (normalized)
                    500.0,  # battery_level (kWh)
                    100.0,  # battery_percentage
                    1.0,  # is_charging
                    float(self.num_stops),  # deliveries_remaining
                    1000.0,  # nearest_charger_distance (km)
                    1.0,  # can_reach_next_delivery
                    1000.0,  # time_elapsed (hours)
                    5000.0,  # distance_traveled (km)
                    self.max_time,  # global_time
                    float(self.num_trucks),  # active_trucks
                    100.0,  # events_pending
                ]
            ),
            dtype=np.float64,
        )

        # Event-driven simulation state
        self.global_clock = 0.0  # Current simulation time
        self.event_queue = []  # Priority queue of events (min-heap)
        self.active_truck_id = None  # ID of truck that needs to make a decision

        # Charging station queue/occupancy tracking
        self.charger_occupancy = {
            node: [] for node in self.charging_nodes
        }  # List of truck IDs
        self.charger_capacity = {
            node: self.transport_graph.get_charger_capacity(node)
            for node in self.charging_nodes
        }
        self.charger_type = {
            node: self.transport_graph.get_charger_type(node)
            for node in self.charging_nodes
        }

        # Charging station utilization tracking
        self.charger_stats = {
            node: {
                "total_charge_sessions": 0,
                "total_charge_time": 0.0,
                "total_trucks_served": set(),
                "occupancy_time": 0.0,  # Total time with at least one truck
                "last_update_time": 0.0,
            }
            for node in self.charging_nodes
        }

        # Current episode state
        self.trucks = []
        self.truck_states = (
            {}
        )  # truck_id -> "active", "routing", "charging", "complete", "failed"
        self.episode_reward = 0.0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Reset simulation time and event queue
        self.global_clock = 0.0
        self.event_queue = []
        self.episode_reward = 0.0

        # Reset charger occupancy and statistics
        self.charger_occupancy = {node: [] for node in self.charging_nodes}
        self.charger_stats = {
            node: {
                "total_charge_sessions": 0,
                "total_charge_time": 0.0,
                "total_trucks_served": set(),
                "occupancy_time": 0.0,
                "last_update_time": 0.0,
            }
            for node in self.charging_nodes
        }

        # Track actual routes for visualization
        self.truck_routes = {}  # truck_id -> list of (node, time, event_type)
        self.truck_initial_plans = (
            {}
        )  # truck_id -> {'start': node, 'deliveries': [nodes]}

        # Create trucks with random delivery sequences
        self.trucks = []
        self.truck_states = {}
        for i in range(self.num_trucks):
            self._create_truck(i)
            self.truck_states[i] = "active"  # All trucks start active
            # Schedule initial TRUCK_READY event for each truck
            heapq.heappush(
                self.event_queue,
                Event(
                    time=0.0,
                    event_type=EventType.TRUCK_READY,
                    truck_id=i,
                    data={"reason": "initial"},
                ),
            )

        # Process event queue until we find a truck that needs a decision
        self._advance_to_next_decision()

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"NEW EPISODE - {self.num_trucks} Trucks (Event-Driven)")
            print(f"{'='*80}")
            print(f"Global Clock: {self.global_clock:.2f} hours")
            print(f"Active Truck: {self.active_truck_id}")
            for truck in self.trucks:
                print(f"\n  Truck {truck.truck_id}:")
                print(f"    Delivery sequence: {truck.delivery_sequence}")
                print(
                    f"    Battery: {truck.current_battery:.1f}/{truck.battery_capacity:.1f} kWh"
                )

        # Generate initial route plot if plotting is enabled
        if self.enable_plotting and self.plotter:
            self.plotter.plot_initial_state(
                self.transport_graph,
                self.truck_initial_plans,
                self.charging_nodes,
                self.num_trucks,
            )

        return obs, info

    def _create_truck(self, truck_id: int):
        """Create a new truck with random delivery sequence."""
        truck, delivery_sequence, start_node = create_truck(
            truck_id=truck_id,
            transport_graph=self.transport_graph,
            config=self.config,
            num_stops=self.num_stops,
            min_hop_distance=self.min_hop_distance,
            max_hop_distance=self.max_hop_distance,
            charging_nodes=self.charging_nodes,
        )

        self.trucks.append(truck)

        # Store initial plan for visualization
        self.truck_routes[truck_id] = [(start_node, 0.0, "start")]
        self.truck_initial_plans[truck_id] = {
            "start": start_node,
            "deliveries": delivery_sequence.copy(),
        }

    def _advance_to_next_decision(self):
        """
        Advance simulation clock to next event that requires a decision.
        Process events until we find a TRUCK_READY event.
        """
        while self.event_queue:
            # Get next event
            event = heapq.heappop(self.event_queue)

            # Advance clock
            self.global_clock = event.time

            # Process event
            if event.event_type == EventType.TRUCK_READY:
                # Truck needs a decision - stop here
                self.active_truck_id = event.truck_id
                return

            elif event.event_type == EventType.ROUTE_COMPLETE:
                self.event_handler.handle_route_complete(
                    event,
                    self.trucks,
                    self.truck_states,
                    self.truck_routes,
                    self.event_queue,
                    self.global_clock,
                    self.enable_plotting,
                )

            elif event.event_type == EventType.CHARGE_COMPLETE:
                self.event_handler.handle_charge_complete(
                    event,
                    self.trucks,
                    self.truck_states,
                    self.charger_occupancy,
                    self.charger_stats,
                    self.event_queue,
                    self.global_clock,
                )

            elif event.event_type == EventType.TRUCK_TERMINATED:
                self.event_handler.handle_truck_terminated(event, self.trucks)

        # No more events - episode is over
        self.active_truck_id = None

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step for the active truck.

        Args:
            action: Discrete action for the active truck
                    - 0 to num_charging_nodes-1: Go to charging station
                    - num_charging_nodes: Go to next delivery
                    - num_charging_nodes+1 to end: Charge for 1-4 hours at current location

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.active_truck_id is None:
            # No active truck - episode is over
            return self._get_observation(), 0.0, True, False, self._get_info()

        truck = self.trucks[self.active_truck_id]
        reward = 0.0

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"STEP at t={self.global_clock:.2f}h - Truck {self.active_truck_id}")
            print(f"Action: {self._action_to_string(action)}")
            print(f"{'='*80}")

        # Decode and execute action
        if action < self.num_navigation_actions:
            # Navigation action
            reward = self._execute_navigation_action(truck, action)
        else:
            # Charging action
            charge_idx = action - self.num_navigation_actions
            charge_hours = charge_idx + 1
            reward = self._execute_charge_action(truck, charge_hours)

        # Accumulate reward
        self.episode_reward += reward

        # Advance to next decision point
        self._advance_to_next_decision()

        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self._check_truncated()

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info["reward"] = reward
        info["active_truck_id"] = self.active_truck_id
        info["global_clock"] = self.global_clock

        return obs, reward, terminated, truncated, info

    def _execute_navigation_action(self, truck: Truck, action: int) -> float:
        """Execute navigation action and schedule route completion event."""
        if action < self.num_charging_nodes:
            # Go to charging station
            target_node = self.charging_nodes[action]
        elif action == self.num_charging_nodes:
            # Go to next delivery
            target_node = truck.get_next_delivery_target()
            if target_node is None:
                # No more deliveries - penalize
                if self.verbose:
                    print(f"  ERROR: No more deliveries for truck {truck.truck_id}")
                return -10.0
        else:
            # Invalid action
            return -10.0

        # Convert numpy types to native Python int
        if hasattr(target_node, "item"):
            target_node = int(target_node.item())
        else:
            target_node = int(target_node)

        current_node = int(truck.current_node)

        # Check if already at target
        if current_node == target_node:
            if self.verbose:
                print(f"  Already at node {target_node}")
            return -1.0  # Small penalty for redundant action

        # Calculate distance directly
        distance = self.transport_graph.get_distance(current_node, target_node)
        if distance == float("inf"):
            if self.verbose:
                print(f"  ERROR: No path from {current_node} to {target_node}")
            return -10.0

        # Get the full path for visualization
        path = self.transport_graph.get_shortest_path(current_node, target_node)

        travel_time = distance / truck.base_speed
        discharge = distance * truck.discharge_rate

        # Check if truck can make it
        if discharge > truck.current_battery:
            if self.verbose:
                print(
                    f"  ERROR: Insufficient battery ({truck.current_battery:.1f} kWh < {discharge:.1f} kWh needed)"
                )
            # Truck will fail - schedule failure event
            truck.failed = True
            heapq.heappush(
                self.event_queue,
                Event(
                    time=self.global_clock,
                    event_type=EventType.TRUCK_TERMINATED,
                    truck_id=truck.truck_id,
                    data={"reason": "insufficient_battery"},
                ),
            )
            return -50.0

        # Schedule route completion event
        completion_time = self.global_clock + travel_time
        heapq.heappush(
            self.event_queue,
            Event(
                time=completion_time,
                event_type=EventType.ROUTE_COMPLETE,
                truck_id=truck.truck_id,
                data={
                    "destination": target_node,
                    "distance": distance,
                    "travel_time": travel_time,
                    "discharge": discharge,
                    "path": path,  # Full path for visualization
                },
            ),
        )

        # Update truck state
        self.truck_states[truck.truck_id] = "routing"

        if self.verbose:
            print(f"  Routing to node {target_node}")
            print(f"    Distance: {distance:.2f} km, Time: {travel_time:.2f}h")
            print(f"    Will arrive at t={completion_time:.2f}h")

        # Calculate reward
        time_penalty = -travel_time * self.reward_config["time_penalty"]
        distance_penalty = -distance * self.reward_config["distance_penalty"]

        # Bonus if this is a delivery
        if target_node == truck.get_next_delivery_target():
            delivery_bonus = self.reward_config["delivery_bonus"]
            return time_penalty + distance_penalty + delivery_bonus

        return time_penalty + distance_penalty

    def _get_waiting_time(self, charger_node: int, current_utilization: float) -> float:
        """
        Get expected waiting time at a charger based on current utilization.

        Args:
            charger_node: The charging station node
            current_utilization: Current utilization rate (0-1)

        Returns:
            Expected waiting time in hours
        """
        charger_type = self.charger_type[charger_node]
        capacity = int(self.charger_capacity[charger_node])

        # Get lookup table for this charger type and capacity
        if charger_type not in self.waiting_time_lookup:
            return 0.0

        capacity_str = str(capacity)
        if capacity_str not in self.waiting_time_lookup[charger_type]:
            # Use closest available capacity
            available_capacities = sorted(
                [int(c) for c in self.waiting_time_lookup[charger_type].keys()]
            )
            closest_capacity = min(
                available_capacities, key=lambda x: abs(x - capacity)
            )
            capacity_str = str(closest_capacity)

        # Round utilization to nearest 0.05
        util_rounded = round(current_utilization / 0.05) * 0.05
        util_rounded = max(0.05, min(0.95, util_rounded))  # Clamp to available range
        util_str = f"{util_rounded:.2f}"

        # Get waiting time in minutes and convert to hours
        waiting_minutes = self.waiting_time_lookup[charger_type][capacity_str].get(
            util_str, 0.0
        )
        waiting_hours = waiting_minutes / 60.0

        return waiting_hours

    def _execute_charge_action(self, truck: Truck, charge_hours: int) -> float:
        """Execute charging action and schedule charge completion event."""
        # Check if at a charging station
        if truck.current_node not in self.charging_nodes:
            if self.verbose:
                print(f"  ERROR: Truck {truck.truck_id} not at charging station")
            return -10.0

        # Check charger availability and calculate waiting time
        charger_node = truck.current_node
        current_occupancy = len(self.charger_occupancy[charger_node])
        capacity = self.charger_capacity[charger_node]

        # Calculate current utilization (based on occupancy)
        current_utilization = current_occupancy / capacity if capacity > 0 else 0.0

        # Get waiting time based on current utilization
        wait_time = 0.0
        if current_occupancy >= capacity:
            # Charger at capacity - use maximum utilization
            wait_time = self._get_waiting_time(charger_node, 0.95)
            if self.verbose:
                print(f"  Charger at capacity! Expected wait: {wait_time:.2f}h")
        elif current_occupancy > 0:
            # Charger partially occupied - use actual utilization
            wait_time = self._get_waiting_time(charger_node, current_utilization)
            if self.verbose and wait_time > 0.01:
                print(
                    f"  Charger utilization: {current_utilization*100:.1f}% - Expected wait: {wait_time:.2f}h"
                )

        # Apply waiting time
        if wait_time > 0:
            truck.add_waiting_time(wait_time)

        # Add to charger occupancy
        self.charger_occupancy[charger_node].append(truck.truck_id)

        # Update utilization stats - track occupancy time
        stats = self.charger_stats[charger_node]
        if (
            len(self.charger_occupancy[charger_node]) == 1
        ):  # First truck at this charger
            # Add occupancy time from last update until now
            if stats["last_update_time"] > 0:
                stats["occupancy_time"] += self.global_clock - stats["last_update_time"]
        stats["last_update_time"] = self.global_clock
        stats["total_charge_sessions"] += 1
        stats["total_trucks_served"].add(truck.truck_id)

        # Get charger type and determine charge rate
        charger_type = self.charger_type[charger_node]
        charging_config = self.config["charging"]

        if charger_type == "DCFast":
            charger_config = charging_config["dcfast"]
            charge_rate = charger_config["charge_rate"]  # kW
            efficiency = charger_config["efficiency"]
        else:  # Level2
            charger_config = charging_config["level2"]
            charge_rate = charger_config["charge_rate"]  # kW
            efficiency = charger_config["efficiency"]

        # Calculate charge amount (accounting for efficiency)
        charge_amount = min(
            charge_hours * charge_rate * efficiency,
            truck.battery_capacity - truck.current_battery,
        )

        # Track total charge time at this station
        stats["total_charge_time"] += charge_hours

        # Schedule charge completion event
        completion_time = self.global_clock + charge_hours
        heapq.heappush(
            self.event_queue,
            Event(
                time=completion_time,
                event_type=EventType.CHARGE_COMPLETE,
                truck_id=truck.truck_id,
                data={"charge_amount": charge_amount, "charge_duration": charge_hours},
            ),
        )

        # Update truck state
        self.truck_states[truck.truck_id] = "charging"
        truck.start_charging(self.global_clock)

        if self.verbose:
            print(f"  Charging for {charge_hours}h")
            print(f"    Will charge {charge_amount:.1f} kWh")
            print(f"    Will complete at t={completion_time:.2f}h")

        # Calculate reward (penalty for time spent charging)
        charge_penalty = -charge_hours * self.reward_config["charge_penalty"]
        return charge_penalty

    def _check_terminated(self) -> bool:
        """Check if episode is terminated (all trucks done)."""
        if self.active_truck_id is None:
            return True

        all_done = all(
            state in ["complete", "failed"] for state in self.truck_states.values()
        )
        return all_done

    def _check_truncated(self) -> bool:
        """Check if episode is truncated (time limit exceeded)."""
        return self.global_clock >= self.max_time

    def _get_observation(self) -> np.ndarray:
        """Get observation for the active truck."""
        return get_observation(
            trucks=self.trucks,
            active_truck_id=self.active_truck_id,
            transport_graph=self.transport_graph,
            charging_nodes=self.charging_nodes,
            truck_states=self.truck_states,
            event_queue=self.event_queue,
            observation_space_shape=self.observation_space.shape,
            global_clock=self.global_clock,
        )

    def _get_info(self) -> Dict:
        """Get info dictionary."""
        all_complete = all(truck.is_complete for truck in self.trucks)
        any_failed = any(truck.failed for truck in self.trucks)

        # Calculate charger utilization statistics
        charger_utilization = EnvironmentStatistics.get_charger_utilization_stats(
            self.charging_nodes,
            self.charger_stats,
            self.charger_type,
            self.charger_capacity,
            self.charger_occupancy,
            self.global_clock,
        )

        return {
            "global_clock": self.global_clock,
            "active_truck_id": self.active_truck_id,
            "episode_reward": self.episode_reward,
            "all_complete": all_complete,
            "any_failed": any_failed,
            "num_active_trucks": sum(
                1
                for state in self.truck_states.values()
                if state not in ["complete", "failed"]
            ),
            "events_pending": len(self.event_queue),
            "trucks": [truck.get_state_dict() for truck in self.trucks],
            "truck_states": self.truck_states.copy(),
            "charger_utilization": charger_utilization,
        }

    def _action_to_string(self, action: int) -> str:
        """Convert action to human-readable string."""
        return action_to_string(
            action=action,
            num_charging_nodes=self.num_charging_nodes,
            num_navigation_actions=self.num_navigation_actions,
            charging_nodes=self.charging_nodes,
        )

    def close(self):
        """Clean up resources and generate final visualizations."""
        if self.enable_plotting and self.plotter and self.stats_collector:
            # Generate final plots
            self.plotter.plot_final_routes(
                self.transport_graph,
                self.truck_routes,
                self.charging_nodes,
                self.num_trucks,
                self.global_clock,
                self.truck_initial_plans,
            )

            # Print and save statistics
            charger_util = EnvironmentStatistics.get_charger_utilization_stats(
                self.charging_nodes,
                self.charger_stats,
                self.charger_type,
                self.charger_capacity,
                self.charger_occupancy,
                self.global_clock,
            )

            self.stats_collector.print_statistics(
                self.trucks,
                self.truck_states,
                self.truck_routes,
                charger_util,
                self.global_clock,
                self.num_trucks,
            )
