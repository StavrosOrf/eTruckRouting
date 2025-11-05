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

from truck_env.utils import (
    get_graph,
    discharge_function as original_discharge_function,
    charge_function as original_charge_function,
)
from simple_truck_env.transportation_graph import TransportationGraph
from simple_truck_env.truck import Truck
from simple_truck_env.config_utils import load_config
from simple_truck_env.event_handlers import EventType, Event, EventHandler
from simple_truck_env.plotter import EnvironmentPlotter
from simple_truck_env.statistics import EnvironmentStatistics


class EventDrivenTruckEnv(gym.Env):
    """
    Event-driven truck routing environment with global clock.
    
    Single-agent paradigm: controls whichever truck is currently active.
    Time advances to the next event, and step() is called when a truck needs a decision.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        config: Optional[Union[str, Dict]] = None,
        num_trucks: Optional[int] = None,
        num_stops: Optional[int] = None,
        min_hop_distance: Optional[float] = None,
        max_hop_distance: Optional[float] = None,
        max_time: Optional[float] = None,
        verbose: Optional[bool] = None,
        enable_plotting: Optional[bool] = None,
        run_id: Optional[str] = None
    ):
        """
        Initialize the event-driven environment.
        
        Args:
            config: Path to config.yaml file or config dictionary
            num_trucks: Number of trucks (overrides config if provided)
            num_stops: Number of delivery stops per truck (overrides config)
            min_hop_distance: Minimum distance between delivery stops (overrides config)
            max_hop_distance: Maximum distance between delivery stops (overrides config)
            max_time: Maximum simulation time in hours (overrides config)
            verbose: Print detailed information (overrides config)
            enable_plotting: Enable plotting and statistics (default: False)
            run_id: Identifier for this run (used in output folder name)
        """
        super().__init__()
        
        # Load configuration
        if isinstance(config, str):
            from simple_truck_env.config_utils import load_config
            self.config = load_config(config)
        elif isinstance(config, dict):
            self.config = config
        elif config is None:
            from simple_truck_env.config_utils import load_config
            self.config = load_config()
        else:
            raise ValueError(f"config must be str, dict, or None, got {type(config)}")
        
        # Extract parameters from config
        env_config = self.config.get('environment', {})
        truck_config = self.config.get('truck', {})
        advanced_config = self.config.get('advanced', {})
        
        # Apply overrides if provided
        self.num_trucks = num_trucks if num_trucks is not None else advanced_config.get('num_trucks', 1)
        self.num_stops = num_stops if num_stops is not None else env_config.get('num_stops', 3)
        self.min_hop_distance = min_hop_distance if min_hop_distance is not None else env_config.get('min_hop_distance', 20.0)
        self.max_hop_distance = max_hop_distance if max_hop_distance is not None else env_config.get('max_hop_distance', 150.0)
        self.max_time = max_time if max_time is not None else env_config.get('max_time', 48.0)  # 48 hours default
        self.verbose = verbose if verbose is not None else env_config.get('verbose', False)
        
        # Visualization and output settings
        self.enable_plotting = enable_plotting if enable_plotting is not None else env_config.get('enable_plotting', False)
        self.run_id = run_id if run_id is not None else env_config.get('run_id', 'default')
        
        # Create output directory and helpers if plotting is enabled
        if self.enable_plotting:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'results',
                f"{self.run_id}_{timestamp}"
            )
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Initialize plotter and statistics collector
            self.plotter = EnvironmentPlotter(self.output_dir, self.verbose)
            self.stats_collector = EnvironmentStatistics(self.output_dir, self.verbose)
            
            if self.verbose:
                print(f"Plotting enabled. Output directory: {self.output_dir}")
        else:
            self.plotter = None
            self.stats_collector = None
        
        # Initialize event handler
        self.event_handler = EventHandler(self.verbose)
        
        # Extract reward and charging config
        self.reward_config = self.config.get('rewards', {})
        self.charging_config = self.config.get('charging', {})
        
        # Load graph and initialize transportation network
        graph = get_graph()
        self.transport_graph = TransportationGraph(graph)
        
        # Load waiting time lookup table for queue simulation
        waiting_time_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'simple_truck_env', 'data', 'waiting_time_lookup.json'
        )
        with open(waiting_time_path, 'r') as f:
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
        charge_durations = self.charging_config.get('charge_durations', [1, 2, 3, 4])
        self.num_navigation_actions = self.num_charging_nodes + 1  # Chargers + next delivery
        self.num_charge_actions = len(charge_durations)  # Charge for 1-4 hours
        
        # Discrete action space (single agent)
        self.action_space = spaces.Discrete(self.num_navigation_actions + self.num_charge_actions)
        
        # Define observation space - Box for current active truck + global state
        # Observation: [truck_state (10), global_time (1), active_trucks (1), events_pending (1)]
        self.observation_space = spaces.Box(
            low=np.array([
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
            ]),
            high=np.array([
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
            ]),
            dtype=np.float64
        )
        
        # Event-driven simulation state
        self.global_clock = 0.0  # Current simulation time
        self.event_queue = []  # Priority queue of events (min-heap)
        self.active_truck_id = None  # ID of truck that needs to make a decision
        
        # Charging station queue/occupancy tracking
        self.charger_occupancy = {node: [] for node in self.charging_nodes}  # List of truck IDs
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
                'total_charge_sessions': 0,
                'total_charge_time': 0.0,
                'total_trucks_served': set(),
                'occupancy_time': 0.0,  # Total time with at least one truck
                'last_update_time': 0.0,
            }
            for node in self.charging_nodes
        }
        
        # Current episode state
        self.trucks = []
        self.truck_states = {}  # truck_id -> "active", "routing", "charging", "complete", "failed"
        self.episode_reward = 0.0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
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
                'total_charge_sessions': 0,
                'total_charge_time': 0.0,
                'total_trucks_served': set(),
                'occupancy_time': 0.0,
                'last_update_time': 0.0,
            }
            for node in self.charging_nodes
        }
        
        # Track actual routes for visualization
        self.truck_routes = {}  # truck_id -> list of (node, time, event_type)
        self.truck_initial_plans = {}  # truck_id -> {'start': node, 'deliveries': [nodes]}
        
        # Create trucks with random delivery sequences
        self.trucks = []
        self.truck_states = {}
        for i in range(self.num_trucks):
            self._create_truck(i)
            self.truck_states[i] = "active"  # All trucks start active
            # Schedule initial TRUCK_READY event for each truck
            heapq.heappush(self.event_queue, Event(
                time=0.0,
                event_type=EventType.TRUCK_READY,
                truck_id=i,
                data={"reason": "initial"}
            ))
        
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
                print(f"    Battery: {truck.current_battery:.1f}/{truck.battery_capacity:.1f} kWh")
        
        # Generate initial route plot if plotting is enabled
        if self.enable_plotting and self.plotter:
            self.plotter.plot_initial_routes(
                self.transport_graph,
                self.truck_initial_plans,
                self.charging_nodes,
                self.num_trucks,
                self.num_stops
            )
        
        return obs, info
    
    def _create_truck(self, truck_id: int):
        """Create a new truck with random delivery sequence."""
        # Select random start node (avoid charging nodes as start)
        all_nodes = self.transport_graph.get_all_nodes()
        non_charging_nodes = [n for n in all_nodes if n not in self.charging_nodes]
        start_node = np.random.choice(non_charging_nodes)
        
        # Generate delivery sequence
        delivery_sequence = self.transport_graph.generate_delivery_sequence(
            start_node=start_node,
            num_stops=self.num_stops,
            min_hop_distance=self.min_hop_distance,
            max_hop_distance=self.max_hop_distance,
            exclude_charging_nodes=True
        )
        
        # Get truck specifications (single type)
        truck_config = self.config.get('truck', {})
        battery_capacity = truck_config.get('battery_capacity', 400.0)
        base_speed = truck_config.get('base_speed', 40.0)
        discharge_rate = truck_config.get('discharge_rate', 0.25)
        
        # Determine initial battery
        initial_battery_setting = truck_config.get('initial_battery', 'full')
        if initial_battery_setting == 'full':
            initial_battery = battery_capacity
        elif initial_battery_setting == 'random':
            initial_battery = np.random.uniform(0.3, 1.0) * battery_capacity
        elif isinstance(initial_battery_setting, (int, float)):
            initial_battery = (initial_battery_setting / 100.0) * battery_capacity
        else:
            initial_battery = battery_capacity
        
        # Create truck
        truck = Truck(
            truck_id=truck_id,
            truck_type="electric",  # Single truck type
            delivery_sequence=delivery_sequence,
            initial_battery=initial_battery,
            battery_capacity=battery_capacity,
            base_speed=base_speed,
            discharge_rate=discharge_rate
        )
        
        self.trucks.append(truck)
        
        # Store initial plan for visualization
        self.truck_routes[truck_id] = [(start_node, 0.0, 'start')]
        self.truck_initial_plans[truck_id] = {
            'start': start_node,
            'deliveries': delivery_sequence.copy()
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
            
            if self.verbose:
                print(f"\n[Clock: {self.global_clock:.2f}h] Processing {event}")
            
            # Process event
            if event.event_type == EventType.TRUCK_READY:
                # Truck needs a decision - stop here
                self.active_truck_id = event.truck_id
                return
            
            elif event.event_type == EventType.ROUTE_COMPLETE:
                self.event_handler.handle_route_complete(
                    event, self.trucks, self.truck_states, self.truck_routes,
                    self.event_queue, self.global_clock, self.enable_plotting
                )
            
            elif event.event_type == EventType.CHARGE_COMPLETE:
                self.event_handler.handle_charge_complete(
                    event, self.trucks, self.truck_states, self.charger_occupancy,
                    self.charger_stats, self.event_queue, self.global_clock
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
        info['reward'] = reward
        info['active_truck_id'] = self.active_truck_id
        info['global_clock'] = self.global_clock
        
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
        if hasattr(target_node, 'item'):
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
        if distance == float('inf'):
            if self.verbose:
                print(f"  ERROR: No path from {current_node} to {target_node}")
            return -10.0
        
        travel_time = distance / truck.base_speed
        discharge = distance * truck.discharge_rate
        
        # Check if truck can make it
        if discharge > truck.current_battery:
            if self.verbose:
                print(f"  ERROR: Insufficient battery ({truck.current_battery:.1f} kWh < {discharge:.1f} kWh needed)")
            # Truck will fail - schedule failure event
            truck.failed = True
            heapq.heappush(self.event_queue, Event(
                time=self.global_clock,
                event_type=EventType.TRUCK_TERMINATED,
                truck_id=truck.truck_id,
                data={"reason": "insufficient_battery"}
            ))
            return -50.0
        
        # Schedule route completion event
        completion_time = self.global_clock + travel_time
        heapq.heappush(self.event_queue, Event(
            time=completion_time,
            event_type=EventType.ROUTE_COMPLETE,
            truck_id=truck.truck_id,
            data={
                "destination": target_node,
                "distance": distance,
                "travel_time": travel_time,
                "discharge": discharge
            }
        ))
        
        # Update truck state
        self.truck_states[truck.truck_id] = "routing"
        
        if self.verbose:
            print(f"  Routing to node {target_node}")
            print(f"    Distance: {distance:.2f} km, Time: {travel_time:.2f}h")
            print(f"    Will arrive at t={completion_time:.2f}h")
        
        # Calculate reward
        time_penalty = -travel_time * self.reward_config.get('time_penalty', 1.0)
        distance_penalty = -distance * self.reward_config.get('distance_penalty', 0.1)
        
        # Bonus if this is a delivery
        if target_node == truck.get_next_delivery_target():
            delivery_bonus = self.reward_config.get('delivery_bonus', 50.0)
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
            available_capacities = sorted([int(c) for c in self.waiting_time_lookup[charger_type].keys()])
            closest_capacity = min(available_capacities, key=lambda x: abs(x - capacity))
            capacity_str = str(closest_capacity)
        
        # Round utilization to nearest 0.05
        util_rounded = round(current_utilization / 0.05) * 0.05
        util_rounded = max(0.05, min(0.95, util_rounded))  # Clamp to available range
        util_str = f"{util_rounded:.2f}"
        
        # Get waiting time in minutes and convert to hours
        waiting_minutes = self.waiting_time_lookup[charger_type][capacity_str].get(util_str, 0.0)
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
                print(f"  Charger utilization: {current_utilization*100:.1f}% - Expected wait: {wait_time:.2f}h")
        
        # Apply waiting time
        if wait_time > 0:
            truck.add_waiting_time(wait_time)
        
        # Add to charger occupancy
        self.charger_occupancy[charger_node].append(truck.truck_id)
        
        # Update utilization stats - track occupancy time
        stats = self.charger_stats[charger_node]
        if len(self.charger_occupancy[charger_node]) == 1:  # First truck at this charger
            # Add occupancy time from last update until now
            if stats['last_update_time'] > 0:
                stats['occupancy_time'] += (self.global_clock - stats['last_update_time'])
        stats['last_update_time'] = self.global_clock
        stats['total_charge_sessions'] += 1
        stats['total_trucks_served'].add(truck.truck_id)
        
        # Get charger type and determine charge rate
        charger_type = self.charger_type[charger_node]
        charging_config = self.config.get('charging', {})
        
        if charger_type == 'DCFast':
            charger_config = charging_config.get('dcfast', {})
            charge_rate = charger_config.get('charge_rate', 50.0)  # kW
            efficiency = charger_config.get('efficiency', 0.85)
        else:  # Level2
            charger_config = charging_config.get('level2', {})
            charge_rate = charger_config.get('charge_rate', 7.2)  # kW
            efficiency = charger_config.get('efficiency', 0.90)
        
        # Calculate charge amount (accounting for efficiency)
        charge_amount = min(
            charge_hours * charge_rate * efficiency,
            truck.battery_capacity - truck.current_battery
        )
        
        # Track total charge time at this station
        stats['total_charge_time'] += charge_hours
        
        # Schedule charge completion event
        completion_time = self.global_clock + charge_hours
        heapq.heappush(self.event_queue, Event(
            time=completion_time,
            event_type=EventType.CHARGE_COMPLETE,
            truck_id=truck.truck_id,
            data={
                "charge_amount": charge_amount,
                "charge_duration": charge_hours
            }
        ))
        
        # Update truck state
        self.truck_states[truck.truck_id] = "charging"
        truck.start_charging(self.global_clock)
        
        if self.verbose:
            print(f"  Charging for {charge_hours}h")
            print(f"    Will charge {charge_amount:.1f} kWh")
            print(f"    Will complete at t={completion_time:.2f}h")
        
        # Calculate reward (penalty for time spent charging)
        charge_penalty = -charge_hours * self.reward_config.get('charge_penalty', 2.0)
        return charge_penalty
    
    def _check_terminated(self) -> bool:
        """Check if episode is terminated (all trucks done)."""
        if self.active_truck_id is None:
            return True
        
        all_done = all(
            state in ["complete", "failed"]
            for state in self.truck_states.values()
        )
        return all_done
    
    def _check_truncated(self) -> bool:
        """Check if episode is truncated (time limit exceeded)."""
        return self.global_clock >= self.max_time
    
    def _get_observation(self) -> np.ndarray:
        """Get observation for the active truck."""
        if self.active_truck_id is None:
            # Return zeros if no active truck
            return np.zeros(self.observation_space.shape[0], dtype=np.float64)
        
        truck = self.trucks[self.active_truck_id]
        
        # Normalize node IDs
        max_node_id = float(self.transport_graph.num_nodes)
        current_node_norm = truck.current_node / max_node_id
        next_delivery = truck.get_next_delivery_target()
        next_delivery_norm = (next_delivery / max_node_id) if next_delivery is not None else 0.0
        
        # Find nearest charger
        nearest_charger_dist = min(
            self.transport_graph.get_distance(truck.current_node, charger)
            for charger in self.charging_nodes
        )
        
        # Check if can reach next delivery
        can_reach_next = 0.0
        if next_delivery is not None:
            dist_to_next = self.transport_graph.get_distance(truck.current_node, next_delivery)
            can_reach_next = 1.0 if truck.can_reach_node(next_delivery, dist_to_next) else 0.0
        
        # Count active trucks
        active_trucks = sum(1 for state in self.truck_states.values() if state not in ["complete", "failed"])
        
        # Count pending events
        events_pending = len(self.event_queue)
        
        obs = np.array([
            current_node_norm,
            next_delivery_norm,
            truck.current_battery,
            truck.get_battery_percentage(),
            float(truck.is_charging),
            float(len(truck.get_remaining_deliveries())),
            nearest_charger_dist,
            can_reach_next,
            truck.total_time_elapsed,
            truck.total_distance_traveled,
            self.global_clock,
            float(active_trucks),
            float(events_pending),
        ], dtype=np.float64)
        
        return obs
    
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
            self.global_clock
        )
        
        return {
            "global_clock": self.global_clock,
            "active_truck_id": self.active_truck_id,
            "episode_reward": self.episode_reward,
            "all_complete": all_complete,
            "any_failed": any_failed,
            "num_active_trucks": sum(1 for state in self.truck_states.values() if state not in ["complete", "failed"]),
            "events_pending": len(self.event_queue),
            "trucks": [truck.get_state_dict() for truck in self.trucks],
            "truck_states": self.truck_states.copy(),
            "charger_utilization": charger_utilization,
        }
    
    def _action_to_string(self, action: int) -> str:
        """Convert action to human-readable string."""
        if action < self.num_charging_nodes:
            node = self.charging_nodes[action]
            return f"Go to charger @ node {node}"
        elif action == self.num_charging_nodes:
            return "Go to next delivery"
        else:
            charge_idx = action - self.num_navigation_actions
            hours = charge_idx + 1
            return f"Charge for {hours}h"
    
    def close(self):
        """Clean up resources and generate final visualizations."""
        if self.enable_plotting and self.plotter and self.stats_collector:
            # Generate final plots
            self.plotter.plot_actual_routes(
                self.transport_graph,
                self.truck_routes,
                self.charging_nodes,
                self.num_trucks,
                self.global_clock
            )
            
            # Print and save statistics
            charger_util = EnvironmentStatistics.get_charger_utilization_stats(
                self.charging_nodes,
                self.charger_stats,
                self.charger_type,
                self.charger_capacity,
                self.charger_occupancy,
                self.global_clock
            )
            
            self.stats_collector.print_statistics(
                self.trucks,
                self.truck_states,
                self.truck_routes,
                charger_util,
                self.global_clock,
                self.num_trucks
            )

