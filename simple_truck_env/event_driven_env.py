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
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truck_env.utils import (
    get_graph,
    get_truck_types,
    discharge_function as original_discharge_function,
    charge_function as original_charge_function,
)
from simple_truck_env.transportation_graph import TransportationGraph
from simple_truck_env.truck import Truck
from simple_truck_env.config_utils import load_config, get_env_config


class EventType(Enum):
    """Types of events in the simulation."""
    TRUCK_READY = "truck_ready"  # Truck is ready to take an action
    ROUTE_COMPLETE = "route_complete"  # Truck completed routing to a node
    CHARGE_COMPLETE = "charge_complete"  # Truck completed charging
    TRUCK_TERMINATED = "truck_terminated"  # Truck finished or failed


@dataclass(order=True)
class Event:
    """Represents a simulation event."""
    time: float  # When the event occurs
    event_type: EventType = field(compare=False)
    truck_id: int = field(compare=False)
    data: Dict = field(default_factory=dict, compare=False)
    
    def __repr__(self):
        return f"Event(time={self.time:.2f}, type={self.event_type.value}, truck={self.truck_id})"


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
        verbose: Optional[bool] = None
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
        
        # Extract reward and charging config
        self.reward_config = self.config.get('rewards', {})
        self.charging_config = self.config.get('charging', {})
        
        # Load graph and initialize transportation network
        graph = get_graph()
        self.transport_graph = TransportationGraph(graph)
        self.truck_types_config = get_truck_types()
        
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
            node: sum(self.transport_graph.get_charger_info(node).values())
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
        
        # Reset charger occupancy
        self.charger_occupancy = {node: [] for node in self.charging_nodes}
        
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
        
        # Select truck type
        truck_config = self.config.get('truck', {})
        type_selection = truck_config.get('type_selection', 'random')
        
        if type_selection == 'random':
            truck_type = np.random.choice(["standard", "heavy"])
        else:
            truck_type = type_selection
        
        truck_spec = self.truck_types_config[truck_type]
        
        # Determine initial battery
        initial_battery_setting = truck_config.get('initial_battery', 'full')
        if initial_battery_setting == 'full':
            initial_battery = truck_spec["battery_capacity"]
        elif initial_battery_setting == 'random':
            initial_battery = np.random.uniform(0.3, 1.0) * truck_spec["battery_capacity"]
        elif isinstance(initial_battery_setting, (int, float)):
            initial_battery = (initial_battery_setting / 100.0) * truck_spec["battery_capacity"]
        else:
            initial_battery = truck_spec["battery_capacity"]
        
        # Create truck
        truck = Truck(
            truck_id=truck_id,
            truck_type=truck_type,
            delivery_sequence=delivery_sequence,
            initial_battery=initial_battery,
            battery_capacity=truck_spec["battery_capacity"],
            base_speed=truck_spec["base_speed"],
            discharge_rate=truck_spec.get("discharge_rate", 0.2)
        )
        
        self.trucks.append(truck)
    
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
                self._handle_route_complete(event)
            
            elif event.event_type == EventType.CHARGE_COMPLETE:
                self._handle_charge_complete(event)
            
            elif event.event_type == EventType.TRUCK_TERMINATED:
                self._handle_truck_terminated(event)
        
        # No more events - episode is over
        self.active_truck_id = None
    
    def _handle_route_complete(self, event: Event):
        """Handle completion of routing to a node."""
        truck = self.trucks[event.truck_id]
        data = event.data
        
        # Update truck position and state
        truck.move_to_node(
            node=data['destination'],
            distance=data['distance'],
            travel_time=data['travel_time'],
            discharge=data['discharge']
        )
        
        if self.verbose:
            print(f"  Truck {truck.truck_id} arrived at node {truck.current_node}")
            print(f"    Battery: {truck.current_battery:.1f} kWh ({truck.get_battery_percentage():.1f}%)")
        
        # Check if truck failed
        if truck.failed:
            self.truck_states[truck.truck_id] = "failed"
            heapq.heappush(self.event_queue, Event(
                time=self.global_clock,
                event_type=EventType.TRUCK_TERMINATED,
                truck_id=truck.truck_id,
                data={"reason": "battery_depleted"}
            ))
        # Check if truck completed all deliveries
        elif truck.is_complete:
            self.truck_states[truck.truck_id] = "complete"
            heapq.heappush(self.event_queue, Event(
                time=self.global_clock,
                event_type=EventType.TRUCK_TERMINATED,
                truck_id=truck.truck_id,
                data={"reason": "deliveries_complete"}
            ))
        else:
            # Truck is ready for next action
            self.truck_states[truck.truck_id] = "active"
            heapq.heappush(self.event_queue, Event(
                time=self.global_clock,
                event_type=EventType.TRUCK_READY,
                truck_id=truck.truck_id,
                data={"reason": "route_complete"}
            ))
    
    def _handle_charge_complete(self, event: Event):
        """Handle completion of charging."""
        truck = self.trucks[event.truck_id]
        data = event.data
        
        # Complete charging
        truck.finish_charging(
            charge_amount=data['charge_amount'],
            charge_duration=data['charge_duration']
        )
        
        # Remove from charger occupancy
        if truck.current_node in self.charger_occupancy:
            if truck.truck_id in self.charger_occupancy[truck.current_node]:
                self.charger_occupancy[truck.current_node].remove(truck.truck_id)
        
        if self.verbose:
            print(f"  Truck {truck.truck_id} finished charging")
            print(f"    Battery: {truck.current_battery:.1f} kWh ({truck.get_battery_percentage():.1f}%)")
        
        # Truck is ready for next action
        self.truck_states[truck.truck_id] = "active"
        heapq.heappush(self.event_queue, Event(
            time=self.global_clock,
            event_type=EventType.TRUCK_READY,
            truck_id=truck.truck_id,
            data={"reason": "charge_complete"}
        ))
    
    def _handle_truck_terminated(self, event: Event):
        """Handle truck termination (complete or failed)."""
        truck = self.trucks[event.truck_id]
        reason = event.data.get('reason', 'unknown')
        
        if self.verbose:
            print(f"  Truck {truck.truck_id} TERMINATED: {reason}")
            print(f"    Total time: {truck.total_time_elapsed:.2f}h")
            print(f"    Total distance: {truck.total_distance_traveled:.2f} km")
        
        # Truck will not generate any more events
        # State already set in previous handler
    
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
    
    def _execute_charge_action(self, truck: Truck, charge_hours: int) -> float:
        """Execute charging action and schedule charge completion event."""
        # Check if at a charging station
        if truck.current_node not in self.charging_nodes:
            if self.verbose:
                print(f"  ERROR: Truck {truck.truck_id} not at charging station")
            return -10.0
        
        # Check charger availability
        charger_node = truck.current_node
        current_occupancy = len(self.charger_occupancy[charger_node])
        capacity = self.charger_capacity[charger_node]
        
        if current_occupancy >= capacity:
            # Charger full - apply waiting time
            wait_time = self.charging_config.get('queue_wait_time', 0.5)
            truck.add_waiting_time(wait_time)
            if self.verbose:
                print(f"  Charger full! Waiting {wait_time:.2f}h")
        
        # Add to charger occupancy
        self.charger_occupancy[charger_node].append(truck.truck_id)
        
        # Calculate charge amount
        charge_rate = self.charging_config.get('charge_rate', 50.0)  # kWh per hour
        charge_amount = min(
            charge_hours * charge_rate,
            truck.battery_capacity - truck.current_battery
        )
        
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
        """Clean up resources."""
        pass
