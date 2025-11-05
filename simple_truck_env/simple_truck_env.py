"""
Simple Truck Routing Environment - Multi-truck coordination with MultiDiscrete actions.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
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


class SimpleTruckEnv(gym.Env):
    """
    Simple multi-truck routing environment with MultiDiscrete action space.
    
    Controls multiple trucks simultaneously, each making navigation and charging decisions.
    Uses MultiDiscrete action space where each truck has 2 action components:
    1. Navigation action (which node to visit)
    2. Charging action (how long to charge, if at a charging station)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        config: Optional[Union[str, Dict]] = None,
        num_trucks: Optional[int] = None,
        num_stops: Optional[int] = None,
        min_hop_distance: Optional[float] = None,
        max_hop_distance: Optional[float] = None,
        max_steps: Optional[int] = None,
        verbose: Optional[bool] = None
    ):
        """
        Initialize the environment.
        
        Args:
            config: Path to config.yaml file or config dictionary. If None, uses default config.
            num_trucks: Number of trucks (overrides config if provided)
            num_stops: Number of delivery stops per truck (overrides config)
            min_hop_distance: Minimum distance between delivery stops (overrides config)
            max_hop_distance: Maximum distance between delivery stops (overrides config)
            max_steps: Maximum steps per episode (overrides config)
            verbose: Print detailed information (overrides config)
        """
        super().__init__()
        
        # Load configuration
        if isinstance(config, str):
            # Config is a file path
            from simple_truck_env.config_utils import load_config
            self.config = load_config(config)
        elif isinstance(config, dict):
            # Config is already a dictionary
            self.config = config
        elif config is None:
            # Load default config
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
        self.max_steps = max_steps if max_steps is not None else env_config.get('max_steps', 200)
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
            print(f"Environment initialized:")
            print(f"  - Total nodes: {self.transport_graph.num_nodes}")
            print(f"  - Charging nodes: {self.num_charging_nodes}")
            print(f"  - Number of trucks: {self.num_trucks}")
            print(f"  - Delivery stops per truck: {self.num_stops}")
        
        # Define action space - MultiDiscrete for multiple trucks
        # Each truck has a single combined action
        # Actions are: [chargers (0 to num_charging_nodes-1), next_delivery (num_charging_nodes),
        #               charge_1h, charge_2h, charge_3h, charge_4h]
        charge_durations = self.charging_config.get('charge_durations', [1, 2, 3, 4])
        self.num_navigation_actions = self.num_charging_nodes + 1  # Chargers + next delivery
        self.num_charge_actions = len(charge_durations)  # Charge for 1-4 hours
        
        # MultiDiscrete: each truck has one action combining navigation and charging
        # Total actions per truck = navigation_actions + charge_actions
        self.action_space = spaces.MultiDiscrete(
            [self.num_navigation_actions + self.num_charge_actions] * self.num_trucks
        )
        
        # Define observation space - Box for all trucks combined
        # Per truck: current_node, next_delivery_node, battery_level, battery_%, 
        #            is_charging, deliveries_remaining, nearest_charger_dist,
        #            can_reach_next, time_elapsed, distance_traveled
        obs_dim_per_truck = 10
        self.obs_dim = obs_dim_per_truck * self.num_trucks
        
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
                0.0,  # time_elapsed
                0.0,  # distance_traveled
            ] * self.num_trucks),
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
            ] * self.num_trucks),
            dtype=np.float32
        )
        
        # Charging station queue/occupancy tracking
        self.charger_occupancy = {node: [] for node in self.charging_nodes}  # List of truck IDs
        self.charger_capacity = {
            node: sum(self.transport_graph.get_charger_info(node).values())
            for node in self.charging_nodes
        }
        
        # Current episode state
        self.trucks = []
        self.current_step = 0
        self.episode_reward = 0.0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset step counter and reward
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Reset charger occupancy
        self.charger_occupancy = {node: [] for node in self.charging_nodes}
        
        # Create trucks with random delivery sequences
        self.trucks = []
        for i in range(self.num_trucks):
            self._create_truck(i)
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"NEW EPISODE - {self.num_trucks} Trucks")
            print(f"{'='*80}")
            for truck in self.trucks:
                print(f"\n  Truck {truck.truck_id}:")
                print(f"    Delivery sequence: {truck.delivery_sequence}")
                print(f"    Battery: {truck.current_battery:.1f}/{truck.battery_capacity:.1f} kWh")
                print(f"    Total distance: {self.transport_graph.calculate_total_distance(truck.delivery_sequence):.1f} km")
        
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
        
        # Select truck type based on config
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
            # Percentage
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
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: MultiDiscrete action array [nav_0, charge_0, nav_1, charge_1, ..., nav_N, charge_N]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        total_reward = 0.0
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Step {self.current_step}")
            print(f"{'='*80}")
        
        # Process actions for each truck
        for truck_idx, truck in enumerate(self.trucks):
            if truck.is_complete or truck.failed:
                # Skip trucks that are done
                continue
            
            # Extract actions for this truck
            nav_action = action[truck_idx * 2]
            charge_action = action[truck_idx * 2 + 1]
            
            if self.verbose:
                print(f"\nTruck {truck.truck_id}:")
                print(f"  Nav action: {self._nav_action_to_string(nav_action)}")
                print(f"  Charge action: {self._charge_action_to_string(charge_action)}")
                print(f"  Current: node={truck.current_node}, battery={truck.current_battery:.1f} kWh")
            
            # Execute navigation action
            nav_reward = self._execute_navigation_action(truck, nav_action)
            total_reward += nav_reward
            
            # Execute charging action (if applicable)
            if charge_action > 0:  # 0 = no charge
                charge_reward = self._execute_charge_action(truck, charge_action)
                total_reward += charge_reward
        
        # Check termination conditions
        all_complete = all(truck.is_complete for truck in self.trucks)
        any_failed = any(truck.failed for truck in self.trucks)
        
        terminated = all_complete or any_failed
        truncated = self.current_step >= self.max_steps
        
        # Add completion/failure rewards
        if all_complete:
            completion_bonus = self.reward_config.get('completion_bonus', 1000.0)
            total_reward += completion_bonus
            if self.verbose:
                print(f"\n✅ ALL TRUCKS COMPLETED! Bonus: +{completion_bonus}")
        elif any_failed:
            failure_penalty = self.reward_config.get('failure_penalty', -500.0)
            total_reward += failure_penalty
            if self.verbose:
                print(f"\n❌ TRUCK(S) FAILED! Penalty: {failure_penalty}")
        
        if truncated and self.verbose:
            print(f"\n⏱️ Episode truncated - max steps reached")
        
        # Get new observation
        obs = self._get_observation()
        info = self._get_info()
        
        self.episode_reward += total_reward
        
        return obs, total_reward, terminated, truncated, info
    
    def _execute_navigation_action(self, truck: Truck, action: int) -> float:
        """
        Execute a navigation action for a specific truck.
        
        Args:
            truck: The truck to move
            action: Navigation action index
            
        Returns:
            Reward for this action
        """
        if action < self.num_charging_nodes:
            # Go to charging station
            target_node = self.charging_nodes[action]
            action_type = "charge_station"
        else:
            # Go to next delivery
            target_node = truck.get_next_delivery_target()
            if target_node is None:
                # No more deliveries - invalid action
                invalid_penalty = self.reward_config.get('invalid_action_penalty', -10.0)
                return invalid_penalty
            action_type = "delivery"
        
        # Calculate route
        distance = self.transport_graph.get_distance(truck.current_node, target_node)
        
        if distance == float('inf'):
            # No path exists
            if self.verbose:
                print(f"    ❌ No path to node {target_node}")
            return self.reward_config.get('invalid_action_penalty', -10.0)
        
        # Calculate travel time and discharge
        travel_time = distance / truck.base_speed
        edge_data = self.transport_graph.get_edge_data(truck.current_node, target_node)
        terrain_factor = edge_data.get("terrain_factor", 1.0) if edge_data else 1.0
        
        discharge = truck.discharge_rate * distance * terrain_factor
        
        # Check if truck has enough battery
        if truck.current_battery < discharge:
            if self.verbose:
                print(f"    ⚠️ Insufficient battery! Need {discharge:.1f}, have {truck.current_battery:.1f}")
            return self.reward_config.get('insufficient_battery_penalty', -50.0)
        
        # Execute move
        truck.move_to_node(target_node, distance, travel_time, discharge)
        
        # Calculate reward (negative time spent)
        time_penalty = self.reward_config.get('time_penalty', -1.0)
        reward = time_penalty * travel_time
        
        # Bonus for completing a delivery
        if action_type == "delivery":
            delivery_bonus = self.reward_config.get('delivery_bonus', 50.0)
            reward += delivery_bonus
            if self.verbose:
                print(f"    📦 Delivery complete! Remaining: {len(truck.get_remaining_deliveries())}")
        
        if self.verbose:
            print(f"    ➡️ Moved to node {target_node} (distance: {distance:.1f}km, time: {travel_time:.2f}h)")
            print(f"    Battery: {truck.current_battery:.1f} kWh remaining")
        
        return reward
    
    def _execute_charge_action(self, truck: Truck, charge_action: int) -> float:
        """
        Execute a charging action for a specific truck.
        
        Args:
            truck: The truck to charge
            charge_action: Charging action (0=no charge, 1-4=charge for 1-4 hours)
            
        Returns:
            Reward for this action
        """
        if charge_action == 0:
            # No charging requested
            return 0.0
        
        current_node = truck.current_node
        
        # Check if at a charging station
        if not self.transport_graph.has_charger(current_node):
            if self.verbose:
                print(f"    ❌ Not at a charging station!")
            return self.reward_config.get('invalid_action_penalty', -10.0)
        
        # Get charge duration
        charge_durations = self.charging_config.get('charge_durations', [1, 2, 3, 4])
        hours = charge_durations[charge_action - 1] if charge_action <= len(charge_durations) else 1
        
        # Check charger availability and simulate queue
        waiting_time = self._simulate_charging_queue(truck, current_node)
        
        # Calculate charge amount
        charge_rate = self.charging_config.get('charge_rate', 50.0)
        efficiency = self.charging_config.get('efficiency', 0.95)
        charge_amount = min(
            charge_rate * hours * efficiency,
            truck.battery_capacity - truck.current_battery
        )
        
        # Total time = waiting + charging
        total_time = waiting_time + hours
        
        # Execute charging
        truck.finish_charging(charge_amount, total_time)
        
        # Remove from charger queue
        if truck.truck_id in self.charger_occupancy.get(current_node, []):
            self.charger_occupancy[current_node].remove(truck.truck_id)
        
        # Reward is negative time spent
        time_penalty = self.reward_config.get('time_penalty', -1.0)
        reward = time_penalty * total_time
        
        if self.verbose:
            if waiting_time > 0:
                print(f"    ⏳ Waited {waiting_time:.1f}h for charger")
            print(f"    🔌 Charged for {hours}h, gained {charge_amount:.1f} kWh")
            print(f"    Battery: {truck.current_battery:.1f}/{truck.battery_capacity:.1f} kWh")
        
        return reward
    
    def _simulate_charging_queue(self, truck: Truck, charger_node: int) -> float:
        """
        Simulate charging queue - calculate waiting time if chargers are occupied.
        
        Args:
            truck: The truck trying to charge
            charger_node: The charging node
            
        Returns:
            Waiting time in hours
        """
        # Get number of chargers at this station
        capacity = self.charger_capacity.get(charger_node, 1)
        
        # Get current queue
        queue = self.charger_occupancy.get(charger_node, [])
        
        # Add truck to queue if not already there
        if truck.truck_id not in queue:
            queue.append(truck.truck_id)
            self.charger_occupancy[charger_node] = queue
        
        # Calculate waiting time based on queue position
        queue_position = queue.index(truck.truck_id)
        
        if queue_position < capacity:
            # Charger available immediately
            return 0.0
        else:
            # Need to wait - simple model: 0.5 hours per truck ahead in queue
            trucks_ahead = queue_position - capacity + 1
            waiting_time = trucks_ahead * 0.5
            return waiting_time
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation for all trucks."""
        obs_list = []
        
        for truck in self.trucks:
            next_target = truck.get_next_delivery_target()
            
            # Normalize node indices
            current_node_norm = truck.current_node / max(1, self.transport_graph.num_nodes)
            next_node_norm = (next_target / max(1, self.transport_graph.num_nodes)) if next_target is not None else 0.0
            
            # Get nearest charger distance
            _, nearest_charger_dist = self.transport_graph.get_nearest_charging_node(truck.current_node)
            
            # Check if can reach next delivery
            can_reach_next = 0.0
            if next_target is not None:
                dist_to_next = self.transport_graph.get_distance(truck.current_node, next_target)
                can_reach_next = 1.0 if truck.can_reach_node(next_target, dist_to_next) else 0.0
            
            truck_obs = [
                current_node_norm,
                next_node_norm,
                truck.current_battery,
                truck.get_battery_percentage(),
                float(truck.is_charging),
                float(len(truck.get_remaining_deliveries())),
                nearest_charger_dist,
                can_reach_next,
                truck.total_time_elapsed,
                truck.total_distance_traveled,
            ]
            
            obs_list.extend(truck_obs)
        
        return np.array(obs_list, dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        truck_states = [truck.get_state_dict() for truck in self.trucks]
        
        return {
            "trucks": truck_states,
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "num_trucks": self.num_trucks,
            "all_complete": all(t.is_complete for t in self.trucks),
            "any_failed": any(t.failed for t in self.trucks),
        }
    
    def _nav_action_to_string(self, action: int) -> str:
        """Convert navigation action index to human-readable string."""
        if action < self.num_charging_nodes:
            return f"Go to charging station at node {self.charging_nodes[action]}"
        else:
            return f"Go to next delivery"
    
    def _charge_action_to_string(self, action: int) -> str:
        """Convert charging action index to human-readable string."""
        if action == 0:
            return "No charging"
        else:
            charge_durations = self.charging_config.get('charge_durations', [1, 2, 3, 4])
            hours = charge_durations[action - 1] if action <= len(charge_durations) else 1
            return f"Charge for {hours} hour(s)"
    
    def render(self):
        """Render the environment state."""
        if not self.trucks:
            print("Environment not initialized. Call reset() first.")
            return
        
        print(f"\n{'='*60}")
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Number of trucks: {self.num_trucks}")
        
        for truck in self.trucks:
            status = "✅ Complete" if truck.is_complete else ("❌ Failed" if truck.failed else "🚛 Active")
            print(f"\nTruck {truck.truck_id} [{status}]:")
            print(f"  Position: node {truck.current_node}")
            print(f"  Battery: {truck.current_battery:.1f}/{truck.battery_capacity:.1f} kWh ({truck.get_battery_percentage():.1f}%)")
            print(f"  Deliveries: {len(truck.get_remaining_deliveries())} remaining")
            print(f"  Time: {truck.total_time_elapsed:.2f}h, Distance: {truck.total_distance_traveled:.1f}km")
        
        print(f"\nEpisode reward: {self.episode_reward:.2f}")
        print(f"{'='*60}\n")
    
    def close(self):
        """Clean up resources."""
        pass
