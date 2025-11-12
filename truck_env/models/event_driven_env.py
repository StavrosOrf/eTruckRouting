"""
Event-Driven Truck Routing Environment - Single-agent controlling active truck.

Uses a global clock and event queue. Each truck generates two types of events:
- TRUCK_READY: Truck is ready to take an action (initial, after arrival, after charging, after waiting)
- TRUCK_ROUTING: Truck arrives at a destination node (delivery or charger)

The environment steps forward when events finish. Only one truck is active at a time.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Union
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
from truck_env.models.charging_station import ChargingStation
from truck_env.state.state_space import StateSpace, action_to_string
from truck_env.utils.plotter import EnvironmentPlotter
from truck_env.utils.statistics import EnvironmentStatistics


class EventDrivenTruckEnv(gym.Env):
    """
    Event-driven truck routing environment with global clock.

    Single-agent paradigm: controls whichever truck is currently active.
    Time advances to the next event, and step() is called when a truck needs a decision.

    STATE MACHINE:
    - ready: Truck can make a decision (navigate or charge)
    - routing: Truck is en route to destination
    - waiting_to_charge: Truck is at charger but no port available
    - charging: Truck is actively charging
    - complete: All deliveries done
    - failed: Ran out of battery or infeasible action

    EVENT TYPES:
    - TRUCK_READY: Truck needs a decision (after arrival/charge/wait)
    - TRUCK_ROUTING: Truck arrival at node (delivery or charger)
    """

    # metadata = {"render_modes": ["human"]}

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

        # Load all parameters from config (no overrides, no defaults)
        self.num_trucks = env_config["num_trucks"]
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

        # Traffic simulation settings
        self.traffic_config = self.config["traffic"]
        self.enable_traffic = self.traffic_config["enable_traffic"]
        self.traffic_std_factor = self.traffic_config["std_dev_factor"]
        self.traffic_max_std = self.traffic_config["max_std_dev_hours"]

        # Load graph and initialize transportation network
        graph = get_graph(self.config)
        self.transport_graph = TransportationGraph(graph)

        # Get charging nodes
        self.charging_nodes = self.transport_graph.get_charging_nodes()
        self.num_charging_nodes = len(self.charging_nodes)

        # Initialize charging station manager
        waiting_time_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "waiting_time_lookup.json",
        )
        self.charging_station = ChargingStation(
            charging_nodes=self.charging_nodes,
            transport_graph=self.transport_graph,
            waiting_time_lookup_path=waiting_time_path,
            verbose=self.verbose,
        )

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

        # Initialize state space
        self.state_space_manager = StateSpace(
            num_trucks=self.num_trucks,
            num_stops=self.num_stops,
            max_time=self.max_time,
            num_charging_nodes=self.num_charging_nodes,
        )
        self.observation_space = self.state_space_manager.observation_space

        # Event-driven simulation state
        self.global_clock = 0.0  # Current simulation time
        self.event_queue = []  # Priority queue of events (min-heap)
        self.active_truck_id = None  # ID of truck that needs to make a decision

        # Current episode state
        self.trucks = []
        self.truck_states = (
            {}
        )  # truck_id -> "active", "routing", "charging", "complete", "failed"
        self.episode_reward = 0.0
        self.waiting_start_times = {}  # Track when trucks enter waiting_to_charge state
        self.waiting_penalty_buffer = 0.0  # Buffer for waiting penalty to apply on next step

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
        self.waiting_start_times = {}  # Reset waiting time tracking
        self.waiting_penalty_buffer = 0.0  # Reset waiting penalty buffer

        # Reset charging station state
        self.charging_station.reset()

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
            self.truck_states[i] = "ready"  # All trucks start in ready state
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
        Process events until we find a TRUCK_READY event that can actually proceed.
        """
        while self.event_queue:
            # Get next event
            event = heapq.heappop(self.event_queue)

            # Advance clock
            self.global_clock = event.time

            # Process event
            if event.event_type == EventType.TRUCK_READY:
                truck = self.trucks[event.truck_id]

                # Safety check: skip if truck is already complete or failed
                if truck.is_complete or truck.failed:
                    # if self.verbose:
                    raise ValueError("Truck is already complete or failed")
                    status = "complete" if truck.is_complete else "failed"
                    print(
                        f"  Skipping TRUCK_READY for truck {truck.truck_id} (status: {status})"
                    )
                    continue

                # Skip if this is a stale wake event and truck is no longer waiting
                # This can happen when a truck gets woken early (port freed) but also had
                # a scheduled event based on predicted wait time
                reason = event.data.get("reason", "")
                if reason in [
                    "recheck_gating",
                    "recheck_after_arrival",
                    "recheck_charge_attempt",
                    "port_freed_early",
                ]:
                    current_state = self.truck_states.get(truck.truck_id, "")
                    if current_state not in ["waiting_to_charge", "ready"]:
                        if self.verbose:
                            print(
                                f"  Skipping stale TRUCK_READY for truck {truck.truck_id} (state: {current_state})"
                            )
                        continue

                # Check if this is a charge completion event
                reason = event.data.get("reason", "")
                if reason == "charge_complete":

                    charger_node = event.data["charger_node"]
                    charge_amount = event.data["charge_amount"]
                    charge_duration = event.data["charge_duration"]

                    # Complete charging for the truck (update battery)
                    truck.finish_charging(
                        charge_amount=charge_amount, charge_duration=charge_duration
                    )

                    # Finish charging via charging station manager
                    self.charging_station.finish_charging(
                        truck_id=truck.truck_id,
                        charger_node=charger_node,
                        global_clock=self.global_clock,
                    )

                    if self.verbose:
                        print(f"  Truck {truck.truck_id} finished charging")
                        print(
                            f"    Battery: {truck.current_battery:.1f} kWh ({truck.get_battery_percentage():.1f}%)"
                        )
                        print(
                            f"    Charged: {charge_amount:.1f} kWh in {charge_duration:.2f}h"
                        )

                    # Wake trucks waiting at this charger
                    self.charging_station.wake_waiting_trucks(
                        charger_node=charger_node,
                        global_clock=self.global_clock,
                        event_queue=self.event_queue,
                        EventType=EventType,
                        Event=Event,
                    )

                # Charger gating: enforce FCFS waitlist with capacity ports
                node = int(truck.current_node)
                if node in self.charging_nodes:
                    can_proceed, next_check_time = (
                        self.charging_station.check_charger_gating(
                            truck_id=truck.truck_id,
                            charger_node=node,
                            global_clock=self.global_clock,
                        )
                    )

                    if not can_proceed:
                        # raise ValueError("Truck cannot proceed to charge due to gating failure")
                        # Update state to waiting_to_charge
                        self.truck_states[truck.truck_id] = "waiting_to_charge"
                        
                        # Track when truck starts waiting (if not already tracked)
                        if truck.truck_id not in self.waiting_start_times:
                            self.waiting_start_times[truck.truck_id] = self.global_clock

                        # Only schedule recheck if we have a specific time
                        # (first truck in waitlist with predicted wait time)
                        # Otherwise, truck will be woken by wake_waiting_trucks
                        if next_check_time is not None:
                            heapq.heappush(
                                self.event_queue,
                                Event(
                                    time=next_check_time,
                                    event_type=EventType.TRUCK_READY,
                                    truck_id=event.truck_id,
                                    data={"reason": "recheck_gating"},
                                ),
                            )
                            if self.verbose:
                                print(
                                    f"  Truck {truck.truck_id} waiting for charge port at node {node}"
                                )
                                print(f"    Will recheck at t={next_check_time:.2f}h")
                        else:
                            if self.verbose:
                                print(
                                    f"  Truck {truck.truck_id} waiting for charge port at node {node}"
                                )
                                print(f"    Will be woken when port becomes available")
                        continue

                # Not at a charger or gating passed - truck is ready for decision
                # Calculate waiting penalty if truck was waiting
                if truck.truck_id in self.waiting_start_times:
                    waiting_duration = self.global_clock - self.waiting_start_times[truck.truck_id]
                    if waiting_duration > 0:
                        # Calculate time penalty for waiting
                        waiting_penalty = -waiting_duration * self.reward_config["time_multiplier"]
                        self.waiting_penalty_buffer = waiting_penalty
                        
                        # Update truck's waiting time stat
                        truck.add_waiting_time(waiting_duration)
                        
                        if self.verbose:
                            print(f"  Truck {truck.truck_id} finished waiting")
                            print(f"    Waited: {waiting_duration:.2f}h")
                            print(f"    Waiting penalty (to be applied on next action): {waiting_penalty:.2f}")
                    
                    # Clear waiting start time
                    del self.waiting_start_times[truck.truck_id]
                
                self.active_truck_id = event.truck_id
                self.truck_states[truck.truck_id] = "ready"
                return

            elif event.event_type == EventType.TRUCK_ROUTING:
                # Handle truck arrival at destination
                destination = event.data["destination"]

                # First, update the truck's physical state (position, battery, etc.)
                self.event_handler.handle_truck_routing(
                    event,
                    self.trucks,
                    self.truck_states,
                    self.truck_routes,
                    self.event_queue,
                    self.global_clock,
                    self.enable_plotting,
                )

                # Check the truck's state after arrival
                truck = self.trucks[event.truck_id]

                # Only schedule TRUCK_READY if truck is not complete or failed
                if not (truck.is_complete or truck.failed):
                    # If truck arrived at a charger, check if port is available
                    if destination in self.charging_nodes:
                        can_proceed, next_check_time = (
                            self.charging_station.check_charger_gating(
                                truck_id=truck.truck_id,
                                charger_node=destination,
                                global_clock=self.global_clock,
                            )
                        )

                        if not can_proceed:
                            # No free port - truck goes to waiting_to_charge state
                            self.truck_states[truck.truck_id] = "waiting_to_charge"
                            
                            # Track when truck starts waiting (if not already tracked)
                            if truck.truck_id not in self.waiting_start_times:
                                self.waiting_start_times[truck.truck_id] = self.global_clock

                            # Only schedule recheck if we have a specific time
                            if next_check_time is not None:
                                heapq.heappush(
                                    self.event_queue,
                                    Event(
                                        time=next_check_time,
                                        event_type=EventType.TRUCK_READY,
                                        truck_id=truck.truck_id,
                                        data={"reason": "recheck_after_arrival"},
                                    ),
                                )
                                if self.verbose:
                                    print(
                                        f"  Truck {truck.truck_id} waiting for charge port at node {destination}"
                                    )
                                    print(
                                        f"    Will recheck at t={next_check_time:.2f}h"
                                    )
                            else:
                                if self.verbose:
                                    print(
                                        f"  Truck {truck.truck_id} waiting for charge port at node {destination}"
                                    )
                                    print(
                                        f"    Will be woken when port becomes available"
                                    )
                        else:
                            # Port available - schedule immediate TRUCK_READY
                            heapq.heappush(
                                self.event_queue,
                                Event(
                                    time=self.global_clock,
                                    event_type=EventType.TRUCK_READY,
                                    truck_id=truck.truck_id,
                                    data={"reason": "arrived_at_charger"},
                                ),
                            )
                    else:
                        # Arrived at delivery node - schedule immediate TRUCK_READY
                        heapq.heappush(
                            self.event_queue,
                            Event(
                                time=self.global_clock,
                                event_type=EventType.TRUCK_READY,
                                truck_id=truck.truck_id,
                                data={"reason": "arrived_at_delivery"},
                            ),
                        )

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
            print(
                f"Current Node: {truck.current_node}, SoC: {truck.get_battery_percentage():.1f}%"
            )
            print(f"Action: {self._action_to_string(action)} ({action})")
            print(f"Event Queue: {self.event_queue}")
            print(f"{'='*80}")

        # Add any buffered waiting penalty from previous wait
        if self.waiting_penalty_buffer != 0.0:
            reward += self.waiting_penalty_buffer
            if self.verbose:
                print(f"  Adding waiting penalty from queue: {self.waiting_penalty_buffer:.2f}")
            self.waiting_penalty_buffer = 0.0  # Clear the buffer

        # Decode and execute action
        if action < self.num_navigation_actions:
            # Navigation action
            reward += self._execute_navigation_action(truck, action)
        else:
            # Charging action
            charge_idx = action - self.num_navigation_actions
            charge_durations = self.charging_config["charge_durations"]
            charge_hours = charge_durations[charge_idx]
            reward += self._execute_charge_action(truck, charge_hours)

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
        is_charger_nav = False
        if action < self.num_charging_nodes:
            # Go to charging station
            target_node = self.charging_nodes[action]
            is_charger_nav = True
        elif action == self.num_charging_nodes:
            # Go to next delivery
            target_node = truck.get_next_delivery_target()
            if target_node is None:
                raise ValueError("No remaining deliveries for truck")
        else:
            raise ValueError("Invalid navigation action")

        # Convert numpy types to native Python int
        if hasattr(target_node, "item"):
            target_node = int(target_node.item())
        else:
            target_node = int(target_node)

        current_node = int(truck.current_node)

        # Check if already at target
        if current_node == target_node:
            if target_node in self.charging_nodes:
                if self.verbose:
                    print(f"  Already at node {target_node}")
                    print(f"  Simulating charging for 1 hour at current location")
                # Simulate charging for 1 hour at current location
                charge_durations = self.charging_config["charge_durations"]
                charge_hours = charge_durations[0]
                return self._execute_charge_action(truck, charge_hours=charge_hours)
            else:
                # go to next delivery
                if self.verbose:
                    print(f"  Already at node {target_node}")
                    print(f"  go to next delivery")
                return self.execute_navigation_action(
                    truck, action=self.num_charging_nodes
                )

        # Calculate energy used for the trip
        energy_used = self.transport_graph.get_path_energy(current_node, target_node)

        # Check if path is reachable
        if energy_used == float("inf"):
            raise ValueError("No valid path for navigation action")

        travel_time = self.transport_graph.get_time_distance(current_node, target_node)
        discharge = energy_used
        distance = travel_time * truck.base_speed

        # Apply traffic simulation if enabled
        actual_travel_time = self._apply_traffic_simulation(travel_time)

        # Check if truck can make it
        if discharge > truck.current_battery:
            if self.verbose:
                print(
                    f"  ERROR: Insufficient battery ({truck.current_battery:.1f} kWh < {discharge:.1f} kWh needed)"
                )
            # Truck will fail - mark as failed and update state
            truck.failed = True
            self.truck_states[truck.truck_id] = "failed"
            return self.reward_config["failure_penalty"]

        queue_penalty = 0.0
        if is_charger_nav and self.verbose:
            charger_info = self.charging_station.get_charger_info(
                target_node, self.global_clock
            )
            print(f"  Going to charger @ node {target_node}")
            print(
                f"    Current occupancy: {charger_info['current_occupancy']}/{charger_info['capacity']}"
            )

        # If leaving a charger to navigate elsewhere, remove from its waitlist and wake others
        if (not is_charger_nav) and (current_node in self.charging_nodes):
            self.charging_station.remove_from_waitlist(truck.truck_id, current_node)
            # Wake other trucks waiting at this charger since a spot may have opened
            self.charging_station.wake_waiting_trucks(
                charger_node=current_node,
                global_clock=self.global_clock,
                event_queue=self.event_queue,
                EventType=EventType,
                Event=Event,
            )

        # Schedule truck routing (arrival) event
        completion_time = self.global_clock + actual_travel_time
        heapq.heappush(
            self.event_queue,
            Event(
                time=completion_time,
                event_type=EventType.TRUCK_ROUTING,
                truck_id=truck.truck_id,
                data={
                    "destination": target_node,
                    "distance": distance,
                    "travel_time": actual_travel_time,
                    "discharge": discharge,
                },
            ),
        )

        # Update truck state and track route information
        self.truck_states[truck.truck_id] = "routing"
        truck.route_destination = target_node
        truck.route_arrival_time = completion_time

        if self.verbose:
            print(f"  Routing to node {target_node}")
            print(
                f"    Distance: {distance:.2f} km, Time: {actual_travel_time:.2f}h (base: {travel_time:.2f}h)"
            )
            print(f"    Will arrive at t={completion_time:.2f}h")
            print(f"    Current Battery: {truck.current_battery:.1f} kWh")
            print(
                f"    Battery after trip: {truck.current_battery - discharge:.1f} kWh"
            )
            # Waiting at charger will be determined upon arrival via queue gating

        # Calculate reward (using actual travel time, not base time)
        time_penalty = -actual_travel_time * self.reward_config["time_multiplier"]
        # distance_penalty = -distance * self.reward_config["distance_penalty"]

        # Bonus if this is a delivery
        if target_node == truck.get_next_delivery_target():
            delivery_bonus = self.reward_config["delivery_bonus"]
            return time_penalty + delivery_bonus

        return time_penalty

    def _apply_traffic_simulation(self, travel_time: float) -> float:
        """
        Apply traffic simulation to travel time using normal distribution.

        Args:
            travel_time: Base travel time from the graph (hours)

        Returns:
            Travel time with traffic variation applied (hours)
        """
        if not self.enable_traffic or travel_time <= 0:
            return travel_time

        # Calculate standard deviation
        std_dev = travel_time * self.traffic_std_factor

        # Cap the std_dev if max is specified
        if self.traffic_max_std > 0:
            std_dev = min(std_dev, self.traffic_max_std)

        # Sample from normal distribution N(mean=travel_time, std=std_dev)
        actual_travel_time = np.random.normal(loc=travel_time, scale=std_dev)

        # Ensure travel time is positive (at least 1% of original)
        actual_travel_time = max(actual_travel_time, travel_time * 0.01)
        actual_travel_time = min(
            actual_travel_time, travel_time * 2.0
        )  # Cap to 2x original

        if self.verbose:
            variation_percent = ((actual_travel_time - travel_time) / travel_time) * 100
            print(
                f"    Traffic simulation: {travel_time:.2f}h → {actual_travel_time:.2f}h ({variation_percent:+.1f}%)"
            )

        return actual_travel_time

    def _execute_charge_action(self, truck: Truck, charge_hours: int) -> float:
        """Execute charging action and schedule charge completion event."""
        # Check if at a charging station
        if truck.current_node not in self.charging_nodes:
            # go to next delivery instead

            if self.verbose:
                print(f"  Truck {truck.truck_id} not at charging station")
                print(f"  Executing navigation to next delivery instead")

            return self._execute_navigation_action(
                truck, action=self.num_charging_nodes
            )

        charger_node = truck.current_node

        # Check if truck can start charging (enforce waitlist eligibility)
        can_proceed, next_check_time = self.charging_station.check_charger_gating(
            truck_id=truck.truck_id,
            charger_node=charger_node,
            global_clock=self.global_clock,
        )

        if not can_proceed:
            raise ValueError("Truck cannot start charging due to gating failure")
            # This should not happen because gating is checked before truck becomes ready
            # Truck wants to charge but no port available - go to waiting_to_charge state
            self.truck_states[truck.truck_id] = "waiting_to_charge"
            
            # Track when truck starts waiting (if not already tracked)
            if truck.truck_id not in self.waiting_start_times:
                self.waiting_start_times[truck.truck_id] = self.global_clock

            # Only schedule recheck if we have a specific time
            if next_check_time is not None:
                heapq.heappush(
                    self.event_queue,
                    Event(
                        time=next_check_time,
                        event_type=EventType.TRUCK_READY,
                        truck_id=truck.truck_id,
                        data={"reason": "recheck_charge_attempt"},
                    ),
                )
                if self.verbose:
                    print(
                        f"  Truck {truck.truck_id} cannot start charging yet (no free port). Going to waiting state."
                    )
                    print(f"    Will recheck at t={next_check_time:.2f}h")
            else:
                if self.verbose:
                    print(
                        f"  Truck {truck.truck_id} cannot start charging yet (no free port). Going to waiting state."
                    )
                    print(f"    Will be woken when port becomes available")

            # Small time penalty for attempting to charge when no port available
            return -0.01

        # Get charger type and determine charge rate
        charger_type = self.charging_station.charger_type[charger_node]
        charging_config = self.config["charging"]

        if charger_type == "DCFast":
            raise NotImplementedError("DCFast charging not implemented yet")
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

        charge_hours = charge_amount / (charge_rate * efficiency)

        # Start charging via charging station manager
        self.charging_station.start_charging(
            truck_id=truck.truck_id,
            charger_node=charger_node,
            charge_hours=charge_hours,
            global_clock=self.global_clock,
        )

        # Schedule TRUCK_READY event when charging completes
        completion_time = self.global_clock + charge_hours
        heapq.heappush(
            self.event_queue,
            Event(
                time=completion_time,
                event_type=EventType.TRUCK_READY,
                truck_id=truck.truck_id,
                data={
                    "reason": "charge_complete",
                    "charge_amount": charge_amount,
                    "charge_duration": charge_hours,
                    "charger_node": charger_node,
                },
            ),
        )

        # Update truck state to charging
        self.truck_states[truck.truck_id] = "charging"
        truck.start_charging(self.global_clock)

        if self.verbose:
            print(f"  Charging for {charge_hours}h")
            print(f"    Will charge {charge_amount:.1f} kWh")
            print(f"    Will complete at t={completion_time:.2f}h")
            print(
                f"    Charger: {self.charging_station.charger_type[charger_node]} @ node {charger_node}"
            )

        # Calculate reward (penalty for time spent charging only, no queue penalty)
        # charge_penalty = -charge_hours * self.reward_config["charge_penalty"]
        return -charge_hours

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
        if self.global_clock >= self.max_time and self.verbose:
            input("Press Enter to continue...")

        return self.global_clock >= self.max_time

    def _get_observation(self) -> np.ndarray:
        """Get observation/state for the active truck."""
        return self.state_space_manager.get_state(
            trucks=self.trucks,
            active_truck_id=self.active_truck_id,
            transport_graph=self.transport_graph,
            charging_nodes=self.charging_nodes,
            truck_states=self.truck_states,
            event_queue=self.event_queue,
            global_clock=self.global_clock,
        )

    def _get_info(self) -> Dict:
        """Get info dictionary."""
        all_complete = all(truck.is_complete for truck in self.trucks)
        any_failed = any(truck.failed for truck in self.trucks)

        # Get charger utilization statistics from charging station manager
        charger_utilization = self.charging_station.get_utilization_stats(
            self.global_clock
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
            # Expose simplified queue state for debugging/analysis
            "charger_waitlist_lengths": {
                int(node): len(self.charging_station.charger_waitlist[node])
                for node in self.charging_nodes
            },
            "charger_occupancy_counts": {
                int(node): len(self.charging_station.charger_occupancy[node])
                for node in self.charging_nodes
            },
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

            # Generate charging queue visualizations
            self.plotter.plot_charger_queue_dynamics(
                self.charging_station,
                self.transport_graph,
                self.global_clock,
            )

            self.plotter.plot_charger_utilization_heatmap(
                self.charging_station,
                self.transport_graph,
                self.global_clock,
            )

            self.plotter.plot_charger_statistics_summary(
                self.charging_station,
                self.transport_graph,
            )

            # Print and save statistics
            charger_util = self.charging_station.get_utilization_stats(
                self.global_clock
            )

            self.stats_collector.print_statistics(
                self.trucks,
                self.truck_states,
                self.truck_routes,
                charger_util,
                self.global_clock,
                self.num_trucks,
            )
