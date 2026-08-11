"""
Event-Driven Truck Routing Environment - Single-agent controlling active truck.

Uses a global clock and event queue. Each truck generates two types of events:
- TRUCK_READY: Truck is ready to take an action (initial, after arrival, after charging, after waiting)
- TRUCK_ROUTING: Truck arrives at a destination node (delivery or charger)

The environment steps forward when events finish. Only one truck is active at a time.
"""

import datetime
import heapq
import math
import os
import sys
from numbers import Integral

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EVRoutingEnv.evaluation.artifacts import build_scenario_descriptor
from EVRoutingEnv.evaluation.metrics import extract_operational_metrics
from EVRoutingEnv.models.core.transportation_graph import TransportationGraph
from EVRoutingEnv.models.core.truck import Truck
from EVRoutingEnv.models.environment.event_handlers import (
    Event,
    EventHandler,
    EventType,
)
from EVRoutingEnv.models.environment.joint_instance import (
    JointRoutingInstance,
    generate_joint_routing_instance,
)
from EVRoutingEnv.models.environment.loaders import create_truck
from EVRoutingEnv.models.simulation.charging_curve import ChargingCurveModel
from EVRoutingEnv.models.simulation.charging_station import ChargingStation
from EVRoutingEnv.models.simulation.delivery_simulator import DeliverySimulator
from EVRoutingEnv.models.simulation.scenario import ScenarioRandomStreams
from EVRoutingEnv.models.simulation.traffic_simulation import TrafficSimulator
from EVRoutingEnv.state.action_mask import get_action_mask
from EVRoutingEnv.state.feasibility import (
    FeasibilityReason,
    evaluate_duration_charge,
    evaluate_joint_route,
    evaluate_target_soc_charge,
    joint_action_feasibility,
)
from EVRoutingEnv.state.features import SCHEMA_VERSION, extract_canonical_features
from EVRoutingEnv.state.representations import (
    CanonicalShapeSpec,
    canonical_flat_observation,
    canonical_graph_observation,
    pad_canonical_features,
)
from EVRoutingEnv.state.state_space import StateSpace
from EVRoutingEnv.utils.charging_logger import ChargingLogger
from EVRoutingEnv.utils.plotter import EnvironmentPlotter
from EVRoutingEnv.utils.statistics import EnvironmentStatistics
from EVRoutingEnv.utils.utils import (
    check_navigation_feasibility,
    get_graph,
    load_config,
)


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
        config: str | dict,
        verbose: bool | None = None,
        enable_plotting: bool | None = None,
        run_id: str | None = None,
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
        self.num_trucks = _positive_integer(
            env_config["num_trucks"],
            "environment.num_trucks",
        )
        # Delivery stops configuration
        self.fixed_num_stops = _positive_integer(
            env_config["num_stops"],
            "environment.num_stops",
        )
        self.allow_variable_num_stops = env_config["allow_variable_num_stops"]
        if not isinstance(self.allow_variable_num_stops, bool):
            raise TypeError("environment.allow_variable_num_stops must be boolean")
        # num_stops will be (re)set in reset(); initialize with fixed for defaults
        self.num_stops = self.fixed_num_stops
        self.min_hop_distance = _nonnegative_finite(
            env_config["min_hop_distance"],
            "environment.min_hop_distance",
        )
        self.max_hop_distance = _nonnegative_finite(
            env_config["max_hop_distance"],
            "environment.max_hop_distance",
        )
        if self.max_hop_distance < self.min_hop_distance:
            raise ValueError(
                "environment.max_hop_distance must be at least min_hop_distance"
            )
        self.max_time = _positive_finite(
            env_config["max_time"],
            "environment.max_time",
        )
        max_stops_for_bounds = self.fixed_num_stops
        default_step_limit = int(self.num_trucks * max_stops_for_bounds * 7.5)
        self.max_episode_steps = _positive_integer(
            env_config.get("max_episode_steps", default_step_limit),
            "environment.max_episode_steps",
        )
        self.verbose = verbose if verbose is not None else env_config["verbose"]
        if not isinstance(self.verbose, bool):
            raise TypeError("environment.verbose must be boolean")
        _validate_initial_battery_setting(
            self.config["truck"]["initial_battery"]
        )

        # Visualization and output settings
        self.enable_plotting = enable_plotting

        # Generate unique run_id based on timestamp
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
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
        # Note: seed will be set in reset() method for reproducibility
        self.traffic_simulator = TrafficSimulator(
            enable_traffic=self.traffic_config["enable_traffic"],
            std_dev_factor=self.traffic_config["std_dev_factor"],
            max_std_dev_hours=self.traffic_config["max_std_dev_hours"],
            rush_hour_multiplier=self.traffic_config["rush_hour_multiplier"],
            enable_energy_uncertainty=self.traffic_config["enable_energy_uncertainty"],
            energy_uncertainty_factor=self.traffic_config["energy_uncertainty_factor"],
            min_energy_multiplier=self.traffic_config["min_energy_multiplier"],
            max_energy_multiplier=self.traffic_config["max_energy_multiplier"],
            verbose=self.verbose,
            seed=None  # Will be set in reset()
        )
        
        # Delivery simulation settings
        self.delivery_config = self.config["delivery"]
        self.problem_config = self.config.get("problem", {})
        self.problem_mode = self.problem_config.get("mode", "preassigned_routes")
        if self.problem_mode not in {"preassigned_routes", "joint_fleet"}:
            raise ValueError(
                "problem.mode must be 'preassigned_routes' or 'joint_fleet'"
            )
        self.joint_routing = self.problem_mode == "joint_fleet"
        self.enable_flexible_delivery_order = self.joint_routing or self.delivery_config.get(
            "enable_flexible_delivery_order", False
        )
        
        # Note: seed will be set in reset() method for reproducibility
        self.delivery_simulator = DeliverySimulator(
            enable_stochastic_unloading=self.delivery_config["enable_stochastic_unloading"],
            base_unloading_time=self.delivery_config["base_unloading_time"],
            std_dev_factor=self.delivery_config["std_dev_factor"],
            max_std_dev_hours=self.delivery_config["max_std_dev_hours"],
            business_hours_multiplier=self.delivery_config["business_hours_multiplier"],
            min_unloading_multiplier=self.delivery_config["min_unloading_multiplier"],
            max_unloading_multiplier=self.delivery_config["max_unloading_multiplier"],
            verbose=self.verbose,
            seed=None  # Will be set in reset()
        )

        # Load graph and initialize transportation network
        graph = get_graph(self.config)
        self.transport_graph = TransportationGraph(graph)

        # Get charging nodes
        self.charging_nodes = self.transport_graph.get_charging_nodes()
        self.num_charging_nodes = len(self.charging_nodes)

        # If verbose, print charger summary loaded into the simulation
        if self.verbose:
            try:
                charger_details = self.transport_graph.get_charger_details()
                # Aggregate counts by type
                agg: dict[str, int] = {}
                for info in charger_details.values():
                    for t, c in info.get("types", {}).items():
                        agg[t] = agg.get(t, 0) + int(c)

                print("Charger inventory loaded:")
                print(f"  - Charger nodes: {self.num_charging_nodes}")
                if agg:
                    by_type = ", ".join([f"{k}={v}" for k, v in sorted(agg.items())])
                    print(f"  - Totals by type: {by_type}")
                # List each charger node (internal_id -> original_id : types)
                for nid in sorted(charger_details.keys()):
                    info = charger_details[nid]
                    types = info.get("types", {})
                    types_str = ", ".join([f"{k}:{v}" for k, v in sorted(types.items())]) or "(none)"
                    print(f"    • node {nid} (orig {info.get('original_id')}): {types_str}")
            except (AttributeError, KeyError, TypeError, ValueError) as e:
                print(f"[Env] Warning: failed to print charger summary: {e}")

        # Initialize charging station manager
        # Go up two levels from models/environment/ to EVRoutingEnv/
        waiting_time_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "waiting_time_lookup.json",
        )
        station_charging_config = dict(self.charging_config)
        if self.joint_routing:
            station_charging_config.setdefault(
                "station_power_classes_kw",
                [150.0, 350.0, 750.0],
            )
        self.charging_station = ChargingStation(
            charging_nodes=self.charging_nodes,
            transport_graph=self.transport_graph,
            waiting_time_lookup_path=waiting_time_path,
            verbose=self.verbose,
            charging_config=station_charging_config,
        )
        
        # Initialize charging curve model
        self.charging_curve_model = ChargingCurveModel(verbose=self.verbose)
        
        # Initialize charging logger if plotting is enabled
        if self.enable_plotting:
            self.charging_logger = ChargingLogger(
                output_dir=self.output_dir,
                verbose=self.verbose
            )
        else:
            self.charging_logger = None

        if self.verbose:
            print("Event-Driven Environment initialized:")
            print(f"  - Total nodes: {self.transport_graph.num_nodes}")
            print(f"  - Charging nodes: {self.num_charging_nodes}")
            print(f"  - Number of trucks: {self.num_trucks}")
            print(f"  - Max simulation time: {self.max_time} hours")

        # Define action space - Discrete for single active truck
        # Sequential mode: [chargers (0 to num_charging_nodes-1), next_delivery (num_charging_nodes), charge_1h, ...]
        # Flexible mode: [chargers (0 to num_charging_nodes-1), delivery_0, ..., delivery_N-1, depot_return, charge_1h, ...]
        default_charging_mode = "target_soc" if self.joint_routing else "duration"
        self.charging_action_mode = self.charging_config.get(
            "action_mode",
            default_charging_mode,
        )
        if self.charging_action_mode == "target_soc":
            self.charge_action_values = [
                float(value)
                for value in self.charging_config.get(
                    "target_soc_levels",
                    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                )
            ]
            if (
                not self.charge_action_values
                or any(
                    not 0.0 < value <= 1.0
                    for value in self.charge_action_values
                )
                or self.charge_action_values
                != sorted(set(self.charge_action_values))
            ):
                raise ValueError(
                    "charging.target_soc_levels must be unique, sorted, and in (0, 1]"
                )
        elif self.charging_action_mode == "duration":
            self.charge_action_values = [
                float(value)
                for value in self.charging_config["charge_durations"]
            ]
            if not self.charge_action_values or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.charge_action_values
            ):
                raise ValueError(
                    "charging.charge_durations must be finite and positive"
                )
        else:
            raise ValueError(
                "charging.action_mode must be 'target_soc' or 'duration'"
            )
        
        stops_for_action_space = self.fixed_num_stops
        if self.enable_flexible_delivery_order:
            # Flexible mode: separate action for each potential delivery node (size with max to keep action space stable)
            self.num_navigation_actions = self.num_charging_nodes + stops_for_action_space + 1
        else:
            # Sequential mode: single next delivery action
            self.num_navigation_actions = self.num_charging_nodes + 1
        
        self.num_charge_actions = len(self.charge_action_values)

        # Discrete action space (single agent)
        self.action_space = spaces.Discrete(
            self.num_navigation_actions + self.num_charge_actions
        )

        default_observation_mode = (
            "canonical_flat" if self.joint_routing else "legacy_flat"
        )
        self.observation_mode = env_config.get(
            "observation_mode",
            default_observation_mode,
        )
        if self.observation_mode not in {"canonical_flat", "legacy_flat"}:
            raise ValueError(
                "environment.observation_mode must be 'canonical_flat' or "
                "'legacy_flat'"
            )
        if self.observation_mode == "canonical_flat" and not self.joint_routing:
            raise ValueError(
                "canonical_flat observations require problem.mode=joint_fleet"
            )

        self.canonical_shape = CanonicalShapeSpec(
            max_trucks=self.num_trucks,
            max_customers=stops_for_action_space,
            max_chargers=self.num_charging_nodes,
            max_actions=self.action_space.n,
        )
        if self.observation_mode == "canonical_flat":
            self.state_space_manager = None
            float_limit = np.finfo(np.float32).max
            self.observation_space = spaces.Box(
                low=-float_limit,
                high=float_limit,
                shape=(self.canonical_shape.flat_size,),
                dtype=np.float32,
            )
        else:
            self.state_space_manager = StateSpace(
                num_trucks=self.num_trucks,
                num_stops=stops_for_action_space,
                max_time=self.max_time,
                num_charging_nodes=self.num_charging_nodes,
            )
            self.observation_space = self.state_space_manager.observation_space

        # Event-driven simulation state
        self.global_clock = 0.0  # Current simulation time
        self.event_queue = []  # Priority queue of events (min-heap)
        self.active_truck_id = None  # ID of truck that needs to make a decision
        self.truck_ready_times = {}  # truck_id -> time when TRUCK_READY event fired (actual ready time)

        # Current episode state
        self.trucks = []
        self.truck_states = (
            {}
        )  # truck_id -> "ready", "routing", "waiting_to_charge", "charging", "unloading", "complete", "failed"
        self.episode_reward = 0.0
        self.episode_steps = 0  # Track number of steps in current episode
        self.waiting_start_times = {}  # Track when trucks enter waiting_to_charge state
        self.waiting_penalty_buffer = 0.0  # Buffer for waiting penalty to apply on next step
        self.scenario_seed: int | None = None
        self.scenario_random_streams: ScenarioRandomStreams | None = None
        self.instance_rng: np.random.Generator | None = None
        self.joint_instance: JointRoutingInstance | None = None
        self.task_registry = None
        self.joint_idle_trucks: set[int] = set()
        self.last_action_feasibility = None
        self.termination_reason: str | None = None
        self.invalid_action_count = 0
        self.scenario_descriptor: dict | None = None

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        if seed is None:
            scenario_seed = int(
                self.np_random.integers(0, np.iinfo(np.int64).max)
            )
        else:
            scenario_seed = int(seed)

        self.scenario_seed = scenario_seed
        self.scenario_random_streams = ScenarioRandomStreams(scenario_seed)
        self.instance_rng = self.scenario_random_streams.generator(
            "instance_generation"
        )

        self.traffic_simulator.reset_scenario(
            scenario_seed, self.scenario_random_streams
        )
        self.delivery_simulator.reset_scenario(
            scenario_seed, self.scenario_random_streams
        )

        # Sample per-episode number of stops if enabled
        if self.allow_variable_num_stops:
            # Always at least one stop; upper bound fixed_num_stops
            self.num_stops = int(
                self.instance_rng.integers(1, self.fixed_num_stops + 1)
            )
        else:
            self.num_stops = self.fixed_num_stops

        # Reset simulation time and event queue
        self.global_clock = 0.0
        self.event_queue = []
        self.episode_reward = 0.0
        self.episode_steps = 0  # Reset step counter
        self.waiting_start_times = {}  # Reset waiting time tracking
        self.waiting_penalty_buffer = 0.0  # Reset waiting penalty buffer
        self.truck_ready_times = {}  # Reset truck ready time tracking
        self.joint_idle_trucks = set()
        self.last_action_feasibility = None
        self.termination_reason = None
        self.invalid_action_count = 0
        self.scenario_descriptor = None

        # Reset charging station state
        self.charging_station.reset()

        # Track actual routes for visualization
        self.truck_routes = {}  # truck_id -> list of (node, time, event_type)
        self.truck_initial_plans = (
            {}
        )  # truck_id -> {'start': node, 'deliveries': [nodes]}

        # Create either one fleet-owned customer instance or independent legacy
        # truck routes. The latter remains available as a secondary benchmark.
        self.trucks = []
        self.truck_states = {}
        if self.joint_routing:
            self._create_joint_fleet()
        else:
            self.joint_instance = None
            self.task_registry = None

        for i in range(self.num_trucks):
            if not self.joint_routing:
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

        self.scenario_descriptor = build_scenario_descriptor(self)

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

    def _create_joint_fleet(self) -> None:
        """Create trucks sharing one depot and one fleet-owned task registry."""
        if self.instance_rng is None:
            raise RuntimeError("joint fleet generation requires an episode RNG")

        truck_config = self.config["truck"]
        payload_capacity = float(
            self.problem_config.get(
                "payload_capacity",
                truck_config.get("payload_capacity", 0.0),
            )
        )
        if payload_capacity <= 0.0:
            raise ValueError(
                "joint_fleet mode requires a positive problem.payload_capacity "
                "or truck.payload_capacity"
            )

        battery_capacity = float(truck_config["battery_capacity"])
        self.joint_instance = generate_joint_routing_instance(
            transport_graph=self.transport_graph,
            charging_nodes=self.charging_nodes,
            rng=self.instance_rng,
            num_customers=self.num_stops,
            num_trucks=self.num_trucks,
            battery_capacity=battery_capacity,
            payload_capacity=payload_capacity,
            min_customer_demand=float(
                self.problem_config.get("min_customer_demand", 1.0)
            ),
            max_customer_demand=float(
                self.problem_config.get("max_customer_demand", payload_capacity)
            ),
            base_service_time=float(
                self.problem_config.get(
                    "base_service_time",
                    self.delivery_config["base_unloading_time"],
                )
            ),
            time_window_config=self.problem_config.get("time_windows"),
        )
        self.task_registry = self.joint_instance.create_registry()
        shared_sequence = [self.joint_instance.depot_node] + [
            task.node_id for task in self.joint_instance.tasks
        ]

        for truck_id in range(self.num_trucks):
            truck = Truck(
                truck_id=truck_id,
                truck_type="electric",
                delivery_sequence=shared_sequence,
                initial_battery=self._initial_battery(battery_capacity),
                battery_capacity=battery_capacity,
                base_speed=float(truck_config["base_speed"]),
                enable_flexible_delivery_order=True,
                payload_capacity=payload_capacity,
            )
            self.trucks.append(truck)
            initial_soc = truck.get_battery_percentage()
            self.truck_routes[truck_id] = [
                (self.joint_instance.depot_node, 0.0, "start", initial_soc)
            ]
            self.truck_initial_plans[truck_id] = {
                "start": self.joint_instance.depot_node,
                "deliveries": shared_sequence.copy(),
            }

    def _initial_battery(self, battery_capacity: float) -> float:
        """Resolve the configured initial SoC using the instance RNG."""
        setting = self.config["truck"]["initial_battery"]
        if setting == "full":
            return battery_capacity
        if setting == "random":
            if self.instance_rng is None:
                raise RuntimeError("random initial battery requires an episode RNG")
            return float(self.instance_rng.uniform(0.3, 1.0) * battery_capacity)
        if isinstance(setting, (int, float)):
            percentage = float(setting)
            if (
                isinstance(setting, bool)
                or not math.isfinite(percentage)
                or not 0.0 <= percentage <= 100.0
            ):
                raise ValueError(
                    "numeric truck.initial_battery must be in [0, 100]"
                )
            return percentage / 100.0 * battery_capacity
        raise ValueError(f"invalid truck.initial_battery setting: {setting!r}")

    def _complete_joint_customer_service(
        self,
        truck: Truck,
        event_data: dict,
    ) -> None:
        """Commit a claimed task after unloading and update fleet completion."""
        if self.task_registry is None or self.joint_instance is None:
            raise RuntimeError("joint task registry is not initialized")

        node_id = int(event_data["customer_node"])
        expected_task_id = int(event_data["task_id"])
        task = self.task_registry.task_for_node(node_id)
        if task.task_id != expected_task_id:
            raise RuntimeError(
                f"service event task {expected_task_id} does not match node "
                f"{node_id} task {task.task_id}"
            )

        task = self.task_registry.complete_service(
            node_id,
            truck_id=truck.truck_id,
            timestamp=self.global_clock,
        )
        truck.complete_customer_service(
            task_id=task.task_id,
            demand=task.demand,
            timestamp=self.global_clock,
            node_id=node_id,
        )

        # The current legacy encoders infer served nodes from each truck. Keep
        # those compatibility views synchronized until the canonical feature
        # extractor replaces them.
        for fleet_truck in self.trucks:
            fleet_truck.delivered_nodes.add(node_id)

        if not self.task_registry.all_served():
            self._wake_joint_idle_trucks()
            return

        depot_node = self.joint_instance.depot_node
        for fleet_truck in self.trucks:
            if fleet_truck.failed or fleet_truck.is_complete:
                continue
            if (
                int(fleet_truck.current_node) == depot_node
                and fleet_truck.route_destination is None
            ):
                fleet_truck.return_to_depot_pending = False
                fleet_truck.battery_at_completion = fleet_truck.current_battery
                fleet_truck.mark_complete(timestamp=self.global_clock)
                self.truck_states[fleet_truck.truck_id] = "complete"
            else:
                fleet_truck.return_to_depot_pending = True
        self._wake_joint_idle_trucks()

    def _wake_joint_idle_trucks(self) -> None:
        """Wake idle trucks when task availability or depot-return state changes."""
        if self.task_registry is None:
            return
        for truck_id in sorted(self.joint_idle_trucks):
            truck = self.trucks[truck_id]
            can_take_task = bool(
                self.task_registry.available_tasks(truck.remaining_payload)
            )
            if not (can_take_task or self.task_registry.all_served()):
                continue
            heapq.heappush(
                self.event_queue,
                Event(
                    time=self.global_clock,
                    event_type=EventType.TRUCK_READY,
                    truck_id=truck_id,
                    data={"reason": "fleet_task_state_changed"},
                ),
            )
            self.joint_idle_trucks.remove(truck_id)

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
            enable_flexible_delivery_order=self.enable_flexible_delivery_order,
            rng=self.instance_rng,
        )

        self.trucks.append(truck)

        # Store initial plan for visualization
        # Include initial SoC (should be 100% at start)
        initial_soc = truck.get_battery_percentage()
        self.truck_routes[truck_id] = [(start_node, 0.0, "start", initial_soc)]
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
            if event.time < self.global_clock - 1e-9:
                raise RuntimeError(
                    f"event time reversal: {event.time} < {self.global_clock}"
                )
            self.global_clock = event.time

            # Process event
            if event.event_type == EventType.TRUCK_READY:
                truck = self.trucks[event.truck_id]

                # Safety check: skip if truck is already complete or failed
                if truck.is_complete or truck.failed:
                    if self.verbose:
                        status = "complete" if truck.is_complete else "failed"
                        print(
                            f"  Skipping TRUCK_READY for truck {truck.truck_id} (status: {status})"
                        )
                    continue

                # Skip if this is a stale wake event and truck is no longer waiting
                # This can happen when a truck gets woken early (port freed) but also had
                # a scheduled event based on predicted wait time
                reason = event.data.get("reason", "")
                if reason == "service_window_open":
                    if not self.joint_routing or self.task_registry is None:
                        raise RuntimeError(
                            "service-window event requires a joint task registry"
                        )
                    if self.truck_states.get(truck.truck_id) != "waiting_for_service":
                        # A failure or cancellation can leave a stale opening event.
                        continue
                    customer_node = int(event.data["customer_node"])
                    task = self.task_registry.task_for_node(customer_node)
                    expected_task_id = int(event.data["task_id"])
                    if task.task_id != expected_task_id:
                        raise RuntimeError("service-window event task mismatch")
                    self.task_registry.start_service(
                        customer_node,
                        truck_id=truck.truck_id,
                        timestamp=self.global_clock,
                    )
                    wait_duration = float(event.data.get("wait_duration", 0.0))
                    truck.add_time_window_waiting(
                        wait_duration,
                        timestamp=self.global_clock,
                    )
                    unloading_duration = float(
                        event.data.get("unloading_duration", 0.0)
                    )
                    truck.start_unloading(
                        timestamp=self.global_clock,
                        delivery_node=customer_node,
                    )
                    heapq.heappush(
                        self.event_queue,
                        Event(
                            time=self.global_clock + unloading_duration,
                            event_type=EventType.TRUCK_READY,
                            truck_id=truck.truck_id,
                            data={
                                "reason": "unloading_complete",
                                "unloading_duration": unloading_duration,
                                "task_id": expected_task_id,
                                "customer_node": customer_node,
                            },
                        ),
                    )
                    self.truck_states[truck.truck_id] = "unloading"
                    continue
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
                    initial_soc = event.data.get("initial_soc", 0.0)
                    charging_details = event.data.get("charging_details", {})

                    # Complete charging for the truck (update battery and record event)
                    truck.finish_charging(
                        charge_amount=charge_amount,
                        charge_duration=charge_duration,
                        timestamp=self.global_clock
                    )
                    
                    # Log charging session if logger is enabled
                    if self.charging_logger:
                        final_soc = truck.get_battery_percentage() / 100.0
                        charger_type = self.charging_station.charger_type[charger_node]
                        self.charging_logger.log_charging_session(
                            truck_id=truck.truck_id,
                            charger_node=charger_node,
                            charger_type=charger_type,
                            start_time=self.global_clock - charge_duration,
                            end_time=self.global_clock,
                            initial_soc=initial_soc,
                            final_soc=final_soc,
                            charge_amount=charge_amount,
                            battery_capacity=truck.battery_capacity,
                            charging_details=charging_details
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
                        truck_states=self.truck_states,
                    )
                    
                    # Truck just finished charging - skip gating check and become ready immediately
                    # The truck should leave the charger, not wait for other trucks
                    # Continue to make this truck active (skip the gating check below)
                    skip_gating_check = True
                else:
                    skip_gating_check = False

                # Charger gating: enforce FCFS waitlist with capacity ports
                # Skip this check if truck just finished charging
                node = int(truck.current_node)
                if (
                    not skip_gating_check
                    and node in self.charging_nodes
                    and self.charging_station.station_available[node]
                ):
                    can_proceed, next_check_time = (
                        self.charging_station.check_charger_gating(
                            truck_id=truck.truck_id,
                            charger_node=node,
                            global_clock=self.global_clock,
                        )
                    )

                    if not can_proceed:
                        # Update state to waiting_to_charge
                        self.truck_states[truck.truck_id] = "waiting_to_charge"
                        
                        # Track when truck starts waiting (if not already tracked)
                        if truck.truck_id not in self.waiting_start_times:
                            self.waiting_start_times[truck.truck_id] = self.global_clock
                            # Record waiting start event
                            truck.start_waiting(timestamp=self.global_clock, reason="charger_queue")

                        # Pure event-driven: truck will be woken by wake_waiting_trucks
                        # No time-based predictions or scheduled rechecks
                        if self.verbose:
                            print(
                                f"  Truck {truck.truck_id} waiting for charge port at node {node} at time {self.global_clock:.2f}h"
                            )
                            print("    Will be woken when port becomes available")
                            # Print the charger queue status for this specific charger
                            self.charging_station.print_charger_queue(node)
                                
                        continue

                # Not at a charger or gating passed - truck is ready for decision
                # Calculate waiting penalty if truck was waiting
                if truck.truck_id in self.waiting_start_times:
                    waiting_duration = self.global_clock - self.waiting_start_times[truck.truck_id]
                    if waiting_duration > 0:
                        # Calculate time penalty for waiting
                        waiting_penalty = -waiting_duration * self.reward_config["time_multiplier"]
                        self.waiting_penalty_buffer = waiting_penalty
                        
                        # Update truck's waiting time stat and record event
                        truck.add_waiting_time(waiting_duration, timestamp=self.global_clock)
                        
                        if self.verbose:
                            print(f"  Truck {truck.truck_id} finished waiting at {self.global_clock:.2f}h")
                            print(f"    Waited: {waiting_duration:.2f}h")
                            print(f"    Waiting penalty (to be applied on next action): {waiting_penalty:.2f}")
                    
                    # Clear waiting start time
                    del self.waiting_start_times[truck.truck_id]
                
                # Handle unloading completion
                if reason == "unloading_complete":
                    unloading_duration = event.data.get("unloading_duration", 0.0)
                    if unloading_duration > 0:
                        truck.finish_unloading(unloading_duration=unloading_duration, timestamp=self.global_clock)
                    if self.joint_routing:
                        self._complete_joint_customer_service(truck, event.data)

                if (
                    self.joint_routing
                    and self.task_registry is not None
                    and not self.task_registry.all_served()
                    and not self.task_registry.available_tasks(
                        truck.remaining_payload
                    )
                ):
                    truck.mark_ready(
                        timestamp=self.global_clock,
                        reason="waiting_for_fleet_task",
                    )
                    self.truck_states[truck.truck_id] = "waiting_for_task"
                    self.joint_idle_trucks.add(truck.truck_id)
                    continue
                
                # Mark truck as ready with appropriate reason
                truck.mark_ready(timestamp=self.global_clock, reason=reason if reason else "unknown")
                
                # Final check: ensure truck is not complete or failed before setting as active
                # This can happen if a TRUCK_READY event was scheduled before truck completed
                if truck.is_complete or truck.failed:
                    if self.verbose:
                        status = "complete" if truck.is_complete else "failed"
                        print(f"  Skipping TRUCK_READY for truck {truck.truck_id} - just became {status}")
                    continue
                
                # Store the actual time when this truck became ready (event.time, not global_clock)
                # This fixes the bug where global_clock advances during event processing
                self.truck_ready_times[event.truck_id] = event.time
                self.truck_states[truck.truck_id] = "ready"
                self.active_truck_id = event.truck_id
                if self.joint_routing:
                    decisions = joint_action_feasibility(self)
                    self.last_action_feasibility = decisions
                    if not any(item.feasible for item in decisions):
                        truck.mark_failed(
                            reason="no_feasible_action",
                            timestamp=self.global_clock,
                        )
                        self.truck_states[truck.truck_id] = "failed"
                        if self.termination_reason is None:
                            self.termination_reason = "no_feasible_action"
                        self.active_truck_id = None
                        continue

                return

            elif event.event_type == EventType.TRUCK_ROUTING:
                # Handle truck arrival at destination
                destination = event.data["destination"]
                truck = self.trucks[event.truck_id]
                
                # Skip if truck is already complete or failed
                if truck.is_complete or truck.failed:
                    if self.verbose:
                        status = "complete" if truck.is_complete else "failed"
                        print(f"  Skipping TRUCK_ROUTING for truck {truck.truck_id} (status: {status})")
                    continue

                # First, update the truck's physical state (position, battery, etc.)
                self.event_handler.handle_truck_routing(
                    event,
                    self.trucks,
                    self.truck_states,
                    self.truck_routes,
                    self.event_queue,
                    self.global_clock,
                    self.enable_plotting,
                    self.delivery_simulator,
                    self.task_registry,
                )
                if self.joint_routing and truck.failed:
                    self._wake_joint_idle_trucks()

                # Check the truck's state after arrival - it may have become complete or failed
                # Only schedule TRUCK_READY if truck is not complete or failed
                if not (truck.is_complete or truck.failed):
                    # If truck arrived at a charger, check if port is available
                    if (
                        destination in self.charging_nodes
                        and self.charging_station.station_available[destination]
                    ):
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
                                # Record waiting start event
                                truck.start_waiting(timestamp=self.global_clock, reason="charger_queue")

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
                                        "    Will be woken when port becomes available"
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
                    # Check if truck is in unloading state (already scheduled by event_handler)
                    elif self.truck_states[truck.truck_id] not in {
                        "unloading",
                        "waiting_for_service",
                    }:
                        # Arrived at delivery node (but not unloading) - schedule immediate TRUCK_READY
                        # If truck is unloading, the event_handler already scheduled TRUCK_READY
                        heapq.heappush(
                            self.event_queue,
                            Event(
                                time=self.global_clock,
                                event_type=EventType.TRUCK_READY,
                                truck_id=truck.truck_id,
                                data={"reason": "arrived_at_delivery"},
                            ),
                        )

        # No more events - episode is over. Successful completion and an
        # unserved-customer deadlock are distinct terminal outcomes.
        self.active_truck_id = None
        if self.joint_routing and self.task_registry is not None:
            if self.task_registry.all_served() and all(
                truck.is_complete for truck in self.trucks
            ):
                self.termination_reason = "success"
            elif not self.task_registry.all_served():
                if self.termination_reason is None:
                    unassigned = self.task_registry.available_tasks()
                    live_trucks = [
                        truck
                        for truck in self.trucks
                        if not truck.failed and not truck.is_complete
                    ]
                    if unassigned and not any(
                        truck.can_accept_demand(task.demand)
                        for truck in live_trucks
                        for task in unassigned
                    ):
                        self.termination_reason = "payload_capacity_deadlock"
                    else:
                        self.termination_reason = (
                            "no_events_with_unserved_customers"
                        )
                for truck in self.trucks:
                    if not truck.is_complete and not truck.failed:
                        truck.mark_failed(
                            reason=self.termination_reason,
                            timestamp=self.global_clock,
                        )
                        self.truck_states[truck.truck_id] = "failed"

    def step(self, action: int | tuple[int, float, bool]) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step for the active truck.

        Args:
            action: Action for the active truck. Can be either:
                    - Integer (legacy format): 
                      * 0 to num_charging_nodes-1: Go to charging station
                      * num_charging_nodes: Go to next delivery
                      * num_charging_nodes+1 to end: Charge for 1-4 hours at current location
                    - Tuple (new GNN format): (node_id, charge_value, is_charging)
                      * node_id: Target node to navigate to or charge at
                      * charge_value: Target SoC in joint mode or hours in legacy mode
                      * is_charging: Whether this is a charging action

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.active_truck_id is None:
            # No active truck - episode is over
            return self._get_observation(), 0.0, True, False, self._get_info()

        truck = self.trucks[self.active_truck_id]
        reward = 0.0
        
        # Decode action format
        if isinstance(action, tuple):
            # New GNN format: (node_id, charging_duration, is_charging)
            node_id, charge_value, is_charging = action
            charge_label = (
                f"target_soc={charge_value:.0%}"
                if is_charging and self.charging_action_mode == "target_soc"
                else f"charge_hours={charge_value:.2f}"
            )
            action_str = (
                f"{'CHARGE' if is_charging else 'ROUTE'} at node {node_id}, "
                f"{charge_label}"
            )
        else:
            # Legacy integer format
            action_str = self._action_to_string(action)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"STEP at t={self.global_clock:.2f}h - Truck {self.active_truck_id}")
            print(
                f"Current Node: {truck.current_node}, SoC: {truck.get_battery_percentage():.1f}%"
            )
            print(f"Action: {action_str}")
            print(f"Event Queue: {self.event_queue}")
            print(f"{'='*80}")

        # Add any buffered waiting penalty from previous wait
        if self.waiting_penalty_buffer != 0.0:
            reward += self.waiting_penalty_buffer
            if self.verbose:
                print(f"  Adding waiting penalty from queue: {self.waiting_penalty_buffer:.2f}")
            self.waiting_penalty_buffer = 0.0  # Clear the buffer

        # Execute action based on format
        if isinstance(action, tuple):
            # New GNN format: (node_id, charging_duration, is_charging)
            node_id, charge_value, is_charging = action
            if is_charging:
                # Charging action at specified node
                reward += self._execute_charge_action(truck, charge_value, node_id)
            else:
                # Navigation action to specified node
                reward += self._execute_navigation_action(truck, node_id)
        else:
            # Legacy integer format - convert to node_id format
            if action < self.num_navigation_actions:
                # Navigation action
                if action < self.num_charging_nodes:
                    # Go to charging station
                    target_node = self.charging_nodes[action]
                else:
                    # Go to delivery
                    if self.enable_flexible_delivery_order:
                        # Flexible mode: decode which delivery from action index
                        delivery_idx = action - self.num_charging_nodes
                        # Map action index to delivery node in sequence (skip depot at index 0)
                        if delivery_idx < len(truck.delivery_sequence) - 1:
                            target_node = truck.delivery_sequence[delivery_idx + 1]
                        elif delivery_idx == len(truck.delivery_sequence) - 1:
                            if truck.return_to_depot_pending:
                                target_node = truck.delivery_sequence[0]
                            else:
                                target_node = self._select_closest_delivery(truck)
                        else:
                            raise ValueError(f"Invalid delivery action index: {delivery_idx}")
                    else:
                        # Sequential mode: go to next delivery
                        target_node = truck.get_next_delivery_target()
                        if target_node is None:
                            raise ValueError("No remaining deliveries for truck")
                reward += self._execute_navigation_action(truck, target_node)
            else:
                # Charging action
                charge_idx = action - self.num_navigation_actions
                charge_value = self.charge_action_values[charge_idx]
                reward += self._execute_charge_action(
                    truck,
                    charge_value,
                    truck.current_node,
                )

        # Accumulate reward
        self.episode_reward += reward
        self.episode_steps += 1  # Increment step counter

        # Advance to next decision point
        self._advance_to_next_decision()
        
        if self.verbose:
            # print status of all trucks
            print("\nTruck statuses after step:")
            for t in self.trucks:
                state = self.truck_states[t.truck_id]
                print(f"  Truck {t.truck_id}: State={state}, Battery={t.current_battery:.1f} kWh, Completed={t.is_complete}, Failed={t.failed}")

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

    def _execute_navigation_action(self, truck: Truck, target_node: int) -> float:
        """Execute navigation action and schedule route completion event."""
        # Convert numpy types to native Python int
        if hasattr(target_node, "item"):
            target_node = int(target_node.item())
        else:
            target_node = int(target_node)

        if self.joint_routing:
            if self.task_registry is None or self.joint_instance is None:
                raise RuntimeError("joint-routing state is not initialized")
            energy_multiplier = 1.0
            if (
                self.traffic_config["enable_traffic"]
                and self.traffic_config["enable_energy_uncertainty"]
            ):
                energy_multiplier = max(
                    1.0,
                    float(self.traffic_config["max_energy_multiplier"]),
                )
            feasibility = evaluate_joint_route(
                truck=truck,
                truck_state=self.truck_states[truck.truck_id],
                target_node=target_node,
                transport_graph=self.transport_graph,
                charging_nodes=self.charging_nodes,
                task_registry=self.task_registry,
                depot_node=self.joint_instance.depot_node,
                energy_multiplier=energy_multiplier,
                current_time=self.global_clock,
                unavailable_charging_nodes={
                    int(node)
                    for node, available in (
                        self.charging_station.station_available.items()
                    )
                    if not available
                },
            )
            if not feasibility.feasible:
                return self._reject_joint_action(truck, feasibility.reason)

        if self.enable_flexible_delivery_order:
            depot_node = int(truck.delivery_sequence[0])
            if truck.return_to_depot_pending:
                # Force any delivery navigation to go to depot once return is required.
                if target_node != depot_node and target_node not in self.charging_nodes:
                    target_node = depot_node
            elif target_node == depot_node:
                target_node = self._select_closest_delivery(truck)

        current_node = int(truck.current_node)

        # Check if already at target
        if current_node == target_node:
            if self.verbose:
                print(f"  Already at target node {target_node}")
            # If at a charger, default to charging for 1 hour
            if target_node in self.charging_nodes:
                return self._execute_charge_action(truck, 1.0, target_node)
            else:
                # Already at delivery - no movement needed, return 0 reward
                if self.verbose:
                    print(f"  Already at delivery node {target_node}, no movement needed")
                return 0.0

        # Calculate energy and time for the trip
        energy_used = self.transport_graph.get_path_energy(current_node, target_node)

        # Check if path is reachable
        if energy_used == float("inf"):
            raise ValueError("No valid path for navigation action")

        travel_time = self.transport_graph.get_time_distance(current_node, target_node)
        discharge = energy_used
        distance = travel_time * truck.base_speed

        # Apply traffic simulation if enabled (returns time and multiplier for correlation)
        actual_travel_time, traffic_multiplier = self.traffic_simulator.apply_traffic(
            travel_time=travel_time,
            current_time=self.global_clock,
            from_node=current_node,
            to_node=target_node
        )
        
        # Apply energy uncertainty correlated with traffic conditions
        discharge = self.traffic_simulator.apply_energy_uncertainty(
            base_energy=discharge,
            traffic_multiplier=traffic_multiplier,
            current_time=self.global_clock,
            from_node=current_node,
            to_node=target_node
        )
        
        # Check if truck can make it
        if discharge > truck.current_battery:
            if self.verbose:
                print(f"  ERROR: Insufficient battery ({truck.current_battery:.1f} kWh < {discharge:.1f} kWh needed)")
            truck.mark_failed(
                reason="insufficient_energy_after_realization",
                timestamp=self.global_clock,
            )
            self.truck_states[truck.truck_id] = "failed"
            return self.reward_config["failure_penalty"]
        
        # Determine if this is navigation to a charger or delivery.
        is_charger_nav = target_node in self.charging_nodes
        next_delivery = truck.get_next_delivery_target()
        joint_task = None
        if self.joint_routing and self.task_registry is not None:
            try:
                joint_task = self.task_registry.task_for_node(target_node)
            except KeyError:
                joint_task = None
            is_delivery_nav = joint_task is not None
            if is_delivery_nav:
                if not joint_task.is_available:
                    raise ValueError(
                        f"customer task at node {target_node} is not available"
                    )
                if not truck.can_accept_demand(joint_task.demand):
                    raise ValueError(
                        f"truck {truck.truck_id} lacks payload for customer "
                        f"{target_node} demand {joint_task.demand:.3f}"
                    )
        elif self.enable_flexible_delivery_order:
            # Flexible mode: check if target is any remaining delivery
            remaining_deliveries = next_delivery if isinstance(next_delivery, list) else []
            is_delivery_nav = target_node in remaining_deliveries
        else:
            # Sequential mode: check if target is next delivery
            is_delivery_nav = (next_delivery is not None and target_node == next_delivery)

        # Track detour loop state (charge -> charger without delivery)
        just_charged = getattr(truck, "detour_last_action_was_charge", False)
        if is_delivery_nav and not self.joint_routing:
            truck.detour_charger_hops_since_delivery = 0
            truck.detour_last_action_was_charge = False
        elif is_charger_nav:
            if just_charged:
                truck.detour_charger_hops_since_delivery += 1
            truck.detour_last_action_was_charge = False
        else:
            truck.detour_last_action_was_charge = False
        
        # If navigating to a non-terminal delivery, check if truck will have feasible actions after arrival
        if is_delivery_nav and not self.joint_routing:
            # Get energy safety factor for feasibility check
            energy_safety_factor = 1.0
            if self.traffic_config['enable_traffic'] and self.traffic_config['enable_energy_uncertainty']:
                energy_safety_factor = self.traffic_config['max_energy_multiplier']
            
            is_feasible = check_navigation_feasibility(
                truck=truck,
                target_node=target_node,
                discharge=discharge,
                transport_graph=self.transport_graph,
                charging_nodes=self.charging_nodes,
                energy_safety_factor=energy_safety_factor,
                verbose=self.verbose
            )
            
            if not is_feasible:
                truck.mark_failed(
                    reason="legacy_navigation_lookahead_failed",
                    timestamp=self.global_clock,
                )
                self.truck_states[truck.truck_id] = "failed"
                return self.reward_config["failure_penalty"]

        if is_charger_nav and self.verbose:
            charger_info = self.charging_station.get_charger_info(target_node, self.global_clock)
            print(f"  Going to charger @ node {target_node}")
            print(f"    Current occupancy: {charger_info['current_occupancy']}/{charger_info['capacity']}")

        # If leaving a charger to navigate elsewhere, remove from its waitlist and wake others
        if (not is_charger_nav) and (current_node in self.charging_nodes):
            self.charging_station.remove_from_waitlist(truck.truck_id, current_node)
            self.charging_station.wake_waiting_trucks(
                charger_node=current_node,
                global_clock=self.global_clock,
                event_queue=self.event_queue,
                EventType=EventType,
                Event=Event,
                truck_states=self.truck_states,
            )

        # Remove any existing routing events for this truck before scheduling new one
        self._remove_pending_events(truck.truck_id, EventType.TRUCK_ROUTING)
        
        # Schedule truck routing (arrival) event
        # BUG FIX: Use the actual time when truck became ready (event.time from TRUCK_READY event)
        # not the current global_clock which may have advanced during event processing
        departure_time = self.truck_ready_times.get(
            truck.truck_id, self.global_clock
        )
        completion_time = departure_time + actual_travel_time

        actual_unloading_time = 0.0
        depot_node = int(truck.delivery_sequence[0])
        if (
            is_delivery_nav
            and self.delivery_simulator is not None
            and target_node != depot_node
        ):
            actual_unloading_time = self.delivery_simulator.apply_unloading_time(
                delivery_node=target_node,
                current_time=completion_time,
            )

        if joint_task is not None:
            self.task_registry.claim(
                target_node,
                truck_id=truck.truck_id,
                timestamp=departure_time,
            )

        time_window_wait = 0.0
        realized_time_window_violation = False
        if joint_task is not None:
            time_window_wait = max(
                0.0,
                joint_task.earliest_service - completion_time,
            )
            realized_time_window_violation = (
                completion_time > joint_task.latest_service + 1e-9
            )
        
        # Record routing start event
        truck.start_routing(destination=target_node, timestamp=departure_time)
        
        if self.verbose:
            print(f"  DEBUG: Scheduling arrival - Departure: {departure_time:.4f}h, Travel: {actual_travel_time:.4f}h, Arrival: {completion_time:.4f}h")
        
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
                    "departure_time": departure_time,
                    "unloading_time": actual_unloading_time,
                    "task_id": (
                        joint_task.task_id if joint_task is not None else None
                    ),
                },
            ),
        )

        # Update truck state and track route information
        self.truck_states[truck.truck_id] = "routing"
        truck.route_destination = target_node
        truck.route_arrival_time = completion_time

        if self.verbose:
            print(f"  Routing to node {target_node}")
            print(f"    Distance: {distance:.2f} km, Time: {actual_travel_time:.2f}h (base: {travel_time:.2f}h)")
            print(f"    Battery: {truck.current_battery:.1f} → {truck.current_battery - discharge:.1f} kWh")
            print(f"    Will arrive at t={completion_time:.2f}h")

        # Calculate reward (using actual travel time, not base time)
        time_penalty = -actual_travel_time * self.reward_config["time_multiplier"]

        # Bonus if this is a delivery
        if is_delivery_nav:
            if realized_time_window_violation:
                return time_penalty + self.reward_config["failure_penalty"]
            delivery_bonus = self.reward_config["delivery_bonus"]
            # Use the same service-time realization carried by the arrival event.
            unloading_penalty = -actual_unloading_time * self.reward_config["time_multiplier"]
            window_wait_penalty = (
                -time_window_wait * self.reward_config["time_multiplier"]
            )
            
            # Check if this is the last delivery for this truck
            # Apply leftover battery penalty if enabled
            leftover_battery_penalty = 0.0
            remaining_deliveries = truck.get_remaining_deliveries()
            is_last_delivery = (
                target_node != depot_node
                and len(remaining_deliveries) == 1
                and target_node in remaining_deliveries
            )
            
            if is_last_delivery and self.reward_config["enable_leftover_battery_penalty"]:
                # Calculate expected battery at completion (after this delivery)
                expected_battery_at_completion = truck.current_battery - discharge
                remaining_soc_percentage = (expected_battery_at_completion / truck.battery_capacity) * 100.0
                penalty_coef = self.reward_config["leftover_battery_penalty_coef"]
                leftover_battery_penalty = -penalty_coef * remaining_soc_percentage
                
                if self.verbose:
                    print(f"  Last delivery! Expected SOC at completion: {remaining_soc_percentage:.1f}%")
                    print(f"  Leftover battery penalty: {leftover_battery_penalty:.2f}")
            
            return (
                time_penalty
                + delivery_bonus
                + unloading_penalty
                + window_wait_penalty
                + leftover_battery_penalty
            )

        return time_penalty

    def _reject_joint_action(
        self,
        truck: Truck,
        reason: FeasibilityReason,
    ) -> float:
        """Fail an explicit invalid action without silently changing its meaning."""
        cause = f"invalid_action:{reason.value}"
        self.invalid_action_count += 1
        truck.mark_failed(reason=cause, timestamp=self.global_clock)
        self.truck_states[truck.truck_id] = "failed"
        if self.termination_reason is None:
            self.termination_reason = cause
        return float(self.reward_config["failure_penalty"])



    def _execute_charge_action(self, truck: Truck, charge_value: float, charger_node: int) -> float:
        """Execute a target-SoC or legacy duration charging action."""
        # Convert to int if needed
        if hasattr(charger_node, "item"):
            charger_node = int(charger_node.item())
        else:
            charger_node = int(charger_node)

        if self.joint_routing:
            if self.charging_action_mode == "target_soc":
                feasibility = evaluate_target_soc_charge(
                    truck=truck,
                    truck_state=self.truck_states[truck.truck_id],
                    charger_node=charger_node,
                    charging_nodes=self.charging_nodes,
                    target_soc=float(charge_value),
                    station_available=self.charging_station.station_available.get(
                        charger_node,
                        False,
                    ),
                )
            else:
                feasibility = evaluate_duration_charge(
                    truck=truck,
                    truck_state=self.truck_states[truck.truck_id],
                    charger_node=charger_node,
                    charging_nodes=self.charging_nodes,
                    charge_hours=float(charge_value),
                    station_available=self.charging_station.station_available.get(
                        charger_node,
                        False,
                    ),
                )
            if not feasibility.feasible:
                return self._reject_joint_action(truck, feasibility.reason)
        
        # Validate truck is at a charger
        if truck.current_node not in self.charging_nodes:
            if self.verbose:
                print(f"  WARNING: Truck not at charging station (current: {truck.current_node})")
                print("  Fallback: Navigating to next delivery instead")
            # Navigate to next delivery instead
            next_delivery = truck.get_next_delivery_target()
            if next_delivery is not None:
                # In flexible order, next_delivery can be a list; pick a concrete target
                if isinstance(next_delivery, list):
                    if not next_delivery:
                        raise RuntimeError("No remaining deliveries to navigate to from fallback charge")
                    target_node = next_delivery[0]
                else:
                    target_node = next_delivery
                return self._execute_navigation_action(truck, target_node)
            raise RuntimeError("Truck attempted to charge while not at a charging station")
        
        # Validate charger_node matches current location
        if charger_node != truck.current_node:
            if self.verbose:
                print(f"  WARNING: Charger node {charger_node} doesn't match current {truck.current_node}")
                print(f"  Using current location {truck.current_node}")
            charger_node = truck.current_node

        # If battery full, go to next delivery instead
        battery_deficit = truck.battery_capacity - truck.current_battery
        if battery_deficit <= 1e-3:
            next_delivery = truck.get_next_delivery_target()
            if self.verbose:
                print(f"  Truck {truck.truck_id} battery full; rerouting instead of charging")
            if next_delivery is not None:
                if isinstance(next_delivery, list):
                    if not next_delivery:
                        raise RuntimeError("No remaining deliveries to navigate to after full battery check")
                    target_node = next_delivery[0]
                else:
                    target_node = next_delivery
                return self._execute_navigation_action(truck, target_node)
            return 0.0

        # Check charger gating
        can_proceed, next_check_time = self.charging_station.check_charger_gating(
            truck_id=truck.truck_id,
            charger_node=charger_node,
            global_clock=self.global_clock,
        )

        if not can_proceed:
            print(f"  !!!WARNING: Charger gating - no free port at node {charger_node}!!!!!!!\n\n")
            # Port not available: move to waiting state and let wake_waiting_trucks handle it.
            if self.verbose:
                print(f"  WARNING: Cannot charge - no free port at node {charger_node}")
            self.truck_states[truck.truck_id] = "waiting_to_charge"
            if truck.truck_id not in self.waiting_start_times:
                self.waiting_start_times[truck.truck_id] = self.global_clock
                truck.start_waiting(timestamp=self.global_clock, reason="charger_queue")
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
            return -0.01

        # Get charger configuration
        charger_type = self.charging_station.charger_type[charger_node]
        charging_config = self.config["charging"]
        
        if charger_type == "DCFast":
            charger_config_type = charging_config["dcfast"]
        else:  # Level2
            charger_config_type = charging_config["level2"]
        
        # Add global use_realistic_curve flag to charger config
        charger_config_with_curve = charger_config_type.copy()
        charger_config_with_curve["use_realistic_curve"] = charging_config["use_realistic_curve"]
        charger_config_with_curve["charge_rate"] = (
            self.charging_station.charger_power_kw[charger_node]
        )
        
        # Calculate charge using charging curve model
        # Clamp to [0.0, 1.0] to handle any floating point precision issues
        initial_soc = min(1.0, max(0.0, truck.get_battery_percentage() / 100.0))
        if self.charging_action_mode == "target_soc":
            charge_amount, charging_details = (
                self.charging_curve_model.calculate_charge_to_target(
                    initial_soc=initial_soc,
                    target_soc=float(charge_value),
                    battery_capacity=truck.battery_capacity,
                    charger_config=charger_config_with_curve,
                    charger_type=charger_type,
                )
            )
        else:
            charge_amount, charging_details = self.charging_curve_model.calculate_charge(
                initial_soc=initial_soc,
                charge_hours=float(charge_value),
                battery_capacity=truck.battery_capacity,
                charger_config=charger_config_with_curve,
                charger_type=charger_type
            )
        
        # Defensive: Ensure charge_amount doesn't exceed remaining capacity
        remaining_capacity = truck.battery_capacity - truck.current_battery
        if charge_amount > remaining_capacity:
            charge_amount = remaining_capacity
            charging_details["clamped_to_capacity"] = True
        
        # Use actual charge time from curve model
        actual_charge_hours = charging_details["actual_charge_hours"]
        
        # Start charging
        self.charging_station.start_charging(
            truck_id=truck.truck_id,
            charger_node=charger_node,
            charge_hours=actual_charge_hours,
            global_clock=self.global_clock,
        )
        
        # Remove any pending TRUCK_READY events for this truck
        self._remove_pending_events(truck.truck_id, EventType.TRUCK_READY)
        
        # Schedule charge completion
        completion_time = self.global_clock + actual_charge_hours
        heapq.heappush(
            self.event_queue,
            Event(
                time=completion_time,
                event_type=EventType.TRUCK_READY,
                truck_id=truck.truck_id,
                data={
                    "reason": "charge_complete",
                    "charge_amount": charge_amount,
                    "charge_duration": actual_charge_hours,
                    "charger_node": charger_node,
                    "initial_soc": initial_soc,
                    "charging_details": charging_details,
                },
            ),
        )
        
        # Update truck state and record charging start event
        self.truck_states[truck.truck_id] = "charging"
        truck.start_charging(current_time=self.global_clock)
        
        if self.verbose:
            print(f"  Charging for {actual_charge_hours:.2f}h")
            print(f"    Will charge {charge_amount:.1f} kWh")
            print(f"    Battery: {truck.current_battery:.1f} → {truck.current_battery + charge_amount:.1f} kWh")
            print(f"    Will complete at t={completion_time:.2f}h")
        
        # Return time penalty
        return -actual_charge_hours

    def _select_closest_delivery(self, truck: Truck) -> int:
        """Pick the closest remaining delivery (by energy) for fallback routing."""
        if self.joint_routing and self.task_registry is not None:
            remaining = [
                task.node_id
                for task in self.task_registry.available_tasks(
                    truck.remaining_payload
                )
            ]
        else:
            remaining = truck.get_remaining_deliveries()
        if not remaining:
            raise ValueError("No remaining deliveries available for fallback routing")

        current_node = int(truck.current_node)
        best_node = None
        best_energy = float("inf")
        for node_id in remaining:
            if node_id == current_node:
                return node_id
            energy = self.transport_graph.get_path_energy(current_node, node_id)
            if energy < best_energy:
                best_energy = energy
                best_node = node_id

        if best_node is None or best_energy == float("inf"):
            raise ValueError("No reachable remaining deliveries for fallback routing")

        return best_node

    def _remove_pending_events(self, truck_id: int, event_type: EventType = None):
        """
        Remove pending events for a specific truck.
        
        Args:
            truck_id: The truck ID
            event_type: If specified, only remove events of this type. 
                       If None, remove all events for the truck.
        """
        if event_type is None:
            # Remove all events for this truck
            self.event_queue = [
                event for event in self.event_queue 
                if event.truck_id != truck_id
            ]
        else:
            # Remove only events of specific type for this truck
            self.event_queue = [
                event for event in self.event_queue 
                if not (event.truck_id == truck_id and event.event_type == event_type)
            ]
        heapq.heapify(self.event_queue)

    def _check_terminated(self) -> bool:
        """Check if episode is terminated (all trucks done)."""
        if self.active_truck_id is None:
            return True

        all_done = all(
            state in ["complete", "failed"] for state in self.truck_states.values()
        )
        return all_done

    def _check_truncated(self) -> bool:
        """Check if episode is truncated (time limit or step limit exceeded)."""        

        return self.global_clock >= self.max_time or self.episode_steps >= self.max_episode_steps

    def _get_observation(self) -> np.ndarray:
        """Get observation/state for the active truck."""
        if self.observation_mode == "canonical_flat":
            return canonical_flat_observation(
                self.get_canonical_features(),
                self.canonical_shape,
            )
        if self.state_space_manager is None:
            raise RuntimeError("legacy state-space manager is unavailable")
        return self.state_space_manager.get_state(
            trucks=self.trucks,
            active_truck_id=self.active_truck_id,
            transport_graph=self.transport_graph,
            charging_nodes=self.charging_nodes,
            truck_states=self.truck_states,
            event_queue=self.event_queue,
            global_clock=self.global_clock,
            charging_station=self.charging_station,
        )

    def _get_info(self) -> dict:
        """Get info dictionary."""
        all_complete = all(truck.is_complete for truck in self.trucks)
        any_failed = any(truck.failed for truck in self.trucks)
        failure_causes: dict[str, int] = {}
        for truck in self.trucks:
            if truck.failure_reason is not None:
                failure_causes[truck.failure_reason] = (
                    failure_causes.get(truck.failure_reason, 0) + 1
                )
        task_snapshot = (
            self.task_registry.snapshot()
            if self.task_registry is not None
            else None
        )
        all_customers_served = (
            self.task_registry.all_served()
            if self.task_registry is not None
            else all_complete
        )

        # Get charger utilization statistics from charging station manager
        charger_utilization = self.charging_station.get_utilization_stats(
            self.global_clock
        )
        operational_metrics = extract_operational_metrics(self).as_dict()

        return {
            "scenario": self.scenario_descriptor,
            "global_clock": self.global_clock,
            "active_truck_id": self.active_truck_id,
            "episode_reward": self.episode_reward,
            "problem_mode": self.problem_mode,
            "feature_schema_version": (
                SCHEMA_VERSION if self.joint_routing else None
            ),
            "observation_mode": self.observation_mode,
            "all_complete": all_complete,
            "all_customers_served": all_customers_served,
            "successful": bool(all_complete and all_customers_served and not any_failed),
            "any_failed": any_failed,
            "termination_reason": self.termination_reason,
            "failure_causes": failure_causes,
            "invalid_action_count": self.invalid_action_count,
            "operational_metrics": operational_metrics,
            "num_active_trucks": sum(
                1
                for state in self.truck_states.values()
                if state not in ["complete", "failed"]
            ),
            "events_pending": len(self.event_queue),
            "trucks": [truck.get_state_dict() for truck in self.trucks],
            "customer_tasks": task_snapshot,
            "task_counts": (
                self.task_registry.counts()
                if self.task_registry is not None
                else None
            ),
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

    def mask_fn(self) -> np.ndarray:
        """
        Generate feasibility mask for actions using GNN state space logic.
        
        Returns:
            np.ndarray: Boolean array where True indicates feasible actions.
                       Shape: (action_space.n,)
                       Order: [charger_0, ..., charger_N-1, next_delivery, charge_1h, ..., charge_4h]
        """
        return get_action_mask(self)

    def get_canonical_features(self):
        """Return the versioned semantic feature snapshot for joint policies."""
        return extract_canonical_features(self)

    def get_canonical_sets(self):
        """Return fixed-shape typed sets and explicit padding masks."""
        return pad_canonical_features(
            self.get_canonical_features(),
            self.canonical_shape,
        )

    def get_canonical_graph(self):
        """Return a complete heterogeneous view of the canonical features."""
        return canonical_graph_observation(self)

    def set_charger_available(self, charger_node: int, available: bool) -> None:
        """Apply a station closure/reopening and wake trucks released from queue."""
        released_trucks = self.charging_station.set_station_available(
            int(charger_node),
            available,
        )
        for truck_id in released_trucks:
            heapq.heappush(
                self.event_queue,
                Event(
                    time=self.global_clock,
                    truck_id=truck_id,
                    event_type=EventType.TRUCK_READY,
                    data={
                        "reason": "charger_closed",
                        "charger_node": int(charger_node),
                    },
                ),
            )

    def _action_to_string(self, action: int) -> str:
        """Convert action to human-readable string (supports flexible order)."""
        # Charging navigation actions
        if action < self.num_charging_nodes:
            node = self.charging_nodes[action]
            return f"Go to charger @ node {node}"

        # Delivery navigation actions
        if action < self.num_navigation_actions:
            if self.enable_flexible_delivery_order:
                delivery_idx = action - self.num_charging_nodes
                truck = None
                if self.active_truck_id is not None and self.active_truck_id < len(self.trucks):
                    truck = self.trucks[self.active_truck_id]

                if truck is not None:
                    if delivery_idx + 1 < len(truck.delivery_sequence):
                        node = truck.delivery_sequence[delivery_idx + 1]
                        remaining = set(truck.get_remaining_deliveries())
                        status = "pending" if node in remaining else "done"
                        return f"Go to delivery slot {delivery_idx} @ node {node} ({status})"
                    if delivery_idx == len(truck.delivery_sequence) - 1:
                        node = truck.delivery_sequence[0]
                        remaining = set(truck.get_remaining_deliveries())
                        status = "pending" if node in remaining else "done"
                        return f"Return to depot @ node {node} ({status})"
                return f"Go to delivery slot {delivery_idx} (empty)"
            else:
                return "Go to next delivery"

        # Charging actions (use configured durations to avoid negative hours)
        charge_idx = action - self.num_navigation_actions
        if 0 <= charge_idx < len(self.charge_action_values):
            value = self.charge_action_values[charge_idx]
            if self.charging_action_mode == "target_soc":
                return f"Charge to {value:.0%} SoC"
            return f"Charge for {value:g}h"
        return f"Invalid charge action {charge_idx}"

    def get_delivery_sequence_index(self, node_id: int) -> int:
        """
        Get the relative delivery sequence index for a given delivery node.
        
        Returns the minimum index across all trucks that have this node in their
        remaining delivery sequence. Index 1 means it's the next delivery, 
        index 2 means there's one delivery before it, etc.
        
        Args:
            node_id: Delivery node ID
            
        Returns:
            Relative sequence index (1-based), or 0 if node is not in any truck's
            remaining delivery sequence or if all trucks are complete/failed
        """
        min_index = float('inf')
        
        for truck in self.trucks:
            # Skip failed and completed trucks
            if truck.failed or truck.is_complete:
                continue
                
            # Get remaining deliveries for this truck
            remaining_deliveries = truck.get_remaining_deliveries()
            
            # Check if node_id is in remaining deliveries
            if node_id in remaining_deliveries:
                # Find its position (1-based index)
                position = remaining_deliveries.index(node_id) + 1
                min_index = min(min_index, position)
        
        # Return 0 if node not found in any truck's sequence
        return int(min_index) if min_index != float('inf') else 0

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
            
            # Save charging logs if logger is enabled
            if self.charging_logger:
                self.charging_logger.save_session_logs(episode_id=self.run_id)
                self.charging_logger.save_summary_statistics(episode_id=self.run_id)


def _positive_integer(value, label: str) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) <= 0
    ):
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_finite(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _positive_finite(value, label: str) -> float:
    result = _nonnegative_finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _validate_initial_battery_setting(setting) -> None:
    if isinstance(setting, str):
        if setting in {"full", "random"}:
            return
        raise ValueError(
            "truck.initial_battery must be 'full', 'random', or a percentage"
        )
    if isinstance(setting, bool) or not isinstance(setting, (int, float)):
        raise TypeError(
            "truck.initial_battery must be 'full', 'random', or a percentage"
        )
    percentage = float(setting)
    if not math.isfinite(percentage) or not 0.0 <= percentage <= 100.0:
        raise ValueError("numeric truck.initial_battery must be in [0, 100]")
