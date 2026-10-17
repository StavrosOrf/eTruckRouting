"""
Event-Driven Truck Routing Environment - Single-agent controlling active truck.

Uses a global clock and event queue. Each truck generates two types of events:
- TRUCK_READY: Truck is ready to take an action (initial, after arrival, after charging, after waiting)
- TRUCK_ROUTING: Truck arrives at a destination node (delivery or charger)

The environment steps forward when events finish. Only one truck is active at a time.
"""

import copy
import datetime
import hashlib
import heapq
import json
import math
import os
import pickle
import sys
from numbers import Integral, Real

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
from EVRoutingEnv.models.environment.route_certificate import replay_routes
from EVRoutingEnv.models.simulation.charging_curve import ChargingCurveModel
from EVRoutingEnv.models.simulation.charging_station import ChargingStation
from EVRoutingEnv.models.simulation.delivery_simulator import DeliverySimulator
from EVRoutingEnv.models.simulation.scenario import ScenarioRandomStreams
from EVRoutingEnv.models.simulation.traffic_simulation import TrafficSimulator
from EVRoutingEnv.state.action_mask import (
    get_action_mask,
    get_structural_action_mask,
)
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
from EVRoutingEnv.utils.utils import get_graph, load_config, resolve_network_data_dir


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
        if {"min_hop_distance", "max_hop_distance"} & env_config.keys():
            raise ValueError(
                "min_hop_distance/max_hop_distance were removed (they were inactive "
                "and the network has no measured km). Configure "
                "min_hop_energy_kwh/max_hop_energy_kwh instead."
            )
        self.min_hop_energy_kwh = _nonnegative_finite(
            env_config.get("min_hop_energy_kwh", 0.0),
            "environment.min_hop_energy_kwh",
        )
        upper_energy = env_config.get("max_hop_energy_kwh")
        self.max_hop_energy_kwh = (math.inf if upper_energy is None else
            _nonnegative_finite(upper_energy, "environment.max_hop_energy_kwh"))
        if self.max_hop_energy_kwh < self.min_hop_energy_kwh:
            raise ValueError(
                "environment.max_hop_energy_kwh must be at least min_hop_energy_kwh"
            )
        self.generation_attempts = _positive_integer(
            env_config.get("generation_attempts", 64), "environment.generation_attempts",
        )
        self.max_time = _positive_finite(
            env_config["max_time"],
            "environment.max_time",
        )
        max_stops_for_bounds = self.fixed_num_stops
        default_step_limit = max(200, 16 * max_stops_for_bounds + 8 * self.num_trucks)
        self.max_episode_steps = _positive_integer(
            env_config.get("max_episode_steps", default_step_limit),
            "environment.max_episode_steps",
        )
        self.verbose = verbose if verbose is not None else env_config.get("verbose", False)
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
                self.output_dir,
                self.verbose,
                use_osm=False,
                vis_data_dir=os.path.join(
                    resolve_network_data_dir(self.config["network"]), "vis_data"
                ),
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
        self.problem_mode = self.problem_config.get("mode", "joint_fleet")
        if self.problem_mode not in {"joint_fleet", "sequential"}:
            raise ValueError("problem.mode must be 'joint_fleet' or 'sequential'")
        self.sequential_routing = self.problem_mode == "sequential"
        # Both modes use the fleet task registry and the same action envelope.
        self.joint_routing = True
        self.enable_flexible_delivery_order = True
        if "enable_flexible_delivery_order" in self.delivery_config:
            raise ValueError(
                "delivery.enable_flexible_delivery_order was removed; use problem.mode"
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
        # Generalization needs a way to move the road network itself. Travel
        # times and energies are read from precomputed tables rather than
        # derived from truck.base_speed -- which is why changing that parameter
        # leaves every leg unchanged -- so the network is perturbed at its own
        # level, before any distance cache is built, and every consumer
        # (simulator, mask, heuristic, planners, canonical features) sees the
        # same scaled network.
        network_config = self.config.get("network", {})
        time_scale = float(network_config.get("travel_time_scale", 1.0))
        energy_scale = float(network_config.get("energy_scale", 1.0))
        for scale, label in ((time_scale, "travel_time_scale"), (energy_scale, "energy_scale")):
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"network.{label} must be positive")
        self.transport_graph = TransportationGraph(graph)
        self.transport_graph.scale_network(time_scale, energy_scale)

        # Get charging nodes
        self.charging_nodes = self.transport_graph.get_charging_nodes()
        self.num_charging_nodes = len(self.charging_nodes)
        if self.joint_routing and "max_episode_steps" not in env_config:
            # A simple charger path can visit each station once, with a route
            # and charge decision per hop. Cover customer approaches and each
            # truck's final return; this is a guard, not a wall-time budget.
            self.max_episode_steps = max(
                200, 2 * (self.num_charging_nodes + 2) * (self.fixed_num_stops + self.num_trucks)
            )

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
        # The charger queue model ships with the case study (EVRoutingEnv/data/case_studies/<case>/).
        waiting_time_path = os.path.join(
            resolve_network_data_dir(self.config["network"]), "waiting_time_lookup.json"
        )
        if not os.path.exists(waiting_time_path):
            raise FileNotFoundError(
                f"{waiting_time_path} not found: every case study folder needs a waiting_time_lookup.json"
            )
        station_charging_config = dict(self.charging_config)
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
        # Both modes: chargers, global customer slots, depot, charging values.
        # Sequential ownership and order are enforced by the task registry.
        default_charging_mode = "target_soc"
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
        self.num_navigation_actions = self.num_charging_nodes + stops_for_action_space + 1
        
        self.num_charge_actions = len(self.charge_action_values)

        # Discrete action space (single agent)
        self.action_space = spaces.Discrete(
            self.num_navigation_actions + self.num_charge_actions
        )

        default_observation_mode = "canonical_flat"
        self.observation_mode = env_config.get(
            "observation_mode",
            default_observation_mode,
        )
        if self.observation_mode not in {"canonical_flat", "legacy_flat"}:
            raise ValueError(
                "environment.observation_mode must be 'canonical_flat' or "
                "'legacy_flat'"
            )

        # Which mask a learning policy is handed.  "hard" is the proposed method:
        # the centralized feasibility engine removes every infeasible candidate
        # before the policy scores it.  "structural" keeps the identical
        # observation and candidate set but only hides slots that denote no
        # action at all, so the policy must learn feasibility itself.  The
        # simulator's dynamics are the same either way; only the mask differs,
        # which is what makes this a clean ablation of the mask.
        # eTFRP variant: customers belong to a specific truck from the start, so
        # the policy decides sequencing, charging, and routing but not
        # assignment. This is the operational case the main benchmark models,
        # expressed inside the canonical stack so that the same architecture,
        # mask, and artifact contract apply to both settings.
        assignment = self.problem_config.get(
            "assignment", "preassigned" if self.sequential_routing else "fleet"
        )
        if assignment not in {"fleet", "preassigned"}:
            raise ValueError("problem.assignment must be 'fleet' or 'preassigned'")
        if self.sequential_routing and assignment != "preassigned":
            raise ValueError("sequential mode requires problem.assignment=preassigned")
        self.preassigned_customers = assignment == "preassigned"

        self.policy_action_mask = env_config.get("policy_action_mask", "hard")
        if self.policy_action_mask not in {"hard", "structural"}:
            raise ValueError(
                "environment.policy_action_mask must be 'hard' or 'structural'"
            )
        # What happens when an infeasible action is actually executed.
        # "terminate" is the simulator's own semantics: committing a truck to a
        # leg it cannot complete strands it.  "penalize" is the charitable
        # variant used to keep the unmasked ablation from being decided by that
        # one modelling choice: the action is refused, the state is unchanged,
        # and the episode continues until the invalid-action budget is spent.
        self.invalid_action_mode = env_config.get("invalid_action_mode", "terminate")
        if self.invalid_action_mode not in {"terminate", "penalize"}:
            raise ValueError(
                "environment.invalid_action_mode must be 'terminate' or 'penalize'"
            )
        self.invalid_action_budget = _positive_integer(
            env_config.get("invalid_action_budget", 64),
            "environment.invalid_action_budget",
        )

        # Generalization evaluation may run a smaller instance through a policy
        # trained on a larger envelope.  The observation keeps the trained width
        # and the surplus rows are padding, so the checkpoint still loads; the
        # instance itself is genuinely smaller.
        canonical_trucks = _positive_integer(
            env_config.get("canonical_max_trucks", self.num_trucks),
            "environment.canonical_max_trucks",
        )
        if canonical_trucks < self.num_trucks:
            raise ValueError(
                "environment.canonical_max_trucks cannot be smaller than num_trucks"
            )
        self.instance_num_stops = env_config.get("instance_num_stops")
        if self.instance_num_stops is not None:
            self.instance_num_stops = _positive_integer(
                self.instance_num_stops, "environment.instance_num_stops"
            )
            if self.instance_num_stops > self.fixed_num_stops:
                raise ValueError(
                    "environment.instance_num_stops cannot exceed num_stops, which "
                    "fixes the action envelope"
                )

        self.canonical_shape = CanonicalShapeSpec(
            max_trucks=canonical_trucks,
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
            # Keep the historical prefix, but include the lossless canonical
            # state too: legacy rows omit task ownership and pending activity.
            float_limit = np.finfo(np.float32).max
            self.observation_space = spaces.Box(
                low=-float_limit, high=float_limit,
                shape=(self.state_space_manager.observation_space.shape[0]
                       + self.canonical_shape.flat_size,), dtype=np.float32,
            )

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
        self.completed_delivery_reward = 0.0
        self._episode_ended = False
        self._episode_end_flags = (False, False)
        self.episode_steps = 0  # Track number of steps in current episode
        self.waiting_start_times = {}  # Track when trucks enter waiting_to_charge state
        self.waiting_penalty_buffer = 0.0  # Buffer for waiting penalty to apply on next step
        self.scenario_seed: int | None = None
        self.scenario_random_streams: ScenarioRandomStreams | None = None
        self.instance_rng: np.random.Generator | None = None
        self.joint_instance: JointRoutingInstance | None = None
        self.task_registry = None
        self.last_action_feasibility = None
        self.termination_reason: str | None = None
        self.invalid_action_count = 0
        self._penalized_failed_trucks: set[int] = set()
        self._penalized_completed_trucks: set[int] = set()
        self._truncation_reason: str | None = None
        self.scenario_descriptor: dict | None = None
        self.feasibility_witness: tuple[tuple[int, int], ...] = ()

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
        if self.instance_num_stops is not None:
            # Size-transfer evaluation: the action envelope stays at
            # fixed_num_stops so a trained policy still fits, while the instance
            # carries fewer customers and the surplus slots report as empty.
            self.num_stops = self.instance_num_stops
        elif self.allow_variable_num_stops:
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
        self.completed_delivery_reward = 0.0
        self._episode_ended = False
        self._episode_end_flags = (False, False)
        self.episode_steps = 0  # Reset step counter
        self.waiting_start_times = {}  # Reset waiting time tracking
        self.waiting_penalty_buffer = 0.0  # Reset waiting penalty buffer
        self.truck_ready_times = {}  # Reset truck ready time tracking
        self.last_action_feasibility = None
        self.termination_reason = None
        self.invalid_action_count = 0
        self._penalized_failed_trucks: set[int] = set()
        self._penalized_completed_trucks: set[int] = set()
        self._truncation_reason: str | None = None
        self.scenario_descriptor = None

        # Reset charging station state
        self.charging_station.reset()

        # Track actual routes for visualization
        self.truck_routes = {}  # truck_id -> list of (node, time, event_type)
        self.truck_initial_plans = (
            {}
        )  # truck_id -> {'start': node, 'deliveries': [nodes]}

        # Both modes create one fleet-owned instance and use the same event engine.
        self.trucks = []
        self.truck_states = {}
        self._create_joint_fleet()

        self._schedule_initial_decisions()

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

    def _schedule_initial_decisions(self) -> None:
        for i in range(self.num_trucks):
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
        initial_batteries = [self._initial_battery(battery_capacity) for _ in range(self.num_trucks)]
        self.feasibility_witness = ()
        def generate():
            return generate_joint_routing_instance(
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
                preassign=self.preassigned_customers,
                sequential=self.sequential_routing,
                energy_multiplier=(max(1.0, float(self.traffic_config["max_energy_multiplier"]))
                    if self.traffic_config["enable_traffic"] and self.traffic_config["enable_energy_uncertainty"]
                    else 1.0),
                initial_batteries=initial_batteries,
                charge_transition=self._certificate_charge_transition,
                min_hop_energy_kwh=self.min_hop_energy_kwh,
                max_hop_energy_kwh=self.max_hop_energy_kwh,
                generation_attempts=self.generation_attempts,
                certify=lambda instance: self._certify_instance(instance, initial_batteries),
            )

        cache_path = self._scenario_cache_path(initial_batteries)
        cached = None
        if cache_path is not None and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as file:
                    cached = pickle.load(file)
            except (OSError, EOFError, pickle.PickleError):
                cached = None
        if cached is not None:
            # the certificate replay's action trace is part of what certification produces
            self.joint_instance, self.feasibility_witness = cached["instance"], cached["trace"]
        else:
            self.joint_instance = generate()
            if cache_path is not None:
                self._write_scenario_cache(
                    cache_path, {"instance": self.joint_instance, "trace": self.feasibility_witness}
                )
        self._install_joint_instance(self.joint_instance, initial_batteries)

    SCENARIO_CACHE_VERSION = 1  # bump when generation or certification semantics change

    def _scenario_cache_path(self, initial_batteries) -> str | None:
        """Disk path of this scenario's certified instance, or None when caching is off.

        Instance generation is a pure function of the scenario seed (its own named RNG stream),
        the configuration and the network, so a certified instance can be stored once and reused
        across resets, runs and workers. Enable it with ``environment.scenario_cache_dir`` or the
        ``EVRP_SCENARIO_CACHE`` environment variable; leave both unset to generate on every reset.
        """
        directory = self.config["environment"].get("scenario_cache_dir") or os.environ.get("EVRP_SCENARIO_CACHE")
        if not directory:
            return None
        graph = self.transport_graph
        matrices = graph.cost_matrices()
        digest = getattr(self, "_scenario_graph_digest", None)
        if digest is None or digest[0] is not matrices:
            node_ids = [int(graph.graph.nodes[n].get("original_id", n)) for n in sorted(graph.graph.nodes())]
            digest = (matrices, hashlib.sha1(
                matrices[1].tobytes() + matrices[2].tobytes() + json.dumps(node_ids).encode()
                + json.dumps(sorted(int(c) for c in self.charging_nodes)).encode()
            ).hexdigest())
            self._scenario_graph_digest = digest
        key = hashlib.sha1(json.dumps({
            "version": self.SCENARIO_CACHE_VERSION, "graph": digest[1], "seed": self.scenario_seed,
            "stops": self.num_stops, "initial_batteries": [float(b) for b in initial_batteries],
            "config": self.config, "mode": [self.sequential_routing, self.preassigned_customers],
        }, sort_keys=True, default=str).encode()).hexdigest()
        return os.path.join(directory, key[:2], key + ".pkl")

    @staticmethod
    def _write_scenario_cache(path: str, payload) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temporary = f"{path}.{os.getpid()}.tmp"
            with open(temporary, "wb") as file:
                pickle.dump(payload, file)
            os.replace(temporary, path)
        except (OSError, pickle.PickleError):
            pass  # caching is an optimisation only

    def _install_joint_instance(self, instance, initial_batteries) -> None:
        """Install static inputs without sampling or changing their seed."""
        self.joint_instance = instance
        truck_config = self.config["truck"]
        battery_capacity = float(truck_config["battery_capacity"])
        payload_capacity = float(self.problem_config.get(
            "payload_capacity", truck_config.get("payload_capacity", 0.0),
        ))
        self.task_registry = self.joint_instance.create_registry(
            sequential=self.sequential_routing
        )
        shared_sequence = [self.joint_instance.depot_node] + [
            task.node_id for task in self.joint_instance.tasks
        ]

        for truck_id in range(self.num_trucks):
            sequence = shared_sequence
            if self.sequential_routing:
                sequence = [self.joint_instance.depot_node] + [
                    task.node_id
                    for task in sorted(
                        (task for task in self.joint_instance.tasks if task.preassigned_to == truck_id),
                        key=lambda task: task.sequence_position,
                    )
                ]
            truck = Truck(
                truck_id=truck_id,
                truck_type="electric",
                delivery_sequence=sequence,
                initial_battery=initial_batteries[truck_id],
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
                "deliveries": sequence.copy(),
            }

    def _certificate_charge_transition(self, node, battery):
        """Use exactly the station curve and highest configured charge action."""
        capacity = float(self.config["truck"]["battery_capacity"])
        soc = min(1.0, max(0.0, battery / capacity))
        value = max(self.charge_action_values)
        if soc >= (value if self.charging_action_mode == "target_soc" else 1.0) - 1e-9:
            return battery, 0.0, value
        kind = self.charging_station.charger_type[node]
        settings = dict(self.charging_config["dcfast" if kind == "DCFast" else "level2"])
        settings.update(
            use_realistic_curve=self.charging_config["use_realistic_curve"],
            charge_rate=self.charging_station.charger_power_kw[node],
            efficiency=self.charging_station.charger_efficiency[node],
        )
        if "taper_power_min" in settings:
            settings["taper_power_min"] = min(
                float(settings["taper_power_min"]), float(settings["charge_rate"]),
            )
        if (
            self.charging_action_mode == "target_soc"
            and kind == "DCFast"
            and settings["use_realistic_curve"]
            and value > soc + 1e-12
        ):
            # Certification only orders candidate routes by charging time, so it uses the fast
            # fixed-order integral (agrees with the adaptive one to ~3e-9 relative). The energy
            # delivered is the same closed form the adaptive routine returns.
            hours = self.charging_curve_model.charge_hours_to_target_fast(
                soc, value, capacity, float(settings["charge_rate"]), float(settings["efficiency"]),
                float(settings["taper_start_soc"]), float(settings["taper_power_min"]),
            )
            return min(capacity, battery + (value - soc) * capacity), hours, value
        if self.charging_action_mode == "target_soc":
            amount, details = self.charging_curve_model.calculate_charge_to_target(
                soc, value, capacity, settings, kind,
            )
        else:
            amount, details = self.charging_curve_model.calculate_charge(
                soc, value, capacity, settings, kind,
            )
        return min(capacity, battery + amount), details["actual_charge_hours"], value

    def _certify_instance(self, instance, initial_batteries) -> None:
        """Replay on private state; never consume live simulator randomness."""
        shadow = copy.copy(self)
        memo = {id(self.transport_graph): self.transport_graph}
        for name in (
            "charging_station", "traffic_simulator", "delivery_simulator",
            "event_queue", "truck_states", "truck_routes", "truck_initial_plans",
            "trucks", "waiting_start_times", "truck_ready_times",
            "_penalized_failed_trucks", "_penalized_completed_trucks",
        ):
            setattr(shadow, name, copy.deepcopy(getattr(self, name), memo))
        shadow.verbose = shadow.enable_plotting = False
        shadow.event_handler = EventHandler(False)
        shadow.charging_curve_model = ChargingCurveModel(verbose=False)
        shadow.charging_logger = None
        shadow.charging_station.verbose = False
        shadow.traffic_simulator.verbose = shadow.delivery_simulator.verbose = False
        shadow._get_observation = lambda: np.empty(0, dtype=np.float32)
        shadow._get_info = lambda: {}
        shadow._install_joint_instance(instance, initial_batteries)
        shadow._schedule_initial_decisions()
        shadow._advance_to_next_decision()
        self.feasibility_witness = replay_routes(shadow, instance.witness_routes)

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

        for fleet_truck in self.trucks:
            if not self.sequential_routing or fleet_truck.truck_id == truck.truck_id:
                fleet_truck.delivered_nodes.add(node_id)

        depot_node = self.joint_instance.depot_node
        for fleet_truck in self.trucks:
            if (
                fleet_truck.failed
                or fleet_truck.is_complete
                or not self.task_registry.route_complete_for(fleet_truck.truck_id)
            ):
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

    def _advance_to_next_decision(self):
        """
        Advance simulation clock to next event that requires a decision.
        Process events until we find a TRUCK_READY event that can actually proceed.
        """
        while self.event_queue:
            pending_truck = self.trucks[self.event_queue[0].truck_id]
            if pending_truck.is_complete or pending_truck.failed:
                heapq.heappop(self.event_queue)
                continue
            # A journey/service may extend past the episode horizon. Keep its
            # event pending: nothing after the horizon has physically completed.
            if self.event_queue[0].time > self.max_time:
                self.global_clock = self.max_time
                self.active_truck_id = None
                self._truncation_reason = "time_limit_exhausted"
                self.termination_reason = self._truncation_reason
                return
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

                if (
                    self.sequential_routing
                    and self.task_registry.route_complete_for(truck.truck_id)
                    and int(truck.current_node) == self.joint_instance.depot_node
                    and self.truck_states[truck.truck_id] == "ready"
                ):
                    truck.battery_at_completion = truck.current_battery
                    truck.mark_complete(timestamp=self.global_clock)
                    self.truck_states[truck.truck_id] = "complete"
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
                    "charger_port_available",
                    "charger_closed",
                ]:
                    current_state = self.truck_states.get(truck.truck_id, "")
                    wrong_station = "charger_node" in event.data and int(
                        event.data["charger_node"]
                    ) != int(truck.current_node)
                    if current_state not in ["waiting_to_charge", "ready"] or wrong_station:
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
                        self.waiting_penalty_buffer += waiting_penalty
                        
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
                    self.completed_delivery_reward += self.reward_config["delivery_bonus"]
                    unloading_duration = event.data.get("unloading_duration", 0.0)
                    if truck.unloading_start_time is not None:
                        truck.finish_unloading(unloading_duration=unloading_duration, timestamp=self.global_clock)
                    self._complete_joint_customer_service(truck, event.data)

                # Even without an available customer, a ready truck can
                # choose to return to the depot and finish its own route.
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
                if self.global_clock >= self.max_time:
                    self.active_truck_id = None
                    continue
                decisions = joint_action_feasibility(self)
                self.last_action_feasibility = decisions
                if not any(item.feasible for item in decisions):
                    self._fail_truck(truck, "no_feasible_action")
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

        if self.global_clock >= self.max_time and not all(
            truck.is_complete or truck.failed for truck in self.trucks
        ):
            self.active_truck_id = None
            self._truncation_reason = "time_limit_exhausted"
            self.termination_reason = self._truncation_reason
            return

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
                    if not live_trucks:
                        self.termination_reason = (
                            "all_trucks_failed"
                            if all(truck.failed for truck in self.trucks)
                            else "fleet_stopped_with_unserved_customers"
                        )
                    elif unassigned and not any(
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
                    - Integer: chargers, global customer slots, depot, then
                      configured charging values. Empty customer slots are masked.
                    - Tuple: (node_id, charge_value, is_charging)
                      * node_id: Target node to navigate to or charge at
                      * charge_value: Target SoC or hours, according to charging.action_mode
                      * is_charging: Whether this is a charging action

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self._episode_ended:
            return (self._get_observation(), 0.0, *self._episode_end_flags, self._get_info())
        if self.active_truck_id is None:
            truncated = self._check_truncated()
            terminated = self._check_terminated()
            reward = (
                self._consume_failure_penalties()
                + self._consume_completion_penalties()
            )
            self.episode_reward += reward
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        truck = self.trucks[self.active_truck_id]
        reward = 0.0
        invalid_actions_before = self.invalid_action_count
        operating_before = extract_operational_metrics(self).total_operating_time
        
        # Reject malformed/out-of-envelope integers before indexing or logging.
        if not isinstance(action, tuple) and (
            isinstance(action, (bool, np.bool_))
            or not isinstance(action, Integral)
            or not 0 <= action < self.action_space.n
        ):
            raise ValueError(f"discrete action must be an integer in [0, {self.action_space.n})")

        # Tuple actions must preserve their declared meaning. In particular,
        # int(12.5) must never silently navigate to node 12, and a string such
        # as "False" must never be interpreted as a charging flag.
        if isinstance(action, tuple):
            if len(action) != 3:
                raise ValueError("tuple action must contain (node_id, charge_value, is_charging)")
            node_id, charge_value, is_charging = action
            if isinstance(node_id, (bool, np.bool_)) or not isinstance(node_id, Integral):
                raise ValueError("tuple action node_id must be an integer")
            if isinstance(charge_value, (bool, np.bool_)) or not isinstance(charge_value, Real):
                raise ValueError("tuple action charge_value must be a real number")
            if not isinstance(is_charging, (bool, np.bool_)):
                raise ValueError("tuple action is_charging must be boolean")

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
            # Elapsed waiting cost is collected with all other operating time
            # below; the buffer remains useful for queue diagnostics only.
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
                    delivery_idx = action - self.num_charging_nodes
                    if delivery_idx < self.fixed_num_stops:
                        target_node = (
                            self.joint_instance.tasks[delivery_idx].node_id
                            if delivery_idx < len(self.joint_instance.tasks)
                            else None
                        )
                    elif delivery_idx == self.fixed_num_stops:
                        target_node = self.joint_instance.depot_node
                    else:
                        raise ValueError(f"Invalid delivery action index: {delivery_idx}")
                if target_node is None:
                    reward += self._reject_joint_action(
                        truck, FeasibilityReason.EMPTY_ACTION_SLOT
                    )
                else:
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

        self.episode_steps += 1
        refused = (
            self.invalid_action_mode == "penalize"
            and self.invalid_action_count > invalid_actions_before
            and not truck.failed
        )
        # A refused action keeps this exact decision point, including the truck
        # and clock. No event is needed until an action is actually accepted.
        if (
            self.invalid_action_mode == "penalize"
            and self.invalid_action_count >= self.invalid_action_budget
        ):
            self.active_truck_id = None
            self.termination_reason = "invalid_action_budget_exhausted"
        elif not refused:
            self._advance_to_next_decision()
        reward += self._consume_failure_penalties()
        reward += self._consume_completion_penalties()
        reward += self.completed_delivery_reward
        self.completed_delivery_reward = 0.0
        self.waiting_penalty_buffer = 0.0
        elapsed_work = extract_operational_metrics(self).total_operating_time - operating_before
        reward -= max(0.0, elapsed_work) * self.reward_config["time_multiplier"]
        self.episode_reward += reward
        
        if self.verbose:
            # print status of all trucks
            print("\nTruck statuses after step:")
            for t in self.trucks:
                state = self.truck_states[t.truck_id]
                print(f"  Truck {t.truck_id}: State={state}, Battery={t.current_battery:.1f} kWh, Completed={t.is_complete}, Failed={t.failed}")

        # Check termination conditions
        truncated = self._check_truncated()
        terminated = self._check_terminated()
        self._episode_ended = terminated or truncated
        self._episode_end_flags = (terminated, truncated)
        if self._episode_ended:
            self.active_truck_id = None

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

        current_node = int(truck.current_node)

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
        if discharge > truck.current_battery + 1e-9:
            if self.verbose:
                print(f"  ERROR: Insufficient battery ({truck.current_battery:.1f} kWh < {discharge:.1f} kWh needed)")
            self._fail_truck(truck, "insufficient_energy_after_realization")
            return 0.0
        
        # Determine if this is navigation to a charger or delivery.
        is_charger_nav = target_node in self.charging_nodes
        try:
            joint_task = self.task_registry.task_for_node(target_node)
        except KeyError:
            joint_task = None
        is_delivery_nav = joint_task is not None

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

        if is_charger_nav and self.verbose:
            charger_info = self.charging_station.get_charger_info(target_node, self.global_clock)
            print(f"  Going to charger @ node {target_node}")
            print(f"    Current occupancy: {charger_info['current_occupancy']}/{charger_info['capacity']}")

        # If leaving a charger to navigate elsewhere, remove from its waitlist and wake others
        if current_node in self.charging_nodes and target_node != current_node:
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
            service_start = (
                max(completion_time, joint_task.earliest_service)
                if joint_task is not None
                else completion_time
            )
            service_kwargs = (
                {"base_service_time": joint_task.base_service_time}
                if joint_task is not None
                else {}
            )
            actual_unloading_time = self.delivery_simulator.apply_unloading_time(
                delivery_node=target_node,
                current_time=service_start,
                **service_kwargs,
            )

        if joint_task is not None:
            self.task_registry.claim(
                target_node,
                truck_id=truck.truck_id,
                timestamp=departure_time,
            )

        if target_node == depot_node:
            # Completion is committed by the arrival event, after travel time
            # and battery consumption, independently of outstanding customers.
            truck.return_to_depot_pending = True

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

        # Time costs accrue from elapsed work at the step boundary. Delivery
        # bonuses and terminal battery costs accrue only when their events finish.
        return 0.0

    def _reject_joint_action(
        self,
        truck: Truck,
        reason: FeasibilityReason,
    ) -> float:
        """Fail an explicit invalid action without silently changing its meaning."""
        cause = f"invalid_action:{reason.value}"
        self.invalid_action_count += 1

        if self.invalid_action_mode == "penalize":
            # Refuse the action, leave the state untouched, and let the episode
            # continue.  Nothing advances here -- not the clock, not the truck --
            # so a policy that keeps proposing infeasible actions makes no
            # progress and eventually exhausts the budget below.  That bound is
            # what stops an unmasked policy from spinning forever on a single
            # decision point.
            penalty = -abs(
                float(self.reward_config.get("invalid_action_penalty", 100.0))
            )
            if self.invalid_action_count >= self.invalid_action_budget:
                exhausted = "invalid_action_budget_exhausted"
                self._fail_truck(truck, exhausted)
                if self.termination_reason is None:
                    self.termination_reason = exhausted
                return penalty
            return penalty

        self._fail_truck(truck, cause)
        if self.termination_reason is None:
            self.termination_reason = cause
        return 0.0

    def _fail_truck(self, truck: Truck, reason: str) -> None:
        """Fail one truck without leaving queue entries or port reservations."""
        truck.mark_failed(reason=reason, timestamp=self.global_clock)
        self.truck_states[truck.truck_id] = "failed"
        for node in self.charging_nodes:
            self.charging_station.remove_from_waitlist(truck.truck_id, node)
            self.charging_station.wake_waiting_trucks(
                node, self.global_clock, self.event_queue, EventType, Event,
                self.truck_states,
            )



    def _execute_charge_action(self, truck: Truck, charge_value: float, charger_node: int) -> float:
        """Execute a target-SoC or duration charging action."""
        # Convert to int if needed
        if hasattr(charger_node, "item"):
            charger_node = int(charger_node.item())
        else:
            charger_node = int(charger_node)

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
        # Station-specific conversion efficiency, falling back to the class
        # value when the station declares none.
        charger_config_with_curve["efficiency"] = (
            self.charging_station.charger_efficiency[charger_node]
        )
        # The taper floor is a property of the curve, not of the station, so a
        # station rated below the configured floor would otherwise be rejected
        # outright. Clamping keeps a weak station a weak station instead: it
        # simply never tapers below its own rated power.
        if "taper_power_min" in charger_config_with_curve:
            charger_config_with_curve["taper_power_min"] = min(
                float(charger_config_with_curve["taper_power_min"]),
                float(charger_config_with_curve["charge_rate"]),
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
        return 0.0  # Elapsed charging time is charged at the step boundary.

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

    def _consume_failure_penalties(self) -> float:
        """Price every truck failure once, regardless of which handler found it."""
        newly_failed = {
            truck.truck_id for truck in self.trucks if truck.failed
        } - self._penalized_failed_trucks
        self._penalized_failed_trucks.update(newly_failed)
        return len(newly_failed) * float(self.reward_config["failure_penalty"])

    def _consume_completion_penalties(self) -> float:
        """Price terminal battery state once, after each truck actually completes."""
        newly_completed = [
            truck
            for truck in self.trucks
            if truck.is_complete
            and truck.truck_id not in self._penalized_completed_trucks
        ]
        self._penalized_completed_trucks.update(
            truck.truck_id for truck in newly_completed
        )
        if not self.reward_config["enable_leftover_battery_penalty"]:
            return 0.0
        coefficient = float(self.reward_config["leftover_battery_penalty_coef"])
        return -sum(
            coefficient
            * 100.0
            * (
                truck.battery_at_completion
                if truck.battery_at_completion is not None
                else truck.current_battery
            )
            / truck.battery_capacity
            for truck in newly_completed
            if truck.served_task_ids
            or truck.completed_service_count > 0
            or truck.total_distance_traveled > 0.0
        )

    def _check_terminated(self) -> bool:
        """Physical success/failure ends the task; a time cutoff only truncates it."""
        if self._truncation_reason is not None:
            return False
        all_done = all(
            state in {"complete", "failed"} for state in self.truck_states.values()
        )
        ended = all_done or self.active_truck_id is None
        if ended and self.termination_reason is None:
            if self.task_registry.all_served() and all(
                truck.is_complete for truck in self.trucks
            ):
                self.termination_reason = "success"
            elif all(truck.failed for truck in self.trucks):
                self.termination_reason = "all_trucks_failed"
            else:
                self.termination_reason = "fleet_stopped_with_unserved_customers"
        return ended

    def _check_truncated(self) -> bool:
        """Completing within the allowed time/steps takes precedence over a cap."""
        if self.termination_reason == "invalid_action_budget_exhausted":
            return False
        if (
            self._truncation_reason is None
            and all(truck.is_complete or truck.failed for truck in self.trucks)
            and self.global_clock <= self.max_time
        ):
            return False
        if self.global_clock >= self.max_time:
            self._truncation_reason = "time_limit_exhausted"
        elif self.episode_steps >= self.max_episode_steps:
            self._truncation_reason = "step_limit_exhausted"
        if self._truncation_reason is not None:
            self.termination_reason = self._truncation_reason
            return True
        return False

    def _get_observation(self) -> np.ndarray:
        """Get observation/state for the active truck."""
        if self.observation_mode == "canonical_flat":
            return canonical_flat_observation(
                self.get_canonical_features(),
                self.canonical_shape,
            )
        if self.state_space_manager is None:
            raise RuntimeError("legacy state-space manager is unavailable")
        legacy = self.state_space_manager.get_state(
            trucks=self.trucks,
            active_truck_id=self.active_truck_id,
            transport_graph=self.transport_graph,
            charging_nodes=self.charging_nodes,
            truck_states=self.truck_states,
            event_queue=self.event_queue,
            global_clock=self.global_clock,
            charging_station=self.charging_station,
        )
        return np.concatenate((legacy, canonical_flat_observation(
            self.get_canonical_features(), self.canonical_shape,
        )))

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
        # Get charger utilization statistics from charging station manager
        charger_utilization = self.charging_station.get_utilization_stats(
            self.global_clock
        )
        operational_metrics = extract_operational_metrics(self).as_dict()
        all_customers_served = (
            self.task_registry.all_served()
            if self.task_registry is not None
            else operational_metrics["customers_served"]
            == operational_metrics["customers_total"]
        )

        return {
            "scenario": self.scenario_descriptor,
            "global_clock": self.global_clock,
            "active_truck_id": self.active_truck_id,
            "episode_reward": self.episode_reward,
            "problem_mode": self.problem_mode,
            "feature_schema_version": SCHEMA_VERSION,
            "instance_feasibility": {
                "certified": bool(self.feasibility_witness),
                "method": "seeded_event_replay",
                "witness_steps": len(self.feasibility_witness),
            },
            "sequential_routing": self.sequential_routing,
            "prescribed_routes": (
                {truck.truck_id: [*truck.delivery_sequence, self.joint_instance.depot_node]
                 for truck in self.trucks}
                if self.sequential_routing else None
            ),
            "observation_mode": self.observation_mode,
            "all_complete": all_complete,
            "all_customers_served": all_customers_served,
            "successful": operational_metrics["success"],
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

    def structural_mask_fn(self) -> np.ndarray:
        """Return the mask that hides only slots denoting no action at all.

        Every semantically defined candidate stays selectable, including ones
        the feasibility engine would reject, so a policy trained against this
        mask has to learn feasibility from experience.
        """
        return get_structural_action_mask(self)

    def policy_mask_fn(self) -> np.ndarray:
        """Return whichever mask this configuration hands to a learning policy."""
        if self.policy_action_mask == "structural":
            return self.structural_mask_fn()
        return self.mask_fn()

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
        """Describe the global action slot shared by both routing modes."""
        if 0 <= action < self.num_charging_nodes:
            return f"Go to charger @ node {self.charging_nodes[action]}"
        if self.num_charging_nodes <= action < self.num_navigation_actions:
            slot = action - self.num_charging_nodes
            if self.joint_instance is not None:
                if slot == self.fixed_num_stops:
                    return f"Return to depot @ node {self.joint_instance.depot_node}"
                if slot < len(self.joint_instance.tasks):
                    node = self.joint_instance.tasks[slot].node_id
                    status = self.task_registry.task_for_node(node).status.value
                    return f"Go to delivery slot {slot} @ node {node} ({status})"
            return f"Go to delivery slot {slot} (empty)"
        charge_idx = action - self.num_navigation_actions
        if 0 <= charge_idx < len(self.charge_action_values):
            value = self.charge_action_values[charge_idx]
            if self.charging_action_mode == "target_soc":
                return f"Charge to {value:.0%} SoC"
            return f"Charge for {value:g}h"
        return f"Invalid action {action}"

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
