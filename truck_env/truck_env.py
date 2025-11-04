from ray.rllib.env.multi_agent_env import MultiAgentEnv
import networkx as nx
from gymnasium.spaces.utils import flatten_space, flatten

from truck_env.utils import (
    get_graph,
    get_truck_types,
    get_truck_configs,
    get_charger_configs,
    get_charger_occupancy_template,
    discharge_function,
    charge_function,
    get_high_level_action_space,
    get_low_level_action_space,
    get_observation_space,
)

from truck_env.reward import (
    reward_move_to_next_node,
    reward_finish_charging,
    reward_arrive_destination,
    penalty_wait_at_charger,
    penalty_run_out_of_energy,
    penalty_time_elapsed,
)


class HierarchicalTruckRoutingEnv(MultiAgentEnv):
    """Hierarchical environment for truck routing with charging optimization."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        
        # Verbose/debug flag for detailed printing
        self.verbose = self.config.get("verbose", False)
        self.debug = self.config.get("debug", False)

        # Initialize environment components
        self.graph = get_graph()
        self.truck_configs = get_truck_configs()
        self.num_trucks = len(self.truck_configs)
        self.charger_configs = get_charger_configs(self.graph)
        self.charger_occupancy = get_charger_occupancy_template(self.graph)
        self.truck_types = get_truck_types()
        self.done_agents = set()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🏗️  INITIALIZING HIERARCHICAL TRUCK ROUTING ENVIRONMENT")
            print(f"{'='*80}")
            print(f"  📊 Environment Stats:")
            print(f"    - Number of trucks: {self.num_trucks}")
            print(f"    - Graph nodes: {self.graph.number_of_nodes()}")
            print(f"    - Graph edges: {self.graph.number_of_edges()}")
            print(f"    - Charging stations: {len(self.charger_configs)}")
        
        # Set up agents
        self.high_level_agents = [
            f"truck_{i}_route_planner" for i in range(self.num_trucks)
        ]
        self.low_level_agents = [
            f"truck_{i}_charge_manager" for i in range(self.num_trucks)
        ]
        self.all_agents = self.high_level_agents + self.low_level_agents

        self.possible_agents = self.all_agents
        self.agents = self.all_agents.copy()
        # Set _agent_ids for RLlib MultiAgentEnv compatibility
        self._agent_ids = set(self.all_agents)
        # Define action and observation spaces
        self._action_space_dict = {}
        self._observation_space_dict = {}
        
        self._raw_obs_space = get_observation_space(self.graph)  # , self.num_trucks)
        flat_obs_space = flatten_space(self._raw_obs_space)
        for i in range(self.num_trucks):
            high_agent = f"truck_{i}_route_planner"
            low_agent = f"truck_{i}_charge_manager"

            self._action_space_dict[high_agent] = get_high_level_action_space(
                self.graph
            )
            self._action_space_dict[low_agent] = get_low_level_action_space()
            self._observation_space_dict[high_agent] = flat_obs_space
            self._observation_space_dict[low_agent] = flat_obs_space
            # self._observation_space_dict[high_agent] = get_observation_space(self.graph, self.num_trucks)
            # self._observation_space_dict[low_agent] = get_observation_space(self.graph, self.num_trucks)
        # self.action_space = self._action_space_dict
        # self.observation_space = self._observation_space_dict
        self.reset()
        self.current_step = 0
        
        if self.verbose:
            print(f"\n  ✅ Environment initialized successfully!")
            print(f"  🔄 Total agents: {len(self.agents)}")
            print(f"{'='*80}\n")

    def reset(self, *, seed=None, options=None):
        """Reset environment to initial state."""
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🔄 RESETTING ENVIRONMENT")
            print(f"{'='*80}")
        
        # Initialize truck states
        self.trucks = []
        num_trucks = len(self.trucks)

        for i, config in enumerate(self.truck_configs):
            if num_trucks > 1:
                normalized_id = i / (num_trucks - 1)
            else:
                normalized_id = 0

            truck_type = self.truck_types[config["truck_type"]]
            truck_state = {
                "id": int(normalized_id),
                "current_node": config["start_node"],
                "destination_node": config["end_node"],
                "current_battery": config["initial_battery"],
                "battery_capacity": truck_type["battery_capacity"],
                "truck_type": config["truck_type"],
                "is_charging": False,
                "waiting_time": 0.0,
                "time_elapsed": 0.0,
                "total_distance": 0.0,
                "charging_sessions": 0,
            }
            self.trucks.append(truck_state)

        # Initialize charger occupancy
        # self.charger_occupancy = {node: 0 for node in self.charger_configs.keys()}
        self.charger_occupancy = {
            node: {ctype: 0 for ctype in self.charger_configs[node]}
            for node in self.charger_configs
        }
        # Global time tracking
        self.global_time = 0.0
        # Maintain proper agent list
        self.agents = self.all_agents.copy()

        # Generate observations for all agents
        obs = self._get_observations()
        self.done_agents = set()
        self.current_step = 0  # Reset step counter
        
        if self.verbose:
            print(f"\n  Initialized {len(self.trucks)} trucks:")
            for i, truck in enumerate(self.trucks):
                print(f"    Truck {i}:")
                print(f"      Start: {truck['current_node']} → Destination: {truck['destination_node']}")
                print(f"      Battery: {truck['current_battery']:.1f}/{truck['battery_capacity']:.1f} kWh")
                print(f"      Type: {truck['truck_type']}")
            print(f"\n  📦 Generated observations for {len(obs)} agents")
            if self.debug:
                print(f"  🔍 Observation shapes: {{{', '.join([f'{k}: {v.shape}' for k, v in list(obs.items())[:2]])}, ...}}")
            print(f"{'='*80}\n")
        
        return {agent: obs[agent] for agent in self.agents}, {}

    def step(self, action_dict):
        """Execute one environment step."""
        self.current_step += 1
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"📍 STEP {self.current_step}")
            print(f"{'='*80}")
        
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        
        if self.debug:
            print(f"\n  🎬 Actions received:")
            for agent_id, action in action_dict.items():
                action_type = "ROUTE" if "route_planner" in agent_id else "CHARGE"
                print(f"    [{action_type}] {agent_id}: {action}")
        
        # Process high-level actions (route planning)
        for i, truck in enumerate(self.trucks):
            high_agent = f"truck_{i}_route_planner"

            if high_agent in action_dict:
                target_node = action_dict[high_agent]
                
                if self.verbose:
                    print(f"\n  🚛 Truck {i} Route Planning:")
                    print(f"    Current: {truck['current_node']} → Target: {target_node}")
                    print(f"    Battery: {truck['current_battery']:.2f} kWh")
                
                self._execute_route_action(truck, target_node, rewards, terminateds)
                
                if self.verbose and high_agent in rewards:
                    print(f"    Reward: {rewards[high_agent]:.2f}")
                    print(f"    New position: {truck['current_node']}")
                    print(f"    Battery after: {truck['current_battery']:.2f} kWh")

        # Process low-level actions (charging management)
        for i, truck in enumerate(self.trucks):
            low_agent = f"truck_{i}_charge_manager"

            if low_agent in action_dict:
                charge_action = action_dict[low_agent]
                action_names = ["Do Nothing", "Start Charging", "Stop Charging", "Wait for Charger"]
                
                if self.verbose:
                    print(f"\n  ⚡ Truck {i} Charge Management:")
                    print(f"    Action: {action_names[charge_action] if charge_action < len(action_names) else charge_action}")
                    print(f"    Current node: {truck['current_node']}")
                    print(f"    Battery before: {truck['current_battery']:.2f} kWh")
                    print(f"    Is charging: {truck['is_charging']}")
                
                self._execute_charge_action(truck, charge_action, rewards)
                
                if self.verbose and low_agent in rewards:
                    print(f"    Reward: {rewards[low_agent]:.2f}")
                    print(f"    Battery after: {truck['current_battery']:.2f} kWh")
                    print(f"    Is charging: {truck['is_charging']}")
                    
        # Update global time
        self.global_time = max(truck["time_elapsed"] for truck in self.trucks)

        # Check termination conditions
        # Consider the episode done if either condition is met
        """
        global_done = all(
            truck["current_node"] == truck["destination_node"] or truck["current_battery"] <= 0
            for truck in self.trucks
        )

        # Also end if max time exceeded
        if self.global_time > 1000 or global_done:
            print("⏹️ Forcing episode end")
            if self.global_time > 1000:
                print("⏱️ Episode ended due to time limit")
            elif global_done:
                print("✅ Episode ended: all trucks done or stuck")            
            terminateds["__all__"] = True
            if terminateds[agent]:
                self.done_agents.add(agent)
        else:
            terminateds["__all__"] = False
            
        #observations = self._get_observations()
        all_obs = self._get_observations()
        observations = {aid: obs for aid, obs in all_obs.items() if aid not in self.done_agents}        
        print("📦 Observations returned:", list(observations.keys()))
        
        # Important: mark each agent done individually!
        for agent in self.agents:
            if agent not in terminateds:
                terminateds[agent] = terminateds["__all__"]
            if terminateds[agent]:
                self.done_agents.add(agent)                
            truncateds[agent] = False
            infos[agent] = {}        
        print("Step called:", action_dict)
        print("Terminateds:", terminateds)        
        active_agents = [agent for agent in self.agents if agent not in self.done_agents]

        observations = {aid: obs for aid, obs in self._get_observations().items() if aid in active_agents}
        rewards = {aid: rewards.get(aid, 0.0) for aid in active_agents}
        terminateds = {aid: terminateds.get(aid, False) for aid in active_agents}
        truncateds = {aid: truncateds.get(aid, False) for aid in active_agents}
        #infos = {aid: infos.get(aid, {}) for aid in active_agents}

        terminateds["__all__"] = terminateds.get("__all__", False)
        truncateds["__all__"] = False  # or your truncation logic        
        """
        all_done = self.current_step >= 1000
        truck_statuses = []

        for truck in self.trucks:
            truck_done = (
                truck["current_node"] == truck["destination_node"]
                or truck["current_battery"] <= 0
            )
            truck_statuses.append(truck_done)

            # Set individual agent termination
            high_agent = f"truck_{truck['id']}_route_planner"
            low_agent = f"truck_{truck['id']}_charge_manager"

            terminateds[high_agent] = truck_done
            terminateds[low_agent] = truck_done

        terminateds["__all__"] = all(truck_statuses) or all_done
        truncateds["__all__"] = all_done  # Timeout truncation
        
        # Get observations for all agents
        observations = self._get_observations()
        
        # Set rewards, terminateds, truncateds, and infos for all agents
        for agent_id in self.agents:
            if agent_id not in rewards:
                rewards[agent_id] = 0.0
            if agent_id not in terminateds:
                terminateds[agent_id] = terminateds["__all__"]
            if agent_id not in truncateds:
                truncateds[agent_id] = truncateds.get("__all__", False)
            if agent_id not in infos:
                infos[agent_id] = {}
        
        if self.verbose:
            print(f"\n  📊 Step Summary:")
            print(f"    Global time: {self.global_time:.2f}")
            total_reward = sum(rewards.values())
            print(f"    Total rewards this step: {total_reward:.2f}")
            done_count = sum(1 for status in truck_statuses if status)
            print(f"    Trucks done: {done_count}/{len(self.trucks)}")
            print(f"    Episode done: {terminateds['__all__']}")
            
        if self.debug:
            print(f"\n  💰 Rewards breakdown:")
            for agent_id, reward in rewards.items():
                if abs(reward) > 0.01:  # Only show non-zero rewards
                    print(f"    {agent_id}: {reward:.3f}")
            
            print(f"\n  🚚 Truck states:")
            for i, truck in enumerate(self.trucks):
                status = "✅ DONE" if truck_statuses[i] else "🔄 Active"
                print(f"    Truck {i} {status}:")
                print(f"      Position: {truck['current_node']} (dest: {truck['destination_node']})")
                battery_pct = 100*truck['current_battery']/truck['battery_capacity']
                print(f"      Battery: {truck['current_battery']:.1f}/{truck['battery_capacity']:.1f} kWh ({battery_pct:.1f}%)")
                print(f"      Distance: {truck['total_distance']:.1f} km, Time: {truck['time_elapsed']:.1f}")
                
        if self.verbose:
            print(f"{'='*80}\n")
        
        return observations, rewards, terminateds, truncateds, infos

    def get_action_space(self, agent_id):
        return self._action_space_dict[agent_id]

    def get_observation_space(self, agent_id):
        return self._observation_space_dict[agent_id]

    def _execute_route_action(self, truck, target_node, rewards, terminateds):
        """Execute high-level routing action."""
        high_agent = f"truck_{truck['id']}_route_planner"

        if target_node == truck["current_node"]:
            # Stay at current node
            rewards[high_agent] = 0.0
            return

        if not self.graph.has_edge(truck["current_node"], target_node):
            # Invalid move
            rewards[high_agent] = -5.0
            return

        # Calculate travel requirements
        edge_data = self.graph[truck["current_node"]][target_node]
        travel_time = (
            edge_data["distance"] / self.truck_types[truck["truck_type"]]["base_speed"]
        )

        # Calculate discharge
        discharge = discharge_function(truck, edge_data, travel_time, self.global_time)
        """
        if truck["current_battery"] < discharge:
            # Not enough battery
            rewards[high_agent] = penalty_run_out_of_energy()
            terminateds[high_agent] = True
            terminateds[f"truck_{truck['id']}_charge_manager"] = True
            return
        """
        if truck["current_battery"] < discharge:
            # Penalize but don't terminate - let agent recover
            rewards[high_agent] = penalty_run_out_of_energy() / 5  # Smaller penalty
            print(
                f"⚠️ Truck {truck['id']} insufficient energy "
                f"({truck['current_battery']:.2f} < {discharge:.2f})"
            )
            return  # Skip movement but continue episode
        # Execute move
        truck["current_battery"] -= discharge
        truck["current_node"] = target_node
        truck["time_elapsed"] += travel_time
        truck["total_distance"] += edge_data["distance"]

        # Calculate rewards
        reward = reward_move_to_next_node()
        reward += penalty_time_elapsed(travel_time)  # Encourage efficiency

        if truck["current_node"] == truck["destination_node"]:
            reward += reward_arrive_destination()
            terminateds[high_agent] = True
            terminateds[f"truck_{truck['id']}_charge_manager"] = True

        rewards[high_agent] = reward

    def _execute_charge_action(self, truck, action, rewards):
        """Execute low-level charging action with support for multiple charger types per node.
        Also handles rewards and penalties as specified.
        """
        low_agent = f"truck_{truck['id']}_charge_manager"
        current_node = truck["current_node"]

        # If not at a charging node, do nothing
        if current_node not in self.charger_configs:
            rewards[low_agent] = 0.0
            return

        charger_types = list(self.charger_configs[current_node].keys())
        if not charger_types:
            rewards[low_agent] = 0.0
            return
        print(
            f"⚡ Truck {truck['id']} at node {current_node} took charge action: {action}"
        )

        if action == 1:  # Start charging
            if truck["is_charging"]:
                # Already charging
                rewards[low_agent] = 0.0
                return

            # Try to find an available charger type
            for ctype in charger_types:
                if (
                    self.charger_occupancy[current_node][ctype]
                    < self.charger_configs[current_node][ctype]
                ):
                    truck["charging_type"] = ctype
                    self.charger_occupancy[current_node][ctype] += 1
                    truck["is_charging"] = True
                    # Reset waiting time on successful charge start
                    truck["waiting_time"] = 0.0
                    # Small reward for starting to charge
                    rewards[low_agent] = 1.0
                    return

            # If no charger available, increment waiting time and apply penalty
            truck["waiting_time"] += 1.0
            rewards[low_agent] = penalty_wait_at_charger(1.0)
            return

        elif action == 2:  # Stop charging
            if not truck["is_charging"]:
                # Not charging, do nothing
                rewards[low_agent] = 0.0
                return

            # Calculate charge received
            ctype = truck["charging_type"]
            charge_time = 1.0  # Time unit for charging (adjust as needed)

            charge_amount = charge_function(
                self.graph,
                truck,
                current_node,
                charge_time,
                self.global_time,
                charger_type=ctype,
            )

            # Update battery, time, and charger occupancy
            truck["current_battery"] = min(
                truck["battery_capacity"], truck["current_battery"] + charge_amount
            )
            truck["is_charging"] = False
            truck["charging_sessions"] += 1
            truck["time_elapsed"] += charge_time
            self.charger_occupancy[current_node][ctype] -= 1
            truck["charging_type"] = None

            # Apply reward for finishing charging
            rewards[low_agent] = reward_finish_charging()
            return

        elif action == 3:  # Wait for charger
            # Check if any charger is available
            any_available = any(
                self.charger_occupancy[current_node][ctype]
                < self.charger_configs[current_node][ctype]
                for ctype in charger_types
            )
            if not any_available:
                # No charger available, wait and apply penalty
                truck["waiting_time"] += 1.0
                rewards[low_agent] = penalty_wait_at_charger(1.0)
            else:
                # Charger available, but chose to wait (unnecessary)
                rewards[low_agent] = -1.0
            return

        else:  # Do nothing (action 0 or invalid)
            rewards[low_agent] = 0.0

    def _can_charge(self, truck):
        """Check if truck can charge at current location."""
        return (
            truck["current_node"] in self.charger_configs
            and self.charger_occupancy[truck["current_node"]]
            < self.charger_configs[truck["current_node"]]["capacity"]
        )

    def _get_observations(self):
        """Get observations for all agents."""
        observations = {}
        num_trucks = len(self.trucks)
        for i, truck in enumerate(self.trucks):
            high_agent = f"truck_{i}_route_planner"
            low_agent = f"truck_{i}_charge_manager"

            if num_trucks > 1:
                normalized_id = i / (num_trucks - 1)
            else:
                normalized_id = 0
            # Calculate nearest charger distance
            nearest_charger_dist = self._get_nearest_charger_distance(truck)

            # Check if can reach destination with current battery
            can_reach = self._can_reach_destination(truck)

            # Get charger info for current node
            current_node = truck["current_node"]
            charger_available = 1 if current_node in self.charger_configs else 0
            # charger_occupancy = self.charger_occupancy.get(current_node, 0)
            # charger_capacity = self.charger_configs.get(current_node, {}).get("capacity", 0)
            charger_occupancy = self.charger_occupancy.get(current_node, {})
            obs = {
                "id": int(normalized_id),
                "current_node": truck["current_node"],
                "destination_node": truck["destination_node"],
                "battery_level": truck["current_battery"],
                "battery_capacity": truck["battery_capacity"],
                "is_charging": int(truck["is_charging"]),
                "charger_available": charger_available,
                # "charger_occupancy": charger_occupancy,
                # "charger_capacity": charger_capacity,
                "time_elapsed": truck["time_elapsed"],
                "waiting_time": truck["waiting_time"],
                "can_reach_destination": int(can_reach),
                "nearest_charger_distance": nearest_charger_dist,
            }
            for ctype in ["fast", "slow"]:  # or use your actual charger types
                obs[f"charger_occupancy_{ctype}"] = charger_occupancy.get(ctype, 0)
            # pprint.pprint(f"OBS is {obs}")
            # pprint.pprint(f"_raw_obs_space is {self._raw_obs_space}")
            flat_obs = flatten(self._raw_obs_space, obs)
            print(f"Flattened obs shape: {flat_obs.shape}, type: {type(flat_obs)}")
            observations[high_agent] = flat_obs  # .copy()
            observations[low_agent] = flat_obs  # .copy()

        return observations

    def _get_nearest_charger_distance(self, truck):
        """Calculate distance to nearest charger."""
        current_node = truck["current_node"]
        min_distance = float("inf")

        for charger_node in self.charger_configs.keys():
            if charger_node != current_node and self.graph.has_node(charger_node):
                try:
                    path_length = nx.shortest_path_length(
                        self.graph, current_node, charger_node, weight="distance"
                    )
                    min_distance = min(min_distance, path_length)
                except nx.NetworkXNoPath:
                    continue

        return min_distance if min_distance != float("inf") else 0.0

    def _can_reach_destination(self, truck):
        """Check if truck can reach destination with current battery."""
        try:
            path_length = nx.shortest_path_length(
                self.graph,
                truck["current_node"],
                truck["destination_node"],
                weight="distance",
            )
            estimated_discharge = self.truck_types[truck["truck_type"]][
                "base_discharge_function"
            ](truck["current_battery"], path_length)
            return truck["current_battery"] >= estimated_discharge
        except nx.NetworkXNoPath:
            return False

    # MultiAgentEnv interface methods
    def observation_space_contains(self, x):
        """Check if x is a valid observation."""
        if not isinstance(x, dict):
            return False
        for agent_id, obs in x.items():
            if agent_id not in self._observation_space_dict:
                return False
            if not self._observation_space_dict[agent_id].contains(obs):
                return False
        return True

    def action_space_contains(self, x):
        """Check if x is a valid action."""
        if not isinstance(x, dict):
            return False
        for agent_id, action in x.items():
            if agent_id not in self._action_space_dict:
                return False
            if not self._action_space_dict[agent_id].contains(action):
                return False
        return True

    def observation_space_sample(self, agent_ids=None):
        """Sample from the observation space."""
        if agent_ids is None:
            agent_ids = self._agent_ids
        return {
            agent_id: self._observation_space_dict[agent_id].sample()
            for agent_id in agent_ids
        }

    def action_space_sample(self, agent_ids=None):
        """Sample from the action space."""
        if agent_ids is None:
            agent_ids = self._agent_ids
        return {
            agent_id: self._action_space_dict[agent_id].sample()
            for agent_id in agent_ids
        }
