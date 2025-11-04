# Hierarchical Truck Routing Environment - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Environment Flow](#environment-flow)
5. [Action Spaces](#action-spaces)
6. [Observation Spaces](#observation-spaces)
7. [Reward System](#reward-system)
8. [Charging Mechanics](#charging-mechanics)
9. [Multi-Agent Structure](#multi-agent-structure)
10. [Training Integration](#training-integration)

---

## Overview

The **Hierarchical Truck Routing Environment** (`HierarchicalTruckRoutingEnv`) is a multi-agent reinforcement learning environment designed to optimize electric truck routing and charging decisions. It simulates multiple electric trucks navigating a road network, making decisions about routes and when to charge their batteries.

### Key Features
- **Multi-Agent**: Each truck has two agents (route planner and charge manager)
- **Hierarchical Decision Making**: High-level routing and low-level charging decisions
- **Realistic Physics**: Battery discharge based on distance, terrain, and efficiency
- **Dynamic Charging**: Multiple charger types (fast/slow) with occupancy management
- **Graph-Based Network**: Uses NetworkX for road network representation

---

## Architecture

### Class Hierarchy
```
MultiAgentEnv (RLlib)
    └── HierarchicalTruckRoutingEnv
```

### File Structure
```
truck_env/
├── truck_env.py       # Main environment implementation
├── utils.py           # Helper functions, graph, spaces
├── reward.py          # Reward function definitions
├── __init__.py        # Package initialization
└── data/              # Network and charger data
    ├── shortest_path_energy_dict.pkl
    ├── shortest_path_time_dict.pkl
    └── station_info_dict.pkl
```

---

## Core Components

### 1. Environment Initialization

```python
def __init__(self, config=None):
    super().__init__()
    self.config = config or {}
    
    # Debug flags
    self.verbose = self.config.get("verbose", False)
    self.debug = self.config.get("debug", False)
    
    # Core components
    self.graph = get_graph()
    self.truck_configs = get_truck_configs()
    self.charger_configs = get_charger_configs(self.graph)
    self.charger_occupancy = get_charger_occupancy_template(self.graph)
    self.truck_types = get_truck_types()
```

**Components Initialized:**
- **Graph**: Road network (NetworkX DiGraph) with nodes and edges
- **Truck Configs**: Starting positions, destinations, battery levels
- **Charger Configs**: Location and capacity of charging stations
- **Charger Occupancy**: Real-time tracking of charger usage
- **Truck Types**: Vehicle specifications (battery capacity, speed, efficiency)

### 2. Graph Structure

The graph is loaded from pickle files containing real-world data:

```python
def get_graph_new():
    edge_distance = read_file(edge_distance_file)  # Distance between nodes
    edge_time = read_file(edge_time_file)          # Travel time
    chargers = read_file(chargers_file)            # Charger locations
```

**Node Properties:**
- `has_charger`: Boolean indicating if node has charging station
- `charger_type`: Dictionary with charger types and capacities
  - Example: `{"fast": 3, "slow": 2}` (3 fast chargers, 2 slow chargers)
- `original_id`: Original OSM (OpenStreetMap) node ID

**Edge Properties:**
- `distance`: Distance in kilometers
- `time`: Travel time
- `terrain_factor`: Multiplier for discharge (1.0 = flat, >1.0 = hilly)

**Node Indexing:**
The environment uses integer indices (0, 1, 2, ...) internally for efficiency, with mappings:
- `node_to_index`: OSM ID → integer index
- `index_to_node`: integer index → OSM ID

### 3. Truck Configuration

Each truck has a configuration defining its mission:

```python
{
    "id": 0,
    "start_node": 5026447875,      # OSM node ID (converted to index)
    "end_node": 5433392625,        # Destination
    "initial_battery": 300.0,      # Starting battery (kWh)
    "truck_type": "standard"       # Truck specification
}
```

**Truck Types:**
```python
"standard": {
    "battery_capacity": 300.0,           # kWh
    "base_speed": 50.0,                  # km/h
    "base_discharge_function": discharge_standard,
    "base_charge_function": charge_standard
}

"heavy": {
    "battery_capacity": 300.0,
    "base_speed": 40.0,                  # Slower than standard
    "base_discharge_function": discharge_standard,
    "base_charge_function": charge_standard
}
```

### 4. Truck State

During execution, each truck maintains:

```python
truck_state = {
    "id": 0,                           # Normalized ID (0.0 to 1.0)
    "current_node": 42,                # Current position
    "destination_node": 156,           # Goal position
    "current_battery": 250.5,          # Current charge (kWh)
    "battery_capacity": 300.0,         # Max capacity
    "truck_type": "standard",          # Type specification
    "is_charging": False,              # Currently charging?
    "charging_type": "fast",           # Type of charger (if charging)
    "waiting_time": 0.0,               # Time waiting for charger
    "time_elapsed": 12.5,              # Total time since start
    "total_distance": 150.3,           # Total distance traveled
    "charging_sessions": 2,            # Number of charges
    "route": [0, 5, 12, 42]           # Path taken (for evaluation)
}
```

---

## Environment Flow

### Reset Process

```python
def reset(self, *, seed=None, options=None):
    # 1. Initialize truck states from configs
    for i, config in enumerate(self.truck_configs):
        truck_state = {
            "current_node": config["start_node"],
            "destination_node": config["end_node"],
            "current_battery": config["initial_battery"],
            # ... other fields
        }
        self.trucks.append(truck_state)
    
    # 2. Reset charger occupancy
    self.charger_occupancy = {
        node: {ctype: 0 for ctype in self.charger_configs[node]}
        for node in self.charger_configs
    }
    
    # 3. Reset time
    self.global_time = 0.0
    
    # 4. Generate observations
    obs = self._get_observations()
    
    return obs, {}
```

### Step Process

The `step()` function executes one timestep:

```python
def step(self, action_dict):
    # 1. Process high-level actions (route planning)
    for i, truck in enumerate(self.trucks):
        high_agent = f"truck_{i}_route_planner"
        if high_agent in action_dict:
            target_node = action_dict[high_agent]
            self._execute_route_action(truck, target_node, rewards, terminateds)
    
    # 2. Process low-level actions (charging management)
    for i, truck in enumerate(self.trucks):
        low_agent = f"truck_{i}_charge_manager"
        if low_agent in action_dict:
            charge_action = action_dict[low_agent]
            self._execute_charge_action(truck, charge_action, rewards)
    
    # 3. Update global time
    self.global_time = max(truck["time_elapsed"] for truck in self.trucks)
    
    # 4. Check termination conditions
    terminateds["__all__"] = all(truck_done for truck in self.trucks)
    
    # 5. Generate observations
    observations = self._get_observations()
    
    return observations, rewards, terminateds, truncateds, infos
```

---

## Action Spaces

### High-Level Actions (Route Planning)

**Space Type:** `Discrete(num_nodes)`

**Meaning:** The agent selects the next node to move to.

```python
action = 42  # Move to node 42
```

**Valid Actions:**
- Any node index from 0 to `num_nodes - 1`
- **Constraint**: Edge must exist in graph (checked during execution)
- **Penalty**: Invalid moves (no edge) receive -5.0 reward

**Execution Logic:**
```python
def _execute_route_action(self, truck, target_node, rewards, terminateds):
    # Check if staying at current node
    if target_node == truck["current_node"]:
        rewards[high_agent] = 0.0
        return
    
    # Check if edge exists
    if not self.graph.has_edge(truck["current_node"], target_node):
        rewards[high_agent] = -5.0
        return
    
    # Calculate travel requirements
    edge_data = self.graph[truck["current_node"]][target_node]
    travel_time = edge_data["distance"] / truck_speed
    discharge = discharge_function(truck, edge_data, travel_time, self.global_time)
    
    # Check battery
    if truck["current_battery"] < discharge:
        rewards[high_agent] = penalty_run_out_of_energy() / 5
        return  # Don't move
    
    # Execute move
    truck["current_battery"] -= discharge
    truck["current_node"] = target_node
    truck["time_elapsed"] += travel_time
    truck["total_distance"] += edge_data["distance"]
    
    # Calculate rewards
    reward = reward_move_to_next_node()  # +2.0
    reward += penalty_time_elapsed(travel_time)  # -0.1 * time
    
    if truck["current_node"] == truck["destination_node"]:
        reward += reward_arrive_destination()  # +50.0
        terminateds[high_agent] = True
```

### Low-Level Actions (Charging Management)

**Space Type:** `Discrete(4)`

**Actions:**
- **0**: Do nothing
- **1**: Start charging
- **2**: Stop charging
- **3**: Wait for charger

**Execution Logic:**

```python
def _execute_charge_action(self, truck, action, rewards):
    low_agent = f"truck_{truck['id']}_charge_manager"
    current_node = truck["current_node"]
    
    # Check if at charging station
    if current_node not in self.charger_configs:
        rewards[low_agent] = 0.0
        return
    
    if action == 1:  # Start charging
        if truck["is_charging"]:
            rewards[low_agent] = 0.0  # Already charging
            return
        
        # Find available charger
        for ctype in ["fast", "slow"]:
            if self.charger_occupancy[current_node][ctype] < max_capacity:
                truck["charging_type"] = ctype
                self.charger_occupancy[current_node][ctype] += 1
                truck["is_charging"] = True
                truck["waiting_time"] = 0.0
                rewards[low_agent] = 1.0
                return
        
        # No charger available
        truck["waiting_time"] += 1.0
        rewards[low_agent] = penalty_wait_at_charger(1.0)  # -0.5
    
    elif action == 2:  # Stop charging
        if not truck["is_charging"]:
            rewards[low_agent] = 0.0
            return
        
        # Calculate charge received
        ctype = truck["charging_type"]
        charge_time = 1.0
        charge_amount = charge_function(...)
        
        # Update state
        truck["current_battery"] = min(capacity, current + charge_amount)
        truck["is_charging"] = False
        truck["charging_sessions"] += 1
        truck["time_elapsed"] += charge_time
        self.charger_occupancy[current_node][ctype] -= 1
        
        rewards[low_agent] = reward_finish_charging()  # +5.0
    
    elif action == 3:  # Wait for charger
        # Applies penalty if waiting unnecessarily
        any_available = any(occupancy < capacity for all chargers)
        if not any_available:
            truck["waiting_time"] += 1.0
            rewards[low_agent] = penalty_wait_at_charger(1.0)
        else:
            rewards[low_agent] = -1.0  # Unnecessary wait
```

---

## Observation Spaces

### Observation Structure

Each agent receives a **flattened** observation vector containing:

```python
observation_space = Dict({
    "id": Box(0.0, 1.0, shape=(), dtype=np.float32),
    "current_node": Discrete(num_nodes),
    "destination_node": Discrete(num_nodes),
    "battery_level": Box(0.0, 300.0, shape=(), dtype=np.float32),
    "battery_capacity": Box(0.0, 300.0, shape=(), dtype=np.float32),
    "is_charging": Discrete(2),
    "charger_available": Discrete(2),
    "charger_occupancy_fast": Box(0.0, 100.0, (), np.float32),
    "charger_occupancy_slow": Box(0.0, 100.0, (), np.float32),
    "time_elapsed": Box(0.0, 1000.0, shape=(), dtype=np.float32),
    "waiting_time": Box(0.0, 800.0, shape=(), dtype=np.float32),
    "can_reach_destination": Discrete(2),
    "nearest_charger_distance": Box(0.0, 900.0, shape=(), dtype=np.float32),
})
```

### Observation Generation

```python
def _get_observations(self):
    observations = {}
    
    for i, truck in enumerate(self.trucks):
        # Calculate derived features
        nearest_charger_dist = self._get_nearest_charger_distance(truck)
        can_reach = self._can_reach_destination(truck)
        
        # Get charger info
        current_node = truck["current_node"]
        charger_available = 1 if current_node in self.charger_configs else 0
        charger_occupancy = self.charger_occupancy.get(current_node, {})
        
        # Build observation dict
        obs = {
            "id": normalized_truck_id,
            "current_node": truck["current_node"],
            "destination_node": truck["destination_node"],
            "battery_level": truck["current_battery"],
            "battery_capacity": truck["battery_capacity"],
            "is_charging": int(truck["is_charging"]),
            "charger_available": charger_available,
            "charger_occupancy_fast": charger_occupancy.get("fast", 0),
            "charger_occupancy_slow": charger_occupancy.get("slow", 0),
            "time_elapsed": truck["time_elapsed"],
            "waiting_time": truck["waiting_time"],
            "can_reach_destination": int(can_reach),
            "nearest_charger_distance": nearest_charger_dist,
        }
        
        # Flatten to vector
        flat_obs = flatten(self._raw_obs_space, obs)
        
        # Both agents get same observation
        observations[f"truck_{i}_route_planner"] = flat_obs
        observations[f"truck_{i}_charge_manager"] = flat_obs
    
    return observations
```

### Helper Functions

**Nearest Charger Distance:**
```python
def _get_nearest_charger_distance(self, truck):
    current_node = truck["current_node"]
    min_distance = float("inf")
    
    for charger_node in self.charger_configs.keys():
        if charger_node != current_node:
            try:
                path_length = nx.shortest_path_length(
                    self.graph, current_node, charger_node, weight="distance"
                )
                min_distance = min(min_distance, path_length)
            except nx.NetworkXNoPath:
                continue
    
    return min_distance if min_distance != float("inf") else 0.0
```

**Can Reach Destination:**
```python
def _can_reach_destination(self, truck):
    try:
        path_length = nx.shortest_path_length(
            self.graph,
            truck["current_node"],
            truck["destination_node"],
            weight="distance"
        )
        estimated_discharge = discharge_function(...)
        return truck["current_battery"] >= estimated_discharge
    except nx.NetworkXNoPath:
        return False
```

---

## Reward System

### Reward Functions (from `reward.py`)

```python
def reward_move_to_next_node():
    """Small reward for moving to next node."""
    return 2.0

def reward_finish_charging():
    """Medium reward for completing charging."""
    return 5.0

def reward_arrive_destination():
    """Big reward for reaching destination."""
    return 50.0

def penalty_wait_at_charger(wait_time: float):
    """Small penalty based on waiting time."""
    return -0.5 * wait_time

def penalty_run_out_of_energy():
    """Big penalty for running out of energy."""
    return -100.0

def penalty_time_elapsed(time_elapsed: float):
    """Penalty to optimize for minimum time."""
    return -0.1 * time_elapsed
```

### Reward Structure by Agent

**High-Level Agent (Route Planner):**
- **+2.0** per valid move to next node
- **-0.1 × time** for time elapsed during travel
- **+50.0** for reaching destination
- **-5.0** for invalid move (no edge)
- **-20.0** for insufficient battery (penalty_run_out_of_energy / 5)

**Low-Level Agent (Charge Manager):**
- **+1.0** for successfully starting to charge
- **+5.0** for finishing charging session
- **-0.5 × time** for waiting at occupied charger
- **-1.0** for unnecessary waiting (charger available but chose to wait)
- **0.0** for do-nothing action

---

## Charging Mechanics

### Discharge Function

Battery consumption when traveling:

```python
def discharge_function(truck_config, edge_data, travel_time, current_time):
    truck_type = get_truck_types()[truck_config["truck_type"]]
    
    # Base discharge
    base_discharge = truck_type["base_discharge_function"](
        truck_config["current_battery"], 
        edge_data["distance"]
    )
    
    # Terrain modifier (from edge data)
    terrain_modifier = edge_data.get("terrain_factor", 1.0)
    
    # Time-based modifier (currently disabled)
    time_modifier = 1.0
    
    # Battery efficiency (currently constant)
    battery_efficiency = 1.0
    
    total_discharge = (base_discharge * terrain_modifier * 
                      time_modifier * battery_efficiency)
    
    return max(0, total_discharge)
```

**Base Discharge:**
```python
def discharge_standard(current_charge, distance=1):
    return 0.2 * distance  # 0.2 kWh per km
```

**Example:**
- Distance: 50 km
- Terrain factor: 1.2 (hilly)
- Discharge: 0.2 × 50 × 1.2 = **12 kWh**

### Charge Function

Battery gained during charging:

```python
def charge_function(graph, truck_config, charger_node, charge_time, 
                   current_time, charger_type):
    truck_type = truck_types[truck_config["truck_type"]]
    current_battery = truck_config["current_battery"]
    
    # Base charge
    base_charge = truck_type["base_charge_function"](
        current_battery, 
        charge_time
    )
    
    # Efficiency (currently constant)
    efficiency = 1.0
    
    # Non-linear charging (slower as battery fills)
    battery_ratio = current_battery / battery_capacity
    if battery_ratio < 0.5:
        charge_efficiency = 1.0    # Full speed
    elif battery_ratio < 0.8:
        charge_efficiency = 0.7    # 70% speed
    else:
        charge_efficiency = 0.4    # 40% speed (almost full)
    
    charge_efficiency = 1.0  # Currently disabled
    
    # Time modifier (currently disabled)
    time_modifier = 1.0
    
    total_charge = base_charge * efficiency * charge_efficiency * time_modifier
    
    # Don't exceed capacity
    max_possible = battery_capacity - current_battery
    return min(total_charge, max_possible)
```

**Base Charge:**
```python
def charge_standard(current_charge, time=1):
    return 0.8 * time  # 0.8 kWh per time unit
```

**Charger Types:**
- **Fast (DCFC)**: 200 kWh/hour
- **Slow (Level 2)**: 20 kWh/hour
- **Note**: Currently the charger type multiplier is commented out in code

### Charger Occupancy Management

**Data Structure:**
```python
self.charger_occupancy = {
    42: {"fast": 2, "slow": 1},  # Node 42: 2 fast chargers occupied, 1 slow
    56: {"fast": 0, "slow": 3},  # Node 56: 0 fast, 3 slow occupied
}

self.charger_configs = {
    42: {"fast": 5, "slow": 2},  # Node 42 has 5 fast, 2 slow chargers total
    56: {"fast": 3, "slow": 5},  # Node 56 has 3 fast, 5 slow chargers total
}
```

**Occupancy Logic:**
1. When truck starts charging: increment occupancy for that charger type
2. When truck stops charging: decrement occupancy
3. Truck can only charge if `occupancy < capacity` for some charger type

---

## Multi-Agent Structure

### Agent Naming Convention

For `N` trucks, there are `2N` agents:

```
Truck 0:
  - truck_0_route_planner    (high-level)
  - truck_0_charge_manager   (low-level)

Truck 1:
  - truck_1_route_planner
  - truck_1_charge_manager

...

Truck N-1:
  - truck_{N-1}_route_planner
  - truck_{N-1}_charge_manager
```

### Agent Initialization

```python
self.high_level_agents = [
    f"truck_{i}_route_planner" for i in range(self.num_trucks)
]

self.low_level_agents = [
    f"truck_{i}_charge_manager" for i in range(self.num_trucks)
]

self.all_agents = self.high_level_agents + self.low_level_agents
self.possible_agents = self.all_agents
self.agents = self.all_agents.copy()
self._agent_ids = set(self.all_agents)
```

### Action/Observation Spaces per Agent

```python
self._action_space_dict = {}
self._observation_space_dict = {}

for i in range(self.num_trucks):
    high_agent = f"truck_{i}_route_planner"
    low_agent = f"truck_{i}_charge_manager"
    
    # Action spaces
    self._action_space_dict[high_agent] = Discrete(num_nodes)
    self._action_space_dict[low_agent] = Discrete(4)
    
    # Observation spaces (same for both)
    self._observation_space_dict[high_agent] = flat_obs_space
    self._observation_space_dict[low_agent] = flat_obs_space
```

### Termination Logic

**Individual Agent Termination:**
```python
# Truck reaches destination
if truck["current_node"] == truck["destination_node"]:
    terminateds[f"truck_{i}_route_planner"] = True
    terminateds[f"truck_{i}_charge_manager"] = True

# Truck runs out of battery (currently just penalized, not terminated)
```

**Episode Termination:**
```python
# All trucks done OR timeout
all_done = self.current_step >= 1000
truck_statuses = [
    truck["current_node"] == truck["destination_node"] or 
    truck["current_battery"] <= 0
    for truck in self.trucks
]

terminateds["__all__"] = all(truck_statuses) or all_done
truncateds["__all__"] = all_done
```

---

## Training Integration

### RLlib Configuration

The environment is trained using Ray RLlib with PPO (Proximal Policy Optimization).

**Policy Mapping:**
```python
def policy_mapping_fn(agent_id, episode, **kwargs):
    if agent_id.endswith("_route_planner"):
        return "high_level_policy"
    elif agent_id.endswith("_charge_manager"):
        return "low_level_policy"
```

**Two Separate Policies:**

**High-Level Policy (Route Planning):**
- Network: [128, 64, 32] neurons
- Activation: ReLU
- Learning rate: 0.0005
- Entropy coefficient: 0.02 (more exploration)

**Low-Level Policy (Charge Management):**
- Network: [64, 32, 16] neurons
- Activation: Tanh
- Learning rate: 0.0003
- Entropy coefficient: 0.01 (less exploration)

### Training Configuration

```python
config = (
    PPOConfig()
    .environment("hierarchical_truck_env", env_config={})
    .multi_agent(
        policies={
            "high_level_policy": (None, flat_obs_space, 
                                 Discrete(num_nodes), 
                                 high_level_config),
            "low_level_policy": (None, flat_obs_space, 
                                Discrete(4), 
                                low_level_config),
        },
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["high_level_policy", "low_level_policy"],
    )
    .env_runners(
        rollout_fragment_length=10,
        num_env_runners=2,
        num_envs_per_env_runner=1,
    )
    .training(
        train_batch_size=2000,
        minibatch_size=256,
        num_epochs=10,
        lr=0.0003,
        entropy_coeff=0.01,
    )
)
```

### Training Loop

```python
algo = config.build()

for iteration in range(30):
    result = algo.train()
    print(f"Iteration {iteration + 1}:")
    print(f"  Episode Reward Mean: {result['episode_return_mean']}")
    print(f"  High Level Reward: {result['high_level_policy_reward']}")
    print(f"  Low Level Reward: {result['low_level_policy_reward']}")

checkpoint_path = algo.save('./saved_models/')
```

### Evaluation

```python
def eval(checkpoint_path):
    algo = config.build()
    algo.restore(checkpoint_path)
    
    eval_env = HierarchicalTruckRoutingEnv()
    
    for episode in range(3):
        obs, _ = eval_env.reset()
        terminated = {"__all__": False}
        
        while not terminated["__all__"]:
            actions = {}
            for agent_id in eval_env.agents:
                policy_id = policy_mapping_fn(agent_id, None)
                actions[agent_id] = algo.compute_single_action(
                    obs[agent_id], 
                    policy_id=policy_id,
                    explore=False
                )
            
            obs, rewards, terminated, truncated, info = eval_env.step(actions)
        
        # Print results: route, charging sessions, distance, etc.
```

---

## Complete Example Flow

### Scenario: Single Truck Journey

**Initial State:**
```
Truck 0:
  - Position: Node 0
  - Destination: Node 4
  - Battery: 25/300 kWh
```

**Step 1:**
```
Action: {
  "truck_0_route_planner": 1,      # Move to node 1
  "truck_0_charge_manager": 0      # Do nothing
}

Execution:
  - Check edge exists: 0 → 1 ✓
  - Calculate discharge: 10 km × 0.2 = 2 kWh
  - Battery sufficient: 25 > 2 ✓
  - Move truck to node 1
  - Battery: 25 - 2 = 23 kWh
  - Time: +0.2 hours (10 km / 50 km/h)

Rewards:
  - truck_0_route_planner: +2.0 (move) - 0.02 (time) = +1.98
  - truck_0_charge_manager: 0.0
```

**Step 2:**
```
Action: {
  "truck_0_route_planner": 1,      # Stay at node 1
  "truck_0_charge_manager": 1      # Start charging
}

Execution:
  - Route action: No move (same node) → reward 0.0
  - Charge action:
    - Check if at charger: Node 1 has chargers ✓
    - Find available charger: "fast" type available ✓
    - Start charging (occupancy: fast=1)
    - Reset waiting time
  - Battery: Still 23 kWh (charging starts next step)

Rewards:
  - truck_0_route_planner: 0.0
  - truck_0_charge_manager: +1.0 (start charging)
```

**Step 3:**
```
Action: {
  "truck_0_route_planner": 1,      # Stay at node 1
  "truck_0_charge_manager": 2      # Stop charging
}

Execution:
  - Route action: No move → reward 0.0
  - Charge action:
    - Currently charging ✓
    - Calculate charge: 0.8 × 1.0 = 0.8 kWh
    - Battery: 23 + 0.8 = 23.8 kWh
    - Release charger (occupancy: fast=0)
    - Time: +1.0

Rewards:
  - truck_0_route_planner: 0.0
  - truck_0_charge_manager: +5.0 (finish charging)
```

**Continue until destination reached...**

---

## Key Design Decisions

### 1. Hierarchical Structure
**Why?** Separates strategic (where to go) from tactical (when to charge) decisions, making learning more efficient.

### 2. Shared Observations
**Why?** Both agents of same truck need same information about truck state.

### 3. Multiple Charger Types
**Why?** Realistic - fast chargers are scarce but efficient; slow chargers are common but slow.

### 4. Occupancy Tracking
**Why?** Creates coordination challenge - trucks must wait or find alternative chargers.

### 5. Time-Based Penalties
**Why?** Encourages efficient routing and minimal waiting.

### 6. Non-Terminal Battery Depletion
**Why?** Allows agent to recover through learning rather than hard failure.

### 7. Graph-Based Network
**Why?** Flexibility to use real-world road networks from OSM data.

---

## Common Pitfalls & Solutions

### Problem: Agent doesn't learn to charge
**Solution:** Ensure `reward_finish_charging` is high enough to overcome time penalties.

### Problem: Trucks get stuck waiting at chargers
**Solution:** Adjust `penalty_wait_at_charger` to encourage finding alternative chargers.

### Problem: Invalid moves (no edge)
**Solution:** Mask invalid actions in policy or increase penalty for invalid moves.

### Problem: Observation space too large
**Solution:** Currently flattened to vector. Could use graph neural network for better scalability.

### Problem: Training slow
**Solution:** Reduce `num_env_runners`, use smaller networks, or reduce graph size.

---

## Future Enhancements

1. **Dynamic Pricing**: Charger costs vary by time/location
2. **Stochastic Travel Times**: Traffic congestion simulation
3. **Battery Degradation**: Long-term battery health
4. **Multi-Objective**: Balance time, cost, battery health
5. **Communication**: Trucks share charger availability info
6. **Hierarchical Routing**: Include waypoint selection
7. **Action Masking**: Prevent invalid actions at policy level
8. **Graph Neural Networks**: Better handling of large road networks

---

## Conclusion

The Hierarchical Truck Routing Environment provides a rich testbed for multi-agent reinforcement learning in electric vehicle logistics. Its hierarchical structure, realistic physics, and multi-agent interactions create interesting optimization challenges that require coordinated decision-making.

Key strengths:
- ✅ Realistic battery/charging mechanics
- ✅ Real-world road network support
- ✅ Multi-agent coordination
- ✅ Hierarchical decision making
- ✅ Extensible design

The environment successfully balances complexity with learning efficiency, making it suitable for both research and practical applications in EV fleet management.
