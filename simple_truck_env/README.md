# Simple Truck Environment

A simplified single-agent reinforcement learning environment for electric truck routing and charging optimization.

## Overview

`SimpleTruckEnv` is a Gymnasium-compatible environment that models a single electric truck completing a delivery sequence while managing its battery charge. Unlike the hierarchical multi-agent approach in `truck_env`, this environment uses a **single-agent paradigm** where each truck is controlled by one agent.

## Key Features

- ✅ **Single-agent control**: One agent controls one truck per episode
- ✅ **Discrete action space**: Combined navigation and charging decisions
- ✅ **Realistic physics**: Battery discharge based on distance and terrain
- ✅ **Dynamic delivery sequences**: Randomly generated routes with configurable constraints
- ✅ **Charging station network**: Full charging infrastructure integration
- ✅ **Time-based rewards**: Optimize for fastest delivery completion

## Architecture

### Components

1. **TransportationGraph** (`transportation_graph.py`)
   - Manages the road network graph
   - Provides routing and distance calculations
   - Generates delivery sequences with hop distance constraints
   - Finds nearest charging stations

2. **Truck** (`truck.py`)
   - Represents individual truck state
   - Tracks battery level, position, and deliveries
   - Manages movement and charging operations
   - Provides state observations

3. **SimpleTruckEnv** (`simple_truck_env.py`)
   - Main Gymnasium environment
   - Implements MDP (states, actions, rewards, transitions)
   - Handles episode lifecycle (reset, step, termination)

## Action Space

**Discrete(N)** where N = num_charging_nodes + 1 + 4

Actions are divided into two categories:

### Navigation Actions (0 to num_charging_nodes)
- `0` to `num_charging_nodes-1`: Go to specific charging station
- `num_charging_nodes`: Go to next delivery location

### Charging Actions (num_charging_nodes+1 to num_charging_nodes+4)
- `num_charging_nodes+1`: Charge for 1 hour
- `num_charging_nodes+2`: Charge for 2 hours
- `num_charging_nodes+3`: Charge for 3 hours
- `num_charging_nodes+4`: Charge for 4 hours

**Example**: With 25 charging nodes:
- Actions 0-24: Go to charging stations
- Action 25: Go to next delivery
- Actions 26-29: Charge for 1, 2, 3, or 4 hours

## Observation Space

**Box(10,)** - Continuous vector with:

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | current_node | [0, 1] | Current node ID (normalized) |
| 1 | next_delivery_node | [0, 1] | Next delivery target (normalized) |
| 2 | battery_level | [0, 500] | Current battery in kWh |
| 3 | battery_percentage | [0, 100] | Battery as percentage of capacity |
| 4 | is_charging | {0, 1} | Whether currently charging |
| 5 | deliveries_remaining | [0, num_stops] | Number of undelivered stops |
| 6 | nearest_charger_distance | [0, 1000] | Distance to nearest charger (km) |
| 7 | can_reach_next | {0, 1} | Whether battery sufficient for next delivery |
| 8 | time_elapsed | [0, 1000] | Total time spent (hours) |
| 9 | distance_traveled | [0, 5000] | Total distance covered (km) |

## Reward Function

The environment uses a **time-based reward** to encourage fast delivery:

- **Base reward**: `-time_spent` for each action (negative hours)
- **Delivery bonus**: `+50` for each completed delivery
- **Completion bonus**: `+1000` for completing all deliveries
- **Failure penalty**: `-500` for running out of battery
- **Invalid action penalty**: `-10` for illegal moves (e.g., no path exists)
- **Insufficient battery penalty**: `-50` for attempting impossible moves

**Goal**: Maximize cumulative reward = minimize time + complete all deliveries

## Usage

### Basic Example

```python
from simple_truck_env import SimpleTruckEnv

# Create environment
env = SimpleTruckEnv(
    num_stops=3,              # 3 delivery stops per truck
    min_hop_distance=20.0,    # Min 20km between stops
    max_hop_distance=150.0,   # Max 150km between stops
    max_steps=200,            # Episode limit
    verbose=True              # Print detailed info
)

# Reset for new episode
obs, info = env.reset(seed=42)

# Run episode
for step in range(100):
    action = env.action_space.sample()  # Random policy
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Episode finished: {info}")
        break

env.close()
```

### Training with RLlib

```python
from ray.rllib.algorithms.ppo import PPOConfig
from simple_truck_env import SimpleTruckEnv

config = (
    PPOConfig()
    .environment(
        SimpleTruckEnv,
        env_config={
            "num_stops": 5,
            "min_hop_distance": 30.0,
            "max_hop_distance": 120.0,
            "max_steps": 300,
            "verbose": False
        }
    )
    .training(
        lr=0.0003,
        gamma=0.99,
        train_batch_size=4000
    )
    .rollouts(num_rollout_workers=4)
)

algo = config.build()

for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: reward={result['env_runners']['episode_reward_mean']}")
```

### Custom Delivery Routes

```python
env = SimpleTruckEnv(num_stops=10, verbose=False)
obs, info = env.reset()

# Access truck state
truck_state = info['truck_state']
print(f"Delivery sequence: {truck_state['delivery_sequence']}")
print(f"Total route distance: {truck_state['total_distance_to_travel']:.1f} km")
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_stops` | int | 3 | Number of delivery stops per episode |
| `min_hop_distance` | float | 20.0 | Minimum distance between consecutive stops (km) |
| `max_hop_distance` | float | 150.0 | Maximum distance between consecutive stops (km) |
| `max_steps` | int | 200 | Maximum steps before episode truncation |
| `verbose` | bool | False | Print detailed step-by-step information |

## Episode Termination

An episode ends when:

1. ✅ **Success**: All deliveries completed (`truck.is_complete == True`)
2. ❌ **Failure**: Truck runs out of battery (`truck.failed == True`)
3. ⏱️ **Truncation**: Maximum steps reached

## Truck Types

The environment supports two truck types (randomly selected each episode):

### Standard Truck
- Battery capacity: 300 kWh
- Base speed: 40 km/h
- Best for: Short to medium routes

### Heavy Truck
- Battery capacity: 500 kWh
- Base speed: 35 km/h
- Best for: Long-haul deliveries

## Testing

Run the test suite:

```bash
python scripts/test_simple_env.py
```

Tests include:
- ✅ Basic functionality (reset, step, observations)
- ✅ Navigation actions (movement to deliveries and chargers)
- ✅ Charging actions (battery replenishment)
- ✅ Delivery completion (full episode simulation)

## Comparison with `truck_env`

| Feature | `truck_env` (Multi-Agent) | `simple_truck_env` (Single-Agent) |
|---------|---------------------------|-----------------------------------|
| **Paradigm** | Hierarchical: 2 agents per truck | Single agent per truck |
| **Action Space** | Discrete (route) + Box (charge) | Unified Discrete(N) |
| **Complexity** | High (agent coordination) | Low (single decision maker) |
| **Training** | Difficult (multi-agent RL) | Easier (standard RL) |
| **Use Case** | Research, complex scenarios | Production, rapid prototyping |

## Future Enhancements

- [ ] **Charging queue simulation**: Model waiting times when multiple trucks use same charger
- [ ] **Dynamic charger speeds**: Vary charging rates by station type
- [ ] **Traffic conditions**: Time-dependent edge weights
- [ ] **Weather effects**: Dynamic terrain factors
- [ ] **Multi-truck coordination**: Fleet-level optimization
- [ ] **Demand forecasting**: Predict delivery requests
- [ ] **Battery degradation**: Long-term capacity reduction

## File Structure

```
simple_truck_env/
├── __init__.py                    # Package exports
├── simple_truck_env.py            # Main environment class
├── transportation_graph.py        # Graph management
└── truck.py                       # Truck state and operations
```

## Dependencies

- `gymnasium` - RL environment interface
- `numpy` - Numerical operations
- `networkx` - Graph algorithms
- `pickle` - Data persistence
- `ray[rllib]` - Optional, for training

## License

[Same as parent project]

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{simple_truck_env,
  title={Simple Truck Environment: A Single-Agent RL Framework for Electric Vehicle Routing},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]}
}
```
