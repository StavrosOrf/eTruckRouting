# Quick Start: Multi-Truck SimpleTruckEnv

## Installation
```bash
cd /home/sorfanouda/EVPR
source .venv/bin/activate
```

## 1. Basic Usage

```python
from simple_truck_env import SimpleTruckEnv

# Create environment (uses default config.yaml)
env = SimpleTruckEnv()

# Reset environment
obs, info = env.reset(seed=42)

# Get action and step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# Check truck states
for truck in info['trucks']:
    print(f"Truck {truck['truck_id']}: {truck['battery_percentage']:.1f}% battery")
```

## 2. Custom Configuration

```python
from simple_truck_env import SimpleTruckEnv

# Create with custom parameters
env = SimpleTruckEnv(
    num_trucks=5,           # 5 trucks
    num_stops=10,           # 10 deliveries each
    max_steps=500,          # 500 step limit
    verbose=True            # Print detailed logs
)
```

## 3. Using Config File

```python
from simple_truck_env import create_env_from_config, load_config

# Method 1: Direct creation
env = create_env_from_config()

# Method 2: Load and modify config
config = load_config()
config['advanced']['num_trucks'] = 8
config['environment']['num_stops'] = 6
env = SimpleTruckEnv(config=config)
```

## 4. Action Format (MultiDiscrete)

```python
import numpy as np

# For 3 trucks: action shape = (6,)
# [nav_0, charge_0, nav_1, charge_1, nav_2, charge_2]

# Example: All trucks go to deliveries
action = np.array([
    25, 0,  # Truck 0: next delivery, no charge
    25, 0,  # Truck 1: next delivery, no charge
    25, 0,  # Truck 2: next delivery, no charge
])

# Example: Mixed actions
action = np.array([
    0,  2,  # Truck 0: charger 0, charge 2 hours
    25, 0,  # Truck 1: next delivery, no charge
    1,  3,  # Truck 2: charger 1, charge 3 hours
])
```

## 5. Monitoring Trucks

```python
obs, info = env.reset()

# Access truck states
for truck_state in info['trucks']:
    print(f"""
    Truck {truck_state['truck_id']}:
      - Type: {truck_state['truck_type']}
      - Battery: {truck_state['battery_percentage']:.1f}%
      - Position: node {truck_state['current_node']}
      - Deliveries left: {truck_state['deliveries_remaining']}
      - Time elapsed: {truck_state['total_time']:.2f} hours
      - Distance: {truck_state['total_distance']:.2f} km
      - Complete: {truck_state['is_complete']}
      - Failed: {truck_state['failed']}
    """)
```

## 6. Edit Config File

Edit `simple_truck_env/config.yaml`:

```yaml
environment:
  num_stops: 5
  max_steps: 300

advanced:
  num_trucks: 3  # Change number of trucks here

truck:
  type_selection: "random"  # "random", "standard", or "heavy"
  initial_battery: "full"    # "full", "random", or percentage

rewards:
  delivery_bonus: 50.0
  completion_bonus: 1000.0
```

## 7. Training with RLlib

```python
from ray.rllib.algorithms.ppo import PPOConfig
from simple_truck_env import SimpleTruckEnv

config = PPOConfig()
config.environment(SimpleTruckEnv, env_config={
    "num_trucks": 4,
    "num_stops": 5,
    "max_steps": 300
})

algo = config.build()
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: {result['env_runners']['episode_reward_mean']}")
```

## 8. Test Scripts

```bash
# Test multi-truck environment
python scripts/test_multitruck_env.py

# Run examples
python scripts/example_multitruck.py

# Config demonstrations
python scripts/demo_config.py
```

## Key Files

- `simple_truck_env/config.yaml` - Main configuration
- `simple_truck_env/simple_truck_env.py` - Environment class
- `simple_truck_env/config_utils.py` - Config utilities
- `simple_truck_env/truck.py` - Truck class
- `simple_truck_env/transportation_graph.py` - Graph utilities

## Action Space Reference

| Component | Values | Description |
|-----------|--------|-------------|
| Navigation | 0 to num_charging_nodes-1 | Go to specific charging station |
| Navigation | num_charging_nodes | Go to next delivery |
| Charging | 0 | No charging |
| Charging | 1-4 | Charge for 1, 2, 3, or 4 hours |

**Total action length = 2 × num_trucks**

## Common Patterns

### Pattern 1: Greedy Delivery
```python
# Always go to next delivery
action = np.array([env.num_charging_nodes, 0] * env.num_trucks)
```

### Pattern 2: Charge When Low
```python
action = []
for truck in info['trucks']:
    if truck['battery_percentage'] < 30:
        nav = 0  # Go to charger
        charge = 2  # Charge 2 hours
    else:
        nav = env.num_charging_nodes  # Next delivery
        charge = 0
    action.extend([nav, charge])
action = np.array(action)
```

### Pattern 3: Random Exploration
```python
action = env.action_space.sample()
```

## Troubleshooting

**Issue: Action shape mismatch**
```python
# Check expected shape
print(env.action_space.nvec)  # e.g., [26, 5, 26, 5, 26, 5]
print(len(action))  # Should match length of nvec
```

**Issue: All trucks failed**
```python
# Increase charging or adjust strategy
# Check battery levels in verbose mode
env = SimpleTruckEnv(verbose=True)
```

**Issue: Config not loading**
```python
# Verify config path
from simple_truck_env import load_config
config = load_config()
print(config.keys())
```
