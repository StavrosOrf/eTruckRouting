# Simple Truck Environment - Event-Driven Simulation

## Overview

This package provides an **event-driven** simulation environment for electric truck routing and charging optimization. The environment uses a **global clock** and processes events chronologically, with a **single-agent** controlling whichever truck is ready for a decision.

## Quick Start

```python
from simple_truck_env import EventDrivenTruckEnv, load_config

# Create environment
config = load_config()
env = EventDrivenTruckEnv(config=config)

# Reset
obs, info = env.reset(seed=42)

# Run episode
done = False
while not done:
    action = env.action_space.sample()  # Your policy here
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"Episode finished at t={info['global_clock']:.2f}h")
print(f"Total reward: {info['episode_reward']:.2f}")
```

## Key Features

### 🕐 Global Clock
- Time advances to the next event (not fixed timesteps)
- Realistic simulation of routing and charging durations
- Episode ends when `global_clock >= max_time` or all trucks done

### 📋 Event Queue
- Priority queue of events ordered by time
- Event types: `TRUCK_READY`, `ROUTE_COMPLETE`, `CHARGE_COMPLETE`, `TRUCK_TERMINATED`
- Automatic event processing between decision points

### 🤖 Single-Agent Control
- Controls one truck at a time (the active truck)
- Standard `Discrete` action space (not multi-agent)
- Compatible with single-agent RL algorithms (PPO, DQN, A2C, etc.)

## Environment Spaces

### Action Space
`Discrete(30)` - Single action for the active truck

**Actions** (with 25 charging stations):
- `0-24`: Navigate to charging station 0-24
- `25`: Navigate to next delivery stop
- `26-29`: Charge for 1-4 hours (if at charging station)

### Observation Space
`Box(13,)` - Float array with:
1. Current node (normalized)
2. Next delivery node (normalized)
3. Battery level (kWh)
4. Battery percentage (0-100)
5. Is charging (0/1)
6. Deliveries remaining
7. Nearest charger distance (km)
8. Can reach next delivery (0/1)
9. Truck time elapsed (hours)
10. Truck distance traveled (km)
11. **Global clock** (hours)
12. **Active trucks count**
13. **Events pending in queue**

## Configuration

Edit `config.yaml`:

```yaml
environment:
  num_stops: 5              # Delivery stops per truck
  min_hop_distance: 30.0    # Min km between stops
  max_hop_distance: 120.0   # Max km between stops
  max_time: 48.0           # Max simulation time (hours)
  verbose: false

advanced:
  num_trucks: 3            # Number of trucks

rewards:
  time_penalty: -1.0       # Per hour
  distance_penalty: -0.1   # Per km
  charge_penalty: -2.0     # Per charging hour
  delivery_bonus: 50.0     # Per delivery
```

## Usage Examples

### Basic Usage
```python
from simple_truck_env import create_env_from_config

env = create_env_from_config()
obs, info = env.reset()

print(f"Active truck: {info['active_truck_id']}")
print(f"Clock: {info['global_clock']:.2f}h")
```

### Smart Strategy
```python
def smart_policy(env, truck):
    battery_pct = truck.get_battery_percentage()
    at_charger = truck.current_node in env.charging_nodes
    
    if battery_pct < 20.0:
        return 0  # Go to nearest charger
    elif battery_pct < 50.0 and at_charger:
        return env.num_navigation_actions + 1  # Charge 2h
    else:
        return env.num_charging_nodes  # Next delivery

# Use it
action = smart_policy(env, env.trucks[env.active_truck_id])
obs, reward, terminated, truncated, info = env.step(action)
```

### Monitor Progress
```python
while not done:
    action = policy(env, env.trucks[env.active_truck_id])
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"t={info['global_clock']:6.2f}h | "
          f"Truck {info['active_truck_id']} | "
          f"R={reward:+6.2f} | "
          f"Active: {info['num_active_trucks']}")
    
    done = terminated or truncated
```

## Event-Driven Architecture

### How It Works

1. **Reset**: All trucks start at t=0, each gets a `TRUCK_READY` event
2. **Step**: Active truck takes action, which schedules a future event
3. **Advance**: Clock jumps to next `TRUCK_READY` event
4. **Repeat**: Until all trucks complete or time limit exceeded

### Example Timeline
```
t=0.0h  → Truck 0 READY → [action: go to delivery] → routing...
          Truck 1 READY → [action: go to charger] → routing...

t=1.5h  → Truck 1 arrives at charger → READY
          [action: charge 2h] → charging...

t=2.3h  → Truck 0 arrives at delivery → READY
          [action: next delivery] → routing...

t=3.5h  → Truck 1 charging complete → READY
          ...
```

## Truck States

- `"active"`: Ready for decision (waiting for `step()`)
- `"routing"`: Traveling to a node
- `"charging"`: Charging at station
- `"complete"`: All deliveries done ✅
- `"failed"`: Ran out of battery ❌

## Termination

**Terminated** (natural): All trucks complete or failed

**Truncated** (time limit): `global_clock >= max_time`

## Rewards

- Navigation: `-time_penalty * hours - distance_penalty * km`
- Delivery reached: `+delivery_bonus`
- Charging: `-charge_penalty * hours`
- Invalid action: `-10.0`

## Files

- `event_driven_env.py` - Main environment class
- `truck.py` - Truck state and operations
- `transportation_graph.py` - Graph utilities
- `config.yaml` - Configuration file
- `config_utils.py` - Config loading utilities

## Testing

```bash
# Run all tests
python scripts/test_event_driven_env.py --test all

# Visual demo
python scripts/test_event_driven_env.py --demo
```

## Dependencies

```bash
pip install gymnasium networkx pyyaml numpy
```

## See Also

- `EVENT_DRIVEN_GUIDE.md` - Detailed architecture documentation
- `config.yaml` - Full configuration options
- `scripts/test_event_driven_env.py` - Test suite and examples
