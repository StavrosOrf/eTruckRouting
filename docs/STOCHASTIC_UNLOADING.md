# Stochastic Unloading Time at Delivery Locations

## Overview

This feature adds configurable stochastic unloading time at delivery locations, simulating realistic variability in unloading operations. The implementation follows the same reproducible pattern as the existing traffic simulation system to ensure consistency across different RL algorithm runs.

## Key Features

- **Time-of-Day Dependent Variation**: Unloading time variance increases during business hours (9am-5pm) to represent variable conditions like dock availability, facility traffic, and staffing levels
- **Reproducible Randomness**: Same seed produces identical unloading times across different algorithm runs, ensuring fair comparison
- **Bounded Values**: Unloading times are capped to prevent extreme outliers
- **Configurable Parameters**: All settings controlled via `config.yaml`
- **Event-Driven Integration**: Seamlessly integrates with the event-driven simulation architecture

## Configuration

### Config File: `EVRoutingEnv/config_files/config.yaml`

```yaml
delivery:
  # Enable/disable stochastic unloading time
  enable_stochastic_unloading: True
  
  # Base unloading time (hours)
  base_unloading_time: 0.5  # 30 minutes
  
  # Standard deviation as fraction of base time
  std_dev_factor: 0.20  # 20% variation during off-hours
  
  # Maximum std_dev cap (hours)
  max_std_dev_hours: 0.25  # 15 minutes max deviation
  
  # Business hours multiplier (9am-5pm)
  business_hours_multiplier: 1.5  # 1.5x variance during business hours
  
  # Bounds for unloading time
  min_unloading_multiplier: 0.75  # -25% (fast unload)
  max_unloading_multiplier: 1.5   # +50% (slow unload)
```

### Parameters Explained

| Parameter | Description | Example |
|-----------|-------------|---------|
| `enable_stochastic_unloading` | Toggle feature on/off | `True` / `False` |
| `base_unloading_time` | Base unloading duration (hours) | `0.5` = 30 minutes |
| `std_dev_factor` | Std dev as fraction of base time | `0.20` = 20% variation |
| `max_std_dev_hours` | Maximum allowed std dev (hours) | `0.25` = 15 min max |
| `business_hours_multiplier` | Variance multiplier for 9am-5pm | `1.5` = 1.5x variance |
| `min_unloading_multiplier` | Lower bound multiplier | `0.75` = minimum 75% of base |
| `max_unloading_multiplier` | Upper bound multiplier | `1.5` = maximum 150% of base |

## How It Works

### 1. Gaussian Distribution with Time-of-Day Effects

Unloading time is sampled from a Gaussian distribution:

```
actual_time ~ N(base_time, std_dev²)
```

Where `std_dev` depends on whether it's business hours (9am-5pm):

```python
# Calculate std_dev
base_std_dev = base_unloading_time * std_dev_factor
if is_business_hours:  # 9am-5pm
    std_dev = base_std_dev * business_hours_multiplier
else:
    std_dev = base_std_dev

# Cap std_dev
std_dev = min(std_dev, max_std_dev_hours)

# Sample from distribution
actual_time = base_time + std_dev * random_normal()

# Apply bounds
actual_time = clip(actual_time, 
                   base_time * min_multiplier, 
                   base_time * max_multiplier)
```

### 2. Reproducibility Mechanism

The simulator uses a deterministic seeding approach based on:
- **delivery_node**: Location of delivery
- **time_bucket**: Time discretized to 0.5-hour buckets
- **delivery_idx**: Counter for multiple deliveries at same node/time

This ensures the same delivery (node, time, occurrence) always gets the same random value across different algorithm runs with the same seed.

### 3. Event-Driven Integration

When a truck arrives at a delivery location:

1. **Arrival Event**: `TRUCK_ROUTING` event fires, truck arrives at delivery node
2. **Apply Unloading Time**: `DeliverySimulator.apply_unloading_time()` calculates stochastic time
3. **Update Truck State**: Truck state changes to `"unloading"`
4. **Schedule Ready Event**: `TRUCK_READY` event scheduled after unloading completes
5. **Resume Operations**: When ready event fires, truck state changes to `"ready"` and can take next action

### 4. State Machine

New truck state added:
- **`"unloading"`**: Truck is currently unloading at a delivery location

State transitions:
```
"routing" → "unloading" → "ready"
     ↓
(arrival at delivery)
```

## Implementation Files

### Core Files Modified/Created

1. **`EVRoutingEnv/utils/delivery_simulator.py`** (NEW)
   - `DeliverySimulator` class
   - `apply_unloading_time()`: Calculate stochastic unloading time
   - `_get_uncertainty_value()`: Reproducible random value generation
   - `reset_delivery_counters()`: Reset for new episode

2. **`EVRoutingEnv/config_files/config.yaml`** (MODIFIED)
   - Added `delivery:` section with all configuration parameters

3. **`EVRoutingEnv/models/environment/event_driven_env.py`** (MODIFIED)
   - Added `DeliverySimulator` import and initialization
   - Seed propagation in `reset()`
   - Counter reset for new episodes
   - Updated truck state comments to include `"unloading"`
   - Modified TRUCK_ROUTING event handling to skip scheduling TRUCK_READY if truck is unloading

4. **`EVRoutingEnv/models/environment/event_handlers.py`** (MODIFIED)
   - Updated `handle_truck_routing()` to accept `delivery_simulator` parameter
   - Apply unloading time when truck arrives at delivery
   - Schedule delayed TRUCK_READY event after unloading
   - Set truck state to `"unloading"`

5. **`EVRoutingEnv/models/core/truck.py`** (MODIFIED)
   - Added `total_unloading_time` attribute
   - Track cumulative unloading time across episode
   - Include in `get_state_dict()` output

## Usage Examples

### Basic Usage

```python
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv

# Create environment (uses config.yaml)
env = EventDrivenTruckEnv(config="EVRoutingEnv/config_files/config.yaml", verbose=True)

# Reset with seed for reproducibility
obs, info = env.reset(seed=42)

# Run episode
for step in range(100):
    action_mask = info["action_mask"]
    action = select_action(obs, action_mask)  # Your policy
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check unloading status
    for truck_id, state in env.truck_states.items():
        if state == "unloading":
            print(f"Truck {truck_id} is unloading")
    
    if terminated or truncated:
        break

# Check final statistics
for truck in env.trucks:
    print(f"Truck {truck.truck_id}:")
    print(f"  Total unloading time: {truck.total_unloading_time:.2f}h")
    print(f"  Total time: {truck.total_time_elapsed:.2f}h")
```

### Disable Feature

To disable stochastic unloading time:

```yaml
delivery:
  enable_stochastic_unloading: False
  base_unloading_time: 0.5  # Constant time used instead
```

### Adjust Time-of-Day Effects

To increase variance during business hours:

```yaml
delivery:
  business_hours_multiplier: 2.0  # 2x variance during 9am-5pm
```

To disable time-of-day effects:

```yaml
delivery:
  business_hours_multiplier: 1.0  # Same variance all day
```

## Testing

Run the test suite to verify the implementation:

```bash
python test_stochastic_unloading.py
```

Tests include:
1. **Reproducibility**: Same seed produces identical unloading times
2. **Different Seeds**: Different seeds produce different results
3. **Disabled Feature**: Feature can be disabled via config
4. **Verbose Output**: Debug information is displayed correctly

## Impact on RL Training

### Benefits

1. **Realistic Modeling**: Captures real-world variability in delivery operations
2. **Reproducibility**: Same seed ensures consistent training across algorithms
3. **Fairness**: All algorithms see identical uncertainty for same state-action pairs
4. **Robustness**: Trains policies to handle variable unloading times

### Considerations

- **Total Episode Time**: Unloading time contributes to `total_time_elapsed`, affecting time-based rewards
- **State Space**: Unloading state is tracked but not directly observable in state vector
- **Action Masking**: No impact on action feasibility during unloading (truck waits automatically)
- **Reward Calculation**: Unloading time adds to episode duration, potentially decreasing time-based rewards

## Comparison with Traffic Simulation

| Aspect | Traffic Simulation | Unloading Time |
|--------|-------------------|----------------|
| **When Applied** | During navigation (travel) | After arrival at delivery |
| **Duration Base** | Travel time between nodes | Fixed base unloading time |
| **Location** | Edge-based (from→to) | Node-based (delivery location) |
| **Time-of-Day** | Rush hours (7-9am, 5-7pm) | Business hours (9am-5pm) |
| **State Impact** | Extends "routing" state | New "unloading" state |
| **Event Type** | Delays TRUCK_ROUTING event | Delays TRUCK_READY after delivery |
| **Correlation** | Affects energy consumption | Standalone (no correlation) |

## Future Extensions

Potential enhancements:

1. **Location-Specific Multipliers**: Different base times for urban vs. rural deliveries
2. **Load Size Dependency**: Unloading time could depend on delivery size/weight
3. **Crew Availability**: Model limited unloading crew affecting multiple trucks
4. **Queue System**: Multiple trucks unloading at same location with limited docks
5. **Historical Data**: Train distribution parameters from real-world data

## References

- Traffic simulation: `docs/TRAFFIC_SIMULATION.md`
- State space design: `docs/STATE_SPACE_FEATURES.md`
- Truck state machine: `docs/TRUCK_STATE_MACHINE.md`
