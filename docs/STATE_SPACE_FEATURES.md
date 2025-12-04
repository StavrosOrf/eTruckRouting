# Flattened State Space Features

## Overview
The flattened state space provides a complete representation of the environment state as a 1D numpy array. It mirrors the GNN state space structure but in a flattened format suitable for traditional RL algorithms.

**Total Dimensions**: Variable based on configuration
- For default setup (2 trucks, 3 deliveries, 25 chargers): **167 features**

## State Space Structure

### 1. Truck Features (14 features per truck)
Position in state: `[0:num_trucks*14]`

For each active truck, the following 14 features are included:

| Index | Feature Name | Description | Normalization | Range |
|-------|--------------|-------------|---------------|-------|
| 0 | `node_type` | Type of current location (0=depot, 1=delivery, 2=charger) | `/3` | [0.0, 0.667] |
| 1 | `current_position` | Current node ID | `/num_nodes` | [0.0, 1.0] |
| 2 | `battery_soc` | State of charge | Direct | [0.0, 1.0] |
| 3 | `state_idle` | Truck in IDLE state | Binary | {0.0, 1.0} |
| 4 | `state_ready` | Truck in READY state | Binary | {0.0, 1.0} |
| 5 | `state_moving` | Truck in MOVING state | Binary | {0.0, 1.0} |
| 6 | `state_charging` | Truck in CHARGING state | Binary | {0.0, 1.0} |
| 7 | `state_waiting` | Truck in WAITING state | Binary | {0.0, 1.0} |
| 8 | `state_delivering` | Truck in DELIVERING state | Binary | {0.0, 1.0} |
| 9 | `state_failed` | Truck has failed | Binary | {0.0, 1.0} |
| 10 | `deliveries_completed` | Number of completed deliveries | `/num_stops` | [0.0, 1.0] |
| 11 | `time_elapsed` | Total time elapsed | `/max_time` | [0.0, 1.0] |
| 12 | `distance_traveled` | Total distance traveled | `/1000` | [0.0, ∞) |
| 13 | `total_charging_time` | Total time spent charging | `/max_time` | [0.0, 1.0] |

**Note**: For completed or failed trucks, all 14 features are set to **zero**.

### 2. Delivery Features (5 features per delivery)
Position in state: `[num_trucks*14 : num_trucks*14 + num_stops*5]`

For each delivery location:

| Index | Feature Name | Description | Normalization | Range |
|-------|--------------|-------------|---------------|-------|
| 0 | `node_type` | Type of node (always 1 for delivery) | `/3` | 0.333 |
| 1 | `node_id` | Delivery node ID | `/num_nodes` | [0.0, 1.0] |
| 2 | `delivery_sequence_index` | Position in delivery sequence (min across active trucks) | `/num_stops` | [0.0, 1.0] |
| 3 | `energy_to_reach` | Energy required from active truck's position | `/1000` | [0.0, ∞) |
| 4 | `time_to_reach` | Time required from active truck's position | `/max_time` | [0.0, ∞) |

**Note**: 
- Energy and time features are calculated from the active truck's current position
- If no active truck, energy and time are set to zero
- Sequence index is the minimum position across all active trucks' remaining deliveries

### 3. Charger Features (6 features per charger)
Position in state: `[num_trucks*14 + num_stops*5 : num_trucks*14 + num_stops*5 + num_charging_nodes*6]`

For each charging station:

| Index | Feature Name | Description | Normalization | Range |
|-------|--------------|-------------|---------------|-------|
| 0 | `node_type` | Type of node (always 2 for charger) | `/3` | 0.667 |
| 1 | `node_id` | Charger node ID | `/num_charging_nodes` | [0.0, 1.0] |
| 2 | `occupancy_rate` | Current occupancy / capacity | Direct ratio | [0.0, 1.0] |
| 3 | `queue_length` | Number of trucks waiting | `/num_trucks` | [0.0, 1.0] |
| 4 | `energy_to_reach` | Energy required from active truck's position | `/1000` | [0.0, ∞) |
| 5 | `time_to_reach` | Time required from active truck's position | `/max_time` | [0.0, ∞) |

**Note**: 
- Energy and time features are calculated from the active truck's current position
- If no active truck, energy and time are set to zero

### 4. Global Features (2 features)
Position in state: `[-2:]` (last 2 features)

| Index | Feature Name | Description | Normalization | Range |
|-------|--------------|-------------|---------------|-------|
| 0 | `current_time` | Current simulation time | `/max_time` | [0.0, 1.0] |
| 1 | `active_truck_mask` | Binary mask indicating which truck is acting | Binary | {0.0, 1.0} |

## Dimension Calculation

For a configuration with:
- `N` trucks
- `D` deliveries
- `C` charging nodes

**Total state dimensions** = `14*N + 5*D + 6*C + 2`

### Example Configurations

| Config | Trucks | Deliveries | Chargers | Total Dimensions |
|--------|--------|------------|----------|------------------|
| Default | 2 | 3 | 25 | 14×2 + 5×3 + 6×25 + 2 = **181** |
| Small | 1 | 2 | 10 | 14×1 + 5×2 + 6×10 + 2 = **86** |
| Large | 5 | 10 | 50 | 14×5 + 5×10 + 6×50 + 2 = **422** |

## Feature Sources

### From Truck Objects
- Position and battery state
- State machine status (IDLE, READY, MOVING, CHARGING, WAITING, DELIVERING, FAILED)
- Delivery progress
- Time and distance metrics

### From Transportation Graph
- Node types and IDs
- Energy consumption calculations
- Travel time estimates

### From Charging Station
- Charger occupancy and capacity
- Queue information

### From Environment
- Global time
- Active truck identification

## Normalization Schemes

| Normalization Type | Formula | Purpose |
|-------------------|---------|---------|
| Node Type | `value / 3` | Standard scale for 3 node types (depot, delivery, charger) |
| Node ID | `id / num_nodes` | Relative position in graph |
| Time | `time / max_time` | Normalize to episode duration |
| Distance | `distance / 1000` | Scale to reasonable range |
| Energy | `energy / 1000` | Scale to reasonable range |
| State of Charge | Direct | Already in [0, 1] range |
| Binary States | `{0, 1}` | One-hot encoding |
| Counts | `count / max_count` | Normalize by maximum possible value |

## Special Cases

### Completed/Failed Trucks
When a truck completes all deliveries or fails:
- All 14 truck features are set to **0.0**
- This creates a clear distinction between active and inactive trucks
- Maintains consistent state space dimensions

### Missing Active Truck
When calculating energy/time features:
- If no active truck is specified: features set to **0.0**
- If path is invalid (inf distance): features set to **0.0**

### Zero Normalization Constants
When denominators are zero:
- Feature is set to **0.0** to avoid division errors
- Examples: `node_id / 0`, `time / 0`

## Comparison with GNN State Space

The flattened state space includes the **exact same information** as the GNN state space:

| GNN Component | Flattened Equivalent | Features |
|---------------|---------------------|----------|
| Truck nodes | Truck features | 14 per truck |
| Delivery nodes | Delivery features | 5 per delivery |
| Charger nodes | Charger features | 6 per charger |
| Global attributes | Global features | 2 total |

**Key Differences**:
1. **Structure**: GNN uses heterogeneous graph; flattened uses 1D array
2. **Edges**: GNN has explicit edges; flattened has implicit relationships
3. **Processing**: GNN uses message passing; flattened uses standard neural networks
4. **Active Truck Context**: Both calculate energy/time from active truck's position

## Usage Example

```python
from EVRoutingEnv.state.state_space import FlattenedStateSpace

# Initialize
state_space = FlattenedStateSpace(
    num_trucks=2,
    num_stops=3,
    num_charging_nodes=25,
    max_time=480,  # 8 hours in minutes
    graph=transport_graph
)

# Get observation space
obs_space = state_space.get_observation_space()
print(f"State dimensions: {obs_space.shape}")  # (181,)

# Get current state
state = state_space.get_state(
    trucks=trucks,
    transport_graph=graph,
    charging_station=charging_station,
    current_time=current_time,
    active_truck_id=0
)

# Access specific features
truck_features = state[:28]  # First 2 trucks
delivery_features = state[28:43]  # 3 deliveries
charger_features = state[43:193]  # 25 chargers
global_features = state[193:]  # Last 2 features
```

## Validation

The state space includes comprehensive validation:
- Dimension checking against GNN state space
- Feature-by-feature comparison tests
- 100-episode consistency validation
- Immediate error reporting on mismatches

See `test_state_space.py` for full test suite.
