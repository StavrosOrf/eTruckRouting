# Event-Driven Truck Environment

## Overview

The `EventDrivenTruckEnv` implements an **event-driven simulation** with a **global clock** and **single-agent control**. Unlike the multi-agent `SimpleTruckEnv`, this environment advances time to the next event and only requests actions when a truck needs to make a decision.

## Key Concepts

### 1. Global Clock
- **Time Variable**: `self.global_clock` tracks the current simulation time in hours
- **Time Progression**: Time jumps forward to the next event (not fixed timesteps)
- **Max Time**: Episode ends if clock exceeds `max_time` (default: 48 hours)

### 2. Event Queue
- **Priority Queue**: Events are stored in a min-heap ordered by time
- **Event Processing**: Events are processed chronologically
- **Event Types**:
  - `TRUCK_READY`: Truck needs a decision (triggers `step()`)
  - `ROUTE_COMPLETE`: Truck finished traveling to a node
  - `CHARGE_COMPLETE`: Truck finished charging
  - `TRUCK_TERMINATED`: Truck completed deliveries or failed

### 3. Single-Agent Paradigm
- **Active Truck**: Only one truck is "active" at a time (needs a decision)
- **Action Space**: `Discrete(30)` - single action for the active truck
- **Observation**: State of the currently active truck + global state
- **Control Flow**: Agent controls whichever truck becomes ready first

## Architecture

```
Episode Timeline:

t=0.0h  ─┬─> Truck 0 READY ──[action]──> Routing to delivery...
         │
         ├─> Truck 1 READY ──[action]──> Routing to charger...
         │
t=1.5h  ─┼─> Truck 1 ROUTE_COMPLETE ──> Arrived at charger
         │                               Truck 1 READY ──[action]──> Charging 2h...
         │
t=2.3h  ─┼─> Truck 0 ROUTE_COMPLETE ──> Arrived at delivery
         │                               Truck 0 READY ──[action]──> Routing to next...
         │
t=3.5h  ─┼─> Truck 1 CHARGE_COMPLETE ──> Charged 100 kWh
         │                                Truck 1 READY ──[action]──> ...
         │
         ⋮
```

## Event Flow

### Reset
1. Create all trucks with delivery sequences
2. Add `TRUCK_READY` event for each truck at t=0.0
3. Process event queue until first `TRUCK_READY` is found
4. Return observation of active truck

### Step(action)
1. Execute action for active truck:
   - **Navigation**: Schedule `ROUTE_COMPLETE` at t + travel_time
   - **Charging**: Schedule `CHARGE_COMPLETE` at t + charge_hours
2. Advance clock by processing events until next `TRUCK_READY`
3. Return observation of new active truck

### Event Handlers
- **ROUTE_COMPLETE**: Update truck position, check if complete/failed, schedule `TRUCK_READY`
- **CHARGE_COMPLETE**: Update truck battery, schedule `TRUCK_READY`
- **TRUCK_TERMINATED**: Mark truck as complete/failed (no more events)

## Truck States

Each truck can be in one of these states:
- `"active"`: Waiting for a decision (can call `step()`)
- `"routing"`: Traveling to a node (waiting for event)
- `"charging"`: Charging at a station (waiting for event)
- `"complete"`: All deliveries done ✅
- `"failed"`: Ran out of battery ❌

## Action Space

**Type**: `Discrete(30)` for environment with 25 chargers

**Actions**:
- `0-24`: Navigate to charging station 0-24
- `25`: Navigate to next delivery
- `26`: Charge for 1 hour (if at charger)
- `27`: Charge for 2 hours (if at charger)
- `28`: Charge for 3 hours (if at charger)
- `29`: Charge for 4 hours (if at charger)

## Observation Space

**Type**: `Box(13,)` - float array

**Features** (for active truck):
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
12. **Number of active trucks**
13. **Events pending in queue**

## Rewards

- **Time Penalty**: `-1.0` per hour (routing or charging)
- **Distance Penalty**: `-0.1` per km traveled
- **Charge Penalty**: `-2.0` per hour charging
- **Delivery Bonus**: `+50.0` when reaching delivery node
- **Invalid Action**: `-10.0` for illegal moves

## Termination Conditions

**Terminated** (natural end):
- All trucks have completed their deliveries OR failed

**Truncated** (time limit):
- `global_clock >= max_time`

## Comparison: Step-Based vs Event-Driven

| Aspect | SimpleTruckEnv (Step-Based) | EventDrivenTruckEnv (Event-Driven) |
|--------|----------------------------|-----------------------------------|
| **Time** | Fixed timesteps | Continuous, event-based |
| **Control** | All trucks simultaneously | One active truck at a time |
| **Action Space** | `MultiDiscrete([30, 30, 30])` | `Discrete(30)` |
| **Agent** | Multi-agent (3 trucks) | Single-agent (active truck) |
| **Step Semantics** | "All trucks act" | "Active truck acts" |
| **Termination** | Max steps reached | Max time or all done |
| **Observation** | All truck states | Active truck + global |
| **Use Case** | Coordination problems | Sequential decision-making |

## Usage Example

```python
from simple_truck_env import EventDrivenTruckEnv, load_config

# Create environment
config = load_config()
config['advanced']['num_trucks'] = 3
config['environment']['max_time'] = 48.0  # 48 hours

env = EventDrivenTruckEnv(config=config)

# Reset
obs, info = env.reset(seed=42)
print(f"Clock: {info['global_clock']:.2f}h")
print(f"Active truck: {info['active_truck_id']}")

# Run episode
done = False
while not done:
    # Select action for active truck
    action = env.action_space.sample()  # Random policy
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Clock: {info['global_clock']:.2f}h | "
          f"Truck {info['active_truck_id']} | "
          f"Reward: {reward:.2f}")
    
    done = terminated or truncated

print(f"Final time: {info['global_clock']:.2f}h")
print(f"Total reward: {info['episode_reward']:.2f}")
print(f"All complete: {info['all_complete']}")
```

## Implementation Details

### Event Queue (Min-Heap)
```python
import heapq
from dataclasses import dataclass

@dataclass(order=True)
class Event:
    time: float  # Primary sort key
    event_type: EventType
    truck_id: int
    data: Dict
    
# Usage
heapq.heappush(event_queue, Event(time=5.2, ...))
next_event = heapq.heappop(event_queue)
```

### Time Advancement
```python
def _advance_to_next_decision(self):
    while self.event_queue:
        event = heapq.heappop(self.event_queue)
        self.global_clock = event.time  # Jump to event time
        
        if event.event_type == EventType.TRUCK_READY:
            self.active_truck_id = event.truck_id
            return  # Stop and wait for step()
        else:
            self._handle_event(event)  # Process automatically
```

### Navigation Event Scheduling
```python
def _execute_navigation_action(self, truck, action):
    # Calculate travel time
    distance = self.transport_graph.get_distance(...)
    travel_time = distance / truck.base_speed
    
    # Schedule completion event
    completion_time = self.global_clock + travel_time
    heapq.heappush(self.event_queue, Event(
        time=completion_time,
        event_type=EventType.ROUTE_COMPLETE,
        truck_id=truck.truck_id,
        data={'destination': target_node, 'distance': distance, ...}
    ))
```

## Advantages of Event-Driven Approach

1. **Realistic Time Modeling**: Time progresses naturally based on actual durations
2. **Efficiency**: No wasted steps when trucks are traveling/charging
3. **Single-Agent RL**: Can use standard single-agent algorithms (PPO, DQN, etc.)
4. **Clear Semantics**: Each `step()` is a decision point, not an arbitrary timestep
5. **Scalability**: Adding more trucks doesn't change action space dimensionality

## Testing

Run comprehensive tests:
```bash
# All tests
python scripts/test_event_driven_env.py --test all

# Specific tests
python scripts/test_event_driven_env.py --test basic
python scripts/test_event_driven_env.py --test events
python scripts/test_event_driven_env.py --test charging
python scripts/test_event_driven_env.py --test termination

# Visual demo
python scripts/test_event_driven_env.py --demo
```

## Next Steps

1. **Install dependencies**: `pip install gymnasium networkx pyyaml`
2. **Run tests**: `python scripts/test_event_driven_env.py --demo`
3. **Train an agent**: Use single-agent RL algorithm on `EventDrivenTruckEnv`
4. **Compare**: Benchmark against step-based `SimpleTruckEnv`

## Files Created

- `simple_truck_env/event_driven_env.py` - Main environment implementation
- `scripts/test_event_driven_env.py` - Test suite and demos
- `EVENT_DRIVEN_GUIDE.md` - This documentation

## Configuration

Add to `config.yaml`:
```yaml
environment:
  max_time: 48.0  # Maximum simulation time (hours)

rewards:
  time_penalty: -1.0
  distance_penalty: -0.1
  charge_penalty: -2.0
  delivery_bonus: 50.0
```
