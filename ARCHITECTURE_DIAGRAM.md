# SimpleTruckEnv Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     SimpleTruckEnv                               │
│                                                                   │
│  ┌────────────────┐                                              │
│  │  config.yaml   │──────┐                                       │
│  └────────────────┘      │                                       │
│                          ▼                                       │
│              ┌───────────────────┐                               │
│              │  Configuration    │                               │
│              │  - num_trucks: 3  │                               │
│              │  - num_stops: 5   │                               │
│              │  - rewards        │                               │
│              └───────────────────┘                               │
│                          │                                       │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────┐          │
│  │         TransportationGraph                        │          │
│  │  - Road network (NetworkX)                         │          │
│  │  - Charging stations (25 nodes)                    │          │
│  │  - Distance calculations                           │          │
│  │  - Route generation                                │          │
│  └───────────────────────────────────────────────────┘          │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────┬───────────────┬───────────────┐              │
│  │   Truck 0    │   Truck 1     │   Truck 2     │              │
│  │ Battery: 85% │ Battery: 45%  │ Battery: 92%  │              │
│  │ Location: 45 │ Location: 128 │ Location: 201 │              │
│  │ Deliver: 2/5 │ Deliver: 4/5  │ Deliver: 1/5  │              │
│  └──────────────┴───────────────┴───────────────┘              │
│                          │                                       │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────┐          │
│  │       Charger Occupancy Tracking                  │          │
│  │  Station 11:  [Truck 0, Truck 1]   (2/3 slots)    │          │
│  │  Station 58:  []                   (0/2 slots)    │          │
│  │  Station 106: [Truck 2]            (1/4 slots)    │          │
│  └───────────────────────────────────────────────────┘          │
└───────────────────────────────────────────────────────────────┘
```

## Action Space (MultiDiscrete)

```
For num_trucks = 3:

MultiDiscrete([26, 5, 26, 5, 26, 5])
               ┬   ┬  ┬   ┬  ┬   ┬
               │   │  │   │  │   └─── Truck 2: Charge Action [0-4]
               │   │  │   │  └─────── Truck 2: Nav Action [0-25]
               │   │  │   └────────── Truck 1: Charge Action [0-4]
               │   │  └────────────── Truck 1: Nav Action [0-25]
               │   └─────────────────  Truck 0: Charge Action [0-4]
               └─────────────────────  Truck 0: Nav Action [0-25]

Action Example: [25, 0, 0, 2, 1, 3]
                 │   │  │  │  │  │
                 │   │  │  │  │  └─ Truck 2: charge 3h
                 │   │  │  │  └──── Truck 2: go to charger 1
                 │   │  │  └─────── Truck 1: charge 2h  
                 │   │  └────────── Truck 1: go to charger 0
                 │   └───────────── Truck 0: no charge
                 └───────────────── Truck 0: go to next delivery
```

## Observation Space (Box)

```
For num_trucks = 3, obs_dim = 30 (10 per truck):

┌─────────────────────────────────────────────────────────┐
│ Truck 0 (indices 0-9):                                  │
│  [0] current_node_normalized                            │
│  [1] next_delivery_node_normalized                      │
│  [2] battery_level (kWh)                                │
│  [3] battery_percentage                                 │
│  [4] is_charging                                        │
│  [5] deliveries_remaining                               │
│  [6] nearest_charger_distance                           │
│  [7] can_reach_next_delivery                            │
│  [8] time_elapsed                                       │
│  [9] distance_traveled                                  │
├─────────────────────────────────────────────────────────┤
│ Truck 1 (indices 10-19):                                │
│  [same structure]                                        │
├─────────────────────────────────────────────────────────┤
│ Truck 2 (indices 20-29):                                │
│  [same structure]                                        │
└─────────────────────────────────────────────────────────┘
```

## Step Execution Flow

```
1. Receive MultiDiscrete Action
   ┌───────────────────────────┐
   │ [nav_0, ch_0, nav_1, ch_1]│
   └───────────────────────────┘
                │
                ▼
2. For Each Truck:
   ┌──────────────────────────────┐
   │ Execute Navigation Action     │
   │  - Calculate distance         │
   │  - Check battery feasibility  │
   │  - Move truck                 │
   │  - Update battery             │
   │  - Calculate reward           │
   └──────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────┐
   │ Execute Charging Action       │
   │  - Check if at charger        │
   │  - Simulate queue/waiting     │
   │  - Charge battery             │
   │  - Calculate reward           │
   └──────────────────────────────┘
                │
                ▼
3. Check Termination:
   ┌──────────────────────────────┐
   │ All trucks complete?   → Done │
   │ Any truck failed?      → Done │
   │ Max steps reached?     → Trun │
   └──────────────────────────────┘
                │
                ▼
4. Return:
   ┌──────────────────────────────┐
   │ (obs, reward, done, trunc,   │
   │  info)                        │
   └──────────────────────────────┘
```

## Reward Structure

```
Per-Step Reward = Σ (truck rewards)

For each truck:
  Navigation Reward:
    - Base: time_penalty × travel_time
    - Delivery bonus: +50 (if delivery completed)
    - Invalid action: -10
    - Insufficient battery: -50
  
  Charging Reward:
    - Base: time_penalty × (charge_time + wait_time)
    - Invalid (not at charger): -10

Episode Completion:
  - All complete: +1000
  - Any failed: -500
```

## Data Flow

```
┌─────────────┐
│ Agent       │
│ (RL Policy) │
└─────────────┘
      │
      │ action [nav_0, ch_0, nav_1, ch_1, ...]
      ▼
┌─────────────────────────────────────┐
│ SimpleTruckEnv                      │
│                                     │
│  1. Parse actions per truck         │
│  2. Execute navigation              │
│  3. Execute charging                │
│  4. Update truck states             │
│  5. Calculate rewards               │
│  6. Build observation               │
└─────────────────────────────────────┘
      │
      │ (obs, reward, done, info)
      ▼
┌─────────────┐
│ Agent       │
│ (Training)  │
└─────────────┘
```

## File Structure

```
EVPR/
├── simple_truck_env/
│   ├── __init__.py              # Package exports
│   ├── config.yaml              # 📝 Configuration file
│   ├── config_utils.py          # Config loading utilities
│   ├── simple_truck_env.py      # 🚛 Main environment
│   ├── truck.py                 # Truck state/logic
│   ├── transportation_graph.py  # Graph utilities
│   └── README.md                # Documentation
│
├── scripts/
│   ├── test_multitruck_env.py   # ✅ Tests
│   ├── example_multitruck.py    # 📖 Examples
│   └── demo_config.py           # Config demos
│
└── truck_env/
    ├── data/                    # Graph data files
    └── utils.py                 # Shared utilities
```

## Configuration Hierarchy

```
config.yaml
├── environment          # Basic settings
│   ├── num_stops
│   ├── max_steps
│   └── verbose
│
├── advanced            # Multi-truck settings
│   ├── num_trucks     ← KEY: Number of trucks
│   └── action_masking
│
├── truck              # Truck specifications
│   ├── type_selection
│   ├── initial_battery
│   ├── standard {...}
│   └── heavy {...}
│
├── charging           # Charging config
│   ├── charge_rate
│   ├── charge_durations
│   └── efficiency
│
└── rewards            # Reward function
    ├── time_penalty
    ├── delivery_bonus
    ├── completion_bonus
    └── failure_penalty
```

## Typical Episode Timeline

```
Step 0: Reset
  ┌────────────────────────────────┐
  │ Truck 0: Start @ node 45       │
  │ Truck 1: Start @ node 128      │
  │ Truck 2: Start @ node 201      │
  └────────────────────────────────┘

Step 1-5: Initial Deliveries
  ┌────────────────────────────────┐
  │ Truck 0: 45 → 67 (delivery 1)  │
  │ Truck 1: 128 → 89 (delivery 1) │
  │ Truck 2: 201 → 156 (delivery 1)│
  └────────────────────────────────┘

Step 6-10: Mixed Actions
  ┌────────────────────────────────┐
  │ Truck 0: 67 → 92 (delivery 2)  │
  │ Truck 1: 89 → Charger @ 11     │
  │   └─ Charging 2h (battery low) │
  │ Truck 2: 156 → 178 (delivery 2)│
  └────────────────────────────────┘

Step 15: Completion
  ┌────────────────────────────────┐
  │ Truck 0: ✅ Complete (5/5)     │
  │ Truck 1: 🚛 Active (3/5)       │
  │ Truck 2: ✅ Complete (5/5)     │
  └────────────────────────────────┘

Step 25: All Complete
  ┌────────────────────────────────┐
  │ Truck 0: ✅ Complete           │
  │ Truck 1: ✅ Complete           │
  │ Truck 2: ✅ Complete           │
  │ → Episode ends, +1000 bonus    │
  └────────────────────────────────┘
```
