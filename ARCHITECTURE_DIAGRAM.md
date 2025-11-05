# Event-Driven Truck Environment Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                 EventDrivenTruckEnv                              │
│                                                                   │
│  ┌────────────────┐                                              │
│  │  config.yaml   │──────┐                                       │
│  └────────────────┘      │                                       │
│                          ▼                                       │
│              ┌───────────────────┐                               │
│              │  Configuration    │                               │
│              │  - num_trucks: 3  │                               │
│              │  - num_stops: 5   │                               │
│              │  - max_time: 48h  │                               │
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
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           🕐 GLOBAL CLOCK: 5.42h                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────┬───────────────┬───────────────┐              │
│  │   Truck 0    │   Truck 1     │   Truck 2     │              │
│  │ Battery: 85% │ Battery: 45%  │ Battery: 92%  │              │
│  │ Location: 45 │ Location: 128 │ Location: 201 │              │
│  │ State: active│ State:charging│ State: routing│              │
│  │ Deliver: 2/5 │ Deliver: 4/5  │ Deliver: 1/5  │              │
│  └──────────────┴───────────────┴───────────────┘              │
│                          │                                       │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────┐          │
│  │       📋 EVENT QUEUE (Priority Heap)              │          │
│  │  t=6.50h: TRUCK_READY (Truck 1)                   │          │
│  │  t=7.12h: ROUTE_COMPLETE (Truck 2)                │          │
│  │  t=8.30h: CHARGE_COMPLETE (Truck 0)               │          │
│  └───────────────────────────────────────────────────┘          │
│                          │                                       │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────┐          │
│  │  🚛 ACTIVE TRUCK: Truck 1 (waiting for action)    │          │
│  └───────────────────────────────────────────────────┘          │
└───────────────────────────────────────────────────────────────┘
```

## Action Space (Discrete - Single Agent)

```
For the ACTIVE truck only:

Discrete(30)  # 25 chargers + 1 delivery + 4 charge durations
         │
         └─── Single action for active truck

Action Breakdown:
  [0-24]  → Navigate to charging station 0-24
  [25]    → Navigate to next delivery stop
  [26]    → Charge for 1 hour (if at charger)
  [27]    → Charge for 2 hours (if at charger)
  [28]    → Charge for 3 hours (if at charger)
  [29]    → Charge for 4 hours (if at charger)

Example Actions:
  action = 25  →  "Go to next delivery"
  action = 0   →  "Go to charging station at node 11"
  action = 27  →  "Charge for 2 hours"
```

## Observation Space (Box)

```
Fixed dimension: 13 (active truck state + global info)

┌─────────────────────────────────────────────────────────┐
│ Active Truck State (indices 0-9):                       │
│  [0] current_node_normalized                            │
│  [1] next_delivery_node_normalized                      │
│  [2] battery_level (kWh)                                │
│  [3] battery_percentage                                 │
│  [4] is_charging                                        │
│  [5] deliveries_remaining                               │
│  [6] nearest_charger_distance                           │
│  [7] can_reach_next_delivery                            │
│  [8] time_elapsed (truck)                               │
│  [9] distance_traveled                                  │
├─────────────────────────────────────────────────────────┤
│ Global State (indices 10-12):                           │
│  [10] global_clock (hours)                              │
│  [11] active_trucks_count                               │
│  [12] events_pending_in_queue                           │
└─────────────────────────────────────────────────────────┘
```

## Step Execution Flow (Event-Driven)

```
1. Receive Discrete Action for Active Truck
   ┌───────────────────────────┐
   │ action = 25 (next delivery)│
   └───────────────────────────┘
                │
                ▼
2. Execute Action:
   ┌──────────────────────────────┐
   │ If Navigation Action:         │
   │  - Calculate distance/time    │
   │  - Check battery feasibility  │
   │  - Schedule ROUTE_COMPLETE    │
   │    event at t + travel_time   │
   │  - Set truck state: "routing" │
   └──────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────┐
   │ If Charging Action:           │
   │  - Check if at charger        │
   │  - Schedule CHARGE_COMPLETE   │
   │    event at t + charge_hours  │
   │  - Set truck state: "charging"│
   └──────────────────────────────┘
                │
                ▼
3. Advance Global Clock:
   ┌──────────────────────────────┐
   │ Pop events from queue until: │
   │  - TRUCK_READY found → STOP  │
   │  - Process other events:     │
   │    * ROUTE_COMPLETE          │
   │    * CHARGE_COMPLETE         │
   │    * TRUCK_TERMINATED        │
   └──────────────────────────────┘
                │
                ▼
4. Check Termination:
   ┌──────────────────────────────┐
   │ All trucks complete?   → Done │
   │ Any truck failed?      → Done │
   │ Clock >= max_time?     → Trun │
   │ Otherwise → active_truck_id   │
   └──────────────────────────────┘
                │
                ▼
5. Return:
   ┌──────────────────────────────┐
   │ (obs, reward, terminated,    │
   │  truncated, info)            │
   │  - obs: active truck state   │
   │  - info['global_clock']      │
   │  - info['active_truck_id']   │
   └──────────────────────────────┘
```

## Reward Structure

```
Per-Step Reward (for active truck):

Navigation Reward:
  - Time penalty: -1.0 × travel_time (hours)
  - Distance penalty: -0.1 × distance (km)
  - Delivery bonus: +50.0 (if delivery completed)
  - Invalid action: -10.0
  - Insufficient battery: -50.0

Charging Reward:
  - Charge penalty: -2.0 × charge_time (hours)
  - Invalid (not at charger): -10.0

Episode Completion:
  - All complete: Natural termination
  - Any failed: Natural termination
  - Time limit: Truncated
```

## Data Flow (Event-Driven)

```
┌─────────────┐
│ Agent       │
│ (RL Policy) │
└─────────────┘
      │
      │ action (single discrete value)
      ▼
┌─────────────────────────────────────┐
│ EventDrivenTruckEnv                 │
│                                     │
│  1. Get active truck                │
│  2. Execute action (nav or charge)  │
│  3. Schedule future event           │
│  4. Advance clock to next READY     │
│  5. Calculate reward                │
│  6. Build observation (active truck)│
└─────────────────────────────────────┘
      │
      │ (obs, reward, terminated, truncated, info)
      │ info includes: global_clock, active_truck_id
      ▼
┌─────────────┐
│ Agent       │
│ (Training)  │
└─────────────┘
```

## Event Types & Flow

```
EventType Enum:
  ┌────────────────────────────────────┐
  │ TRUCK_READY       → Needs decision │
  │ ROUTE_COMPLETE    → Arrived        │
  │ CHARGE_COMPLETE   → Charged        │
  │ TRUCK_TERMINATED  → Done/Failed    │
  └────────────────────────────────────┘

Event Processing:
  t=0.00 → TRUCK_READY(0) → step() called → action=25 (delivery)
  t=0.00 → Schedule ROUTE_COMPLETE(0, t=1.5h)
  t=0.00 → TRUCK_READY(1) → step() called → action=27 (charge 2h)
  t=0.00 → Schedule CHARGE_COMPLETE(1, t=2.0h)
  t=1.50 → ROUTE_COMPLETE(0) → Truck 0 at delivery
  t=1.50 → Schedule TRUCK_READY(0)
  t=1.50 → TRUCK_READY(0) → step() called...
  t=2.00 → CHARGE_COMPLETE(1) → Truck 1 charged
  t=2.00 → Schedule TRUCK_READY(1)
  ...
```

## File Structure

```
EVPR/
├── simple_truck_env/
│   ├── __init__.py              # Package exports
│   ├── config.yaml              # 📝 Configuration file
│   ├── config_utils.py          # Config loading utilities
│   ├── event_driven_env.py      # 🚛 Main environment (EVENT-DRIVEN)
│   ├── truck.py                 # Truck state/logic
│   ├── transportation_graph.py  # Graph utilities
│   └── README.md                # Documentation
│
├── scripts/
│   ├── test_event_driven_env.py # ✅ Tests
│   └── visualize_simple_env.py  # � Visualization
│
├── EVENT_DRIVEN_GUIDE.md        # Architecture guide
├── RESEARCH_OVERVIEW.md         # Research problem
└── truck_env/
    ├── data/                    # Graph data files
    └── utils.py                 # Shared utilities
```

## Configuration Hierarchy

```
config.yaml
├── environment          # Basic settings
│   ├── num_stops
│   ├── max_time        ← KEY: Time limit (hours)
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
    ├── distance_penalty  ← NEW
    ├── charge_penalty    ← NEW
    └── delivery_bonus
```

## Typical Episode Timeline (Event-Driven)

```
t=0.00h: Reset
  ┌────────────────────────────────┐
  │ Truck 0: Start @ node 45       │
  │ Truck 1: Start @ node 128      │
  │ Truck 2: Start @ node 201      │
  │ → All trucks get READY events  │
  └────────────────────────────────┘

t=0.00h: Truck 0 Decision
  ┌────────────────────────────────┐
  │ Active: Truck 0                │
  │ Action: 25 (go to delivery)    │
  │ → Schedule ROUTE_COMPLETE @1.2h│
  └────────────────────────────────┘

t=0.00h: Truck 1 Decision  
  ┌────────────────────────────────┐
  │ Active: Truck 1                │
  │ Action: 0 (go to charger)      │
  │ → Schedule ROUTE_COMPLETE @0.8h│
  └────────────────────────────────┘

t=0.80h: Truck 1 Arrives at Charger
  ┌────────────────────────────────┐
  │ Event: ROUTE_COMPLETE (Truck 1)│
  │ → Truck 1 now at charger       │
  │ → Schedule TRUCK_READY         │
  └────────────────────────────────┘

t=0.80h: Truck 1 Decision
  ┌────────────────────────────────┐
  │ Active: Truck 1                │
  │ Action: 27 (charge 2 hours)    │
  │ → Schedule CHARGE_COMPLETE @2.8h│
  └────────────────────────────────┘

t=1.20h: Truck 0 Completes Delivery
  ┌────────────────────────────────┐
  │ Event: ROUTE_COMPLETE (Truck 0)│
  │ → Delivery 1/5 complete        │
  │ → Schedule TRUCK_READY         │
  └────────────────────────────────┘

t=1.20h: Truck 0 Decision
  ┌────────────────────────────────┐
  │ Active: Truck 0                │
  │ Action: 25 (next delivery)     │
  │ → Schedule ROUTE_COMPLETE @2.5h│
  └────────────────────────────────┘

t=2.50h: Truck 0 Completes Another Delivery
  ...continuing until all trucks complete...

t=15.30h: All Complete
  ┌────────────────────────────────┐
  │ Truck 0: ✅ Complete (5/5)     │
  │ Truck 1: ✅ Complete (5/5)     │
  │ Truck 2: ✅ Complete (5/5)     │
  │ → Episode terminates           │
  │ → Final time: 15.30 hours      │
  └────────────────────────────────┘
```
