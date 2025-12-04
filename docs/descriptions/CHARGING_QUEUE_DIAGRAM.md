# Charging Queue System - Flow Diagram

## Overview
The charging queue system manages truck access to charging stations using FCFS (First-Come-First-Served) with multiple charging ports per station.

---

## System Components

### 1. **Data Structures**
```
charger_occupancy[node] → List[truck_id]  # Trucks currently charging
charger_waitlist[node]  → List[{truck_id, planned_plug_time}]  # Trucks waiting
truck_charge_end_time   → Dict[truck_id → time]  # When charging completes
```

### 2. **Key Functions**
- `check_charger_gating()` - Decides if truck can charge or must wait
- `start_charging()` - Begins charging session
- `finish_charging()` - Completes charging session
- `wake_waiting_trucks()` - Notifies waiting trucks when port opens

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRUCK ARRIVES AT CHARGER                      │
│                     (TRUCK_ROUTING event completes)                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  TRUCK_READY event     │
                    │  (truck ready to act)  │
                    └────────┬───────────────┘
                             │
                             ▼
              ╔══════════════════════════════╗
              ║  check_charger_gating()      ║
              ║  Check if truck can proceed  ║
              ╚══════════════╤═══════════════╝
                             │
                    ┌────────┴────────┐
                    │                 │
         ┌──────────▼─────────┐      │
         │  Truck in waitlist? │      │
         └──────────┬──────────┘      │
                    │                 │
              ┌─────┴─────┐          │
              │           │          │
           YES│           │NO        │
              │           │          │
              ▼           ▼          │
    ┌─────────────┐   ┌──────────────────────┐
    │  SCENARIO A │   │    NEW ARRIVAL       │
    │  (In Queue) │   │                      │
    └──────┬──────┘   └──────┬───────────────┘
           │                 │
           │          ┌──────┴──────┐
           │          │             │
           │     ┌────▼────┐   ┌────▼────┐
           │     │Scenario 1│   │Scenario 2│
           │     │Free Slots│   │No Slots │
           │     │No Wait   │   │Single/  │
           │     │          │   │Multi    │
           │     └────┬─────┘   └────┬────┘
           │          │              │
           └──────────┴──────────────┴─────────────┐
                                                    │
                                                    ▼
                                    ╔═══════════════════════════╗
                                    ║  DECISION POINT           ║
                                    ║  Can truck charge now?    ║
                                    ╚═════════╤═════════════════╝
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                           ┌────────▼────────┐   ┌──────▼────────┐
                           │   CAN PROCEED   │   │  CANNOT       │
                           │   (True, None)  │   │  PROCEED      │
                           └────────┬────────┘   │  (False, time)│
                                    │            └───────┬───────┘
                                    │                    │
                    ┌───────────────▼──────────┐         │
                    │  Agent Selects Action    │         │
                    │  (navigate or charge)    │         │
                    └───────────┬──────────────┘         │
                                │                        │
                    ┌───────────┴───────────┐            │
                    │                       │            │
         ┌──────────▼──────────┐   ┌────────▼─────────┐ │
         │ Navigate Elsewhere  │   │  Charge Action   │ │
         │ (leave charger)     │   │                  │ │
         └──────────┬──────────┘   └────────┬─────────┘ │
                    │                       │            │
                    │                       ▼            │
                    │          ╔════════════════════╗    │
                    │          ║ start_charging()   ║    │
                    │          ║ - Add to occupancy ║    │
                    │          ║ - Remove from wait ║    │
                    │          ║ - Schedule end     ║    │
                    │          ╚═════════╤══════════╝    │
                    │                    │               │
                    │                    ▼               │
                    │          ┌──────────────────┐      │
                    │          │  TRUCK_READY     │      │
                    │          │  (charge_complete│      │
                    │          │   after N hours) │      │
                    │          └─────────┬────────┘      │
                    │                    │               │
                    │                    ▼               │
                    │          ╔════════════════════╗    │
                    │          ║ finish_charging()  ║    │
                    │          ║ - Remove occupancy ║    │
                    │          ║ - Update stats     ║    │
                    │          ╚═════════╤══════════╝    │
                    │                    │               │
                    │                    ▼               │
                    │          ╔════════════════════╗    │
                    │          ║ wake_waiting_      ║    │
                    │          ║ trucks()           ║    │
                    │          ║ - Wake up to K     ║    │
                    │          ║   trucks (K=free)  ║    │
                    │          ╚═════════╤══════════╝    │
                    │                    │               │
                    └────────────────────┴───────────────┘
                                         │
                             ┌───────────┴────────────┐
                             │                        │
                    ┌────────▼────────┐      ┌────────▼────────┐
                    │ Waiting trucks   │      │ Truck continues │
                    │ get TRUCK_READY  │      │ to next delivery│
                    │ events (t=now)   │      │                 │
                    └──────────────────┘      └─────────────────┘
                             │
                             │
                    ┌────────▼────────────┐
                    │ Enter waiting_to_   │
                    │ charge state        │
                    │ - Track start time  │
                    │ - Wait for recheck  │
                    │   or wake event     │
                    └─────────────────────┘
```

---

## Detailed Scenarios

### **Scenario 1: Free Slots, No Waitlist**
```
Charger Status:
├─ Capacity: N ports
├─ Occupancy: M trucks (M < N)
├─ Waitlist: Empty
└─ Free Slots: N - M > 0

Action:
├─ Truck uses waiting_time_lookup to predict wait
├─ If wait_time > 0:
│  ├─ Add to waitlist with planned_plug_time
│  └─ Schedule TRUCK_READY at predicted time
└─ Else:
   ├─ Add to waitlist with planned_plug_time = now
   └─ Return (True, None) → Can proceed immediately

Issue: Currently applies lookup table even when slots available!
```

### **Scenario 2: Single-Port Charger (Occupied)**
```
Charger Status:
├─ Capacity: 1 port
├─ Occupancy: 1 truck
├─ Waitlist: 0+ trucks
└─ Free Slots: 0

Action:
├─ Add truck to waitlist (planned_plug_time = None)
├─ NO self-scheduled event
└─ Wait for wake_waiting_trucks() when port frees
```

### **Scenario 3: Multi-Port Charger (All Busy/Has Waitlist)**
```
Charger Status:
├─ Capacity: N ports (N > 1)
├─ Occupancy: M trucks (M > 0)
├─ Waitlist: W trucks (W > 0)
└─ Free Slots: Variable

Action:
├─ Calculate utilization: M / N
├─ Sample wait_time from lookup table
├─ plug_time = now + max(wait_time, 0.1)
├─ Add to waitlist with planned_plug_time
└─ Schedule TRUCK_READY at plug_time
```

### **Scenario A: Already in Waitlist**
```
Check Position:
├─ Index: i in waitlist
├─ Free Slots: K
└─ planned_plug_time: T (or None)

Decision:
├─ If (K > 0) AND (i < K) AND (T ≤ now or T = None):
│  └─ Return (True, None) → Can charge now
└─ Else:
   ├─ If T exists and T > now:
   │  └─ Return (False, T) → Wait until T
   └─ Else:
      └─ Return (False, None) → Wait for wake
```

---

## Event Timeline Example

### Example: 2 Trucks, 1 Charger (Capacity=1)

```
Time  Event                        Occupancy  Waitlist         Action
────────────────────────────────────────────────────────────────────────
0.0   Truck 0: TRUCK_READY         []         []               Navigate to charger
0.5   Truck 0: TRUCK_ROUTING       []         []               Arrive at charger
      └─> check_gating()           []         [0]              Can't proceed (lookup: wait 30min)
0.5   Truck 0: TRUCK_READY         []         [0]              scheduled at t=1.0
1.0   Truck 0: TRUCK_READY         []         [0]              Can proceed now
      └─> Agent: Charge 2h         []         [0]              Select charge action
1.0   Truck 0: start_charging()    [0]        []               Charging begins
      └─> Schedule end at t=3.0    [0]        []               

1.5   Truck 1: TRUCK_READY         [0]        []               Navigate to charger
2.0   Truck 1: TRUCK_ROUTING       [0]        []               Arrive at charger
      └─> check_gating()           [0]        [1]              Can't proceed (single-port busy)
2.0   Truck 1: waiting_to_charge   [0]        [1]              No scheduled event (wait for wake)

3.0   Truck 0: TRUCK_READY         [0]        [1]              Charge complete
      └─> finish_charging()        []         [1]              Port freed
      └─> wake_waiting_trucks()    []         [1]              Wake Truck 1
3.0   Truck 1: TRUCK_READY         []         [1]              Woken up (port_freed_early)
      └─> check_gating()           []         [1]              Can proceed now
3.0   Truck 1: start_charging()    [1]        []               Charging begins
```

---

## Waiting Penalty Tracking

```
┌─────────────────────────────────────────────────────────────┐
│ When truck enters waiting_to_charge state:                  │
│   waiting_start_times[truck_id] = global_clock              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ When truck transitions out of waiting_to_charge:            │
│   waiting_duration = global_clock - waiting_start_times[id] │
│   waiting_penalty = -duration * time_multiplier             │
│   waiting_penalty_buffer = waiting_penalty                  │
│   truck.add_waiting_time(duration)                          │
│   del waiting_start_times[truck_id]                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ On next step() call:                                        │
│   reward += waiting_penalty_buffer                          │
│   waiting_penalty_buffer = 0.0                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Decision Points

### **When is `check_charger_gating()` called?**
1. **On TRUCK_READY event** - Before truck can take action
   - After arrival at charger
   - After waiting period expires (recheck)
   - After being woken by `wake_waiting_trucks()`
   
2. **Triggers waiting state if:**
   - No free ports available
   - Truck not at front of queue
   - Predicted wait time > 0 (Scenario 1)

### **When is `start_charging()` called?**
- Only when agent selects a "charge" action
- After `check_charger_gating()` returns `(True, None)`
- Adds truck to occupancy, removes from waitlist

### **When is `finish_charging()` called?**
- When TRUCK_READY event with reason="charge_complete" fires
- Scheduled during `start_charging()` based on charge duration
- Frees port, calls `wake_waiting_trucks()`

### **When is `wake_waiting_trucks()` called?**
1. After `finish_charging()` - Port became available
2. After truck navigates away from charger - Port freed without charging
3. Wakes up to K trucks (K = number of free slots)

---

## Current Issues

### **Issue 1: Scenario 1 Wait Time**
**Problem:** When a truck arrives at a charger with free slots and no waitlist, it still waits based on the lookup table prediction.

**Location:** `check_charger_gating()` - Scenario 1
```python
if free_slots > 0 and len(waitlist) == 0:
    util = occupancy / float(capacity)
    wait_h = self.get_waiting_time(charger_node, util)
    if wait_h > 0:  # ← Truck waits even with free slot!
        return False, plug_time
```

**Expected:** Truck should proceed immediately when port is available.

---

## State Transitions

```
ready ──────────────┐
                    │ Navigate to charger
                    ▼
              TRUCK_ROUTING
                    │ Arrive
                    ▼
         check_charger_gating()
                    │
        ┌───────────┴───────────┐
        │                       │
     CAN'T                    CAN
     PROCEED                PROCEED
        │                       │
        ▼                       ▼
  waiting_to_charge          ready
  (track start time)            │
        │                       │ Agent: charge action
        │                       ▼
        │                  start_charging()
        │                       │
        │                       ▼
        │                    charging
        │                       │
        │                       │ After N hours
        │                       ▼
        │                  finish_charging()
        │                       │
        │                       ▼
        └──> wake_waiting_trucks()
                    │
                    ▼
              TRUCK_READY
              (woken up)
                    │
                    └──> Loop back to check_gating()
```

---

## Lookup Table Usage

The `waiting_time_lookup` table contains empirical wait times based on:
- **Charger Type:** Level2 or DCFast
- **Capacity:** Number of ports (1, 2, 3, etc.)
- **Utilization:** Current occupancy ratio (0.05 to 0.95)

**Returns:** Average wait time in minutes (converted to hours)

**Used in:**
- Scenario 1: Predicting initial wait when arriving
- Scenario 3: Estimating wait time at busy multi-port chargers

**Not used in:**
- Scenario 2: Single-port (wait for wake only)
- Scenario A: Already in queue (uses planned time)
