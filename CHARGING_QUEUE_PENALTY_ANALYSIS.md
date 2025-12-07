# Charging Queue Penalty Analysis

## Overview
The charging queue penalty is applied when a truck waits at a charger for a free port. This document traces the exact flow.

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ TRUCK ARRIVES AT CHARGER (TRUCK_ROUTING event)                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                  ┌─────────────────────┐
                  │ Check port available│
                  │ (charger_gating)    │
                  └─────────────────────┘
                    ↙               ↘
        ┌──────────────────┐    ┌──────────────────┐
        │ Port available   │    │ No free port     │
        └──────────────────┘    └──────────────────┘
              ↓                          ↓
     ┌──────────────────┐      ┌──────────────────────────────┐
     │ TRUCK_READY      │      │ truck_state = "waiting..."   │
     │ scheduled        │      │ waiting_start_times[id] =    │
     │ immediately      │      │   global_clock               │
     └──────────────────┘      │ truck.start_waiting()        │
                               └──────────────────────────────┘
                                        ↓
                               ┌──────────────────────────┐
                               │ Wait for port to free    │
                               │ (wake_waiting_trucks)    │
                               └──────────────────────────┘
                                        ↓
                               ┌──────────────────────────┐
                               │ TRUCK_READY event fired  │
                               │ at time port becomes free│
                               └──────────────────────────┘
```

---

## Step-by-Step Execution

### Step 1: Truck Arrives at Charger (TRUCK_ROUTING Event)
**File:** `event_driven_env.py`, lines 573-640

```python
# When truck arrives at charger node (destination in self.charging_nodes):
can_proceed, next_check_time = self.charging_station.check_charger_gating(
    truck_id=truck.truck_id,
    charger_node=destination,
    global_clock=self.global_clock,
)

if not can_proceed:
    # No free port available
    self.truck_states[truck.truck_id] = "waiting_to_charge"
    
    if truck.truck_id not in self.waiting_start_times:
        self.waiting_start_times[truck.truck_id] = self.global_clock  # ← START TIMER
        truck.start_waiting(timestamp=self.global_clock, reason="charger_queue")
```

**What happens:**
- `waiting_start_times[truck_id]` is set to the current global clock time
- Truck enters "waiting_to_charge" state
- Truck waits in the charging station queue
- No TRUCK_READY event is scheduled (waits for port to free)

---

### Step 2: Truck is Woken When Port Becomes Available
**File:** `models/simulation/charging_station.py` (wake_waiting_trucks method)

```python
# When another truck finishes charging and frees a port:
def wake_waiting_trucks(self, charger_node, global_clock, event_queue, ...):
    if charger_node in self.charger_queues:
        while len(self.charger_queues[charger_node]) > 0:
            # Remove next truck from queue
            truck_id = self.charger_queues[charger_node].pop(0)
            
            # Schedule TRUCK_READY event for this truck
            heapq.heappush(
                event_queue,
                Event(
                    time=global_clock,  # Wake up immediately
                    event_type=EventType.TRUCK_READY,
                    truck_id=truck_id,
                    data={"reason": "port_freed_early"}  # ← Reason for waking
                )
            )
```

**What happens:**
- When a port becomes available, waiting trucks are woken
- TRUCK_READY event is scheduled with `reason="port_freed_early"`

---

### Step 3: TRUCK_READY Event Processed
**File:** `event_driven_env.py`, lines 460-542

When the TRUCK_READY event fires:

```python
if event.event_type == EventType.TRUCK_READY:
    truck = self.trucks[event.truck_id]
    reason = event.data.get("reason", "")
    
    # ... (skip some checks)
    
    # ╔════════════════════════════════════════════════════════════╗
    # ║ CRITICAL: Calculate waiting penalty if truck was waiting    ║
    # ╚════════════════════════════════════════════════════════════╝
    if truck.truck_id in self.waiting_start_times:
        waiting_duration = self.global_clock - self.waiting_start_times[truck.truck_id]
        #                  ↑ Wake-up time   ↑ Start waiting time
        
        if waiting_duration > 0:
            # Calculate time penalty for waiting
            waiting_penalty = -waiting_duration * self.reward_config["time_multiplier"]
            
            # ← BUFFER THE PENALTY (don't apply immediately)
            self.waiting_penalty_buffer = waiting_penalty
            
            # Update truck's waiting time stat
            truck.add_waiting_time(waiting_duration, timestamp=self.global_clock)
            
            if self.verbose:
                print(f"  Truck {truck.truck_id} finished waiting at {self.global_clock:.2f}h")
                print(f"    Waited: {waiting_duration:.2f}h")
                print(f"    Waiting penalty (to be applied on next action): {waiting_penalty:.2f}")
        
        # Clear the waiting start time
        del self.waiting_start_times[truck.truck_id]
    
    # Mark truck as ready
    truck.mark_ready(timestamp=self.global_clock, reason=reason)
    self.truck_states[truck.truck_id] = "ready"
    self.active_truck_id = truck.truck_id
    return  # ← TRUCK IS NOW READY FOR NEXT ACTION
```

**What happens:**
- Waiting duration is calculated: `current_time - waiting_start_time`
- Penalty is calculated: `-waiting_duration * time_multiplier`
- **PENALTY IS BUFFERED** in `self.waiting_penalty_buffer`
- Truck becomes "ready" for the next action

---

### Step 4: Penalty Applied on Next Action
**File:** `event_driven_env.py`, lines 708-712

When `step()` is called for the next action:

```python
def step(self, action):
    # ... (get active truck)
    
    # ╔════════════════════════════════════════════════════════════╗
    # ║ APPLY BUFFERED WAITING PENALTY FROM PREVIOUS WAIT          ║
    # ╚════════════════════════════════════════════════════════════╝
    if self.waiting_penalty_buffer != 0.0:
        reward += self.waiting_penalty_buffer  # ← ADD TO REWARD
        
        if self.verbose:
            print(f"  Adding waiting penalty from queue: {self.waiting_penalty_buffer:.2f}")
        
        self.waiting_penalty_buffer = 0.0  # ← CLEAR BUFFER
    
    # Execute the action (navigation or charging)
    # reward += self._execute_navigation_action(...)  or
    # reward += self._execute_charge_action(...)
    
    # Return step results
    return obs, reward, terminated, truncated, info
```

**What happens:**
- The buffered penalty from the previous wait is added to the current action's reward
- The buffer is cleared after use

---

## Summary of Charging Queue Penalty

### When is the penalty applied?
- **Calculation time:** When truck is woken up from waiting (TRUCK_READY event)
- **Application time:** On the NEXT action the truck takes (buffered)

### What is the penalty amount?
```
waiting_penalty = -waiting_duration * time_multiplier

Where:
  waiting_duration = time_woken - time_started_waiting
  time_multiplier = config["rewards"]["time_multiplier"] (default: 1.0)
```

### Example Timeline:
```
Time 0.0h: Truck arrives at charger
           waiting_start_times[truck_1] = 0.0
           
Time 2.5h: Port becomes available
           Port freed by another truck
           waiting_duration = 2.5 - 0.0 = 2.5 hours
           waiting_penalty = -2.5 * 1.0 = -2.5
           waiting_penalty_buffer = -2.5
           Truck becomes "ready"
           
Time 2.5h: Agent takes next action (e.g., charge)
           reward = -2.5 (waiting penalty) + -charge_time + other_rewards
           waiting_penalty_buffer = 0.0 (cleared)
```

---

## Key Observations

### ✅ Correct Behavior:
1. **Exact duration tracking:** Uses `global_clock - waiting_start_time`
2. **Buffering mechanism:** Penalty waits until truck takes next action
3. **One-time application:** Penalty is cleared after being applied once
4. **Consistent with unloading:** Same pattern could be used for unloading (but currently isn't for exact time)

### ⚠️ Issues Compared to Unloading:

| Aspect | Waiting Penalty | Previous Unloading |
|--------|-----------------|-------------------|
| **When calculated** | When truck woken | When truck navigates |
| **Using actual time** | Yes (exact duration) | No (base time approximation) |
| **Timing of penalty** | Buffered to next action | Immediate with delivery |
| **Applied multiple times** | No (buffer cleared) | Only once per delivery |

---

## Unloading Time Bug Fix Context

The current unloading implementation calls `apply_unloading_time()` to get the exact time when calculating the delivery reward:

```python
# In _execute_navigation_action() for delivery navigation:
if is_delivery_nav:
    delivery_bonus = self.reward_config["delivery_bonus"]
    # Get exact unloading time that WILL occur
    actual_unloading_time = self.delivery_simulator.apply_unloading_time(
        delivery_node=target_node,
        current_time=completion_time
    )
    unloading_penalty = -actual_unloading_time * self.reward_config["time_multiplier"]
    return time_penalty + delivery_bonus + unloading_penalty
```

**Advantage over the old waiting buffer pattern:**
- The exact unloading time is pre-computed and included in the immediate reward
- No need for buffering since we know the exact time upfront
- Agent gets full visibility into the cost at decision time

---

## Configuration

**Waiting time multiplier:** From `config["rewards"]["time_multiplier"]`
- Default: 1.0 (1 hour of waiting = -1.0 reward)
- Affects both charging queue waiting and (now) unloading penalties

