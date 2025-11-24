# Event-Driven Truck Routing MDP

This document describes the Markov Decision Process that underpins the `EventDrivenTruckEnv`, the environment used throughout the EVPR project. The agent is a single controller that is sequentially handed whichever truck currently requires a routing/charging decision, while the event queue advances the global simulation clock between decisions.

---

## State Space (𝕊)

### Environment State (latent, Markovian)
At any decision point the environment is fully described by:
- **Global clock and event queue** – the continuous time stamp and priority queue of `TRUCK_READY`/`TRUCK_ROUTING` events (heap stored in `self.event_queue`). Whichever truck produced the earliest `TRUCK_READY` event becomes the controllable agent (`self.active_truck_id`).
- **Per-truck physical state** – for each truck: current node, delivery sequence & pointer, battery level/percentage, cumulative time and distance, routing target & arrival time, charging timers, and terminal flags (`is_complete`, `failed`). See `truck_env/models/truck.py` for the tracked attributes.
- **Per-truck logical state** – categorical label in `self.truck_states` (`"ready"`, `"routing"`, `"waiting_to_charge"`, `"charging"`, `"complete"`, `"failed"`), plus auxiliary timers such as `waiting_start_times` for trucks queued at a charger.
- **Charging station manager** – occupancy lists, waitlists, FCFS gating queues, charger capacities/types, and recorded utilization maintained inside `ChargingStation` (`charger_occupancy`, `charger_waitlist`, `charger_queue`, `truck_charge_end_time`, etc.).
- **Transportation network** – shortest-path energy/time matrices, charging node IDs/capacities/types loaded via `TransportationGraph` from the files referenced in `config.yaml`.

Together, these variables ensure the process is Markovian: once an action is chosen, the subsequent event scheduling and truck/charger state updates are determined (up to explicit stochastic effects described under the transition kernel).

### Agent Observations
Two observation formats are used:
1. **Vector observation (`StateSpace.get_state`)** – a 13-dimensional vector exposing the active truck’s normalized node IDs, battery level & %, charging flag, deliveries remaining, nearest charger distance, reachability flag for the next delivery, truck-level time/distance traveled, global clock, number of unfinished trucks, and pending-event count (`truck_env/state/state_space.py`).
2. **Graph observation (`GNNStateSpace`)** – a heterogenous PyTorch Geometric graph with truck/delivery/charger node sets, edge features (`[energy, time]`), feasibility masks, and metadata required by the GNN policy (`truck_env/state/gnn_state_space.py`). Only active/feasible entities appear, which keeps the observation Markov while shrinking the graph.

The simulator maintains the full environment state internally, while policies consume either observation.

---

## Action Space (𝔸)

Two equivalent parameterizations target the same physical decisions:

1. **Legacy discrete action** – `spaces.Discrete(num_chargers + 1 + num_charge_actions)` (`truck_env/models/event_driven_env.py`, action-space initialization):
   - Indices `0…num_charging_nodes-1`: navigate to a specific charger.
   - Index `num_charging_nodes`: navigate to the next outstanding delivery for the active truck.
   - Remaining indices: stay at the current charger and charge for one of the predefined durations (`charging.charge_durations` in `config.yaml`).

2. **Tuple action (GNN policies)** – `(node_id, charge_hours, is_charging)`:
   - `is_charging=False`: route toward `node_id` (charger or delivery).
   - `is_charging=True`: remain at the current charger and request `charge_hours` of plug-in time. The environment clamps the actual duration based on charger power, efficiency, and remaining capacity.

Regardless of encoding, the environment enforces feasibility checks before scheduling the action:
- Routes require enough battery and a valid path (`transport_graph.get_path_energy/time`, with optional guard `check_navigation_feasibility` to ensure another feasible action exists upon arrival).
- Charging actions require the truck to already occupy a charger; otherwise the agent is redirected toward the next delivery.
- Charger gating is FCFS: `ChargingStation.check_charger_gating` may delay an action until a plug becomes free, pushing a future `TRUCK_READY` event into the queue.

---

## Reward Function (ℝ)

Let `Δt` denote elapsed hours for the action (travel time with traffic or charge duration). The reward components are:

| Event | Reward contribution |
| --- | --- |
| Navigation (delivery or relocation) | `r = -Δt * time_multiplier`, where `time_multiplier` is `rewards.time_multiplier` (default `1.0`). |
| Successful delivery | Additional `+delivery_bonus` (`rewards.delivery_bonus`, default `+100`). Applied whenever the navigation target equals the truck’s next delivery node. |
| Charging | `r = -actual_charge_hours` (time penalty only; configurable charge penalty hook left commented). |
| Waiting for a charger | When a truck transitions out of `"waiting_to_charge"`, the time spent in the queue incurs `-waiting_duration * time_multiplier`. The penalty is buffered and added to the reward of the next decision step. |
| Terminal failure | Immediate `failure_penalty` (`rewards.failure_penalty`, default `-1000`) when a truck runs out of battery, attempts an infeasible route, or cannot guarantee a feasible continuation after a delivery. |
| Invalid micro-actions | Small `-0.01` penalties for attempting useless actions (e.g., charging without a plug, navigating to the current location). |

Episode return is the sum of these step rewards until all trucks are complete/failed or the `max_time` horizon is reached.

---

## Transition Kernel (𝒫)

Transitions are event-driven and mostly deterministic given the environment state and chosen action:

1. **Routing actions**
   - Energy usage and base travel time are read from the deterministic transportation graph.
   - Actual travel time is optionally perturbed by the traffic simulator: `Δt ~ 𝒩(mean=travel_time, std=std_dev)`, truncated to `[0.01·travel_time, 2·travel_time]`, where `std_dev = travel_time * traffic.std_dev_factor` capped by `traffic.max_std_dev_hours`.
   - Upon scheduling a `TRUCK_ROUTING` event, truck batteries are decremented immediately; if insufficient, the truck fails.
   - Arrival triggers the event handler, which updates routes, delivery completion, and potentially pushes a new `TRUCK_READY` event (or transitions into charger waiting state).

2. **Charging actions**
   - Charging duration is derived from charger type (`Level2`/`DCFast`), efficiency, and remaining deficit.
   - FCFS gating mediated by `ChargingStation` may postpone start times via waitlists and next-check events. Once a plug is allocated, the truck state switches to `"charging"` until the scheduled completion.

3. **Waiting dynamics**
   - If no plug is free, a truck is parked in the waitlist. It receives a deterministic estimation of when to re-check, derived from charger utilization and historical waiting-time lookup tables (`waiting_time_lookup.json`). When another truck finishes, `wake_waiting_trucks` moves the next truck into the plug and schedules its `TRUCK_READY` event immediately.

4. **Initial state randomness**
   - Each episode samples truck start nodes and delivery sequences via `create_truck`, ensuring stochastic initial states (independent identically distributed across episodes).

Because all stochasticity (initial delivery samples and traffic perturbations) has known distributions, the environment defines a valid MDP. Rewards depend solely on the current state, action, and sampled randomness, and the combination of global clock, event queue, truck states, and charging-station data renders the next state conditionally independent of the past.

---

## Termination

Episodes terminate when every truck is `"complete"` (all deliveries served) or `"failed"`, or when the global clock exceeds `environment.max_time`, in which case the step is marked truncated. Statistics and visualizations are emitted via `EnvironmentPlotter`/`EnvironmentStatistics` during `env.close()`.
