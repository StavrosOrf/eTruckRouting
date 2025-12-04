# Improved GNN State Space Summary

## Node Types & Features

### 1. **Truck Nodes** (10 features) ✨ IMPROVED
Active trucks only (excludes failed/completed trucks)

**Features (all normalized):**
1. **Node type** (0-1): Normalized by number of node types
2. **Current position** (0-1): Normalized by total number of nodes in graph
3. **Current battery SoC estimate** (0-1): Accounts for energy consumed during routing 🆕
   - For routing trucks: estimates battery based on route progress
   - Provides real-time battery estimate, not just start-of-route value
4. **Is active truck** (0/1): Binary flag indicating if this is the truck we control 🆕
5. **Deliveries completed** (0-1): Normalized by total deliveries for this truck
6. **Deliveries remaining** (0-1): Normalized by total deliveries for this truck
7. **Time elapsed** (0-1): Normalized by max simulation time
8. **Distance traveled** (0-1): Normalized by dividing by 1000
9. **Time to destination** (0-1): Normalized by max time; 0 if not routing
10. **Time to finish charging** (0-1): Normalized by max time; 0 if not charging 🆕

**Changes Made:**
- ✅ Removed redundant battery percentage (was duplicated)
- ✅ Removed one-hot state encoding (ready/routing/waiting/charging) - reduces 4 features to 0
- ✅ Added is_active_truck flag to identify the controllable truck
- ✅ Added time_to_finish_charging for better charge state awareness
- ✅ Enhanced battery SoC to estimate current charge during routing (accounts for consumed energy)
- **Net change:** 13 features → 10 features (more informative, less redundant)

---

### 2. **Delivery Nodes** (3 features) ✨ IMPROVED
Undelivered nodes from active trucks only

**Features (all normalized):**
1. **Node type** (0-1): Normalized by number of node types
2. **Node ID** (0-1): Normalized by total number of nodes
3. **Delivery sequence index** (0-1): Position in **active truck's** remaining deliveries, normalized by max stops 🆕

**Changes Made:**
- ✅ Changed from minimum position across all trucks → position in **active truck only**
- **Benefit:** More relevant to the current decision-maker; removes ambiguity from multi-truck scenarios

---

### 3. **Charger Nodes** (5 features) ✨ IMPROVED
All charging stations in the environment

**Features (all normalized):**
1. **Node type** (0-1): Normalized by number of node types
2. **Node ID** (0-1): Normalized by number of charging stations
3. **Occupancy rate** (0-1): Current occupancy / capacity
4. **Queue length** (0-1): Normalized by number of trucks
5. **Time for queue to empty** (0-1): Normalized by max simulation time 🆕
   - Estimates remaining time for current charging trucks + queued trucks
   - Accounts for parallel charging capacity

**Changes Made:**
- ✅ Added time_for_queue_to_empty feature
- **Net change:** 4 features → 5 features
- **Benefit:** Helps agent understand wait times at busy chargers

---

## Edge Types & Features

### Edge Structure (9 bidirectional edge types)
- `truck ↔ delivery`
- `truck ↔ charger`
- `truck ↔ truck`
- `charger ↔ charger`
- `charger ↔ delivery`
- `delivery ↔ delivery`

### Edge Features (2 per edge)
1. **Energy distance** (0-1): Normalized by dividing by 1000
2. **Time to traverse** (0-1): Normalized by max simulation time

### Edge Construction Rules (State-Dependent)
- **READY trucks**: Connect to next delivery + all feasible chargers
- **CHARGING/WAITING trucks**: Only connect to current charger (0 energy/time)
- **ROUTING trucks**: Only connect to destination node
- **Feasibility**: Energy required < current battery capacity

---

## Action Space Encoding

Discrete action metadata for the active truck:

**Action Structure:**
```
[next_delivery, charger_0, charger_1, ..., charger_N, charge_here]
```

**Metadata (stored in graph):**
- `action_to_node_map`: Maps action index → (node_id, is_charging_action)
- `feasible_action_mask`: Boolean tensor marking valid actions
- `action_node_type`: Node type code per action
- `action_local_index`: Index within node type tensor
- `action_is_charging`: Boolean flag for charging action
- `node_id_to_type`: Dictionary mapping node IDs to (type, local_idx)

---

## Summary of Improvements

### Truck Nodes
- **Before:** 13 features with redundancy
- **After:** 10 features, more informative
- **Key improvements:**
  - Real-time SoC estimation during routing
  - Active truck identification flag
  - Charging completion time
  - Removed redundant battery percentage
  - Removed verbose one-hot state encoding

### Delivery Nodes
- **Before:** Sequence index = min across all trucks (ambiguous)
- **After:** Sequence index for active truck only (decision-relevant)

### Charger Nodes
- **Before:** 4 features
- **After:** 5 features with queue wait time estimation
- **Key improvement:** Time for queue to empty helps with charging planning

---

## Design Benefits

✅ **More efficient:** Reduced truck features from 13 → 10  
✅ **More informative:** Real-time SoC estimation, charging time awareness  
✅ **More relevant:** Features aligned with active truck's decision-making  
✅ **Better charging awareness:** Queue wait time estimation for chargers  
✅ **Cleaner representation:** Removed redundant features (battery %, one-hot states)

---

## Total Feature Counts

- **Truck nodes:** 10 features each
- **Delivery nodes:** 3 features each
- **Charger nodes:** 5 features each
- **Edges:** 2 features each
- **Dynamic graph size:** No padding, adapts to environment state
