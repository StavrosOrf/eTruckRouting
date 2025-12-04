# GNN State Space Architecture

**Heterogeneous Graph Neural Network for Electric Vehicle Routing**

---

## Overview

The GNN State Space converts the truck routing environment into a **Heterogeneous Graph** representation suitable for Graph Neural Networks. This enables the agent to learn routing policies by reasoning over the spatial relationships between trucks, deliveries, and charging stations.

```mermaid
graph LR
    Env[Event-Driven Environment] --> GNN[GNN State Space]
    GNN --> HeteroGraph[Heterogeneous Graph]
    HeteroGraph --> Agent[GNN Agent]
    Agent --> Action[Routing Decision]
    
    style GNN fill:#aaf,stroke:#33d,stroke-width:3px
    style HeteroGraph fill:#afa,stroke:#3d3,stroke-width:3px
```

---

## Graph Structure

### Node Types (3 Types)

```mermaid
graph TD
    subgraph "Node Types"
        T[🚛 Truck Nodes<br/>Active trucks only]
        D[📦 Delivery Nodes<br/>Undelivered stops]
        C[🔌 Charger Nodes<br/>All charging stations]
    end
    
    T --> T1[Features: 13]
    D --> D1[Features: 3]
    C --> C1[Features: 4]
    
    style T fill:#faa,stroke:#d33,stroke-width:2px
    style D fill:#afa,stroke:#3d3,stroke-width:2px
    style C fill:#aaf,stroke:#33d,stroke-width:2px
    style T1 fill:#fdd
    style D1 fill:#dfd
    style C1 fill:#ddf
```

**Key Properties:**
- ✅ No padding - each node type has different feature dimensions
- ✅ Dynamic - only active/undelivered nodes included
- ✅ No depot nodes - simplified representation
- ❌ Completed/failed trucks excluded
- ❌ Already delivered nodes excluded

---

## Node Features

### 🚛 Truck Node Features (13 features)

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `node_type` | Node type identifier | `÷ 3` (0.0 for truck) |
| `current_position` | Current node ID | `÷ num_nodes` |
| `battery_level` | Current battery (kWh) | `÷ capacity` |
| `battery_percentage` | Battery % | `÷ 100` |
| `is_ready` | State: ready for action | Binary (0/1) |
| `is_routing` | State: traveling | Binary (0/1) |
| `is_waiting_to_charge` | State: in queue | Binary (0/1) |
| `is_charging` | State: charging | Binary (0/1) |
| `deliveries_done` | Completed deliveries | `÷ total_deliveries` |
| `deliveries_remaining` | Remaining deliveries | `÷ total_deliveries` |
| `time_elapsed` | Total time spent | `÷ max_time` |
| `distance_traveled` | Total distance (km) | `÷ 1000` |
| `time_to_destination` | ETA if routing | `÷ max_time` |

### 📦 Delivery Node Features (3 features)

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `node_type` | Node type identifier | `÷ 3` (0.33 for delivery) |
| `node_id` | Delivery node ID | `÷ num_nodes` |
| `sequence_index` | Position in route | `÷ max_stops` |

### 🔌 Charger Node Features (4 features)

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `node_type` | Node type identifier | `÷ 3` (0.67 for charger) |
| `node_id` | Charger node ID | `÷ num_charging_nodes` |
| `occupancy_rate` | Current/capacity | `0.0 - 1.0` |
| `queue_length` | Waiting trucks | `÷ num_trucks` |

---

## Edge Types & Connectivity

### Edge Structure (9 Edge Types)

```mermaid
graph LR
    T[🚛 Truck]
    D[📦 Delivery]
    C[🔌 Charger]
    
    T <-->|State-dependent| D
    T <-->|State-dependent| C
    T <-->|Self-edges| T
    C <-->|Always feasible| C
    C <-->|Always feasible| D
    D <-->|Always feasible| D
    
    style T fill:#faa,stroke:#d33,stroke-width:2px
    style D fill:#afa,stroke:#3d3,stroke-width:2px
    style C fill:#aaf,stroke:#33d,stroke-width:2px
```

**Edge Types:**
1. `(truck, to, delivery)` - Truck to delivery navigation
2. `(delivery, to, truck)` - Bidirectional
3. `(truck, to, charger)` - Truck to charger navigation
4. `(charger, to, truck)` - Bidirectional
5. `(truck, to, truck)` - Inter-truck relationships
6. `(charger, to, charger)` - Charger network
7. `(charger, to, delivery)` - Infrastructure-delivery links
8. `(delivery, to, charger)` - Bidirectional
9. `(delivery, to, delivery)` - Delivery sequence relationships

### Edge Features (2 features per edge)

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `energy_distance` | Energy required (kWh) | `÷ 1000` |
| `time_distance` | Travel time (hours) | `÷ max_time` |

**Special Case:** When truck is routing, edge to destination has:
- `energy_distance = 0.0` (already committed)
- `time_distance = remaining_time / max_time`

---

## State-Dependent Edge Construction

### When Truck is READY

```mermaid
graph TD
    Truck[🚛 Truck<br/>State: READY]
    
    Truck -->|energy < battery| NextDel[📦 Next Delivery]
    Truck -->|energy < battery| C1[🔌 Charger 1]
    Truck -->|energy < battery| C2[🔌 Charger 2]
    Truck -->|energy < battery| CN[🔌 Charger N]
    
    Truck -->|0 energy/time| CurrentLoc[🔌 Current Location<br/>if at charger]
    
    style Truck fill:#faa,stroke:#d33,stroke-width:3px
    style NextDel fill:#afa,stroke:#3d3,stroke-width:2px
    style C1 fill:#aaf,stroke:#33d,stroke-width:2px
    style C2 fill:#aaf,stroke:#33d,stroke-width:2px
    style CN fill:#aaf,stroke:#33d,stroke-width:2px
    style CurrentLoc fill:#aaf,stroke:#33d,stroke-width:3px
```

**Rules:**
- ✅ Connect to **next delivery** (if energy feasible)
- ✅ Connect to **all chargers** (if energy feasible)
- ✅ Include self-loop if at charger (0 cost)

### When Truck is ROUTING

```mermaid
graph TD
    Truck[🚛 Truck<br/>State: ROUTING]
    
    Truck -->|only connection| Dest[🎯 Destination<br/>Delivery or Charger]
    
    style Truck fill:#faa,stroke:#d33,stroke-width:3px
    style Dest fill:#ffa,stroke:#da3,stroke-width:3px
```

**Rules:**
- ✅ Connect **only to destination** node
- ✅ Edge weight = remaining time (energy = 0)

### When Truck is CHARGING/WAITING

```mermaid
graph TD
    Truck[🚛 Truck<br/>State: CHARGING]
    
    Truck -->|only connection| Charger[🔌 Current Charger]
    
    style Truck fill:#faa,stroke:#d33,stroke-width:3px
    style Charger fill:#aaf,stroke:#33d,stroke-width:3px
```

**Rules:**
- ✅ Connect **only to current charger**
- ✅ Edge weight = 0 (at location)

---

## Feasible Actions Architecture

### Action Space Structure

```mermaid
graph TD
    Start[Action Selection] --> ActionTypes{Action Types}
    
    ActionTypes -->|Navigate| NavActions[Navigation Actions]
    ActionTypes -->|Charge| ChargeActions[Charging Actions]
    
    NavActions --> A1[Action 0: Go to Charger 0]
    NavActions --> A2[Action 1: Go to Charger 1]
    NavActions --> AN[Action N-1: Go to Charger N-1]
    NavActions --> ADel[Action N: Go to Next Delivery]
    
    ChargeActions --> C1[Action N+1: Charge 0.5h]
    ChargeActions --> C2[Action N+2: Charge 1.0h]
    ChargeActions --> CM[Action N+M: Charge 2.0h]
    
    style NavActions fill:#afa,stroke:#3d3,stroke-width:2px
    style ChargeActions fill:#aaf,stroke:#33d,stroke-width:2px
    style ADel fill:#ffa,stroke:#da3,stroke-width:2px
```

**Action Ordering:**
1. **Actions 0 to N-1:** Navigate to chargers (sorted by charger ID)
2. **Action N:** Navigate to next delivery
3. **Actions N+1 to N+M:** Charge at current location (by duration)

### Feasibility Mask Generation

```mermaid
graph TD
    Start([Generate Action Mask]) --> TruckState{Truck State?}
    
    TruckState -->|At Charger| AtCharger[At Charger Logic]
    TruckState -->|Not at Charger| NotAtCharger[Not at Charger Logic]
    
    AtCharger --> MustLeave{must_leave_charger?}
    
    MustLeave -->|Yes| DisableCharge[❌ Disable all charging<br/>✅ Enable navigation]
    MustLeave -->|No| EnableCharge[✅ Enable charging<br/>if battery gain sufficient]
    
    NotAtCharger --> NavOnly[✅ Enable navigation only<br/>❌ Disable all charging]
    
    DisableCharge --> CheckNav[Check Navigation Feasibility]
    EnableCharge --> CheckNav
    NavOnly --> CheckNav
    
    CheckNav --> Path{Path exists?}
    Path -->|No| Fail1[❌ Infeasible]
    Path -->|Yes| Battery{Battery sufficient?}
    
    Battery -->|No| Fail2[❌ Infeasible]
    Battery -->|Yes| PostCheck{Would strand truck?}
    
    PostCheck -->|Yes| Fail3[❌ Infeasible]
    PostCheck -->|No| Success[✅ Feasible]
    
    style Fail1 fill:#f88,stroke:#d33,stroke-width:2px
    style Fail2 fill:#f88,stroke:#d33,stroke-width:2px
    style Fail3 fill:#f88,stroke:#d33,stroke-width:2px
    style Success fill:#8f8,stroke:#3d3,stroke-width:2px
    style DisableCharge fill:#faa
    style EnableCharge fill:#afa
```

### Action Graph Features (3 features per action)

For each **feasible** action, we compute:

| Feature | Description | Values |
|---------|-------------|--------|
| `action_type` | Type of action | `1/3` = delivery, `2/3` = charger, `3/3` = charge |
| `resulting_soc` | Battery after action | `0.0 - 1.0` |
| `charge_duration` | Charging time (if charging) | `÷ max_charge_duration` |

```mermaid
graph LR
    A[Action] --> T[Action Type]
    A --> S[Resulting SOC]
    A --> D[Charge Duration]
    
    T --> T1[Navigate to Delivery: 0.33]
    T --> T2[Navigate to Charger: 0.67]
    T --> T3[Charge at Location: 1.00]
    
    S --> S1[Battery % after action]
    D --> D1[Normalized 0.0-1.0]
    
    style A fill:#ffa,stroke:#da3,stroke-width:2px
```

---

## Action-to-Node Mapping

The GNN agent selects actions by attending to node embeddings:

```mermaid
graph TD
    Action[Discrete Action] --> Map{Action Type?}
    
    Map -->|Navigate to Charger| ChargerNode[🔌 Charger Node Embedding]
    Map -->|Navigate to Delivery| DeliveryNode[📦 Delivery Node Embedding]
    Map -->|Charge Here| CurrentNode[🔌 Current Charger Embedding]
    
    ChargerNode --> Index1[action_node_type = CHARGER<br/>action_local_index = charger_idx]
    DeliveryNode --> Index2[action_node_type = DELIVERY<br/>action_local_index = delivery_idx]
    CurrentNode --> Index3[action_node_type = CHARGER<br/>action_local_index = current_charger_idx]
    
    Index1 --> Embed[Node Embedding Lookup]
    Index2 --> Embed
    Index3 --> Embed
    
    Embed --> Score[Action Score Computation]
    
    style Action fill:#ffa,stroke:#da3,stroke-width:2px
    style Embed fill:#afa,stroke:#3d3,stroke-width:2px
```

**Metadata Tensors:**
- `action_node_type`: Maps action → node type (truck/delivery/charger)
- `action_local_index`: Maps action → index within node type tensor
- `action_is_charging`: Boolean flag for charging actions
- `action_charge_durations`: Charge time for each action
- `feasible_action_mask`: Boolean mask of valid actions

---

## Complete State Graph Example

### Scenario: 1 Truck, 2 Deliveries, 2 Chargers

```mermaid
graph TD
    T1[🚛 Truck 1<br/>Battery: 80%<br/>State: READY<br/>At Node 5]
    
    D1[📦 Delivery 1<br/>Node 10<br/>Next in sequence]
    D2[📦 Delivery 2<br/>Node 15<br/>Second in sequence]
    
    C1[🔌 Charger 1<br/>Node 2<br/>Occupancy: 0/2<br/>Queue: 0]
    C2[🔌 Charger 2<br/>Node 7<br/>Occupancy: 1/2<br/>Queue: 1]
    
    T1 -->|energy: 20kWh<br/>time: 1.5h| D1
    T1 -->|energy: 15kWh<br/>time: 1.0h| C1
    T1 -->|energy: 10kWh<br/>time: 0.8h| C2
    
    D1 -->|energy: 25kWh| D2
    D1 -->|energy: 18kWh| C1
    D1 -->|energy: 12kWh| C2
    
    D2 -->|energy: 22kWh| C1
    D2 -->|energy: 30kWh| C2
    
    C1 -->|energy: 20kWh| C2
    C2 -->|energy: 20kWh| C1
    
    style T1 fill:#faa,stroke:#d33,stroke-width:3px
    style D1 fill:#ffa,stroke:#da3,stroke-width:3px
    style D2 fill:#afa,stroke:#3d3,stroke-width:2px
    style C1 fill:#aaf,stroke:#33d,stroke-width:2px
    style C2 fill:#aaf,stroke:#33d,stroke-width:2px
```

**Feasible Actions:**
- ✅ **Action 0:** Navigate to Charger 1 (energy feasible)
- ✅ **Action 1:** Navigate to Charger 2 (energy feasible)
- ✅ **Action 2:** Navigate to Delivery 1 (energy feasible + won't strand)
- ❌ **Action 3-5:** Charge here (not at charger)

---

## Data Flow Pipeline

```mermaid
graph TB
    Env[Event-Driven Environment] --> Extract[Extract State]
    
    Extract --> Nodes[Build Node Features]
    Extract --> Edges[Build Edge Connectivity]
    Extract --> Actions[Generate Action Space]
    
    Nodes --> NT[Truck Nodes]
    Nodes --> ND[Delivery Nodes]
    Nodes --> NC[Charger Nodes]
    
    Edges --> ET[Truck Edges<br/>State-dependent]
    Edges --> EI[Infrastructure Edges<br/>Always present]
    
    Actions --> AM[Action Mask]
    Actions --> AF[Action Features]
    Actions --> AN[Action-to-Node Map]
    
    NT --> HeteroData[HeteroData Object]
    ND --> HeteroData
    NC --> HeteroData
    ET --> HeteroData
    EI --> HeteroData
    AM --> HeteroData
    AF --> HeteroData
    AN --> HeteroData
    
    HeteroData --> GNN[GNN Agent]
    GNN --> Policy[Policy Output]
    Policy --> Env
    
    style Env fill:#faa,stroke:#d33,stroke-width:2px
    style HeteroData fill:#afa,stroke:#3d3,stroke-width:3px
    style GNN fill:#aaf,stroke:#33d,stroke-width:2px
```

---

## Key Advantages

### 1. Dynamic Graph Structure
- Automatically excludes completed/failed trucks
- Removes delivered nodes
- Adapts to current problem state

### 2. State-Aware Connectivity
- Truck edges reflect current state (ready/routing/charging)
- Prevents invalid actions at graph level
- Encodes temporal constraints

### 3. Heterogeneous Design
- Different node types capture different semantics
- Flexible feature dimensions per type
- Enables specialized message passing

### 4. Action Grounding
- Direct mapping from actions to node embeddings
- GNN can attend to action targets
- Feasibility encoded in both graph structure and mask

### 5. Scalability
- No padding → efficient memory usage
- Sparse connectivity → O(E) message passing
- Batch processing support

---

## Implementation Notes

### Normalization Strategy
- **Spatial features:** Divided by graph size
- **Battery features:** Divided by capacity (0-1 range)
- **Time features:** Divided by max simulation time
- **Distance features:** Divided by 1000 (km → normalized)

### Edge Feasibility Criteria
```python
# Edge is added if:
energy_required < truck_battery  # For truck edges
energy_required < max_battery_capacity  # For infrastructure edges
energy_required != infinity  # Path exists
```

### Action Feasibility Criteria
```python
# Navigation action is feasible if:
1. Path exists (energy != infinity)
2. Battery sufficient (energy < current_battery)
3. Won't strand truck (post-delivery check)
4. Truck not required to charge now

# Charging action is feasible if:
1. At charger location
2. Not forced to leave (must_leave_charger == False)
3. Charge provides enough energy to reach next destination
```

---

## Summary

The GNN State Space provides a **rich, dynamic, heterogeneous graph representation** that:

✅ Captures spatial relationships between trucks, deliveries, and chargers  
✅ Encodes truck states and temporal constraints  
✅ Provides feasible action space with direct node grounding  
✅ Scales efficiently with problem size  
✅ Enables end-to-end learning of routing policies  

**Result:** GNN agents can learn to reason about multi-step planning, battery constraints, and charging infrastructure to solve the EV routing problem.
