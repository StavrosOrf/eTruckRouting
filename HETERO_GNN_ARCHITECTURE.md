# HeteroGNN Architecture Documentation

## Overview

This document describes the Heterogeneous Graph Neural Network (HeteroGNN) architecture used for the TD3 Action-GNN agent in the Electric Vehicle Routing Problem (EVPR).

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     INPUT: Heterogeneous Graph Data                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Node Features:                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
│  │   Trucks    │  │ Deliveries  │  │  Chargers   │                     │
│  │ [N_t × 13]  │  │ [N_d × 2]   │  │ [N_c × 5]   │                     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                     │
│         │                 │                 │                            │
│  Edge Features (all edge types):                                        │
│  [E × 2] where E = num_edges                                            │
│  Features: [energy_cost, time_cost]                                     │
│                                                                          │
│  Feasible Actions:                                                       │
│  • Reachable chargers (within current truck energy)                     │
│  • Next delivery (if reachable with current energy)                     │
│  • Current charger (if truck is connected & ready to charge)            │
│                                                                          │
│  Edge Types (Bidirectional):                                            │
│  • truck ↔ delivery                                                      │
│  • truck ↔ charger                                                       │
│  • charger ↔ charger                                                     │
│  • truck ↔ truck                                                         │
│  • delivery ↔ delivery                                                   │
│  • charger ↔ delivery                                                    │
└─────────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      INPUT PROJECTION LAYERS                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  nn.Linear(in_features → hidden_dim=64) for each node type:            │
│                                                                          │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐            │
│  │   Trucks    │      │ Deliveries  │      │  Chargers   │            │
│  │ 13 → 64     │      │  2 → 64     │      │  5 → 64     │            │
│  │ [N_t × 13]  │      │ [N_d × 2]   │      │ [N_c × 5]   │            │
│  │     ↓       │      │     ↓       │      │     ↓       │            │
│  │   ReLU()    │      │   ReLU()    │      │   ReLU()    │            │
│  │     ↓       │      │     ↓       │      │     ↓       │            │
│  │ [N_t × 64]  │      │ [N_d × 64]  │      │ [N_c × 64]  │            │
│  └─────────────┘      └─────────────┘      └─────────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│         HETERO INTERACTION LAYERS (num_layers=3, repeated 3×)          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  For EACH edge type (src → dst):                                        │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  EdgeConditionedConv(in_channels=64, out_channels=64)   │          │
│  │                                                           │          │
│  │  MESSAGE COMPUTATION:                                    │          │
│  │  ┌────────────────────────────────────────────────────┐ │          │
│  │  │ Input: [h_i || h_j || edge_attr]                  │ │          │
│  │  │        [64  || 64  || 2] = 130 dims               │ │          │
│  │  │                  ↓                                 │ │          │
│  │  │ message_mlp: MLP(in=130, hidden=64, out=64)       │ │          │
│  │  │   Layer 1: Linear(130 → 64)                       │ │          │
│  │  │            LayerNorm(64)                           │ │          │
│  │  │            ReLU()                                  │ │          │
│  │  │   Layer 2: Linear(64 → 64)                        │ │          │
│  │  │                  ↓                                 │ │          │
│  │  │ Output: message [E × 64]                          │ │          │
│  │  └────────────────────────────────────────────────────┘ │          │
│  │                                                           │          │
│  │  AGGREGATION: Sum over incoming edges                    │          │
│  │  aggregated_msg: [N_dst × 64]                           │          │
│  │                                                           │          │
│  │  UPDATE COMPUTATION:                                     │          │
│  │  ┌────────────────────────────────────────────────────┐ │          │
│  │  │ Input: [h_i || aggregated_msg]                    │ │          │
│  │  │        [64  || 64] = 128 dims                     │ │          │
│  │  │                  ↓                                 │ │          │
│  │  │ update_mlp: MLP(in=128, hidden=64, out=64)        │ │          │
│  │  │   Layer 1: Linear(128 → 64)                       │ │          │
│  │  │            LayerNorm(64)                           │ │          │
│  │  │            ReLU()                                  │ │          │
│  │  │   Layer 2: Linear(64 → 64)                        │ │          │
│  │  │                  ↓                                 │ │          │
│  │  │ Output: h_i' [N_dst × 64]                         │ │          │
│  │  └────────────────────────────────────────────────────┘ │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                          │
│  Applied to all edge types, outputs averaged per node type              │
│  Final output per layer: {node_type: [N × 64]} → ReLU() applied        │
└─────────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        ACTOR: TWO OUTPUT HEADS                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PATH 1: DISCRETE NODE SELECTION HEAD                                   │
│  ┌────────────────────────────────────────────────────────┐            │
│  │  Input: Node embeddings {node_type: [N × 64]}         │            │
│  │                           ↓                             │            │
│  │  Output projection per node type:                      │            │
│  │    Linear(64 → 1) for each node type                  │            │
│  │                           ↓                             │            │
│  │  Map to actions via action_to_node_map:               │            │
│  │    For each action:                                    │            │
│  │      - Routing actions: use node score                │            │
│  │      - Charging action: use charge_action_head output │            │
│  │                           ↓                             │            │
│  │  Apply feasibility mask (optional, via apply_mask):   │            │
│  │    masked = where(feasible, scores, -1e9)             │            │
│  │                           ↓                             │            │
│  │  Tanh() for squashing: [num_actions] ∈ [-1, 1]       │            │
│  │                           ↓                             │            │
│  │  action_scores: [num_actions] (continuous values)     │            │
│  └────────────────────────────────────────────────────────┘            │
│                                                                          │
│  CHARGE ACTION HEAD (Global):                                           │
│  ┌────────────────────────────────────────────────────────┐            │
│  │  Input: Graph embedding [1 × 192]                     │            │
│  │    (pooled from all node types: 3 × 64)               │            │
│  │                           ↓                             │            │
│  │  MLP: 192 → 64 → 1                                    │            │
│  │                           ↓                             │            │
│  │  charge_score: [1] (scalar, used for charge action)   │            │
│  └────────────────────────────────────────────────────────┘            │
│                                                                          │
│  PATH 2: CHARGING DURATION HEAD                                         │
│  ┌────────────────────────────────────────────────────────┐            │
│  │  Input: Graph embedding [1 × 192]                     │            │
│  │    (pooled from all node types: 3 × 64)               │            │
│  │                           ↓                             │            │
│  │  MLP: 192 → 64 → 1                                    │            │
│  │                           ↓                             │            │
│  │  Sigmoid() → value ∈ [0, 1]                           │            │
│  │                           ↓                             │            │
│  │  Scale by max_charging_duration (10.0)                │            │
│  │                           ↓                             │            │
│  │  charging_duration: [1] ∈ [0, 10]                     │            │
│  └────────────────────────────────────────────────────────┘            │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    ACTION SELECTION LOGIC                        │  │
│  │                                                                  │  │
│  │  EXPLORATION MODE (during environment interaction):             │  │
│  │  ┌────────────────────────────────────────────────────┐        │  │
│  │  │ 1. Call with apply_mask=True                       │        │  │
│  │  │    action_scores, charging_duration = actor(data)  │        │  │
│  │  │                                                     │        │  │
│  │  │ 2. selected_action = argmax(action_scores)         │        │  │
│  │  │                                                     │        │  │
│  │  │ 3. Return: (selected_action, charging_duration)    │        │  │
│  │  └────────────────────────────────────────────────────┘        │  │
│  │                                                                  │  │
│  │  TRAINING MODE (for critic):                                    │  │
│  │  ┌────────────────────────────────────────────────────┐        │  │
│  │  │ 1. Call with apply_mask=False to get continuous   │        │  │
│  │  │    action values for critic gradient flow          │        │  │
│  │  │                                                     │        │  │
│  │  │ 2. Pass (action_scores, charging_duration) to      │        │  │
│  │  │    critic for Q-value estimation                   │        │  │
│  │  └────────────────────────────────────────────────────┘        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  FINAL OUTPUT:                                                           │
│    (action_scores, charging_duration)                                   │
│    where action_scores: [num_actions] ∈ [-1, 1] (after tanh)          │
│          charging_duration: [1] ∈ [0, max_charging_duration]           │
└─────────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CRITIC Q-VALUE ESTIMATION                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT: State + Action Index + Charging Duration                        │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ Augment Node Features:                                   │          │
│  │                                                           │          │
│  │ For each node:                                           │          │
│  │   base_features: [original_dim]                          │          │
│  │   action_value: normalized action index (scalar)        │          │
│  │   charge_value: charging_duration value                 │          │
│  │                                                           │          │
│  │   augmented = [base || action_value || charge_value]    │          │
│  │                                                           │          │
│  │ Truck:    [13 + 1 + 1] = 15 → Linear → 64               │          │
│  │ Delivery: [2 + 1 + 1]  = 4  → Linear → 64               │          │
│  │ Charger:  [5 + 1 + 1]  = 7  → Linear → 64               │          │
│  └──────────────────────────────────────────────────────────┘          │
│                              ▼                                           │
│         Apply HeteroInteractionLayers (3 layers, same as Actor)        │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ Global Mean Pool per node type:                          │          │
│  │   [N_t × 64] → [1 × 64]                                 │          │
│  │   [N_d × 64] → [1 × 64]                                 │          │
│  │   [N_c × 64] → [1 × 64]                                 │          │
│  │                           ↓                               │          │
│  │ Concatenate: [1 × 192]                                   │          │
│  │                           ↓                               │          │
│  │ MLP(in=192, hidden=256, out=1, num_layers=3):           │          │
│  │   Layer 1: Linear(192 → 256) + LayerNorm + ReLU         │          │
│  │   Layer 2: Linear(256 → 256) + LayerNorm + ReLU         │          │
│  │   Layer 3: Linear(256 → 1)                              │          │
│  │                           ↓                               │          │
│  │ Q-value: [1] (scalar)                                    │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                          │
│  Twin Critics: Q1 and Q2 (identical architecture, separate weights)    │
│  Used in TD3 for reduced overestimation bias                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Message Passing Detail

```
┌─────────────────────────────────────────────────────────────────────────┐
│           EDGE-CONDITIONED MESSAGE PASSING (Single Edge Type)           │
│                                                                          │
│  Node i (target)          Node j (source)         Edge (i,j)            │
│      h_i                       h_j                 e_ij                 │
│     [64]                      [64]                 [2]                  │
│       ┃                         ┃                   ┃                   │
│       ┃                         ┃                   ┃                   │
│       ┗━━━━━━━━━┳━━━━━━━━━━━━━━┛━━━━━━━━━━━━━━━━━━┛                   │
│                  ▼                                                       │
│         Concatenate: [h_i || h_j || e_ij]                               │
│                      [64 + 64 + 2 = 130]                                │
│                  ▼                                                       │
│         ┌──────────────────┐                                            │
│         │   Message MLP    │                                            │
│         │    130 → 64      │                                            │
│         │  (2 layers with  │                                            │
│         │   LayerNorm)     │                                            │
│         └────────┬─────────┘                                            │
│                  ▼                                                       │
│            m_j→i [64] (message)                                         │
│                                                                          │
│  Aggregate all messages to node i:                                      │
│            m_i = Σ m_j→i  (sum aggregation)                             │
│                j∈N(i)                                                    │
│                  ▼                                                       │
│         Concatenate: [h_i || m_i]                                       │
│                      [64 + 64 = 128]                                    │
│                  ▼                                                       │
│         ┌──────────────────┐                                            │
│         │   Update MLP     │                                            │
│         │    128 → 64      │                                            │
│         │  (1 layer with   │                                            │
│         │   LayerNorm)     │                                            │
│         └────────┬─────────┘                                            │
│                  ▼                                                       │
│               h_i' [64] (updated node embedding)                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Dimension Summary

| Component | Input Dims | Output Dims | Notes |
|-----------|-----------|-------------|-------|
| **Actor Input Projection** | Truck: 13, Delivery: 2, Charger: 5 | All → 64 | Separate Linear layer per type |
| **EdgeConditionedConv Message** | [h_i \|\| h_j \|\| edge] = 130 | 64 | 2-layer MLP with LayerNorm |
| **EdgeConditionedConv Update** | [h_i \|\| msg] = 128 | 64 | 1-layer MLP with LayerNorm |
| **Actor Output Heads** | 64 per node | 1 per node | Linear projection |
| **Graph Embedding** | 3 node types × 64 | 192 | Global mean pool + concatenate |
| **Charging Duration Head** | 192 → 64 → 1 | 1 | Sigmoid activation, scaled to [0, 10] |
| **Charge Action Head** | 192 → 64 → 1 | 1 | Used for charging action score |
| **Critic Input Projection** | Truck: 15, Delivery: 4, Charger: 7 | All → 64 | +2 dims for action & charge info |
| **Critic Q-Net** | 192 → 256 → 256 → 1 | 1 | 3-layer MLP, outputs Q-value |

## Key Features

### 1. Heterogeneous Nodes
- **Trucks**: 13 features (position, energy, capacity, etc.)
- **Deliveries**: 2 features (position/demand info)
- **Chargers**: 5 features (position, capacity, availability)

### 2. Edge-Conditioned Messages
- Messages incorporate both node features and edge features
- Edge features: [energy_cost, time_cost]
- Enables cost-aware routing decisions

### 3. Bidirectional Edge Types
All edge types work in both directions:
- truck ↔ delivery
- truck ↔ charger
- charger ↔ charger
- truck ↔ truck
- delivery ↔ delivery
- charger ↔ delivery

### 4. Action Mapping
- Uses `action_to_node_map` to map discrete actions to graph nodes
- Distinguishes between:
  - **Routing actions**: Go to delivery/charger node
  - **Charging action**: Charge at current location

### 5. Feasibility Masking
- Optional masking via `apply_mask` parameter
- When `apply_mask=True`: Infeasible actions get score -1e9
- When `apply_mask=False`: Continuous values for critic training
- Feasible actions include:
  - Reachable chargers (within current energy)
  - Next delivery (if reachable)
  - Current charger (if connected and ready)

### 6. Dual Output (Actor)
- **Path 1**: Discrete action scores (node selection)
  - One score per feasible action
  - Uses per-node Linear projections for routing
  - Uses global charge_action_head for charging
  - Tanh activation for bounded output ∈ [-1, 1]
- **Path 2**: Continuous charging duration
  - Global graph-level feature
  - Sigmoid activation scaled to [0, max_charging_duration]

### 7. Twin Critics (TD3)
- Two identical Q-networks (Q1 and Q2)
- Separate weights, same architecture
- Reduces overestimation bias in Q-learning

## Training vs. Exploration Mode

### Exploration (Environment Interaction)
```python
# Apply feasibility mask
action_scores, charging_duration = actor(state, apply_mask=True)

# Select best action
selected_action = torch.argmax(action_scores)
```

### Training (Critic Update)
```python
# No mask for continuous gradient flow
action_scores, charging_duration = actor(state, apply_mask=False)

# Pass to critic
q_value = critic(state, action_index, charging_duration)
```

## Implementation Notes

1. **Batch Processing**: Supports batched graphs via PyTorch Geometric's batching mechanism
2. **Device Management**: All tensors properly moved to specified device (CPU/GPU)
3. **Edge Cases**: Handles empty edge types and missing node types gracefully
4. **Normalization**: LayerNorm used in MLPs for stable training
5. **Activation**: ReLU for hidden layers, Tanh/Sigmoid for outputs

## File Location
Implementation: `/home/sorfanouda/EVPR/algo/networks.py`
