# PPO Variable Action GNN Architecture

**Proximal Policy Optimization with Variable-Sized Discrete Action Spaces**

---

## Overview

PPO Variable Action GNN is a reinforcement learning architecture that handles **variable-sized action spaces** at each timestep by using a **Graph Convolutional Network (GCN) over feasible actions**. This enables the agent to reason over different numbers of valid actions as the environment state changes.

```mermaid
graph LR
    Env[Environment State] --> GNN[GNN Feature Encoder]
    GNN --> Embed[Graph Embedding]
    
    Embed --> Actor[Action Graph Head]
    Embed --> Critic[Value Head]
    
    Actor --> Policy[Policy Distribution]
    Critic --> Value[State Value]
    
    Policy --> Action[Selected Action]
    Value --> PPO[PPO Update]
    
    style GNN fill:#aaf,stroke:#33d,stroke-width:3px
    style Actor fill:#faa,stroke:#d33,stroke-width:3px
    style Critic fill:#afa,stroke:#3d3,stroke-width:3px
```

---

## Architecture Components

### 1. GNN Feature Encoder (Shared Backbone)

```mermaid
graph TD
    subgraph Input[Heterogeneous Graph Input]
        T[🚛 Truck Nodes<br/>13 features]
        D[📦 Delivery Nodes<br/>3 features]
        C[🔌 Charger Nodes<br/>4 features]
        E[Edge Features<br/>energy, time]
    end
    
    subgraph Projection[Input Projection Layer]
        TP[Truck Proj<br/>→ hidden_dim]
        DP[Delivery Proj<br/>→ hidden_dim]
        CP[Charger Proj<br/>→ hidden_dim]
    end
    
    subgraph Layers[Heterogeneous GNN Layers]
        L1[Layer 1:<br/>Edge-Conditioned Message Passing]
        L2[Layer 2:<br/>Edge-Conditioned Message Passing]
        L3[Layer N:<br/>Edge-Conditioned Message Passing]
    end
    
    subgraph Pooling[Global Pooling]
        Pool[Pool each node type<br/>Concatenate]
    end
    
    T --> TP
    D --> DP
    C --> CP
    
    TP --> L1
    DP --> L1
    CP --> L1
    E --> L1
    
    L1 --> L2 --> L3
    L3 --> Pool
    
    Pool --> Output[Graph Embedding<br/>dim = hidden_dim × 3]
    
    style Input fill:#ddf
    style Projection fill:#ffd
    style Layers fill:#dfd
    style Pooling fill:#fdd
    style Output fill:#aaf,stroke:#33d,stroke-width:3px
```

**Key Features:**
- **Heterogeneous:** Separate projections for each node type
- **Edge-Conditioned:** Messages depend on edge features (energy, time)
- **Multi-Layer:** Typically 3 layers for information propagation
- **Global Pooling:** Aggregates all node types into single embedding

---

## 2. Action Graph Head (Actor)

### Action Graph Construction

For each timestep, we build a **fully connected graph over feasible actions**:

```mermaid
graph TD
    State[Current State] --> Feasible[Filter Feasible Actions]
    
    Feasible --> A1[Action 1:<br/>Go to Charger 2]
    Feasible --> A2[Action 2:<br/>Go to Charger 5]
    Feasible --> A3[Action 3:<br/>Go to Delivery 10]
    Feasible --> A4[Action 4:<br/>Charge 1.0h]
    
    A1 <--> A2
    A1 <--> A3
    A1 <--> A4
    A2 <--> A3
    A2 <--> A4
    A3 <--> A4
    
    subgraph Features[Action Features: 3D]
        F1[Action Type<br/>1/3, 2/3, or 3/3]
        F2[Resulting SOC<br/>0.0 - 1.0]
        F3[Charge Duration<br/>normalized]
    end
    
    A1 --> Features
    
    style Feasible fill:#ffa,stroke:#da3,stroke-width:2px
    style A1 fill:#afa,stroke:#3d3,stroke-width:2px
    style A2 fill:#afa,stroke:#3d3,stroke-width:2px
    style A3 fill:#ffa,stroke:#da3,stroke-width:2px
    style A4 fill:#aaf,stroke:#33d,stroke-width:2px
```

### Action Graph Head Architecture

```mermaid
graph TD
    GraphEmbed[Graph Embedding<br/>from encoder] --> StateProj[State Projection<br/>Linear + ReLU<br/>→ mlp_dim]
    
    ActionFeats[Action Features<br/>3D per action] --> ActionProj[Action Projection<br/>Linear + ReLU<br/>→ mlp_dim]
    
    ActionProj --> GCN1[GCN Layer 1<br/>Message Passing]
    GCN1 --> GCN2[GCN Layer 2<br/>Message Passing]
    
    GCN2 --> ActionEmbed[Action Embeddings<br/>mlp_dim per action]
    
    StateProj --> DotProduct[Dot Product<br/>state • action_i]
    ActionEmbed --> DotProduct
    
    DotProduct --> Logits[Action Logits<br/>1 per feasible action]
    
    Logits --> Softmax[Softmax]
    Softmax --> Distribution[Categorical Distribution]
    
    style GraphEmbed fill:#aaf,stroke:#33d,stroke-width:2px
    style ActionFeats fill:#ffa,stroke:#da3,stroke-width:2px
    style GCN1 fill:#afa,stroke:#3d3,stroke-width:2px
    style GCN2 fill:#afa,stroke:#3d3,stroke-width:2px
    style Distribution fill:#faa,stroke:#d33,stroke-width:3px
```

**Action Graph Head Process:**

1. **Project State:** `state_repr = ReLU(Linear(graph_embedding))`
2. **Project Actions:** `action_features = ReLU(Linear(action_feats))`
3. **GCN Layers:** Message passing between feasible actions
4. **Compute Logits:** `logit_i = dot(state_repr, action_embedding_i)`
5. **Sample:** Categorical distribution over feasible actions

---

## 3. Value Head (Critic)

```mermaid
graph TD
    GraphEmbed[Graph Embedding<br/>from encoder] --> MLP1[Linear Layer<br/>→ mlp_dim]
    MLP1 --> ReLU[ReLU Activation]
    ReLU --> MLP2[Linear Layer<br/>→ 1]
    MLP2 --> Value[State Value<br/>scalar]
    
    style GraphEmbed fill:#aaf,stroke:#33d,stroke-width:2px
    style Value fill:#afa,stroke:#3d3,stroke-width:3px
```

**Value Head:** Simple 2-layer MLP that estimates the state value `V(s)`

---

## Complete Forward Pass

```mermaid
graph TB
    subgraph Input[Input]
        State[HeteroData State]
        ActionMask[Feasible Action Mask]
    end
    
    subgraph Encoder[GNN Feature Encoder]
        Project[Project Node Features<br/>to hidden_dim]
        GNN[3× Hetero GNN Layers<br/>Edge-Conditioned MP]
        Pool[Global Mean Pooling<br/>per node type]
        Concat[Concatenate<br/>Truck + Delivery + Charger]
    end
    
    subgraph ActionHead[Action Graph Head]
        Filter[Filter Feasible Actions<br/>using mask]
        BuildGraph[Build Fully Connected<br/>Action Graph]
        ActionGCN[2× GCN Layers<br/>on action graph]
        Score[Dot Product Scoring<br/>state • action]
    end
    
    subgraph ValueHead[Value Head]
        ValueMLP[2-Layer MLP]
    end
    
    subgraph Output[Output]
        Logits[Action Logits<br/>variable size]
        Value[State Value V(s)]
        Policy[Categorical Policy<br/>π(a|s)]
    end
    
    State --> Project
    Project --> GNN
    GNN --> Pool
    Pool --> Concat
    
    Concat --> ActionGCN
    Concat --> ValueMLP
    
    ActionMask --> Filter
    Filter --> BuildGraph
    BuildGraph --> ActionGCN
    
    ActionGCN --> Score
    Score --> Logits
    Logits --> Policy
    
    ValueMLP --> Value
    
    style Encoder fill:#ddf
    style ActionHead fill:#fdd
    style ValueHead fill:#dfd
    style Policy fill:#faa,stroke:#d33,stroke-width:3px
    style Value fill:#afa,stroke:#3d3,stroke-width:3px
```

---

## Action Features (3 Dimensions)

Each feasible action is represented by a 3D feature vector:

| Feature | Description | Value Range | Example |
|---------|-------------|-------------|---------|
| **action_type** | Type of action | • `1/3` = Navigate to delivery<br/>• `2/3` = Navigate to charger<br/>• `3/3` = Charge at current location | `0.33` for delivery |
| **resulting_soc** | Battery % after action | `0.0 - 1.0` (normalized) | `0.65` = 65% battery |
| **charge_duration** | Charging time (if applicable) | `0.0 - 1.0` (normalized by max) | `0.5` for 1.0h charge |

```mermaid
graph LR
    A1[Navigate to Delivery 10] --> F1[0.33, 0.52, 0.0]
    A2[Navigate to Charger 2] --> F2[0.67, 0.71, 0.0]
    A3[Charge 1.0 hour] --> F3[1.00, 0.85, 0.5]
    
    F1 --> Description1[Type: delivery<br/>SOC after: 52%<br/>No charging]
    F2 --> Description2[Type: charger<br/>SOC after: 71%<br/>No charging]
    F3 --> Description3[Type: charge<br/>SOC after: 85%<br/>Duration: 1.0h]
    
    style A1 fill:#ffa,stroke:#da3,stroke-width:2px
    style A2 fill:#aaf,stroke:#33d,stroke-width:2px
    style A3 fill:#afa,stroke:#3d3,stroke-width:2px
```

---

## Variable Action Space Handling

### Problem: Action Space Changes Each Timestep

```mermaid
graph TD
    T1[Timestep 1<br/>5 feasible actions] --> Batch1[Action Graph<br/>5 nodes]
    T2[Timestep 2<br/>8 feasible actions] --> Batch2[Action Graph<br/>8 nodes]
    T3[Timestep 3<br/>3 feasible actions] --> Batch3[Action Graph<br/>3 nodes]
    
    Batch1 --> PTR[Pointer Array<br/>[0, 5, 13, 16]]
    Batch2 --> PTR
    Batch3 --> PTR
    
    PTR --> ActionGCN[Process all actions<br/>in single batch]
    
    style T1 fill:#ffa
    style T2 fill:#faa
    style T3 fill:#afa
    style PTR fill:#aaf,stroke:#33d,stroke-width:3px
```

**Solution: Pointer Array (`ptr`)**

- Concatenate all actions into single tensor
- Track boundaries with pointer array
- Process all actions in single GCN forward pass
- Extract per-timestep logits using pointers

**Example:**
```
Timestep 1: 5 actions → indices [0, 1, 2, 3, 4]
Timestep 2: 8 actions → indices [5, 6, 7, 8, 9, 10, 11, 12]
Timestep 3: 3 actions → indices [13, 14, 15]

ptr = [0, 5, 13, 16]
```

---

## PPO Training Loop

```mermaid
graph TD
    Start[Start Episode] --> Collect[Collect Rollout]
    
    Collect --> Act{For each step}
    Act --> Forward[Forward Pass:<br/>Actor + Critic]
    Forward --> Sample[Sample Action<br/>from π(a|s)]
    Sample --> EnvStep[Environment Step]
    EnvStep --> Store[Store in Buffer:<br/>s, a, r, log π(a|s), V(s)]
    Store --> Done{Episode done?}
    Done -->|No| Act
    Done -->|Yes| GAE[Compute GAE<br/>Returns & Advantages]
    
    GAE --> Update{PPO Epochs}
    
    Update --> Minibatch[Sample Minibatch]
    Minibatch --> NewForward[New Forward Pass]
    NewForward --> Ratio[Compute Ratio:<br/>π_new / π_old]
    
    Ratio --> ClipLoss[Clipped Policy Loss:<br/>min(ratio·A, clip(ratio)·A)]
    NewForward --> ValueLoss[Value Loss:<br/>MSE(V_new, returns)]
    NewForward --> Entropy[Entropy Bonus]
    
    ClipLoss --> TotalLoss[Total Loss:<br/>-policy + value - entropy]
    ValueLoss --> TotalLoss
    Entropy --> TotalLoss
    
    TotalLoss --> Backprop[Backpropagation<br/>+ Gradient Clipping]
    Backprop --> NextBatch{More batches?}
    
    NextBatch -->|Yes| Minibatch
    NextBatch -->|No| NextEpoch{More epochs?}
    
    NextEpoch -->|Yes| Update
    NextEpoch -->|No| ClearBuffer[Clear Buffer]
    ClearBuffer --> Start
    
    style Collect fill:#ddf
    style Update fill:#fdd
    style TotalLoss fill:#ffa,stroke:#da3,stroke-width:3px
```

### PPO Loss Components

**1. Clipped Policy Loss**
```
ratio = π_θ(a|s) / π_θ_old(a|s)
L_CLIP = -min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)
```

**2. Value Loss**
```
L_VF = (V_θ(s) - V_target)²
```

**3. Entropy Bonus**
```
L_ENT = -H[π_θ(·|s)]
```

**Total Loss:**
```
L = L_CLIP + c₁·L_VF - c₂·L_ENT
```

---

## Key Advantages

### 1. Variable Action Handling
✅ **No padding** - Only process feasible actions  
✅ **Dynamic sizing** - Handles 1 to N actions per step  
✅ **Efficient batching** - Process multiple timesteps together

### 2. Action Reasoning via GCN
✅ **Message passing** between actions  
✅ **Contextual scoring** - Actions influence each other  
✅ **Learned relationships** - GCN discovers action correlations

### 3. Shared Encoding
✅ **Single encoder** for both actor and critic  
✅ **Efficient computation** - Feature extraction once  
✅ **Better sample efficiency** - Shared representations

### 4. Scalability
✅ **Handles large action spaces** efficiently  
✅ **GPU-friendly** - Parallel GCN computation  
✅ **Flexible architecture** - Easy to modify

---

## Comparison: Fixed vs Variable Action Space

```mermaid
graph TB
    subgraph Fixed[Fixed Action Space PPO]
        F1[State] --> F2[GNN Encoder]
        F2 --> F3[Linear Layer<br/>→ N actions]
        F3 --> F4[Apply Mask<br/>-inf for infeasible]
        F4 --> F5[Softmax<br/>always N values]
    end
    
    subgraph Variable[Variable Action Space PPO]
        V1[State] --> V2[GNN Encoder]
        V2 --> V3[Action Graph Head<br/>only K feasible actions]
        V3 --> V4[Action GCN<br/>K action nodes]
        V4 --> V5[Softmax<br/>K values only]
    end
    
    Fixed --> Waste[❌ Wastes computation<br/>on infeasible actions]
    Variable --> Efficient[✅ Only processes<br/>feasible actions]
    
    style Fixed fill:#fdd
    style Variable fill:#dfd
    style Waste fill:#faa
    style Efficient fill:#afa
```

**Key Differences:**

| Aspect | Fixed Action Space | Variable Action Space |
|--------|-------------------|----------------------|
| **Output Size** | Always N logits | K logits (K varies) |
| **Masking** | Masking via -inf | No infeasible actions |
| **Computation** | Processes all N actions | Only K feasible actions |
| **Efficiency** | Lower (wastes compute) | Higher (adaptive) |
| **Action Reasoning** | Independent scoring | GCN message passing |

---

## Implementation Details

### Forward Pass Code Flow

```python
# 1. Encode state
embedding = encoder(hetero_data)  
# → [batch_size, hidden_dim * 3]

# 2. Build action graph
action_features, ptr = build_action_graph(
    states, feasible_indices
)
# → action_features: [total_actions, 3]
# → ptr: [batch_size + 1]

# 3. Action head forward
action_output = action_head(
    embedding, action_features, ptr
)
# → logits: [total_actions]

# 4. Value head forward
value = value_head(embedding)  
# → [batch_size]

# 5. Sample action for each timestep
for i in range(batch_size):
    start, end = ptr[i], ptr[i+1]
    logits_i = action_output.logits[start:end]
    dist = Categorical(logits=logits_i)
    action = dist.sample()
```

### Minibatch Processing

```mermaid
graph LR
    Buffer[Rollout Buffer<br/>T transitions] --> Sample[Sample Minibatch<br/>M transitions]
    
    Sample --> Batch[Batch HeteroData<br/>PyG batching]
    
    Batch --> Actions[Extract Action Features<br/>per transition]
    
    Actions --> PTR[Build Pointer Array<br/>track boundaries]
    
    PTR --> Forward[Forward Pass<br/>Actor + Critic]
    
    Forward --> Loss[Compute Losses]
    Loss --> Update[Update Parameters]
    
    style Buffer fill:#ddf
    style Batch fill:#ffd
    style Forward fill:#dfd
    style Update fill:#faa,stroke:#d33,stroke-width:2px
```

---

## Hyperparameters

### Network Architecture
- **hidden_dim**: 64 (GNN hidden dimension)
- **num_layers**: 3 (GNN layers)
- **mlp_dim**: 128 (Action/Value head dimension)
- **edge_dim**: 2 (Energy + Time features)

### PPO Training
- **lr**: 3e-4 (Learning rate)
- **gamma**: 0.99 (Discount factor)
- **gae_lambda**: 0.95 (GAE parameter)
- **clip_coef**: 0.2 (PPO clipping)
- **value_coef**: 0.5 (Value loss weight)
- **entropy_coef**: 0.01 (Entropy bonus)
- **ppo_epochs**: 10 (Update epochs per rollout)
- **minibatch_size**: 128 (Batch size)

---

## Summary

The PPO Variable Action GNN architecture provides:

✅ **Efficient handling** of variable-sized action spaces  
✅ **Action reasoning** via GCN message passing  
✅ **Shared encoder** for sample efficiency  
✅ **Scalable** to large and dynamic action spaces  
✅ **State-of-the-art** for EV routing with charging constraints  

**Result:** The agent learns to make intelligent routing and charging decisions by reasoning over the current state graph and the set of feasible actions simultaneously.
