# Variable-Action PPO + GNN Policy

This note documents the **variable-action PPO** agent (`algo/PPO_VariableActionGNN.py`). The agent consumes the heterogenous graph state built by `truck_env/state/gnn_state_space.py` and reasons over a dynamically defined action graph instead of a fixed-size discrete space.

---

## 1. State + Action Metadata from `GNNStateSpace`

Each call to `get_state_GNN(env)` produces a `torch_geometric.data.HeteroData` bundle with:

1. **Graph structure**
   - Node types: `truck`, `delivery`, `charger` (each has a minimal feature vector; see `descriptions/GNN_STATE_SUMMARY.md`).
   - Edge types: fully connected, type-aware edges with normalized `[energy, time]` features.
2. **Action metadata** (attached as attributes on the `HeteroData` object):

| Attribute | Meaning |
| --- | --- |
| `action_to_node_map` | Python list mapping action indices → `(node_id, is_charging)` tuples. |
| `feasible_action_mask` | Bool tensor denoting which indices can actually be executed (battery/path constraints). |
| `action_node_type`, `action_local_index` | Tie each action back to the node embedding it references. |
| `action_is_charging`, `action_charge_durations` | Indicate whether the action is routing or charging (and for how long). |
| `action_features` | Dense per-action vector `[action_type_norm, resulting_soc, charge_duration_norm]` produced by `_build_action_graph_features`. |
| `action_ptr` | Prefix-sum vector used to build per-batch fully connected action graphs. |
| `num_actions` | Torch scalar with the count of candidate actions for the active truck. |

This metadata allows the policy to batch states with different action counts while retaining alignment between logits and simulator actions.

---

## 2. Architecture Overview

```
EVENT STATE GRAPH (HeteroData)
   │
   ▼
GNNFeatureEncoder  ──> shared embedding h_s
   │
   ├─ Value head (MLP) ──> V(s)
   │
   └─ ActionGraphHead
        ├─ project action features + referenced node embeddings
        ├─ run 2× GCNConv over fully connected per-truck action graphs (edges built from ptr)
        └─ combine with h_s to obtain variable-length logits
```

- **Shared encoder:** Reuses `GNNFeatureEncoder` from the standard PPO agent (per-type linear projections → stacked `HeteroInteractionLayer`s → type-wise mean pooling).
- **ActionGraphHead:** Takes the shared embedding plus action features, builds a small fully connected graph per truck, applies two `GCNConv` layers, and dot-products the resulting action embeddings with the state embedding.
- **Masking:** The logits are filtered with `feasible_action_mask`; impossible actions get `-1e9`, ensuring the `Categorical` distribution samples only valid indices.
- **Value head:** Standard MLP that predicts the scalar value for PPO advantage estimation.

---

## 3. Tensor Dimensionalities

The following shapes use the defaults from `GNNStateSpace` and `train.py`. Adjust accordingly if you override hidden dimensions via CLI flags.

| Component | Symbol | Dimension |
| --- | --- | --- |
| Truck node features | `x_truck` | 13 (see `_get_truck_node_features`) |
| Delivery node features | `x_delivery` | 3 |
| Charger node features | `x_charger` | 4 |
| Edge features | `e_*` | 2 (`[energy_norm, time_norm]`) |
| Action feature vector | `f_action` | 3 (`[type_norm, resulting_soc, charge_duration_norm]`) |
| GNN hidden size | `hidden_dim` | CLI `--gnn-hidden-dim` (default 64) |
| Encoder output | `h_s` | `hidden_dim × (#node types)` = `hidden_dim × 3` → default 192 |
| Value head MLP | `Linear(h_s → mlp_dim)` + `Linear(mlp_dim → 1)` where `mlp_dim = --mlp-hidden-dim` (default 128) |
| Action head projections | `Linear(h_s → mlp_dim)` for state, `Linear(f_action → mlp_dim)` for actions |
| Action GCN layers | Two `GCNConv(mlp_dim → mlp_dim)` layers operating over each per-truck action graph |
| Output logits | Variable length = `num_actions` for the active truck |

These dimensions ensure the state embedding has enough capacity to attend to multiple node types, while the action head stays lightweight enough to run per decision.

---

## 3. Rollout & Training Flow (Variable Action Specifics)

```
ROLLOUT
-------
1. Env step → GNNStateSpace builds HeteroData (graph + action metadata).
2. `policy.select_action(state)`:
      • batches PyG data (respecting VARIABLE_BATCH_EXCLUDE_KEYS)
      • runs encoder + action head to get masked logits
      • samples an action index and stores logπ, value, mask
3. Env executes `policy.to_env_action(...)` which maps the index back to `(node_id, charge_hours, is_charging)`.
4. `VariableRolloutBuffer.add` saves (state, action_idx, reward, value, done, mask, logπ).
5. Repeat until `ppo-steps-per-update` transitions are collected (episodes may span multiple rollouts).

UPDATE
------
1. Compute returns + GAE advantages using saved rewards/values (`VariableRolloutBuffer.compute_returns_and_advantages`).
2. For `ppo-epochs`:
      a. Sample minibatches from the stored PyG graphs (batched via `Batch.from_data_list`).
      b. Recompute logits/values → obtain new logπ.
      c. ratio = exp(logπ_new - logπ_old); apply PPO clipping (`ppo-clip`).
      d. Loss = actor_loss + c1 * critic_loss - c2 * entropy, where `c2=entropy_coef`.
      e. Optimize network parameters (shared encoder + action head + value head).
3. Clear the buffer and continue collecting new rollouts.
```

Important differences from the fixed-action PPO:
- The action dimension changes per state, so everything is keyed by `action_ptr` and `num_actions`.
- No explicit `compute_action_mask` call inside `train_ppo_variable`; feasibility is encoded in the state object itself.
- `to_env_action` enforces a guard against stale metadata by raising if `action_to_node_map` is missing or truncated.

---

## 4. Inside the Action Graph Head

For one batch element with `N` feasible actions (varies per truck), the operations are:

1. **Project inputs**
   - State embedding: `h_s ∈ ℝ^{encoder_dim}` → `z_s = ReLU(W_s h_s)` where `W_s ∈ ℝ^{mlp_dim × encoder_dim}`.
   - Action features: each `f_i ∈ ℝ^3` → `z_i = ReLU(W_a f_i)` (`W_a ∈ ℝ^{mlp_dim × 3}`).
2. **Build fully connected action graph**
   - Using the prefix sums `ptr`, create an edge list that connects every pair of actions belonging to the same truck (bidirectional edges). This graph expresses interactions such as “two charger options compete” or “routing vs. charging alternatives.”
3. **Message passing**
   - Apply two stacked `GCNConv(mlp_dim → mlp_dim)` layers to propagate information across the action nodes. Each layer updates `z_i` using neighboring actions, letting the policy reason about relative desirability.
4. **State-action fusion**
   - For each truck, take the processed action embeddings `z_i` and compute logits via dot product with the state projection: `logit_i = z_i ⋅ z_s`.
   - The result is a length-`N` vector (different per truck).
5. **Masking**
   - Apply `feasible_action_mask` to set invalid logits to `-1e9`, ensuring the softmax/Categorical distribution only samples executable moves.

Because the head is graph-based, its computational cost scales with the number of candidate actions rather than a fixed action dimension.

---

## 4. Integration inside `train.py`

```
CLI args (with --algo ppo-variable)
      ↓
train_ppo_variable(...)
      ↓
EventDrivenTruckEnv + GNNStateSpace
      ↓
PPOVariableActionGNN policy
      ↓
Training loop
      ├─ Data collection via select_action()/to_env_action()
      ├─ Policy updates every ppo-steps-per-update
      ├─ Evaluation runs evaluate_policy() (also calls to_env_action for deployment)
      └─ Checkpoints best + final models, logs to W&B/console
```

- Evaluation uses greedy selection (`expl_noise=0`) but still leverages the variable action metadata to map indices to environment commands.
- WandB logging captures success rate, reward, per-truck statistics, and any new metrics (e.g., total truck time) independent of the action-space size.

---

## 5. Hyperparameters & Tunable Components

| Component | Flags / Sources | Notes |
| --- | --- | --- |
| **GNN encoder** | `--gnn-hidden-dim`, `--actor-gcn-layers`, `--critic-gcn-layers` | Controls feature dimensionality and depth. |
| **Action head** | `--mlp-hidden-dim` (projection width), internal `GCNConv` layers in `ActionGraphHead` | You can extend the head (add layers, change nonlinearity) directly in `PPO_VariableActionGNN.py`. |
| **PPO core** | `--ppo-steps-per-update`, `--ppo-epochs`, `--ppo-minibatch-size`, `--ppo-clip`, `--ppo-entropy-coef`, `--lr` | Same semantics as vanilla PPO. |
| **Environment** | `truck_env/config_files/config.yaml`, runtime flags (`--num-trucks`, `--num-stops`, `--enable-traffic`) | Affects graph size, number of feasible actions, and reward landscape. |

Always tune encoder capacity in tandem with environment complexity. Large numbers of trucks/deliveries imply many candidate actions, so increasing `hidden_dim` or the action-head width can help the policy disambiguate them.

---

## 6. File Map & Further Reading

- `algo/PPO_VariableActionGNN.py`
  - `PPOVariableActionGNN`: orchestrates batching, action sampling, PPO updates.
  - `ActionGraphHead`: implements the per-action GCN logic.
  - `VariableRolloutBuffer`: stores variable-action rollouts and computes GAE.
- `truck_env/state/gnn_state_space.py`
  - `_build_action_graph_features`, `action_to_node_map` construction, and metadata alignment.
  - `graph_to_numpy` helper for debugging state tensors.
- `train.py`
  - `train_ppo_variable` and evaluation hooks.
- Additional references: `descriptions/GNN_STATE_SUMMARY.md` (full node/edge feature list) and `descriptions/HETERO_GNN_ARCHITECTURE.md` (global view of the heterogenous graph encoder).

Use this document as the definitive description of the variable-action PPO workflow when modifying the agent or onboarding new contributors.
