# File-Level Implementation Map

This map prepares the approved work without selecting an unvalidated modeling option. Items labeled `D#` depend on the corresponding decision in `00_modeling_validation.md`.

## Architectural observation

The inherited code stores a delivery sequence inside each `Truck`. Even flexible mode chooses among the remaining nodes in that truck's preassigned sequence. A genuine joint-assignment formulation therefore requires a fleet-level customer/task registry; it cannot be implemented correctly by merely enabling the current flexible-order flag.

## Phase 1 — Foundation independent of D1–D10

These corrections are required under every reasonable modeling choice.

### Scenario and RNG subsystem

Files:

- `EVRoutingEnv/models/simulation/traffic_simulation.py`
- `EVRoutingEnv/models/simulation/delivery_simulator.py`
- `EVRoutingEnv/models/environment/event_driven_env.py`
- new `EVRoutingEnv/models/simulation/scenario.py`

Changes:

- introduce an immutable scenario identifier and episode-scoped RNG streams;
- derive independent travel, energy, service, and instance-generation streams from `SeedSequence`;
- remove hash-only pseudo-random generation that omits episode identity;
- make common random numbers independent of policy-internal RNG consumption;
- save scenario metadata in episode `info` and evaluation artifacts.

Tests:

- same-scenario replay;
- different-scenario variation;
- policy-independent exogenous draws;
- distribution moment/clipping checks.

### Evaluation objectives and outcome schema

Files:

- `EVRoutingEnv/models/environment/event_driven_env.py`
- `EVRoutingEnv/utils/statistics.py`
- `scripts/evaluation/eval_parallel_policies.py`
- `scripts/evaluation/generalization_eval.py`
- new `scripts/evaluation/metrics.py`

Changes:

- separate shaped training reward from operational evaluation metrics;
- define a structured terminal outcome with success and failure cause;
- retain all scenarios in aggregation;
- add full-service, conditional time, queue, energy, tail-risk, and runtime metrics;
- replace hardcoded policy registries with configuration-driven evaluation manifests.

### Run configuration and provenance

Files:

- `scripts/training/train_PPO_Variable_parallel.py`
- `scripts/training/train_sb3_event_driven.py`
- `scripts/runners/runner_train_ppo-variable.py`
- new `scripts/campaign/`
- new `campaign_configs/`

Changes:

- resolve every run from immutable YAML/JSON configuration;
- store git commit, worktree state, dependencies, hardware, seeds, and command;
- enforce disjoint train/validation/test seed namespaces;
- make checkpoint selection deterministic and validation-only;
- generate, rather than hand-edit, campaign commands.

## Phase 2 — Problem model

### Fleet-level task ownership (`D1`, `D2`)

Current ownership points:

- `EVRoutingEnv/models/core/truck.py` stores `delivery_sequence` and delivered nodes per truck;
- `EVRoutingEnv/models/environment/loaders.py` independently generates a feasible sequence per truck;
- `EVRoutingEnv/models/environment/event_driven_env.py` maps delivery actions through the active truck's sequence;
- state builders infer remaining deliveries from those per-truck sequences.

Required joint-routing design if D1 recommended option is approved:

- add a `CustomerTask` data model with node, demand, service time, optional time window, and service state;
- add a fleet-level task registry owned by the environment;
- store vehicle capacity/load separately from task ownership;
- let the active truck select any eligible unserved task;
- atomically claim/service tasks so no customer is served twice;
- mark a truck complete only after its assigned work is done and it returns to depot;
- keep a compatibility adapter for the preassigned-route execution benchmark.

Likely files:

- new `EVRoutingEnv/models/core/customer.py`
- `EVRoutingEnv/models/core/truck.py`
- `EVRoutingEnv/models/environment/loaders.py`
- `EVRoutingEnv/models/environment/event_driven_env.py`
- `EVRoutingEnv/models/environment/event_handlers.py`
- all state-space implementations
- all routing baselines.

### Charging actions (`D6`)

Current duration-action points:

- `EVRoutingEnv/config_files/config.yaml`
- `EVRoutingEnv/models/environment/event_driven_env.py`
- `EVRoutingEnv/models/simulation/charging_curve.py`
- `EVRoutingEnv/state/gnn_state_space.py`
- `EVRoutingEnv/state/gnn_state_space_detour.py`
- `EVRoutingEnv/state/gnn_state_space_vrp.py`
- optimizer and heuristic policies.

If target-SoC is approved:

- represent charge actions by target SoC;
- compute duration and energy through the same charging-curve integrator used by simulation;
- expose target SoC, expected duration, and expected energy as action features;
- mask targets at or below current SoC;
- ensure optimizer and heuristics use equivalent charging physics.

### Queue semantics (`D7`)

Files:

- `EVRoutingEnv/models/simulation/charging_station.py`
- `EVRoutingEnv/models/environment/event_driven_env.py`
- `EVRoutingEnv/models/environment/event_handlers.py`
- state-space and mask implementations.

Changes:

- define queue joining as a first-class state transition if approved;
- expose only currently observable queue workload;
- invariant-test FCFS order and port capacity;
- ensure the mathematical description matches actual admission and waiting behavior.

## Phase 3 — State, mask, and policy

### One canonical observation specification

Problem:

- `state_space.py`, `gnn_state_space.py`, `gnn_state_space_detour.py`, and `gnn_state_space_vrp.py` independently encode related but non-equivalent information.

Plan:

- introduce canonical typed feature extraction before representation-specific encoding;
- build flat, DeepSets, and graph observations from the same typed feature tensors;
- include every customer/task needed by the decision problem;
- add schema/version metadata and shape tests;
- test numerical equality of shared semantic features across encoders.

Likely new module:

- `EVRoutingEnv/state/features.py`

### Hard feasibility engine (`D2`, `D8`)

Problem:

- feasibility logic is distributed across state builders and contains pruning, loop guards, safety relaxation, and fallback behavior.

Plan:

- move hard feasibility rules into a pure, testable engine;
- return action validity plus a reason code for every rejected candidate;
- separate physical validity from optional candidate pruning;
- remove hidden mask mutation and forced action enabling;
- treat `no feasible continuation` as an explicit terminal outcome.

Likely files:

- replace/expand `EVRoutingEnv/state/action_mask.py`
- simplify all GNN state builders;
- add `EVRoutingEnv/state/feasibility.py`.

### Action heads (`D9`)

Files:

- `algo/PPO_VariableActionGNN.py`
- `algo/PPO_actionGNN.py`
- `algo/networks.py`

Plan:

- factor a shared action-head interface;
- implement independent, complete-GCN, and self-attention heads;
- preserve ragged batching through `ptr`;
- prevent cross-instance edges/attention;
- add action permutation-equivariance tests;
- select the primary head using validation configurations only.

## Phase 4 — Baselines

### Optimization

Files:

- `EVRoutingEnv/baselines/optimal_gurobi.py`
- `EVRoutingEnv/baselines/optimal_gurobi_simple.py`
- `EVRoutingEnv/baselines/optimal_vrp_single_truck.py`

Plan:

- rename inherited policies to describe their actual assumptions;
- retain them as legacy references;
- add a fleet-level exact/bounded model for tiny deterministic instances;
- add solver-status and fallback instrumentation;
- optionally add rolling-horizon/scenario optimization after the deterministic formulation is validated.

### Heuristics and learning baselines

Files:

- `EVRoutingEnv/baselines/heuristic_policy.py`
- `EVRoutingEnv/baselines/classic_vrp_heuristics.py`
- `scripts/training/train_sb3_event_driven.py`
- new baseline modules as approved by D10.

Plan:

- implement or adapt a strong ALNS/routing-and-charging heuristic;
- add DeepSets and attention/transformer policy baselines using the canonical observation;
- give all learning methods equal observable information and candidate semantics;
- allocate tuning budgets prospectively.

## Phase 5 — Test structure

Proposed test tree:

```text
tests/
  unit/
    test_scenario_rng.py
    test_charging_curve.py
    test_queue_fcfs.py
    test_task_registry.py
    test_feasibility.py
    test_feature_parity.py
    test_action_permutation.py
  integration/
    test_tiny_known_instance.py
    test_every_customer_once.py
    test_depot_return.py
    test_policy_common_scenarios.py
    test_evaluation_no_exclusions.py
  smoke/
    test_train_graphppo.py
    test_train_maskppo.py
    test_campaign_manifest.py
```

## Dependency order after validation

1. scenario/RNG and outcome schema;
2. approved fleet/task and charging models;
3. canonical features;
4. feasibility engine;
5. policy action heads;
6. baseline parity;
7. exact tiny-instance validation;
8. campaign runner and manifests;
9. smoke training;
10. architecture selection and full experiments.

This order prevents spending compute on data generated by an invalid simulator or on baselines that receive incomparable information.

