# Prioritized Execution TODO

Updated: 2026-08-11

This is the operational checklist. The broader experiment matrix remains in `01_experimental_campaign.md`; acceptance criteria remain in `02_acceptance_and_reproducibility.md`.

## Completed foundation

- [x] Create `campaign/ieee-tits-revision` from a clean `Restructuring` baseline.
- [x] Record approval of D1–D10 and mark the recommended choices authoritative.
- [x] Add stable scenario IDs and isolated RNG streams for instance, travel, energy, and service randomness.
- [x] Add same-seed replay, different-seed variation, and policy-RNG-isolation tests.
- [x] Sample unloading time once per service event.
- [x] Add fleet-owned `CustomerTask` and atomic task-registry lifecycle.
- [x] Add truck payload capacity, remaining payload, and served-task accounting.
- [x] Add joint instance generation with a common depot, unique customers, static battery reachability, and fleet-capacity-feasible demand.
- [x] Prevent joint customer completion at arrival; commit it after unloading.
- [x] Exercise one deterministic two-truck/four-customer episode with exactly-once service, nonnegative payload, and depot return.
- [x] Pass the current correctness suite: 24 tests.

The full Gate A is not yet passed: the complete-episode check is one deterministic integration case, not a randomized/property-based invariant campaign.

## P0 — Finish model correctness before training

- [ ] Replace compatibility views based on per-truck delivery sequences with one canonical fleet feature model.
- [ ] Build a pure hard-feasibility engine with rejection reason codes.
- [ ] Remove primary top-k pruning, safety relaxation, and forced fallback behavior.
- [ ] Detect and record `no feasible continuation` explicitly.
- [ ] Add randomized every-customer-once, capacity, battery, event-order, and depot-return invariant tests.
- [ ] Add a controlled time-window variant without enabling it in every base instance.
- [ ] Define whether depot return allows payload reload; base campaign should use one-tour CVRP semantics unless explicitly changed.
- [ ] Validate stochastic distribution moments, clipping, time-of-day effects, and travel-energy correlation.
- [ ] Version and persist the full scenario descriptor in evaluation artifacts.

## P0 — Charging and queue semantics

- [ ] Replace duration actions with target SoC 50/60/70/80/90/100%.
- [ ] Use the same nonlinear curve integrator for simulation, masks, heuristic, and optimization models.
- [ ] Preserve station types instead of mapping every station to `DCFast`.
- [ ] Add 150/350/750 kW station classes plus the 50 kW legacy stress case.
- [ ] Preserve station-specific efficiency and port counts from data/configuration.
- [ ] Make joining a full station's FCFS queue a valid first-class action.
- [ ] Add queue-order, simultaneous-arrival, port-capacity, wake-up, and station-closure tests.

## P0 — Fair observations and actions

- [ ] Introduce typed canonical truck, customer, charger, route, and global features.
- [ ] Generate flat, DeepSets, and graph observations from the same features.
- [ ] Verify semantic parity numerically across encoders.
- [ ] Stabilize variable customer/action sizes with explicit masks and schema versions.
- [ ] Implement independent, complete-GCN, and self-attention action-head interfaces.
- [ ] Add action-order permutation-equivariance and cross-batch isolation tests.
- [ ] Select the action head on validation configurations only and freeze it before test runs.

## P0 — Objectives, metrics, and provenance

- [ ] Create a terminal-outcome schema with success and explicit failure causes.
- [ ] Compute feasibility and fleet makespan independently of shaped reward.
- [ ] Add total time, travel, service, charging, queue, energy, reserve, vehicles used, tail risk, and inference runtime.
- [ ] Retain failures in every aggregate and avoid method-dependent scenario exclusions.
- [ ] Add Wilson intervals for success and paired bootstrap intervals/effect sizes for continuous metrics.
- [ ] Create immutable campaign manifests with git/worktree, dependency, hardware, seed, configuration, checkpoint, and command provenance.
- [ ] Enforce disjoint train/validation/test seed namespaces.

## P1 — Baselines and validation

- [ ] Exact or bounded fleet optimization for tiny deterministic instances.
- [ ] Exhaustive-enumeration cross-check for the tiniest cases.
- [ ] Rolling-horizon/scenario optimization for small stochastic instances.
- [ ] Strong ALNS/routing-and-charging heuristic.
- [ ] Constructive attention/transformer policy.
- [ ] PPO, MaskPPO, DeepSets-PPO, state-GNN PPO, and GraphPPO under equivalent information.
- [ ] Rename and instrument the inherited per-truck deterministic MILP.
- [ ] Publish solver status/gap/runtime and every retry/fallback.

## P1 — Smoke campaign before expensive runs

- [ ] Resolve a GPU-capable training environment; the current host exposes no working NVIDIA driver.
- [ ] Run one short train/evaluate cycle for every learning implementation.
- [ ] Verify manifests, checkpoints, validation-only selection, raw episode rows, and aggregate metrics.
- [ ] Run tiny deterministic comparisons against exact bounds.
- [ ] Freeze campaign configurations and artifact schemas.

## P2 — Main evidence campaign

- [ ] Run the XS–L2 scale grid from `01_experimental_campaign.md`.
- [ ] Use ten independent training seeds for headline learning methods.
- [ ] Use at least 500 paired test scenarios per main setting; increase to 1,000 for rare failures or small success gaps.
- [ ] Run the complete component-ablation matrix.
- [ ] Run charging/congestion, uncertainty/robustness, and optimizer-budget sensitivities.
- [ ] Run interpolation and genuine OOD generalization campaigns.
- [ ] Preserve negative and incomplete runs with documented status.

## P3 — Manuscript and response artifacts

- [ ] Freeze claims after final results; do not retrofit experiments to desired wording.
- [ ] Rewrite title, abstract, introduction, contributions, formulation, architecture, and results from the tested implementation.
- [ ] Replace normalized-reward headline tables with feasibility-first operational tables.
- [ ] Generate every LaTeX table/figure from versioned result artifacts.
- [ ] Complete `04_reviewer_response_todo.md` with exact page/line and artifact links.
- [ ] Add formulation, notation, optimizer, heuristic, random-variable, and reproducibility appendices.
- [ ] Compile LaTeX, check citations/references, lint, and archive final artifacts.

## Stop/go gates

- [ ] **Go to training:** all Campaign 0 correctness and smoke checks pass.
- [ ] **Go to headline test:** architecture and checkpoints are frozen using validation only.
- [ ] **Go to manuscript rewrite:** primary, ablation, robustness, and generalization result manifests are complete.
- [ ] **Go to resubmission:** Gates A–F and every reviewer-response item have traceable evidence.
