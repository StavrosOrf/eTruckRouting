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
- [x] Add a primary hard-feasibility engine with stable rejection reason codes and no graph-mask dependency.
- [x] Add explicit invalid-action, empty-feasible-set, and payload-deadlock outcomes.
- [x] Implement exact target-SoC actions at 50/60/70/80/90/100% for joint routing.
- [x] Preserve Level2/DCFast source classes and assign configurable 150/350/750 kW joint-fleet station powers.
- [x] Harden FCFS queues against duplicate wake-ups, stale waiters, port overbooking, duplicate charging, and cross-station double occupancy.
- [x] Add feasibility-first per-episode operational metrics independent of shaped reward.
- [x] Replace the primary joint flat observation with a versioned canonical fleet representation and lossless padded-set/graph adapters.
- [x] Add the disabled-by-default hard-time-window variant with early waiting and explicit late failure.
- [x] Replace the adjacent-action chain with selectable independent, genuinely complete-GCN, and self-attention action heads.
- [x] Make the PPO hard-mask path fail explicitly on empty, disjoint, or length-mismatched masks instead of relaxing constraints.
- [x] Add a failure-retaining campaign runner that verifies manifest seed/config identity and publishes immutable raw rows and aggregates.
- [x] Pass the current correctness suite: 219 tests; pass targeted Ruff, coverage, and warning-free Gymnasium contract checks.

The core Gate A battery accounting now has randomized mixed travel/charge conservation checks and 20 seeded complete-episode checks. Broader generated mixed-event property testing remains desirable before expensive training.

## P0 — Finish model correctness before training

- [x] Replace the primary joint observation with one canonical fleet feature model; retain legacy views only for the secondary preassigned benchmark.
- [x] Build a pure hard-feasibility engine with rejection reason codes.
- [x] Remove primary top-k pruning, safety relaxation, and forced fallback behavior.
- [x] Detect and record `no feasible continuation` explicitly.
- [x] Add randomized every-customer-once, capacity, event-order, and depot-return regression tests; broader property testing of energy/charging combinations remains.
- [x] Add a controlled time-window variant without enabling it in every base instance.
- [x] Use one-tour CVRP semantics in the base campaign: no payload reload occurs at the depot.
- [x] Validate stochastic clipping, approximate means, time-of-day variance effects, and travel-energy correlation with deterministic Monte Carlo tests.
- [x] Version and expose the full scenario descriptor in episode information; persist it from every runner remains open.

## P0 — Charging and queue semantics

- [x] Replace duration actions with target SoC 50/60/70/80/90/100% in the primary joint model.
- [ ] Use the same nonlinear curve integrator for simulation, masks, heuristic, and optimization models.
- [x] Preserve station types instead of mapping every station to `DCFast`.
- [x] Add configurable 150/350/750 kW station classes while retaining legacy type-rate configuration.
- [ ] Preserve station-specific efficiency and port counts from data/configuration.
- [x] Make joining a full station's FCFS queue a valid routing choice.
- [x] Queue order, port capacity, duplicate wake-up, stale-waiter, full-environment one-port handoff, and station-closure scenarios pass.

## P0 — Fair observations and actions

- [x] Introduce typed canonical truck, customer, charger, route/action, edge, and global features.
- [x] Generate flat, padded DeepSets-ready, and complete heterogeneous graph observations from the same node/action/global features.
- [x] Verify numerical parity for every shared semantic row across representation adapters.
- [ ] Close the remaining information-parity gap before baseline training: graph observations currently add pairwise energy/time/reachability edges that flat and padded-set observations do not receive. Either expose an equivalent padded pairwise tensor to all policies or isolate edge information as an explicit ablation.
- [x] Stabilize variable customer/action sizes with overflow rejection, explicit padding masks, and schema versions.
- [x] Implement independent, complete-GCN, and self-attention action-head interfaces with identical state/action inputs.
- [x] Add action-order permutation-equivariance, complete-edge, singleton/empty-set, finite-gradient, and cross-batch isolation tests.
- [x] Expose action-head selection in the PPO training CLI and persist the selection and head hyperparameters in checkpoint configuration.
- [x] Smoke-test every head through a real seeded environment, heterogeneous state encoder, and hard feasible-action mask on CPU.
- [ ] Select the action head on validation configurations only and freeze it before test runs.

## P0 — Objectives, metrics, and provenance

- [x] Create versioned per-episode success, explicit failure-cause, and artifact schemas.
- [x] Compute feasibility and conditional fleet makespan independently of shaped reward.
- [x] Add per-episode total operating, travel, service, charging, queue, time-window wait, energy, terminal-SoC, vehicles-used, and inference-time metrics; broader tail-risk reporting remains campaign-table work.
- [x] Add a failure-retaining aggregate primitive with explicit success conditioning and missing counts and wire it into the generic runner.
- [x] Add Wilson intervals and deterministic paired bootstrap intervals/effect sizes; campaign tables must consume these primitives.
- [x] Create strict-JSON immutable manifest primitives with git/worktree, dependency, hardware, seed, configuration, checkpoint, and command provenance.
- [x] Define and test provably disjoint train/validation/test seed namespaces.
- [x] Add a generic evaluation runner that writes the manifest before execution, preserves full scenario descriptors and diagnostic rewards, records inference time, retains failed episodes, and leaves an explicit failure artifact for interrupted runs.
- [ ] Migrate every inherited evaluation/training entry point to the generic artifact contract.

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

- [x] Install and import-check Torch, PyTorch Geometric, Stable-Baselines3, sb3-contrib, and W&B in `.venv`.
- [ ] Resolve a GPU-capable training environment; the installed Torch build supports CUDA but this host exposes no working GPU.
- [ ] Run one short train/evaluate cycle for every learning implementation.
- [x] Integration-test manifest, scenario descriptor, raw row, aggregate summary, inference timing, overwrite refusal, and interrupted-run artifacts in the generic evaluation runner.
- [ ] Verify checkpoints and validation-only selection in short train/evaluate runs, and migrate legacy runners.
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
