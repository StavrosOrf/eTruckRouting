# Prioritized Execution TODO

Updated: 2026-08-17

This is the operational checklist. The broader experiment matrix remains in `01_experimental_campaign.md`; acceptance criteria remain in `02_acceptance_and_reproducibility.md`. Reviewer-facing obligations remain in `04_reviewer_response_todo.md`.

`[x]` complete with cited evidence; `[~]` partially satisfied, with the missing part named on the following line; `[ ]` not started.

## Where this stands

The correctness phase (P0) and the primary evidence campaign are complete; the supporting-evidence phase (P1/P2 ablations, baselines, generalization) and the whole manuscript phase (P3) are not. The current headline, from `10_travel_time_objective_campaign.md`: GraphPPO `c_tm15`, trained from random initialisation with no imitation, reaches **119.9 fleet travel hours at 0.857 success** on 300 held-out test scenarios, against 120.5 h / 0.607 for the tuned CP-SAT planner, and holds the best mean and median ratio against the best-known travel reference.

Verification state at `a11f92c`: **281 tests pass** in ~64 s.

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
- [x] Pass the current correctness suite: 281 tests; pass targeted Ruff, coverage, and warning-free Gymnasium contract checks.

The core Gate A battery accounting has randomized mixed travel/charge conservation checks and 20 seeded complete-episode checks. Broader generated mixed-event property testing remains desirable but is no longer gating, since the expensive training runs have been executed and audited.

## Completed learning and campaign phase

- [x] Close the pairwise information-parity gap so every encoder receives the same semantic content (doc 08 §1).
- [x] Tune every non-learning baseline on validation only, then re-tune all three for the travel-time objective including CP-SAT's own nominal objective (`frozen_baselines_travel.json`, doc 10 §3).
- [x] Make the baselines plan with the simulator's own nonlinear charging physics (doc 08 §3).
- [x] Run the encoder x head architecture sweep at a shared interaction budget and freeze on validation with a disjoint re-score (`selected_architecture.json`, doc 08 §9).
- [x] Demonstrate GraphPPO trained from random initialisation with no imitation, via the energy-ramp curriculum (doc 09).
- [x] Add travel time, distance, and charging stops as first-class harness metrics and make `selection_score` objective-aware (doc 10 §2.1).
- [x] Add `ROUTING_ACTION_FEATURES` and prove their effect with a matched-budget, matched-width, matched-seed ablation (`v2_tm10` 0.700 vs `v2_ablate` 0.213, doc 10 §6.2).
- [x] Parallelise rollouts with `WorkerCanonicalVecEnv` and prove step-for-step equivalence against the synchronous implementation (`test_worker_vec_env_matches_the_synchronous_one`).
- [x] Run the four-stage travel-time ladder (A–D, 16 runs) and select under a pre-declared banded rule on 150 validation scenarios disjoint from those used in training (`selected_travel_d.json`, doc 10 §6.4).
- [x] Score the frozen method set on the 300-scenario test split with Wilson intervals, paired bootstrap differences, and a best-known optimality reference (`campaign_travel_final/test/`).
- [x] Preserve negative and stalled runs with documented status (`v2_tm20` collapse, the two selection defects found and fixed in doc 10 §6.4, the makespan-era policy carried in for comparison).

## P0 — Finish model correctness before training

- [x] Replace the primary joint observation with one canonical fleet feature model; retain legacy views only for the secondary preassigned benchmark.
- [x] Build a pure hard-feasibility engine with rejection reason codes.
- [x] Remove primary top-k pruning, safety relaxation, and forced fallback behavior.
- [x] Detect and record `no feasible continuation` explicitly.
- [x] Add randomized every-customer-once, capacity, event-order, and depot-return regression tests; broader property testing of energy/charging combinations remains.
- [x] Add a controlled time-window variant without enabling it in every base instance.
- [x] Use one-tour CVRP semantics in the base campaign: no payload reload occurs at the depot.
- [x] Validate stochastic clipping, approximate means, time-of-day variance effects, and travel-energy correlation with deterministic Monte Carlo tests.
- [x] Version and expose the full scenario descriptor in episode information, and persist it from the canonical evaluation runner.
  - Legacy runners still do not persist it; see the migration item under P1.

## P0 — Charging and queue semantics

- [x] Replace duration actions with target SoC 50/60/70/80/90/100% in the primary joint model.
- [x] Use the same nonlinear curve integrator for simulation, masks, heuristic, and optimization models.
  - Closed for the canonical baselines: `nominal_charge_hours` prices every recharge with `ChargingCurveModel.calculate_charge_to_target`, the routine the simulator executes, with the station's own power and the global realistic-curve flag injected as the environment injects them (doc 08 §3).
- [x] Preserve station types instead of mapping every station to `DCFast`.
- [x] Add configurable 150/350/750 kW station classes while retaining legacy type-rate configuration.
- [ ] Preserve station-specific efficiency and port counts from data/configuration.
- [x] Make joining a full station's FCFS queue a valid routing choice.
- [x] Queue order, port capacity, duplicate wake-up, stale-waiter, full-environment one-port handoff, and station-closure scenarios pass.
- [ ] Fix or document `ChargingCurveModel.estimate_charge_time`, which does not converge at a target of 1.0 and silently returns its search midpoint of 10 h.

## P0 — Fair observations and actions

- [x] Introduce typed canonical truck, customer, charger, route/action, edge, and global features.
- [x] Generate flat, padded DeepSets-ready, and complete heterogeneous graph observations from the same node/action/global features.
- [x] Verify numerical parity for every shared semantic row across representation adapters.
- [x] Close the remaining information-parity gap before baseline training.
  - Closed in doc 08 §1: the pairwise energy/time/reachability content is now owned canonically and exposed to every encoder, so the flat and padded-set policies are not scored against a graph policy that sees more.
- [x] Stabilize variable customer/action sizes with overflow rejection, explicit padding masks, and schema versions.
- [x] Implement independent, complete-GCN, and self-attention action-head interfaces with identical state/action inputs.
- [x] Add action-order permutation-equivariance, complete-edge, singleton/empty-set, finite-gradient, and cross-batch isolation tests.
- [x] Expose action-head selection in the PPO training CLI and persist the selection and head hyperparameters in checkpoint configuration.
- [x] Smoke-test every head through a real seeded environment, heterogeneous state encoder, and hard feasible-action mask on CPU.
- [~] Select the action head on validation configurations only and freeze it before test runs.
  - Done for the makespan-era headline (`selected_architecture.json`: validation-only ranking at the largest shared budget, finalists re-scored on 150 disjoint validation scenarios). The travel-time model inherits `hetero_graph + complete_gcn` from that sweep rather than re-selecting under the new objective, schema, and routing features. Either re-run the sweep under the current objective or state the inheritance explicitly as a limitation.

## P0 — Objectives, metrics, and provenance

- [x] Create versioned per-episode success, explicit failure-cause, and artifact schemas.
- [x] Compute feasibility and conditional fleet makespan independently of shaped reward.
- [x] Add per-episode total operating, travel, service, charging, queue, time-window wait, energy, terminal-SoC, vehicles-used, and inference-time metrics.
- [x] Add per-episode travel time, distance, and charging-session metrics, and make checkpoint selection objective-aware.
- [x] Add a failure-retaining aggregate primitive with explicit success conditioning and missing counts and wire it into the generic runner.
- [x] Add Wilson intervals and deterministic paired bootstrap intervals/effect sizes, and consume them in the campaign tables.
- [x] Create strict-JSON immutable manifest primitives with git/worktree, dependency, hardware, seed, configuration, checkpoint, and command provenance.
- [x] Define and test provably disjoint train/validation/test seed namespaces.
- [x] Add a generic evaluation runner that writes the manifest before execution, preserves full scenario descriptors and diagnostic rewards, records inference time, retains failed episodes, and leaves an explicit failure artifact for interrupted runs.
- [~] Classify every retained failure.
  - Retention is complete; 17 of GraphPPO's 43 test-split failures still fall through to `unspecified_failure` because no termination reason was set (`EVRoutingEnv/evaluation/statistics.py:53`). R1.5 asks for the cause, so these need a real label.
- [ ] Migrate every inherited evaluation/training entry point to the generic artifact contract.
  - Still on the old contract: `scripts/evaluation/generalization_eval.py`, `eval_policies.py`, `eval_parallel_policies.py`, `eval_parallel_by_size.py`, and the `train_sb3_event_driven.py` / `runner_train_ppo-variable.py` training entry points.

## P1 — Baselines and validation

- [~] Exact or bounded fleet optimization for tiny deterministic instances.
  - CP-SAT planner and optimality reference implemented and scored; no instance proved optimal at an ~85% bound gap (doc 09 §5), so no valid optimality claim is available yet. A stronger formulation — branch-and-price, or a labelling DP over customer subsets and battery levels — is what would make optimality measurable.
- [ ] Exhaustive-enumeration cross-check for the tiniest cases.
- [x] Rolling-horizon/scenario optimization for small stochastic instances.
  - `RollingHorizonMPCPolicy`, re-tuned for the travel objective and scored on the test split.
- [ ] Strong ALNS/routing-and-charging heuristic.
- [ ] Constructive attention/transformer policy.
- [~] PPO, MaskPPO, DeepSets-PPO, state-GNN PPO, and GraphPPO under equivalent information.
  - GraphPPO is complete. Pure PPO without demonstrations was measured (`results/canonical/pure_ppo`, reaching 0 success, which is what motivated the curriculum). Missing entirely: **PPO without the feasibility mask** — there is no such option in `train_canonical_ppo.py` — plus MaskPPO under identical candidate semantics, DeepSets-PPO, and state-GNN PPO with independent scoring.
- [ ] Rename and instrument the inherited per-truck deterministic MILP.
- [~] Publish solver status/gap/runtime and every retry/fallback.
  - Status and runtime are captured; bound, gap, retry, and fallback counters are not in the artifacts.

## P1 — Smoke campaign before expensive runs

- [x] Install and import-check Torch, PyTorch Geometric, Stable-Baselines3, sb3-contrib, and W&B in `.venv`.
- [x] Resolve a GPU-capable training environment.
  - Two A30s are present; `cuda:1` trains at ~273 steps/s against ~115 on CPU. Note for reproduction: all *reported test numbers* through doc 08 were produced on CPU.
- [~] Run one short train/evaluate cycle for every learning implementation.
  - Run for every implementation that has been trained. DeepSets-PPO, MaskPPO, and the no-mask arm do not exist yet, so they cannot be smoke-tested.
- [x] Integration-test manifest, scenario descriptor, raw row, aggregate summary, inference timing, overwrite refusal, and interrupted-run artifacts in the generic evaluation runner.
- [~] Verify checkpoints and validation-only selection in short train/evaluate runs, and migrate legacy runners.
  - Checkpoint retention and validation-only selection are verified and were exercised across all 16 runs of the travel ladder. Legacy runner migration is untouched.
- [ ] Run tiny deterministic comparisons against exact bounds.
- [x] Freeze campaign configurations and artifact schemas.
  - `frozen_baselines_travel.json`, `selected_travel_d.json`, `travel_methods_final.json`, versioned observation and outcome schemas.

## P2 — Main evidence campaign

- [ ] Run the XS–L2 scale grid from `01_experimental_campaign.md`.
  - Nothing beyond the single target configuration has been scored. This also blocks the quality-versus-runtime curve E3 asks for.
- [ ] Use ten independent training seeds for headline learning methods.
  - Every run in the ladder used `--seed 0`. Seed variance in the 0.857 headline is unquantified, and this is the cheapest remaining threat to the main claim.
- [ ] Use at least 500 paired test scenarios per main setting; increase to 1,000 for rare failures or small success gaps.
  - Currently 300. The GraphPPO-versus-CP-SAT travel difference is -1.0 h [-5.5, +3.7], i.e. exactly the kind of small gap that motivates the larger sample.
- [~] Run the complete component-ablation matrix.
  - Done: encoder x head (makespan era), routing action features, curriculum, reward-shaping weights across four stages. Missing: mask, global pooling, active-truck embedding, queue features, relation types.
- [~] Run charging/congestion, uncertainty/robustness, and optimizer-budget sensitivities.
  - Only a linear-versus-nonlinear charging tuning pass exists (`results/canonical/tuning_linear_charging`). Congestion, uncertainty, and optimizer-budget sensitivities are not run.
- [ ] Run interpolation and genuine OOD generalization campaigns.
  - Needs a canonical replacement for `generalization_eval.py` first; the inherited script targets the old environment and baselines.
- [x] Preserve negative and incomplete runs with documented status.

## P3 — Manuscript and response artifacts

- [ ] Freeze claims after final results; do not retrofit experiments to desired wording.
- [ ] Rewrite title, abstract, introduction, contributions, formulation, architecture, and results from the tested implementation.
- [ ] Replace normalized-reward headline tables with feasibility-first operational tables.
- [ ] Generate every LaTeX table/figure from versioned result artifacts.
- [ ] Complete `04_reviewer_response_todo.md` with exact page/line and artifact links.
- [ ] Add formulation, notation, optimizer, heuristic, random-variable, and reproducibility appendices.
- [ ] Compile LaTeX, check citations/references, lint, and archive final artifacts.

`latex/main.tex` has not been modified since `1459e54`, before this branch existed. Six `near-optimal`-class claims and two zero-shot claims remain in the text and are not currently supported by any artifact on disk.

## Stop/go gates

- [x] **Go to training:** all Campaign 0 correctness and smoke checks pass.
- [x] **Go to headline test:** architecture and checkpoints are frozen using validation only.
  - Satisfied for the primary campaign: the pre-declared banded selection rule was applied to all 16 runs on 150 validation scenarios and no test scenario was read before the choice was frozen. The action head itself was frozen in the earlier makespan-era sweep; see the P0 caveat.
- [ ] **Go to manuscript rewrite:** primary, ablation, robustness, and generalization result manifests are complete.
  - Primary is complete. Ablation is partial, robustness is nearly absent, generalization has not started.
- [ ] **Go to resubmission:** Gates A–F and every reviewer-response item have traceable evidence.

## Critical path

1. Add a no-mask/soft-mask training path and run the mask ablation (R1.2, E3). Needs code; nothing else in the revision substitutes for it.
2. Build a canonical generalization runner on the artifact contract, then run interpolation, size-transfer, and OOD campaigns with baselines in every regime (R1.7).
3. Re-run the headline arm across independent training seeds and extend the test split to at least 500 paired scenarios (R1.7, P2).
4. Add the missing baseline family: ALNS, a constructive attention policy, DeepSets-PPO, MaskPPO (R1.6, R2.8).
5. Rewrite the manuscript and the response letter against the frozen artifacts (P3, all of R1.3 and the minor comments).
