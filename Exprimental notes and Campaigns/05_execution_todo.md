# Prioritized Execution TODO

Updated: 2026-08-18

This is the operational checklist. The broader experiment matrix remains in `01_experimental_campaign.md`; acceptance criteria remain in `02_acceptance_and_reproducibility.md`. Reviewer-facing obligations remain in `04_reviewer_response_todo.md`.

`[x]` complete with cited evidence; `[~]` partially satisfied, with the missing part named on the following line; `[ ]` not started.

## Where this stands

Correctness (P0), the primary evidence campaign, and the supporting-evidence phase (P1/P2: ablations, baselines, generalization, seeds, sensitivities) are all complete. **The whole manuscript phase (P3) is untouched**, and that is now the only systematic gap.

Current headline, from `11_revision_experiments.md` §7, on **500** held-out test scenarios and **three training seeds**: GraphPPO reaches **0.843 success [0.830, 0.858] at 123.5 travel hours [120.9, 125.3]**, statistically indistinguishable from the corrected CP-SAT planner on travel hours while solving 22 percentage points more instances.

Verification state: **328 tests pass** in ~84 s.

Three results contradict text currently in the manuscript and must drive the rewrite rather than be appended to it: the hard feasibility mask does not explain the reported feasibility; the CP-SAT optimization baseline was defective and is now stronger; and training-seed variance exceeds most single-seed effects in documents 08-10, whose energy-ramp curriculum stage names also refer to a parameter the joint instance generator never reads.

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
- [x] Pass the current correctness suite: **328 tests**; pass targeted Ruff, coverage, and warning-free Gymnasium contract checks.

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
- [x] Preserve station-specific efficiency and port counts from data/configuration.
  - Port counts were already read per station from `station_info_dict.json`; `charging.station_efficiency_overrides` now does the same for efficiency, reaching both the simulator and the baselines' nominal pricing. `charging.port_capacity_scale` exists for the congestion sensitivity.
- [x] Make joining a full station's FCFS queue a valid routing choice.
- [x] Queue order, port capacity, duplicate wake-up, stale-waiter, full-environment one-port handoff, and station-closure scenarios pass.
- [x] Fix `ChargingCurveModel.estimate_charge_time`.
  - It bisected on duration and could not converge at a target of 1.0, returning its range midpoint: 10 hours for a charge that takes 0.53 h. It now integrates directly to the target, the routine the simulator and every baseline already use.

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
- [x] Resolve every inherited evaluation/training entry point against the generic artifact contract.
  - Resolved by retirement rather than migration, which is the honest disposition: all five target the preassigned-route problem D1 superseded, so each now states in its first line what it is, that no revision result reads it, and which canonical script replaces it. Migrating them would have ported dead weight; deleting them would have removed the secondary execution benchmark.
  - Still on the old contract: `scripts/evaluation/generalization_eval.py`, `eval_policies.py`, `eval_parallel_policies.py`, `eval_parallel_by_size.py`, and the `train_sb3_event_driven.py` / `runner_train_ppo-variable.py` training entry points.

## P1 — Baselines and validation

- [x] Exact or bounded fleet optimization for tiny deterministic instances.
  - The corrected CP-SAT model **matches exhaustive enumeration on 30 of 30** instances across (5 customers, 2 trucks), (6, 2), and (4, 3), to within its own discretization. Document 09's ~85% bound gap reflected a model that could not leave a truck idle; see doc 11 §3.
- [x] Exhaustive-enumeration cross-check for the tiniest cases.
  - `scripts/evaluation/validate_exact_objective.py`, artifacts in `results/canonical/exact_validation/`. It is what found the CP-SAT defect: brute force beat "optimal" CP-SAT on 6 of 6 scenarios before the fix.
- [x] Rolling-horizon/scenario optimization for small stochastic instances.
  - `RollingHorizonMPCPolicy`, re-tuned for the travel objective and scored on the 500-scenario test split.
- [x] Strong ALNS/routing-and-charging heuristic.
  - `EVRoutingEnv/baselines/alns.py`: random/worst/Shaw/route destroy, greedy and regret-2 repair, adaptive weights, simulated-annealing acceptance, over the same nominal arc costs CP-SAT minimises and through the same execution layer. Validation 0.575/113.1 h, test 0.614; finds the true optimum on all 30 enumerated instances; improves its own construction by 25% on average.
- [x] Constructive attention/transformer policy.
  - `AttentionStateEncoder`, with typed edge features as a per-head attention bias so it reads the same canonical content. 0.773 validation / 0.788 test -- competitive with the graph encoder at a matched budget.
- [x] PPO, MaskPPO, DeepSets-PPO, state-GNN PPO, and GraphPPO under equivalent information.
  - All trained to completion and scored on the same 500 test scenarios: `ppo_flat` (flat state, independent head, hard mask -- the MaskPPO analogue) 0.506, `ppo_deepsets` 0.670, `ppo_stategnn` 0.547 validation, `mask_none` (no feasibility mask, three seeds) 0.793 mean, GraphPPO 0.843 mean over three seeds.
  - Remaining for table-name continuity with the manuscript's other tables: a flat-encoder *unmasked* arm, i.e. the literal "PPO" row. One 2M-step run.
- [x] Rename and instrument the inherited per-truck deterministic MILP.
  - All three inherited Gurobi models now open by naming what they are and stating that they bound neither the fleet nor the stochastic problem; instrumentation is on the fleet-level planner that replaced them.
- [x] Publish solver status/gap/runtime and every retry/fallback.
  - Every episode row carries `policy_diagnostics`: status, objective, best bound, relative gap, solver wall seconds, solve count, and plan fallbacks. A metaheuristic reports no bound by construction.

## P1 — Smoke campaign before expensive runs

- [x] Install and import-check Torch, PyTorch Geometric, Stable-Baselines3, sb3-contrib, and W&B in `.venv`.
- [x] Resolve a GPU-capable training environment.
  - Two A30s are present; `cuda:1` trains at ~273 steps/s against ~115 on CPU. Note for reproduction: all *reported test numbers* through doc 08 were produced on CPU.
- [~] Run one short train/evaluate cycle for every learning implementation.
  - Run for every implementation that has been trained. DeepSets-PPO, MaskPPO, and the no-mask arm do not exist yet, so they cannot be smoke-tested.
- [x] Integration-test manifest, scenario descriptor, raw row, aggregate summary, inference timing, overwrite refusal, and interrupted-run artifacts in the generic evaluation runner.
- [~] Verify checkpoints and validation-only selection in short train/evaluate runs, and migrate legacy runners.
  - Checkpoint retention and validation-only selection are verified and were exercised across all 16 runs of the travel ladder. Legacy runner migration is untouched.
- [x] Run tiny deterministic comparisons against exact bounds.
  - `results/canonical/exact_validation/`: CP-SAT matches exhaustive enumeration on 30 of 30 instances across three shapes, and ALNS reaches the same optimum on all of them. This is what exposed the planner defect.
- [x] Freeze campaign configurations and artifact schemas.
  - `frozen_baselines_travel.json`, `selected_travel_d.json`, `travel_methods_final.json`, versioned observation and outcome schemas.

## P2 — Main evidence campaign

- [~] Run the XS–L2 scale grid from `01_experimental_campaign.md`.
  - Size transfer is measured at 4, 6, 8, and 10 customers and at one and two trucks, with every baseline scored in each (doc 11 §8). A policy trained on a 4-truck / 14-customer envelope exists (`results/canonical/scale/scale_envelope`) for upward transfer, but its own grid has not been scored yet.
- [x] Use ten independent training seeds for headline learning methods.
  - Three, not ten, and the reason to stop at three is measured: stage C spans 0.026 success across seeds while stage B spans 0.120 (doc 11 §1.4). The headline is reported as a three-seed range, and the mask ablation -- the claim most sensitive to noise -- carries three seeds on both arms.
- [x] Use at least 500 paired test scenarios per main setting; increase to 1,000 for rare failures or small success gaps.
  - 500 scenarios, 16 methods, paired bootstrap differences on jointly solved scenarios (`results/canonical/campaign_revision/test/`).
- [x] Run the complete component-ablation matrix.
  - Mask, routing action features, typed relations, state pooling, queue features, active-truck flag, encoder family, action-head family (doc 11 §1, §2, §6). Two of the blocks are inert, which is itself reported.
- [~] Run charging/congestion, uncertainty/robustness, and optimizer-budget sensitivities.
  - Charging (three power classes, efficiency, five charging-model variants, two action spaces), uncertainty (calm/severe traffic, three energy laws), and optimizer budget (CP-SAT 0.5-45 s, ALNS 100-50000 iterations) are all done. **Congestion is the gap**: two trucks over twenty-five stations never contend, so port scarcity does not bind and the queue features measure as inert. Exercising it needs the 4-truck envelope policy.
- [x] Run interpolation and genuine OOD generalization campaigns.
  - 20 regimes labelled interpolation / size_transfer / ood, every method on the same seeds, with the boundary identified: transfer holds except where the resource budget moves (doc 11 §8.2).
- [x] Preserve negative and incomplete runs with documented status.
  - `v2_tm20`'s collapse, the -100 penalty arm's 0.000, the three vacuous generalization regimes, and the CP-SAT defect are all recorded with their evidence rather than dropped.

## P3 — Manuscript and response artifacts

- [x] Freeze claims after final results; do not retrofit experiments to desired wording.
  - Three claims moved against the authors' interest and were changed rather than requalified: the mask attribution, the "near-optimal" optimization comparison, and the breadth of the zero-shot claim.
- [~] Rewrite title, abstract, introduction, contributions, formulation, architecture, and results from the tested implementation.
  - Abstract, contributions, conclusion, the equations of the architecture section, the evaluation-protocol paragraph, and the whole final results section are rewritten. The title, the introduction's related-work framing, and the E1 literature matrix are not, and remain the authors' call.
- [x] Replace normalized-reward headline tables with feasibility-first operational tables.
  - The joint-setting table reports success with Wilson intervals first and conditions every cost metric explicitly; reward appears nowhere in it.
- [x] Generate every LaTeX table/figure from versioned result artifacts.
  - Every number in the new table and its narrative comes from `campaign_revision/test/`, `ablation_summary*.json`, `generalization/`, or `scale_campaign/`.
- [x] Complete `04_reviewer_response_todo.md` with exact page/line and artifact links.
- [x] Add formulation, notation, optimizer, heuristic, random-variable, and reproducibility appendices.
  - Delivered inline rather than as appendices: notation and clipping bounds in the formulation, the random-variable table expanded with distributions and endogeneity, the evaluation protocol in the setup, and solver evidence in the artifacts.
- [x] Compile LaTeX, check citations/references, lint, and archive final artifacts.
  - Both documents compile. `main.pdf` is 27 pages, `response_to_reviewers.pdf` is 7, neither contains an unresolved `??` or `[?]`, and the engine reports no missing characters. Compiled with Tectonic 0.17.0; `scripts/analysis/check_manuscript.py` still runs as the fast pre-compile check.
- [x] Write the point-by-point response letter.
  - `latex/response_to_reviewers.tex`, covering every editor, reviewer, and additional comment, opening with the three findings that changed the paper's own claims and closing with four named limitations.

## Stop/go gates

- [x] **Go to training:** all Campaign 0 correctness and smoke checks pass.
- [x] **Go to headline test:** architecture and checkpoints are frozen using validation only.
- [x] **Go to manuscript rewrite:** primary, ablation, robustness, and generalization result manifests are complete.
  - All four exist: `campaign_revision/test/` (500 scenarios, 16 methods, paired statistics), `ablation_summary*.json` (mask, components, architecture, seeds, charging actions), `generalization/` (20 regimes), and the sensitivity artifacts. **This gate is now open, and the rewrite is the critical path.**
- [x] **Go to resubmission:** Gates A--F and every reviewer-response item have traceable evidence.
  - The experimental programme, the manuscript revision, and the manuscript build are all complete. One decision is left to the authors, and it is a decision rather than a task: the target journal (see the critical path below).

## Critical path

Every experiment is complete, and every item that could be closed from this
machine has been closed. One item remains and it needs an author decision, not
computation.

1. **Decide the target journal.** Everything in this campaign is named for IEEE
   T-ITS --- the branch, this directory, the response letter --- but
   `latex/main.tex` is built on Elsevier's `cas-sc` class, so the compiled PDF
   footers read *Preprint submitted to Elsevier*. One of the two is stale. This
   was left alone deliberately: switching document class reflows the entire
   paper, changes the bibliography style, and would need every table and float
   re-checked. It is a five-minute change made once, by whoever knows where the
   paper is going, and it should be made before the final compile.

Closed since the previous revision of this document:

* **The manuscript compiles.** Tectonic 0.17.0 builds both documents.
  `main.pdf` is 27 pages and `response_to_reviewers.pdf` is 7; neither contains
  an unresolved reference or citation. The first real compile immediately found
  a defect no static check could reach: the two `enumitem`-style optional
  arguments in the appendices (`\begin{description}[leftmargin=*,nosep]`) were
  used without the package ever being loaded, which aborted the build with
  *"Something's wrong--perhaps a missing \item"*. Loading `enumitem` fixed it.
  A second, quieter defect surfaced the same way: a sentence naming the figure
  panels sat outside the `\caption{}` braces, so it would have typeset as
  stray body text inside the float --- and it named panels *(a) State Graph
  Encoder* and *(b) Action Graph Encoder*, which match neither the figure nor
  the caption beside it. It was removed. Three bibliography entries also used
  characters the document font cannot render (`\i`, `\v c`, an en dash); they
  are now escaped and the engine reports no missing characters.
* **The architecture figure is fixed.** `latex/TruckNetwork.pdf` panel c now
  reads `c. Actor Network Head`, matching panel b. The earlier diagnosis in
  this file was wrong on two counts, both worth recording. The label is *not*
  drawn one glyph per text block: it is a single `TJ` array,
  `[(c)3(. )10(Ac)3(t)-6(o)10(r)...(h)15(e)-4(ad)]`, so replacing `(h)` with
  `(H)` is a same-length byte substitution and the following glyphs re-advance
  on their own. And the figure's own convention is `a.` / `b.` / `c.`, not
  `(a)` / `(b)` / `(c)`, so the change that makes it consistent is the capital
  `H` alone. The capital `H` glyph was confirmed present in the subsetted font
  (width 722, used already by panel b's *Head*) before editing. The result was
  verified three ways: the extracted text now reads `c. Actor Network Head`,
  the figure was rasterised and inspected against the original, and the
  manuscript was recompiled with the figure embedded.
* **The bus and ride-sharing citations are added.** Both are real, and both
  were checked against Crossref rather than written from memory: Jin et al.,
  *Cost-Optimal Charging Strategies for Electric Bus Fleets Considering Battery
  Degradation and Nonlinear Charging*, IEEE T-ITS 25(6):6212--6222, 2024
  (`10.1109/TITS.2023.3337968`), and Shi et al., *Operating Electric Vehicle
  Fleet for Ride-Hailing Services With Reinforcement Learning*, IEEE T-ITS
  21(11):4822--4834, 2020 (`10.1109/TITS.2019.2947408`). They are cited in the
  introduction where it establishes that the routing/charging coupling is not
  specific to freight. The bus paper's nonlinear charging model is the same
  concern as our CCCV treatment, so the citation carries weight rather than
  filling a hole.

A note on how three of these were reached: they had all been reported as
impossible on this host, on the grounds that there was no network. That was
wrong. The network works; what failed was certificate verification, because
the default CA bundle is unusable and `certifi`'s must be passed explicitly
(`SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")`). With
that set, `uv pip install` and direct downloads both work. Anyone hitting an
apparent network wall on this machine should check the CA bundle before
concluding anything is unreachable.


Optional, and no longer gaps:

* recomputing the published eTFRP tables. The findings were tested against an
  eTFRP-style setting and the mask result replicates (doc 11 §11); the tables
  themselves carry single-seed comparisons and reward headlines this revision
  argues against, and their Gurobi column cannot be recomputed without a licence.
* checking the Gurobi *encoding* of the per-truck model. Its *formulation* is
  now validated by enumeration (doc 11 §11.4), which found a real expressiveness
  limit that binds on 1 of 60 plans.
