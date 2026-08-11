# Reviewer Revision and Experimental Campaign Plan

Last audited: 2026-08-11

This document records the repository audit, the planned response to the handling editor and reviewers, and the experimental campaigns required before resubmission.

## Executive decision

The current paper requires a substantive revision and new experiments. A prose-only revision will not resolve the concerns because the audit found mismatches in problem scope, baseline information, stochastic evaluation, action-graph implementation, benchmark interpretation, and manuscript-to-code descriptions.

### Recommended scope

Reframe the main problem as **online charging-aware execution and coordination of preassigned electric-truck routes under shared charging capacity and travel/energy uncertainty**.

The main experiments currently assume that deliveries are preassigned, delivery order is usually fixed, and depot return is generally not required. Consequently, the work should not claim to solve full fleet routing, customer assignment, or general route construction.

### Alternative scope

If the paper must retain a full electric-truck fleet-routing claim, extend the environment and policy to decide:

- customer-to-truck assignment;
- delivery sequence;
- complete route construction;
- charging-station choice and charge quantity;
- mandatory depot return where applicable.

This alternative requires a new formulation, new actions, stronger routing baselines, retraining, and complete experimental regeneration.

## Repository audit findings

### P0 validity blockers

- [ ] **Fix stochastic scenario seeding.** Travel and unloading perturbations are keyed by edge/node, time bucket, and journey index, but not by episode seed. Different evaluation seeds therefore do not necessarily generate the distinct uncertainty scenarios claimed in the paper.
  - Evidence: `EVRoutingEnv/models/simulation/traffic_simulation.py`
  - Evidence: `EVRoutingEnv/models/simulation/delivery_simulator.py`
  - Acceptance: identical seed and action sequence reproduce a trajectory; different seeds produce different uncertainty; every policy receives identical exogenous draws for a common scenario ID.

- [ ] **Give PPO, MaskPPO, and GraphPPO equivalent observations.** The flat baseline allocates only `num_stops` delivery slots instead of `num_trucks * num_stops`. In 100T3S it can expose at most three delivery-node slots despite potentially 300 delivery tasks.
  - Evidence: `EVRoutingEnv/state/state_space.py`
  - Acceptance: all baselines receive the same trucks, deliveries, chargers, global state, and candidate destinations.

- [ ] **Correct the action graph.** The function named `_build_fully_connected_edges` currently constructs a bidirectional chain between adjacent action nodes.
  - Evidence: `algo/PPO_VariableActionGNN.py`
  - Acceptance: implement the claimed complete graph, or rename and justify the chain; add candidate-order permutation tests.

- [ ] **Correct benchmark naming and interpretation.** The main Gurobi reference solves each truck separately, ignores charger contention and stochastic travel, permits at most one charger detour per delivery leg, and uses continuous charging durations.
  - Evidence: `EVRoutingEnv/baselines/optimal_gurobi.py`
  - Acceptance: call it a conservative per-truck deterministic MILP reference, or replace it with a fleet-level benchmark that matches the simulator.

- [ ] **Record solver fallbacks.** The optimization policy may retry with a modified safety factor and then invoke an emergency heuristic.
  - Acceptance: report solver status, MIP gap, time limit, retry count, and fallback count for every experiment.

- [ ] **Make checkpoint selection reproducible.** Evaluation scripts contain hardcoded, commented alternatives and manually selected seeds/checkpoints.
  - Evidence: `scripts/evaluation/eval_parallel_policies.py`
  - Acceptance: choose checkpoints using a predefined validation metric and save the rule and selected checkpoint in a run manifest.

- [ ] **Remove post-hoc evaluation exclusions.** Single-truck evaluation removes all instances where the optimization reference is marked infeasible; ten nominal evaluation seeds were excluded.
  - Evidence: `scripts/evaluation/eval_parallel_policies.py`
  - Evidence: `results/vrp_excluded_seeds.txt`
  - Acceptance: validate instances before evaluation with a method-independent rule, generate the required number of valid scenarios, and never exclude a scenario based on one method's result.

- [ ] **Audit the action mask as an algorithmic component.** It contains charger top-k pruning, hop limits, lookahead, forced charger departure, safety-factor relaxation below 1, and fallback actions.
  - Acceptance: disclose every rule, log every relaxation/fallback, and ablate each material component.

- [ ] **Align charging semantics.** The manuscript permits charging only when a port is free, while the simulator can queue a request at a full station.
  - Acceptance: choose one FCFS queueing interpretation and use it consistently in code, equations, and pseudocode.

- [ ] **Restore charger heterogeneity.** `map_charger_type` currently maps every station type to `DCFast`, and the main configuration uses the same nominal power for Level 2 and DC fast charging.
  - Evidence: `EVRoutingEnv/utils/utils.py`
  - Acceptance: preserve charger type, power, efficiency, and capacity from input data; document the evaluated distributions.

### Reward and metric blockers

- [ ] Treat full-service completion as the primary outcome.
- [ ] Report completion time/cost conditional on success.
- [ ] Report completed-delivery fraction only as a diagnostic.
- [ ] Report failure causes: depletion, stranded/no action, timeout, invalid instance, and solver fallback.
- [ ] Report charging time, queue time, detour time, number of charging sessions, terminal SoC, and inference time.
- [ ] Add 95th percentile and CVaR measures for completion and queue time.
- [ ] Stop interpreting normalized reward as an optimality gap.
- [ ] Remove `near-optimal` unless a valid exact or bounded reference supports the claim.
- [ ] Explain that the current reward uses time coefficient `1`, per-delivery bonus `500`, and failure penalty `-1000`.
- [ ] Reconsider reward shaping: a failed truck retains bonuses for partial deliveries, which explains high reward at zero full-service success.

## Literature and novelty campaign

- [ ] Remove the claim that conventional eVRP is a single-vehicle problem.
- [ ] Remove the claim that fleet-level operational uncertainty is largely unstudied.
- [ ] Conduct a structured literature review covering:
  - electric freight fleets;
  - electric bus fleets;
  - ride-hailing and shared electric fleets;
  - finite-capacity/shared charging stations;
  - deterministic, time-dependent, stochastic, and endogenous queues;
  - travel-time and energy-consumption uncertainty;
  - nonlinear and partial charging;
  - rolling-horizon, robust, stochastic, heuristic, and RL methods;
  - graph, attention, transformer, and multi-agent policies.
- [ ] Build a comparison matrix with these columns:
  - fleet coordination;
  - customer assignment;
  - route construction/sequence decisions;
  - shared finite-capacity chargers;
  - endogenous queueing;
  - travel-time uncertainty;
  - energy-use uncertainty;
  - nonlinear charging;
  - event-driven decisions;
  - variable action sets;
  - explicit feasibility control;
  - scale and evaluation setting.
- [ ] Base the novelty statement on the populated matrix, not on broad claims about fleet size or uncertainty.
- [ ] Position the contribution around the exact combination that remains defensible:
  - event-driven preassigned-route execution with endogenous shared-charger congestion;
  - feasibility-aware variable-action control;
  - heterogeneous state representation;
  - controlled decomposition of masking and graph components;
  - rapid fleet-scale online inference, if supported by revised experiments.

## Problem formulation and methods campaign

- [ ] Define the exact problem and decision boundary on the first manuscript page.
- [ ] State explicitly what `operational uncertainty` includes and what is deterministic.
- [ ] Distinguish exogenous randomness from endogenous charger congestion.
- [ ] Add an explicit constraint requiring every assigned delivery to be served.
- [ ] Define the consequence of incomplete service and all terminal conditions.
- [ ] Define depot-return requirements separately for fixed-sequence and flexible single-truck modes.
- [ ] Use one symbol consistently for nominal and realized edge energy.
- [ ] Define clipping bounds and clarify that energy bounds do not bound travel time.
- [ ] Rewrite the charging integral using a time-varying SoC trajectory and a distinct integration variable.
- [ ] Correct malformed movement/energy constraint indices.
- [ ] Define unloading time as fixed or stochastic in the formulation and in each experiment.
- [ ] Replace `stochastic edge traversing` with `realized travel duration on the selected road-network transition`.
- [ ] Publish the numerical reward coefficients.
- [ ] Correct the sender/receiver embeddings in the heterogeneous message equation.
- [ ] Correct the action-GCN neighbor aggregation equation.
- [ ] Rewrite the state encoder, action head, and critic equations from the actual code.
- [ ] State the actual node, edge, and action feature vectors.
- [ ] Correct the stated architecture: selected models use 32-dimensional graph embeddings rather than 256-dimensional graph layers; the 256 dimension belongs to the MLP configuration.
- [ ] Explain whether SMDP discounting uses one `gamma` per decision or holding-time-aware discounting such as `gamma ** delta_t`.
- [ ] Add full optimizer and heuristic formulations/pseudocode to appendices.

## Baseline campaign

All learning baselines must share observations, action candidates, mask definitions where applicable, reward, training steps, validation scenarios, and tuning effort.

- [ ] PPO with full equivalent state and no mask.
- [ ] MaskPPO with full equivalent state.
- [ ] Pooled MLP or DeepSets encoder with masking.
- [ ] Heterogeneous state GNN with independent per-action MLP scoring.
- [ ] Heterogeneous state GNN with action interaction.
- [ ] Attention/transformer action scorer with identical inputs and mask.
- [ ] Strong constructive or local-search electric-routing baseline.
- [ ] ALNS/ILS or rolling-horizon charging-aware heuristic.
- [ ] Conservative deterministic per-truck MILP, clearly labeled as a reference rather than an optimum.
- [ ] Exact fleet-level solution or lower/upper bounds for small instances.
- [ ] Scenario-based or rolling-horizon optimization reference for uncertainty, if computationally feasible.
- [ ] Remove the unsupported claim that PPO is the most capable discrete-action RL algorithm.

## Ablation campaign

- [ ] PPO without masking.
- [ ] PPO with masking only.
- [ ] Flat/DeepSets state versus heterogeneous state graph.
- [ ] State graph with an independent action scorer.
- [ ] State graph plus chain action graph.
- [ ] State graph plus complete action graph.
- [ ] State graph plus attention-based action interaction.
- [ ] Global pooled embedding only.
- [ ] Global pooling plus active-truck embedding.
- [ ] Remove queue/occupancy features.
- [ ] Remove uncertainty-related edge features.
- [ ] Remove charger-to-charger and delivery-to-delivery relations.
- [ ] Disable or vary detour top-k pruning.
- [ ] Disable hop-limit logic.
- [ ] Fixed conservative mask versus relaxed/fallback mask.
- [ ] Report action-mask fallback and safety-relaxation frequency.

For each ablation, report full-service probability, conditional completion time, queue time, tail risk, inference time, and confidence intervals. Reward alone is insufficient.

## Sensitivity campaign

- [ ] Charging duration granularity: 15, 30, and 60 minutes.
- [ ] Target-SoC or continuous-duration charging, if implementable.
- [ ] Heavy-duty charger powers rather than only 50 kW.
- [ ] Fleet-to-port ratios and station capacities.
- [ ] Charger-location and charger-density variations.
- [ ] No, low, nominal, and high travel-time uncertainty.
- [ ] No, low, nominal, and high energy uncertainty.
- [ ] Independent versus correlated travel and energy perturbations.
- [ ] Fixed versus stochastic unloading time.
- [ ] Linear versus nonlinear/CCCV charging.
- [ ] Validate the charging curve parameters against a suitable source and compare with the Montoya-style piecewise nonlinear formulation.
- [ ] Energy safety multiplier sensitivity.
- [ ] Detour candidate-count and hop-limit sensitivity.
- [ ] Reward-weight sensitivity if scalar reward remains part of evaluation.

## Generalization campaign

- [ ] Separate interpolation from out-of-distribution evaluation.
- [ ] Use multiple training seeds and policies, not one selected 100T3S checkpoint.
- [ ] Test unseen fleet sizes and stop counts.
- [ ] Test unseen road graphs or regions.
- [ ] Test unseen charging-station locations, capacities, and powers.
- [ ] Test unseen battery capacities and vehicle-consumption parameters.
- [ ] Test unseen uncertainty distributions and severities.
- [ ] Test unseen operational policies, including depot-return requirements where meaningful.
- [ ] Evaluate credible baselines in every generalization regime.
- [ ] If only fleet/task size changes within the California simulator, call the result `size transfer within the training simulator distribution`, not broad zero-shot generalization.

## Statistical and reproducibility campaign

- [ ] Use at least five independent training seeds per learning method.
- [ ] Use common held-out scenario IDs for paired evaluation.
- [ ] Separate training-seed variability from evaluation-scenario variability.
- [ ] Choose checkpoints only on predefined validation scenarios.
- [ ] Use Wilson confidence intervals for full-service success rates.
- [ ] Use paired bootstrap intervals for time, queue, and cost differences.
- [ ] Report effect sizes, not only averages.
- [ ] Avoid mean +/- raw standard deviation for Bernoulli success indicators.
- [ ] Predefine invalid-instance handling and never filter on a method's outcome.
- [ ] Publish per-episode results and scenario seeds.
- [ ] Save a run manifest containing:
  - git commit;
  - full environment configuration;
  - policy configuration;
  - training and evaluation seeds;
  - training budget;
  - validation/checkpoint rule;
  - dependency versions;
  - hardware;
  - solver version/settings;
  - wall-clock training, evaluation, and inference times.

## Manuscript rewrite campaign

### Title, abstract, introduction, and contributions

- [ ] Replace full fleet-routing language with the selected scope.
- [ ] Rewrite the abstract after final experiments are available.
- [ ] Remove unsupported statements about heuristic infeasibility at scale.
- [ ] Remove `near-optimal`, `optimization-level`, `consistently outperforms`, and `robust generalization` unless the revised results support them.
- [ ] Clarify the uncertainty sources in the introduction.
- [ ] Explain the fixed/preassigned delivery setting before describing GraphPPO.
- [ ] Present contributions as testable statements tied to sections and experiments.

### Experimental section

- [ ] Add a table of every deterministic, stochastic, and endogenous quantity.
- [ ] For each random variable, state distribution, parameters, clipping, correlation, source, and whether enabled.
- [ ] Correct the five-training-seed claim unless the final tables truly aggregate five seeds.
- [ ] Provide the policy-selection protocol.
- [ ] Provide optimizer and heuristic assumptions.
- [ ] Define primary, secondary, and diagnostic metrics before results.
- [ ] Replace normalized reward tables with feasibility-first tables.
- [ ] Distinguish scenario count from training-seed count.

### Technical and minor corrections

- [ ] Change `demonstrated` to `demonstrates` in the literature review.
- [ ] Add the missing comma in Equation 12.
- [ ] Standardize all figure captions and capitalization.
- [ ] Change `c. Actor Network head` to `(c) Actor Network Head` in the relevant figure asset/caption.
- [ ] Correct Table 2 so the energy-consumption row shows realized energy, not only coefficient `xi`.
- [ ] Correct all node/edge/action feature definitions.
- [ ] Remove commented duplicate literature notes and the commented placeholder ablation section.
- [ ] Add a notation/glossary table.
- [ ] Normalize unusual BibTeX formatting and add the missing recent literature.
- [ ] Add a reproducibility appendix.
- [ ] Add a LaTeX build command and CI workflow.
- [ ] Run LaTeX compilation, reference checks, and a linter before submission.

## Point-by-point reviewer response map

### Handling editor

- [ ] **Fleet eVRP literature:** concede the incorrect single-vehicle characterization and replace it with a structured literature taxonomy.
- [ ] **Operational uncertainty:** concede that it is established and specify the exact uncertainty/decision combination studied here.
- [ ] **Novelty:** provide the literature matrix and component ablations; narrow the contribution to what is empirically demonstrated.

### Reviewer 1

- [ ] **R1.1:** reframe as preassigned-route execution or implement full fleet routing.
- [ ] **R1.2:** run mask, state graph, action graph, pooling, and active-truck ablations.
- [ ] **R1.3:** correct energy notation, clipping, charging integral, sender/receiver message, and action-neighbor equation.
- [ ] **R1.4:** rename/replace the optimization reference and remove unsupported proximity-to-optimum claims.
- [ ] **R1.5:** make success the primary metric and explain partial-delivery reward behavior.
- [ ] **R1.6:** fully specify the heuristic and add fair attention/transformer and strong search baselines.
- [ ] **R1.7:** replace the single-policy same-distribution analysis with multi-seed OOD evaluation or narrow the claim.

### Reviewer 2

- [ ] **R2.1:** define the selected problem and its distinction from classical EVRP in the introduction.
- [ ] **R2.2:** define each source of operational uncertainty early.
- [ ] **R2.3:** add the all-deliveries-served constraint and incomplete-service behavior.
- [ ] **R2.4:** replace `stochastic edge traversing` with precise language.
- [ ] **R2.5:** provide numerical reward weights.
- [ ] **R2.6:** run charging-discretization sensitivity.
- [ ] **R2.7:** provide the full optimization formulation and queue/stochastic/offline assumptions.
- [ ] **R2.8:** provide heuristic pseudocode and compare against closely related heuristics.
- [ ] **R2.9:** add attention/transformer baselines or provide a defensible scope-based explanation.
- [ ] Apply all requested tense, punctuation, caption, and Table 2 corrections.

### Additional comments from Ziyan

- [ ] Define unloading time and clarify that station waiting is endogenous through finite port capacity and FCFS queueing.
- [ ] Cite and validate the charging function; compare it with established piecewise nonlinear charging models.
- [ ] Expand the random-variable table and state which uncertainties are enabled in each experiment.

## Suggested execution order

1. Freeze the intended scope and contribution statement.
2. Fix seeding, action graph, baseline state parity, queue semantics, and charger mapping.
3. Add tests and a reproducible experiment manifest.
4. Implement fair baselines and the ablation matrix.
5. Run small-scale correctness and optimizer-validation experiments.
6. Run the main multi-seed campaign.
7. Run charging, uncertainty, congestion, and architecture sensitivities.
8. Run genuine OOD evaluation or narrow the transfer claim.
9. Regenerate all tables and figures from final result files.
10. Rewrite the mathematical formulation and methods from the corrected code.
11. Rewrite the abstract, introduction, contributions, results, and conclusion.
12. Prepare a point-by-point response letter linking each response to manuscript changes and new evidence.
13. Compile, lint, cross-check references, and complete a reproducibility audit.

## Resubmission gate

Do not resubmit until all of the following are satisfied:

- [ ] Problem name and claims match the optimized decisions.
- [ ] Scenario seeding is correct and tested.
- [ ] Baselines receive equivalent information and candidates.
- [ ] The action graph matches its description.
- [ ] Checkpoint and scenario selection are predefined and reproducible.
- [ ] Results aggregate multiple independent training seeds.
- [ ] Full-service feasibility is the primary metric.
- [ ] Strong baselines and component ablations are complete.
- [ ] The optimization reference is described honestly.
- [ ] Generalization claims match the tested distribution.
- [ ] Manuscript equations and feature definitions match the implementation.
- [ ] All reviewer comments have an explicit response and corresponding change/evidence.
- [ ] Tests, run manifests, LaTeX build, and final artifacts are available.
