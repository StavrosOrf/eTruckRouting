# Modeling Decisions Requiring Author Validation

The author approved the recommended choice for every decision on 2026-08-11. This file now serves as the authoritative modeling decision record for implementation and the experimental campaign.

## D1 — Primary problem scope

### Recommended: joint fleet routing, assignment, sequencing, and charging

Use a primary problem in which a central policy dynamically chooses, for the active truck:

- which unserved customer to serve next;
- which charging station to visit;
- how much to charge;
- when to return to the depot.

Customers are not preassigned. Every customer must be served exactly once, truck and battery feasibility must hold, and every used truck must return to the depot.

Retain the existing fixed/preassigned sequence mode as a secondary **route-execution and charging-coordination** benchmark. It is useful for isolating shared-charger effects but should no longer carry the main fleet-routing claim.

### Lower-risk alternative

Keep preassigned routes as the only problem and reframe the paper narrowly. This requires less implementation but offers a weaker response to the scope criticism.

### Validation

- [x] Approve recommended joint routing scope.
- [ ] Select lower-risk preassigned-route scope instead.

## D2 — Customer demand, vehicle capacity, and time windows

### Recommended

Add customer demand and payload-capacity constraints to the primary problem. Add service time at every customer. Treat hard or soft time windows as a separate experimental variant rather than forcing them into every base experiment.

Rationale: without customer demand/capacity, the task remains closer to ordering and charging than a standard fleet-routing formulation. Time windows add relevance but also make feasibility and reward design substantially harder, so they should be introduced as a controlled variant.

### Validation

- [x] Add demand and capacity to the base problem.
- [x] Add time windows as a separate campaign.
- [ ] Add hard time windows to every primary instance.
- [ ] Do not add payload capacity/time windows; focus on energy and charging coordination.

## D3 — Centralized versus multi-agent control

### Recommended: centralized event-driven controller

Preserve one centralized policy acting whenever a truck becomes decision-ready. The active-truck identity is part of the state, and the graph represents every truck, customer, and charging station. This retains the event-driven contribution and avoids conflating architecture changes with decentralized credit assignment.

Add a decentralized/shared-policy MARL baseline only after the centralized model is correct.

### Validation

- [x] Approve centralized event-driven control as the main formulation.
- [ ] Make decentralized multi-agent control the main formulation.

## D4 — Objective and success definition

### Recommended: feasibility-first lexicographic evaluation

Primary comparison:

1. maximize probability that all customers are served and all required depot returns occur without energy failure;
2. among successful solutions, minimize fleet makespan;
3. report total operating time, travel time, charging time, queue time, distance/energy, and number of vehicles used as secondary metrics.

Use reward shaping only for training. Do not use shaped reward as the main measure of operational quality.

For training, use potential-based or normalized shaping where possible, with explicit terminal success/failure signals. The evaluation code must compute operational objectives independently from training reward.

### Validation

- [x] Approve feasibility, then makespan, as the primary lexicographic objective.
- [ ] Use total operating cost instead of makespan; provide desired cost weights.
- [ ] Use a multi-objective Pareto study rather than one primary objective.

## D5 — Operational uncertainty

### Recommended

Use episode-seeded stochastic processes with common random numbers across policies:

- time-of-day-dependent travel-time uncertainty;
- energy use correlated with realized travel conditions;
- stochastic service/unloading time;
- endogenous FCFS charger queues caused by fleet actions.

Start with a transparent parametric model, then add at least one distribution-shift evaluation. Do not describe endogenous queues as random inputs.

### Validation

- [x] Enable all three exogenous uncertainties in the primary campaign.
- [ ] Keep unloading deterministic in the primary campaign and vary it only in sensitivity tests.
- [ ] Provide an empirical dataset/model that should replace the parametric uncertainty model.

## D6 — Charging decisions and infrastructure

### Recommended: target-SoC actions with heterogeneous heavy-duty chargers

Replace 1–12 hour actions with discrete target-SoC actions. Proposed primary levels are 50%, 60%, 70%, 80%, 90%, and 100%, with actions below current SoC masked. Charging duration is computed by integrating the nonlinear power curve.

Use heterogeneous station powers representative of truck operations. Proposed base station classes are 150, 350, and 750 kW, plus station-specific port capacity and efficiency. Retain a 50 kW stress case only as slow legacy infrastructure.

Run 5% versus 10% target-SoC granularity and 15/30/60-minute duration actions as sensitivity studies.

### Validation

- [x] Approve target-SoC actions and 150/350/750 kW station classes.
- [ ] Keep time-duration actions but use 15-minute increments.
- [ ] Provide preferred charging powers or a vehicle/charger dataset.

## D7 — Queueing discipline

### Recommended

Model finite station ports with FCFS queues. A truck may choose a full station and join its queue. State features expose current occupancy, queue length, and estimated workload, but not future arrivals.

Add scheduled/reservation-aware charging as a later variant, not as the base model.

### Validation

- [x] Approve FCFS queue joining as the base model.
- [ ] Disallow selecting a station without a free port.
- [ ] Add reservations/scheduled priority to the primary model.

## D8 — Feasibility mask and candidate pruning

### Recommended: hard constraints only in the primary mask

The primary mask should remove only actions that are provably invalid under the information available at decision time:

- already served customer;
- capacity/time-window impossibility where applicable;
- physically unreachable destination under the defined conservative energy model;
- charging target not above current SoC;
- invalid depot-return/action-state combinations.

Remove safety-factor relaxation below 1 and forced fallback actions from the primary implementation. If no feasible continuation exists, terminate and record the failure cause.

Do not use top-k charger/customer pruning in the headline comparison. Evaluate pruning separately as a scalability tradeoff.

### Validation

- [x] Approve hard-constraint-only mask and no primary top-k pruning.
- [ ] Retain top-k pruning in the primary model; define the required candidate counts.
- [ ] Retain recovery/fallback logic as an explicitly reported safety controller.

## D9 — GraphPPO action architecture

### Recommended: compare three permutation-equivariant heads, preselect on validation

Implement and compare:

1. independent state-conditioned action scorer;
2. genuinely complete action-graph GCN;
3. self-attention/set-transformer action encoder.

All three receive identical action features and the same state encoder. Choose the main GraphPPO head using a predefined validation suite, then freeze it before the test campaign. Report all three as ablations.

This avoids retaining the current arbitrary adjacent-action chain and prevents choosing an architecture from final test results.

### Validation

- [x] Approve validation-based selection among the three heads.
- [ ] Keep a corrected complete action-GCN as the only proposed head.
- [ ] Make action self-attention the new primary architecture.

## D10 — Benchmark and evidence standard

### Recommended

Use the following benchmark hierarchy:

- exact or bounded fleet-level optimization on small deterministic instances;
- rolling-horizon/scenario optimization on small stochastic instances;
- ALNS or another strong routing-and-charging metaheuristic;
- constructive attention/transformer policy;
- PPO, MaskPPO, DeepSets-PPO, state-GNN PPO, and full GraphPPO;
- legacy per-truck conservative MILP, clearly labeled as a reference.

Train at least 10 independent seeds for headline learning methods, use 5 seeds for expensive secondary ablations, and evaluate on at least 500 paired scenarios per main setting. Increase to 1,000 scenarios when success-rate differences are small or failures are rare.

### Validation

- [x] Approve this evidence standard.
- [ ] Use five training seeds for all methods instead of ten.
- [ ] Identify specific optimization, heuristic, or transformer implementations that must be included.

## Author validation record

Record the approved decisions here before implementation:

| Decision | Approved choice | Date | Notes |
|---|---|---|---|
| D1 | Joint fleet routing, assignment, sequencing, charging, and depot return | 2026-08-11 | Recommended path approved |
| D2 | Capacity in base problem; time windows as a separate variant | 2026-08-11 | Recommended path approved |
| D3 | Centralized event-driven controller; MARL as a baseline | 2026-08-11 | Recommended path approved |
| D4 | Feasibility-first lexicographic evaluation, then makespan | 2026-08-11 | Recommended path approved |
| D5 | Seeded travel, energy, and service uncertainty plus endogenous queues | 2026-08-11 | Recommended path approved |
| D6 | Target-SoC actions; heterogeneous 150/350/750 kW chargers | 2026-08-11 | Recommended path approved |
| D7 | Finite-capacity FCFS queues that policies may join | 2026-08-11 | Recommended path approved |
| D8 | Hard-constraint-only primary mask; pruning only as an ablation | 2026-08-11 | Recommended path approved |
| D9 | Validation selection among independent, complete-GCN, and attention heads | 2026-08-11 | Recommended path approved |
| D10 | Full benchmark hierarchy; ten headline seeds and paired scenarios | 2026-08-11 | Recommended path approved |
