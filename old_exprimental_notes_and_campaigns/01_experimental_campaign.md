# Proposed Experimental Campaign

This campaign begins only after the decisions in `00_modeling_validation.md` are approved. Run IDs and result folders must be generated from immutable configuration files rather than edited lists of hardcoded checkpoint paths.

## Campaign 0 — Correctness and reproducibility

Purpose: ensure subsequent compute is scientifically usable.

- [x] Implement episode-scoped RNG and common scenario IDs.
- [x] Test same-seed replay and different-seed variation.
- [x] Test policy-RNG-independent exogenous streams.
- [x] Correct flat-state delivery capacity and establish observation parity through the versioned canonical flat/set/graph adapters.
- [ ] Correct action connectivity and add permutation-equivariance tests.
- [x] Remove mask relaxation and fallback behavior from the primary joint model; legacy migration remains separate.
- [x] Preserve charger types, allow FCFS queue joining, and invariant-test finite ports.
- [x] Split operational episode metrics from shaped training reward; campaign aggregation remains.
- [x] Add boundary configuration validation and strict immutable run-manifest primitives; runner wiring remains a smoke-campaign task.
- [ ] Add CI; the local suite currently contains 178 unit/integration checks and all three campaign configurations pass Gymnasium's environment checker without warnings.

Gate: no large training job starts until every Campaign 0 test passes.

## Campaign 1 — Environment and formulation validation

Purpose: prove that the simulator implements the mathematical model.

- [ ] Hand-construct tiny instances with analytically known energy and timing outcomes.
- [x] Verify every-customer-once, payload capacity, battery conservation, charging/queue, service, and depot-return invariants across randomized unit sequences and seeded stochastic episodes.
- [ ] Compare nonlinear charging integration with reference curves.
- [x] Verify deterministic event ties, unloading completion, one-port FCFS handoff, station closure, and early time-window waiting; expand to randomized mixed-event property tests before the main campaign.
- [ ] Verify exact-solver objective against exhaustive enumeration on very small instances.
- [ ] Validate uncertainty mean, variance, clipping, time-of-day dependence, and correlation empirically.
- [ ] Publish invariant and distribution-check reports.

## Campaign 2 — Hyperparameter and architecture selection

Purpose: select architecture and training hyperparameters without using test results.

Representative validation settings, subject to D1:

- small: 2 trucks, 10 customers;
- medium: 10 trucks, 50 customers;
- congested: 20 trucks, 100 customers with low charger capacity;
- transfer: one unseen intermediate fleet/customer combination.

Candidate factors:

- state encoders: flat MLP, DeepSets, heterogeneous GNN;
- action heads: independent scorer, complete GCN, self-attention;
- active-truck conditioning: pooled only versus pooled plus active embedding;
- graph depth: 2, 3, 4;
- hidden width: 64, 128, 256;
- entropy coefficient and learning rate;
- target-SoC granularity;
- candidate pruning disabled versus controlled top-k.

Protocol:

- [ ] Use three seeds for screening.
- [ ] Use successive halving or another predefined resource allocation rule.
- [ ] Rank by feasibility first, then conditional makespan.
- [ ] Confirm finalists with five seeds.
- [ ] Freeze the chosen configuration before Campaigns 3–7.
- [ ] Retain all screening results, including negative findings.

## Campaign 3 — Main benchmark

Purpose: establish performance across scale and congestion.

Proposed joint-routing grid:

| Scale | Trucks | Customers | Role |
|---|---:|---:|---|
| XS | 1 | 10 | single-vehicle sanity/exact comparison |
| S1 | 2 | 10 | exact fleet-level comparison |
| S2 | 3 | 20 | exact/bounded comparison |
| M1 | 5 | 50 | standard learning/heuristic comparison |
| M2 | 10 | 100 | main medium-scale result |
| L1 | 20 | 200 | large dynamic fleet result |
| L2 | 50 | 500 | scalability stress test |

If D1 retains preassigned routes, replace `Customers` with explicit stops per truck and label the campaign as route execution rather than fleet route construction.

Methods:

- exact/bounded optimization where tractable;
- rolling-horizon optimization;
- ALNS/metaheuristic;
- constructive attention/transformer;
- PPO;
- MaskPPO;
- DeepSets-PPO;
- state-GNN PPO with independent action scoring;
- GraphPPO.

Protocol:

- [ ] Ten training seeds for headline learning methods.
- [ ] Five seeds for secondary learning baselines when computationally justified.
- [ ] At least 500 common test scenarios per setting.
- [ ] Fixed solver time budgets reported at several levels.
- [ ] Hardware-normalized training and inference time.
- [ ] Paired scenario-level analysis.

Primary outputs:

- full-service probability and Wilson interval;
- conditional fleet makespan;
- total operating time;
- travel, charging, queue, and service time;
- energy consumed and terminal reserve;
- vehicles used;
- failure-cause distribution;
- inference latency and scaling curve.

## Campaign 4 — Component ablations

Purpose: identify why GraphPPO works.

- [ ] no mask;
- [ ] hard feasibility mask only;
- [ ] mask plus candidate pruning;
- [ ] flat state plus mask;
- [ ] DeepSets state plus mask;
- [ ] heterogeneous state graph plus independent action scorer;
- [ ] complete action GCN;
- [ ] self-attention action encoder;
- [ ] without active-truck embedding;
- [ ] without global pooling;
- [ ] without queue features;
- [ ] without uncertainty edge features;
- [ ] relation removal study;
- [ ] two/three/four graph layers;
- [ ] mask safety policy variations.

Run on at least one medium, one congested, and one transfer setting with five seeds each.

## Campaign 5 — Charging and congestion sensitivity

- [ ] charger power mix: 50, 150, 350, and 750 kW scenarios;
- [ ] port capacity and fleet-to-port ratio;
- [ ] homogeneous versus heterogeneous infrastructure;
- [ ] target-SoC action resolution: 5% versus 10%;
- [ ] duration actions: 15, 30, and 60 minutes;
- [ ] linear versus nonlinear charging;
- [ ] FCFS queueing versus reservation-aware variant;
- [ ] station closure and charger outage stress tests;
- [ ] action-pruning/runtime tradeoff.

## Campaign 6 — Uncertainty and robustness

- [ ] deterministic reference;
- [ ] travel uncertainty only;
- [ ] energy uncertainty only;
- [ ] service-time uncertainty only;
- [ ] combined uncertainty;
- [ ] correlated travel-energy uncertainty;
- [ ] low/nominal/high variance;
- [ ] heavier-tailed or misspecified test distribution;
- [ ] temporal shift in peak-congestion periods;
- [ ] risk-sensitive/CVaR training as an optional variant.

Report reliability-performance tradeoffs rather than only average reward.

## Campaign 7 — Generalization

- [ ] interpolation across fleet/customer size;
- [ ] extrapolation to larger fleets and customer sets;
- [ ] unseen road graph or geographic region;
- [ ] unseen charger density/location layout;
- [ ] unseen port capacities and charger powers;
- [ ] unseen truck battery and consumption parameters;
- [ ] unseen demand spatial distribution;
- [ ] unseen uncertainty distributions;
- [ ] curriculum-trained universal policy versus size-specific policies;
- [ ] zero-shot versus limited fine-tuning.

The paper must distinguish within-distribution size transfer from genuine out-of-distribution generalization.

## Campaign 8 — Optimizer and heuristic validation

- [ ] Publish the exact small-instance formulation.
- [ ] Publish rolling-horizon/scenario optimizer assumptions.
- [ ] Publish heuristic pseudocode and complexity.
- [ ] Report optimality gaps where bounds exist.
- [ ] Report MIP status, gap, node count, runtime, and time-limit termination.
- [ ] Report every emergency/fallback action separately.
- [ ] Compare methods under equal information assumptions.
- [ ] Plot solution quality versus compute budget.

## Campaign 9 — Paper artifacts

- [ ] Generate every table and figure from versioned result files.
- [ ] Include training-seed and scenario-level uncertainty.
- [ ] Create a machine-readable table manifest mapping manuscript artifacts to scripts/data.
- [ ] Create a point-by-point reviewer response document.
- [ ] Add a notation table and formulation appendix.
- [ ] Add an experiment configuration appendix.
- [ ] Add code tests and LaTeX CI.
- [ ] Archive final checkpoints, manifests, logs, and per-episode results.
