# Generalist GraphPPO campaign: beat the conservative day-ahead schedule

Plan date: 2026-10-20. Branch `campaign/ieee-tits-revision` at `eecdaab`. This is a campaign design. No job was submitted and no code was changed while preparing it. Every number quoted from earlier work comes from `california_graphppo_progress_20261019.md` and `california_graphppo_status.md`; those are reused development results on 128 opened cases, not validation or test evidence.

**Recommendation.** Keep the proposed method (heterogeneous variable state graph + variable feasible-action graph + PPO) and attack the two reasons it currently loses, in this order:

0. **Move the primary benchmark to a long-haul case where charging is unavoidable.** On the current California 10/30 case, the day-ahead plan barely charges (about 2 sessions, 6% of operating time). The new `config_joint_texas_longhaul.yaml` and `config_joint_pacific_northwest_longhaul.yaml` place every customer at least 225 kWh from the depot, so every route must charge. Charging then takes 22–24% of operating time (Section 1b).
1. **Measure headroom before training.** Quantify, per uncertainty regime, how far the day-ahead schedule is from a hindsight (clairvoyant) reference. Superiority is only claimable where that gap exists. Freeze the regime list before any new policy is trained.
2. **Close the combinatorial gap.** The policy loses on travel and fleet activation, not on charging or adaptation. Give the state graph fleet-assignment relations, train against a per-scenario day-ahead baseline, and test a plan-conditioned variant whose starting point *is* the day-ahead schedule.
3. **Make "variable state / variable action" real.** Train one policy over mixed fleet sizes, customer counts, uncertainty regimes and three state-scale maps, then evaluate zero-shot on held-out case studies. This is the generalization claim that a per-instance planner cannot make cheaply, and the one the method's design is built for.

---

## 1. Where the method stands and why it loses

### The bar

"Conservative day-ahead schedule" in this codebase means `alns_planned` / `cpsat_planned`: one nominal plan solved at reset with energy safety factor 1.2 (ALNS: 20,000 iterations, 10 s), executed open loop through the greedy energy-safe navigator in `EVRoutingEnv/baselines/heuristics/alns.py` (`respect_planned_routes=True`, `replan_on_completion=False`). Charging is decided reactively by the navigator; routes and assignment come from the plan.

| Method (128 dev cases, California 10 trucks / 30 customers) | Completion | Successful hours | Paired BKS excess |
| --- | --- | --- | --- |
| Day-ahead ALNS (`alns_planned`) | 98.44% | 112.11 | 1.73% |
| Day-ahead CP-SAT (`cpsat_planned`) | 99.22% | 116.27 | 5.15% |
| Best GraphPPO (`edge_route_cost40`, 3 seeds, 1M) | 98.96% | 129.37 | 16.92% |
| GraphPPO standard action GCN (`edge_control_cost40`) | 99.22% | 131.36 | 18.04% |
| Re-planning ALNS (`alns`) | 100.00% | 263.17 | 137.98% |
| Heuristic / MPC | 96.88% / 94.53% | 277.30 / 316.57 | 150.19% / 179.34% |

### Diagnosis

- **The gap is combinatorial.** For the best GraphPPO, travel contributes 16.59 of the 16.92 excess points; charging contributes 0.32. The policy activates 5.72 trucks on average against 4.27 for the references. This is assignment and sequencing quality, which is exactly what a 20,000-iteration ALNS optimizes well.
- **The nominal California regime gives closed-loop control little to win.** Traffic σ-factor 0.15, energy multiplier 0.90–1.20 and 209 ports let the day-ahead plan succeed 98.4–99.2% of the time. The paper's joint eVRP result (GraphPPO 84.3% vs CP-SAT 62.2% success) came from a regime where open-loop plans broke. In this regime, the plan rarely breaks.
- **Further PPO alone has plateaued.** Warm-cost 1M → 4M moved excess from 18.14% to 18.68% (selected) / 19.91% (final). Shields, progress features, capacity links, retained imitation and critic detachment were all screened out. Repeating these does not help.
- **The "other baselines" row needs repair.** Re-planning ALNS at 263 h is more than twice the cost of the same solver used once. A reviewer will read it as a broken strawman. It must be audited or replaced by a correct rolling re-planner before it appears in a table.
- **The implementation is not yet variable-schema.** `CanonicalPolicyShape` carries fixed `max_trucks` / `max_customers`, the action head is ragged but the state is padded, and physical-node relabeling changed 22–26% of greedy choices. The identity-relation adapter (array 28367215) fixes the relabeling sensitivity on 128 train states; it is not yet a route-cost result.

### Consequence for the claim

"Better than day-ahead" must be a falsifiable, preregistered statement per regime, not a regime chosen after seeing results. The campaign therefore uses two claim types:

| Regime class (decided in Phase 0, frozen) | Claim | Test |
| --- | --- | --- |
| **Adaptation regimes:** day-ahead completion ≤ 97% *or* day-ahead-to-hindsight gap ≥ 5% | Superiority | Completion ≥ day-ahead and paired joint cost (below) lower, 95% upper bound < 0 |
| **Nominal regimes:** neither condition holds | Non-inferiority | Completion ≥ 99%, paired successful-hours difference upper bound ≤ +2% |
| **Held-out case studies** (zero-shot) | Generalization | Same tests as their regime class, policy never trained on that map |

---

## 1b. Realistic long-haul case: trips that require charging

### Why the California case is not enough

Measured on 2026-10-20 by executing the day-ahead ALNS plan (`alns_planned` settings) on fresh seeds:

| Case (10 trucks / 30 customers unless noted) | Seeds | Day-ahead completion | Makespan (h) | Operating (h) | Charging share | Sessions | Queue (h) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| California default, 200 h horizon | 10 | 10/10 | 34.8 | 92.4 | 5.7% | 2.2 | 0.0 |
| California, depot ring 200 kWh, 200 h | 10 | 9/10 | 72.9 | 220.9 | 9.9% | 9.1 | 0.0 |
| California, depot ring 200 kWh, 48 h | 10 | 7/10 (2 not generable) | 38.2 | 138.7 | 8.8% | 4.7 | 0.0 |
| Texas default, 200 h | 10 | 10/10 | 35.5 | 88.5 | 20.8% | 6.6 | 0.0 |
| **Texas long-haul** (ring 225 kWh, 48 h) | 15 | 15/15 | 39.0 | 130.2 | 22.2% | 10.8 | 0.0 |
| Pacific Northwest default, 200 h | 10 | 10/10 | 20.9 | 58.7 | 18.6% | 3.8 | 0.0 |
| **Pacific Northwest long-haul** (ring 225 kWh, 48 h) | 15 | 15/15 | 35.7 | 108.9 | 24.4% | 9.7 | 0.0 |
| Texas, 20 customers, ring 200 kWh, 36 h | 10 | 6/10 | 32.0 | 87.2 | 22.1% | 7.0 | 0.0 |

These are small development samples for choosing a benchmark, not evaluation results.

- **California's time data are the problem.** Energy is plausible (median 1.57 kWh/km), but the median leg covers about 75 km in 14 h, an effective 5 km/h. A 400 kWh battery therefore lasts about 19 driving hours. Forcing long trips on California needs multi-day horizons, and inside 24–36 h most ring instances cannot be generated.
- **The OSM-built state cases are consistent.** Texas, Pacific Northwest, Germany/Netherlands/Belgium and China use the documented heavy-truck model at about 40 km/h effective speed. Their median legs (466–525 kWh) exceed the battery.
- **Queues never bind at 10 trucks** on these maps (hundreds of real DC-fast sites). Congestion must come from the `ports_scarce` regime or a larger fleet, not from the base case.
- **The day-ahead ALNS ignores the horizon.** With 20 customers and a 36 h horizon it packs routes onto 3 trucks and misses the deadline on 4 of 10 Texas instances. Before horizons are tightened, the planner must model `max_time`, or that comparison is a strawman (Phase 0 step 0.7).

### New generator knob

`environment.min_depot_energy_kwh` / `max_depot_energy_kwh` (nominal kWh, the smaller of the two directions) restrict customers to a ring around the depot. With a floor `F` and minimum energy multiplier 0.90, every round trip needs at least `2 × 0.90 × F`. At `F = 225` this is 405 kWh, above the 400 kWh battery, so **every truck that serves a customer must charge at least once**. Unset bounds keep the original generator draws exactly (tested). The scenario cache key already hashes the full config.

### Exact workload (10TxD): withdrawn on 2026-10-20

An exact per-truck workload (`problem.deliveries_per_truck: x`) was prototyped and **withdrawn before it was committed**. The classic E-VRP (Schneider et al. 2014, Montoya et al. 2017, Froger et al. 2022) has no per-truck customer quota; routes are bounded only by payload, battery, duration and time windows. The environment now rejects the key with a `ValueError`. Every benchmark uses free assignment, so fleet sizing remains part of the decision and the Section 1 diagnosis (5.72 against 4.27 active trucks) still applies. See `docs/campaigns/dynamic_stochastic_etruck_model_plan.md`, Phase 0.

Free-workload rows, measured with the day-ahead ALNS on 12 fresh seeds per row:

| Case | Completion | Trucks used | Operating hours | Charging hours | Charging sessions |
| --- | --- | --- | --- | --- | --- |
| California 10T30C | 12/12 | 4.0 | 88.3 | 4.7 | 1.8 |
| Texas long-haul | 12/12 | 4.2 | 138.5 | 31.4 | 11.7 |

The greedy heuristic completed 9 of 12 Texas long-haul instances.

### Benchmark roles

| Role | Config |
| --- | --- |
| Primary benchmark (training, selection, headline test) | `config_joint_texas_longhaul.yaml` |
| Second training map | `config_joint_pacific_northwest_longhaul.yaml` |
| California 10T30C | `config_joint_california_10t30c.yaml` (free assignment, as in the earlier `ca10t3d_v1` cohort) |
| Held-out long-haul case studies | Germany/Netherlands/Belgium and China with the same ring and horizon, after the Phase 0.5 generation check |

---

## 2. Metrics and success criteria (frozen before Phase 2)

Retain the existing success-first ordering and add one joint metric so that a method which declines hard cases is not flattered.

- **Completion:** all customers served and every truck at the depot, from operational `success`. Wilson 95% interval.
- **Successful operating hours:** travel + charging + queueing + waiting + service, on successful episodes. Paired against each baseline on jointly successful scenarios.
- **Joint cost (new, primary for superiority):** realized operating hours on success; on failure, realized hours plus a recovery charge equal to the scenario's hindsight-reference cost (a full replacement run). This is fixed now, applied identically to every method, and reported next to the lexicographic view.
- **Paired difference vs day-ahead:** per-scenario `(J_policy − J_dayahead) / J_dayahead`, hierarchical bootstrap over scenarios and training seeds. "Day-ahead" is the better of `alns_planned` and `cpsat_planned` *as a method* (chosen on validation), not a per-case minimum.
- **References, not bars:** per-case BKS pool, hindsight bound, perfect-information solver where tractable. Report excess against them as before.

**Campaign success** requires all of the following with five training seeds on the locked test banks:

1. Nominal Texas long-haul: completion ≥ 99% and successful hours non-inferior to day-ahead within +2%. California is reported with the same tests for continuity.
2. At least half of the frozen adaptation regimes: superiority on joint cost and completion not lower than day-ahead.
3. At least two held-out case studies: the same result as their regime class, zero-shot.
4. Plan-free GraphPPO must meet (2) and (3) on its own for the generalization headline. The plan-conditioned variant is reported as a separate method.

---

## 3. Phase 0: headroom and baseline repair (CPU work, no training)

This phase decides where superiority is possible. It runs while Phase 1 code is written.

| Step | Work | Output | Gate |
| --- | --- | --- | --- |
| 0.1 | Audit re-planning ALNS (`replan_on_completion=True`). Build a correct rolling re-planner: re-solve from the current fleet state at every customer completion, warm-started from the previous plan, same nominal costs and safety factor. | Fixed `alns_rolling` baseline with replay-verified episodes | Rolling cost must not exceed day-ahead cost on nominal California, or the bug is not found |
| 0.2 | Tune the day-ahead safety factor per regime on validation (1.05 / 1.10 / 1.20 / 1.30). | Strongest conservative day-ahead per regime | Choose on validation only |
| 0.3 | Hindsight reference: ALNS solved on realized costs. The seed defines the full uncertainty field, and `TrafficSimulator.realize_journey(..., occurrence=)` reads it without consuming state. Use `solve_perfect_information --information-mode realized` on small cases to validate it. | Per-scenario optimistic reference | Replay-verified |
| 0.4 | Regime sweep on 128 development cases using the existing `REGIMES` in `scripts/evaluation/run_generalization_campaign.py`, plus two combined stress regimes. | Day-ahead completion, cost and hindsight gap per regime | Classify adaptation vs nominal regimes, then **freeze** |
| 0.5 | Case-study screen: fraction of scenarios needing a charging stop, generation acceptance and day-ahead completion for every map in `EVRoutingEnv/data/case_studies/`. | Held-out case list | Only charging-relevant maps count as held-out tests |
| 0.6 | Imitation ceiling: execute the day-ahead plan through the policy's own action space and mask. | Cost a perfect plan follower would get | Confirms the plan-conditioned variant can reach day-ahead quality |
| 0.7 | Give the day-ahead ALNS and CP-SAT planners the episode horizon (`max_time`) as a route-duration constraint, and add a 36 h long-haul variant once they respect it. | Horizon-aware day-ahead baseline | No horizon-only failures on witness-feasible instances |

Regimes evaluated in 0.4 on Texas long-haul (primary), Pacific Northwest long-haul and California default (all already defined except the two combined ones):

| Regime | Change | Why it matters |
| --- | --- | --- |
| `in_distribution` | none | Nominal California |
| `traffic_severe` | σ 0.30, rush-hour ×3 | Plans miss arrival times |
| `energy_severe` | energy 0.8–1.5× | Conservative reserve breaks or wastes |
| `ports_scarce` | ⅓ of ports | Queues bind; FCFS congestion is endogenous |
| `service_slow` | 0.4 h service | Paper's known transfer weakness |
| `battery_small` | 300 kWh | Paper's known transfer weakness |
| `stress_realistic` (new) | rush-hour ×2.5, ports ×0.5, energy 0.85–1.35 | Combined, plausible bad day |
| `stress_outage` (new) | 20% of stations unavailable, unannounced at plan time, observable online | Pure recourse value; needs a station-availability event in the simulator |

Expected result, stated so it can be falsified: nominal Texas long-haul and California have a small hindsight gap and becomes a non-inferiority regime; `ports_scarce`, `traffic_severe`, `stress_realistic` and `stress_outage` become adaptation regimes.

---

## 4. Method changes (Phase 1 implementation)

All changes keep the hetero state GNN, complete-GCN variable action head with routing gates, masked categorical policy and PPO. Each change has its own switch so Phase 2 can attribute it.

### 4.1 Genuinely variable state (portability)

- **Ragged state batching.** Replace the fixed `max_trucks` / `max_customers` envelope with per-graph segment indices, in the same way the action head is already ragged. Gate: on 10/30 California, padded and ragged forward passes give identical logits, values and gradients within float tolerance.
- **Relational identity.** Adopt the identity adapter from array 28367215 as the default for generalist arms: no ordinal node, task or truck IDs; equality, ownership, destination and claim as relations. Gate: logits invariant under node relabeling (already passing on 128 train states).
- **Bounded charger subgraph.** Station counts range from 25 (California) to 432 (Texas) and 469 (China). Each state includes the K chargers ranked by detour cost from active trucks and open customers, with K a fixed hyperparameter (start 32). Distances to excluded chargers are summarized in a pooled feature. Charger destinations offered as actions come from the same subgraph, and the gate is that it never removes a destination the full mask would allow on the day-ahead reference routes.
- **Physically normalized features.** Energy over battery capacity, time over horizon and over map median leg time, payload over capacity. Gate: feature distributions on California, Texas and Pacific Northwest overlap (report per-feature quantiles).
- **Context node.** A global node carrying battery capacity, charger power classes, travel σ-factor and rush multiplier, energy band, base service time and port scale. Operators know these values, and the planners receive them too. This targets the paper's documented miscalibration under battery and service shifts.

### 4.2 Fleet-assignment relations (cost)

Targets the 5.72 vs 4.27 truck gap and the travel excess.

- For each customer candidate: the active truck's marginal insertion cost, the best other truck's observable marginal cost from its committed ETA and remaining payload, and the regret between them.
- For the depot / retire candidate: remaining unclaimed demand and whether trucks already active can cover it by payload and energy.
- Truck–customer "cheapest-server" edges in the state graph, recomputed each event from observable commitments only.

### 4.3 Day-ahead-relative reward (training signal)

- Precompute, for every training scenario, the executed cost of the day-ahead schedule on that same seed. The seed fixes the uncertainty field, so this is a paired per-scenario baseline. It is used only in the reward and is never observed by the policy.
- Dense reward `−Δhours / C_dayahead(scenario)` plus a terminal term `(C_dayahead − C_policy) / C_dayahead` and the existing failure penalty. Returns are then comparable across sizes, maps and regimes, which mixed training needs. The objective becomes "beat day-ahead on this scenario" directly.
- Keep gamma 1 and the cost40 shaping as the control arm.

### 4.4 Plan-conditioned GraphPPO (separate method, `GraphPPO-P`)

- At reset, solve the same day-ahead plan the baseline uses. Add it to the state graph as relations: truck → planned next customer, customer → planned successor, planned charging stops. Add per-candidate "on plan" and "deviation cost versus plan" features.
- The policy still selects every action and may deviate at any event. BC from the plan executor gives a day-ahead-quality start (Phase 0.6 measures the ceiling), and PPO with the reward in 4.3 learns when deviation pays.
- **Plan dropout** of 0.3 during training, so one network works with or without a plan. The plan-free mode is the generalization headline. The plan-conditioned mode is reported separately, with planner time included in its latency.
- Report the deviation rate and the cost change on deviating decisions, so the result can be attributed to recourse rather than imitation.

---

## 5. Training distributions

| Arm | Maps | Fleet / customers | Regimes | Purpose |
| --- | --- | --- | --- | --- |
| C (Texas long-haul specialist) | Texas long-haul | 10 / 30 | Nominal | In-distribution control; BC from Texas day-ahead plans, warm start from `edge_route_cost40` |
| G-size | Texas long-haul | trucks 5–20, customers 15–60 | Randomized nominal → severe, context node on | Size and regime generalization on one map |
| G-map | Texas + Pacific Northwest long-haul, plus California default | as G-size | as G-size | Map generalization (headline) |
| G-LOMO | Pacific Northwest long-haul + California only | as G-size | as G-size | Texas becomes a held-out map: tests "new case study" on the primary benchmark |

**Held-out case studies** (final list frozen by Phase 0.5): `de_nl_be` and `china_jinghu` with the long-haul ring and horizon, `europe_10k` (its `c300` / `c500` charger configs are missing from this checkout), and the metros (`chicago`, `seattle`) with the battery reduced so that charging binds. Texas has 432 charger nodes and China 469, so the bounded charger subgraph (4.1) is required, not optional. These include a different continent, OSM-derived chargers and modelled service-area chargers.

Demonstrations for BC come from the day-ahead executor plus the existing mask-respecting recovery expert (DAgger-style mixtures at 80% and 50% teacher control). Use 4,096 scenarios per map and size band. BC labels never enter evaluation.

---

## 6. Phases, arms and gates

Interaction budgets count environment decisions. Each stage saves full model, optimizer, normalizer and RNG state, and continues rather than restarting, as the current campaign already does.

| Phase | Runs | Budget per run | Decision gate |
| --- | --- | --- | --- |
| P0 headroom (Section 3) | none | CPU only | Regime classes and held-out maps frozen |
| P1 implementation gates | 5 pilot arms | 4,096 decisions | Ragged = padded equivalence, relabel invariance, finite gradients in new branches, checkpoint reload |
| P2 Texas long-haul screen | 4 new arms × 3 seeds = 12, plus control C0 reused | BC20 + 1M | Promote at most 2 arms: completion ≥ 98.5% and lowest paired hours versus day-ahead |
| P3 Texas long-haul promotion | 2 arms × 3 seeds | 1M → 4M | Nominal non-inferiority trend (upper bound ≤ +5% at 4M) or stop |
| P4 generalist training | G-size, G-map, G-LOMO × 3 seeds = 9, using the P3 recipe | BC + 4M, gated to 12M | G-map must not lose more than 2 points of Texas long-haul excess versus the specialist; adaptation-regime superiority on development banks |
| P5 finalists | ≤ 2 recipes (specialist + generalist) × 5 seeds | 12M | Selection on 512 disjoint validation cases per regime and map |
| P6 locked evaluation | Frozen | Evaluation only | Test once; no redesign after opening |

P2 arms (each adds one change to C0, `edge_route_cost40` with BC warm start):

| Arm | Change |
| --- | --- |
| C0 | Control, reused 3-seed actors (not retrained) |
| C1 | + fleet-assignment relations (4.2) |
| C2 | + day-ahead-relative reward (4.3) |
| C3 | + plan-conditioned input with dropout (4.4) |
| C4 | C1 + C2 + C3 |

C4 is a bundle, so P5 adds leave-one-out controls (C4 − C1, C4 − C2, C4 − C3) at 4M if C4 is promoted.

Not repeated: critic-encoder detachment, completion shields, progress/links/capacity cohorts, retained imitation. These were screened out with recorded evidence.

### Evaluation matrix (P6)

Every method is scored on identical seeds per cell.

| Methods | Rows |
| --- | --- |
| Proposed | GraphPPO plan-free (specialist and generalist), GraphPPO-P |
| Day-ahead | `alns_planned`, `cpsat_planned`, each with the per-regime tuned safety factor |
| Online | `alns_rolling` (Phase 0.1), MPC, greedy heuristic |
| Learned | MaskPPO and the constructive attention model, trained on the same distribution and budget as the generalist |
| References | Hindsight ALNS, per-case BKS pool; perfect-information solver on small instances |

| Bank | Size per cell |
| --- | --- |
| Nominal Texas long-haul test | 1,000 (primary) |
| Nominal California test | 500 (continuity) |
| Each adaptation regime | 500 |
| Each held-out case study (nominal + one stress regime) | 500 |
| Validation for selection | 512 per regime and map, disjoint from test |

Report per cell: completion (Wilson), joint cost, paired difference versus day-ahead with hierarchical bootstrap, failure categories, trucks used, charging sessions and queue time, and decision latency including any planner time.

---

## 7. Compute envelope

Estimates use the measured rates from the current campaign: BC20 + 1M on one MIG slice took 93–112 minutes, PPO runs at roughly 190–240 decisions/s, and MIG costs 64 SBU per slice-hour. Mixed-size and multi-map graphs are assumed to run at 120 decisions/s until P1 measures them. Only `gpu_mig` and `gpu_a100` are used, and concurrent allocations stay ≤ 30 across all arrays, including the arrays still running.

| Item | Estimate |
| --- | --- |
| P0 hindsight, rolling and tuned day-ahead on 128 cases × 8 regimes, plus case-study screen | ~1.5k SBU |
| Precomputed day-ahead costs for training banks (≈60k scenarios × 10 s) | ~1.2k SBU |
| Demonstration collection across maps and size bands | ~0.6k SBU |
| P2 (12 runs × 1M) and P3 (6 runs, 1M → 4M) | ~3k SBU |
| P4 (9 generalist runs to 12M at 120 decisions/s) | ~16k SBU |
| P5 (≈10 additional or continued runs to 12M) | ~15k SBU |
| P6 references and evaluation | ~4k SBU |
| **Total** | **~41k SBU**, inside the 50k internal planning ceiling; the 18 October snapshot showed about 504k remaining, to be rechecked before launch |

---

## 8. Risks and stopping rules

- **The nominal long-haul case may stay at parity.** If P0 finds a hindsight gap below 3% and no P2 arm reaches non-inferiority by 4M, report the nominal case as a non-inferiority-or-worse result and move the compute to adaptation regimes and held-out maps. Do not extend training to chase it.
- **The plan-conditioned variant may only imitate.** If its deviation rate is near zero, or deviations do not lower cost, it is a plan follower and is reported as such. The plan-free policy keeps the generalization claim.
- **Multi-map training may dilute California quality.** The P4 gate (≤ 2 points of excess lost) catches this. If it fails, report the specialist for California and the generalist for transfer.
- **Baseline strength.** Tuned safety factors, the fixed rolling re-planner, and learned baselines trained on the same distribution are required. Without them, a superiority claim will not survive review.
- **The `stress_outage` regime needs a simulator change.** It needs a station-availability event revealed online. If that is not ready by P2, drop the regime rather than approximating it with port scaling.
- **Compute fairness.** Report the planner's 10 s per scenario and the policy's per-decision latency. GraphPPO-P pays both.

---

## 9. Implementation order

1. Done: depot-energy ring in the generator (`EVRoutingEnv/models/environment/route_walk.py`), long-haul Texas and Pacific Northwest configs, tests. Next: BC demonstrations and precomputed day-ahead costs on Texas long-haul.
2. Phase 0 scripts: rolling ALNS audit and fix (`EVRoutingEnv/baselines/heuristics/alns.py`), hindsight ALNS on realized costs, two new regimes in `run_generalization_campaign.py`, case-study screen. Launch these on CPU while step 2 proceeds.
3. Ragged state batching and bounded charger subgraph in `algo/canonical_state.py` / `algo/canonical_policy.py`, with the padded-equivalence test.
4. Context node and normalized features in `EVRoutingEnv/state/features.py`; multi-map and mixed-size environment sampler, reusing `curriculum_env.py` sampling but with ragged shapes.
5. Fleet-assignment relations, day-ahead plan relations and plan dropout, each behind its own trainer flag in `scripts/training/train_canonical_ppo.py`.
6. Day-ahead-relative reward using a precomputed per-seed cost table, with the scenario bank hashed into the run manifest.
7. P1 GPU pilot, then P2 through the existing controller and promotion tooling.

## 10. What this campaign will not claim

- Certified optimality, unless a valid lower bound exists for the exact simulator problem.
- Support for arbitrary graph schemas. The claim covers the EVRP state and action semantics across maps, sizes and regimes.
- Superiority in regimes classified as nominal in Phase 0.
- That the plan-conditioned result is plan-free GraphPPO.
