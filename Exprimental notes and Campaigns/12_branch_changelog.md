# Branch Changelog: `campaign/ieee-tits-revision`

Compiled: 2026-10-16, from `git log main..HEAD` (198 commits, merge-base
`e58ffdd`) plus the uncommitted working tree.

Scope against `main`: 248 files changed, +73,323 / -3,634 lines. Two eras:

| Era | Commits | Dates | Diff |
| --- | --- | --- | --- |
| A. Environment and method build (pre-revision) | `7a060d5` … `beec506` | 2025-11-05 → 2026-04-27 | 108 files, +33,283 / -3,633 |
| B. IEEE T-ITS revision campaign | `1459e54` … `361a9b1` | 2026-08-11 → 2026-08-18 | 176 files, +40,854 / -815 |
| C. Uncommitted | working tree | 2026-08-20 | 3 paths |

---

## Era C — Uncommitted working tree

- `10_travel_time_objective_campaign.md`: new §7.5, *Re-verification against
  ALNS (2026-08-20)*. Re-tunes all four baselines (40 validation scenarios,
  travel objective) and re-scores the frozen `c_tm15` checkpoint on the same
  300 test seeds. Results: GraphPPO 0.843 success / 121.5 h; ALNS 0.603 /
  119.9 h; CP-SAT 0.607 / 121.1 h. Paired vs ALNS +1.4 h [-3.4, +6.7], i.e.
  *tied*. §7.1's 0.857 is marked superseded. Adds an artifact note: the
  2026-08-18 `results/` restructure deleted `campaign_travel_final/`,
  `selected_travel_d.json`, `frozen_baselines_travel.json`, `tuning_travel/`
  and all of stage D. The live set is now `*_v3`.
- `README.md` (campaign dir): status table row for ALNS marked done, and the
  doc 10 summary extended with the ALNS re-verification. **The rest of this
  table is stale**: it still says ablations, attention baselines,
  generalization and the manuscript are "not started", though all are done.
- `results`: committed as a symlink to its own absolute path
  (`/home/sorfanoudakis/EVRP/results`, mode 120000). The working tree has
  replaced it with a real directory, so git reports it as deleted. **Resolve
  this before committing**, either by removing the symlink from the index and
  ignoring `results/`, or by restoring it.

---

## Era B — IEEE T-ITS revision campaign (2026-08-11 → 2026-08-18)

### B1. Planning and tracking (2026-08-11)
- `1459e54` Code-structure refactor that serves as the revision baseline.
- `be712ca` Reviewer-revision and experimental-campaign plan (docs 00–03).
- `f212708` Reviewer-response matrix (doc 04) and execution checklist (doc 05).
- `5558702` Unit tests for the `EVRoutingEnv` components.

### B2. Simulator correctness and the joint fleet model
Delivered across `a11f92c`, `a12bdfe` and the feature commits below (docs 05
P0, 06, 07):
- Stable scenario IDs, with isolated RNG streams for instance, travel, energy
  and service randomness. Replay and isolation tests added.
- Unloading time is sampled once per service event.
- Fleet-owned `CustomerTask` registry (`models/core/customer.py`), plus truck
  payload capacity and served-task accounting.
- Joint instance generator (`joint_instance.py`): common depot, unique
  customers, static reachability, capacity-feasible demand. Customer
  completion is committed after unloading. Depot return is enforced.
- Pure hard-feasibility engine (`state/feasibility.py`) with rejection reason
  codes. Explicit invalid-action, empty-feasible-set and payload-deadlock
  outcomes. Top-k pruning and forced fallbacks removed.
- Charging is now a target-SoC action at 50/60/70/80/90/100%. Level2/DCFast
  classes are preserved. 150/350/750 kW joint-fleet powers are configurable.
- FCFS queues hardened against duplicate wake-ups, stale waiters, port
  overbooking and double occupancy. Joining a full queue is now a valid choice.
- Hard-time-window variant, disabled by default.
- `3baa359` Station-specific charging efficiency
  (`charging.station_efficiency_overrides`).
- `6e2526c` Port-capacity scaling (`charging.port_capacity_scale`).
- `43dd7f3` **Fix:** `ChargingCurveModel.estimate_charge_time` could not
  converge at target 1.0 and returned its range midpoint (10 h for a 0.53 h
  charge). It now integrates directly to the target.
- `44e52d5` **Fix:** weak stations broke on the taper floor.

### B3. Canonical learning stack
- Versioned canonical fleet observation (`state/features.py`,
  `representations.py`) with flat, padded-set and heterogeneous-graph
  adapters. The pairwise information-parity gap is closed (doc 08 §1).
- `algo/canonical_encoders.py`, `canonical_policy.py`, `canonical_ppo.py`,
  `action_heads.py`: independent, complete-GCN and self-attention action heads.
  Permutation-equivariance and isolation tests added. The hard-mask path fails
  loudly on bad masks.
- `ROUTING_ACTION_FEATURES`. Matched ablation: 0.700 vs 0.213 without them
  (doc 10 §6.2).
- `WorkerCanonicalVecEnv` parallel rollouts, proven step-equivalent to the
  synchronous version. `4322b0a` makes the worker picklable. `897c5ea` pins
  one torch thread per worker.
- Behaviour cloning and demonstration collection (`behavior_cloning.py`,
  `collect_demonstrations.py`).
- GraphPPO trained from scratch via the energy-ramp curriculum (doc 09).
- **Objective change:** fleet makespan → total fleet travel time, with a
  re-tuned reward and objective-aware checkpoint selection (doc 10).

### B4. Baselines
- Fleet-level CP-SAT nominal planner (`exact_optimization.py`) and
  rolling-horizon MPC. Every baseline prices charging with the simulator's own
  nonlinear integrator.
- `893d7c1` **Fix:** CP-SAT could not leave a truck idle, so it returned plans
  labelled OPTIMAL that were worse than optimal. An exhaustive-enumeration
  cross-check was added (`validate_exact_objective.py`). After the fix CP-SAT
  matches enumeration on 30/30 tiny instances.
- `bc6202b` ALNS (`baselines/alns.py`): random, worst, Shaw and route destroy
  operators; greedy and regret-2 repair; adaptive weights; simulated-annealing
  acceptance. It reaches the enumerated optimum on all 30 instances.
- `bc6202b` Attention/transformer encoder (`AttentionStateEncoder`), with
  typed edge features used as an attention bias.
- DeepSets-PPO, state-GNN PPO, flat MaskPPO-analogue and a no-mask arm, all
  trained under equal information.
- `88608da`, `5bba705` The inherited per-truck Gurobi MILP is renamed in code.
  Legacy entry points are marked retired, with their replacements named.
- Solver diagnostics (status, bound, gap, runtime, fallbacks) are stored on
  every episode row.
- `700ca9c` Per-truck MILP *formulation* validated by enumeration
  (`validate_per_truck_model.py`). Its "≤1 charger per leg" restriction binds
  on 1 of 60 plans.
- Per-truck Gurobi *encoding* validated (`validate_per_truck_gurobi.py`, doc
  11 §11.5). All 11 executable plans match the enumerated optimum. 42/80 plans
  cannot be executed under nonlinear charging.

### B5. Evaluation infrastructure
- `EVRoutingEnv/evaluation/` (`runner.py`, `artifacts.py`,
  `statistics.py`): immutable manifests with git, dependency, hardware and
  seed provenance; failure-retaining aggregates; Wilson intervals; paired
  bootstrap.
- Disjoint train/validation/test seed namespaces.
- Scripts: `run_canonical_campaign.py`, `compare_campaign.py`,
  `build_best_known.py`, `tune_heuristics.py`, `select_architecture.py`,
  `run_generalization_campaign.py`, `canonical_harness.py`, and the batch
  runners under `scripts/runners/`.
- `0556362` Ablation and seed summary that re-scores each run in its own
  environment.
- `9828049` **Fix:** three generalization regimes were no-ops. The network is
  now perturbed.
- `dc9e602` **Fix:** generalization scored each method under the wrong
  environment.
- `44e1078` eTFRP (pre-assigned) setting expressed inside the canonical stack.

### B6. Experiments (docs 08–11)
- 500-scenario test campaign: 16 methods, with three seeds for GraphPPO.
  Headline: 0.843 [0.830, 0.858] success at 123.5 h [120.9, 125.3]. CP-SAT
  0.622, ALNS 0.614. Paired vs CP-SAT +0.2 h [-3.1, +3.7] on 285 jointly
  solved scenarios.
- Mask ablation at three seeds per arm: masked 0.760, unmasked 0.773. The mask
  does not explain feasibility. The penalty sweep shows what does: a
  failure-commensurate cost.
- Seed variance: identical configurations span 0.107 success, which dissolves
  several single-seed conclusions in docs 08–10.
- Component ablations: routing features, typed relations and pooling carry the
  model. Queue features and the active-truck flag are inert at two trucks.
- Attention baseline: 0.788 test, statistically level with the graph encoder.
- Charging: five charging-model variants. Action space: `soc5` gains nothing,
  while duration actions cost 0.073 success and 14 h.
- Optimizer budget: CP-SAT from 0.5 s to 45 s returns an identical plan, and
  ALNS saturates below 2000 iterations.
- Generalization over 20 regimes: strong size transfer. The ranking reverses
  at 300 and 500 kWh batteries and at doubled service times.
- Upward size transfer fails. Under congestion (4 trucks, 1 port) the queue
  mechanism binds, and GraphPPO waits 32% less than CP-SAT.
- eTFRP setting: the mask finding replicates, and pre-assignment makes the
  problem harder. The published eTFRP tables were **not** recomputed.

### B7. Manuscript
- `1d44c9d` Corrects the claims the new evidence contradicts and replaces the
  final results table with the joint fleet eVRP study.
- `5ddb301` Point-by-point response letter (`latex/response_to_reviewers.tex`).
- `ccf5f76` States the evaluation protocol and tempers the seed claim.
- `db3cc0c` Corrects the PPO and MaskPPO columns of the joint table.
- `e4572ad` Appendices for baseline specifications and formulation.
- `4382f71` Tests the findings in the eTFRP setting.
- `abde94e` E1 fleet-literature matrix, plus the static checker
  `scripts/analysis/check_manuscript.py`.
- `5b55347` First real compile: fixes the missing `enumitem`, removes a stray
  caption sentence, and escapes bib characters. `TruckNetwork.pdf` panel c now
  reads "Head". Crossref-verified bus (Jin 2024) and ride-hailing (Shi 2020)
  citations added.
- `361a9b1` **IEEE T-ITS port** in `latex/latexIEEE/`: IEEEtran v1.8b,
  `IEEEtranN`, anonymized, 10-page main paper, plus a 9-page `supplementary.tex`
  holding the appendices, taxonomy, eTFRP transfer studies, component sweep,
  PPO recap, settings table and four figures.

### B8. Documentation
- docs 00–11 in `Exprimental notes and Campaigns/`. Commits `a12bdfe`,
  `1625947`, `698e193`, `0736d3d`, `c3e4f05` and `be44710` refresh the
  checklists as the work progressed.

---

## Era A — Environment and method build (2025-11-05 → 2026-04-27)

- **Nov 2025:** `SimpleTruckEnv` → event-driven architecture
  (`event_handlers`, plotter, statistics), station and waiting-time data, the
  first heuristic, GNN state representation and action graphs, TD3 Action-GNN
  (later dropped), MaskPPO, variable-action PPO (`PPO_VariableActionGNN`),
  charge-duration actions, eval script.
- **Dec 2025:** charging-queue and feasibility-mask debugging, the
  generalization evaluator, Gurobi per-truck optimal solver, SB3 training
  scripts, curriculum learning (mixed/staged/uniform), the CCCV charging-curve
  model, traffic simulation, stochastic unloading, energy uncertainty, flexible
  delivery order, detour-based GNN state, and variable stop counts.
- **Feb 2026:** VRP formulation and optimal VRP policy, top-K deliveries in the
  GNN state, configurable chargers, a parallel evaluator, classic VRP
  heuristics, loop-avoiding feasible actions, comparison and visualization
  scripts, seed-averaged tables.
- **Apr 2026:** experimental-setup documentation, the CCCV power method and
  charging-curve figure, open-source preparation.
- The legacy `truck_env/` package was removed (`truck_env.py`, `utils.py`,
  `reward.py`).
