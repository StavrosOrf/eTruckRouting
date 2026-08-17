# Point-by-Point Reviewer Response TODO

Updated: 2026-08-17

This matrix translates every item in `latex/reviewer_comments.txt` and `latex/more_comments_and_suggestions.txt` into a code, experiment, and manuscript obligation. Responses remain open until they cite a tested implementation, saved result, or exact revised manuscript location.

**State legend.** `[x]` complete with cited evidence; `[~]` partially satisfied, with the missing part named on the following line; `[ ]` not started.

**Standing evidence base.** Correctness suite: 281 tests pass at `a11f92c`. Headline campaign: `results/canonical/campaign_travel_final/test/`, selection frozen in `results/canonical/selected_travel_d.json`, baselines frozen in `results/canonical/frozen_baselines_travel.json`, narrative in `10_travel_time_objective_campaign.md`.

**Standing gap.** `latex/main.tex` is unmodified since `1459e54`, i.e. since before this campaign branch existed. Every item below whose deliverable is manuscript text is therefore open regardless of how much of its code and experiment obligation is discharged.

## Handling editor

### E1 — Incorrect characterization of eVRP as single-vehicle

- [ ] Concede the overstatement explicitly in the response letter.
- [ ] Replace the introduction's single-vehicle/fleet dichotomy, including the claims around `main.tex:122`, `main.tex:371`, and Figure 2 near `main.tex:475`.
- [ ] Build a literature matrix covering electric freight, bus, ride-sharing, shared charging, and fleet-level RL.
- [ ] Cite fleet-level eVRP and electric-fleet scheduling work from primary sources.
- [ ] State that novelty is not the existence of multiple EVs or shared chargers alone.

Evidence required: completed literature matrix, revised related-work taxonomy, and response-letter citations to the new text.

### E2 — Operational uncertainty is already well studied

- [ ] Concede that stochastic/time-dependent EV routing is established.
- [ ] Define the paper's exact combination: joint online assignment/routing, finite-port endogenous queues, nonlinear partial charging, correlated travel-energy uncertainty, and event-driven centralized control.
- [~] Separate exogenous travel/energy/service draws from endogenous queue delays in the formulation and random-variable table.
  - Implemented and tested in the simulator: isolated RNG streams per source (`tests/unit/test_scenario_rng.py`), FCFS finite-port queues (`tests/unit/test_charging_station.py`), correlation and clipping validated in `tests/unit/test_stochastic_distributions.py`. The formulation text and random-variable table do not yet make the separation.
- [ ] Compare against stochastic/robust/rolling-horizon literature rather than claiming an unfilled uncertainty gap.

Evidence required: taxonomy table plus formulation and experiment table that enumerate every stochastic and endogenous quantity.

### E3 — Unclear methodological novelty

- [ ] Recast contributions as falsifiable component claims.
- [~] Run mask-only, state-encoder, action-head, pooling, and active-truck ablations.
  - Done: state-encoder x action-head factorial at a shared budget (`results/canonical/selected_architecture.json`, doc 08 §9) and the routing-action-feature ablation at matched budget, width, and seed stream (`v2_tm10` 0.700 vs `v2_ablate` 0.213 success, doc 10 §6.2).
  - Missing: the mask-only ablation — there is no no-mask training path in `scripts/training/train_canonical_ppo.py` — plus the pooling and active-truck ablations.
- [x] Compare independent, complete-GCN, and self-attention action heads on validation only.
  - `results/canonical/selected_architecture.json`, ranked at the largest shared budget and re-scored on 150 disjoint validation scenarios; doc 08 §9. Caveat to state in the response: this sweep predates the travel-time objective and the `joint-fleet-v3`/routing-feature schema, and the current model inherits `complete_gcn` rather than re-selecting under the new objective.
- [~] Support any scalability claim with quality-versus-runtime curves.
  - Per-decision inference time is recorded for every method in the campaign artifacts (GraphPPO 11.1 ms, CP-SAT 3.9 ms, heuristic 0.8 ms). No quality-versus-runtime curve across instance sizes exists; the XS–L2 scale grid has not been run.
- [ ] Remove broad novelty language if the component studies do not support it.

## Reviewer 1

### R1.1 — Main task is not full fleet routing

- [x] Choose joint fleet assignment, sequencing, routing, charging, and depot return as the primary problem (D1).
- [x] Add fleet-owned customer tasks, payload fields, and service lifecycle foundations.
- [x] Finish canonical joint observation/action semantics at the environment representation layer.
- [x] Add invariant regressions across randomized mixed travel/charge sequences and 20 seeded instances for battery accounting, every-customer-once, capacity, service completion, and depot return.
- [~] Retain preassigned routes only as a clearly named secondary execution benchmark.
  - The preassigned path still exists in `EVRoutingEnv/models/environment/event_driven_env.py` and is no longer the primary formulation, but it has not been renamed as a secondary benchmark and no campaign scores it in that role.
- [ ] Rewrite the problem definition and all main claims around the implemented joint model.

### R1.2 — Mask may explain most of GraphPPO's gain

- [ ] Train PPO without mask using the full equivalent observation.
  - Blocked on code: no no-mask or soft-mask option exists in the canonical PPO trainer. This is the single most load-bearing missing ablation in the revision.
- [ ] Train MaskPPO with identical observation and candidate semantics.
- [~] Compare flat MLP, DeepSets, heterogeneous GNN, and active-truck conditioning.
  - `flat` and `hetero_graph` were swept against all three heads at a shared budget (doc 08 §9). `DeepSetsStateEncoder` is implemented and covered by the same permutation, masking, and gradient tests but was never trained — dropped for compute, not on merit. Active-truck conditioning has not been ablated.
- [x] Compare no action interaction, complete GCN, and self-attention heads.
  - `results/canonical/selected_architecture.json`; same caveat as E3 about the makespan-era schema.
- [ ] Ablate global pooling, active-truck embedding, queue features, and relation types.
- [x] Report feasibility, conditional makespan, queue time, and runtime—not reward alone.
  - `run_evaluation_campaign` publishes success with Wilson intervals, makespan and travel time conditioned on successful episodes with the conditioning stated, queue and charging time, terminal SoC, and seconds per decision; `compare_campaign.py` adds paired bootstrap differences on jointly solved scenarios.

### R1.3 — Notation and equation defects

- [ ] Use distinct symbols for nominal and realized edge energy throughout.
- [ ] Define travel and energy clipping bounds precisely.
- [ ] Rewrite the charging integral with a time-varying SoC trajectory and unambiguous integration variable.
- [ ] Correct Eq. 19 to use sender and receiver embeddings.
- [ ] Correct Eq. 22 to aggregate neighbor action embeddings.
- [ ] Regenerate equations from the final tested feature and network implementations.

All six are pure manuscript work and all six are unstarted. The implementations they must be regenerated from are now stable and tested: `algo/canonical_encoders.py`, `algo/action_heads.py` (the complete action graph is now genuinely complete rather than an adjacent-node chain), `EVRoutingEnv/state/features.py`, and `EVRoutingEnv/models/simulation/charging_curve.py`.

### R1.4 — Optimization benchmark does not establish near-optimality

- [ ] Rename the inherited method `conservative per-truck deterministic MILP` in artifacts and manuscript.
- [~] Add exact/bounded fleet optimization for tiny deterministic instances.
  - `EVRoutingEnv/baselines/exact_optimization.py` (CP-SAT nominal planner) and `optimality_reference.py` are implemented, tuned on validation, and scored in the headline campaign. They do not yet establish optimality: doc 09 §5 reports zero instances proved optimal at an ~85% bound gap, so only a best-known reference (`build_best_known.py`) is quotable.
- [x] Add rolling-horizon/scenario optimization where tractable.
  - `RollingHorizonMPCPolicy` in `EVRoutingEnv/baselines/canonical_baselines.py`, grid-searched on 40 validation scenarios under the travel objective (horizon 6, branching 2, safety 1.15, target SoC 0.8) and scored on the test split.
- [~] Save solver status, bound, gap, runtime, retry, and fallback counters.
  - Solver status is captured (`optimality_reference.py:283`) and runtime is recorded per decision. Bound, gap, retry, and fallback counters are not published in the campaign artifacts.
- [ ] Remove `near-optimal`, `optimization-level`, and equivalent wording unless supported by valid gaps/bounds.
  - Six such claims remain in `latex/main.tex` and none is currently supported.

### R1.5 — High reward despite zero success

- [x] Approve feasibility-first, then makespan evaluation (D4).
- [x] Separate training reward from per-episode operational evaluation metrics.
  - Statistical aggregation is now also complete: Wilson intervals and deterministic paired bootstrap intervals are consumed by the campaign tables (`compare_campaign.py`, `comparison_vs_*.json`).
- [~] Retain every failed episode and classify the failure cause.
  - Retention is guaranteed by the runner and failures stay in every non-conditioned aggregate. Classification is incomplete: on the 300-scenario test split GraphPPO's 43 failures split into 26 `no_feasible_action` and 17 `unspecified_failure`, the latter being the fallback in `EVRoutingEnv/evaluation/statistics.py:53` when no termination reason is set.
- [x] Report full-service probability first and time/cost only with explicit conditioning.
  - Enforced in the artifact schema itself: every metric carries a `conditioning` field, and cost metrics are marked `successful_episodes`. Manuscript tables still have to consume this.
- [~] Explain the inherited `+500` delivery, `-1` time, and `-1000` failure shaping and replace it if needed for stable training.
  - Replaced and re-tuned: the current shaping is success bonus, incompletion penalty, per-leg travel multiplier, terminal travel-time bonus, stranding penalty, and energy-margin bonus, all CLI-exposed in `train_canonical_ppo.py` and persisted per checkpoint. Doc 10 §2.2 and §4 explain the design and measure the penalty-weight sweep. The manuscript explanation and the published coefficient table are open.

### R1.6 — Weak and underspecified baselines

- [ ] Publish current heuristic pseudocode and its exact information assumptions.
  - Doc 08 §3 documents the three rounds of repair and the information the baselines read, in prose. No pseudocode or manuscript appendix exists.
- [ ] Add ALNS or an equivalently strong routing-and-charging metaheuristic.
- [ ] Add a constructive attention/transformer baseline.
- [~] Add DeepSets-PPO and state-GNN PPO with independent action scoring.
  - Every component exists (`deep_sets` and `hetero_graph` encoders, `independent` head) and is unit-tested, but neither has been trained as a baseline under the current objective and schema.
- [x] Equalize observations, masks, training steps, tuning budget, and evaluation scenarios.
  - Holds for the method set actually implemented: pairwise information parity closed (doc 08 §1), identical hard mask for learned and classical policies, shared execution layer, architecture runs compared only at the largest common budget, all baselines re-tuned on 40 validation scenarios under `--objective travel_time`, and all methods scored on the same 300 test seeds.
- [ ] Remove the unsupported statement that PPO is the most capable discrete-action RL algorithm.

### R1.7 — Generalization evidence is too narrow

- [ ] Freeze multiple training seeds before test evaluation.
  - One training seed (`--seed 0`) across the entire v2 ladder. Training-seed variance in the 0.857 headline is unquantified.
- [ ] Use at least 500 paired test scenarios per main setting.
  - Currently 300.
- [ ] Test unseen graphs/regions, charger layouts/powers/capacities, fleet/customer sizes, battery parameters, demand patterns, and uncertainty distributions.
  - No generalization campaign exists under the canonical stack. `scripts/evaluation/generalization_eval.py` is the inherited script: it targets `EventDrivenTruckEnv` with the old GNN space and Gurobi baselines, has hardcoded policy paths, and does not write the artifact contract.
- [ ] Distinguish interpolation, within-simulator size transfer, and genuine OOD tests.
- [ ] Evaluate credible baselines in every transfer regime.
- [ ] Replace the current broad zero-shot claims if only size transfer remains tested.
  - Two zero-shot claims remain in `latex/main.tex`, currently unsupported by any artifact on disk.

## Reviewer 2

### R2.1–R2.3 — Definition, uncertainty, and mandatory service

- [ ] Define the joint problem on page 1 without inventing an eTFRP/eVRP dichotomy.
- [ ] Enumerate demand, travel-time, energy, service-time, and queue uncertainty/endogeneity early.
- [~] Add explicit every-customer-once, capacity, energy, time-window-variant, and depot-return constraints.
  - All five are implemented and regression-tested (`tests/unit/test_customer_registry.py`, `test_truck_payload.py`, `test_feasibility.py`, `tests/integration/test_joint_environment.py`; the hard time-window variant is present and disabled by default). The manuscript does not yet state them.
- [~] Define success, infeasibility, timeout, and incomplete-service outcomes.
  - Implemented as a versioned per-episode outcome schema with explicit failure causes and truncation accounting; the definitions are not yet written into the paper.

### R2.4–R2.6 — Language, reward coefficients, charging discretization

- [ ] Replace `stochastic edge traversing` near `main.tex:776` with `realized travel duration on the selected road-network transition`.
- [~] Publish every numerical training-reward coefficient.
  - Every coefficient is CLI-exposed and persisted in checkpoint configuration, and the selected arm's values are recorded in the stage runners and `results/canonical/travel_methods_final.json`. The manuscript table does not exist.
- [x] Select target-SoC actions at 50/60/70/80/90/100% for the primary model (D6).
- [x] Implement target-SoC actions through the nonlinear charging integrator.
- [ ] Compare 5% versus 10% targets and 15/30/60-minute duration actions.

### R2.7 — Optimization model unclear

- [ ] Add a complete appendix formulation with queue, uncertainty, horizon, information, and fallback assumptions.
- [~] State which optimization baselines are offline, rolling-horizon, deterministic, robust, or scenario-based.
  - Documented in doc 08 §3 and doc 10 §3 — all three plan on the nominal network, CP-SAT offline with energy-safe repair, MPC receding-horizon — but not in the manuscript.
- [ ] Validate tiny exact objectives against exhaustive enumeration.

### R2.8–R2.9 — Heuristic and neural baselines

- [ ] Link each implemented heuristic to the closest cited problem variant and disclose adaptations.
- [ ] Add attention/transformer and strong search baselines under equal information.
- [ ] If a cited method cannot be transferred, document the exact incompatible assumption rather than giving a generic justification.

### Minor comments

- [ ] Change `demonstrated` to `demonstrates` near `main.tex:364`.
- [ ] Add the missing comma in Eq. 12.
- [ ] Standardize all figure-caption capitalization.
- [ ] Change `c. Actor Network head` to `(c) Actor Network Head` in the figure asset/caption.
- [ ] Correct Table 2 so realized energy, rather than only coefficient `xi`, is shown.

## Additional comments from Ziyan

- [~] Define unloading/service time as a model parameter and state when it is stochastic.
  - Implemented: unloading time is sampled once per service event from its own RNG stream and the draw is fixed for that event. The manuscript statement is open.
- [~] Explain that queue delay is endogenous under finite ports and FCFS admission.
  - Implemented and hardened against duplicate wake-ups, stale waiters, port overbooking, and cross-station double occupancy; joining a full station's queue is an explicit routing choice. The explanation is not in the paper.
- [~] Source and validate the nonlinear charging equation.
  - A CCCV formulation is implemented (`charging_curve.py:cccv_power_at_soc`) and compared against the linear model in `results/charging_curves/` and `latex/charging_curve_comparison.pdf`; doc 08 §3 quantifies a 55% underestimate by the naive model at high SoC. No primary source is cited and no validation against published measurements exists.
  - Known defect to disclose or fix: `ChargingCurveModel.estimate_charge_time` does not converge at a target of 1.0 and silently returns its search midpoint of 10 h. The baselines avoid it by using `calculate_charge_to_target`.
- [ ] Compare with an established three-segment/piecewise charging formulation, including Montoya-style curves.
  - Not implemented anywhere in the codebase.
- [ ] Expand the random-variable table to list distribution, parameters, clipping, correlation, source, scenario stream, and campaign status.
  - Every column except `source` is now derivable from the versioned scenario descriptor and the stochastic-distribution tests; the table itself has not been written.

## Where the revision stands

| Block | State |
| --- | --- |
| Simulator, feasibility, queues, charging semantics | complete and regression-tested |
| Canonical observation, encoders, action heads, artifact contract | complete |
| Headline campaign under the travel-time objective | complete (doc 10) |
| Mask, pooling, and active-truck ablations | not started, needs code |
| ALNS, attention/transformer, learned-baseline family | not started |
| Generalization and multi-seed campaigns | not started, needs a canonical replacement for the legacy script |
| Manuscript and response letter | not started |

The two items that gate resubmission independently of any writing are the mask ablation (R1.2/E3) and the generalization campaign (R1.7). Both require new code, not just compute.

## Response-letter completion rule

For every item above, the final response letter must contain:

1. a direct acknowledgement or disagreement;
2. the concrete change made;
3. the manuscript page/line or appendix location;
4. the relevant test, table, figure, or artifact ID;
5. any remaining limitation stated without overclaiming.
