# Point-by-Point Reviewer Response TODO

Updated: 2026-08-18 (final)

This matrix translates every item in `latex/reviewer_comments.txt` and `latex/more_comments_and_suggestions.txt` into a code, experiment, and manuscript obligation. Responses remain open until they cite a tested implementation, saved result, or exact revised manuscript location.

**State legend.** `[x]` complete with cited evidence; `[~]` partially satisfied, with the missing part named on the following line; `[ ]` not started.

**Standing evidence base.** Correctness suite: 320 tests pass. Headline campaign: `results/canonical/campaign_revision/test/` (500 scenarios, 11 methods, paired comparisons and best-known reference). Generalization: `results/canonical/generalization/` (20 regimes). Ablations and seeds: `results/canonical/ablation_summary.json`. Optimality validation: `results/canonical/exact_validation/`. Charging: `results/charging_curves/model_comparison.json`. Optimizer budget: `results/canonical/optimizer_budget/sweep.json`. Narrative: `11_revision_experiments.md`.

**Standing gap.** `latex/main.tex` is unmodified since `1459e54`, i.e. since before this campaign branch existed. Every item below whose deliverable is manuscript text is therefore open regardless of how much of its code and experiment obligation is discharged. That is now the *only* systematic gap: the experimental obligations are discharged.

**Three findings that change what the manuscript may claim**, and which must be carried into the response letter rather than buried:

1. The hard feasibility mask does **not** explain the reported feasibility (three seeds per arm: masked 0.760, unmasked 0.773, overlapping). Any text attributing the result to masking is wrong.
2. The CP-SAT optimization baseline was **defective** -- it could not leave a truck idle and returned worse-than-optimal plans labelled `OPTIMAL`. Corrected, validated against exhaustive enumeration on 30/30 tiny instances, and the headline is now reported against the stronger planner.
3. **Training-seed variance (0.107 success) exceeds most effects being measured.** Every single-seed claim in documents 08-10 must be re-read against it, and the energy-ramp curriculum's stage names refer to a parameter (hop distance) the joint instance generator never reads.

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
  - Done: the **mask ablation** at three seeds per arm plus a penalty-magnitude sweep (doc 11 §1); the state-encoder x action-head factorial (doc 08 §9) extended with the attention encoder (0.773, doc 11 §2); **pooling** (0.667) and **typed-relation** (0.633) ablations at matched budget (doc 11 §6); the routing-action-feature ablation (0.213, doc 10 §6.2).
  - Missing: `ablate_queue` and `ablate_active_truck` are training in batch 4; the mechanism and harness for both are implemented and tested (`tests/unit/test_feature_ablations.py`).
- [x] Compare independent, complete-GCN, and self-attention action heads on validation only.
  - `results/canonical/selected_architecture.json` (doc 08 §9), now reinforced by doc 11 §2: every independent-head arm lands at 0.527-0.573 against the complete-GCN model's 0.700-0.807 three-seed band, so the action head is where the family separates. State the inheritance caveat: the head was frozen in the makespan-era sweep and not re-selected under the travel objective.
- [x] Support any scalability claim with quality-versus-runtime curves.
  - `results/canonical/optimizer_budget/sweep.json`: CP-SAT is identical from a 0.5 s to a 45 s limit, ALNS converges by 2000 iterations and buys nothing with 25x more. Per-decision inference time is recorded for every method, and the size-transfer regimes give quality against instance size. The finding to report is that **the nominal planning problem is solved at this instance size**, so no scalability claim rests on search budget.
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

- [x] Train PPO without mask using the full equivalent observation.
  - `environment.policy_action_mask=structural` keeps the identical observation and candidate set and hides only slots denoting no action. Three seeds: 0.793 / 0.820 / 0.707 against the masked control's 0.700 / 0.807 / 0.773 (doc 11 §1.2). **The mask does not explain the result.**
- [x] Train MaskPPO with identical observation and candidate semantics.
  - `ppo_flat` is exactly that -- flat state encoder, independent action scoring, hard mask, identical everything else -- at 0.527 validation and 0.506 on the 500-scenario test split.
- [x] Compare flat MLP, DeepSets, heterogeneous GNN, and active-truck conditioning.
  - All four encoders now trained to completion under equal information: flat 0.527, DeepSets 0.573, hetero 0.547 (independent head), attention 0.773 and hetero 0.700-0.807 (complete-GCN head). DeepSets is no longer an unrun limitation.
- [x] Compare no action interaction, complete GCN, and self-attention heads.
  - `results/canonical/selected_architecture.json`; same caveat as E3 about the makespan-era schema.
- [~] Ablate global pooling, active-truck embedding, queue features, and relation types.
  - Pooling (0.667) and relation types (0.633) done and both clear the seed-noise threshold; queue and active-truck arms are training in batch 4.
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

- [~] Rename the inherited method `conservative per-truck deterministic MILP` in artifacts and manuscript.
  - Done in code: all three inherited Gurobi models now open by naming what they are, state that they bound neither the fleet nor the stochastic problem, and point at the fleet-level CP-SAT planner. The manuscript rename is outstanding.
- [x] Add exact/bounded fleet optimization for tiny deterministic instances.
  - The corrected CP-SAT model **matches exhaustive enumeration on 30 of 30** instances across (5 customers, 2 trucks), (6, 2), and (4, 3), to within its own discretization (`results/canonical/exact_validation/`). This is a validated exact reference on tiny instances, which document 09 could not claim. The defect it replaced is in doc 11 §3.
- [x] Add rolling-horizon/scenario optimization where tractable.
  - `RollingHorizonMPCPolicy`, grid-searched on 40 validation scenarios under the travel objective and scored on the 500-scenario test split.
- [x] Save solver status, bound, gap, runtime, retry, and fallback counters.
  - Every episode row carries `policy_diagnostics` for any policy exposing them: status, objective, best bound, relative gap, solver wall seconds, solve count, and plan fallbacks (how often the executed plan ran out and the shared navigation layer chose instead). A metaheuristic reports no bound and no optimality claim by construction.
- [~] Remove `near-optimal`, `optimization-level`, and equivalent wording unless supported by valid gaps/bounds.
  - The evidence to support a *bounded* statement now exists on tiny instances only; at campaign scale the honest quantity remains best-known. Six such claims remain in `latex/main.tex`.

### R1.5 — High reward despite zero success

- [x] Approve feasibility-first, then makespan evaluation (D4).
- [x] Separate training reward from per-episode operational evaluation metrics.
  - Statistical aggregation is now also complete: Wilson intervals and deterministic paired bootstrap intervals are consumed by the campaign tables (`compare_campaign.py`, `comparison_vs_*.json`).
- [x] Retain every failed episode and classify the failure cause.
  - Retention was already guaranteed; classification is now complete. Truncation was silent, which is where `unspecified_failure` came from: on 40 replayed heuristic episodes the 19 unlabelled failures resolve into 17 `step_limit_exhausted` and 2 `time_limit_exhausted`, with every metric unchanged. `_check_terminated` and `_check_truncated` now name every outcome.
- [x] Report full-service probability first and time/cost only with explicit conditioning.
  - Enforced in the artifact schema itself: every metric carries a `conditioning` field, and cost metrics are marked `successful_episodes`. Manuscript tables still have to consume this.
- [~] Explain the inherited `+500` delivery, `-1` time, and `-1000` failure shaping and replace it if needed for stable training.
  - Replaced and re-tuned: the current shaping is success bonus, incompletion penalty, per-leg travel multiplier, terminal travel-time bonus, stranding penalty, and energy-margin bonus, all CLI-exposed in `train_canonical_ppo.py` and persisted per checkpoint. Doc 10 §2.2 and §4 explain the design and measure the penalty-weight sweep. The manuscript explanation and the published coefficient table are open.

### R1.6 — Weak and underspecified baselines

- [ ] Publish current heuristic pseudocode and its exact information assumptions.
  - Doc 08 §3 documents the three rounds of repair and the information the baselines read, in prose. No pseudocode or manuscript appendix exists.
- [x] Add ALNS or an equivalently strong routing-and-charging metaheuristic.
  - `EVRoutingEnv/baselines/alns.py`: four destroy operators, greedy and regret-2 repair, adaptive weights, simulated-annealing acceptance, over the same nominal arc costs CP-SAT minimises and through the same execution layer. Validation 0.575/113.1 h; **finds the true optimum on all 30 enumerated instances**; improves its own construction by 25% on average.
- [x] Add a constructive attention/transformer baseline.
  - `AttentionStateEncoder` with typed edge features as per-head attention bias, so it reads the same canonical content. 0.773 -- inside the proposed model's three-seed band, i.e. competitive. The deviation from Kool et al. (single vehicle, no charging, no uncertainty) is stated rather than hidden.
- [x] Add DeepSets-PPO and state-GNN PPO with independent action scoring.
  - Both trained to completion under equal information: DeepSets 0.573, state-GNN 0.547, against flat 0.527. DeepSets is no longer the unrun limitation document 08 recorded.
- [x] Equalize observations, masks, training steps, tuning budget, and evaluation scenarios.
  - Holds for the method set actually implemented: pairwise information parity closed (doc 08 §1), identical hard mask for learned and classical policies, shared execution layer, architecture runs compared only at the largest common budget, all baselines re-tuned on 40 validation scenarios under `--objective travel_time`, and all methods scored on the same 300 test seeds.
- [ ] Remove the unsupported statement that PPO is the most capable discrete-action RL algorithm.

### R1.7 — Generalization evidence is too narrow

- [~] Freeze multiple training seeds before test evaluation.
  - Three seeds are trained and re-scored at the stage-A configuration for both the masked control and the unmasked arm, and the variance is the headline finding: **0.107 success across seeds, larger than most measured effects** (doc 11 §1.3). Seed replication of the *full* A→B→C ladder is running in batches 4 and 5; until it lands, the 0.858 test figure is still a single-seed number and must be quoted as such.
- [x] Use at least 500 paired test scenarios per main setting.
  - `results/canonical/campaign_revision/test/`: 500 scenarios, 11 methods, paired bootstrap differences on jointly solved scenarios. The headline reproduced exactly (0.858 against 0.857 on 300).
- [x] Test unseen graphs/regions, charger layouts/powers/capacities, fleet/customer sizes, battery parameters, demand patterns, and uncertainty distributions.
  - 20 regimes in `results/canonical/generalization/`, every method on the same held-out seeds. Note two regimes that *look* like tests and are not, and were removed: `truck.base_speed` does not affect travel time, and hop distances are not read by the joint generator (doc 11 §8.3). The road network is perturbed through `network.travel_time_scale` and `network.energy_scale` instead.
- [x] Distinguish interpolation, within-simulator size transfer, and genuine OOD tests.
  - Every regime carries a `kind` label and the summary keeps it attached, so a size-transfer result can never be quoted as OOD evidence.
- [x] Evaluate credible baselines in every transfer regime.
  - CP-SAT, ALNS, MPC, the heuristic, and random are scored in all 16 shared regimes alongside both learned arms.
- [~] Replace the current broad zero-shot claims if only size transfer remains tested.
  - The evidence now supports a *bounded* claim: size transfer and most parameter shift hold; **budget shift does not** -- at a 300 kWh battery the learned policies fall to 0.34 against CP-SAT's 0.43, and at 500 kWh they reach 0.87 against 0.98 (doc 11 §8.2). The manuscript text is unwritten; two zero-shot claims remain in `latex/main.tex`.

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
  - Quantified against both alternatives in `results/charging_curves/model_comparison.json`. The `estimate_charge_time` defect is **fixed**: it bisected on duration, could not converge at a target of 1.0, and returned its range midpoint -- 10 hours for a charge that takes 0.53 h. It now integrates directly to the target, the routine the simulator and every baseline already use.
  - Still missing: a primary source citation and validation against published measurements. The curve's *shape* is now documented and its consequences measured, but its provenance is not.
- [x] Compare with an established three-segment/piecewise charging formulation, including Montoya-style curves.
  - `ChargingCurveModel.montoya_breakpoints` builds a piecewise-linear charging function interpolating the integrated curve at its phase boundaries. At 350-750 kW the classical three-segment form errs by 4.7% mean / 27.8% max, **barely better than assuming constant power** (5.0% / 28.7%), because its pieces straddle both curved regions. Four segments give 2.6% / 10.0%, five give 0.6% / 4.0%.
  - The reason is recorded as a test: this curve ramps to peak before tapering, so it is **not concave**, and concavity is what a Montoya-style approximation assumes. That is the "exact incompatible assumption" R2.9 asks for rather than a generic justification.
- [ ] Expand the random-variable table to list distribution, parameters, clipping, correlation, source, scenario stream, and campaign status.
  - Every column except `source` is now derivable from the versioned scenario descriptor and the stochastic-distribution tests; the table itself has not been written.

## Where the revision stands

| Block | State |
| --- | --- |
| Simulator, feasibility, queues, charging semantics | complete, 328 tests pass |
| Canonical observation, encoders, action heads, artifact contract | complete |
| Headline campaign: 500 scenarios, 16 methods, three seeds | complete |
| Mask ablation: three seeds per arm plus penalty sweep | complete |
| Baseline family: ALNS, attention, DeepSets, state-GNN, PPO, MaskPPO | complete |
| Optimality validated against exhaustive enumeration | complete |
| Charging: model comparison, granularity, action semantics, curve defect | complete |
| Generalization: 20 regimes, plus scale grid and congestion | complete |
| Component ablations | complete |
| Manuscript revision | complete except the items below |
| Response letter | complete (`latex/response_to_reviewers.tex`) |

**Remaining, all authorial rather than computational:**

1. Compile the LaTeX. No toolchain exists on the machine this revision was
   prepared on, so `main.tex` and the response letter were verified
   structurally (60 labels, 33 references resolved, environments and columns
   balanced) but never typeset.
2. Regenerate the architecture figure with the panel label `(c) Actor Network
   Head`; it lives in the figure asset, not the source.
3. Write the E1 literature matrix, and settle how the introduction frames the
   contribution now that neither the mask nor the graph encoder is claimed as
   its source.
4. Decide what to do about the main eTFRP experiments, which were not re-run.
   The three findings that changed the joint-setting claims have not been
   checked against them, and the corrected CP-SAT model is not the planner
   those tables used.

**The three findings the response letter leads with**, because they change
claims rather than adding to them:

1. the hard feasibility mask does not explain the reported feasibility;
2. the CP-SAT optimization baseline was defective and is now stronger;
3. training-seed variance exceeds most single-seed effects in documents 08-10.

Two further results are reported against interest and should not be quietly
dropped in redrafting: upward size transfer fails when the training envelope is
widened, and the attention baseline is competitive with the proposed encoder.

## Response-letter completion rule

For every item above, the final response letter must contain:

1. a direct acknowledgement or disagreement;
2. the concrete change made;
3. the manuscript page/line or appendix location;
4. the relevant test, table, figure, or artifact ID;
5. any remaining limitation stated without overclaiming.
