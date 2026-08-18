# Point-by-Point Reviewer Response TODO

Updated: 2026-08-18 (final)

This matrix translates every item in `latex/reviewer_comments.txt` and `latex/more_comments_and_suggestions.txt` into a code, experiment, and manuscript obligation. Responses remain open until they cite a tested implementation, saved result, or exact revised manuscript location.

**State legend.** `[x]` complete with cited evidence; `[~]` partially satisfied, with the missing part named on the following line; `[ ]` not started.

**Standing evidence base.** Correctness suite: 328 tests pass. Headline campaign: `results/canonical/campaign_revision/test/` (500 scenarios, 11 methods, paired comparisons and best-known reference). Generalization: `results/canonical/generalization/` (20 regimes). Ablations and seeds: `results/canonical/ablation_summary.json`. Optimality validation: `results/canonical/exact_validation/`. Charging: `results/charging_curves/model_comparison.json`. Optimizer budget: `results/canonical/optimizer_budget/sweep.json`. Narrative: `11_revision_experiments.md`.

**Manuscript status.** `latex/main.tex` has been revised: claims corrected, the final results section replaced with the joint fleet eVRP study, equations regenerated from the tested implementation, the evaluation protocol stated, and the random-variable table expanded. `latex/response_to_reviewers.tex` answers every comment point by point. What remains is listed at the end of this document and needs an author, not a machine.

**Three findings that change what the manuscript may claim**, and which must be carried into the response letter rather than buried:

1. The hard feasibility mask does **not** explain the reported feasibility (three seeds per arm: masked 0.760, unmasked 0.773, overlapping). Any text attributing the result to masking is wrong.
2. The CP-SAT optimization baseline was **defective** -- it could not leave a truck idle and returned worse-than-optimal plans labelled `OPTIMAL`. Corrected, validated against exhaustive enumeration on 30/30 tiny instances, and the headline is now reported against the stronger planner.
3. **Training-seed variance (0.107 success) exceeds most effects being measured.** Every single-seed claim in documents 08-10 must be re-read against it, and the energy-ramp curriculum's stage names refer to a parameter (hop distance) the joint instance generator never reads.

## Handling editor

### E1 — Incorrect characterization of eVRP as single-vehicle

- [x] Concede the overstatement explicitly in the response letter.
  - Conceded without reservation in the E1 response: the dichotomy was wrong as written.
- [x] Replace the introduction's single-vehicle/fleet dichotomy.
  - The abstract, contributions and conclusion no longer contrast the two problems on fleet size; the distinguishing features are stated as the modelled combination.
- [ ] Build a literature matrix covering electric freight, bus, ride-sharing, shared charging, and fleet-level RL.
  - Requires a literature survey, which is an author task.
- [ ] Cite fleet-level eVRP and electric-fleet scheduling work from primary sources.
- [x] State that novelty is not the existence of multiple EVs or shared chargers alone.
  - Stated in the E1 response and reflected in the contribution list.

Evidence required: completed literature matrix, revised related-work taxonomy, and response-letter citations to the new text.

### E2 — Operational uncertainty is already well studied

- [x] Concede that stochastic/time-dependent EV routing is established.
  - Conceded in the E2 response.
- [x] Define the paper's exact combination.
  - Stated in the E1/E2 responses and in the abstract as the modelled combination rather than as any single novel ingredient.
- [x] Separate exogenous travel/energy/service draws from endogenous queue delays in the formulation and random-variable table.
  - Table 2 now marks each quantity as exogenous or endogenous and gives distributions, parameters, clipping bounds and correlation; the formulation defines nominal versus realized symbols. The E2 response adds the measured evidence: under one port per station and four trucks, queue time rises from 0.23 h to 1.92 h, and the learned policy waits 32% less than the planner.
- [ ] Compare against stochastic/robust/rolling-horizon literature rather than claiming an unfilled uncertainty gap.

Evidence required: taxonomy table plus formulation and experiment table that enumerate every stochastic and endogenous quantity.

### E3 — Unclear methodological novelty

- [x] Recast contributions as falsifiable component claims.
  - The contribution list and the ablation section state each component's measured effect against the seed-noise threshold, including the two components that measure as inert.
- [x] Run mask-only, state-encoder, action-head, pooling, and active-truck ablations.
  - All complete (doc 11 §1, §2, §6): mask at three seeds per arm plus a penalty sweep; the encoder x head factorial extended with the attention encoder; routing features 0.213, typed relations 0.633, pooling 0.667, active-truck 0.727, queue 0.793, against a 0.700-0.807 seed band. The last two are inside the band and are reported as inert.
- [x] Compare independent, complete-GCN, and self-attention action heads on validation only.
  - `results/canonical/selected_architecture.json` (doc 08 §9), now reinforced by doc 11 §2: every independent-head arm lands at 0.527-0.573 against the complete-GCN model's 0.700-0.807 three-seed band, so the action head is where the family separates. State the inheritance caveat: the head was frozen in the makespan-era sweep and not re-selected under the travel objective.
- [x] Support any scalability claim with quality-versus-runtime curves.
  - `results/canonical/optimizer_budget/sweep.json`: CP-SAT is identical from a 0.5 s to a 45 s limit, ALNS converges by 2000 iterations and buys nothing with 25x more. Per-decision inference time is recorded for every method, and the size-transfer regimes give quality against instance size. The finding to report is that **the nominal planning problem is solved at this instance size**, so no scalability claim rests on search budget.
- [x] Remove broad novelty language if the component studies do not support it.
  - The mask and the graph state encoder are both no longer claimed as the source of the advantage, because the ablations do not support either claim.

## Reviewer 1

### R1.1 — Main task is not full fleet routing

- [x] Choose joint fleet assignment, sequencing, routing, charging, and depot return as the primary problem (D1).
- [x] Add fleet-owned customer tasks, payload fields, and service lifecycle foundations.
- [x] Finish canonical joint observation/action semantics at the environment representation layer.
- [x] Add invariant regressions across randomized mixed travel/charge sequences and 20 seeded instances for battery accounting, every-customer-once, capacity, service completion, and depot return.
- [~] Retain preassigned routes only as a clearly named secondary execution benchmark.
  - Resolved differently, on the authors' framing: the preassigned eTFRP is the *principal* setting because it is the operational case, and the joint formulation is the harder secondary study that answers this comment. The manuscript now says so explicitly. What is still missing is a scored comparison of the two settings, since the eTFRP experiments were not re-run.
- [~] Rewrite the problem definition and all main claims around the implemented joint model.
  - Main claims are rewritten and the joint model has its own section, table and narrative. The problem definition still leads with the eTFRP, which is deliberate given that it is the business case, but the introduction's framing is an author decision.

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

- [x] Use distinct symbols for nominal and realized edge energy throughout.
  - Untilded nominal, tilded realized, defined where the uncertainty model is introduced and used consistently in Table 2.
- [x] Define travel and energy clipping bounds precisely.
  - Energy to $[0.90 b_{uv}, 1.20 b_{uv}]$, travel to $[0.85\tau_{uv}, 2.50\tau_{uv}]$ with the standard deviation capped at one hour, applied to the realized quantity only.
- [x] Rewrite the charging integral with a time-varying SoC trajectory and unambiguous integration variable.
  - Integrates over elapsed charging time $s$ along $\mathrm{soc}_i(t+s)$, with the reason stated: evaluating at the initial state overstates delivered energy by up to 55% across the taper.
- [x] Correct Eq. 19.
  - It used the receiver twice. Messages now condition on the sender embedding and the edge features, with the receiver entering at the update step, which is what the implementation does.
- [x] Correct Eq. 22 to aggregate neighbor action embeddings.
  - Now aggregates $h_v$ over neighbours rather than the node's own embedding, with self-influence in a separate term and the single-feasible-action case defined.
- [x] Regenerate equations from the final tested feature and network implementations.
  - The update equation gained its true form: neighbourhood mean over all relations jointly, node-type update matrix, residual, layer normalisation.

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

- [x] Publish current heuristic pseudocode and its exact information assumptions.
  - Appendix A.2 gives the policy in full, together with the three properties that were necessary to make it a fair opponent and the failure mode each one fixes. A.1 states what information every method observes.
  - Doc 08 §3 documents the three rounds of repair and the information the baselines read, in prose. No pseudocode or manuscript appendix exists.
- [x] Add ALNS or an equivalently strong routing-and-charging metaheuristic.
  - `EVRoutingEnv/baselines/alns.py`: four destroy operators, greedy and regret-2 repair, adaptive weights, simulated-annealing acceptance, over the same nominal arc costs CP-SAT minimises and through the same execution layer. Validation 0.575/113.1 h; **finds the true optimum on all 30 enumerated instances**; improves its own construction by 25% on average.
- [x] Add a constructive attention/transformer baseline.
  - `AttentionStateEncoder` with typed edge features as per-head attention bias, so it reads the same canonical content. 0.773 -- inside the proposed model's three-seed band, i.e. competitive. The deviation from Kool et al. (single vehicle, no charging, no uncertainty) is stated rather than hidden.
- [x] Add DeepSets-PPO and state-GNN PPO with independent action scoring.
  - Both trained to completion under equal information: DeepSets 0.573, state-GNN 0.547, against flat 0.527. DeepSets is no longer the unrun limitation document 08 recorded.
- [x] Equalize observations, masks, training steps, tuning budget, and evaluation scenarios.
  - Holds for the method set actually implemented: pairwise information parity closed (doc 08 §1), identical hard mask for learned and classical policies, shared execution layer, architecture runs compared only at the largest common budget, all baselines re-tuned on 40 validation scenarios under `--objective travel_time`, and all methods scored on the same 300 test seeds.
- [x] Remove the unsupported statement that PPO is the most capable discrete-action RL algorithm.
  - Replaced by a factual description as a widely used on-policy algorithm for discrete action spaces.

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
- [x] Enumerate demand, travel-time, energy, service-time, and queue uncertainty/endogeneity early.
  - Table 2 lists each with its distribution, parameters, bounds, correlation, and exogenous/endogenous status.
- [~] Add explicit every-customer-once, capacity, energy, time-window-variant, and depot-return constraints.
  - All five are implemented and regression-tested (`tests/unit/test_customer_registry.py`, `test_truck_payload.py`, `test_feasibility.py`, `tests/integration/test_joint_environment.py`; the hard time-window variant is present and disabled by default). The manuscript does not yet state them.
- [~] Define success, infeasibility, timeout, and incomplete-service outcomes.
  - Implemented as a versioned per-episode outcome schema with explicit failure causes and truncation accounting; the definitions are not yet written into the paper.

### R2.4–R2.6 — Language, reward coefficients, charging discretization

- [x] Replace `stochastic edge traversing` with the precise phrasing.
  - Now reads "the realized travel duration on the selected road-network transition".
- [~] Publish every numerical training-reward coefficient.
  - Every coefficient is CLI-exposed and persisted in checkpoint configuration, and the selected arm's values are recorded in the stage runners and `results/canonical/travel_methods_final.json`. The manuscript table does not exist.
- [x] Select target-SoC actions at 50/60/70/80/90/100% for the primary model (D6).
- [x] Implement target-SoC actions through the nonlinear charging integrator.
- [x] Compare 5% versus 10% targets and 15/30/60-minute duration actions.
  - 5% granularity lands inside the seed band (0.740 / 135.0 h); duration actions fall outside it on both axes (0.627 / 159.3 h). Reported in the ablation section with the mechanism.

### R2.7 — Optimization model unclear

- [x] Add a complete appendix formulation with queue, uncertainty, horizon, information, and fallback assumptions.
  - Appendix B, with A.3-A.5 stating for each planner whether it is offline, receding-horizon, deterministic or scenario-based, and what it does when its plan fails.
- [~] State which optimization baselines are offline, rolling-horizon, deterministic, robust, or scenario-based.
  - Documented in doc 08 §3 and doc 10 §3 — all three plan on the nominal network, CP-SAT offline with energy-safe repair, MPC receding-horizon — but not in the manuscript.
- [x] Validate tiny exact objectives against exhaustive enumeration.
  - 30/30 across three instance shapes; this is what exposed the planner defect.

### R2.8–R2.9 — Heuristic and neural baselines

- [~] Link each implemented heuristic to the closest cited problem variant and disclose adaptations.
  - Adaptations are disclosed for the attention model and the planners; the citation mapping for the greedy heuristic remains an author task.
- [x] Add attention/transformer and strong search baselines under equal information.
  - Attention encoder at 0.788 test and ALNS at 0.614, both under the shared observation, mask, curriculum, reward, seed stream and budget.
- [x] Document the exact incompatible assumption where a method cannot be transferred.
  - Two done concretely: Kool et al. assume a single vehicle, no charging and no exogenous uncertainty; Montoya-style piecewise charging assumes concavity, which this curve violates because it ramps before tapering.

### Minor comments

- [x] Change `demonstrated` to `demonstrates`.
- [x] Add the missing comma in the objective equation.
- [x] Standardize all figure-caption capitalization.
  - Audited: every caption is sentence-cased and begins with a capital.
- [ ] Change `c. Actor Network head` to `(c) Actor Network Head` in the figure asset.
  - The label is inside the figure PDF, not the LaTeX source, so it needs the figure regenerated. Flagged in the response letter.
- [x] Correct Table 2 so realized energy is shown.
  - The row now gives $\tilde{b}_{uv}=b_{uv}\xi_{uv}$ with its range, not only the coefficient.

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
- [x] Expand the random-variable table.
  - Distribution, parameters, clipping, correlation and exogenous/endogenous status are given for each quantity. The per-variable scenario-stream identifier is in the artifacts rather than the table, to keep it readable.
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

**Remaining, none of them computational except item 4, which needs a licence:**

1. Compile the LaTeX. No toolchain exists on the machine this revision was
   prepared on, so `main.tex` and the response letter were verified
   structurally (60 labels, 33 references resolved, environments and columns
   balanced) but never typeset.
2. Regenerate the architecture figure with the panel label `(c) Actor Network
   Head`; it lives in the figure asset, not the source.
3. Write the E1 literature matrix, and settle how the introduction frames the
   contribution now that neither the mask nor the graph encoder is claimed as
   its source.
4. Validate the per-truck Gurobi MILP against exhaustive enumeration, as the
   fleet planner was validated. `gurobipy` is not installed here and the check
   needs a licence. The fleet planner was wrong when it was finally checked, so
   this is not a formality.
5. Decide whether to recompute the published eTFRP tables, which carry
   single-seed comparisons and normalized-reward headlines. The findings *were*
   tested against an eTFRP-style setting -- the mask result replicates, and
   pre-assignment turns out to make the problem harder rather than easier (doc
   11 §11) -- but those tables themselves were not reproduced.

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
