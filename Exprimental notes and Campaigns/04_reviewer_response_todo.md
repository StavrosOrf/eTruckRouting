# Point-by-Point Reviewer Response TODO

This matrix translates every item in `latex/reviewer_comments.txt` and `latex/more_comments_and_suggestions.txt` into a code, experiment, and manuscript obligation. Responses remain open until they cite a tested implementation, saved result, or exact revised manuscript location.

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
- [ ] Separate exogenous travel/energy/service draws from endogenous queue delays in the formulation and random-variable table.
- [ ] Compare against stochastic/robust/rolling-horizon literature rather than claiming an unfilled uncertainty gap.

Evidence required: taxonomy table plus formulation and experiment table that enumerate every stochastic and endogenous quantity.

### E3 — Unclear methodological novelty

- [ ] Recast contributions as falsifiable component claims.
- [ ] Run mask-only, state-encoder, action-head, pooling, and active-truck ablations.
- [ ] Compare independent, complete-GCN, and self-attention action heads on validation only.
- [ ] Support any scalability claim with quality-versus-runtime curves.
- [ ] Remove broad novelty language if the component studies do not support it.

## Reviewer 1

### R1.1 — Main task is not full fleet routing

- [x] Choose joint fleet assignment, sequencing, routing, charging, and depot return as the primary problem (D1).
- [x] Add fleet-owned customer tasks, payload fields, and service lifecycle foundations.
- [x] Finish canonical joint observation/action semantics at the environment representation layer.
- [x] Add invariant regressions across randomized mixed travel/charge sequences and 20 seeded instances for battery accounting, every-customer-once, capacity, service completion, and depot return.
- [ ] Retain preassigned routes only as a clearly named secondary execution benchmark.
- [ ] Rewrite the problem definition and all main claims around the implemented joint model.

### R1.2 — Mask may explain most of GraphPPO's gain

- [ ] Train PPO without mask using the full equivalent observation.
- [ ] Train MaskPPO with identical observation and candidate semantics.
- [ ] Compare flat MLP, DeepSets, heterogeneous GNN, and active-truck conditioning.
- [ ] Compare no action interaction, complete GCN, and self-attention heads.
- [ ] Ablate global pooling, active-truck embedding, queue features, and relation types.
- [ ] Report feasibility, conditional makespan, queue time, and runtime—not reward alone.

### R1.3 — Notation and equation defects

- [ ] Use distinct symbols for nominal and realized edge energy throughout.
- [ ] Define travel and energy clipping bounds precisely.
- [ ] Rewrite the charging integral with a time-varying SoC trajectory and unambiguous integration variable.
- [ ] Correct Eq. 19 to use sender and receiver embeddings.
- [ ] Correct Eq. 22 to aggregate neighbor action embeddings.
- [ ] Regenerate equations from the final tested feature and network implementations.

### R1.4 — Optimization benchmark does not establish near-optimality

- [ ] Rename the inherited method `conservative per-truck deterministic MILP` in artifacts and manuscript.
- [ ] Add exact/bounded fleet optimization for tiny deterministic instances.
- [ ] Add rolling-horizon/scenario optimization where tractable.
- [ ] Save solver status, bound, gap, runtime, retry, and fallback counters.
- [ ] Remove `near-optimal`, `optimization-level`, and equivalent wording unless supported by valid gaps/bounds.

### R1.5 — High reward despite zero success

- [x] Approve feasibility-first, then makespan evaluation (D4).
- [x] Separate training reward from per-episode operational evaluation metrics; statistical aggregation remains pending.
- [ ] Retain every failed episode and classify the failure cause.
- [ ] Report full-service probability first and time/cost only with explicit conditioning.
- [ ] Explain the inherited `+500` delivery, `-1` time, and `-1000` failure shaping and replace it if needed for stable training.

### R1.6 — Weak and underspecified baselines

- [ ] Publish current heuristic pseudocode and its exact information assumptions.
- [ ] Add ALNS or an equivalently strong routing-and-charging metaheuristic.
- [ ] Add a constructive attention/transformer baseline.
- [ ] Add DeepSets-PPO and state-GNN PPO with independent action scoring.
- [ ] Equalize observations, masks, training steps, tuning budget, and evaluation scenarios.
- [ ] Remove the unsupported statement that PPO is the most capable discrete-action RL algorithm.

### R1.7 — Generalization evidence is too narrow

- [ ] Freeze multiple training seeds before test evaluation.
- [ ] Use at least 500 paired test scenarios per main setting.
- [ ] Test unseen graphs/regions, charger layouts/powers/capacities, fleet/customer sizes, battery parameters, demand patterns, and uncertainty distributions.
- [ ] Distinguish interpolation, within-simulator size transfer, and genuine OOD tests.
- [ ] Evaluate credible baselines in every transfer regime.
- [ ] Replace the current broad zero-shot claims if only size transfer remains tested.

## Reviewer 2

### R2.1–R2.3 — Definition, uncertainty, and mandatory service

- [ ] Define the joint problem on page 1 without inventing an eTFRP/eVRP dichotomy.
- [ ] Enumerate demand, travel-time, energy, service-time, and queue uncertainty/endogeneity early.
- [ ] Add explicit every-customer-once, capacity, energy, time-window-variant, and depot-return constraints.
- [ ] Define success, infeasibility, timeout, and incomplete-service outcomes.

### R2.4–R2.6 — Language, reward coefficients, charging discretization

- [ ] Replace `stochastic edge traversing` near `main.tex:776` with `realized travel duration on the selected road-network transition`.
- [ ] Publish every numerical training-reward coefficient.
- [x] Select target-SoC actions at 50/60/70/80/90/100% for the primary model (D6).
- [x] Implement target-SoC actions through the nonlinear charging integrator.
- [ ] Compare 5% versus 10% targets and 15/30/60-minute duration actions.

### R2.7 — Optimization model unclear

- [ ] Add a complete appendix formulation with queue, uncertainty, horizon, information, and fallback assumptions.
- [ ] State which optimization baselines are offline, rolling-horizon, deterministic, robust, or scenario-based.
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

- [ ] Define unloading/service time as a model parameter and state when it is stochastic.
- [ ] Explain that queue delay is endogenous under finite ports and FCFS admission.
- [ ] Source and validate the nonlinear charging equation.
- [ ] Compare with an established three-segment/piecewise charging formulation, including Montoya-style curves.
- [ ] Expand the random-variable table to list distribution, parameters, clipping, correlation, source, scenario stream, and campaign status.

## Response-letter completion rule

For every item above, the final response letter must contain:

1. a direct acknowledgement or disagreement;
2. the concrete change made;
3. the manuscript page/line or appendix location;
4. the relevant test, table, figure, or artifact ID;
5. any remaining limitation stated without overclaiming.
