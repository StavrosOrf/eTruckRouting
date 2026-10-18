# California GraphPPO campaign on Snellius

Plan date: 2026-10-18. Repository inspected at commit `866708a`, with existing local changes. This is a campaign design, not a record of completed experiments. No training jobs were submitted and no implementation tests were run while preparing it.

**Recommendation:** first correct the numerical/training issues below, then run a staged, multi-seed campaign on **10 trucks / 30 customers**, retaining the proposed heterogeneous state graph + variable feasible-action graph + PPO. Promote configurations using feasibility and paired operational cost, and establish the strength of the optimization reference independently. Profile two training processes per A100 allocation before committing to that packing. Cap concurrent campaign training allocations at 30 across all arrays.

## 1. Freeze the problem and the meaning of near-optimal

Working assumptions, pending any preference supplied by the user:

- Main task: `problem.mode: joint_fleet`, with **joint customer assignment**. Thirty customers are generated using three witness deliveries per truck on average; the learned policy may distribute them differently and may retire unused trucks. An exactly-three-per-truck requirement is a different constraint. Use `problem.assignment: preassigned` for a separate fixed-ownership comparison.
- Primary ordering: maximize successful fleet completion, then minimize **total operating truck-hours**: travel + charging + queueing + time-window waiting + service. Report fleet makespan and driving hours separately. This proposed objective must be frozen before tuning; the existing manuscript/canonical selection default uses driving hours.
- A successful episode serves every customer and returns every truck to the depot without failure or budget exhaustion. Use operational `success`/`info['successful']`, not `all_complete`.
- Keep the actual California map, payload constraints, shared charger capacities/FCFS queues, stochastic travel, energy and service, and depot return. A witness certifies existence under the generated seeded scenario; it does not certify success under every future disturbance.
- All optimality statements initially concern the **configured simulator**: fixed minimum-energy road paths, its available decisions, charging choices, scheduler, and horizon. Solving unrestricted real California logistics would additionally need validated road/vehicle data, alternative road paths, and an operational specification beyond this simulator.

The current map metadata reports 258 nodes, 25 charging nodes and 233 noncharging nodes. A depot and eligibility/reachability rules further reduce the customer pool. All 25 stations must remain available to the main policy and reference. The stored map is directed and asymmetric.

Create these explicit, versioned targets during implementation:

| Target | Purpose | Specification |
| --- | --- | --- |
| `ca10x3_reference` | Reproduce current model at requested size | Copy `config_joint.yaml`; set `num_trucks: 10`, `num_stops: 30`, fixed customer count; retain 400 kWh, payload 20, horizon 200 h, current traffic/service noise, six SoC targets, no windows |
| `ca10x3_full` | Primary model with effective nonlinear charging | Same fleet/map, effective charging taper, documented charger power assumptions, shared queues; activate calibrated feasible windows if windows belong to the intended target |
| `ca10x3_stress` | Robustness, never a substitute for main target | Separate regimes for scarce ports/outages, stronger noise, tight windows and reduced initial SoC |

Do not silently call `ca10x3_reference` a nonlinear, time-window benchmark: its windows are disabled and its 150 kW taper floor equals its 150 kW peak. For an initial *sensitivity* target, a 150 kW DC peak with a 30 kW taper floor and taper start 0.8 is a concrete choice; it is not a calibration to California equipment. If testing Level2 at 22 kW, remove the overriding `[150.0]` station power class and apply the type-specific configuration explicitly. These are experimental assumptions until supported by station data. Keep the reference target in every final report so the modeling change is visible.

Before freezing windows, measure generated witness schedules and generation acceptance on train-only pilot instances. Choose window widths in **hours** from those schedules; California nominal OD times can be long, so arbitrary one-hour windows can make reset generation fail. Current window controls are `earliest_min`, `earliest_max`, `window_width_min`, `window_width_max`. Freeze the resulting settings before validation. If changing how windows are generated around witness schedules, version that generator and report the resulting distribution bias.

Three planned acceptance levels:

1. **Reliable:** at least 99% observed success on 1,000 locked test scenarios, with Wilson 95% interval and failure categories reported. Treat this as a target, not a promise or proof of 99% population reliability. A stronger claim requires the confidence bound itself to clear the desired threshold.
2. **Competitive quality:** mean paired excess cost at most 5% against independently built best-known feasible plans, median at most 3%, p95 at most 15%, with reference coverage and a joint success/quality metric. Report all five independent training seeds.
3. **Certified near-optimal:** where a valid lower bound L exists for exactly the stated problem/objective, require `(J_policy - L) / L <= 5%`, L > 0. Report the share of test instances with this certificate. If this cannot be established, say “within X% of best-known,” not “within X% of optimal.”

## 2. Code audit: fix these before spending on ablations

Use `scripts/training/train_canonical_ppo.py` with explicit `--state-encoder hetero_graph --action-head complete_gcn` as the primary implementation. The trainer defaults to `self_attention`, so omitting the head flag changes the proposed method. Keep the historical `PPO_VariableActionGNN.py` runner out of the main comparison unless separately auditing feature and action parity.

| Finding from current source | Consequence | Required response |
| --- | --- | --- |
| `CanonicalActorCritic.forward` extracts raw action rows and sends them directly to the action head; only the encoder invokes the normalizer | Actor inputs mix IDs, hours, kWh and small categorical values; the existing action normalization statistics are not used on this path | Normalize continuous action features explicitly; preserve raw feasibility masks and integer action mapping outside normalization |
| `_collect_rollout` updates observation normalizer statistics after collecting old log probabilities and before PPO optimization | The recomputed policy changes even before an optimizer step | Freeze a normalization snapshot for collection and all PPO epochs of that rollout; update statistics at the next rollout boundary; check initial ratios are one within numerical tolerance |
| Every simulator truncation gets value bootstrapping | Time/action budgets are observed problem constraints and exhaustion is scored as failure; bootstrapping may optimize a continuation outside the stated task | Separate task-terminal deadline/action-budget failures from external training cutoffs; bootstrap only the latter |
| Trainer help and older scripts call `time_multiplier` a travel penalty, but `EventDrivenTruckEnv.step` penalizes the increment of total operating time | Training objective and checkpoint selection can disagree | Introduce one explicit objective contract shared by dense rewards, terminal shaping, selection and references; correct labels |
| `operating_time` selection currently uses mean operating time over successes **and failures** | Early termination can artificially lower its cost tiebreak | Add successful operating-time summary and paired comparisons; retain failed rows and the success-first ranking |
| `--init-from` loads policy state only; `CanonicalPPO.save` saves policy/config | Successive halving cannot be a true continuation and jobs cannot reliably resume | Save/restore optimizer, schedule, normalizers, reward-scale state, counters, curriculum, RNG and environment/seed-stream state; otherwise explicitly label fresh fine-tuning and count its budget |
| Canonical shapes are set from the environment; curricula reject width changes | A variable feasible-action head does not by itself permit mixed fleet/customer sizes | Implement explicit padding ceilings + validity masks or ragged state batching before mixed-size training; test transfer separately |
| IDs are scalar numerical features; pairwise lookup IDs are reconstructed from float32 feature rows | False ordinal information and a portability risk for large external IDs | Keep integer IDs/index maps separately; encode identity through relations, not scalar magnitude. California is currently remapped to 0–257 by the loader, so float precision is not a demonstrated California defect |

The original six SoC levels give **62 action slots** at 10/30: 25 station destinations + 30 customer destinations + depot + 6 charging actions. There is no need for aggressive top-k pruning at this size. Only feasible candidates enter the action graph.

Additional correctness gates during implementation: exact action-index round trips, no probability on invalid/padded choices, mask/step agreement, duplicate-service prevention, asymmetric path-cost consistency, queue event accounting, valid depot retirement, nonfinite handling, no hidden witness/RNG information in observations, and schema rejection for incompatible checkpoints. Record failures with exceptions and context: `canonical_harness.evaluate_policy` currently catches `RuntimeError`/`ValueError` and breaks, which can conceal implementation errors as ordinary policy failures.

## 3. Improve the formulation without abandoning GraphPPO

### State graph and state-to-action connections

The current `joint-fleet-v6` state already includes ownership/order, deadlines, action/time budgets, committed ETAs/energy, charging/service timing, FCFS positions/reservations, directed energy/time edges and depot features. Preserve these. Avoid claiming queue or depot information is entirely absent.

The more consequential limitation is the use of that information. `_TypedMessagePassingLayer` uses an affine message in source embedding and edge features, then averages over all sources. With dense validity masks, this aggregates neighbor state separately from its edge cost. It loses distinctions such as “the large-demand customer is the nearby one.” The action head then receives a pooled state vector and an action row, without an explicit destination entity embedding.

Recommended representation:

- Return per-truck/customer/station embeddings plus a pooled fleet embedding from the state encoder. Connect each action to its active-truck and target-node embeddings through integer indices.
- Score action a from `[h_fleet, h_active, h_target(a), normalized_action_features]`, then retain complete action-graph message passing and a masked categorical policy.
- Replace affine edge averaging in an ablation with an edge-conditioned nonlinear/gated message, e.g. `m_ij = MLP(h_i, h_j, energy_ij/B, hours_ij/H, reachable_ij)`. Mask nonexistent/unreachable transport edges as appropriate; retain relation type and direction. Profile its memory before expanding width.
- Encode action kind/truck state/task status as categorical embeddings or one-hot vectors; ownership and route destinations as relations. Remove scalar node/task/truck IDs from neural inputs after their relationships are available.
- Add candidate-specific arrival SoC, escape energy margin, expected time-window slack, next-port-available delay, charge duration/energy, and usable remaining fleet payload. Forecast queues from observable commitments; never use unrevealed future traffic or charging arrivals as facts.
- Condition customer assignment on other trucks' observable ETA, remaining payload, current commitment and cost advantage. Fixed event-order dispatch can otherwise induce poor assignment even when each immediate action is feasible.
- Retain clock/budgets and add bounded time-of-day encoding if traffic depends on daily phase. If introducing heterogeneous charging curves/efficiencies or changing noise across episodes, expose those parameters to the policy. The existing station row does not describe every curve parameter.

Use physical units/scales (battery capacity, payload capacity, horizon) for continuous features, then frozen running normalization if still needed. Structural flags and masks must remain exact.

### Action graph and charging decisions

Keep the basic two-stage decision: navigate to a station, then choose a charge target after arrival when its current state is observed. This preserves feedback and avoids multiplying every destination by every charge amount.

Evaluate charging sets in order:

1. Current `{0.5, 0.6, 0.7, 0.8, 0.9, 1.0}` as the compatibility baseline.
2. A fixed grid including lower levels, e.g. `{0.1, 0.2, ..., 1.0}`. Targets below current SoC remain invalid. The six-level baseline can force unnecessary charging when only a small top-up is needed.
3. A variable event-based set: fixed-grid anchors plus deduplicated targets sufficient for each eligible next customer **and an escape to a charger/depot**, direct depot return, and taper breakpoints. Clip to capacity; include full charge. Compute thresholds using the same energy convention as the execution mask. Keep a separate candidate table with exact target values.

Dynamic targets require a candidate/action mapping extension; the current canonical discrete layout enumerates configuration-level charge values. Do not treat this as an existing CLI feature. Retain a fine-grid reference and report discretization sensitivity; optimality for six SoC levels does not establish optimality with continuous charging. Freeze physical scenario descriptors across charging/mask ablations: rerunning configuration-dependent rejection sampling can change the accepted customers and confound the comparison. Preserve the coarse grid inside richer candidate sets, remap witness action semantics, and recheck certification on the same physical instances.

The current complete action GCN has no action-pair edge features. It communicates the mean of other actions with a residual self term, so it need not collapse all actions, but it cannot directly represent pair-specific substitution/competition. First implement its **exact** efficient aggregation:

`neighbor_mean_i = (sum_j h_j - h_i) / (K - 1)` within each state, with zero neighbors at K=1.

This replaces explicit K(K-1) edges with segmented sums while preserving the proposed head mathematically. Check forward/gradient equivalence during implementation. Evaluate relation-aware action messages or self-attention as explicitly named architecture ablations after the baseline is sound.

Do not add a cross-product of all trucks and all actions initially: the simulator chooses the ready truck. Optional dispatch selection or a bounded WAIT-until-event action changes the MDP and may help with traffic/queues, but needs its own semantics, anti-loop rules, baselines and reference. Similarly, offering fastest/energy-efficient Pareto road paths would broaden the present fixed-path problem and belongs in a separate extension.

### Feasibility masks

Keep three concepts distinct in manifests and evaluation:

- **Execution legality:** the shared engine checks ready state, ownership/order, payload, available station, immediate-leg energy, charge legality and nominal deadline arrival. Under enabled traffic/energy uncertainty it uses the configured worst-case energy multiplier (currently 1.20).
- **Proven necessary completion checks:** safe pruning only when impossibility follows from the stated model. Examples include retiring the last usable owner of an unserved preassigned task or leaving insufficient aggregate payload for remaining unclaimed demand. Avoid double-counting claimed/in-service tasks; account for in-flight trucks. Document proofs and boundary cases.
- **Optional conservative lookahead:** post-delivery escape reserve, station-to-depot reachability, window risk and assignment lookahead. Robust reserve can remove successful realized paths, so report it as a policy restriction, not universal physical infeasibility.

The current route mask checks immediate reachability; it does not generally certify that a delivery leaves enough energy to continue or return. Begin by adding escape margins as features. Compare the existing mask to a shield that requires a direct charger/depot escape under a declared reserve rule. For any stronger claim, examine multi-customer continuations and the selected road-path semantics; do not assume a travel-time triangle inequality for minimum-energy paths.

Keep depot retirement legal when a truck is genuinely surplus. Do not globally mask the depot until all customers are served: that forces unnecessary fleet use and changes the feasible problem. Do not ban consecutive charger visits just because the generator's witness uses at most one station per segment. Do not impose “always charge to full.”

Log feasible-action counts by kind, mask reasons, empty candidate sets, post-delivery dead ends, rejected witness/reference actions and extra mask time. For a heuristic shield, a fallback may restore the base legal set when the shield empties it, but the fallback must be explicit and counted; never relax physical constraints. For tiny proven instances, measure how often a shield removes every optimal action and its resulting objective increase. Use these comparisons to reject harmful pruning.

## 4. PPO, rewards and curriculum

Start with hidden width 128, two state layers, two action layers, 16 environments, rollout length 128 (2,048 transitions/update), four epochs, minibatch 256, learning rate 3e-4, clipping 0.2, GAE lambda 0.95 and entropy coefficient 0.02. These are starting settings, not tuned recommendations. A new graph layer may require minibatch 64/128 with gradient accumulation to retain effective batch size.

Expose gamma/lambda/KL and schedule controls in the trainer rather than invent unsupported flags. Compare gamma 0.99 to 0.999 and, in the objective-aligned arm, gamma 1.0 for the finite task horizon. Per-decision discounting changes the preference for plans with different numbers of event decisions; use elapsed-time discounting only if the desired objective is explicitly time-discounted. Monitor KL, clipping fraction, entropy divided by log(feasible actions) where K>1, value explained variance, gradient norms and success by termination cause.

For the proposed operating-time objective, use dense `-delta(total_operating_time)` plus feasibility shaping with all components logged. Scale completion rewards by total customer count when training mixed sizes. Prefer a documented potential-based progress term (terminal potential zero for every absorbing outcome) over arbitrary milestones; keep a baseline with current shaping. A finite scalar success bonus alone does not mathematically enforce lexicographic expected-success optimization. Select models under an explicit success constraint and tune cost only once reliability is adequate. Avoid terminal battery bonuses that reward buying unneeded energy.

Curriculum stays at 10 trucks / 30 customers during the first campaign; every validation uses the frozen target:

- Train a direct-to-target control from random initialization.
- For the curriculum arm, begin with reduced traffic/service uncertainty and a controlled train-only mix of easier, certified routes; keep capacity 400 kWh and physical units honest.
- Gradually restore target noise, charging physics, windows and congestion. Sampling easier instances must not silently remove difficult target regions from the final distribution.
- Advance when recent training success is at least 90% over 200 completed episodes and target validation is not deteriorating; require a minimum training dwell. Reserve at least half the total budget for the exact target distribution. If early stages stall, diagnose/stop rather than finishing an entire campaign on the easy stage.
- Keep 20% easier examples only during the transition; final convergence and reporting use the exact target. A mixed sampler is an implementation extension; current curriculum switches whole environments.

The primary claim remains PPO from scratch. A separate rescue track may use train-only heuristic/ALNS demonstrations or witness imitation if pure PPO fails, but label it “GraphPPO + imitation,” account for demonstrator compute and regenerate compatible data. A witness is never part of policy observations, validation targets, or online action selection.

## 5. Scenario bank, references and statistical protocol

Create immutable scenario descriptors before the broad sweep. Record map/config/generator/feature/action schema hashes, original and internal node maps, seed, accepted-generation attempt and witness certificate separately. Preserve a reproducible code/environment snapshot including the relevant dirty diff; a Git hash alone does not capture this workspace.

Suggested initial bank: 20,000 training instances, 128 frequent validation instances, a disjoint 512-instance promotion validation set, and 1,000 locked test instances. Extend training to 100,000 distinct instances if repetition or generalization diagnostics justify it. Use the repository split namespaces (`train`, `validation`, `test`) with base seed 0 and disjoint index ranges. Fix the evaluation bank across all training seeds; vary initialization, minibatches and training scenario order. Account explicitly for finite-bank cycling.

Use separate distributions to distinguish generalization from a changed problem: no-window reference; full target; tight-window/queue stress; initial-SoC and noise shifts; and size transfer. Measure generation acceptance, reset cost, geography, charging requirements, route lengths, workload balance and queue incidence. Certified construction is a conditional distribution, not a uniform sample of arbitrary California deliveries. If the 209 stored ports yield little queueing at 10 trucks, say so; a scarce-port stress set is needed to evaluate coordination.

Build three reference layers:

1. **Tiny exact sanity set:** on the California map, start with 1 truck / 1–3 customers, then 2 trucks / 4–6 customers. Run `perfect_information.py` in `information_mode='simulator'` for identical execution constraints and in `realized` mode only as a separately labeled clairvoyant reference. Retain actual search status, lower bound, incumbent, time/node limits and verified replay. Current solver objectives are `makespan` and `total_time`; driving-hour optimality needs implementation work.
2. **10/30 best-known feasible bank:** tune heuristic, rolling-horizon MPC, CP-SAT and multi-start ALNS on validation. Allocate fixed solve budgets, initially 30/120/600 seconds per instance for a reference pilot. Replay candidates through the identical simulator and retain only fully successful plans for that scenario. The generator witness supplies a debugging feasible upper bound, not a deployable nonclairvoyant baseline. Keep independent references without the candidate policy and also report the pooled best-known solution.
3. **Bounds:** use validated exact-simulator frontier bounds when available. Develop a relaxed routing/assignment or branch-and-price reference only with an explicit proof that every original feasible solution is represented and costs are underestimated. Dropping queues can relax cost; restricting stations/routes, limiting route positions without justification, or fixing ownership can exclude better solutions and invalidates a global lower bound. Nominal travel/energy is not automatically a lower bound on stochastic realizations, since draws can be below nominal. Integer rounding also needs directionally safe bounds. The old `optimality_reference.py` prose alone is insufficient to establish these conditions.

The current perfect-information profile artifact already shows the limits of exhaustive search: one 1/1 instance is proved optimal, while an 8-truck/32-customer instance at 100 nodes remains feasible with roughly 49% solver gap. This is a small profiling sample, not a runtime forecast. Do not allocate the entire campaign assuming exact 10/30 solutions will be easy. Existing branch-and-price files in the dirty workspace are ongoing work, not a certified reference.

Distinguish deterministic/clairvoyant discrete optima from the optimal **nonanticipative stochastic policy**. Pathwise clairvoyant cost is an optimistic reference; it does not prove the achievable stochastic policy optimum. For repeated-disturbance evaluation, freeze customer/task instances and resample exogenous randomness independently of policies; this separation needs an explicit scenario-bank API if the current seed couples both. Do not retain only disturbance realizations on which a witness happens to succeed.

For successful policy/reference pairs, report `100 * (J_policy - J_BKS) / J_BKS`; allow negative values when the policy improves an independently frozen BKS. For a valid lower bound, report the certificate separately. Include paired-bootstrap confidence intervals over scenarios, seed-level summaries and clustered intervals for repeated disturbances. Report success on all scenarios, jointly solved coverage, cost quantiles, per-failure category, customer completion and latency. Add the unconditional fraction of all test cases that both succeed and lie within 5% of an available reference; report missing references explicitly so quality cannot improve by dropping difficult cases.

Greedy GraphPPO is the primary deployment mode. Evaluate optional multi-start decoding with fixed 1/8/32-candidate budgets and measured end-to-end latency. Candidate selection must use observable state/nominal or independent predictive simulations, not the held-out environment's unrevealed future draws. Realized best-of-32 rollouts may be reported only as an oracle diagnostic. A GraphPPO-seeded ALNS solution is a separate hybrid method with its search time included.

## 6. Experiment schedule and promotion gates

For true successive halving, separate the stop-at-transition budget from the learning-rate schedule horizon. Use the same predeclared schedule (for example a 12M horizon) for all screened arms; do not anneal to zero at 1M and silently restart the optimizer on promotion. Preserve counters and optimizer state on continuation. Fresh finalist seeds must follow the same recipe and schedule from initialization. This requires trainer extensions beyond the illustrative existing-CLI command below.

All counts below are **training runs**, distinct from SLURM allocations. Budgets count environment decisions, summed across vector environments. Match both interaction budget and report actual SBU/wall cost. Reuse pilots only when their configuration and provenance satisfy the main protocol.

| Phase | Runs | Interaction allocation | Decision gate |
| --- | ---: | ---: | --- |
| P0: correctness and hardware pilot | Up to 6 | 0.1M each; 0.6M total | Numerical/semantic gates pass; choose resource packing from measured throughput/memory |
| A: formulation screening | 10 arms × 3 seeds = 30 | 1M each; 30M total | Compare completion, success, learning curves and paired cost; retain four arms across seeds |
| B: promotion | 4 arms × 3 seeds = 12 | Continue 1M to 4M; +36M | At least 95% success on promotion validation to enter cost refinement; if none qualify, diagnose failures |
| C: convergence | 2 arms × 5 seeds = 10 | Reach 12M per seed; +96M assuming six inherited 4M runs and four new runs | Reliability target and near-BKS quality, plateau analysis; lock architecture/training recipe |
| D: attribution | 8 controls × 3 seeds = 24 | 4M each; 96M | Explain the selected changes at matched budget; use stored 4M winner checkpoints as controls |
| E: scaling | 4 sizes × 3 seeds = 12 | 6M each; 72M | Measure transfer first, then fine-tune/train; report size-specific success/quality/runtime |
| F: locked evaluation | Frozen checkpoints/baselines | Evaluation/reference compute booked separately | Test once after selection; no test-driven redesign |

The full optional training envelope is **330.6M transitions**. The essential path through C is **162.6M**. Phases D/E are released after the essential result is useful. If phase A cannot learn, a screen can be extended to 2M before rejection based on improving completion; log this as an amended budget and extend comparable controls. Do not automatically declare slow learners inferior at 1M.

Screening arms, all sharing the mandatory fixes in section 2:

| Arm | Difference from corrected A0 |
| --- | --- |
| A0 | Proposed hetero state + complete action GCN, current features/charge grid/mask, direct full-target training |
| A1 | Semantic ID/categorical handling and physically scaled features |
| A2 | A1 + explicit active/target embeddings in each action |
| A3 | A2 + nonlinear edge-conditioned state messages |
| A4 | A3 + candidate escape/slack/charge-duration/queue forecasts; base legal mask |
| A5 | A4 + conservative escape/retirement lookahead shield |
| A6 | A4 + 0.1–1.0 fixed SoC grid |
| A7 | A4 + event-based charging candidates |
| A8 | A4 + feasibility curriculum |
| A9 | A4 + curriculum + event-based candidates + objective-aligned gamma/reward shaping; keep shield as a validation-selected option |

This is an incremental screening design with a bundled arm, not a full factorial experiment. It cannot alone identify every interaction. Phase D should include: remove entity links, revert edge-conditioned messages, remove candidate lookahead features, revert charging grid, remove curriculum, revert reward/gamma treatment, replace complete GCN with independent scoring, and replace it with self-attention. If a shield is selected, substitute/add its removal control and update the budget. Compare hard mask vs structural mask as a short diagnostic, with identical invalid-action semantics and feasibility-indicator leakage documented; it is lower priority than making the proposed masked policy strong.

Keep the third-best candidate if confidence intervals overlap the promotion boundary; budget amendments must be explicit. Never select the best random seed and call it the expected method performance.

Scaling starts after 10/30 succeeds: **10/50, 20/60, 30/90, 50/150** (trucks/total customers). Run zero-shot size transfer before fine-tuning. Mix 10/30 replay into fine-tuning to measure retention. Add 10/100, 10/200 or 70/210 only after checking demand, eligible customer count, action budget and generation acceptance. **100 trucks × 3 deliveries = 300 unique customers is impossible on this stored map's 233 noncharging nodes.** It requires a larger map or repeat-location customer tasks, which changes the generator/task representation. Do not oversize padding to a physically impossible workload.

## 7. Snellius allocation and packing

Read-only cluster/account observations on 2026-10-18:

- Account `tdsei17344`; enabled products: `cbuild`, `gpu_a100`, `gpu_h100`, `gpu_mig`, `staging`. Do not assume access to `rome`/`genoa` merely because `sinfo` lists them.
- Approximately **503,554 SBU remain**, expiring 2027-01-27. Snapshot only; recheck before submission.
- A100: 4 GPUs and 72 CPU cores/node; default 18 cores and 120 GiB host RAM/GPU. Partition billing weight 128 SBU per GPU-hour at that resource ratio.
- H100: 4 GPUs and 64 cores/node; default 16 cores and 180 GiB host RAM/GPU. Billing weight 192 SBU/GPU-hour at that ratio.
- MIG: 8 A100 `3g.20gb` slices/node; 9 CPU cores and 60 GiB host RAM/slice. Billing weight 64 SBU/slice-hour. This is a 20 GB device-memory slice, not 60 GB GPU memory.
- `cbuild`: 10 nodes, 32 cores/node; node-exclusive allocations. A nominal request for a few cores may still occupy a whole node; pack reference processes and confirm billed usage with `sacct`.
- `scontrol show config` reports `PriorityFlags=MAX_TRES`, consistent with the resource-ratio billing estimates below; confirm the completed allocation in accounting.
- Listed training partitions have a five-day maximum; prefer restartable 12–24 hour chunks. Thirty jobs is this campaign's concurrency cap, not evidence that the scheduler will immediately run all thirty.

P0 profiles identical workloads under:

| Allocation | Runs sharing it | Initial layout |
| --- | ---: | --- |
| A100, 18 CPUs, 120G RAM | 1 | 16 envs, 8 rollout workers, 1 Torch thread |
| A100, 18 CPUs, 120G RAM | 2 | Each: 16 envs, 7 rollout workers, 1 Torch thread; 9 CPU affinity cores per run |
| A100, 18 CPUs, 120G RAM | 3, only if two underutilize it | Each: 8–16 envs, 4 workers; 6 affinity cores; retain comparable PPO batch size |
| MIG, 9 CPUs, 60G RAM | 1 | 16 envs, 7 workers, 1 Torch thread |
| H100, 16 CPUs, 180G RAM | 2 | Each: 16 envs, 6 workers, 1 Torch thread; 8 affinity cores |
| cbuild, full 32-CPU node | 2–4 CPU runs | Compare cost efficiency if GPU duty cycle is consistently low |

Measure cold/warm reset and certification time separately, rollout transitions/sec, PPO updates/sec, validation fraction, host RSS, peak device allocation, GPU duty cycle and billed SBU. Include contention from concurrent resets. Prefer packing only if aggregate transitions/SBU improves, peak memory retains 20% headroom, and per-run throughput remains at least 70% of the single-run value. Test three-way packing separately; do not assume it works.

H100 must deliver more than 1.5× A100 aggregate throughput to win on these GPU billing weights. An A100 allocation must exceed 2× one MIG slice's throughput to beat its SBU efficiency. These comparisons also depend on queue delays and requested CPU resources; verify actual `AllocTRES`/billing rather than assuming the nominal weights apply to every layout.

Canonical observation storage itself is material: with the current schema, 10/30/25 and 62 actions uses 18,945 float scalars per state, about 74 KiB. A 2,048-transition rollout stores about 148 MiB of observations before gradients/activations. The state relation block grows as `(trucks + customers + chargers)^2`; large GPU batches and nonlinear edge messages can dominate memory. Cache immutable map costs, gather per-state relations efficiently, and consider CPU rollout storage/static-table reconstruction at larger sizes before buying bigger GPUs.

### Enforce the 30-job ceiling globally

Use one training array at a time, or a campaign controller with a shared allocation ledger. `--array=0-29%30` limits one array only. Two independent `%30` arrays can run 60 jobs. If waves overlap, assign caps whose sum is at most 30 (for example screening 18 + promotion 8 + pilots 4), and include reruns/manual training allocations in that sum. Count array elements, not just the parent job ID. Dependencies do not implement the statistical promotion decision; the controller must read validation results and generate the next immutable manifest.

With two runs per A100 allocation, phase A needs **15 allocations**, B needs 6, C needs 5, D needs 12, and E needs 6. The ceiling permits 60 simultaneous runs if enough work is ready, but there is no benefit to submitting weak experiments simply to fill 30 jobs. MIG/single-run layouts use up to 30 allocations for phase A.

Suggested allocation header after P0 (template, not a submitted script):

```bash
#SBATCH --account=tdsei17344
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-14%15
#SBATCH --signal=B:USR1@180
#SBATCH --output=results/california_graphppo/logs/%A_%a.out
```

The future bundle launcher reads two manifest rows per array element and launches both child training processes inside its one GPU allocation. Preserve SLURM's `CUDA_VISIBLE_DEVICES`; each process uses logical `cuda:0`. Partition `os.sched_getaffinity(0)` into disjoint CPU sets, bind each process plus its workers, and set `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. A single enclosing job step with a bundle supervisor avoids two exclusive steps competing for the same GPU. Sharing one allocated GPU between processes is distinct from assigning two independent exclusive GPU steps.

The supervisor must forward USR1/TERM, request atomic checkpoint writes, wait for every child, record exit codes, and fail the bundle when a child fails. Requeue/resubmit only missing work. The batch-shell signal directive alone does not forward signals or save anything. Use unique outputs, caches and per-process scratch directories under `$TMPDIR`; copy final/periodic checkpoints and manifests to persistent storage. Avoid simultaneous environment installation and shared writable transport/scenario caches; prebuild the locked environment and stage read-only artifacts once. Do not use the historical nohup launchers or hard-coded physical GPU indices.

References for scheduler semantics: [Slurm job arrays](https://slurm.schedmd.com/job_array.html), [Slurm sbatch](https://slurm.schedmd.com/sbatch.html), [Slurm GRES](https://slurm.schedmd.com/gres.html). The resource/account numbers above came from this machine's `sinfo`, `scontrol show partition`, `sacctmgr` and `accinfo`, not from generic hardware assumptions.

### Budget scenarios, not an unmeasured runtime promise

For S total transitions, K runs per GPU, packed per-run rate r, and b SBU/GPU-hour:

`GPU-hours = S / (3600 * K * r)` and `training SBU = b * GPU-hours`.

For the optional 330.6M envelope, two-way A100 packing and **hypothetical measured rates** of 100/200/400 transitions/sec per run imply about 459/230/115 GPU-hours, or 58.8k/29.4k/14.7k SBU before validation/reference overhead. A 20% allowance gives about 70.5k/35.3k/17.6k SBU; separately budget long solver sweeps because 20% may not cover them. At 200 transitions/sec a 12M run alone takes about 16.7 hours plus validation, so restart support is necessary for 12-hour allocations.

Release at most 2k SBU for P0 and bounded reference pilots, then approve the phase A–C envelope from measured rates. A suggested internal planning ceiling is 50k SBU before reassessing; it is not a request to spend it now. Daily accounting should report completed transitions, usable checkpoints, failure/restart cost and projected remaining SBU. Faster hardware or extra parallelism cannot fix a formulation failure.

## 8. Concrete implementation order and launch handoff

1. Freeze objective, assignment semantics and reference/full configurations; create scenario manifests and train/validation/test partitions. Record the conditional generator and actual window/charging assumptions.
2. Implement numerical/termination/metric fixes, complete training checkpoints and manifest-driven launch supervision. Version schemas. Retain a short unmodified diagnostic run so numerical corrections are distinguishable from architecture changes.
3. Implement state/action integer links, candidate features and exact segmented action-GCN aggregation. Add the separate ablation switches. Dynamic charging candidates and mixed-size batching follow their explicit gates.
4. Validate the critical invariants and tiny solver replay, then run P0 on compute nodes. Do not run training or expensive generation on the interactive/login service.
5. Build and replay reference incumbents for a train/validation pilot while phase A runs. Review optimality bounds before committing large solver budgets.
6. Execute A → B → C with frozen promotion rules. Add D/E after the primary 10/30 result passes reliability/quality gates. Run the locked final evaluation once.

A baseline command using **existing flags** after a resolved 10/30 YAML has been created is:

```bash
PYTHONPATH=. .venv/bin/python scripts/training/train_canonical_ppo.py \
  --config results/california_graphppo/configs/ca10x3_reference.yaml \
  --state-encoder hetero_graph --action-head complete_gcn \
  --num-envs 16 --rollout-workers 7 --torch-threads 1 \
  --rollout-steps 128 --epochs 4 --minibatch-size 256 \
  --hidden-dim 128 --encoder-layers 2 --action-head-layers 2 \
  --learning-rate 3e-4 --entropy-coefficient 0.02 \
  --total-timesteps 1000000 --train-scenarios 20000 \
  --validation-scenarios 128 --validate-every 100 \
  --selection-objective operating_time \
  --policy-action-mask hard --device cuda:0 --seed 0 \
  --output results/california_graphppo/phase_a --run-name a0_seed0
```

This illustrates the launcher interface; the YAML and new launch/checkpoint features are deliverables of implementation, not files created by this plan. The canonical trainer has no `--num-trucks` or `--num-stops` CLI flags: set 10/30 in YAML. The example operating-time selection requires the successful-cost fix above. For a driving-time campaign, change both reward and reference objective deliberately; `--selection-objective travel_time` alone does not change the current dense reward. For the full target use its frozen full YAML, not the reference YAML.

Completion deliverables: resolved configs, immutable run/bundle manifests, profiling report, reproducible snapshots/checkpoints, per-episode failure-retaining results, reference plans and proof-status records, five-seed learning curves, paired quality/reliability tables, inference latency, and actual SLURM/SBU accounting. A trustworthy result may be “99% success and within 5% of best-known, with only a subset certified near-optimal.” The campaign must preserve that distinction.

## Source map for implementation

- Problem/configuration: `EVRoutingEnv/config_files/config_joint.yaml`; `EVRoutingEnv/data/case_studies/california/metadata.json`; `EVRoutingEnv/models/environment/joint_instance.py`.
- Feature extraction and graph schemas: `EVRoutingEnv/state/features.py`; `EVRoutingEnv/state/representations.py`; node relabeling in `EVRoutingEnv/utils/utils.py`.
- Actor/encoders: `algo/canonical_policy.py`; `algo/canonical_state.py`; `algo/canonical_encoders.py`; `algo/action_heads.py`.
- Masks/execution: `EVRoutingEnv/state/feasibility.py`; `EVRoutingEnv/state/action_mask.py`; `EVRoutingEnv/models/environment/event_driven_env.py`.
- Charging/metrics: `EVRoutingEnv/models/simulation/charging_curve.py`; `EVRoutingEnv/models/simulation/charging_station.py`; `EVRoutingEnv/evaluation/metrics.py`.
- PPO/training: `algo/canonical_ppo.py`; `scripts/training/train_canonical_ppo.py`.
- Evaluation: `scripts/evaluation/canonical_harness.py`; `scripts/evaluation/run_canonical_campaign.py`; `EVRoutingEnv/evaluation/artifacts.py`.
- References: `EVRoutingEnv/baselines/optimization/perfect_information.py`; `EVRoutingEnv/baselines/optimization/optimality_reference.py`; `scripts/evaluation/solve_perfect_information.py`; `EVRoutingEnv/baselines/heuristics/alns.py`.
- Existing exact-search evidence: `results/perfect_information/profile_20261018/fixed_nodes/results.csv`.
- Historical context only: `old_exprimental_notes_and_campaigns/09_graphppo_from_scratch_campaign.md`; `scripts/runners/run_graphppo_v2_campaign.sh`. Old widths, generator versions, distance knobs and result claims must not be inherited without inspection.
