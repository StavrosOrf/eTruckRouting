# California 10T3D GraphPPO campaign status

Updated 2026-10-20T12:12:49.785496+02:00 (Europe/Berlin). Earlier observations below are historical.

The goal remains **active and unmet**: at least 99% successful depot-return completion and at most 5% mean paired operating-hours excess against independently built feasible BKS, with five training seeds and a locked 1,000-scenario evaluation. These BKS comparisons are not certified optimality gaps.

## Actual GPU gate and full-cohort launch

MIG pilot 28365564 completed successfully: three recipes each completed 4,096 actual PPO decisions with seven workers. Encoder, action and value weights changed and remained finite; enabled target and semantic branches changed too. All full checkpoints loaded. This is optimizer/checkpoint evidence, not held-out route performance.

The audited bootstrap submitted array 28367215 with six new runs, reusing three completed controls. It permits three concurrent MIG allocations. Current training states appear in the table below. The warm-cost 4M readout now runs in a two-CPU, zero-GPU step of the existing warm-MC allocation because the pilot had insufficient remaining wall time.

## Message and identity architecture ablation

Prepared six new runs across two recipes and three seeds, with three completed matched controls frozen from v16 instead of retraining. The original graph messages condition on source, target and edge, while the canonical encoder previously conditioned on source and edge before aggregation. An optional zero-initialized target projection restores this dependency; a second option replaces ordinal IDs with explicit physical equality, destination, claim and assignment relations. This is an architectural ablation, not an exact replica of the original implementation or arbitrary-schema support.

Legacy-default logits and values are exactly preserved on 128 training states across separate source processes. New branches receive finite gradients. The identity variant gives identical logits across three physical-node relabelings and retains that property after three tiny optimizer updates; saved predictions reload exactly. These are train-only representation/optimization findings, not route-performance evidence. GPU pilot 28365564 completed 4,096 actual PPO decisions per recipe and released the full cohort. The protected final hold now covers ten cohorts. Five future families reserve at most 25 allocations in total, below 30.

## Paired action-edge uncertainty readout

On jointly successful cases, routing gates minus the standard control change paired BKS excess by -0.97 percentage points (hierarchical bootstrap 95% interval -4.22 to +2.47). Per-seed differences are -2.20, +2.56 and -3.28 points. This does not establish a consistent advantage across three fine-tuning seeds. Critic detachment minus routing gates is +30.26 points (22.51 to 39.28), with worse successful cost in every seed. These are post-hoc development contrasts, conditional on the common parent; the existing promotion protocol is unchanged. Physical scenario payloads and execution configurations match aside from declared policy-mask flags.

## Completed warm-cost 4M snapshots

The three warm-cost runs have reached 4,001,792 decisions. Six selected/final actors are now frozen for a three-seed readout while the MC recipe remains unfinished. The CPU readout is attached after the required GPU pilot gate and bootstrap, with a twelve-minute remaining-time guard and no extra allocation. The full 4M stage may reuse completed sealed outputs only when weights, configuration, source, masks and case bank match. No 4M score is available yet.

## Latest completed action-edge comparison

Three fine-tuning seeds conditional on the same fixed parent, 128 reused development cases; costs condition on success. Standard end-to-end control: 99.21875% completion, 131.36 hours, 18.04% paired BKS excess. Routing gates: 98.95833%, 129.37 hours, 16.92% excess. Gates with critic-encoder detachment: 100%, 161.92 hours, 46.75% excess. Only standard control and routing gates were promoted to 4M. The 5% cost target remains unmet; final validation and test remain unopened.

Physical-node relabeling on 128 fixed train-only states changed 22.40–25.78% of frozen PPO greedy choices across three permutations while preserving physical features, masks and action semantics. This implementation is not strictly node-label invariant. This is a representation diagnostic, not evidence that ID sensitivity causes the California cost gap.

The prior owned-MIG packing step ended with the warm parent. Its first recovery model completed 1M; its second failed before creating a resumable checkpoint. No steady two-recovery-model packing throughput result is claimed. The one unfinished recovery actor is now queued independently. Initial completed single-model MIG BC20+1M runs took 93–112 minutes, including startup and BC; queue wait and separate evaluation are excluded.


## Current training

| Cohort | Array | Running / queued allocations | Target decisions per run |
| --- | --- | --- | --- |
| Capacity / graph messages | 28279539 | 0 / 0 | screened out; partial checkpoints retained |
| Imitation + progress | 28279541 | 0 / 0 | screened out; partial checkpoints retained |
| Undiscounted / cost emphasis | 28360730 | 0 / 2 | 12M; five seeds per recipe |
| Progress features | 28279546 | 0 / 0 | completed 12M; five seeds per recipe |
| Original finalists | 28279550 | 0 / 0 | completed 12M; five seeds per recipe |
| Larger expert dataset | 28360521 | 1 / 0 | 4M; gated 12M |
| Matched recovery data (gated) | 28360534 | 1 / 0 | 1M; gated 12M |
| Routing action edges / critic gradients (gated) | 28360721 | 0 / 2 | 4M; gated 12M |
| Target-conditioned messages / identity relations | 28367215 | 3 / 3 | 1M; gated 12M |
| Packing profiles (two / three runs per A100) | 28305475 / 28305476 | 0 / 0 | cancelled at zero runtime; no measurement |
| Message/identity GPU PPO pilot | 28365564 | 0 / 0 | 4,096 PPO decisions × three sequential models |

Training queue entries: **12**, including **5 running** and **7 pending tasks**. Array throttles currently bound simultaneous training allocations to **9**, within the global limit of 30. Only MIG and A100 are permitted for new campaign submissions; H100 and cbuild are excluded. Warm and recovery continuations use MIG. Action-edge models pack three per A100 allocation; objective models pack two. Every model keeps nine CPUs and seven rollout workers. Node-local scratch quota failures were repaired by moving runtime caches to shared workspace storage. Completed models are skipped and full states resume. Coordinators and evaluation jobs are excluded from training counts.

Campaign coordination and evaluation use gpu_mig; the message/identity GPU pilot uses gpu_mig after zero-runtime migration from A100 based on scheduler estimates. Historical H100 tasks had already terminated before the latest user restriction. New campaign submissions permit only MIG/A100. Completed results and frozen scientific source inputs are retained.

## Overnight checkpoint recovery

Previous long-running arrays failed around 02:20 on October 19 while PyTorch wrote checkpoints. The specific filesystem cause was not established. Recovery audited 55 valid full training states and one missing state. Five cohorts resume model, optimizer, simulator and RNG state; the one run without a full checkpoint repeats its original BC initialization. One damaged selection-policy file was archived, and that run re-evaluates its restored policy before further optimization.

Full states now use compressed, flushed atomic writes with explicit I/O errors and bounded retries; inference weights remain ordinary Torch files written atomically. A real checkpoint compression round trip preserved model, optimizer, simulator and RNG values while reducing 193 MiB to 107 MiB. Running recovery logs show restored interaction counts and further updates. The separate live checkpoint audit passed for all ten capacity policies after each advanced by at least 102,400 decisions; their compressed files passed archive CRC checks and full-state loading. Artifacts are under `recovery_20261019/`.

## Larger-data PPO development comparison

The comparison after 1M additional PPO steps completed on the same opened 128 development cases, with three training seeds and six independent cached references. The table reports checkpoint-selection best actors; BC0 and final actors were also evaluated. Final validation and test remain unopened. Costs include successful episodes only; completion includes all failures.

| Recipe | Mean completion | Mean successful operating hours | Mean paired BKS excess |
| --- | --- | --- | --- |
| large_fresh_cost40 | 98.44% | 139.62 | 25.52% |
| large_warm_cost40 | 98.44% | 131.12 | 18.14% |
| large_warm_mc_cost40 | 97.92% | 134.60 | 20.28% |

The warm cost recipe improves the earlier leading 32.50% excess to 18.14%, with completion rising from 97.92% to 98.44%. It still misses both targets. The warm cost and warm MC recipes advanced to 4M and now resume on MIG array 28320619, coordinator 28320621, with the same seven-worker saved rollout layout. The MC treatment combines removal of the delivery bonus with GAE one; this screen does not isolate their effects.

## Completed matured-control development comparison

Three training seeds, the same 128 opened development cases and six independent cached references. Old control runs completed 12M decisions; larger-data warm recipes have different inherited training histories. This is not a comparison at equal total compute. Costs include successful episodes only.

| Recipe | Mean completion | Successful operating hours | Mean paired BKS excess |
| --- | --- | --- | --- |
| large_warm_cost40 | 98.44% | 131.12 | 18.14% |
| large_warm_mc_cost40 | 97.92% | 134.60 | 20.28% |
| bc_nonlinear_cost40 | 96.88% | 138.77 | 26.39% |
| bc_capacity_nonlinear_links | 97.92% | 142.09 | 28.97% |
| bc_nonlinear_gamma1 | 98.96% | 146.55 | 32.76% |
| bc_progress | 97.66% | 145.94 | 33.91% |
| v2_shield | 95.05% | 147.99 | 34.86% |
| bc_capacity_links | 98.18% | 151.88 | 36.90% |
| bc_links_progress | 97.92% | 151.23 | 37.06% |
| links_shield | 96.61% | 155.16 | 39.30% |
| shield | 97.66% | 152.66 | 40.18% |
| progress_links_shield | 98.70% | 157.82 | 41.40% |

## Earlier PPO development comparison

CPU jobs 28278836 and 28278837 completed a 42-policy comparison: 14 recipes, three training seeds each, 128 shared scenarios at validation offset 2,000, and six cached independent reference methods. The bank is reused development validation. Training budgets differ between recipes. Costs exclude failed episodes; completion statistics retain every failure.

| Recipe | Mean completion | Mean successful operating hours | Mean paired BKS excess |
| --- | --- | --- | --- |
| bc_nonlinear_cost40 | 97.92% | 145.98 | 32.50% |
| bc_progress | 98.44% | 149.94 | 35.78% |
| warm_cost_mc | 97.66% | 150.98 | 36.55% |
| bc_retain01 | 88.80% | 152.46 | 37.38% |
| bc_links_progress | 97.40% | 152.40 | 37.65% |
| v2_shield | 95.31% | 154.28 | 39.52% |
| shield | 96.61% | 153.34 | 40.32% |
| warm_control | 97.66% | 154.06 | 40.90% |
| links_shield | 97.40% | 156.12 | 41.26% |
| bc_nonlinear_gamma1 | 97.66% | 156.84 | 42.03% |
| bc_capacity_nonlinear_links | 97.92% | 156.53 | 42.75% |
| bc_capacity_links | 97.40% | 157.53 | 42.80% |
| progress_links_shield | 97.66% | 161.43 | 45.42% |
| bc_gamma1_control | 96.61% | 157.73 | 46.73% |

In this earlier comparison no recipe satisfies both targets. Cost emphasis has the lowest cost among recipes completing over 95% of cases: 97.92% completion, 145.98 hours and 32.50% mean paired excess. Travel still accounts for about 31.25 percentage points of its pooled successful-pair excess. The stronger retained-expert-loss recipes lose reliability; the retention cohort was screened out during its interrupted 4M stage and is not resumed. This is an early stopping decision, not a convergence result.

Artifacts: `results/california_graphppo/intermediate_20261019_1020/` contains frozen policies, scenario results, provenance hashes, independent paired costs, training-seed uncertainty and cost components.

## Larger train-only expert experiment

New cohort `ca10t3d_large_expert_v14` compares fresh initialization, a fixed trained-policy warm start, and a matched warm-start treatment with GAE lambda one and delivery bonus zero. All three use 20 BC epochs followed by end-to-end PPO. The third treatment is motivated by the earlier matched cost-fine-tuning screen (36.55% versus 40.90% excess); it combines two changes and does not isolate their individual effects. It retains the heterogeneous graph encoder and complete variable-action GCN, nonlinear edge messages, action-to-entity links, gamma one and cost-emphasis rewards. The exact capacity restriction is used consistently in BC and PPO; additional work-escape restrictions remain disabled.

The verified archive contains 183,697 decisions from 4,007 successful episodes among 4,096 training scenarios, with train seeds 600,000–604,095. Whole episodes are assigned to the BC holdout. BC normalization observes the training portion; warm starts also retain previously acquired training statistics. The archive is eight times the earlier teacher bank by episode count. Remasking pruned 3,363 candidate rows and zero expert labels. All 34 model/state/utils files match the anchored benchmark source; the YAML physics is unchanged.

All warm-start seeds use the same frozen seed-zero parent. Their seed variability is conditional on that parent; the fresh-initialization arm varies all initial model weights. Every graph and action-network parameter remains trainable. No solver is invoked to select deployed neural-policy actions.

Initial array **28279916** retains its original six run definitions in three allocations. Supplementary array **28280299** adds three disjoint runs in two allocations, giving nine runs and five allocations in this cohort. CPU job **28280300** waits for both arrays and compares BC-only, selection-best PPO and final PPO checkpoints on the opened 128-case development bank. Coordinator **28280303** waits for that report before the standard validation-gated 4M promotion, which retains at most two recipes. The original pending CPU jobs (28279949 / 28279953) were cancelled with zero runtime when this treatment was registered; no GPU training was restarted. A zero-runtime premature submission (28279715) was cancelled before input setup completed; it produced no training artifacts.

## Recovery-data continuation

The frozen BC-only failure replays matched all nine original development outcomes. These time-limit failures repeatedly charged at full battery and moved between chargers while customers remained. A controlled training-only gradient audit measured median weighted critic-to-policy encoder gradient norm ratio 2.05 and near-zero mean cosine; this does not establish critic interference as the cause.

On 128 train-only cases, the original-plan teacher retained 68 cases at 80% teacher control and 18 at 50%, largely because its labels became invalid after student deviations. The explicit mask-respecting recovery expert retained 126 and 119 cases, respectively, with zero final rejected labels. The pure nominal control retained 127. These are mixed collection behaviors, not deployed GraphPPO results.

Pilot data audits passed in an existing campaign allocation and standalone CPU job 28296798. Full collection 28297279 completed in 1:03:01 on 4,096 fresh training cases (700128–704223), with one shared initial ALNS plan per case and teacher probabilities 100%, 80% and 50%. The full archive integrity audit passed. It retained 4,008 nominal-teacher episodes, 3,965 at 80% teacher control and 3,886 at 50%, with zero final rejected labels. Archives are in campaign-owned scratch storage. These collection outcomes are not deployed neural-policy performance.

Recovery array 28320624 has been submitted after recording all ten completed original runs and verifying all data/source/parent/resource gates. Coordinator 28320625 runs on gpu_mig. The recovery array uses one model per MIG slice and at most five simultaneous allocations. The metadata handoff was executed inline after cancelling its unused pending jobs at zero runtime. It verifies full data hashes, unchanged physical configuration and parent weights, all original five-seed 12M completion records, absence of original training jobs, and the 30-allocation limit. The audited nine BC+PPO runs are now submitted as nine single-model MIG tasks, at most five simultaneously. It compares nominal, 80% and 50% data from the same cases; all start from the fixed seed-zero 1M warm-cost actor. It selects at most two recipes for 4M, then expands to five seeds for 12M. No solver chooses actions at neural-policy deployment.

Both the larger-data and recovery cohorts freeze and score best/final actors at 4M before further training can overwrite them. All readouts use the opened 128-case development bank; the shared final validation and locked test remain protected.

## Trained-policy failure readout

Six first-failure replays from the 1M warm-cost and warm-MC actors matched their original outcomes. Warm-cost failures still looped through chargers. The selected MC trajectories included two exhausted feasible-action sets and a retirement with one customer left. Counterfactual queries exposed legal-action fallback in all three MC cases and one warm-cost case. Enabling the existing work-escape predicate rejected no additional recorded choices. This is a diagnosis of selected failure trajectories, not measured population frequency or a demonstrated fix. Physical execution, current training masks and expert-data collection settings were preserved. Artifacts: `results/california_graphppo/trained_1m_mask_diagnostic_20261019/`.

## Energy-mask inference experiments and initialization budget

The optimistic energy/payload check uses observable state, minimum energy multipliers and charging connectivity. It preserved all 11,920 decisions in 252 successful independent reference routes. On six frozen warm actors and the same 128 development cases, warm-cost metrics were unchanged (98.44%, 131.12 hours, 18.14% paired excess). Warm MC completion changed from 97.92% to 98.18%, with essentially unchanged excess (20.28% to 20.29%). This is not a substantial improvement or a newly trained policy.

The separate 1.2-factor energy-risk restriction is stronger than a necessary physical-feasibility test. With fallback included, it rejects two decisions on one successful independent route; that scenario has no remaining observed successful route in the two-reference audit pool. It failed the predeclared coverage gate and is not promoted into training. The existing BKS pool is unchanged. Vectorized candidate calculations matched scalar results on 9,035 legal candidates across 542 recorded decisions; this is an implementation audit on those trajectories.

The 1M counts in the larger-data warm recipes are additional PPO interaction budgets. Their common parent was selected after 3,174,400 PPO steps. The fixed seed-zero actor used by recovery v15 inherits a selected chain of 4,096,000 PPO steps plus the earlier and larger-data BC phases. Fresh and warm recipes have matched additional PPO budgets, not matched total training histories. Warm fine-tuning seed variability is conditional on the common parent. Details: `results/california_graphppo/initialization_lineage_20261019.json`.

The completed matured-control comparison scored 18 selected original, progress and imitation actors after their three-seed 12M run budgets, alongside six cached larger-data warm actors. The inherited training budgets differ. Both imitation recipes met the prespecified dominated-cohort screen; their paired excess was 37.06% and 33.91%, versus 18.14% for warm cost, with lower completion. Their pending coordinator was cancelled and the last four runs stopped gracefully with checkpoints retained. The progress cohort remains active because one recipe achieved 98.70% completion, so it did not meet the screening rule. This is compute screening on opened development cases; final validation and test remain unopened.

## New MIG training progress

New stages use one model per MIG slice, nine CPUs and seven rollout workers. The six-run warm stage resumes its prior 1M state; its logged counters include that prior stage. Recovery and action-edge models first perform 20 BC epochs. BC losses and rollout completion are training diagnostics, not held-out benchmark performance.

| Run | Phase | BC epochs completed | PPO decisions logged |
| --- | --- | --- | --- |
| large_warm_cost40_s0 | completed milestone | 0 | 3,993,600 |
| large_warm_mc_cost40_s0 | completed milestone | 0 | 3,993,600 |
| large_warm_cost40_s1 | completed milestone | 0 | 3,993,600 |
| large_warm_mc_cost40_s1 | completed milestone | 0 | 3,993,600 |
| large_warm_cost40_s2 | completed milestone | 0 | 3,993,600 |
| large_warm_mc_cost40_s2 | PPO | 0 | 1,413,120 |
| recovery_nominal_s0 | completed milestone | 20 | 993,280 |
| recovery_teacher80_s0 | completed milestone | 20 | 993,280 |
| recovery_teacher50_s0 | completed milestone | 20 | 993,280 |
| recovery_nominal_s1 | completed milestone | 20 | 993,280 |
| recovery_teacher80_s1 | completed milestone | 20 | 993,280 |
| recovery_teacher50_s1 | completed milestone | 20 | 993,280 |
| recovery_nominal_s2 | completed milestone | 20 | 993,280 |
| recovery_teacher80_s2 | completed milestone | 20 | 993,280 |
| recovery_teacher50_s2 | PPO | 20 | 174,080 |
| edge_control_cost40_s0 | queued | 0 | 0 |
| edge_route_cost40_s0 | queued | 0 | 0 |
| edge_control_cost40_s1 | queued | 0 | 0 |
| edge_route_cost40_s1 | queued | 0 | 0 |
| edge_control_cost40_s2 | queued | 0 | 0 |
| edge_route_cost40_s2 | queued | 0 | 0 |
| message_control_s0 | completed milestone | 0 | 0 |
| message_target_s0 | BC | 8 | 0 |
| message_target_identity_s0 | BC | 7 | 0 |
| message_control_s1 | completed milestone | 0 | 0 |
| message_target_s1 | BC | 8 | 0 |
| message_target_identity_s1 | queued | 0 | 0 |
| message_control_s2 | completed milestone | 0 | 0 |
| message_target_s2 | queued | 0 | 0 |
| message_target_identity_s2 | queued | 0 | 0 |

## Frozen-policy decoding study

CPU job 28309962 compares the three selected warm-cost actors under greedy decoding and sampling temperatures 0.25, 0.5 and 1.0. Each positive temperature has three predefined policy RNG repetitions per actor, independently seeded from the simulator. Every executed trajectory and failure is retained; there is no best-over-rollouts selection. Repetitions are not treated as independent trained policies.

All three zero-temperature replays must exactly match the cached greedy outcomes, costs and action-call counts on 384 actor/case pairs. The study uses only the same opened 128 development cases and six cached references, with unchanged weights and physical configuration. A discrepancy in a live run between sampled training completion and one greedy validation checkpoint motivated this comparison; different scenario banks also contribute, so that observation alone does not establish a decoding effect.

| Decoder | Mean completion | Successful operating hours | Mean paired BKS excess |
| --- | --- | --- | --- |
| greedy_repeat | 98.44% | 131.12 | 18.14% |
| sampling_0p25 | 98.78% | 132.64 | 18.90% |
| sampling_0p5 | 98.78% | 131.95 | 18.93% |
| sampling_1p0 | 98.18% | 135.10 | 21.58% |

Job 28309962 completed in 30:16. All 384 greedy replay controls exactly matched cached costs and outcomes. Positive temperatures modestly changed reliability but all worsened BKS excess; none meets both targets, so sampling is not promoted as a solution.

## Mature-cohort compute screens

Objective MIG job 28320175 completed. The ready capacity screen now executes in owned warm-training step 28320619_0.0 with two CPUs and no GPU access, after its unused pending MIG job was cancelled at zero runtime. Both first verify that both recipes completed 12M decisions in seeds 0, 1 and 2, freeze the selected actors, and evaluate the same 128 opened development cases against the unchanged six-method reference pool. The anchor is the cached three-seed larger-data warm-cost result; inherited training histories differ.

The prespecified rule stops remaining training only when neither recipe improves mean completion over the anchor and both have at least ten percentage points more paired BKS excess. A graceful stop retains partial checkpoints; a dependent CPU job verifies terminal accounting and full-state loading, then rechecks shared evaluation readiness. Already completed five-seed cohorts and active coordinators are preserved. These screens add no GPU allocations and do not open final validation or test.

## Routing-edge GraphPPO ablation

A new optional action-GCN residual learns routing relationships between feasible candidate actions from observed graph travel and energy features. The physical action set, hard masks and simulator are unchanged. An additional treatment blocks only critic-loss gradients into the shared encoder; actor gradients still train the complete encoder and action network. Zero-output gates preserved parent logits and values exactly on 128 training states. The actual 4,096-step GPU PPO pilot completed, changed encoder and gate parameters, and produced loadable policy and full training checkpoints. These are implementation and training-path results, not evidence of improved held-out performance.

Registered cohort `ca10t3d_action_edges_v16` has nine matched runs: three seeds each for the standard GCN, routing-edge gates, and gates with critic-gradient isolation. All use the same 4,008 nominal-teacher episodes, fixed parent and 20 BC epochs. The audited bootstrap completed inline and submitted action-edge MIG array 28320681, coordinator 28320683. It verified source/parent/data hashes, released imitation slots, the five-allocation family throttle, and explicit zero-runtime cancellation of the unexecuted A100 packing profiles. The operational GPU layout keeps one model and seven rollout workers per allocation; no profile-based performance claim is made. It evaluates BC0, selected PPO and final PPO at 1M, promotes at most two recipes to 4M, then expands to five seeds at 12M.

A live full-campaign audit loaded the actual post-BC PPO states: 153,600 decisions for the control and 102,400 for both routing-gate variants. Encoder, action-network and value-network parameters changed after BC; both gated models also changed routing-gate parameters. All checked neural parameters were finite. The critic-isolation actor encoder still updates through PPO. This establishes actual end-to-end optimization, not improved held-out performance.

The new cohort replaces imitation training slots; recovery v15 separately replaces original-cohort slots after all five original seeds complete. Both remain in the shared nine-cohort final-evaluation hold. The common evaluator now uses the audited source that reads both legacy and edge-gated model configurations.

## Larger-data startup and BC-only readout

The following 64-case results are reused checkpoint-selection observations. They are not the independent 128-case development comparison. Operating hours include successful episodes only.

| Run | BC epochs | PPO decisions logged | Initial completion | Initial operating hours |
| --- | --- | --- | --- | --- |
| large_fresh_cost40_s0 | 20/20 | 993,280 | 93.75% | 126.40 |
| large_warm_cost40_s0 | 20/20 | 993,280 | 95.31% | 120.87 |
| large_fresh_cost40_s1 | 20/20 | 993,280 | 93.75% | 124.66 |
| large_warm_cost40_s1 | 20/20 | 993,280 | 96.88% | 122.19 |
| large_fresh_cost40_s2 | 20/20 | 993,280 | 96.88% | 125.03 |
| large_warm_cost40_s2 | 20/20 | 993,280 | 95.31% | 120.04 |
| large_warm_mc_cost40_s0 | 20/20 | 993,280 | 96.88% | 121.66 |
| large_warm_mc_cost40_s1 | 20/20 | 993,280 | 98.44% | 124.29 |
| large_warm_mc_cost40_s2 | 20/20 | 993,280 | 96.88% | 124.13 |

Early evaluation monitor: 28280824; state: `completed`. It uses one CPU while awaiting the nine immutable BC-only policies, then submits one packed CPU evaluation on 128 opened development cases. The 1M comparison reuses those results only after checking policy weights, configuration, source, masks and scenario coverage; otherwise it evaluates BC-only policies itself.

Early BC-only scoring job: **28281262**.

| BC-only recipe | Mean completion | Mean successful operating hours | Mean paired BKS excess |
| --- | --- | --- | --- |
| bc0_large_fresh_cost40 | 92.97% | 141.29 | 28.00% |
| bc0_large_warm_cost40 | 90.62% | 136.21 | 25.56% |
| bc0_large_warm_mc_cost40 | 91.67% | 139.41 | 27.29% |

These are the additional-BC checkpoints before PPO updates in this cohort. Warm starts inherit a parent already selected after 3,174,400 PPO steps; seed variability is conditional on that parent. Final validation and test remain unopened.

## Completed reference-mask availability audit

CPU audit **28280487** completed 256 new independent solver executions on the same opened 128 development scenarios. ALNS and CP-SAT each completed 126/128 cases, with successful operating-hour means of 111.99 and 118.68 respectively. Every initial physical descriptor matched the cached cases. None of the three queried masks (completion only, relaxed capacity, exact capacity) rejected any of the **11,920 actions in 252 successful reference routes**. The references executed their ordinary actions without mask repair.

The combined new reference pool contains a mask-admissible successful route on all 128 cases. Selecting the cheapest successful reference separately for each case averages 0.35% excess over the cached six-reference BKS (paired bootstrap interval [-0.61%, 1.29%]). This is oracle reference-route selection and establishes observed route availability, not a deployed policy result or an optimality certificate. It supports concentrating the next experiments on learning those efficient choices. The original BKS pool was not revised. Artifacts are in `results/california_graphppo/reference_mask_audit_20261019/`.

## Final evaluation protocol

All ten registered cohorts share one final selection; capacity, retention and imitation are explicitly screened out. Five potentially active families reserve at most 25 future training allocations, below the global limit of 30; completed qualifying original policies remain selection candidates. The evaluator supports the trained mask flags and excludes only four declared policy restriction flags when comparing execution configurations. Physical descriptors and frozen source inputs are checked. The disjoint 512-case final-validation bank and locked 1,000-case test remain unopened. Opening the test requires both validation targets and complete independent feasible-reference coverage.

Current machine-readable observations: `results/california_graphppo/status.json`.

## Earlier campaign observations

All five cohorts use identical target YAML configurations. The initial corrected-mask and progress cohorts share all 34 simulator/state/utils files. The new capacity variant changes only the optional policy-mask code among those files; physical transitions and hard execution legality are preserved. The proposed heterogeneous state encoder, complete variable-action GCN, hard execution mask and PPO are retained. New features describe nominal energy-safe charging connectivity to remaining eligible customers and depot return; they use no witness or future disturbance draws.

## Evidence behind the change

A known validation failure made 535 charger revisits after serving 28/30 customers. The fleet still had sufficient payload. The remaining customers (nodes 71 and 72) can be served safely from station 233; the policy circulated between stations 42 and 193. Replaying the certified witness succeeds and the corrected mask rejects none of its actions in this scenario. The new features expose useful-station reachability and charging-hop distance to the policy.

## Measured pilot

The 65,536-decision completion-mask checkpoint solved 31/32 paired validation scenarios (96.9%). Its mean excess operating cost against the best independent heuristic/MPC/ALNS/CP-SAT result was 6.58% on solved cases, with 95% paired-bootstrap interval [1.70%, 11.60%]. This is a pilot, not the final locked evaluation or a certified optimality gap.

## Promotion and evaluation

Validation-gated coordinators 28260658 and 28261578 can promote qualifying formulations to 4M and 12M decisions and expand finalists to five seeds. A singleton bundle retains the original worker count, promoted sources are pinned, and the final loop limit is rounded to 12,001,280 decisions so the 12M milestone is reachable by the trainer. Clean exits before a milestone resume their checkpoints.

The last completed cohort will gather qualifying five-seed finalists from both cohorts and schedule one shared, disjoint validation rescore (512 scenarios, validation offset 10,000). Recipe selection uses validation only. A qualifying recipe then opens one locked 1,000-scenario test set in eight CPU shards, retaining failures, verifying physical/configuration parity and checkpoint/source hashes, and reporting uncertainty across scenarios and training seeds. No completed result is inferred from job completion alone.

The original benchmark configuration remains fixed: joint assignment of 30 total customers among 10 trucks, 400 kWh batteries, shared chargers, stochastic travel/energy/service, depot return and existing charging parameters, with time windows disabled. Three deliveries per truck defines the workload rather than fixing learned ownership.

Tracked campaign allocations have used approximately 6,091.2 SBU as of 2026-10-18T20:14:48.195486+02:00 (elapsed allocated time × billing TRES, excluding batch/extern/step duplicates and including running allocations). The training concurrency ceiling remains 30 jobs.

Current machine-readable progress: `results/california_graphppo/status.json`.

## Latest checkpoint observation

At 2026-10-18T16:18:28.541326+02:00, ten original runs had reached the 1M milestone; two were still finishing. The new cohort had reached 204,800–307,200 decisions at its latest recorded validations. One remaining-work-feature run (`progress_links_shield_s1`) reached 64/64 validation successes; the other seeds had not all reached the reliability target. This is checkpoint selection on reused validation scenarios, not locked-test evidence. Eight training allocations were live at the last queue check.

## First promotion and independent 1M rescore

At 2026-10-18T16:24:41.973552+02:00, all twelve original runs completed their 1,001,472-decision first stage with exit code zero. The coordinator selected `shield` and `links_shield` across all three seeds and started array **28262168** (three A100 allocations, six resumed runs) toward 4M decisions. Runtime logs confirm restoration from 1,001,472 decisions under the corrected-mask source. Coordinator **28262169** follows this array.

The completion-mask arm's best selection success rates were 62/64, 63/64, and 64/64 across seeds 0–2, averaging 98.44%; this remains below the final target and uses reused selection scenarios. All twelve original best checkpoints were frozen with their 1M-stage provenance for this separate comparison. CPU job **28262186** is rescoring them on 128 disjoint intermediate validation scenarios (namespace validation, offset 2,000), alongside heuristic, MPC, ALNS and CP-SAT references. Input hashes pin the source, map, configuration, checkpoints and helper files. Results will retain failures, check physical/execution parity and report paired costs plus uncertainty across training seeds. This check uses the corrected completion mask for shielded policies and does not open the final test set.

Nine training allocations were live at this check: six remaining-work-feature allocations and three promoted original-cohort allocations. The initial architecture is general, but current progress-feature checkpoints cache the training map's station connectivity and all checkpoints retain configured padding limits; no cross-map generalization is claimed.

## Stronger references and train-only imitation cohort

At 2026-10-18T16:33:10.875068+02:00, the execution audit found that the original ALNS/CP-SAT policies fall back to greedy customer selection when a truck's planned route is empty or exhausted, often consuming other trucks' assigned work. On the first pilot scenario, the original ALNS plan recorded 48 fallbacks in 60 decisions; its nominal objective was 79.35 hours and realized operating cost 299.28 hours. This weakens the reference and cannot support a strong near-optimality claim.

An opt-in `respect_planned_routes` execution variant returns empty/exhausted routes to depot while retaining recovery for incomplete plans and unfinished own-route work. Existing defaults and running training sources are preserved. Independent CPU audit **28262250** compares old and new executors and the three frozen 1M shielded policies on 32 pilot validation scenarios. The plan-respecting ALNS has completed: **31/32 successful, 100.93 mean operating hours on successful cases**, versus the old ALNS's 32/32 and 254.19 hours. These means use different successful subsets; the completed audit will report paired per-scenario costs. Its one failure timed out after serving 29 customers. CP-SAT comparison remains running.

All three cohorts now share six final independent references, retaining the original four and adding plan-respecting ALNS (20,000 iterations / 10-second cap) and CP-SAT. Evaluation source `ca10t3d_v1/source_reference_v4` differs from the progress-v3 source only in the two reference implementations and method-alias handling; actor and physical simulator code match. Final reference settings are hashed when evaluation inputs are frozen. The original four-reference pilot gap remains historical evidence, not the final criterion.

New cohort `ca10t3d_imitation_v5` contains two GraphPPO recipes × three seeds: remaining-work features with and without action-to-entity links, initialized by behavior cloning and continued with PPO. CPU collection job **28262302** gathers 512 plan-respecting ALNS demonstrations exclusively from train seeds at offset 500,000. Only successful trajectories whose actions obey the same completion mask are retained. Seed lists, configuration and archive hashes are recorded. Dependent kickoff **28262303** will start three bundled A100 allocations toward 1M PPO decisions; successful recipes use the existing validation-gated 4M/12M/five-seed schedule. BC uses 20 epochs at learning rate 0.0003; subsequent PPO uses learning rate 0.0001 and entropy coefficient 0.005. BC is skipped when resuming a full training checkpoint.

The teacher is an additional initialization method; this cohort is explicitly recorded as imitation-assisted GraphPPO. Selection and final test remain shared across all cohorts. The goal remains active and unverified against the strengthened references.

## Validation gate and reference failure trace

At 2026-10-18T16:39:20.487517+02:00, the final selection pipeline was strengthened to score all six independent reference methods on the same 512 disjoint validation scenarios as the frozen five-seed finalists. Opening the 1,000-scenario test now requires both at least 99% mean successful completion and at most 5% mean paired operating-hours excess, with complete feasible-reference coverage. Recipe selection uses paired excess rather than unpaired successful-case mean costs. Recipes that miss either target leave test data unopened.

Diagnostic job **28262475** replayed the observed plan-respecting ALNS failure on validation seed 1,000,000,027. After 29 deliveries, truck 4 retained enough payload for customer 195 but repeatedly alternated between stations 52 and 172 and charged to full. The final 100 decisions consisted of 25 repetitions of the same four actions; the episode hit the 200-hour horizon. This is a charging-connectivity navigation failure rather than lack of payload. The full trace is `ca10t3d_v1/evaluation/reference_executor_v4/failure_trace.json`. The new GraphPPO progress features expose charging-hop distance to useful stations; no hardcoded location or witness is added to the policy.

The imitation cohort's physical source was compared with the pinned reference-v4 evaluator: all **34** simulator/state/utils Python files match, as do all **10** algorithm Python files other than the deliberately revised behavior-cloning module. Detailed hashes are in `ca10t3d_imitation_v5/physical_parity.json`. Demonstration collection and both reference evaluations remain live; no final performance result has been inferred from partial rows.

## Completed reference audit

At 2026-10-18T16:41:17.098122+02:00, CPU audit **28262250** completed successfully with all physical/configuration checks passing. Plan-respecting CP-SAT solved all 32 pilot cases with mean operating time **110.07 hours**; plan-respecting ALNS solved 31 with **100.93 hours** on its solved subset. The six independent methods collectively provide a feasible reference on every case.

The three frozen 1M completion-mask GraphPPO checkpoints each solved 32/32, but their mean paired excesses against the strengthened independent best-known reference were **56.32%, 74.56%, and 67.76%**. Their three-seed average is **66.21%**, well above the 5% target. These are the reused pilot validation seeds, not the independent 128-scenario check or the final test. The earlier apparent closeness to the original executors does not establish near-optimality. The imitation-assisted cohort and continued PPO optimization must improve cost substantially; success rates alone are insufficient. The raw paired report and three-seed uncertainty assessment are preserved under `ca10t3d_v1/evaluation/reference_executor_v4/`.

At 2026-10-18T16:44:15.866815+02:00, CPU reference job **28262603** was queued to add plan-respecting ALNS and CP-SAT to the disjoint 128-scenario comparison (validation offset 2,000). Four scenario shards run inside one CPU allocation; raw rows and input hashes are preserved. Dependent report job **28262604** will combine these results with the twelve frozen 1M policies and original four references from job 28262186, checking every shard's execution manifest and full case coverage. No test seeds are used. Demonstration collection 28262302 remains live and its dependent GPU kickoff 28262303 has not yet run.

At 2026-10-18T16:45:46.514599+02:00, demonstration collection **28262302** and kickoff **28262303** completed successfully. The archive retains **499/512** successful train episodes and **22,923** transitions (634 MiB compressed), with SHA-256 `a0d263fdbfe9099a6a2e7135cddec3de1922c051c4552c832710e3ee7f74692f`. GPU array **28262666** has started all three bundled allocations (six runs: two recipes × three training seeds); coordinator **28262667** follows the 1M stage. There are now **12 live training allocations**, below the 30-job limit. Strong-reference validation **28262603** has also started its four CPU shards.

## Capacity, message-passing and objective experiments

At 2026-10-18T20:08:22.785996+02:00, the first three cohorts had all promoted to 4M training. The imitation finalists reached best reused-selection scores of 64/64 across all three training seeds, with mean successful operating costs around 139–148 hours. These checkpoint-selection observations do not establish held-out near-optimality.

The completed disjoint 128-case capacity ablation compares identical six BC-only checkpoints. Added necessary packing checks raised pooled completion from **365/768 (47.53%)** to **406/768 (52.86%)** and preserved all 32 replayed feasibility witnesses. This modest gain requires on-policy learning and does not satisfy the target by itself.

Travel contributes **60.35 of 62.31 percentage points** of pooled excess on solved 1M paired cases; candidate policies use **7.73 trucks** versus **4.25** for independent references. The affine graph message separately aggregates source states and transport edges on dense relations, losing their association. New optional nonlinear messages activate each source/edge pair before pooling, with the same parameter count and canonical inputs. Cohort `ca10t3d_capacity_v7` compares both message rules under the capacity mask.

Teacher collection **28265619** uses 512 train scenarios with the new mask. Kickoff **28265621** starts the capacity/message cohort after its archive is complete. Cohort `ca10t3d_objective_v8` shares the archive; kickoff **28265725** starts two nonlinear-message recipes with `gamma=1`: the existing operating-time weight and a stronger cost-emphasis weight. The latter scales time, success and failure coefficients by four while keeping the delivery reward fixed. Benchmark physics and evaluation rewards remain common.

All five cohorts share the 512-case validation gate and one locked 1,000-case final test. Their final-stage envelope is at most **25 concurrent training jobs**, each packing up to two runs, below the limit of 30. Equations and evidence are in `california_graphppo_objective_audit.md`. SLURM automation continues; the goal tracker currently reports `usageLimited`, and no completion claim has been made.

At 2026-10-18T20:13:56.728339+02:00, collection **28265619** and both kickoffs completed with exit code zero. Arrays **28265804** (capacity/message ablations) and **28265802** (objective ablations) started all six GPU bundles, totaling twelve new training runs. Their coordinators are **28265805** and **28265803**. Both cohorts loaded the same archive, retaining 499/512 successful train episodes and 22,923 transitions; SHA-256 `88a3b4ed6c21f7759d12530eba87094041b91d35f10e8dbb3d1d56393ba7aa1b`. Runtime configuration checks confirm all intended mask, graph-message, discount, time-penalty and seed settings. All 47 core physical/model/trainer/collector files match between these two cohorts. Nonlinear-message models are executing BC gradient updates successfully. The latest queue contains **12 running training allocations**, below the 30-job ceiling.

At 2026-10-18T20:17:39.737624+02:00, the original cohort completed its 4M stage and promoted `shield` and `links_shield` to **12M decisions with five training seeds each**. Array **28265867** runs ten policies in five bundles; coordinator **28265872** follows it. The other two earlier cohorts remain in their 4M stages. The new twelve-run cohorts completed all 20 BC epochs without runtime or memory errors and entered initial episode validation. Effective objective coefficients were checked against the resolved run configurations. The latest queue has **17 running training allocations**; all planned final-stage allocations total at most 25. Near-optimality remains unverified.
