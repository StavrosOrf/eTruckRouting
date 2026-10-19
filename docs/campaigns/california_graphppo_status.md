# California 10T3D GraphPPO campaign status

Updated 2026-10-19T12:22:29.178304+02:00 (Amsterdam time). Earlier observations below are historical.

The goal remains **active and unmet**: at least 99% successful depot-return completion and at most 5% mean paired operating-hours excess against independently built feasible BKS, with five training seeds and a locked 1,000-scenario evaluation. These BKS comparisons are not certified optimality gaps.

## Current training

| Cohort | Array | Running / queued allocations | Target decisions per run |
| --- | --- | --- | --- |
| Capacity / graph messages | 28279539 | 5 / 0 | 12M; five seeds per recipe |
| Imitation + progress | 28279541 | 5 / 0 | 12M; five seeds per recipe |
| Undiscounted / cost emphasis | 28279544 | 5 / 0 | 12M; five seeds per recipe |
| Progress features | 28279546 | 5 / 0 | 12M; five seeds per recipe |
| Original finalists | 28279550 | 5 / 0 | 12M; five seeds per recipe |
| Larger expert dataset | 28279916 + 28280299 | 5 / 0 | 1M initially; gated 4M / 12M |

Observed training allocations: **30 total**, including **30 running** and **0 pending**. Six active cohorts need at most 30 allocations at their final stage. Each A100 allocation packs up to two runs; coordinators and CPU evaluation jobs are not training allocations.

## Overnight checkpoint recovery

Previous long-running arrays failed around 02:20 on October 19 while PyTorch wrote checkpoints. The specific filesystem cause was not established. Recovery audited 55 valid full training states and one missing state. Five cohorts resume model, optimizer, simulator and RNG state; the one run without a full checkpoint repeats its original BC initialization. One damaged selection-policy file was archived, and that run re-evaluates its restored policy before further optimization.

Full states now use compressed, flushed atomic writes with explicit I/O errors and bounded retries; inference weights remain ordinary Torch files written atomically. A real checkpoint compression round trip preserved model, optimizer, simulator and RNG values while reducing 193 MiB to 107 MiB. Running recovery logs show restored interaction counts and further updates. The separate live checkpoint audit passed for all ten capacity policies after each advanced by at least 102,400 decisions; their compressed files passed archive CRC checks and full-state loading. Artifacts are under `recovery_20261019/`.

## Latest independent development comparison

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

No recipe satisfies both targets. Cost emphasis currently has the lowest cost among recipes completing over 95% of cases: 97.92% completion, 145.98 hours and 32.50% mean paired excess. Travel still accounts for about 31.25 percentage points of its pooled successful-pair excess. The stronger retained-expert-loss recipes lose reliability; the retention cohort was screened out during its interrupted 4M stage and is not resumed. This is an early stopping decision, not a convergence result.

Artifacts: `results/california_graphppo/intermediate_20261019_1020/` contains frozen policies, scenario results, provenance hashes, independent paired costs, training-seed uncertainty and cost components.

## Larger train-only expert experiment

New cohort `ca10t3d_large_expert_v14` compares fresh initialization, a fixed trained-policy warm start, and a matched warm-start treatment with GAE lambda one and delivery bonus zero. All three use 20 BC epochs followed by end-to-end PPO. The third treatment is motivated by the earlier matched cost-fine-tuning screen (36.55% versus 40.90% excess); it combines two changes and does not isolate their individual effects. It retains the heterogeneous graph encoder and complete variable-action GCN, nonlinear edge messages, action-to-entity links, gamma one and cost-emphasis rewards. The exact capacity restriction is used consistently in BC and PPO; additional work-escape restrictions remain disabled.

The verified archive contains 183,697 decisions from 4,007 successful episodes among 4,096 training scenarios, with train seeds 600,000–604,095. Whole episodes are assigned to the BC holdout. BC normalization observes the training portion; warm starts also retain previously acquired training statistics. The archive is eight times the earlier teacher bank by episode count. Remasking pruned 3,363 candidate rows and zero expert labels. All 34 model/state/utils files match the anchored benchmark source; the YAML physics is unchanged.

All warm-start seeds use the same frozen seed-zero parent. Their seed variability is conditional on that parent; the fresh-initialization arm varies all initial model weights. Every graph and action-network parameter remains trainable. No solver is invoked to select deployed neural-policy actions.

Initial array **28279916** retains its original six run definitions in three allocations. Supplementary array **28280299** adds three disjoint runs in two allocations, giving nine runs and five allocations in this cohort. CPU job **28280300** waits for both arrays and compares BC-only, selection-best PPO and final PPO checkpoints on the opened 128-case development bank. Coordinator **28280303** waits for that report before the standard validation-gated 4M promotion, which retains at most two recipes. The original pending CPU jobs (28279949 / 28279953) were cancelled with zero runtime when this treatment was registered; no GPU training was restarted. A zero-runtime premature submission (28279715) was cancelled before input setup completed; it produced no training artifacts.

## Larger-data startup and BC-only readout

The following 64-case results are reused checkpoint-selection observations. They are not the independent 128-case development comparison. Operating hours include successful episodes only.

| Run | BC epochs | PPO decisions logged | Initial completion | Initial operating hours |
| --- | --- | --- | --- | --- |
| large_fresh_cost40_s0 | 20/20 | 358,400 | 93.75% | 126.40 |
| large_warm_cost40_s0 | 20/20 | 358,400 | 95.31% | 120.87 |
| large_fresh_cost40_s1 | 20/20 | 358,400 | 93.75% | 124.66 |
| large_warm_cost40_s1 | 20/20 | 368,640 | 96.88% | 122.19 |
| large_fresh_cost40_s2 | 20/20 | 286,720 | 96.88% | 125.03 |
| large_warm_cost40_s2 | 20/20 | 307,200 | 95.31% | 120.04 |
| large_warm_mc_cost40_s0 | 20/20 | 348,160 | 96.88% | 121.66 |
| large_warm_mc_cost40_s1 | 20/20 | 348,160 | 98.44% | 124.29 |
| large_warm_mc_cost40_s2 | 20/20 | 563,200 | 96.88% | 124.13 |

Early evaluation monitor: 28280824; state: `completed`. It uses one CPU while awaiting the nine immutable BC-only policies, then submits one packed CPU evaluation on 128 opened development cases. The 1M comparison reuses those results only after checking policy weights, configuration, source, masks and scenario coverage; otherwise it evaluates BC-only policies itself.

Early BC-only scoring job: **28281262**.

| BC-only recipe | Mean completion | Mean successful operating hours | Mean paired BKS excess |
| --- | --- | --- | --- |
| bc0_large_fresh_cost40 | 92.97% | 141.29 | 28.00% |
| bc0_large_warm_cost40 | 90.62% | 136.21 | 25.56% |
| bc0_large_warm_mc_cost40 | 91.67% | 139.41 | 27.29% |

These are BC-only policies, before any PPO update. Warm starts share one pretrained parent; seed variability is conditional on that parent. Final validation and test remain unopened.

## Completed reference-mask availability audit

CPU audit **28280487** completed 256 new independent solver executions on the same opened 128 development scenarios. ALNS and CP-SAT each completed 126/128 cases, with successful operating-hour means of 111.99 and 118.68 respectively. Every initial physical descriptor matched the cached cases. None of the three queried masks (completion only, relaxed capacity, exact capacity) rejected any of the **11,920 actions in 252 successful reference routes**. The references executed their ordinary actions without mask repair.

The combined new reference pool contains a mask-admissible successful route on all 128 cases. Selecting the cheapest successful reference separately for each case averages 0.35% excess over the cached six-reference BKS (paired bootstrap interval [-0.61%, 1.29%]). This is oracle reference-route selection and establishes observed route availability, not a deployed policy result or an optimality certificate. It supports concentrating the next experiments on learning those efficient choices. The original BKS pool was not revised. Artifacts are in `results/california_graphppo/reference_mask_audit_20261019/`.

## Final evaluation protocol

All seven registered cohorts share one final selection; retention is explicitly screened out, leaving six active cohorts and a maximum of 30 final-stage training allocations. The evaluator supports all trained mask flags and normalizes only four declared policy restrictions; physical/configuration and source hashes are checked. The disjoint 512-case final validation bank and locked 1,000-case test remain unopened. The test requires both target metrics and complete independent feasible-reference coverage on validation.

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
