# California 10T3D GraphPPO campaign status

Updated 2026-10-18 around 22:40 Amsterdam time. Earlier observations below are historical.

The campaign is **running**. Its target is: at least 99% successful completion and at most 5% mean paired excess operating hours to independently built best-known feasible references, evaluated on 1,000 locked scenarios with five training seeds. The goal has not been achieved or verified yet.

## Running experiments

| Cohort | Training allocation | Runs / A100 jobs | Current target |
| --- | --- | --- | --- |
| Original finalists | 28265867 | 10 / 5 running | 12M decisions; five training seeds per recipe |
| Progress-feature finalists | 28266976 | 10 / 5 running | 12M decisions; five training seeds per recipe |
| Imitation-assisted finalists | 28267420 | 10 / 5 running | 12M decisions; five training seeds per recipe |
| Capacity mask / nonlinear messages | 28267125 | 6 / 3 running | 4M decisions per run |
| Undiscounted / cost-emphasis objectives | 28267175 | 6 / 3 running | 4M decisions per run |
| Retained expert loss, with matched PPO control | 28267391 | 9 / 5 running | BC then 1M decisions; top two recipes eligible for promotion |
| Cost-focused warm start, with matched control | 28267725 | 6 / 3 running | Additional 1M decisions from frozen trained parents; exploratory screening |

The goal tracker is active. There are 29 training allocations in this snapshot.
Six cohorts with at most two five-seed finalists each require at most 30 packed
training allocations. The retention cohort uses the same graph encoder,
complete variable-action GCN, simulator, mask and train-only demonstration bank
as the progress/imitation formulation. Its treatment is a separate expert
cross-entropy term during PPO, with initial weight 0, 0.1 or 1 and linear decay
to zero at 12M decisions. It is an ablation, not a demonstrated improvement.

The three extra warm-start screening allocations overlap at most 26 legacy
allocations. Capacity/objective promotions are gated behind this screen, keeping
the current bound at 29; after it ends the six legacy cohorts again need at
most 30. This screening stage has no automatic long-run promotion. It compares
the current delivery reward / GAE settings against delivery bonus zero and GAE
lambda one, with gamma one in both, and retains the same learned GraphPPO
architecture. All six runtime configurations and initial updates were inspected;
their independent quality results are pending.

The exact packing mask comparison completed: 98.44% completion (378/384 pairs)
versus 97.92% without it, with no lost successes and unchanged costs on common
successes. Mean paired excess remains 38.66%, and neither target is met. Its
128 feasible witness routes were preserved. An additional work-escape reserve
rule produced no further completion gains and remains experimental.

## Current independent development-validation evidence

CPU array **28267364** and report **28267365** completed successfully. Thirty
frozen checkpoints (ten variants, three seeds each) were evaluated on the same
128 scenarios at validation offset 2,000 and compared against six cached
independent references. This bank is now reused development validation. The
512-case final validation bank and locked 1,000-case test remain unopened.

| Variant | Mean completion | Mean successful operating hours | Mean paired BKS excess |
| --- | --- | --- | --- |
| Imitation + progress | 97.92% | 152.49 | 38.57% |
| Imitation + progress + links | 95.83% | 160.87 | 45.16% |
| Original shield | 97.40% | 157.54 | 45.60% |
| Revised shield | 96.61% | 159.59 | 46.52% |
| Nonlinear + gamma 1 + cost emphasis | 97.14% | 163.42 | 47.26% |
| Capacity + nonlinear graph messages | 96.88% | 164.13 | 49.98% |
| Progress + links + shield | 97.92% | 169.00 | 51.45% |
| Capacity + links | 93.23% | 161.69 | 51.92% |
| Nonlinear + gamma 1 | 96.35% | 164.67 | 52.40% |
| Original links + shield | 97.14% | 170.21 | 53.61% |

All values average per-seed results; cost excludes failed episodes, which are
retained in success statistics and failure reports. The leading recipe's
hierarchical bootstrap interval for mean excess is [35.49%, 41.72%]. Its pooled
successful cases use 6.77 trucks versus references' 4.26; travel contributes
37.31 of 38.57 percentage points of cost excess. Neither target is met. The
137-hour scores on the reused 64-case checkpoint-selection bank should not be
interpreted as independent performance.

Artifacts: `results/california_graphppo/intermediate_20261018_2200/` contains
frozen weights, input hashes, reference rows, episode results, independent
paired report, training-seed intervals and cost-component breakdown. CPU job
**28267429** replays selected failures with full action traces.

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
