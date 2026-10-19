# California GraphPPO progress and available performance

Updated 2026-10-19T23:31:12.400392+02:00 (Europe/Berlin).

Goal is active and unmet: at least 99% successful depot-return completion and at most 5% mean paired operating-hours excess against independent feasible BKS, with five training seeds and protected final evaluation.

## Completed performance comparisons

Every row below uses the same opened 128-case development bank. Neural recipes have three evaluated training seeds; inherited training budgets differ. Warm starts share one fixed parent. Sampling repetitions are policy RNG repetitions, not extra training seeds. Operating hours and paired excess condition on successful episodes; completion retains every failure. Excess is the mean of per-case ratios. The per-case BKS pool is an oracle reference, not one deployed policy or a certified optimum.

| Approach | Stage | Completion | Successful operating hours | Mean paired BKS excess |
| --- | --- | --- | --- | --- |
| Independent BKS pool (per-case oracle) | Reference coverage, not one policy | 100.00% | 113.20 | 0.00% |
| heuristic | Independent reference | 96.88% | 277.30 | 150.19% |
| mpc | Independent reference | 94.53% | 316.57 | 179.34% |
| alns | Independent reference | 100.00% | 263.17 | 137.98% |
| cpsat | Independent reference | 100.00% | 264.60 | 138.75% |
| alns_planned | Independent reference | 98.44% | 112.11 | 1.73% |
| cpsat_planned | Independent reference | 99.22% | 116.27 | 5.15% |
| bc0_large_fresh_cost40 | 20 BC epochs; zero new PPO steps | 92.97% | 141.29 | 28.00% |
| bc0_large_warm_cost40 | 20 BC epochs; zero new PPO steps | 90.62% | 136.21 | 25.56% |
| bc0_large_warm_mc_cost40 | 20 BC epochs; zero new PPO steps | 91.67% | 139.41 | 27.29% |
| large_fresh_cost40 | Selected checkpoint through 1M additional PPO decisions | 98.44% | 139.62 | 25.52% |
| large_warm_cost40 | Selected checkpoint through 1M additional PPO decisions | 98.44% | 131.12 | 18.14% |
| large_warm_mc_cost40 | Selected checkpoint through 1M additional PPO decisions | 97.92% | 134.60 | 20.28% |
| ppo1m_final_large_fresh_cost40 | Final at 1M additional PPO decisions | 98.18% | 138.77 | 25.40% |
| ppo1m_final_large_warm_cost40 | Final at 1M additional PPO decisions | 98.70% | 132.18 | 19.65% |
| ppo1m_final_large_warm_mc_cost40 | Final at 1M additional PPO decisions | 98.44% | 135.97 | 21.38% |
| bc_links_progress | Selected checkpoint after completed 12M run budgets | 97.92% | 151.23 | 37.06% |
| bc_progress | Selected checkpoint after completed 12M run budgets | 97.66% | 145.94 | 33.91% |
| links_shield | Selected checkpoint after completed 12M run budgets | 96.61% | 155.16 | 39.30% |
| progress_links_shield | Selected checkpoint after completed 12M run budgets | 98.70% | 157.82 | 41.40% |
| shield | Selected checkpoint after completed 12M run budgets | 97.66% | 152.66 | 40.18% |
| v2_shield | Selected checkpoint after completed 12M run budgets | 95.05% | 147.99 | 34.86% |
| sampling_0p25 | Same frozen warm-cost actors; 3 policy RNG repetitions per actor | 98.78% | 132.64 | 18.90% |
| sampling_0p5 | Same frozen warm-cost actors; 3 policy RNG repetitions per actor | 98.78% | 131.95 | 18.93% |
| sampling_1p0 | Same frozen warm-cost actors; 3 policy RNG repetitions per actor | 98.18% | 135.10 | 21.58% |
| bc_nonlinear_cost40 | Selected checkpoint after completed 12M run budgets | 96.88% | 138.77 | 26.39% |
| bc_nonlinear_gamma1 | Selected checkpoint after completed 12M run budgets | 98.96% | 146.55 | 32.76% |
| bc_capacity_links | Selected checkpoint after completed 12M run budgets | 98.18% | 151.88 | 36.90% |
| bc_capacity_nonlinear_links | Selected checkpoint after completed 12M run budgets | 97.92% | 142.09 | 28.97% |

## Main findings

- The leading selected warm-cost actor family remains at 98.44% completion, 131.12 successful operating hours and 18.14% paired BKS excess. Its excess bootstrap 95% interval is 15.69–20.45%; the 5% target is not supported.
- PPO improves its warm BC-only starting policy from 90.63% completion and 25.56% excess. The parent had already received 3,174,400 selected PPO decisions. These are warm-started end-to-end runs; teacher or solver inference is not used when deploying the neural policy.
- Travel contributes 17.37 of the 18.14 excess percentage points; charging contributes 0.75. On successful paired cases the policy uses 5.65 trucks versus 4.26 for the references. Assignment and routing efficiency remain the main measured cost weaknesses.
- Completed sampling variants reached up to 98.78% completion but worsened cost excess. All 384 greedy replay controls matched; sampling is not promoted as a solution.
- Existing completion/capacity masks preserved all 11,920 actions on 252 successful independent reference routes. Stronger risk masking failed its reference-coverage gate. Observed route availability supports improving learned decisions; it does not prove every mask is harmless.
- Actual full-campaign checkpoints confirm PPO changes the graph encoder, action network and value network after BC. Both routing-edge variants also changed their gates; all inspected neural weights were finite. Critic-gradient detachment still permits actor gradients through the encoder. This establishes end-to-end training, not a held-out performance gain.
- The reward printed in training logs is the simulator episode reward. The vector worker separately adds terminal success/failure shaping to the PPO reward. The logged reward therefore omits the configured 24,000 success bonus; source inspection found no missing-bonus defect.

## Current campaign

Scheduler observation: 2026-10-19T23:31:11.752163+02:00. New training uses one model per GPU allocation, nine CPUs and seven rollout workers; current warm/recovery/action-edge families allow six/eight/nine allocations plus four objective continuations within a global bound of 27. Future stages keep the five-allocation default. Four objective models resume on one-model MIG allocations after clean A100 checkpoint exits. One queued recovery pilot packs two models per MIG, with staggered BC. No GraphPPO campaign work is submitted to cbuild. The global training limit remains 30 allocations. The original new-family arrays use MIG; nine unstarted tasks moved to supplemental H100 arrays, and replacement coordinators wait for both arrays. SLURM disallowed live throttle changes and mixed-partition shared jobs; no running model was restarted.

| Family | Training array | Current milestone |
| --- | --- | --- |
| Larger-data warm cost / MC | 28320619 + 28344289 | Three seeds per recipe; continue 1M → 4M |
| Nominal / 80% teacher / 50% teacher recovery data | 28320624 + 28344292 + 28348662 | Three seeds per recipe; BC20 then 1M PPO |
| Standard action GCN / routing gates / gates with critic detachment | 28320681 + 28344295 | Three seeds per recipe; BC20 then 1M PPO |
| Objective controls | 28348611 | Six completed models skipped; four full states resume toward 12M |
| Original and progress cohorts | Completed | All five seeds per recipe reached 12M; await shared selection |
| Capacity controls | 28279539 | Screened out after completed three-seed 12M comparison; partial final seeds retained |

The completed capacity evaluation used two CPUs and zero GPUs in the owned warm-training MIG allocation, step 28320619_0.0. Its checkpoint/terminal-state verification also completed in that allocation without another GPU reservation. Its pending GPU allocation was cancelled at zero runtime. Unexecuted A100 packing profiles were also cancelled; there are no measured packing-profile results. Future continuation arrays skip finished bundles while preserving the full seed manifest and saved rollout-worker layouts.

## Policy drift diagnostic and resource pilot

On 128 fixed teacher-80 pilot training states from 119 scenarios, exact teacher-action loss rose during PPO. These are disjoint pilot training cases, not the full-data BC holdout or held-out performance evidence. Snapshots use seed zero; equivalent actions can exist. The gated variants were compared at the same 460,800 PPO decisions.

| Actor | BC teacher-action loss | Current PPO teacher-action loss | Mean KL from BC to PPO |
| --- | --- | --- | --- |
| Standard action GCN (563,200 PPO decisions) | 1.50 | 7.24 | 5.01 |
| Routing gates (460,800) | 1.41 | 5.61 | 3.31 |
| Routing gates + critic detachment (460,800) | 1.45 | 20.56 | 17.53 |

This flags stronger off-policy drift with critic detachment in one seed. It does not establish a causal mechanism, a route-cost regression, or final model quality. The three-seed BC/selected/final evaluation remains required before promotion.

Live warm PPO memory was 4.58 GiB of proportional host memory and 2,176 MiB of resident GPU memory. Pilot 28348662 packs two unstarted recovery runs in one nine-CPU/60-GiB MIG allocation, retains seven workers per model, and staggers BC until the first model enters PPO below a conservative host-memory guard. It is queued; packing throughput and peak memory have not been measured.

An overlapping two-CPU watcher will score the six seed-zero actors after terminal 1M completion, using the existing 128 cases and cached seed-zero warm anchor. This exploratory preview will be labelled separately; five-seed final evidence and unopened final/test banks remain required.

## Observed training speed

Measured from current training logs at 2026-10-19T23:31:11.752163+02:00; intervals include intervening selection validation, exclude BC, and are short-run observations.

| GPU family | Models in PPO | PPO decisions so far per running model | Decisions/s per model |
| --- | --- | --- | --- |
| Larger-data 4M | 5 | 2,140,160–2,170,880 | 218–225 |
| Recovery 1M | 5 | 829,440–942,080 | 216–224 |
| Action-edge 1M | 5 | 716,800–921,600 | 189–222 |

At roughly 190–240 decisions/s, one additional million decisions takes about 1.2–1.5 hours of PPO wall time per model. Add BC, startup, checkpoint I/O, the stage evaluation and any queue delay; this is an estimate, not a completed-stage runtime guarantee. Larger-data counters include the completed first 1M stage.

## Next steps

1. The capacity screen completed and met the preregistered dominated-cohort stopping rule: completion 98.18% / 97.92%, excess 36.90% / 28.97%. Both remaining allocations exited cleanly. All four partial full states loaded and matched saved counters (9.67M–9.96M decisions); two A100 allocations were released. The objective screen did not meet its stopping rule, so its remaining seeds continue.
2. Complete the matched recovery and action-edge 1M comparisons: evaluate BC-only, selected PPO and final PPO actors using the same six independent reference methods. Judge reliability and cost together.
3. Continue at most two qualifying recipes per new family to 4M. Freeze and evaluate selected and final 4M actors before further updates; then expand promoted recipes to five training seeds and 12M.
4. After all registered cohorts finish or are explicitly screened out, perform shared selection on 512 disjoint final-validation cases. Open the locked 1,000-case test only if both validation targets pass. Neither bank has been opened.

Sources: [comparison CSV](../../results/california_graphppo/performance_comparison_20261019.csv), [live status](california_graphppo_status.md), and scientific summaries referenced in the comparison JSON.
