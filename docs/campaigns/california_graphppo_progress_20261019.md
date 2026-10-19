# California GraphPPO progress and available performance

Updated 2026-10-19T21:26:31.412083+02:00 (Europe/Berlin).

Goal remains active and unmet: at least 99% successful depot-return completion and at most 5% mean paired operating-hours excess, with five training seeds and protected final evaluation.

## Completed performance comparisons

All rows use the same opened 128-case development bank. Neural recipes have three evaluated training seeds. Positive sampling temperatures have three policy RNG repetitions per trained actor; those repetitions are not extra independent training seeds. Training histories differ. Warm starts share one fixed pretrained parent.

Operating hours and paired excess exclude failed episodes; completion retains failures. Excess is the mean of scenario-wise cost ratios, not a ratio of overall means. The per-case BKS pool is an oracle reference with full case coverage, not one deployed policy. Different successful-case subsets can make a reference policy's conditional mean hours lower than the BKS mean over all cases.

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

## Main findings

- The strongest selected warm-cost recipe remains at 98.44% completion, 131.12 successful hours and 18.14% paired BKS excess. Its 95% bootstrap interval for excess is 15.69–20.45%; the current evidence does not support the 5% target.
- End-to-end PPO improves the warm BC-only checkpoint from 90.63% completion and 25.56% excess to 98.44% and 18.14%. All encoder and action-network parameters remain trainable. The common parent already had 3,174,400 selected PPO steps before the larger-data BC/PPO stage.
- Travel accounts for 17.37 of the 18.14 excess percentage points, compared with 0.75 from charging. On successful paired cases the policy uses 5.65 trucks versus 4.26 for the best references. This supports prioritizing assignment and route efficiency.
- The sampled-decoding comparison completed. Temperatures 0.25 and 0.5 reached 98.78% completion but worsened excess to 18.90% and 18.93%. All 384 greedy replay controls matched exactly. Sampling is not promoted as a solution.
- Existing completion/capacity masks rejected none of 11,920 actions from 252 successful independent reference routes. This establishes observed route availability, not a proof that masks never cause problems. Stronger risk masking failed its reference-coverage gate and was not promoted.
- The action-edge/critic-gradient variants passed training-state and 4,096-step real PPO audits. They do not yet have a completed performance comparison. All 4,096-case recovery data archives passed integrity checks.

## Live campaign and next actions

- Latest checked queue: 9 running and 5 pending training allocations, below the limit of 30. Capacity and objective maturity screens wait for completed first-three-seed 12M runs.
- All ten original runs completed 12M. Pending 32-CPU metadata jobs were replaced, without restarting training, by completion job 28318314 and recovery bootstrap 28318317, each using one CPU and 8 GiB.
- Continue the larger-data warm recipes to 4M; execute the action-edge packing profiles, then the nine matched action-edge runs. Compare standard action GCN, routing-edge gates, and gates with value-loss encoder detachment.
- Launch the nine recovery-data runs once original completion metadata is recorded. Compare nominal-teacher, 80%-teacher and 50%-teacher datasets with matched parent and optimizer settings.
- Score BC0/selected/final actors at 1M and frozen selected/final actors at 4M. Advance at most two recipes to five training seeds and 12M; use conservative opened-development screens to release dominated cohorts.
- Only after the shared training barrier closes: 512 disjoint final-validation cases with complete independent reference coverage, followed by the locked 1,000-case test if both targets pass. Report BKS excess, without a certified-optimality claim.

Sources: `ca10t3d_large_expert_v14/development_1m_summary.json`, `matured_controls_20261019_1735/summary.json`, and `policy_temperature_20261019/summary.json` under `results/california_graphppo/`.
