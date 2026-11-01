# Generalist GraphPPO suite: results

**Verdict: the campaign's gate is met on the locked test.** Two of the three trained runs of one arm, s0 and s2, pass the near-optimal gate on all five problems on 500 unseen test seeds, one set of weights each. The third run, s1, fails on congested and Northwest success.

The passing system is not GraphPPO alone. It is a trained policy that reads the long-haul planner's plan at every decision, with two rule-based filters on its actions. Section 3 states what the result does and does not claim.

- Arm: `full_dagger_plan_charge` (`results/generalist_graphppo/suite_v4/`), `best.pt`, chosen during training at 0.768M steps (s0, s2) and 1.536M steps (s1).
- Decision-time settings: feasibility mask at margin 0 with the energy-rule fallback, and the retirement guard (`evaluation/test/final/decision.json`).
- Gate: `scripts/campaigns/generalist_near_optimal_gate.py`, unchanged; record in `evaluation/test/final/near_optimal_gate.md`.
- The verdict was recomputed from the raw rows separately from the gate script and matches.

## Post-test audit (2026-10-31, in progress)

The result below is reported as the gate computed it. Two independent audits ran afterwards; some checks are still running, and this section will be replaced by their outcome.

- **The computation holds.** Same scenarios, same metrics, no dropped or mixed rows; a separate recomputation gives the same verdicts. Rerunning validation at a different evaluation batch size changed 0 of 1,920 episodes.
- **A narrow information leak exists on the dynamic problems.** A waiting truck's wake-up time is computed from the hidden request stream and reaches the policy's features and the rolling planner: at most one bit, at most 0.5 h ahead, usually none. Static problems are clean (746 scrambled decisions, 0 differences). With the wake-up replaced by its public value, the evaluated weights changed 0 actions on all 112 validation seeds outside checkpoint selection, for s0 and s2 (95 % upper bound 3.2 % of seeds); the rolling planner changed on none of the 128 same-day and drayage seeds. So the leak looks harmless to this result, but the test seeds were not rerun. An earlier 13-seed pilot showing a lost service was an artefact of a flawed formula and is retracted. A full leak-closed validation rerun is in progress.
- **`lh_rolling` never re-planned.** It equals `lh_plan` on 500/500 test seeds because of a defect, so the BKS has one fewer distinct method than documented. Adding the corrected method (`lh_rolling_served`) to the portfolio tightens best-known hours by 0.06 / 0.26 / 0.71 % on tx / congested / pnw. s0 and s2 still pass the static gate on the test seeds under this stricter comparator: s0 +1.06 / +1.77 / +4.30 %, s2 +0.44 / +1.71 / +4.19 %. On validation s2 passes and s0 misses pnw by 0.07 points (+5.07 %). The recorded gate result stays as computed; this is a sensitivity check. The policy's plan input still uses the defective trigger, since it was trained on it.
- **"Near-optimal" means near best-known.** The static lower bounds leave 30-88 % gaps and were closed on only 2-8 of 500 seeds, although `summary_test.json` marks them proven. True optimality is unknown on every problem.
- **The static pass mainly shows the policy following the planner it is compared with.** s2 reproduces `lh_plan`'s episode exactly on 483/500 Texas and 455/500 congested seeds, and `lh_plan` alone passes all three static problems.
- **The pass is seed-sensitive.** Resampling the seeds, the probability of passing all five is about 0.55 for s0 and 0.73 for s2.
- **The drayage gain is genuine within the simulator's rules, but it is narrower than the ratio suggests.** 80 reproduced episodes showed no rule violations (11 injected faults were all caught). The mechanism is pairing exports with imports for less empty driving. About half of the ratio to the BKS comes from another method's late deliveries; against `rolling_request` alone the policy completes 27-29 % more jobs. At 30-minute terminal turns, on 8 seeds, its lead over `rolling_request` falls from 1.27 to 1.01; it never trained on those turns.

## 1. The test result

Gate limits, against the BKS on the same seeds: static problems need failures ≤ 1 %, success ≥ 98 % and mean paired hours excess ≤ 5 %; dynamic problems need failures ≤ 1 % and on-time services ≥ 95 % of the BKS. BKS on the 500 test seeds: 0 failures on all five problems, 500/500 success on the three static ones.

Δhours vs teacher is paired on seeds where neither episode failed. Served fraction is the mean of each episode's served share of its tasks. Success is not a gate criterion on the dynamic problems (BKS success: 0.368 same-day, 0 drayage), where most episodes end with trucks still out at the horizon.

### Run s2 (seed 2 of the arm): PASS

| Problem | Served fraction | Success [95 % Wilson] | Failed episodes | Mean hours | Δhours vs teacher [95 %] | vs BKS [95 %] | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Texas long-haul | 1.000 | 1.000 [0.992, 1.000] | 0/500 | 117.6 | -228.6 [-234.1, -223.1] | +0.4 % [+0.3, +0.5], n=500 | 3.65 |
| Texas congested | 0.999 | 0.984 [0.969, 0.992] | 0/500 | 125.8 | -294.7 [-302.8, -286.4] | +1.4 % [+1.2, +1.7], n=492 | 3.67 |
| Northwest (driver rest rules) | 1.000 | 0.986 [0.971, 0.993] | 2/500 | 159.4 | -296.9 [-306.1, -287.7] | +3.4 % [+2.7, +4.1], n=493 | 3.61 |
| California same-day | 0.910 | 0.200 [0.167, 0.237] | 1/500 | 263.6 | -17.4 [-19.4, -15.4] | services 0.960× BKS | 2.33 |
| California drayage | 0.576 | 0.000 [0.000, 0.008] | 0/500 | 320.1 | -39.9 [-41.5, -38.3] | services 1.233× BKS | 3.63 |

### Run s0 (seed 0 of the arm): PASS

| Problem | Served fraction | Success [95 % Wilson] | Failed episodes | Mean hours | Δhours vs teacher [95 %] | vs BKS [95 %] | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Texas long-haul | 1.000 | 0.996 [0.986, 0.999] | 2/500 | 118.4 | -227.7 [-233.1, -222.2] | +1.0 % [+0.8, +1.3], n=498 | 3.64 |
| Texas congested | 0.999 | 0.988 [0.974, 0.994] | 0/500 | 125.9 | -294.6 [-303.3, -286.5] | +1.5 % [+1.2, +1.8], n=494 | 3.67 |
| Northwest (driver rest rules) | 1.000 | 0.980 [0.964, 0.989] | 0/500 | 159.9 | -296.6 [-305.7, -287.7] | +3.5 % [+3.0, +4.1], n=490 | 3.61 |
| California same-day | 0.927 | 0.206 [0.173, 0.244] | 1/500 | 266.3 | -14.2 [-16.1, -12.2] | services 0.962× BKS | 2.37 |
| California drayage | 0.587 | 0.000 [0.000, 0.008] | 0/500 | 324.8 | -35.2 [-36.8, -33.8] | services 1.256× BKS | 3.77 |

### Run s1 (seed 1 of the arm): FAIL

| Problem | Served fraction | Success [95 % Wilson] | Failed episodes | Mean hours | Δhours vs teacher [95 %] | vs BKS [95 %] | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Texas long-haul | 1.000 | 0.998 [0.989, 1.000] | 0/500 | 118.3 | -228.0 [-233.4, -222.3] | +0.8 % [+0.6, +1.0], n=499 | 3.65 |
| Texas congested | 0.999 | 0.978 [0.961, 0.988] | 3/500 | 127.9 | -291.9 [-300.1, -283.6] | +2.8 % [+2.3, +3.3], n=489 | 3.65 |
| Northwest (driver rest rules) | 0.996 | 0.942 [0.918, 0.959] | 4/500 | 162.0 | -294.7 [-303.6, -285.6] | +4.0 % [+3.2, +4.9], n=471 | 3.54 |
| California same-day | 0.944 | 0.272 [0.235, 0.313] | 0/500 | 265.1 | -14.3 [-16.1, -12.4] | services 0.977× BKS | 2.43 |
| California drayage | 0.640 | 0.000 [0.000, 0.008] | 0/500 | 326.8 | -33.2 [-34.6, -32.0] | services 1.371× BKS | 4.61 |

**Thin margins.**

- s0 Northwest success is exactly at the limit: 490 of 500, where 490 is the minimum that passes.
- Same-day services are 0.962 (s0) and 0.960 (s2), about 0.01 above the limit.
- Hours are comfortable: the upper end of the 95 % interval is at most +4.1 % for s0 and s2.

## 2. Decision latency

CPU, 2 threads, one episode at a time, s2, validation seeds +32 and +64, plan read from the cache, on a shared login node (`agent/pilots/latency/`). The cycle includes the simulator step and the planner's per-decision solve, so these numbers bound the policy's cost from above.

| Problem | Network only | Full decision cycle |
| --- | --- | --- |
| Texas long-haul | 18 ms | 81 ms |
| Texas congested | 15 ms | 70 ms |
| Northwest (driver rest rules) | 17 ms | 164 ms |
| California same-day | 9 ms | 16 ms |
| California drayage | 11 ms | 143 ms |

Static episodes also need the time-0 plan before they start: about 37–60 core-seconds per seed (500 seeds took 17–28 minutes on 18 cores).

## 3. What this result does and does not claim

**It claims** that a single trained set of weights, with the planner's plan as input and the two filters at decision time, is near-optimal by the campaign's gate on 500 unseen seeds of all five problems.

**It does not claim** that a learned policy replaces the planner.

- **The plan is essential.** With the plan withheld on validation, the same weights fall to distance 1.205, and every problem fails. The planner has to run alongside the policy.
- **On the static problems the planner alone is at least as good.** `lh_plan` alone is at +0.1 to +3.5 % on validation with no failures. The policy tracks it; it does not beat it.
- **The PPO phase was needed for this pass.** With the same filters on the 128 validation seeds, the weights from before PPO pass 0 of 3 runs (distance 0.017, against 0.000 after). For the passing runs s0 and s2, it raised drayage services by 32 % and 38 %, cut congested hours by 4.1 and 1.3 points of the BKS, and raised Northwest success from 113 to 127 and from 122 to 127 of 128; it also cut Northwest hours by 4.7 points on s0. It lowered same-day services by 0.26 and 0.64 per episode. s1 gained only on drayage. Imitation terms stayed active in this phase, so it is not pure PPO. After them, later training scored lower on the selection seeds, and why is unresolved.
- **The filters matter.** Without the guard the same weights do not pass (validation distance 0.007, Northwest fails). Without the mask at all, the distance is 0.033. On all 128 validation seeds of s0 and s2, the guard changed the policy's own pick at 260 of 85,310 static decisions (0.30 %), in 198 of 768 episodes; for s2 that is 0 on Texas, 1 on congested and 79 on Northwest. The depot option was hidden at 19.7 % of decisions, but mostly the policy would not have chosen it. The test rows do not record this.
- **Seed sensitivity.** Two of three runs pass. s1 passes neither on validation nor on test. The arm's training does not reliably give a passing policy, and the test reports all three.
- **Drayage is not evidence of near-optimality.** The policy serves 1.23–1.37× the BKS and more than the "hindsight" value, which is an unproven heuristic, so the portfolio there is weak rather than the policy shown optimal.

## 4. Integrity of the test

- `agent/TEST_OPENED` was written on 2026-10-31 with the user's approval, after the user was told the validation pass depends on the guard.
- The guard, the mask fallback and every other design choice were made on validation; test seeds were first touched after that file. The evaluation's frozen code is identical to the validation tag `4M_mask0_fix_guard`.
- One test evaluation (`final`, job 28787745), of every seed of the passing arm. No design change followed.
- All three static BKS jobs stopped at their final merge step because the merge also read `plans__test.jsonl`. They were re-merged with `--methods` and the same five-method portfolio; no row changed.
- The fixes came from failures on validation seeds, so the validation pass was optimistic. The test results are the ones to cite, and they held: s0 and s2 passed both.

## 5. How the campaign got here

Distance to the gate on the 128 validation seeds; 0 passes. Best arm per stage.

| Stage | What changed | Distance |
| --- | --- | --- |
| P2, 2M | seven arms | 0.512 |
| P3, 8M | three arms promoted | 0.237 |
| suite_v2 | planner's route order as input | 0.158 |
| + feasibility mask | decision-time filter | 0.151 |
| suite_v4 + mask | planner's charging moves as input | 0.015 |
| + mask fallback fix | energy rule kept when the time rule empties the safe set | 0.007 |
| + retirement guard | no retirement while the plan still routes customers to the truck | 0.000 (s0, s2 pass) |

Decision-time search (best-of-16, lookahead) made results worse and was dropped. suite_v3 (deeper network, imitation held on) reached 0.056 and did not pass.

## 6. Where to find everything

- Test rows and gate: `results/generalist_graphppo/suite_v4/evaluation/test/final/`
- Validation record: `.../evaluation/validation/4M_mask0_fix_guard/`, and the comparison tags `4M`, `4M_noplan`, `4M_mask0_fix`, `interim_mask0`
- Test BKS: `results/generalist_graphppo/bks/<problem>/test.json`
- Diagnostics: `results/generalist_graphppo/agent/pilots/` (`failure_replays`, `guard_rate`, `latency`, `plan_order`, `plan_adherence`, `nominal_lookahead`)
- Campaign log: `results/generalist_graphppo/agent/status.log`, analysis of 2026-10-29: `results/generalist_graphppo/agent/analysis_2026-10-29.md`
