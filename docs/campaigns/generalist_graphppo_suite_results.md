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

### Generalisation to unseen problems (claim 2, 2026-11-01)

Pre-registered in `results/generalist_graphppo/suite_v4_heldout/PREREGISTRATION.md` before any held-out result existed: the frozen passing weights (suite_v4 `best.pt`, mask at margin 0 with the energy-rule fallback, guard, plan on, no tuning) on eight held-out problems, 128 validation-namespace seeds each, against a BKS built per problem with the suite's portfolio plus `lh_rolling_served`, judged by the unchanged gate. Table: `results/generalist_graphppo/suite_v4_heldout/evaluation/validation/heldout/near_optimal_gate.md`.

**Claim 2 does not hold under the pre-registered rule, for any kind.** No run passes all eight (arm distance 0.090). What passes and what fails is the same for all three runs:

| Kind | Problem | s0 / s1 / s2 |
|---|---|---|
| Unseen network | `de_nl_be_longhaul` | PASS (hours +0.8 / +0.5 / +0.3 %, success 100 %) |
| Unseen network | `de_nl_be_longhaul_hos` | FAIL (success 88 / 83 / 86 % vs 98 needed; hours +7.5 / +4.9 / +6.4 %) |
| Unseen size | `tx_longhaul_t5c15` | PASS (+0.7 / +0.4 / +0.3 %) |
| Unseen size | `tx_longhaul_t20c60` | PASS (+3.0 / +4.5 / +1.3 %) |
| Unseen size | `ca_sameday_t5` | FAIL (services 0.825 / 0.837 / 0.862 of BKS; s1, s2 failure 1.6 / 3.9 %) |
| Unseen size | `ca_sameday_t20` | PASS (0.975 / 0.984 / 0.962) |
| Unseen combination | `tx_sameday` | FAIL (0.840 / 0.857 / 0.809) |
| Unseen combination | `pnw_drayage_combined` | FAIL (0.603 / 0.576 / 0.641) |

Reading: on long-haul problems without hours-of-service the policy with its planner transfers to a new network and to fleets 2.5× smaller and 2× larger within the gate's hours margin, which is consistent with the static pass being the policy following the planner. Hours-of-service on an unseen network breaks it (one truck in eight fails to finish on time). On the dynamic side it transfers to a larger same-day fleet but not to a smaller one, and both unseen combinations fail, drayage badly: the drayage gain seen on the suite problem does not carry to Seattle–Tacoma, where the policy serves 60 % of the BKS. No held-out problem was used for any decision, so these numbers are a clean read of the frozen policy.

### PPO on the deployed problem (suite_v5, pre-registered, 2026-11-01)

Question (pre-registered 2026-10-31 in `suite_v4_heldout/PREREGISTRATION.md`, amended before any result): does PPO noticeably improve the passing policy when it trains on the deployed problem (both filters on, plan on, `--public-wake`, per-problem advantages, learning rate 1e-4, 2M steps, warm start from each seed's suite_v4 weights)? Two arms: `ppo_deployed` (pure PPO) and `ppo_deployed_anchor` (imitation held at bc 0.05, DAgger β 0.3). Checkpoints chosen on the 64 selection seeds by gate distance (score breaking ties), then read on the 128 validation seeds against the same seed's starting weights (`4M_mask0_fix_guard_publicwake`); "noticeable" = no gate criterion turns pass → fail, and a gain of ≥ 1 point of static hours excess or success, or ≥ 0.02 same-day services ratio, on a long-haul or same-day problem, with a paired 95 % bootstrap interval excluding zero (drayage gains are reported but do not qualify alone).

| Run | Chosen checkpoint | Gate on 128 seeds | Qualifying gain | Losses | Verdict |
|---|---|---|---|---|---|
| pure PPO s0 / s1 / s2 | step 0 (= the start weights, bit-for-bit) | as the start | none: every later checkpoint is farther from the gate on the selection seeds (static hours +4…+29 %, Northwest success 0.73–0.94) | — | **no improvement** |
| anchored s0 | 1.54M | PASS all five | Northwest hours excess −2.75 pts [−4.04, −1.48]; same-day +0.01; drayage +0.17 | congested hours +1.20 pts [0.43, 2.03] | **noticeable** (also on seeds 64–127) |
| anchored s1 | 1.92M | PASS all five (its start failed congested and Northwest) | Northwest hours −3.29 pts [−5.44, −1.34]; drayage +0.05 | Texas hours +0.96 pts; on seeds 64–127 same-day turns PASS → FAIL (failure 1.6 %, services −0.02) | **noticeable** on the primary read, regression on the second half |
| anchored s2 | 1.02M | FAIL: Northwest success 96.9 % (124/128, limit 98 %) | same-day +0.02, drayage +0.13 | Northwest success −2.34 pts, Texas and congested hours +1.9 pts each | **not noticeable** (gate regression) |

Reading: pure PPO at this batch only degrades the policy, which matches the gradient probe (per-problem PPO gradients are sampling noise at 128–1 000 decisions per problem; `agent/pilots/gradient_conflict/CONCLUSIONS.md`). With the imitation anchor PPO moves hours between problems — Northwest better by 2–3 points on two seeds, Texas and congested worse by 1–2 points on all three — and lifts drayage, and whether the net is a pass depends on the seed. By the pre-registered rule the anchored arm shows a noticeable improvement on 2 of 3 seeds; the claim that PPO reliably improves the deployed policy is not supported. Variance-reduction arms (bigger batches, a seed-paired return baseline, family critic heads) and an imitation-only control are training as `suite_v5b` / `suite_v5c` (pre-registered in `suite_v5b/PREREGISTRATION.md`) to attribute the gains to PPO or to the anchor.

### Variance reduction and the imitation-only control (suite_v5b / suite_v5c, pre-registered, verdicts 2026-11-06/07)

Question (`suite_v5b/PREREGISTRATION.md`, written before any run): which of the anchored arm's gains are PPO's rather than the anchor's, and does reducing the gradient's sampling noise make them reliable? Four arms, all warm-started as the anchored arm and trained for 2M steps: family critic heads (`ppo_deployed_anchor_critics`, one value head per mechanism family), big batches (`ppo_deployed_anchor_bigbatch`, 512-step rollouts and 1 024-decision gradient steps instead of 128 / 256, cadences divided by four), a seed-paired group baseline on top of the big batch (`ppo_deployed_anchor_group`), and the control `imitation_only_deployed` (the same recipe with the PPO actor term weighted 0). Checkpoints chosen as in suite_v5, read on all 128 validation seeds against each seed's starting weights, and PPO's share read as the arm minus the control, paired per seed. The rule: a change enters suite_v6 if its arm shows a noticeable improvement attributable to PPO on at least 2 of 3 seeds.

| Arm | Seed | Chosen checkpoint | Gate (128 seeds) | Qualifying gain vs start, and vs control | Losses | Verdict |
|---|---|---|---|---|---|---|
| imitation-only control | s0, s2 | step 0 (= the start weights) | as the start | none (every later checkpoint is farther from the gate) | — | the anchor alone does not improve |
| imitation-only control | s1 | update 175 | PASS (its start failed congested and Northwest) | none | drayage −0.41 | the anchor alone moves s1 through the gate at a large drayage cost |
| family critics | s0 | 250 | PASS | Northwest hours −1.55 pts [−2.76, −0.39] (control: same) | Texas +1.31, congested +1.13 pts | **noticeable, PPO's** |
| family critics | s1 | 375 | PASS (start failed congested and Northwest) | Northwest −3.49 vs start; −2.67 [−3.72, −1.59] vs control | Texas +2.00 pts; seeds 64–127 add one same-day failure | **noticeable, PPO's** |
| family critics | s2 | 75 | PASS | same-day services +0.021 [0.010, 0.033] (control: same) | congested +2.29, Texas +0.40 pts | **noticeable, PPO's** |
| big batch | s0 | 66 | PASS | Northwest −2.45 pts [−3.67, −1.23] (control: same) | congested +1.84 pts; on seeds 64–127 Northwest success turns PASS → FAIL | **noticeable, PPO's** |
| big batch | s1 | 96 | PASS (start failed congested and Northwest) | Northwest −3.62 vs start; −2.87 [−4.02, −1.69] vs control | Texas +0.71 pts | **noticeable, PPO's** (also on seeds 64–127) |
| big batch | s2 | 97 | PASS | same-day +0.03 [0.02, 0.04] and Northwest −1.84 [−3.01, −0.66] (control: same) | congested +1.12, Texas +0.74 pts | **noticeable, PPO's** (also on seeds 64–127) |
| group baseline | s0 | 54 | FAIL: Northwest success 97.7 % | Northwest hours −2.67 | Northwest success −1.56 pts | not noticeable (gate regression) |
| group baseline | s1 | 96 | PASS | Northwest −3.01 vs start; −2.42 [−3.39, −1.48] vs control | none beyond noise | noticeable, PPO's |
| group baseline | s2 | 36 | FAIL: Northwest success 96.1 % | same-day +0.02 | Northwest success −3.12 pts | not noticeable (gate regression) |

Drayage services rise on every PPO run (+0.05 to +0.19 vs start) and are reported, not counted. **Verdicts by the pre-registered rule: the family critics qualify on 3 of 3 seeds, the big batch on 3 of 3, the group baseline on 1 of 3. The first two enter the suite_v6 recipe; the group baseline does not.**

Reading: the imitation-only control settles the attribution question. On two seeds its best checkpoint is the start itself, so the anchor alone changes nothing and every gain of the PPO arms is PPO's; on seed 1 the anchor alone does move the policy through the gate, but at −41 % drayage services, and the PPO arms still beat it by 2.4–2.9 points of Northwest hours. PPO's signature is the same in every arm: Northwest long-haul hours better by 1.5–3.6 points, drayage services up, Texas and congested hours worse by 0.4–2.3 points. Neither change removes the Texas cost; what the critics and the big batch buy is that all three seeds now pass the gate after training (the anchored arm alone passed on 2 of 3), i.e. a more reliable trade rather than a larger one. The group baseline (common random numbers across four env slots) cost Northwest success on two seeds. The checkpoint selector hard-coded 5 120 decisions per update and would have mis-numbered the 20 480-decision big-batch runs; it now reads the value from each run's configuration (`agent/pilots/suite_v5_select/select.py`). Per-seed outputs: `agent/pilots/suite_v5_select/improvement_suite_v5{b,c}_*.json`, `ppo_share_suite_v5{b,c}.json`.

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
