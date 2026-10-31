# SWARM.md: generalist GraphPPO campaign, near-optimal gate

Goal status: **active and unmet** (reinstated 2026-10-23). This file is the shared goal and coordination record of every agent working on the campaign; the design is `docs/campaigns/generalist_graphppo_suite_campaign.md`, the procedure `docs/campaigns/generalist_graphppo_suite_runbook.md`, and the runbook's "User decisions" table is binding on every agent.

## 1. The goal

One GraphPPO policy, one set of weights, trained on the five suite problems of `EVRoutingEnv/config_files/generalist_suite/` (`ca_sameday`, `ca_drayage_combined`, `tx_longhaul`, `tx_longhaul_congested`, `pnw_longhaul_hos`), that passes `scripts/campaigns/generalist_near_optimal_gate.py` on **every** problem, first on the 128 validation seeds and then once on the locked 500-seed test bank:

| Problems | Criteria (per run, per problem, against the best-known solution of the same seeds) |
| --- | --- |
| Static (`tx_longhaul`, `tx_longhaul_congested`, `pnw_longhaul_hos`) | failure rate ≤ max(1 %, BKS failure); success rate ≥ BKS − 2 points; mean paired operating-hours excess on jointly successful seeds ≤ 5 % |
| Dynamic (`ca_sameday`, `ca_drayage_combined`) | failure rate ≤ max(1 %, BKS failure); on-time services ≥ 95 % of the BKS's, paired per seed |

The goal is **met** when `results/generalist_graphppo/<campaign>/evaluation/test/final/near_optimal_gate.json` holds a run with `"overall": "PASS"` and that run passed the same gate on validation first. Until then it is unmet, and no partial claim is made. "Near-optimal" means within these margins of a best-known solution, not optimality: the BKS is a portfolio of planners (Section 1 of the design doc), the static bound is nominal and the dynamic bound a clairvoyant heuristic floored at the BKS.

## 2. Binding decisions (user, 2026-10-22; details in the runbook)

- Cluster Snellius, account `tdsei17344`, `gpu_a100` (overflow `gpu_h100`, `gpu_mig`), at most 20 concurrent allocations; CPU stages hold one GPU each.
- Arms: `base`, `fleet`, `paired`, `full`, `full_bks`, `full_bks_retained`, `full_dagger` × 3 seeds at BC + 2M; promote the 3 arms closest to the gate (mean shortfall; amended 2026-10-24 when no arm met the teacher-relative eligibility) to 8M; if no run passes at 8M, one `suite_v2` iteration with one change per arm, then stop and write up.
- Scope: the five problems only; leave-one-problem-out, specialists and size probes are out.
- BKS: static portfolio with nominal bounds; dynamic portfolio with the rolling request planner, bound = max(clairvoyant planner at the raised budget, BKS); the planner never rests on top of a known request.
- DAgger overhead (pilot, 2026-10-23): rounds take about 38 % of `full_dagger`'s wall time; the user kept the arm as decided (2M in about 7 h, 8M in about 25 h with resumes).
- Authority: agents submit P0–P3 without asking and open P5 automatically on a validation PASS; agents **never commit**; changes to arms, thresholds, scope or authority need the user.

## 3. Swarm roles

Any agent session may take a role; it announces the claim and every step with one line in `results/generalist_graphppo/agent/status.log` (`<ISO time> <role>: <what>`). Two agents never work the same stage at once; a claim older than a day with no further lines may be taken over.

| Role | Does | Evidence it leaves |
| --- | --- | --- |
| Coordinator | Runs the stage board below, applies the gates, promotes, opens P5 (writes `agent/TEST_OPENED`), writes `docs/campaigns/generalist_graphppo_suite_results.md` | status lines, `promotion.json`, `near_optimal_gate.{json,md}` |
| BKS builder | P0a plan files and `generalist_bks.sbatch` jobs, `merge`, `demos`, `check`; the P5 test-split BKS | `results/generalist_graphppo/bks/<p>/{validation,test}.json`, `summary*.json`, `train_demonstrations.json`, `hindsight_check.json`, `bks/logs/` |
| Reference builder | P0b `prepare`, `submit --stage references`, `merge-references` | `<campaign>/references/summary.json` |
| Trainer | `submit --stage train` and resubmissions after USR1 time-outs, pilot measurements | `<campaign>/logs/<run>_<budget>.log`, `training/<run>/*_history.json` |
| Evaluator | `submit --stage evaluate`, `report --promote`, the gate per tag | `evaluation/<split>/<tag>/` |

## 4. Shared state and formats

- `results/generalist_graphppo/agent/status.log`: append-only, the swarm's memory. `agent/TEST_OPENED`: written once, by the coordinator, when P5 opens.
- `results/generalist_graphppo/bks/<problem>/<split>.json`: `{"<seed>": {"best": {method, success, failed, dynamic, tasks, served, late, lost, unserved, hours}, "methods": {<method>: {...}}, "bound": {"kind", "value", "proven"[, "clairvoyant", "raised_to_bks"]}}}` with `kind` ∈ {`nominal_lower_bound_hours`, `clairvoyant_rolling_served`, `none`}; `summary.json` (validation) or `summary_<split>.json`; `train_demonstrations.json`: replay-verified rows `{problem, seed, policy, actions, ...episode fields}` on train seeds only.
- Campaign directory (`results/generalist_graphppo/suite_v1`): immutable `manifest.json` (arms, flags, seeds, SLURM block, the frozen `bks/` copy), `references/`, `training/<run>/`, `evaluation/<split>/baselines/<problem>__<policy>.json`, `evaluation/<split>/<tag>/<run>.json` (episode rows with `score`), `promotion.json`, `near_optimal_gate.{json,md}`, `logs/submissions.jsonl`.
- Test seeds never enter training, references or the BKS before P5; the builders and the gate refuse the test split without `--allow-test`.

## 5. Stage board

| Stage | State | Evidence |
| --- | --- | --- |
| Pre-flight code and tests | done 2026-10-22 | 776 unit tests, 2 integration tests, mini check end to end |
| P0a best-known solutions (10 jobs) | **done** 2026-10-23 (jobs 28476569–28476581, 2–63 min each); gates passed: 128 seeds and 256 demonstrations per problem, BKS beats the teacher everywhere | `bks/<p>/summary.json`, design doc Section 1 |
| P0b references (3 jobs) | **done** 2026-10-23 (array 28476582); 45 shards merged, ≥ 1,348 non-failed train episodes per problem | `suite_v1_refs/references/summary.json` |
| P1 GPU pilot (`full_s0`, `full_dagger_s0`, 200k) | **done** 2026-10-23: 175 / 171 decisions/s, 7.8 / 10.9 GB, DAgger rounds 38 % of wall (kept), BC misses 0.77 % on BKS demos at budget 32 → common charger budget raised to 64 (Texas 0.32 % at 64), resume verified (job 28509592) | `suite_v1_pilot/logs/`, design doc Section 4 |
| P2 screen (21 runs, 2M) | training since 2026-10-23 19:40 in `suite_v1` (charger budget 64; references and BKS frozen in); **done** 2026-10-24: all 21 runs evaluated on 128 seeds; no run passes; by distance to the gate full_bks 0.51, full_dagger 0.63, full_bks_retained 0.65 (original arms 0.89–1.26). Design doc Section 3 "P2 result" | `suite_v1/logs/`, then `evaluation/validation/2M/{report.md,promotion.json,near_optimal_gate.md}` |
| P3 promotion (3 arms, 8M) | **done** 2026-10-25: no run passes; best arm full_dagger at distance 0.24 (static failure 0–4 %, success 95–100 %, hours +32–40 % vs BKS; same-day 0.78–0.84; drayage passes). Design doc Section 3 "P3 result" | `evaluation/validation/8M/near_optimal_gate.json` |
| Track B `suite_v3` (user decision 2026-10-27) | `full_dagger_plan_deep` × 3 seeds to 4M (imitation held at 0.1, DAgger β 0.5 → 0.2, 4-layer encoder, 64-seed selection): array 28639609 submitted 2026-10-27 | `suite_v3/logs/`, then `evaluation/validation/{4M,4M_noplan}/` |
| Track A decision-time search (user decision 2026-10-27) | code done (`algo/generalist_search.py`); validation on suite_v2: mask `8M_mask0` 28640780, `8M_mask1` 28640781 (six stopped runs); `8M_mask1_bestofn16` 28641937, `8M_mask1_lookahead4` 28641940–42 (full_dagger_plan s0–s2). 2026-10-28: distance full_dagger_plan 0.178 → 0.151 with the mask (margin 0, chosen on validation; margin 1 0.155), best-of-16 0.166 (worse than the mask alone); no pass; static hours unchanged (+11–35 %); lookahead: `8M_mask1_lookahead4` s1/s2 crashed on a copy bug at congested public stations (fixed, fallback + resumable episodes), rerun as `8M_mask0_lookahead4` 28650997–99. Lookahead s0 (28641940, margin 1) finished: distance 0.203 vs 0.152 for the mask alone (failures down, paired hours excess up 9–19 points); one rollout per candidate has ~5 h of noise, so the search picks almost at random; 28650997–99 cancelled before start. Pilot of a noise-free lookahead (candidates finished under nominal legs/energy/unloading), s0, 16 unselected seeds × 3 static problems, job 28658722 (`agent/pilots/nominal_lookahead/`). Validation seeds 0–15 were the checkpoint-selection seeds, so 16 of the 128 gate seeds are optimistic (the locked test set is clean). Pilot result: success 43/48 → 48/48 but hours unchanged (−0.7 / +2.2 / −1.9 %), not evaluated further. Diagnosis (`agent/pilots/plan_order`, `plan_adherence`): forcing the plan's customer order changes nothing; the gap is travel from charging execution (which charger, how much), which the plan input does not cover. **Track A closed: keep the mask (margin 0), no search** | `suite_v2/evaluation/validation/<search tag>/` |
| P5 locked test | locked; opens on a validation PASS | `agent/TEST_OPENED`, `evaluation/test/final/` |
| `suite_v4` charging plan (user decisions 2026-10-28) | the planner's moves to the active truck's next stop (stations, target SoCs, rests, queue join or leave), re-solved at every decision, join the plan input (`--charge-plan`, `PlanProvider.moves`); arm `full_dagger_plan_charge` = `full_dagger_plan` + `--charge-plan`; following the flagged move: 12/12 validation episodes succeed at +0.0..+7.1 % hours vs BKS; array 28671987, 3 seeds, 4M, 48 h; 45-54 decisions/s; pretrained checkpoint (16 selection seeds, no mask) already at static hours +0..+11 % vs BKS (s1: +0.0/+1.7/+4.8 %, failures 0), dynamic weak before PPO | `results/generalist_graphppo/suite_v4` |
| suite_v4 failure replays (2026-10-30/31) | same-day failures: the mask's fallback dropped the energy rule (fixed in `safe_actions`); static failures and most pnw time limits: a planned truck retiring for good (`--retire-guard`, a new plan-reading rule). **First validation PASS: `4M_mask0_fix_guard`, s0 and s2, arm distance 0.000** (fix alone 0.007, no pass; suite_v3 0.056 with the mask, no pass). `TEST_OPENED` not written: the pass depends on the guard, so it waits for the user | `suite_v4/evaluation/validation/4M_mask0_fix_guard/`, `agent/pilots/failure_replays/` |
| `suite_v2` plan-conditioned (user decision 2026-10-25) | plan cache done 2026-10-26 (identical to the BKS time-0 plans); 9 runs to 8M: array 28586156 in `results/generalist_graphppo/suite_v2` (first submission 28583920 aborted in BC replay by a rolling-planner crash when no truck can work, fixed 2026-10-26); stop rule at 4M stopped `full_bks_retained_plan` (2.215 -> 2.193) and `full_dagger_plan` (2.223 -> 2.214); `full_bks_plan` (1.92 -> 2.16) continues to 8M (reached 8M 2026-10-28 via 28641948; evaluations 28648943–45: `8M`, `8M_noplan`, `8M_mask0`; `8M` done 2026-10-28: no pass, arm distance 0.158, the best plan-on arm without the mask: static hours +20–26 %, same-day 0.78–0.87 of BKS; `8M_noplan` 0.359; `8M_mask0` 0.152. suite_v2 final with the mask: full_dagger_plan 0.151, full_bks_plan 0.152, full_bks_retained_plan 0.238; no run passes); validation evaluation of the six stopped runs (best.pt): 28627901 (`8M`), 28627905 (`8M_noplan`): all six FAIL (distance full_dagger_plan 0.178, full_bks_retained_plan 0.318 plan-on; plan-off 0.64 / 1.01) | `suite_v2/logs/`, then `evaluation/validation/{8M,8M_noplan}/` |

## 6. Stop rules

- No run passes at 8M: one `suite_v2` iteration, then stop and write up whatever the outcome. **Amended 2026-10-27 by the user:** after `suite_v2`, tracks A (decision-time search + feasibility mask) and B (`suite_v3`) run in parallel; write-up after both.
- A DAgger round above about 30 % of wall time in the pilot: stop and ask the user (the arm's schedule is a decision).
- A gate that fails because of MISSING BKS or NO_ROWS is a tooling failure to fix, not a result.
- Never more than 20 allocations, never a job on a GPU partition without a GPU, never a submission while `sinfo` shows the partitions down, and every composed command through `sbatch --test-only` before its first real submission.
