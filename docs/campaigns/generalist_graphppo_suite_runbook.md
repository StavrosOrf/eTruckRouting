# Runbook: generalist GraphPPO suite campaign (near-optimal gate, Snellius)

Written 2026-10-22 for the campaign agent that runs this campaign end to end. Design: `generalist_graphppo_suite_campaign.md`. Branch `NewFormulation`; the pre-flight code below is in the working tree, uncommitted.

## User decisions (2026-10-22)

| Topic | Decision |
|---|---|
| Case studies | The five suite problems in `EVRoutingEnv/config_files/generalist_suite/`: `ca_sameday`, `ca_drayage_combined` (dynamic), `tx_longhaul`, `tx_longhaul_congested`, `pnw_longhaul_hos` (static). |
| Goal | One policy, one set of weights, that **passes the near-optimal gate** (`scripts/campaigns/generalist_near_optimal_gate.py`) on all five problems: static failure ≤ max(1 %, BKS), success ≥ BKS − 2 pts, paired hours excess vs BKS ≤ 5 %; dynamic failure ≤ max(1 %, BKS), on-time services ≥ 95 % of BKS. Thresholds as coded. |
| Cluster | **Snellius**, account `tdsei17344`. `gpu_a100` first (18 cores, ~117 GB per GPU share); overflow `gpu_h100` (16 cores) and `gpu_mig` (9 cores) through `submit --partition`. CPU stages (references, BKS) also run on `gpu_a100` with one GPU; `cbuild` is a manual fallback. **At most 20 concurrent allocations** for this campaign (`max_concurrent`). |
| Static BKS | `generalist_bks_static.py`: greedy_teacher, alns_respect, lh_plan, lh_rolling, lh_plan_safe (40 s × 3 restarts) + nominal lower bound (120 s); 128 validation + 256 train seeds per problem. |
| Dynamic BKS | `generalist_bks_dynamic.py`: greedy_dispatch, wait_at_depot, kullman_myopic (drayage only), rolling request planner (non-clairvoyant); bound = **max(the clairvoyant rolling planner at the raised ALNS budget, the BKS)** per validation seed (`proven: false`, `raised_to_bks` marks seeds where the clairvoyant alone fell short); the planner never starts a daily rest while a known request is due before it would end (fixed 2026-10-22); `solve_request_hindsight` only in `check`, on tiny one-truck instances (one to three requests at 12 h; the search cannot exhaust larger ones). |
| Arms (P2) | 7 × 3 seeds = 21 runs at BC + 2M: `base`, `fleet`, `paired`, `full`, `full_bks`, `full_bks_retained`, `full_dagger`. |
| Imitation arms | BKS-paired reward (`--reward-references <campaign>/bks`), `--bc-episodes 256`; retained BC 0.1 → 0 over 2M; DAgger every 10 updates, 8 episodes/problem, β 0.5 → 0 over 2M, 2 pretrain rounds. `full_dagger` keeps the retained term (DAgger labels enter the PPO loss only through it). |
| DAgger experts | Static: `LonghaulPlanPolicy(replan="events")` at 8 s / 1500 iterations / 1 restart (`algo/generalist_experts.py`); dynamic: the rolling planner (same parameters as the BKS method). |
| DAgger cost (2026-10-23) | Measured in the pilot: a round (40 episodes) takes 194–218 s against 334 s of PPO per 10 updates, about 38 % of wall time. **Kept as decided**: a 2M run of `full_dagger` takes about 7 h (12 h limit), an 8M run about 25 h (36 h limit, resumable). |
| Promotion | P2 → the **3** best eligible arms (served ≥ teacher and failure ≤ teacher on every problem; highest mean score) → 8M. No PASS at 8M → one diagnostic iteration `suite_v2` (one flag change per arm), then stop and write up. **Amended 2026-10-24** after the 2M evaluation left no arm eligible (the teacher never strands, every arm strands on the static problems): promote the **3 arms closest to the gate** by mean shortfall over problems and seeds (`generalist_near_optimal_gate.py --promote 3`, section `Arms by distance to the gate`); eligibility stays reported. |
| After `suite_v2` (2026-10-27) | The six stopped `suite_v2` runs fail the gate (best arm `full_dagger_plan`, distance 0.178). Instead of stopping, the user chose **two parallel tracks, started at once**: **A** decision-time search on the existing checkpoints (both a per-decision policy lookahead and best-of-N policy rollouts feeding the plan input, the better one chosen on validation; rollouts use nominal or freshly sampled conditions, never the episode's own future draws) plus a **feasibility mask** on all five problems (no move after which a truck can no longer return before its shift ends or reach a charger); **B** `suite_v3`: `full_dagger_plan_deep` (imitation term held at 0.1, DAgger β 0.5 → 0.2 over 4M, 4-layer encoder, selection on 64 validation seeds), 3 seeds, fresh from BC, to 4M with the learning rate decaying over 4M. A `full_bks_plan` PASS in `suite_v2` still opens P5 as agreed and cancels B. |
| After tracks A and B (2026-10-28) | No `suite_v2` run passes (with the mask: `full_dagger_plan` 0.151, `full_bks_plan` 0.152); track A is closed with the mask only (lookahead and best-of-N do not reduce hours). Diagnosis: the static gap is travel from charging (which station, how much), not customer order. The user chose **`suite_v4` now, alongside `suite_v3`**: the planner's moves to the active truck's next stop (stations, target SoCs, rests, queue join or leave) join the plan input, **re-solved at every decision** (the user's second choice, after re-solving only at plan redos left stale charge levels that stranded trucks). |
| Scope | Near-optimal on the five problems only. P4 (LOPO, specialists) and the size probes are **out**; `prepare --probes` can still write them for a later campaign. |
| Authority | Submit every P0–P3 stage without asking. Change code without a review gate. **Open P5 automatically** when a run passes the gate on validation. **Never commit unless the user asks.** |

The agent appends one line per turn to `results/generalist_graphppo/agent/status.log` and writes `results/generalist_graphppo/agent/TEST_OPENED` when it opens P5.

## 0. What exists (read first)

| File | Role |
|---|---|
| `SWARM.md` (repo root) | The swarm's goal (active and unmet until the locked-test gate passes), roles, shared state, stage board and stop rules |
| `docs/campaigns/generalist_graphppo_suite_campaign.md` | Design: arms, gates, compute, risks |
| `algo/generalist_observation.py`, `generalist_policy.py`, `generalist_ppo.py`, `generalist_imitation.py` | Size-agnostic observation, network, PPO with BC warm start, retained imitation and DAgger |
| `algo/generalist_experts.py` | DAgger expert factories (`longhaul_plan`, `rolling_request`); `ROLLING` is also the dynamic BKS planner |
| `scripts/training/train_generalist_ppo.py` | Trainer CLI; writes `training.pt`, `best.pt`, `best_config.json`, `final.pt`, `*_history.json`, `run_config.json` |
| `scripts/campaigns/generalist_graphppo.py` | Driver: `prepare` (`--bks-from`, `--arm-set screen|named`, `--probes`), `references`, `merge-references`, `train`, `train-pack`, `submit` (`--partition`, `--test-only`), `evaluate`, `report --promote N` |
| `scripts/campaigns/generalist_graphppo_{gpu,cpu,eval}.sbatch` | Allocation wrappers; the GPU wrapper forwards USR1 at T−300 s and trainers checkpoint on it |
| `scripts/campaigns/generalist_bks_common.py` | Shared BKS machinery and the `<split>.json` format; the test-split guard |
| `scripts/campaigns/generalist_bks_static.py`, `generalist_bks_dynamic.py` | BKS builders: `run`, `bound`, `merge`, `demos`, `snapshot` (+ `check` for dynamic) |
| `scripts/campaigns/generalist_bks.sbatch` | One gpu_a100 share works through a plan file: `sbatch --job-name bks_<p>_<split> scripts/campaigns/generalist_bks.sbatch SOURCE_DIR {static|dynamic} PLAN_FILE [BKS_OUT]` |
| `scripts/campaigns/generalist_near_optimal_gate.py` | The gate; `--allow-test` opens the test split (P5 only) |
| Tests | `tests/unit/test_generalist_{campaign,experts,bks,near_optimal_gate,imitation,observation,policy,ppo}.py`, `tests/integration/test_generalist_suite_policy.py` |

Fixed in pre-flight: `LonghaulPlanPolicy._replan` passed a `descent_rounds` argument the planner's `solve` does not take, so every re-plan (`lh_rolling`, the DAgger expert) crashed; `solve` already ends with a descent to a fixpoint.

Also fixed in pre-flight (user decision 2026-10-22): the drayage model of the rolling request planner could start a daily rest while a known request was due, because resting is free in its objective and a shift counts as started at t = 0; on 12 h sanity instances it rested at t = 0 and missed every request, and at full scale the clairvoyant bound fell below the causal BKS. `DrayageModel.rest_blocked` now stops a planned rest (REST token, rest-before-a-long-wait, idle rest) whenever a known or revealed job is released before the rest would end; forced rests (`required_rest`) are unchanged. Covered by `tests/unit/test_rolling_request_planner.py`.

## A. Pre-flight (done; verify before the real `prepare`)

```
pytest -q -m "not statistical" tests/unit                      # baseline 701 + the new tests
pytest -q tests/integration/test_generalist_suite_policy.py
grep -rn "research-eemcs-ese\|gpu-a100\|compute-p1\|/scratch/sorfanoudakis" scripts/campaigns docs/campaigns/generalist_graphppo_suite_* --exclude=generalist_graphppo_suite_runbook.md   # nothing
```

## B. Mini check (login node, CPU, about 20 minutes)

`B=results/generalist_graphppo/mini_check/bks`, `P=.venv/bin/python`, `PYTHONPATH=.`, `BKS_OUT=$B` exported for the BKS commands.

1. BKS on two seeds per problem so that every problem has files:
   - static, each of `tx_longhaul tx_longhaul_congested pnw_longhaul_hos`: `run --problem P --methods greedy_teacher lh_plan --count 2 --workers 2`; `bound --problem P --count 2 --workers 2`; `merge --problem P --count 2`; `run --split train --count 4 --methods greedy_teacher lh_plan --workers 2`; `demos --problem P --count 4 --workers 2`;
   - dynamic, each of `ca_sameday ca_drayage_combined`: `run --problem P --methods greedy_dispatch rolling_request --count 2 --workers 2`; `bound --problem P --count 2 --workers 2`; `merge --problem P --count 2`; `run --split train --count 4 ...`; `demos --problem P --count 4 --workers 2`; `check --problem P --seeds 8`.
   - **Record** the rolling planner's `seconds` per episode (`runs/rolling_request__validation.jsonl`) as `c`, and `lh_plan`'s.
   - The drayage `check` keeps hours of service on (the rest rule is part of what it verifies); `--no-hos` isolates the routing logic if it ever fails again.
2. `prepare --directory results/generalist_graphppo/mini_check --seeds 1 --bks-from $B`, then shrink `manifest.json`: seeds train 10 / validation 4 / test 2, `reference_shard` 5, `schedule_timesteps` 10240, `common_flags` = `--num-envs 5 --rollout-steps 8 --minibatch-size 20 --hidden-dim 32 --encoder-layers 1 --bc-episodes 1 --pretrain-epochs 1 --train-scenarios 10 --validation-scenarios 1 --validate-every 1 --charger-budget 32 --epochs 1 --learning-rate 0.0003 --entropy-coefficient 0.01 --gamma 1.0 --gae-lambda 0.95 --target-kl 0.05 --action-head-layers 1 --pretrain-learning-rate 0.0003 --checkpoint-every 1`, runs = `full_dagger_s0` only (edit its `--bc-episodes` to 4, `--pretrain-epochs` to 2, `--dagger-episodes` to 1, `--dagger-every` to 1, `--dagger-pretrain-rounds` to 1, and add `--dagger-envs-per-problem 1`; the arm's own flags override the common ones).
3. `references --tasks 0,1,...,14 --workers 4`, `merge-references`, `train --run 0 --budget 80 --device cpu`, `evaluate --split validation --tag mini --count 2 --workers 4`, `report --split validation --tag mini --promote 1`, `generalist_near_optimal_gate.py --campaign results/generalist_graphppo/mini_check --split validation --tag mini --bks-dir $B`.

**Gate:** `report.md` lists the five problems and no probes and ends with a Promotion table; the trainer log shows behaviour cloning from `bks/<p>/train_demonstrations.json`, `[dagger]` rounds with about zero expert errors and an agreement figure for every problem; the gate prints a table with no MISSING row; every `hindsight_check.json` says `PASS` (`planner_exceeds == 0`).

## P0a. Best-known solutions (10 jobs, gpu_a100, 6 h each)

Submit nothing while `sinfo -p gpu_a100` shows the partition down. `B=results/generalist_graphppo/bks`.

```
PYTHONPATH=. $P scripts/campaigns/generalist_bks_static.py snapshot --name v1      # $B/_source/v1 and $B/logs
mkdir -p $B/plans
```

Plan files (`{W}` becomes the core count):

- `$B/plans/<p>_val.txt` for each static problem:
  `run --problem <p> --methods greedy_teacher alns_respect lh_plan lh_rolling lh_plan_safe --split validation --count 128 --workers {W}`, then `bound --problem <p> --count 128 --workers {W}`, then `merge --problem <p>`;
- `$B/plans/<p>_train.txt`: `run --problem <p> --methods greedy_teacher alns_respect lh_plan lh_rolling lh_plan_safe --split train --count 256 --workers {W}`, then `demos --problem <p> --count 256 --workers {W}`;
- dynamic, the same two files with `--methods greedy_dispatch wait_at_depot rolling_request` (`ca_sameday`) or `--methods greedy_dispatch wait_at_depot kullman_myopic rolling_request` (`ca_drayage_combined`); `bound --problem <p> --count 128 --workers {W}` runs the clairvoyant planner at the raised budget (default `--iterations 300`, up to ~500 s per episode) and `merge` floors it at the BKS.

```
sbatch --job-name bks_<p>_<split> scripts/campaigns/generalist_bks.sbatch $B/_source/v1 {static|dynamic} $B/plans/<p>_<split>.txt $B
```

Check each command first with `sbatch --test-only ...`. A job that hits its limit is resubmitted unchanged: rows are appended per seed and finished seeds are skipped.

**Gates:** all five `$B/<p>/validation.json` hold 128 seeds and `train_demonstrations.json` at least 200 replay-verified non-failed episodes; in every `summary.json` the BKS beats the teacher (`bks.mean_hours_successful` below the teacher's for static problems, `bks.mean_served` above for dynamic ones) and `best_method_counts` is not all greedy; `bound.exceeded_seeds` is recorded for the dynamic problems.

## P0b. References (3 jobs, about 20 minutes)

```
D=results/generalist_graphppo/suite_v1
PYTHONPATH=. $P scripts/campaigns/generalist_graphppo.py prepare --directory $D --arm-set screen --seeds 3 --bks-from $B
PYTHONPATH=. $P scripts/campaigns/generalist_graphppo.py submit --directory $D --stage references --pack 3 --test-only
PYTHONPATH=. $P scripts/campaigns/generalist_graphppo.py submit --directory $D --stage references --pack 3
PYTHONPATH=. $P scripts/campaigns/generalist_graphppo.py merge-references --directory $D
```

**Gates:** 45 shards; `references/summary.json` close to the 14-seed table of the design doc; at least 64 non-failed train episodes per problem.

## P1. GPU pilot (one allocation, 2 GPUs, 3 h)

```
prepare --directory ${D}_pilot --seeds 1 --references-from $D/references --bks-from $B
submit --directory ${D}_pilot --stage train --budget 200000 --pack 2 --arms full full_dagger
```

**Gates** (from `logs/<run>_200000.log` and `training/<run>/*_history.json`): `full_s0` runs on CUDA at ≥ 150 decisions/s (about 250 expected) and `gpu_peak_gb` fits; `full_dagger_s0` clones from the BKS demonstrations with `misses` ≤ 0.5 % (`pretrain_history.json`), logs two DAgger pretrain rounds, about zero `dagger/expert_errors`, an agreement per problem, and checkpoints and resumes. **Measure the DAgger round time** (`dagger/seconds` per round): if rounds take more than about 30 % of wall time, stop and report to the user (changing `--dagger-every` or `--dagger-episodes` changes the decided arm). If GPU memory fails, set `--minibatch-size 128` in `COMMON`, delete the untrained `suite_v1`, and re-prepare with `--references-from` and `--bks-from`. Rewrite `train_hours` from the measured rate if it is far from 250 decisions/s.

**P1 outcome (2026-10-23):** every gate passed except BC coverage on the BKS demonstrations: 0.77 % misses at `--charger-budget 32`, all of them charger destinations outside the subgraph (teacher demonstrations: 0.04 %). The remedy is the observation-only change the design doc names (Section 5 and 9.2): the common charger budget is raised to the smallest of 48 or 64 that brings the misses under 0.5 % (measured by replaying the demonstrations through the observation at each budget); if neither does, 64 is used and the residual miss rate is recorded. Memory headroom is ample (10.9 GB peak of 40 GB at budget 32). The DAgger cost (38 % of wall time) was kept as decided by the user.

**Decision (2026-10-23 19:40): `--charger-budget 64` in `COMMON` for `suite_v1`.** Measured by replaying the BKS demonstrations through the observation: California problems miss nothing at any budget; `tx_longhaul` misses 0.78 % of its decisions at 48 and 0.32 % at 64; `tx_longhaul_congested` 1.5 % at 48 (64 not finished before the measurement's time limit; Pacific Northwest not reached). A budget of 48 would pass the 0.5 % gate only because the California demonstrations dilute the average, while the misses sit exactly where charging binds, so 64 was taken. The pilot ran at 32; P2 throughput and memory at 64 are read from the first screen runs (`gpu_<job>.csv`, `training_history.json`).

## P2. Screen (21 runs, BC + 2M)

```
submit --directory $D --stage train --budget 2000000                       # 21 single-GPU jobs, throttle 20
submit --directory $D --stage evaluate --split validation --tag 2M --runs <7 runs>   # three groups of seven
report --directory $D --split validation --tag 2M --promote 3
PYTHONPATH=. $P scripts/campaigns/generalist_near_optimal_gate.py --campaign $D --split validation --tag 2M --bks-dir $B --promote 3
```

`--pack 4` packs four runs on one node when whole nodes are idle; `--partition gpu_h100` or `gpu_mig` moves a stage to an overflow partition. **Promotion (amended 2026-10-24):** the gate's `--promote 3` names the three arms with the smallest mean shortfall to the gate (static: hours excess above 5 %, success below 98 %, failure above the limit; dynamic: services below 95 % of the BKS, failure above the limit; means over problems and seeds). `report --promote 3` still reports the teacher-relative eligibility, which no arm met at 2M.

## P3. Promotion (three arms × three seeds, 2M → 8M)

```
submit --stage train --budget 8000000 --arms A B C          # continues from training.pt; 36 h limit, resubmit on USR1
submit --stage evaluate --split validation --tag 8M --runs <their 9 runs>
report --split validation --tag 8M --promote 3
generalist_near_optimal_gate.py --campaign $D --split validation --tag 8M --bks-dir $B
```

Stop an arm early when its 16-seed in-training validation score has not improved between 2M and 4M (`validation_history.json`). **Success:** a run with `"overall": "PASS"` in `evaluation/validation/8M/near_optimal_gate.json`.

## P5. Locked test (automatic once a run passes on validation)

1. Write `results/generalist_graphppo/agent/TEST_OPENED` (date, the passing runs, the gate file).
2. `cmp $D/bks/<p>/validation.json $B/<p>/validation.json` for every problem (the frozen copy must equal the global one).
3. BKS on the test seeds, same portfolio and budgets, no demonstrations: plan files `<p>_test.txt` with `run ... --split test --count 500 --allow-test --workers {W}`, `bound ... --split test --count 500 --allow-test --workers {W}`, `merge --problem <p> --split test --allow-test`; submit through `generalist_bks.sbatch` (static ≈ 4 h per problem).
4. `submit --stage evaluate --split test --tag final --runs <every seed of the passing arms>`.
5. `report --split test --tag final`; `generalist_near_optimal_gate.py --campaign $D --split test --tag final --bks-dir $B --allow-test`.
6. Write `docs/campaigns/generalist_graphppo_suite_results.md`: per problem served fraction, success with Wilson interval, failures, hours, Δhours vs teacher and vs BKS with bootstrap intervals, score, the gate table, and decision latency.

Make no design change after `TEST_OPENED`.

## If no run passes at 8M: `suite_v2`, one iteration

Add recipes `<arm>_v2` to `RECIPES`, one flag change each, chosen from the 8M diagnosis (Section 9 of the design doc, "Out-of-the-box solutions", is the idea bank; its 9.1 diagnostics come first): `--charger-budget 64` when failures are charging strandings on Texas or the Pacific Northwest; `--bc-coefficient-final 0.02` when the imitation arms regress after the anneal; `--dagger-beta-final 0.2` when DAgger agreement collapses; `--entropy-coefficient 0.02` when the entropy collapses. Then

```
prepare --directory results/generalist_graphppo/suite_v2 --arm-set named --arms <v2 arms> --seeds 3 --references-from $D/references --bks-from $B
```

and P2 → P3 for those arms only. Afterwards stop and write up whatever the outcome.

## suite_v2 (plan-conditioned; user decisions 2026-10-25)

1. Time-0 plans for the static problems (one gpu_a100 share per problem, about 2–3 h): plan files `bks/plans/<p>_plans.txt` with `plans --problem <p> --split validation --count 128 --workers {W}` and `plans --problem <p> --split train --count 2048 --workers {W}`, run through `generalist_bks.sbatch` from the source snapshot `bks/_source/v2`. Gate: `bks/<p>/plans_{train,validation}.json` hold 2,048 and 128 plans.
2. `prepare --directory results/generalist_graphppo/suite_v2 --arm-set named --arms full_bks_plan full_bks_retained_plan full_dagger_plan --seeds 3 --references-from results/generalist_graphppo/suite_v1_refs/references --bks-from results/generalist_graphppo/bks` (refuses without the plan files).
3. `submit --stage train --budget 8000000` (9 runs, 36 h limit, resumable). Stop rule at 4M as in P3. Expect lower throughput than `suite_v1` (planner reads every decision); measure it on the first updates.
4. `evaluate --split validation --tag 8M` (plan on) and `--tag 8M_noplan` (plan withheld), `report --promote 3`, and the gate on `8M`. A PASS opens P5 as before; at P5 the static test plans are built first with `plans ... --split test --count 500 --allow-test` into the global BKS directory, which test evaluation reads.
5. If no run passes: stop and write `docs/campaigns/generalist_graphppo_suite_results.md`.

Fixed on the way (2026-10-26): the plan input reads a plan at every observation, including states where every truck has ended its shift; the rolling planner's ALNS then drew from an empty truck set (`IndexError` in BC replay). `FleetPlanner.plan` now returns the empty plan when no truck can work (`tests/unit/test_rolling_request_planner.py`); all 128 sampled dynamic BKS demonstrations replay through the plan view.

## After suite_v2: tracks A and B (user decisions 2026-10-27)

- **B, `suite_v3`.** `prepare --directory results/generalist_graphppo/suite_v3 --arm-set named --arms full_dagger_plan_deep --seeds 3 --references-from results/generalist_graphppo/suite_v1_refs/references --bks-from results/generalist_graphppo/bks --schedule 4000000`; `submit --stage train --budget 4000000 --hours 36` (a 4-layer encoder at about 45 decisions/s needs about 26 h; the default for 4M is 24 h). Then `evaluate --tag 4M` and `--tag 4M_noplan`, `report --promote 3`, the gate on `4M`.
- **A, decision-time search** on the existing `suite_v2` checkpoints (and `suite_v3` once trained): feasibility mask on all five problems; per-decision lookahead and best-of-N; chosen on validation, decision latency reported with the gate. A PASS opens P5 as agreed. Code: `algo/generalist_search.py`.
  - Mask: `submit --stage evaluate --split validation --tag 8M_mask<m> --runs ... --safety-mask --safety-margin <m>` (`EVRP_SAFETY_MASK`, `EVRP_SAFETY_MARGIN`). Rules: a customer move must leave the energy (nominal × worst-case multiplier) to reach a charger or the depot; a move or charge must leave time to deliver what is on board and reach the depot before the horizon minus the margin (nominal travel, exact charge time, one charging stop at full power). When the rules remove every feasible action the hard mask stands. The margin (0 or 1 h) is chosen on validation.
  - Search (static problems only; dynamic problems keep the batched path with the mask): `--search lookahead --search-k 4` or `--search best_of_n --search-n 16`. A copy of the simulator (`fresh_clone`) keeps what has happened and redraws every future draw (travel, energy, service, station traffic) from a new seed, never the episode's own; it carries a copy of the plan provider, which keeps re-planning on deviations (a frozen plan sent the policy into charging loops). Lookahead plays the policy's 4 likeliest actions in copies sharing one seed and finishes greedily, taking the best outcome (no failure, then success, then hours). Best-of-N samples 16 rollouts at the first decision; the best one's routes, if it served everyone, become the planner's time-0 plan.
  - Every decision-time tag freezes `decision.json` and a code snapshot (`<tag>/source`) at submission; the evaluation wrapper runs that snapshot.
  - Measured on one CPU core (Texas): plain episode 6 s (82 decisions); copy 20 ms; one rollout 4–5 s; best-of-8 50–70 s per episode.

## suite_v4: charging plan (user decisions 2026-10-28)

- **What changes.** `ObservationSpec.charge_plan` (`--charge-plan`, needs `--plan-conditioned`) appends three action columns (`planned_move`, `planned_station`, `planned_station_level`) and one context column (`charge_plan_available`) after the plan columns, and keeps the active truck's planned stations in the charger subgraph. The moves come from `PlanProvider.moves`: the long-haul planner's continuation search along the truck's planned customers, from its state at each decision, with the same searches as `LonghaulPlanPolicy.__call__` (queue join or leave, charge-only at the head of a queue). Static problems only; zero on the dynamic problems and whenever the plan is withheld (dropout 0.3 on train seeds, `EVRP_PLAN_MODE=off`).
- **Check before training.** Following the flagged move on 12 validation episodes (seeds 64–67, three static problems) succeeds 12/12 at +0.0 to +7.1 % hours vs BKS (re-solving only at plan redos: 3/6). The moves cost 14–20 ms per step on Texas and about 160 ms on `pnw_longhaul_hos` (hours-of-service labels) on the loaded login node.
- **Arm.** `full_dagger_plan_charge` = `full_dagger_plan` + `--charge-plan`, nothing else (same 8M schedule, 16-seed selection), so it compares with `suite_v2`'s `full_dagger_plan` at ≤ 4M.
- **Commands.** `prepare --directory results/generalist_graphppo/suite_v4 --arm-set named --arms full_dagger_plan_charge --seeds 3 --references-from results/generalist_graphppo/suite_v1_refs/references --bks-from results/generalist_graphppo/bks`; `submit --stage train --budget 4000000 --hours 48`; then `evaluate --tag 4M`, `--tag 4M_noplan` and the mask (`4M_mask0`), the gate on each.

## suite_v4 failure replays (2026-10-30)

Gate on `interim_mask0` (the same weights as a `4M_mask0` tag): distance 0.015, no pass. Replays of the failed episodes: `results/generalist_graphppo/agent/pilots/failure_replays/`.

- **Mask fallback fixed.** On same-day the time rule could empty the safe set, and the fallback then also dropped the energy rule, so the policy delivered into a stranding. `safe_actions` now falls back to the energy rule first, then to the hard mask. Tags from before 2026-10-30 ran the old fallback (frozen in `<tag>/source`).
- **Retirement guard (`--retire-guard`, needs `--safety-mask`; static problems; off by default).** The depot action retires a truck for good. The guard hides it from a truck whose planned route is not empty while anything else is left. It reads the plan at decision time, so **a PASS that depends on it goes to the user before `TEST_OPENED` is written**.
- Tags on suite_v4: `4M_mask0_fix` (fallback only), `4M_mask0_fix_guard` (fallback + guard). Gate them as any other tag.

## Monitoring and failure handling

- `sinfo -p gpu_a100,gpu_h100,gpu_mig,cbuild` before submitting; `squeue -u $USER` and `sacct -j <job>` every 30–60 minutes during training; never busy-wait on the login node.
- Every composed command goes through `--test-only` (or `sbatch --test-only`) once before its first real submission.
- Logs: `$D/logs/<run>_<budget>.log`, `$D/logs/slurm-%A_%a.out`, `$D/logs/gpu_<job>.csv`; progress in `training/<run>/validation_history.json`; submissions in `logs/submissions.jsonl`; BKS logs in `$B/logs/`.
- Re-submitting a stage skips completed runs and resumes partial ones from `training.pt`; in-flight episodes restart from their own seed, so a resume is not bit-identical to an uninterrupted run.
- A finished run whose allocation lingers (2026-10-24, `full_s2`: the trainer printed `done in`, a forked worker stayed in uninterruptible disk sleep on GPFS and the job idled): once `completed_run` is true and the run log has `done in`, cancel the array element; the coordinator's watcher does this automatically for `suite_v1_train` jobs.
- Snellius refuses `scontrol update` on jobs (time limit, array throttle); resubmit instead. A stale `JobArrayTaskLimit` can hold the last element of an array after earlier elements finish: cancel it and resubmit the run as a single-element array through the GPU wrapper (`--array=0-0 ... <budget> 1 <run index>`), recording it in `logs/submissions.jsonl`.
- `submit --stage train` refuses to run before `merge-references`; `prepare` refuses the imitation arms without `--bks-from`.
- Partition fallbacks: `submit --partition gpu_h100` (16 cores per GPU) or `gpu_mig` (9); for a CPU stage on `cbuild` compose the command with `--dry-run` and submit by hand with `--partition=cbuild --gpus-per-task=0 --cpus-per-task=16`.

## Invariants (do not break)

- Decisions use validation only. Test seeds never enter training, references or the BKS before P5; the BKS builders and the gate refuse the test split without `--allow-test`.
- Worker processes fork before any CUDA use; `evaluate` builds its environments before a policy moves to the GPU.
- `schedule_timesteps` = 8M is frozen in the manifest; do not extend past 8M in a directory.
- The reward weights (2 / 1 / 0.5 / 1 / 2 / 2) and the gate thresholds are fixed a priori.
- Arms differ only in their flags; the observation width is identical across arms.
- `prepare` is immutable; the campaign's `bks/` copy is the frozen training input, the global `$B` may grow (test seeds at P5) and is what the gate reads.
- The BKS-paired reward is exactly paired on the 256 demonstration seeds and the 128 validation seeds; other train seeds use the BKS median.

## Verification

- `pytest -q -m "not statistical" tests/unit` (701 baseline plus the new tests) and `pytest -q tests/integration/test_generalist_suite_policy.py` pass.
- The mini check (§B) passes its gate, including every `hindsight_check.json`.
- The P1 gates pass before P2 is submitted; the gate tables for 2M and 8M are written under `evaluation/validation/<tag>/`.
