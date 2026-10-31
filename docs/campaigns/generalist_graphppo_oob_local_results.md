# oob_local campaign results (interim, 2026-10-27)

Status: **4 of 5 problems pass the near-optimal validation gate; ca_sameday stands at 84 % of the required services bar.** One run (`oob_privcritic_s1`) is still training; this document records everything measured so far and is finalized when it lands. All numbers: 128 validation seeds, local BKS bar (which reproduced the Snellius P0a bar nearly exactly — identical best-method counts, all 384 static CP-SAT bounds proven).

## Headline gate table (champion: `oob_v1/oob_privcritic_s0` @ 8M, tag `8M_repair`)

| Problem | Failure (≤ max(1 %, BKS)) | Success / services | Hours excess vs BKS | Status |
| --- | --- | --- | --- | --- |
| tx_longhaul | 0.0 % | 100 % | **0.1 % [−0.0, 0.3]** | **PASS** |
| tx_longhaul_congested | 0.0 % | 100 % | **1.3 % [0.7, 2.0]** | **PASS** |
| pnw_longhaul_hos | 0.0 % | 99.2 % | **3.1 % [2.1, 4.2]** | **PASS** |
| ca_drayage_combined | 0.0 % | **117 % of BKS** (35.3 vs 30.1 served; hindsight bound 31.5) | — | **PASS** |
| ca_sameday | 3.1 % | 84.4 % of BKS (19.71 vs 23.37) | — | FAIL |

The three statics pass **through the policy + planner-repair wrapper** (design 9.6: the policy's closed-loop visit order warm-starts the long-haul planner's continuation search). Measured decision latency, which any claim here must carry (`evaluation/validation/8M_repair/wrappers/`): mean per decision 0.234 s (tx) / 0.273 s (txc) / 0.462 s (pnw), p95 per decision 0.045 / 0.056 / 0.423 s, total **24.7 / 27.8 / 56.1 s of planning per episode** (worst episode 101 s). The mean exceeds the p95 on the Texas maps because the cost is concentrated in a few planner solves rather than spread over decisions. For reference the BKS methods behind the bar cost far more (`lh_plan` 35–87 s per episode at 40 s × 3 restarts), so the wrapper is not buying its gain with more compute than the bar it is measured against — but it is decidedly not a single forward pass, and the plain-policy statics do **not** pass (31–38 % hours excess). Drayage passes with the plain policy and **beats both the BKS portfolio and the clairvoyant hindsight bound**. Sameday's failure criterion passes (0.8–3.1 % across checkpoints); only its services criterion is short.

## What moved sameday (ratio of BKS services, 128 seeds)

| Lever | Result |
| --- | --- |
| Training 2M → 8M (privileged critic arm) | **0.56 → 0.844** (the dominant gain; curve still rising at 8M) |
| Privileged (asymmetric) critic vs control at 2M | equal services, failures 3.1 % → **0.8 %** (fixes the failure bar) |
| Scheduled tempered decoding (new `t_end`/`top_p`) | **+30 % at 2M** (0 failures at t=1.5→0.2, p=0.9); **neutral at 8M** — training absorbed the conservatism |
| 4× demonstrations (1,024 planner demos; BC holdout 0.74 → 0.83) | end-to-end **0.810** — better imitation fit did not survive PPO |
| Clairvoyant DAgger (dynamic-only, "learning by cheating") | sameday 0.604; best drayage ever (33.1) but statics degraded |
| LR-dropped polish continuation (9.2) | regressed (0.798, 6.2 % failures) — chained fresh-optimizer runs stall |
| Regret-normalized reward (9.4) | training collapse on dynamics (drayage served 0) |
| 2-member ensemble (champion + demos1k) | neutral (paired 20.28 vs 20.53 on 32 seeds) |
| Simulated lookahead | +2.2 served but **excluded**: the environment clones replay the episode's own draw/request stream, which is mildly clairvoyant; also 7 % failures |

Diagnosis (9.1), two stages:

- **At 2M** the policy *abandoned* 7.8 requests/seed at BKS-level hours — over-conservatism, where greedy decoding suppressed service commitments the policy's own distribution already contained (hence the +30 % from tempered decoding).
- **At 8M** that is gone and the residual gap is a **load-scaling deficit**, measured per seed against the BKS of the same seed:

| Seed group (champion, 128 seeds) | n | Served (policy vs BKS) | Operating hours | Requests offered | Policy lost |
| --- | --- | --- | --- | --- | --- |
| Service gap ≥ 5 | 49 | 25.7 vs 33.0 | 307.4 vs 283.7 (**108 % of BKS hours**) | 33.9 | 3.6 |
| Service gap ≤ 0 | 32 | **11.0 vs 11.0** | 249.0 vs 245.7 (101 %) | 11.2 | 0.0 |

On light-load seeds the policy is **exactly at the best-known solution with zero losses**; the entire shortfall sits on dense seeds, where it works *harder* than the BKS (108 % of its hours, 68 % of them driving) and still loses requests. This is combinatorial packing of many simultaneous requests, not idleness and not timidity — the rolling planner re-optimizes all revealed requests at every decision, and one forward pass of a GNN policy does not reproduce that on dense instances. Mean hours overall are 281.8 (policy) vs 266.2 (BKS).

## Honest assessment

- The statics are solved to near-optimality in the gate's exact sense (≤ 5 % paired hours excess against planner-grade BKS, zero failures) by policy + repair — with per-decision latencies reported.
- On drayage the learned policy is **better than every planner in the portfolio and the hindsight bound** — the strongest single result of the campaign.
- On sameday every incremental lever plateaus at 0.81–0.85 of a near-clairvoyant bar. Reaching 0.95 likely needs a structural change, not tuning. Ranked next bets (not run here): **plan-conditioned input (GraphPPO-P, 9.5)** — the rolling planner costs only ~2 s/episode on sameday, so conditioning on its live plan is cheap at inference and directly imports its anticipation; potential-based shaping from planner cost-to-go (9.4); DAgger seeded with failure states (9.3).

## Campaign record

- Infrastructure: local BKS (`bks/`), references (`refs_v1/`), campaign dirs `oob_v1` … `oob_v6` on `/data/scratch/sorfanoudakis/evrp_generalist`; swarm log `agent/status.log`; caps held throughout (cuda:1 only, CPU cores 32–63, ≤ ~83 GB RAM observed).
- Four parallel implementation agents delivered the OOB code (privileged critic, clairvoyant expert, inference toolkit, signal pack) — 102 generalist unit tests green after integration; decoding extensions (`t_end`, `top_p`) and tooling (`generalist_assemble_tag.py`, `generalist_selfimit_harvest.py`, `generalist_bks_local.sh`) added during the campaign with tests.
- GPU contention from other users was handled by pack-size reduction (1 run/GPU), the minibatch-128 fallback (`oob_v3`), allocator `expandable_segments`, and preemptible checkpointing; two OOM casualties were restarted without loss.
- Test discipline: the 500-seed test bank remains **locked**; every decision above used validation only.

## Seed variance and ensembles (final, 2026-10-28)

Three independent 8M privileged-critic seeds landed at sameday ratios **0.844 / 0.810 / 0.732** (s0 champion / demos1k / s1; s1 had the best statics and selection 2.100). The 2- and 3-member logit ensembles were neutral on sameday (paired 20.28 and 20.22 vs the champion's 20.53 on 32 seeds) — weaker members dilute. Every incremental lever is now measured; all plateau at 0.81–0.85 of the near-clairvoyant bar.

## W5 (in progress): plan-conditioned input, GraphPPO-P

Per the campaign design, the W5 workstream opens after the gap reading, and the gap reading points at it: the deficit is the *packing* of dense request sets (table above), which is precisely what the rolling planner's re-optimization produces and a single forward pass does not. Conditioning the policy on that live plan imports the planner's combinatorial work as an input while leaving the policy free to deviate — the one lever that changes what the policy can represent rather than how it is trained or decoded.

Built and integrated (`algo/generalist_plan.py`, 113 generalist unit tests green): a `PlanTracker` caching the rolling planner's fleet plan per environment, re-planning only on a new request reveal or when the active truck's assignments run out; per-task (`plan_next_active`, `plan_next_any`, `plan_rank_active`), per-truck (`plan_has_route`) and per-candidate (`plan_on_next`) features kept out of `FEATURE_DIMS` so old checkpoints still load; plan dropout 0.3 on training episodes so the plan-free mode survives; measured plan cost **0.75 s/episode on sameday**, 4 s on drayage, reported per episode as `plan_calls`/`plan_seconds` for any gate claim. Flags: `--plan-conditioning`, `--plan-dropout`. Arm `oob_plancond` = privileged critic + plan conditioning, training to 8M (a plan-conditioned policy cannot `--init-from` a plain checkpoint, so it starts from BC).
