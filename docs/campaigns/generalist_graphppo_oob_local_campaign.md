# Out-of-the-box ideas campaign: generalist GraphPPO, local box (oob_local)

Started 2026-10-23 on user direction: "start a new out-of-the-box ideas campaign to solve the generalist PPO problem; run multiple parallel agents to solve it near-optimally with the generalist GraphPPO; use cuda:1 and up to half the RAM and CPU cores; use the scratch partition for large files." This campaign runs on **iepg-compute1.ewi.tudelft.nl** and is independent of (and concurrent with) the Snellius `suite_v1` campaign; it implements and screens the Section 9 idea bank of `generalist_graphppo_suite_campaign.md`. The goal, gate, reward, thresholds and invariants of the suite campaign and its runbook stay binding; only cluster, directories and arms differ, under the user's 2026-10-23 authority.

## Goal and gate (unchanged)

One GraphPPO policy that passes `scripts/campaigns/generalist_near_optimal_gate.py` on all five suite problems, first on 128 validation seeds, then once on the locked 500-seed test bank. Criteria as in SWARM.md Section 1. Decisions on validation only; test opens only on a validation PASS (write `agent/TEST_OPENED`).

## Resources (hard caps)

| Resource | Cap | Enforcement |
| --- | --- | --- |
| GPU | **cuda:1 only** (A30 24 GB; GPU 0 belongs to another user) | `CUDA_VISIBLE_DEVICES=1` on every GPU process |
| CPU | **cores 32–63** (half of 64) | `taskset -c` on every compute process; BKS: 30 cores, training: 3×10-core packs |
| RAM | ≤ 500 GB of 1007 | worker counts; check `free -g` before each wave |
| Disk | `/home` is 99 % full — **everything heavy on scratch** | `results/generalist_graphppo` → symlink → `/data/scratch/sorfanoudakis/evrp_generalist` |

## Layout (all under the symlinked `results/generalist_graphppo/`)

- `bks/` — local BKS, same portfolio/budgets as P0a (`_source/v1` snapshot, `plans/`, `logs/`). Local BKS is built on this box because Snellius results are unreachable from here; ALNS is time-budgeted, so the local bar may differ slightly from the Snellius bar. The gate always reads this local bar.
- `refs_v1/` — references-only campaign dir (teacher on 2,048 train + 128 val seeds per problem, 45 shards, claim-based).
- `oob_v1/` — control arms on existing code, prepared with `--references-from refs_v1/references --bks-from bks`.
- `oob_v2_<idea>/` — one campaign dir per integrated new-code idea (prepare freezes source, so new code ⇒ new dir).
- `agent/status.log` — swarm log, same conventions as SWARM.md Section 3; `patches/` — agent-delivered patches + reports.

## Stages

| Stage | What | Gate |
| --- | --- | --- |
| L0 BKS (running since 2026-10-23 18:40) | 5 background jobs, one per problem: val 128 run+bound+merge, train 256 run+demos (+check for dynamic); cores: pnw 8, tx 6, txc 6, sameday 4, drayage 6 | P0a gates of the runbook (128 seeds, ≥200 verified demos, BKS beats teacher) |
| L0b references | shards claimed on freed cores as BKS jobs finish; `merge-references` | ≥64 non-failed train episodes per problem |
| L1 pilot | `oob_v1` run 0 to 200k on cuda:1, 10 cores | ≥60 dec/s local, GPU mem ≤ 7 GB/run (3 runs must share 24 GB), BC misses ≤ 0.5 % |
| L2 controls | `full_bks`, `full_bks_retained`, `full_dagger` × 1 seed at BC+2M (3-run pack on cuda:1) | evaluate 128 val seeds, report, near-optimal gate per problem |
| L3 OOB arms | integrated ideas (below), 1 seed at BC+2M each, rotating through the 3-slot pack; best arms continued to 8M (`schedule_timesteps` stays 8M) | same gate; 9.1 gap diagnosis decides what runs next |
| L4 inference toolkit | lookahead / planner-repair / temperature / ensemble applied to every ≥2M checkpoint; decision latency reported | the gate on wrapped policies; latency must be stated with any claim |
| L5 locked test | only on a validation PASS, same rules as P5 | `evaluation/test/final/near_optimal_gate.json` |

## Out-of-the-box workstreams (parallel agents, worktree-isolated)

One idea per arm, flags default-off, no behavior change when off; each agent delivers a patch + tests to `patches/`; the coordinator integrates, re-runs the generalist tests, and prepares the next `oob_v2_*` dir. From Section 9's ranked shortlist:

| # | Workstream | Design-doc entry | Touches |
| --- | --- | --- | --- |
| W1 | Privileged (asymmetric) critic: value head sees the hidden request stream and realized draws during training; actor stays causal | 9.4 / ranked #1 | observation (privileged view), policy (value path), ppo (collection); `--privileged-critic` |
| W2 | Clairvoyant DAgger expert for the dynamic problems ("learning by cheating") | 9.3 / ranked #2 | experts (`rolling_request_clairvoyant`); arm flags only |
| W3 | Inference-time toolkit: simulated lookahead over top-k candidates, planner warm-start/repair for statics, decode temperature, seed ensembles; latency measured | 9.6 | new `algo/generalist_inference.py` + eval script; no training change |
| W4 | Training-signal pack: problem-balanced advantage normalization; regret-normalized terminal (BKS–teacher spread as the unit) | 9.4 | ppo update loop + reward; `--balanced-advantages`, `--regret-normalized-reward` |
| W5 (later, on diagnosis) | Plan-conditioned input (GraphPPO-P); self-imitation from policy wins; prioritized scenario sampling | 9.5 / 9.3 / 9.4 | larger; only after L2 gap reading |

## Rules carried over

- Never commit unless the user asks. Decisions on validation only. Test bank locked until a validation PASS. Reward weights and gate thresholds fixed. `prepare` immutable. One idea per arm.
- Every agent appends `<ISO time> <role>: <what>` to `agent/status.log`.
- Stop rule: if after the first OOB wave (W1–W4 arms at 2M) no run improves on the controls, write up the diagnosis before training anything else; cheap knobs (9.2) are allowed as single-flag variants.
