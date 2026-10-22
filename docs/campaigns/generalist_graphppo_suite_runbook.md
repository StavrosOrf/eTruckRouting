# Runbook: generalist GraphPPO suite campaign

Written 2026-10-22 for the campaign agent that runs this campaign end to end. Design: `generalist_graphppo_suite_campaign.md`.

User decisions (2026-10-22):
- **Cluster:** DelftBlue, account `research-eemcs-ese`. These are the defaults in the manifest `slurm` block of `scripts/campaigns/generalist_graphppo.py`.
- **Authority:** the campaign agent may submit every stage (P0–P4) without asking. It **must ask before opening the locked test bank (P5)**.
- **Scope:** full P0–P5.

The agent appends one line per turn to `results/generalist_graphppo/agent/status.log`, and writes `results/generalist_graphppo/agent/AWAITING_TEST_APPROVAL` with a results summary before P5.

## 0. What exists (read first)

| File | Role |
|---|---|
| `docs/campaigns/generalist_graphppo_suite_campaign.md` | Design: which plan ideas are used, arms, gates, compute, risks |
| `algo/generalist_observation.py` | Size-agnostic graph view: `GeneralistEnv`, `build_observation`, the charger subgraph `_charger_order`, `collate`, `GraphBatch` |
| `algo/generalist_policy.py` | `GeneralistActorCritic`: 1.15 M parameters, none depending on size |
| `algo/generalist_ppo.py` | `SuiteReward`, `ReferenceTable`, `SuiteVecEnv`, `GeneralistPPO`, `replay_demonstrations`, `evaluation_envs`, `evaluate_policy_suite`, `summarize_suite` |
| `scripts/training/train_generalist_ppo.py` | Trainer CLI. Outputs `training.pt`, `best.pt`, `final.pt`, `pretrained.pt`, `*_history.json`, `run_config.json` |
| `scripts/campaigns/generalist_graphppo.py` | Driver: `prepare`, `references`, `merge-references`, `train`, `submit`, `evaluate`, `report` |
| `scripts/campaigns/generalist_graphppo_{gpu,cpu}.sbatch` | Allocation wrappers. The GPU wrapper forwards USR1 at T−300 s, and the trainer checkpoints on it |
| Tests | `tests/unit/test_generalist_{observation,policy,ppo,campaign}.py` (24 pass), `tests/integration/test_generalist_suite_policy.py` |
| Refactors (behaviour-neutral; 701 unit tests pass) | `TRUCK_STATES` in `EVRoutingEnv/state/features.py`; `RunningStandardizer` and a parameterized `_TypedMessagePassingLayer` in `algo/canonical_encoders.py` |

Teacher reference on 14 seeds per problem (development sample):

| Problem | Success | Failed | Served fraction | Median hours |
|---|---|---|---|---|
| ca_sameday | 0/14 | 7/14 (strands near 35.4 of 36 h) | 0.86 | 276 |
| ca_drayage_combined | 0/14 | 0/14 | 0.19 | 360 |
| tx_longhaul | 14/14 | 0 | 1.00 | 355 |
| tx_longhaul_congested | 12/14 | 0 | 1.00 | 409 |
| pnw_longhaul_hos | 10/14 | 0 | 1.00 | 512 |

Do not commit unless the user asks. `prepare` snapshots the source tree and writes `source.diff` and `source.untracked`, so provenance is kept without a commit.

## A. Pre-flight code tasks (before the real `prepare`, which is immutable)

Make each change in `scripts/campaigns/generalist_graphppo.py` unless noted, and cover each with a case in `tests/unit/test_generalist_campaign.py`.

1. **`evaluate` and `report` take `--tag`** (for example `2M`, `8M`, `final`). Rows go to `evaluation/<split>/<tag>/<run>.json`; baselines stay shared in `evaluation/<split>/baselines/`. Today an existing file is skipped, so a P3 re-evaluation would silently reuse stale 2M results.
2. **`submit --stage evaluate`** plus `generalist_graphppo_eval.sbatch`: one GPU and 16 CPUs, running `evaluate --split S --tag T --device cuda:0 --workers 15 [--runs ...]`. Today evaluation has no sbatch path. Keep the invariant: `evaluate` builds `evaluation_envs` (fork) before any policy goes to CUDA.
3. **`submit --stage train --arms A B`**: submit only the runs of the given arms (needed for P3).
4. **`prepare --references-from DIR`**: copy a merged `references/*.json` and `summary.json`, so the pilot and P4 directories reuse the P0 references. Teacher runs are deterministic per seed, so the copy is exact.
5. In `train_generalist_ppo.py`, add `torch.cuda.max_memory_allocated()` (when on CUDA) and decisions/s to the progress print and to `training_history.json`.
6. **Check that the size probes generate** (`PROBES`; `tx_longhaul_t20c60` is unverified under the 225 kWh ring). Reset each probe on 3 validation seeds. If one fails to generate, change it (for example to `t20c45`) before `prepare`.
7. Run `pytest -q tests/unit/test_generalist_*.py` and `pytest -q tests/integration/test_generalist_suite_policy.py`.

## B. Mini check (login node, about 10 min, CPU)

`prepare --directory results/generalist_graphppo/mini_check --seeds 1`. Then shrink the manifest:
- seeds: train 10, validation 4, test 2; `reference_shard` 5; `schedule_timesteps` 10240;
- common flags: `--num-envs 5 --rollout-steps 8 --minibatch-size 20 --hidden-dim 32 --encoder-layers 1 --bc-episodes 1 --pretrain-epochs 1 --train-scenarios 10 --validation-scenarios 1 --validate-every 1`;
- runs: keep only `full_s0`.

Run `references --task 0..14`, `merge-references`, `train --run 0 --budget 80 --device cpu`, `evaluate --split validation --tag mini --count 2`, and `report`.

**Gate:** `report.md` has rows for all 5 problems and all 4 probes, with teacher and extra baselines plus `full`.

## P0. References (CPU, about 18 CPU-h)

```
D=results/generalist_graphppo/suite_v1
prepare --directory $D --arm-set screen --seeds 3
submit --directory $D --stage references    # 45 tasks x 24 cores on compute-p1, 4 h limit
merge-references --directory $D
```

**Gates:**
- all 45 shards are present;
- `references/summary.json` is close to the table above;
- every problem has at least 64 non-failed train episodes (BC takes the first 64 in seed order).

## P1. GPU pilot (one allocation, about 1 h). Measures what the login node could not

Create `D1=${D}_pilot` with `--seeds 1 --references-from $D/references` and keep run `full_s0`. Run `submit --stage train --budget 200000`.

**Gates and measurements**, from the log and the histories:
- the CUDA path runs, and BC, validation and checkpoint all happen;
- BC coverage `misses` are at most 0.5 % of teacher decisions (`pretrain_history.json`);
- throughput is at least 150 decisions/s (about 250 expected);
- peak GPU memory fits. The risk is nonlinear edge messages at minibatch 256 with about 114 padded nodes, estimated at 15 GB or more.

If memory fails, set `--minibatch-size 128` in `COMMON`. If throughput is below 100 decisions/s, profile the worker step before going on.

Rewrite the `submit` time limits (`hours` mapping) and the compute table in the design doc from the measured rate. Because `prepare` is immutable and `COMMON` lives in the manifest, any flag change means re-preparing `suite_v1` before P2: delete it while nothing has been trained, and re-copy the references.

## P2. Screen (12 runs: `base`, `fleet`, `paired`, `full` × seeds 0–2; BC + 2M)

```
submit --directory $D --stage train --budget 2000000
submit --directory $D --stage evaluate --split validation --tag 2M     # 128 seeds per problem
report --directory $D --split validation --tag 2M
```

**Promotion gate**, using arm means over three seeds on 128 validation seeds per problem:
- eligible: on every problem, served fraction is at least the teacher's and failure rate at most the teacher's;
- promote the two eligible arms with the highest mean selection score.

**If no arm is eligible:** run one diagnostic iteration only. Check BC holdout accuracy, entropy and value loss per problem, then choose **one** of: `--bc-episodes 256`, entropy 0.02, or `--charger-budget 64` (the last if failures are charging strandings on TX or PNW). Put it in a new directory `suite_v1b` with `--references-from`, and re-run P2 for the best two arms only. If that also fails, write up and stop; do not tune further.

## P3. Promotion (two arms × three seeds, 2M → 8M, continuing from `training.pt`)

```
submit --stage train --budget 8000000 --arms <A1> <A2>
evaluate --tag 8M --runs <their runs>
report --tag 8M
```

Stop an arm early if its in-training validation score (16 seeds) has not improved between 2M and 4M. The **winner recipe** is the eligible arm with the best 8M validation score.

## P4. Generalization (30 runs at 8M, about 300 GPU-h, three waves at concurrency 12)

```
prepare --directory ${D}_lopo --arm-set lopo        --recipe <winner> --seeds 3 --references-from $D/references
prepare --directory ${D}_spec --arm-set specialists --recipe <winner> --seeds 3 --references-from $D/references
```

Submit train at 8M in both, then `evaluate --tag 8M` and `report`.

**Gates:**
- **Generalist vs specialist:** on every problem, the P3 winner is within 2 points of served fraction and 5 % of hours of the specialist.
- **LOPO (zero-shot):** the served fraction on the held-out problem is at least the teacher's. Report each held-out problem separately.

Keep at most 12 GPUs busy across all arrays.

## P5. Locked test (ask the user first)

After explicit approval, run once in each directory: `evaluate --split test --tag final` (500 seeds per problem plus the probes; baselines are computed there) and `report --split test --tag final`. The methods are:
- the P3 winner;
- the specialists;
- the LOPO runs, for their held-out problem;
- the teacher, `wait_at_depot` and `kullman_myopic`.

Write `docs/campaigns/generalist_graphppo_suite_results.md` with these tables:
- per problem: served fraction, success with Wilson interval, failures, hours, Δhours vs teacher with bootstrap interval, and score;
- zero-shot LOPO and size probes, kept separate;
- decision latency.

Make no design change after test is opened.

## Monitoring and failure handling

- Poll `squeue -u $USER` and `sacct -j <job>` at intervals matched to the job (30–60 min for training); do not busy-wait.
- Logs: `$D/logs/<run>_<budget>.log` and `slurm-%A_%a.out`; progress in `training/<run>/validation_history.json`. Submissions go to `logs/submissions.jsonl`.
- Re-submitting the same stage skips completed runs (`completed_run`) and resumes partial ones from `training.pt`. In-flight episodes restart from their own seed, so a resume is not bit-identical to an uninterrupted run.
- `submit --stage train` refuses to run before `merge-references`.

## Invariants (do not break)

- Decisions use validation only. Test seeds never enter training, references, or anything before P5.
- Worker processes fork before any CUDA use. If you touch `train_generalist_ppo.py` or `evaluate`, keep the env and replay construction ahead of the trainer and policy creation.
- `schedule_timesteps` = 8M is frozen in the manifest; the learning rate anneals to 0 at 8M, so do not extend past 8M in these directories.
- The reward weights (2 / 1 / 0.5 / 1 / 2 / 2) are fixed a priori. The episode return equals `SuiteReward.score` exactly (tested).
- Arms differ only in their flags. The observation width is identical, and `--fleet-assignment` only fills zero columns.


## Verification

- `pytest -q -m "not statistical" tests/unit` (701 pass as the baseline; add the new campaign tests).
- `pytest -q tests/integration/test_generalist_suite_policy.py` (2 pass).
- The mini check (§B) passes its report gate.
- The GPU pilot (P1) gates pass before P2 is submitted.
