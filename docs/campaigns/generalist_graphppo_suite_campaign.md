# Generalist GraphPPO campaign: one policy for the five suite problems

Plan date: 2026-10-22. Branch `NewFormulation` at `53a0e65`, with the generalist code in the working tree. Code, tests and a CPU pilot are done; no job has been submitted. The step-by-step runbook for the campaign agent is `generalist_graphppo_suite_runbook.md`.

**Goal.** Train one GraphPPO policy, one set of weights, that solves all five problems in `EVRoutingEnv/config_files/generalist_suite/` whatever their state and action-space sizes. Then show that it is not a bundle of five memorised specialists: hold one problem out and solve it zero-shot, and test unseen fleet and customer sizes.

**Method.** The ideas proposed in `generalist_graphppo_dayahead_plan.md` (Section 4), implemented as follows:

| Plan idea | Here | Switch | Gate |
| --- | --- | --- | --- |
| 4.1 Ragged state batching | Rows exist only for live entities; each batch is padded to its own largest graph | always on | Logits and values alone equal those inside a mixed batch (`test_generalist_policy.py`, `test_generalist_suite_policy.py`) |
| 4.1 Relational identity | No node, task, request or truck id is a feature. *Same node*, *heading to* and *owns* are edge flags, and an action points at its target's row | always on | Reordering trucks, tasks, chargers and actions permutes the logits exactly |
| 4.1 Bounded charger subgraph, K = 32 | For each goal, nearest first, the chargers take turns from two rankings: cheapest stop on the way and longest hop toward the goal. Feasible chargers come first. Chargers a truck is at or heading to are always kept | `--charger-budget` | 677 of 677 greedy-teacher charger choices are kept on the three long-haul maps. A single detour ranking kept 84 %. Every BC replay re-measures this (`pretrain_history.json`, `coverage.misses`) |
| 4.1 Physical normalization | Durations / horizon, energy / battery, loads / payload, clock times relative to now | always on | Unit tests |
| 4.1 Context node | 25 published parameters (horizon, battery, uncertainty band, HOS, utilization, outages, request mode, …), plus progress and a summary of the chargers left out; it conditions every node | always on | |
| 4.2 Fleet-assignment relations | For each task: the active truck's cost, the best other truck's cost from its committed ETA and payload, the regret, and the fraction of trucks that are closer. For depot and retire actions: unclaimed demand against the other trucks' room | `--fleet-assignment` | Arm `fleet` |
| 4.3 Day-ahead-relative reward | **Adapted.** Day-ahead ALNS and CP-SAT raise `NotImplementedError` for request modes, so 2 of the 5 problems have no day-ahead plan. The paired per-scenario reference is therefore the greedy teacher on the same seed | `--reward-reference scenario` | Arm `paired`; control: the problem's median teacher outcome |
| 4.4 Plan-conditioned GraphPPO-P | **Not in this campaign.** No plan exists for unrevealed requests. Revisit for the three static problems once the planners support the HOS and public-station blocks | | |

Kept unchanged from the plan: the heterogeneous state GNN, the complete-GCN action head, the masked categorical policy, PPO, the BC warm start and γ = 1. Not repeated, because they were screened out earlier: completion shields, progress features, capacity links, retained imitation and critic detachment.

---

## 1. The five problems

Measured on 2026-10-21 and 2026-10-22. Teacher rows: 14 development seeds per problem, not evaluation results.

| Problem | Components | Trucks | Task slots | Chargers | Env actions (extra kinds) | Canonical obs | Generalist obs (mean) | Step, ms | Greedy teacher |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ca_sameday` | Same-day requests, EURO epochs, hard deadlines, depot charging | 10 | 200 | 25 | 234 (WAIT, END_SHIFT) | 0.9 MB | 26 KB | 7 | 0/14 succeed; 7/14 fail, stranding a truck at about 35.4 h of 36 h (`no_feasible_action`); served fraction 0.86; median 276 h |
| `ca_drayage_combined` | Drayage (Long Beach, Los Angeles), US HOS (rests anywhere), public stations at u = 0.65 | 10 | 200 | 25 | 237 (QUEUE, REST_BREAK, REST_DAILY, WAIT, END_SHIFT) | 0.9 MB | 64 KB | 11 | 0/14 succeed, 0/14 fail; served fraction 0.19 (all 10 trucks for the full 36 h); median 360 h |
| `tx_longhaul` | Static, every route must charge, 72 h | 10 | 30 | 432 | 469 (none) | 3.6 MB | 54 KB | 45–65 | 14/14 succeed; served fraction 1.00; median 355 h |
| `tx_longhaul_congested` | Same instances, public stations u = 0.65, port outages | 10 | 30 | 432 | 470 (QUEUE) | 3.6 MB | 50 KB | 49–65 | 12/14 succeed, 0/14 fail; served fraction 1.00; median 409 h |
| `pnw_longhaul_hos` | Static long haul, US HOS (rests at stops only) | 10 | 30 | 323 | 362 (REST_BREAK, REST_DAILY) | 2.2 MB | 50 KB | 27–46 | 10/14 succeed, 0/14 fail; served fraction 1.00; median 512 h |

The canonical policy reads one `CanonicalShapeSpec`, so it cannot even load another problem's observation. The union envelope (10 trucks, 200 slots, 432 chargers, 644 actions) would take about 6.6 MB per decision. The generalist observation takes 26–64 KB, and no network parameter depends on any count.

---

## 2. Method

**State graph.** Three node types: every truck; every task that is still live (static customers not yet served, or revealed requests not yet served or lost); and the charger subgraph. All nine typed relations carry six edge features: nominal energy over battery capacity, nominal hours over the horizon, reachable, same node, heading to, owns. A context row conditions every node embedding.

**Actions.** The candidates are the hard-feasible actions from the one feasibility engine, minus charger destinations outside the subgraph. Each candidate row has a one-hot kind over the ten `ActionKind`s, nominal leg and lookahead hours, energy and demand fractions, and the fleet-assignment columns. It is concatenated with the embedding of its target (task or charger) and of the active truck. The policy outputs a position in the candidate list, which the worker maps back to the environment action. Extra action kinds (QUEUE, REST, WAIT, END_SHIFT) are ordinary candidates.

**Network.** Three layers of typed message passing (`_TypedMessagePassingLayer`, nonlinear edge messages, hidden 128), then a masked mean and max readout per type plus the context, then a two-layer complete-graph GCN action head and a value head. The model has 1.15 M parameters, and none of them is indexed by a count.

**Reward** (`SuiteReward`). With `H_ref` the reference operating hours and `N_ref` the reference task count (static: the customers; dynamic: the teacher's on-time services):

- each step: `−Δhours / H_ref + (2·Δserved − 0.5·Δlate − 1·Δlost) / N_ref`;
- at the end: `+1`, so that the hour terms sum to `(H_ref − H) / H_ref` as in plan 4.3; `−2` on a failure (stranding, deadlock, or the fleet stopping with work left); for static problems, `+1` on success or `−2 × (unserved fraction)`; for dynamic problems, requests still unserved at the horizon count as lost.

The episode return equals `SuiteReward.score` of the episode's summary exactly (`test_the_return_of_an_episode_is_its_score`). That score is also the validation selection metric. Retiring every truck at t = 0 scores about −3, against about +3 for the teacher on static problems, so quitting is dominated. `reference = "scenario"` uses the same seed's teacher run, which the policy never observes; `"problem"` uses the problem median.

**Training distribution.** 40 sub-environments, 8 per problem, each on a disjoint slice of 2,048 train seeds. Every rollout of 40 × 128 decisions therefore has the same share of each problem. BC first, on 64 non-failed teacher episodes per problem. The teacher is the greedy navigator for static problems and greedy dispatch for requests. Demonstrations are stored as `(problem, seed, actions)` and replayed through the current observation, so a schema change never invalidates them.

**Checkpoints** hold the policy, optimizer, normalizer, return scale, RNG and seed-stream positions, but not live simulators: a Texas simulator pickles to about 75 MB, and there are 40 of them. On resume, the episodes that were in flight restart from their own seed. Resumes are reproducible but not bit-identical to an uninterrupted run.

---

## 3. Phases, arms and gates

| Phase | Work | Budget | Gate |
| --- | --- | --- | --- |
| P0 references (CPU) | Teacher on 2,048 train and 128 validation seeds per problem: 45 array tasks of 256 seeds | about 18 CPU-h | Reference failure rates recorded (`references/summary.json`); BC uses non-failed episodes only |
| P1 implementation gates | Done locally (Section 6) | | Padding invariance, permutation equivariance, finite gradients, return = score, replay fidelity, checkpoint resume, a mixed 5-problem rollout |
| P2 screen | 4 arms × 3 seeds = 12 runs | BC + 2M decisions | Promote at most two arms: highest validation selection score, with no problem below the teacher's served fraction and no failure rate above the teacher's |
| P3 promotion | 2 arms × 3 seeds | 2M → 8M (continued, not restarted) | Per-problem trend on 128 validation seeds per problem; stop an arm whose score has not improved over 2M → 4M |
| P4 generalization | Promoted recipe: leave one problem out (5 × 3 seeds), and five specialists (5 × 3 seeds) | 8M | The generalist is within 2 points of served fraction and 5 % of hours of the specialist on every problem. A held-out problem at or above the teacher's served fraction, zero-shot, is the generalization claim |
| P5 locked test | 500 test seeds per problem, plus four size probes | evaluation only | Test once; no redesign after opening |

P2 arms (`prepare --arm-set screen`):

| Arm | Change |
| --- | --- |
| `base` | Variable state graph, relational identity, charger subgraph, normalization, context node; problem-median reward |
| `fleet` | + fleet-assignment relations (4.2) |
| `paired` | + per-scenario teacher-relative reward (4.3, adapted) |
| `full` | `fleet` + `paired` |

P4 uses `prepare --arm-set lopo --recipe <promoted>` and `--arm-set specialists`. Each is a new immutable campaign directory.

**Size probes** (evaluation only, never trained on): `tx_longhaul` with 5 trucks and 15 customers, and with 20 trucks and 60 customers; `ca_sameday` with 5 and with 20 trucks. Each probe is written into the campaign's `configs/`.

### Evaluation (`evaluate`, then `report`)

Every method runs on identical seeds per problem: the generalist arms, the greedy teacher, `wait_at_depot` (dynamic problems) and `kullman_myopic` (public-station problems). Reported per problem:

- served fraction;
- success with a Wilson interval;
- failure rate;
- mean operating hours;
- paired Δhours against the teacher, with a bootstrap interval. Static problems use jointly successful seeds; dynamic problems use all seeds;
- the score, the teacher-relative return on the same seed (higher is better).

Lexicographic reading, as in the formal model (Section 6): on-time services, or success for static problems, first; lost and late requests next; operating hours last.

---

## 4. Compute envelope

These are measured rates, not guesses. On a login node with 8 rollout workers and CPU inference, rollouts ran at about 90 decisions/s, and one rollout of 5,120 decisions holds about 1 GB of padded float16 pairwise tensors. Steps cost about 45–65 ms on Texas, 27–46 ms on the Pacific Northwest and 7–11 ms on California, and every problem steps in parallel. On one DelftBlue A100 share (16 cores, so 14 rollout workers), I expect about 250 decisions/s. **Measure this in the first hour of P2 before committing to P3.**

| Item | Estimate |
| --- | --- |
| P0 references: 10.9k teacher episodes × about 6 s | about 18 CPU-h (45 tasks × 24 cores, under 1 h wall) |
| P2: 12 runs × (BC + 2M) | about 12 × 3 = 36 GPU-h |
| P3: 6 runs, 2M → 8M | about 6 × 7 = 42 GPU-h |
| P4: 30 runs to 8M | about 30 × 10 = 300 GPU-h |
| P5: 9 problem configs × 500 seeds × (arms + 3 baselines) | about 40 CPU-h + 5 GPU-h |

SLURM settings live in the manifest's `slurm` block. The defaults target DelftBlue: account `research-eemcs-ese`, `gpu-a100` (16 CPUs and 120 GB per GPU) and `compute-p1`, with at most 12 concurrent tasks. Edit that block before `submit` to run on Snellius (`gpu_a100`, account `tdsei17344`).

---

## 5. Risks and stopping rules

- **The teacher is weak on the two California request problems.** It serves 9–16 of 65 drayage requests and strands trucks in same-day. BC starts low there and PPO has to do the lifting. Because the dynamic score is normalized by the teacher's services, gains there show up as large numbers. Watch the value loss per problem, and compare served counts, not scores, across problems.
- **The charger subgraph is gated on teacher choices only.** A learned policy may want a charger the teacher never picks. If P2 failures concentrate on charging (strandings on Texas or the Pacific Northwest), rerun the best arm with `--charger-budget 64`; it is an observation-only change.
- **Mixed training may dilute per-problem quality.** The P4 specialists measure this. If the gate fails, report the specialists per problem and the generalist for transfer.
- **The reward weights are fixed a priori** (2 / 1 / 0.5 / 1 / 2 / 2) and never tuned on test.
- **Workers fork a process that has already imported torch** (Python warns about forking with threads). The canonical trainer does the same, and the workers never call torch.

---

## 6. P1 gates (passed locally)

| Test | Checks |
| --- | --- |
| `tests/unit/test_generalist_observation.py` | Candidates are exactly the hard mask outside charger destinations, for static, public, HOS and same-day configs; action links are valid; there are no identifier features; a large budget offers every feasible charger; collate keeps the edge blocks |
| `tests/unit/test_generalist_policy.py` | Padding invariance across mixed problems; permutation equivariance; every parameter gets a finite gradient; out-of-set actions are refused; checkpoints round-trip; another schema is refused |
| `tests/unit/test_generalist_ppo.py` | Return = score in both reference modes; paired references with median fallback; disjoint, balanced seed slots; teacher replay reproduces every action; evaluation runs every seed once; checkpoint and resume |
| `tests/unit/test_generalist_campaign.py` | `prepare` freezes problems, arms, probes and the source; LOPO and specialist arms; reference shards cover train and validation only; sbatch composition |
| `tests/integration/test_generalist_suite_policy.py` | One policy scores all five real problems in one batch, with batch-invariant logits; a mixed five-problem rollout trains through forked workers and evaluates |

---

## 7. Commands

```bash
D=results/generalist_graphppo/suite_v1
PYTHONPATH=. .venv/bin/python scripts/campaigns/generalist_graphppo.py prepare --directory $D --arm-set screen
PYTHONPATH=. .venv/bin/python scripts/campaigns/generalist_graphppo.py submit --directory $D --stage references   # --dry-run to print only
PYTHONPATH=. .venv/bin/python scripts/campaigns/generalist_graphppo.py merge-references --directory $D
PYTHONPATH=. .venv/bin/python scripts/campaigns/generalist_graphppo.py submit --directory $D --stage train --budget 2000000
PYTHONPATH=. .venv/bin/python scripts/campaigns/generalist_graphppo.py evaluate --directory $D --split validation
PYTHONPATH=. .venv/bin/python scripts/campaigns/generalist_graphppo.py report --directory $D --split validation
# P3: the same runs continue from training.pt
PYTHONPATH=. .venv/bin/python scripts/campaigns/generalist_graphppo.py submit --directory $D --stage train --budget 8000000
```

Files: `algo/generalist_observation.py` (view, subgraph, batching), `algo/generalist_policy.py` (network), `algo/generalist_ppo.py` (reward, references, workers, PPO, BC replay, evaluation), `scripts/training/train_generalist_ppo.py` (trainer CLI), `scripts/campaigns/generalist_graphppo.py` with `generalist_graphppo_{gpu,cpu}.sbatch` (campaign driver).

## 8. What this campaign will not claim

- Superiority over day-ahead planners on the request problems: they are not defined there.
- Generalization beyond the EVRP state and action semantics, or to components absent from the training mixture. A held-out problem still shares its components with the others, for example HOS appears in both `ca_drayage_combined` and `pnw_longhaul_hos`.
- That a size probe is solved because a training problem is.
