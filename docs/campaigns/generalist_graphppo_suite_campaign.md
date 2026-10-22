# Generalist GraphPPO campaign: one near-optimal policy for the five suite problems

Plan date: 2026-10-22 (revised the same day for the near-optimal gate and Snellius). Branch `NewFormulation` at `72e3d43` plus the pre-flight code in the working tree. Code and tests are done; no job has been submitted. The step-by-step runbook for the campaign agent is `generalist_graphppo_suite_runbook.md`; its "User decisions" table is binding.

**Goal.** Train one GraphPPO policy, one set of weights, that is **near-optimal on all five problems** in `EVRoutingEnv/config_files/generalist_suite/`, whatever their state and action-space sizes. Near-optimal is the gate of `scripts/campaigns/generalist_near_optimal_gate.py`, measured against best-known solutions (BKS) on the same seeds: static problems need a failure rate at most max(1 %, the BKS's), a success rate at least the BKS's minus 2 points and a mean paired operating-hours excess of at most 5 %; dynamic problems need the failure bound and at least 95 % of the BKS's on-time services. The campaign succeeds when one run passes on every problem on validation and then, once, on the locked test bank.

**Method.** The ideas proposed in `generalist_graphppo_dayahead_plan.md` (Section 4), plus imitation of the BKS planners, implemented as follows:

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
| BKS demonstrations | Behaviour cloning starts from the best-known solution of each train seed (256 per problem, replay-verified) instead of the greedy teacher's, and the paired reward normalizes by the BKS outcome of the seed | `--demonstrations`, `--reward-references`, `--bc-episodes 256` | Arm `full_bks`; BC coverage `misses` ≤ 0.5 % in the pilot |
| Retained imitation | The BKS cross-entropy stays in every PPO minibatch, 0.1 → 0 over 2M decisions | `--bc-coefficient 0.1 --bc-coefficient-final 0 --bc-anneal-timesteps 2000000` | Arm `full_bks_retained` |
| DAgger with planner experts | Every 10 updates, 8 episodes per problem are driven by a mixture of the learner and the planner (β 0.5 → 0 over 2M); every visited state is labelled by the planner and joins the retained set; two such rounds precede PPO. Experts: the long-haul planner at a reduced budget (static) and the rolling request planner (dynamic), `algo/generalist_experts.py` | `--dagger-expert ... --dagger-every 10 --dagger-episodes 8 --dagger-beta 0.5 --dagger-beta-final 0 --dagger-beta-anneal-timesteps 2000000 --dagger-pretrain-rounds 2` | Arm `full_dagger` (keeps the retained term: DAgger labels reach the PPO loss only through it); round time measured in the pilot |

Kept unchanged from the plan: the heterogeneous state GNN, the complete-GCN action head, the masked categorical policy, PPO, the BC warm start and γ = 1. Not repeated, because they were screened out earlier on California: completion shields, progress features, capacity links and critic detachment. Retained imitation returns here because its labels now come from planners rather than from the greedy teacher.

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

### Best-known solutions

The bar is a portfolio of independent, non-learning controllers executed closed loop in the simulator on the same seed; the best outcome per seed (not failed, then success, then services, then fewer lost, then fewer late, then fewer hours) is the BKS. It is a best-known reference, not a proven optimum.

| Problems | Portfolio (`scripts/campaigns/generalist_bks_*.py`) | Bound per seed |
| --- | --- | --- |
| `tx_longhaul`, `tx_longhaul_congested`, `pnw_longhaul_hos` | greedy teacher; day-ahead ALNS with the plan-respecting executor; the long-haul ALNS plan executed through the continuation search (40 s × 3 restarts), its rolling re-planning variant and a safer-margin variant | CP-SAT nominal lower bound on operating hours (120 s), `proven` when it closes |
| `ca_sameday`, `ca_drayage_combined` | greedy dispatch; wait at depot; Kullman's myopic station rules (drayage only); the rolling-horizon request planner (ALNS re-optimization of every revealed request at each decision), never clairvoyant | the same planner made clairvoyant at a raised ALNS budget (its cap of four iterations per pending job), floored at the BKS of the seed (user decision 2026-10-22: at full scale the clairvoyant planner can serve less than the causal one; `raised_to_bks` and `clairvoyant_below_bks_seeds` report how often), a heuristic hindsight estimate (`proven: false`); `check` confirms it never beats the exact search `solve_request_hindsight` on tiny one-truck instances (one to three requests, 12 h; a three-request instance already exceeds 20k search nodes). The planner's drayage model never starts a daily rest while a known request is due before the rest would end (fixed 2026-10-22; before that the clairvoyant bound rested on top of known requests) |

Files per problem: `results/generalist_graphppo/bks/<p>/validation.json` (128 seeds), `train_demonstrations.json` (256 train seeds, replay-verified) and `summary.json`; `test.json` is built only when P5 opens. `prepare --bks-from` freezes a copy into the campaign for the imitation arms.

---

## 2. Method

**State graph.** Three node types: every truck; every task that is still live (static customers not yet served, or revealed requests not yet served or lost); and the charger subgraph. All nine typed relations carry six edge features: nominal energy over battery capacity, nominal hours over the horizon, reachable, same node, heading to, owns. A context row conditions every node embedding.

**Actions.** The candidates are the hard-feasible actions from the one feasibility engine, minus charger destinations outside the subgraph. Each candidate row has a one-hot kind over the ten `ActionKind`s, nominal leg and lookahead hours, energy and demand fractions, and the fleet-assignment columns. It is concatenated with the embedding of its target (task or charger) and of the active truck. The policy outputs a position in the candidate list, which the worker maps back to the environment action. Extra action kinds (QUEUE, REST, WAIT, END_SHIFT) are ordinary candidates.

**Network.** Three layers of typed message passing (`_TypedMessagePassingLayer`, nonlinear edge messages, hidden 128), then a masked mean and max readout per type plus the context, then a two-layer complete-graph GCN action head and a value head. The model has 1.15 M parameters, and none of them is indexed by a count.

**Reward** (`SuiteReward`). With `H_ref` the reference operating hours and `N_ref` the reference task count (static: the customers; dynamic: the teacher's on-time services):

- each step: `−Δhours / H_ref + (2·Δserved − 0.5·Δlate − 1·Δlost) / N_ref`;
- at the end: `+1`, so that the hour terms sum to `(H_ref − H) / H_ref` as in plan 4.3; `−2` on a failure (stranding, deadlock, or the fleet stopping with work left); for static problems, `+1` on success or `−2 × (unserved fraction)`; for dynamic problems, requests still unserved at the horizon count as lost.

The episode return equals `SuiteReward.score` of the episode's summary exactly (`test_the_return_of_an_episode_is_its_score`). That score is also the validation selection metric. Retiring every truck at t = 0 scores about −3, against about +3 for the teacher on static problems, so quitting is dominated. `reference = "scenario"` uses the same seed's teacher run, which the policy never observes; `"problem"` uses the problem median.

**Training distribution.** 40 sub-environments, 8 per problem, each on a disjoint slice of 2,048 train seeds. Every rollout of 40 × 128 decisions therefore has the same share of each problem. BC first, on 64 non-failed teacher episodes per problem for the four original arms and on 256 BKS episodes per problem for the imitation arms. The teacher is the greedy navigator for static problems and greedy dispatch for requests. Demonstrations are stored as `(problem, seed, actions)` and replayed through the current observation, so a schema change never invalidates them.

**Imitation arms.** `full_bks` clones the BKS demonstrations and pairs its reward with the BKS outcome of the seed (`ReferenceTable` reads the BKS folder: the 256 demonstration seeds and the 128 validation seeds are exactly paired, the other train seeds fall back to the problem's BKS median, so selection and the gate are fully paired while training is partly so). `full_bks_retained` keeps the BKS cross-entropy in every PPO minibatch with a coefficient annealed from 0.1 to 0 over 2M decisions. `full_dagger` adds DAgger rounds: the long-haul planner (8 s, 1500 iterations, one restart, re-planning on events and deviations) labels the static problems, the rolling request planner the dynamic ones; the learner's own actions are executed with probability 1 − β. Both experts re-plan from whatever state they are handed.

**Checkpoints** hold the policy, optimizer, normalizer, return scale, RNG and seed-stream positions, but not live simulators: a Texas simulator pickles to about 75 MB, and there are 40 of them. On resume, the episodes that were in flight restart from their own seed. Resumes are reproducible but not bit-identical to an uninterrupted run.

---

## 3. Phases, arms and gates

| Phase | Work | Budget | Gate |
| --- | --- | --- | --- |
| P0a BKS (CPU) | Portfolio on 128 validation and 256 train seeds per problem, bounds, demonstrations | about 115 CPU-h, 10 allocations of 1–3 h | Every problem has `validation.json` and ≥ 200 non-failed demonstrations; the BKS beats the teacher on every problem |
| P0b references (CPU) | Teacher on 2,048 train and 128 validation seeds per problem | about 18 CPU-h | `references/summary.json`; ≥ 64 non-failed train episodes per problem |
| P1 pilot (GPU) | `full_s0` and `full_dagger_s0` to 200k decisions in one allocation | 1 h | CUDA path, ≥ 150 decisions/s, memory, BC coverage ≤ 0.5 % misses, DAgger rounds ≤ 30 % of wall time |
| P2 screen | 7 arms × 3 seeds = 21 runs | BC + 2M | Promote the **three** eligible arms with the highest mean selection score; eligible: on every problem served fraction ≥ the teacher's and failure rate ≤ the teacher's (`report --promote 3`) |
| P3 promotion | 3 arms × 3 seeds | 2M → 8M (continued) | Stop an arm whose score has not improved over 2M → 4M. **Success: a run with `overall: PASS` in the near-optimal gate on 128 validation seeds** |
| P5 locked test | BKS on 500 test seeds per problem, then the passing arms' seeds | evaluation only | Opened automatically on a validation PASS; test once; no redesign after opening |
| `suite_v2` (only if no PASS at 8M) | The promoted arms with one flag change each, chosen from the 8M diagnosis | BC + 2M → 8M | One iteration, then stop and write up |

P2 arms (`prepare --arm-set screen --bks-from B`):

| Arm | Change |
| --- | --- |
| `base` | Variable state graph, relational identity, charger subgraph, normalization, context node; problem-median reward; BC from the greedy teacher |
| `fleet` | + fleet-assignment relations (4.2) |
| `paired` | + per-scenario teacher-relative reward (4.3, adapted) |
| `full` | `fleet` + `paired` |
| `full_bks` | `full` + BKS demonstrations (256 per problem) + BKS-paired reward |
| `full_bks_retained` | `full_bks` + retained BKS cross-entropy, 0.1 → 0 over 2M |
| `full_dagger` | `full_bks_retained` + DAgger with the planner experts (β 0.5 → 0 over 2M, two pretrain rounds) |

Dropped from this campaign: the leave-one-problem-out and specialist arms (P4) and the four size probes. `prepare --probes` still writes the probe configurations for a later campaign; `prepare` is immutable and `suite_v2` is a new directory.

### Evaluation (`evaluate`, then `report`)

Every method runs on identical seeds per problem: the generalist arms, the greedy teacher, `wait_at_depot` (dynamic problems) and `kullman_myopic` (public-station problems). Reported per problem:

- served fraction;
- success with a Wilson interval;
- failure rate;
- mean operating hours;
- paired Δhours against the teacher, with a bootstrap interval. Static problems use jointly successful seeds; dynamic problems use all seeds;
- the score, the teacher-relative return on the same seed (higher is better).

Lexicographic reading, as in the formal model (Section 6): on-time services, or success for static problems, first; lost and late requests next; operating hours last.

Then `report --promote N` writes `promotion.json` (arm means over seeds, eligibility, the top N) and `generalist_near_optimal_gate.py` writes `near_optimal_gate.{json,md}` per tag: for every run and problem, the three criteria against the BKS of the same seeds, with a paired bootstrap interval on the hours excess and the hindsight bound as headroom for the dynamic problems.

---

## 4. Compute envelope (Snellius)

Measured rates: on a login node with 8 rollout workers and CPU inference, rollouts ran at about 90 decisions/s; steps cost about 45–65 ms on Texas, 27–46 ms on the Pacific Northwest and 7–11 ms on California. On one gpu_a100 share (18 cores, so 16 rollout workers) about 250 decisions/s is expected. **P1 measures this, and the DAgger round time, before P2.**

BKS method costs per episode, measured in the mini check on 2026-10-22 (two validation seeds per problem, login node, one episode per core):

| Method | Texas long haul | Texas congested | Pacific Northwest HOS | Same-day | Drayage |
| --- | --- | --- | --- | --- | --- |
| Greedy teacher | 7 s | 6–9 s | 3–6 s | 0.7–0.8 s | 1.8–2.1 s |
| `lh_plan` (40 s × 3 restarts + continuation) | 45–49 s | 41–42 s | 82–87 s | | |
| Rolling request planner | | | | 1–6 s | 26–30 s |
| Clairvoyant rolling planner (bound), default budget | | | | 28–79 s | 82–121 s |
| Clairvoyant rolling planner at the raised budget (the bound as recorded) | | | | about 5× | 380–510 s |

On those two seeds the planners are far ahead of the teacher: `lh_plan` finished Texas in 110–122 operating hours against the greedy teacher's 305–377, and the Pacific Northwest in 143–171 against 318 (and one greedy failure); the rolling planner served 34 and 26 same-day requests against greedy dispatch's 26 and 19, and 37 and 32 drayage requests against 9 and 11 (Kullman's rules: 22 and 23). The clairvoyant same-day bound served 35 and 26 on the same seeds, so the heuristic bound sits almost on the BKS there. On drayage, after the rest-rule fix, the causal planner served 35 and 34 (3 and 1 late) and the clairvoyant one 30 and 29: with the whole 65-request stream in view the planner's 60-iteration ALNS does worse than the causal planner's small revealed problems, so the heuristic bound sits below the BKS there (`bound.exceeded_seeds`); the tiny-instance check passes 10 of 10 with hours of service on. The static nominal lower bounds were 42–126 % below the BKS hours on those seeds (nominal costs, no realized draws, no rests), so they bound loosely and the gate's margin is relative to the BKS, not to them. These are development samples; P0a measures the full 128-seed bar. The gate therefore asks the policy for roughly a third of the teacher's hours on the static problems and 30–40 % more services than the teacher on the dynamic ones.

| Item | Estimate |
| --- | --- |
| P0a BKS: 3 static problems × 384 seeds × 3 long-haul methods × ~1 min, plus ALNS and the 120 s bounds; 2 dynamic problems × 384 seeds × 4 methods × ≤ 30 s, plus the clairvoyant bounds at the raised budget (128 seeds × up to ~500 s per problem) | about 70 CPU-h static and 45 CPU-h dynamic; 10 allocations of one GPU share, about 1–3 h wall each |
| P0b references: 10.9k teacher episodes × about 6 s | about 18 CPU-h; 3 allocations, about 20 min |
| P1 pilot | 1 allocation, 2 GPUs, about 1 h |
| P2: 21 runs × (BC + 2M) | about 21 × 4 = 85 GPU-h in one wave (throttle 20) |
| P3: 9 runs, 2M → 8M | about 9 × 8 = 72 GPU-h |
| Evaluations: 2M, 8M (128 seeds per problem, 3 allocations each) | about 20 GPU-h |
| P5: BKS on 500 test seeds per problem (about 67 CPU-h per static problem), then one evaluation | about 25 GPU-share-h |
| `suite_v2` (only if needed) | about 100 GPU-h |

About 300 GPU-allocation-hours in all, roughly 40k SBU at 128 SBU per A100-hour (403k SBU remained on 2026-10-22). Every job on a GPU partition holds a GPU, so the CPU stages leave theirs idle; the alternative, `cbuild`, is 32 cores per node and was saturated. SLURM settings live in the manifest's `slurm` block: account `tdsei17344`, `gpu_a100` for every stage, 18 cores and 6500 MB per core per GPU share, overflow to `gpu_h100` (16 cores) and `gpu_mig` (9) with `submit --partition`, at most 20 concurrent allocations.

## 5. Risks and stopping rules

- **The teacher is weak on the two California request problems.** It serves 9–16 of 65 drayage requests and strands trucks in same-day. The BKS there is the rolling planner, whose quality is unmeasured before P0a; if it is no better than greedy dispatch, the gate is trivially easy on those problems and `summary.json` (`best_method_counts`, served ratio) says so. Report that rather than tuning the planner on validation.
- **The clairvoyant bound is a heuristic.** The BKS may beat it on some seeds (`bound.exceeded_seeds`); the gate reads it as headroom only, and `check` keeps it honest on reduced instances where the exact search exhausts.
- **DAgger rounds run synchronously** inside the trainer every 10 updates (about 39 rounds at 2M). Planner experts cost seconds per decision on Texas and the Pacific Northwest. The pilot measures the round time; above about 30 % of wall time the user decides, because the arm's schedule is a decision.
- **BKS-paired rewards are only partly paired in training** (256 of 2,048 train seeds). Selection and the gate are fully paired.
- **The charger subgraph is gated on teacher choices only.** If failures concentrate on charging (strandings on Texas or the Pacific Northwest), `suite_v2` uses `--charger-budget 64`; it is an observation-only change.
- **Mixed training may dilute per-problem quality.** Without the specialist arms this is not measured here; a gate failure on one problem is reported per problem.
- **The reward weights (2 / 1 / 0.5 / 1 / 2 / 2) and the gate thresholds are fixed a priori** and never tuned on test.
- **Workers fork a process that has already imported torch** (Python warns about forking with threads). The canonical trainer does the same, and the workers never call torch; the planner experts do not either.
- **Snellius was rebooting on 2026-10-22** (every partition down, no reservation). Nothing is submitted until `sinfo` shows the partitions up, and every composed command passes `sbatch --test-only` first.

## 6. P1 gates (passed locally)

| Test | Checks |
| --- | --- |
| `tests/unit/test_generalist_observation.py` | Candidates are exactly the hard mask outside charger destinations, for static, public, HOS and same-day configs; action links are valid; there are no identifier features; a large budget offers every feasible charger; collate keeps the edge blocks |
| `tests/unit/test_generalist_policy.py` | Padding invariance across mixed problems; permutation equivariance; every parameter gets a finite gradient; out-of-set actions are refused; checkpoints round-trip; another schema is refused |
| `tests/unit/test_generalist_ppo.py` | Return = score in both reference modes; paired references with median fallback; disjoint, balanced seed slots; teacher replay reproduces every action; evaluation runs every seed once; checkpoint and resume |
| `tests/unit/test_generalist_campaign.py` | `prepare` freezes problems, the seven arms, the BKS copy and the source; probes only on request; the imitation arms' flags override the common ones; Snellius sbatch composition (every job holds a GPU, throttle 20, overflow partitions, `--test-only`); promotion |
| `tests/unit/test_generalist_experts.py` | The long-haul expert re-plans after the learner deviates; the rolling expert drives a same-day episode; both resolve through `--dagger-expert` for every suite problem |
| `tests/unit/test_generalist_bks.py` | Outcome order; merge into the gate's format with both bound kinds; the test split is locked; the reduced instances keep the request envelope; the clairvoyant planner never beats the exact hindsight search |
| `tests/unit/test_generalist_near_optimal_gate.py` | Static and dynamic pass and fail criteria; the test split needs `--allow-test` |
| `tests/integration/test_generalist_suite_policy.py` | One policy scores all five real problems in one batch, with batch-invariant logits; a mixed five-problem rollout trains through forked workers and evaluates |

---

## 7. Commands

```bash
P=.venv/bin/python; B=results/generalist_graphppo/bks; D=results/generalist_graphppo/suite_v1
# P0a: best-known solutions (plan files per problem and split; see the runbook)
PYTHONPATH=. $P scripts/campaigns/generalist_bks_static.py snapshot --name v1
sbatch --job-name bks_tx_longhaul_val scripts/campaigns/generalist_bks.sbatch $B/_source/v1 static $B/plans/tx_longhaul_val.txt $B
sbatch --job-name bks_ca_sameday_val scripts/campaigns/generalist_bks.sbatch $B/_source/v1 dynamic $B/plans/ca_sameday_val.txt $B
# P0b: references
PYTHONPATH=. $P scripts/campaigns/generalist_graphppo.py prepare --directory $D --arm-set screen --seeds 3 --bks-from $B
PYTHONPATH=. $P scripts/campaigns/generalist_graphppo.py submit --directory $D --stage references --pack 3   # --test-only first
PYTHONPATH=. $P scripts/campaigns/generalist_graphppo.py merge-references --directory $D
# P2 and P3
PYTHONPATH=. $P scripts/campaigns/generalist_graphppo.py submit --directory $D --stage train --budget 2000000
PYTHONPATH=. $P scripts/campaigns/generalist_graphppo.py submit --directory $D --stage evaluate --split validation --tag 2M
PYTHONPATH=. $P scripts/campaigns/generalist_graphppo.py report --directory $D --split validation --tag 2M --promote 3
PYTHONPATH=. $P scripts/campaigns/generalist_near_optimal_gate.py --campaign $D --split validation --tag 2M --bks-dir $B
PYTHONPATH=. $P scripts/campaigns/generalist_graphppo.py submit --directory $D --stage train --budget 8000000 --arms A B C
# P5 (opened automatically on a validation PASS): BKS on test seeds with --allow-test, then
PYTHONPATH=. $P scripts/campaigns/generalist_graphppo.py submit --directory $D --stage evaluate --split test --tag final --runs ...
PYTHONPATH=. $P scripts/campaigns/generalist_near_optimal_gate.py --campaign $D --split test --tag final --bks-dir $B --allow-test
```

Files: `algo/generalist_observation.py` (view, subgraph, batching), `algo/generalist_policy.py` (network), `algo/generalist_ppo.py` (reward, references, workers, PPO, BC replay, evaluation), `algo/generalist_imitation.py` (demonstrations, retained imitation, DAgger), `algo/generalist_experts.py` (planner experts), `scripts/training/train_generalist_ppo.py` (trainer CLI), `scripts/campaigns/generalist_graphppo.py` with `generalist_graphppo_{gpu,cpu,eval}.sbatch` (campaign driver), `scripts/campaigns/generalist_bks_{common,static,dynamic}.py` with `generalist_bks.sbatch` (best-known solutions), `scripts/campaigns/generalist_near_optimal_gate.py` (the gate).

## 8. What this campaign will not claim

- Optimality. The BKS is a portfolio of heuristics and planners; the static lower bound is nominal (deterministic costs) and the dynamic bound is a clairvoyant heuristic. "Near-optimal" means within the gate's margins of the best-known solution.
- Superiority over day-ahead planners on the request problems: they are not defined there.
- Zero-shot generalization to a held-out problem or to unseen fleet and customer sizes: the leave-one-problem-out arms and the size probes are not run here.
- Generalization beyond the EVRP state and action semantics, or to components absent from the training mixture.

---

## 9. Out-of-the-box solutions: ideas to keep improving performance

This part is an idea bank, not a commitment. The frozen plan is Sections 3–5; the user decisions in the runbook stay binding. Use this list when a gate fails or when a problem lags: pick **one** idea per arm for `suite_v2`, keep every choice on validation, record it in `agent/status.log`, and never look at the test bank to choose. Each entry says what evidence should trigger it, where it plugs in, roughly what it costs, and what could go wrong.

### 9.1 Read the gap before changing anything

Every improvement below targets a specific loss, so first decompose the gap per problem from the evaluation rows and the BKS rows on the same seeds (`evaluation/validation/<tag>/<run>.json` against `bks/<p>/validation.json`):

| Diagnostic | How | What it tells you |
| --- | --- | --- |
| Hours decomposition | Paired difference of `travel_hours`, charging hours, queue and wait hours, service hours (fields of `episode_summary` / `extract_operational_metrics`) | On California 10/30 the whole excess was travel and fleet activation (5.7 trucks vs 4.3), not charging; expect the same on the long-haul problems unless strandings appear |
| Fleet activation | Trucks that left the depot, per seed, policy vs BKS | Over-activation is a cheap, large loss; it points at the depot/retire action, fleet-assignment features and the reward's hour scale |
| Failure categories | `termination_reason` counts and the state at failure | Strandings → charging (charger budget, reserve features); deadlocks → masks; `no_feasible_action` near the horizon on same-day → end-of-shift handling |
| Imitation fit | `pretrain_history.json` holdout accuracy, BC coverage misses, DAgger agreement per problem | Low holdout accuracy on one problem means the observation cannot express the expert's rule there (fix features) rather than that PPO is weak |
| Where the policy already beats the BKS | Seeds with negative excess | Those trajectories are new demonstrations (9.3, self-imitation) |

### 9.2 Cheap knobs (one flag, no new code)

| Idea | Trigger | Change | Cost / caveat |
| --- | --- | --- | --- |
| Wider charger subgraph | Strandings or charging detours on Texas / Pacific Northwest; BC coverage misses on charger actions | `--charger-budget 64` (observation only) | More padded nodes per decision, roughly +40 % GPU memory |
| Longer imitation anneal | Imitation arms regress once the BC term reaches zero | `--bc-coefficient-final 0.02`, or `--bc-anneal-timesteps 4000000` | Policy stays closer to the planner; may cap the gain where the planner is beatable |
| Keep the expert in the loop longer | DAgger agreement collapses after β reaches zero | `--dagger-beta-final 0.2`, `--dagger-beta-anneal-timesteps 4000000` | More expert time per round |
| More BKS demonstrations | Holdout accuracy still rising at 256 episodes | BKS on 512 or 1,024 train seeds (CPU only), `--bc-episodes 512` | Also widens the exactly paired seeds of the BKS reward |
| Exploration | Entropy collapses early (training history) while the validation score is flat | `--entropy-coefficient 0.02`, or `--target-kl 0.1` | Slower convergence |
| Lower learning rate for promotion | Validation score oscillates between 2M and 8M | `--learning-rate 1e-4` in a fresh directory with `--init-from` the 2M `best.pt` | Changes the schedule, so a new directory |
| Decoding temperature | Greedy decoding loses to sampled decoding on validation | `scripts/campaigns/evaluate_california_policy_temperature.py` as the template: evaluate `best.pt` at temperatures 0.5–1.5 and choose on validation | Free at train time; must be fixed before test |

### 9.3 Stronger imitation

| Idea | Why it can help | Sketch |
| --- | --- | --- |
| **Privileged (hindsight) expert for DAgger** on the dynamic problems | The clairvoyant rolling planner (`rolling_request(..., clairvoyant=True)`) knows the hidden request stream; distilling its actions into a causal policy ("learning by cheating") teaches anticipation that no causal expert can show. The gate scores realized outcomes, so the training signal is legitimate | New expert spec `algo.generalist_experts:rolling_request_clairvoyant`; `--dagger-expert ca_sameday=... ca_drayage_combined=...`. Caveat: the policy cannot copy decisions that depend on unobservable facts; expect lower agreement and keep the causal expert for the static problems |
| **Self-imitation from the policy's own wins** | Seeds where the policy beats the BKS carry better demonstrations than any planner | After each evaluation on train seeds, write the winning episodes into `bks/<p>/train_demonstrations.json` as `policy` rows (replay-verified through `GeneralistEnv`); the validation BKS stays fixed for the gate |
| **Label with the best of several experts** | On a given state `lh_rolling`, `lh_plan_safe` and the greedy navigator disagree; the best continuation is cheap to pick with the continuation search's own cost | A composite expert in `generalist_experts.py` that queries two planners and returns the action of the cheaper continuation; DAgger declines a state when both decline |
| **Expert budget schedule** | Early DAgger rounds only need rough labels; late rounds decide the last percent | Round-dependent `PlannerSettings.time_limit_seconds` (2 s → 8 s → 20 s) through a factory that reads the round from the environment's `scenario_seed` stream or an environment variable |
| **DAgger rounds on failure states** | Failures are rare in rollouts, so the expert rarely labels the states that matter | Seed the DAgger slots with the seeds of the policy's failed validation episodes (train-namespace twins), or replay the failed episode's prefix and label from there |

### 9.4 A better training signal

| Idea | Why | Sketch |
| --- | --- | --- |
| **Asymmetric actor–critic** | The dynamic problems are partially observed; a critic that sees the hidden requests and the realized draws during training gives far less noisy advantages while the actor stays causal | A second observation for the value head built from `hindsight_problem(env)` (`request_engine.py`) and the realized legs; `GeneralistActorCritic.predict_values` reads it; the actor's input is untouched |
| **Potential-based shaping from the planner's cost-to-go** | Dense, optimal-policy-invariant credit for routing decisions; the planners already compute a nominal remaining-hours estimate from any state | `F = γΦ(s') − Φ(s)` with Φ = −(planner remaining hours / `H_ref`), computed in the workers at a reduced planner budget every k steps (interpolated in between) |
| **Prioritized scenario sampling** | Mixed training spends equal effort on seeds the policy already solves | Weight the train-seed stream per problem by the last known excess vs the BKS (from `train_demonstrations.json` and periodic train-seed evaluations); keep a floor so no seed is forgotten |
| **Regret-normalized reward** | The BKS-paired return still mixes problems with very different absolute hours | Normalize the terminal term by the BKS–teacher spread of the seed, so "closing the gap" is the unit everywhere |
| **Problem-balanced advantage normalization** | One problem's large advantages dominate the update | Normalize advantages per problem inside each minibatch (`RunningReturnScale` per problem) |
| **Checkpoint averaging** | The 8M actor is noisier than the trajectory of actors | Polyak average of the last N `best.pt` candidates, evaluated as one more candidate on validation |

### 9.5 Model and observation

| Idea | Why | Sketch |
| --- | --- | --- |
| **Plan-conditioned input (GraphPPO-P)** | The strongest signal about a good assignment is the planner's plan; the policy keeps the right to deviate | Day-ahead plan 4.4: truck → next planned customer and customer → successor relations, "on plan" and "deviation cost" per candidate, plan dropout 0.3 so the plan-free mode still works; for the dynamic problems the rolling planner's current plan plays the day-ahead role |
| **Learned charger selection instead of a ranked subgraph** | K = 32 ranked chargers is a hand-made filter | A sparse attention layer from each truck to all chargers with distance-bucket features, keeping the K best by attention as the candidate set |
| **Fleet-level context** | Over-activation is a fleet decision, but the policy decides per truck | Add fleet summaries (unclaimed demand vs active payload, planned end times) to the context node; add a "commit to retire" feature that the fleet-assignment relations already approximate |
| **Problem-conditioned adapters** | One network for five problems may underfit the extra action kinds | FiLM layers on the encoder driven by the context node, or small per-component adapters (HOS, public stations, requests) switched by the context flags; no problem id |
| **Deeper message passing for long haul** | 432 chargers and 30 customers need more than three hops | `--encoder-layers 4` with `--hidden-dim 128`; check memory in a pilot |

### 9.6 Inference-time improvements (no retraining)

| Idea | Why | Sketch |
| --- | --- | --- |
| **Simulated lookahead** | The simulator is a model the policy can roll forward; a few policy rollouts per candidate action with nominal (or sampled) draws choose better than one forward pass | At each decision, clone the environment (the perfect-information module already clones it), roll out the policy on the top-k candidates for a horizon of a few events, pick the best mean outcome; report decision latency, because the gate's claim must include it |
| **Policy as the planner's warm start, planner as the policy's repair** | The long-haul planner accepts an initial solution (`LonghaulPlanner.solve(initial=...)`), and its continuation search fixes charging and rests exactly | Static problems: run the policy nominally once to get routes, hand them to `lh_plan` as the initial solution, execute with the continuation search; or run the policy closed loop and let the continuation search choose the charging stops it proposes |
| **Ensemble of seeds** | Three seeds disagree on hard states | Average the three actors' logits over the shared candidate set (same observation spec), evaluated on validation like any other candidate |
| **Greedy end-game** | Near the horizon the policy's value estimates are weak | Switch to the planner's continuation for the last stretch of each episode when the remaining task set is small |

### 9.7 Raising the bar honestly (better BKS), which also finds headroom

| Idea | Effect |
| --- | --- |
| Rolling planner with a larger budget (`PlannerParameters.iterations`, `time_limit`) and anticipatory waiting (sampled future requests, multiple-scenario approach) | A stronger dynamic BKS; if the policy still passes, the claim is stronger; if not, the gap shows where to look |
| Long-haul planner with more restarts and the rolling variant on every train seed | Better demonstrations for 9.3 as a side effect |
| Lower bounds closer to the simulator (a bound that models the charging curve and HOS exactly) | Tells how much of the BKS gap is real |

### 9.8 A ranked shortlist

If one more iteration is affordable after `suite_v2`, the author's order of bets is: asymmetric critic for the dynamic problems (9.4), privileged DAgger expert there (9.3), plan-conditioned input for the static problems (9.5), self-imitation from the policy's wins (9.3), simulated lookahead at inference (9.6). All of them keep the state and action semantics of the method; none of them touches the gate.
