# Fleet Travel Time: Objective Change, Diagnosis, and Campaign

Updated: 2026-08-14
Status: **complete.** GraphPPO is the best method on the campaign objective;
see section 7 for the test-split result and section 8 for what it does not show.

Documents 08 and 09 optimised and reported **fleet makespan**. This campaign
changes the primary objective to **total travel time of all trucks on a plan
that completes every delivery** — the sum of `truck.total_routing_time` over the
fleet, already extracted by `OperationalMetrics.total_travel_time`.

The two are not the same quantity and not minimised by the same plan. Makespan
is the wall-clock length of the longest route, so it rewards balancing the two
trucks; total travel time is the fleet's driving bill, so it rewards short tours
and few charging detours regardless of how the work is split.

**Result in one line.** On 300 held-out test scenarios GraphPPO drives **119.9
fleet hours** at **0.857** success, against 152.1 h / 0.863 for the makespan-era
policy and 120.5 h / 0.607 for the CP-SAT planner. Paired on jointly solved
scenarios it is **-25.0 h** against MPC, **-21.7 h** against the tuned heuristic,
and **-1.0 h [-5.5, +3.7]** against CP-SAT — statistically level with the exact
planner while solving 25 percentage points more instances. It holds the best
mean ratio (0.9412) and a **median ratio of exactly 1.0000** against the
best-known travel reference, and owns 161 of the 273 best-known plans.

## 1. Where GraphPPO stood (test split, 300 scenarios)

Re-read from the stored `campaign_final` episode rows, conditioning each method
on its own successes:

| Method | Success | Travel h | Distance km | Charging stops |
| --- | --- | --- | --- | --- |
| **GraphPPO** (`speed1500`) | **0.863** | 152.1 | 6041 | 16.1 |
| `greedy_heuristic` | 0.543 | 142.7 | 5690 | — |
| `rolling_horizon_mpc` | 0.520 | 142.3 | 5654 | — |
| `cpsat_plan` | 0.470 | **113.1** | **4500** | — |

Paired on the scenarios both methods solve — the only like-for-like reading,
since the methods decline different instances:

| vs | Jointly solved | GraphPPO | Other | Difference | GraphPPO wins |
| --- | --- | --- | --- | --- | --- |
| `cpsat` | 136 | 149.7 | 112.4 | **+37.3 ± 5.1** | 5% |
| `heuristic` | 156 | 154.2 | 141.8 | +12.3 ± 4.8 | 34% |
| `mpc` | 152 | 148.2 | 141.6 | +6.7 ± 5.3 | 43% |

**The diagnosis is not scheduling, it is distance.** All four methods realise
the same ~39.8 km/h average speed, so travel hours track kilometres almost
exactly. Against CP-SAT on jointly solved scenarios GraphPPO drives 33% further
(5945 km vs 4470 km), takes 10.2 charging stops against 6.8, and pushes 41% more
energy through the battery. It is winning feasibility by being willing to detour,
and paying for it in the objective.

## 2. Three things were misaligned, and all three were fixed

### 2.1 The objective was never measured or selected on

`EpisodeOutcome` carried makespan and operating time but not travel time, so no
part of the stack — periodic validation, checkpoint retention, architecture
selection, heuristic tuning — could see the quantity being optimised. Travel
time, distance, and charging stops are now first-class in the harness, and
`selection_score(summary, objective)` takes the objective by name.

The rule is unchanged in shape: **feasibility first, objective second**. A plan
that abandons deliveries is worthless however short its tour, so success rate
always dominates and travel time only ever breaks ties.

### 2.2 The training signal paid for the wrong thing

`RewardShaping.speed_bonus` paid on `fleet_makespan`. Two changes:

* `travel_time_bonus` — terminal, paid only on a complete plan, proportional to
  fleet travel hours saved against a `trucks × horizon` reference.
* `--time-multiplier` — the environment already charges
  `-actual_travel_time × time_multiplier` on every navigation leg, but the
  config default of `1.0` makes a whole episode's driving worth about -150
  against +5000 in delivery bonuses, i.e. rounding error. This is the only
  signal with **per-action** credit assignment for the objective, so it is the
  primary lever.

Charging time is charged at a fixed `-actual_charge_hours` and is *not* scaled
by `time_multiplier`. That is the right asymmetry here: under a travel-time
objective, charging longer at one stop to avoid a second detour is a good trade,
and the reward now prices it that way.

### 2.3 The action head could not see travel time at all

This was the binding constraint. Every head scores a candidate from its own
feature row plus a pooled state embedding, and `ACTION_FEATURES` was:

```
kind_code, target_node, charge_value, customer_demand,
required_energy, feasible, reason_code
```

There is no leg cost in that row. `required_energy` is a distance proxy and
`target_node` is a raw identifier. The policy was being asked to minimise a
quantity it could not observe per action.

`ROUTING_ACTION_FEATURES` adds six nominal-network columns per candidate:

| Column | What it prices |
| --- | --- |
| `leg_travel_hours`, `leg_reachable` | the leg the action commits to |
| `target_depot_hours` | the mandatory return leg from the target |
| `target_pending_min_hours`, `target_pending_mean_hours` | how well the target leaves the truck placed for the work that remains |
| `insertion_detour_hours` | hours added over driving straight to the nearest unserved customer |

Every value is a *nominal* network quantity — the same deterministic table the
heuristic and CP-SAT baselines plan against — so this reveals no realised
traffic or energy draw the simulator has not already disclosed, and the
shared-observation invariant holds: flat, DeepSets, and hetero-graph encoders
all receive the identical vector.

The observation width changes, so **no earlier checkpoint transfers** and the
curriculum has to run again from random initialisation. The simulator dynamics
are untouched: replaying the frozen heuristic on 12 test scenarios reproduces
the stored episode rows exactly, so the earlier campaign's numbers remain
directly comparable.

## 3. The baselines were re-tuned for the new objective

Comparing a travel-time policy against makespan-tuned baselines would measure
the objective mismatch rather than the method. All three were re-searched on 40
validation scenarios under `--objective travel_time`, and the CP-SAT grid was
extended with the **nominal objective itself**: `makespan` (minimise the longest
route) or `total_time` (minimise the sum).

| Method | Success | Travel h | Selected |
| --- | --- | --- | --- |
| `cpsat` | 0.550 | **111.1** | `objective=total_time`, 150 kW nominal charge, safety 1.15 |
| `mpc` | 0.600 | 141.1 | horizon 6, branching 2, safety 1.15, target SoC 0.8 |
| `heuristic` | 0.450 | 136.5 | demand weight 0.5, safety 1.15, target SoC 1.0 |

Switching CP-SAT to `total_time` made it **stronger on both axes** — success
0.475 → 0.550 and travel 113 → 111 — which is the expected direction: summing
route times is an easier CP-SAT objective than a min-max, and the resulting
plans happen to survive execution more often. The bar this campaign has to clear
is therefore higher than the one document 09 reported against.

Frozen in `results/canonical/frozen_baselines_travel.json`.

## 4. Reward shaping alone is not enough (measured)

Eight arms refined the frozen `speed1500` checkpoint under the *old* action
features, varying only the reward. At a matched 102k-step budget, on 40
validation scenarios:

| Arm | Success | Travel h |
| --- | --- | --- |
| `control` (makespan bonus 1500) | 0.800 | 143.4 |
| `tm5` | 0.800 | 140.3 |
| `tm10` | 0.775 | 140.0 |
| `tm20` | 0.800 | **137.3** |
| `tm40` | 0.700 | 147.9 |
| `tm20_tb3000` | 0.800 | 139.9 |
| `tb6000` (terminal only) | 0.875 | 145.1 |
| `tm10_tb3000` | 0.850 | 151.6 |

Three things this establishes:

1. The dense per-leg penalty is the lever that moves the objective; the terminal
   bonus alone (`tb6000`, 145.1) barely separates from the control.
2. It saturates and then reverses — `tm40` is *worse* on both axes, because at
   that weight a truck that drives less is preferred to one that finishes.
3. Even the best arm lands near 137 against CP-SAT's 111. Reward alone cannot
   close a gap that is really a gap in what the policy can observe.

This sweep was stopped at 102k–205k steps once the representation change made
its checkpoints unloadable; the ablation it supports is reproduced properly in
the v2 campaign, where `v2_ablate` zeroes the new columns while keeping the
observation width, network shape, and budget identical.

## 5. Engineering: rollouts now run in worker processes

The simulator is pure Python at ~4 ms per step, so `SyncCanonicalVecEnv` was
bound to one core no matter how many sub-environments it held — 16 envs stepped
sequentially at ~250 steps/s, making a 4M-step run a multi-hour proposition and
leaving 30 of 32 available cores idle.

`WorkerCanonicalVecEnv` splits the sub-environments across forked workers with
identical semantics: same seed stream (worker `w` takes stream positions
`w, w+workers, …`), same terminal shaping, same auto-reset. Terminal shaping and
reset happen *in* the worker so only the small per-step tuple crosses the pipe —
the full `info` carries scenario descriptors and per-truck state that would
dominate the transfer.

`tests/unit/test_canonical_ppo.py::test_worker_vec_env_matches_the_synchronous_one`
steps both implementations through the same actions and asserts identical
observations, rewards, dones, and masks.

Measured: 26 s/update → 10 s/update per run on the v2 configuration. Beyond
about six workers the remaining cost is the PPO update itself on a shared GPU,
not the environment.

## 6. The four-stage campaign

All stages train the proposed GraphPPO architecture (`hetero_graph` state graph
+ `complete_gcn` action graph) from **random initialisation with no imitation**,
2M interaction steps each, and are validated on the *target* configuration
throughout — never on the curriculum stage in progress.

| Stage | Script | From | Arms |
| --- | --- | --- | --- |
| A | `run_graphppo_v2_campaign.sh` | random init + curriculum | `v2_base`, `v2_tm5`, `v2_tm10`, `v2_ablate` |
| B | `run_graphppo_v2_stageb.sh` | stage-A winner | `b_tm10`, `b_tm20`, `b_tm30`, `b_tm20_tb3000` |
| C | `run_graphppo_v2_stagec.sh` | stage-B winner | `c_sb6000`, `c_sb9000`, `c_tm15`, `c_sb6000_margin` |
| D | `run_graphppo_v2_staged.sh` | stage-C best-success | `d_str2000`, `d_str4000`, `d_margin2000`, `d_tm10_sb6000` |

Stage A learns feasibility under the curriculum. Stage B pushes the travel
penalty up once a feasible policy exists. Stage C was built to buy feasibility
back by raising the *success bonus*, but the arm that won simply softened the
penalty from 20 to 15 — see 6.3. Stage D attacks the residual failure mode,
which by then was strandings rather than unfinished routes; none of its four
levers beat stage C under the selection rule.

`v2_tm20` was launched in stage A and stalled: at that weight, from random
initialisation, a policy that drives less outscores one that finishes, and it
never left curriculum stage 1 (0.21 train success at 82k steps, entropy down to
1.44). It was replaced by `v2_tm5`. This is the same failure `tm40` showed in
section 4, and it is the whole reason the campaign is staged rather than
single-shot: **a heavy travel penalty is safe to apply to a policy that is
already feasible, not to one still learning to be.**

### 6.1 Stage A, matched budgets, target-configuration validation

The right-hand column is the makespan-era from-scratch run of document 09 at the
same budget (success / completed fraction, 40 validation scenarios):

| Steps | `v2_tm10` | `v2_base` | `v2_ablate` | doc-09 run |
| --- | --- | --- | --- | --- |
| 102k | 0.000 / 0.578 | 0.000 / 0.628 | 0.000 / 0.537 | 0.000 / 0.570 |
| 205k | 0.100 / 0.795 | 0.225 / 0.863 | 0.000 / 0.610 | 0.025 / 0.725 |
| 307k | 0.100 / 0.823 | 0.300 / 0.905 | 0.025 / 0.775 | 0.050 / 0.725 |
| 410k | 0.450 / 0.950 | 0.325 / 0.898 | 0.025 / 0.828 | 0.075 / 0.765 |
| 614k | 0.550 / 0.957 | 0.650 / 0.985 | 0.025 / 0.825 | 0.175 / 0.957 |
| 1.84M | 0.725 / 0.985 | 0.850 / 0.993 | 0.250 / 0.968 | — |

The routing-aware policy reaches the target curriculum stage at ~300k steps
where the makespan-era run needed ~600k, and is 3-6x its success at every
matched budget through 614k.

### 6.2 The feature ablation

`v2_tm10` and `v2_ablate` differ in exactly one thing: whether the six
`ROUTING_ACTION_FEATURES` columns carry values or zeros. Same observation width,
same network shape, same reward, same seed stream, same 2M budget. On the
150-scenario validation re-score:

| | Success | Completed | Travel h |
| --- | --- | --- | --- |
| `v2_tm10` (features on) | **0.700** | 0.976 | 145.0 |
| `v2_ablate` (features zeroed) | 0.213 | 0.884 | 158.9 |

State it as a learning-speed effect, not an impossibility result: `v2_ablate`
does climb — 0.000 at 205k, 0.025 at 410k, 0.175 at 1.02M, 0.300 at 2M — it is
simply an order of magnitude slower, and it spends most of its budget stuck at
~0.9 completion **unable to close the depot return**, which is precisely the
failure document 09 diagnosed. Given the leg cost on each action row, the same
reward solves it.

### 6.3 The full ladder, 150-scenario validation re-score

Every checkpoint below was chosen on validation scenarios 0-39 during training,
then re-scored on scenarios 40-189 — disjoint, so the finalist comparison is
independent of the selection that produced it.

| Stage | Run | Success | Completed | Travel h |
| --- | --- | --- | --- | --- |
| A | `v2_ablate` | 0.213 | 0.884 | 158.9 |
| A | `v2_tm10` | 0.700 | 0.976 | 145.0 |
| A | `v2_tm5` | 0.747 | 0.989 | 152.1 |
| A | `v2_base` | 0.767 | 0.989 | 149.3 |
| B | `b_tm30` | 0.700 | 0.979 | 136.2 |
| B | `b_tm20_tb3000` | 0.733 | 0.981 | 140.4 |
| B | `b_tm10` | 0.787 | 0.994 | 135.5 |
| B | `b_tm20` | 0.773 | 0.984 | 121.1 |
| C | `c_sb9000` | 0.840 | 0.992 | 127.4 |
| C | `c_sb6000_margin` | 0.840 | 0.992 | 123.2 |
| C | `c_sb6000` | 0.827 | 0.991 | 119.4 |
| C | **`c_tm15`** | **0.853** | 0.991 | **119.7** |
| D | `d_str2000` | 0.853 | 0.992 | 121.2 |
| D | `d_margin2000` | 0.853 | 0.993 | 122.1 |
| D | `d_str4000` | 0.860 | 0.984 | 122.7 |
| D | `d_tm10_sb6000` | 0.880 | 0.993 | 124.1 |

Three things this ladder says:

1. **Stage B is where the travel time came from.** `b_tm20` cut 28 hours off
   the stage-A winner (149.3 → 121.1) and cost nothing in feasibility
   (0.767 → 0.773). The three penalty weights ran on identical budgets from the
   identical checkpoint and are not monotone — 10 → 135.5 h, 20 → 121.1 h,
   30 → 136.2 h — so the gain is the weight, not the extra 2M steps, and there
   is an optimum near 20 rather than "heavier is better".

   Caveat on this arm: stage B carries **no zero-penalty control**. The stage-A
   winner `v2_base` trained at the config default of 1.0, and no stage-B arm
   continued at that weight, so the 10/20/30 comparison isolates penalty
   *strength* but cannot by itself rule out that some of the 149.3 → 121.1 drop
   is continued training. The stage-A ladder makes that unlikely — `v2_base`'s
   travel was flat near 149 h over its last 600k steps — but it is an inference,
   not a measured control.
2. **Stage C recovered feasibility for free, but not by the lever it was built
   to test.** The arms that doubled the success bonus at the stage-B travel
   weight did improve on stage B (0.773 → 0.827 for `c_sb6000`, 0.840 for
   `c_sb9000` and `c_sb6000_margin`), and tripling it was worse than doubling.
   But the best stage-C arm, `c_tm15`, raised nothing: it simply softened the
   travel penalty from 20 to 15 at the default 3000 success bonus, and reached
   0.853 at 119.7 h. So what stage B overshot was the penalty weight itself, and
   the cheapest correction was to back it off rather than to pay more for
   success.
3. **Stage D's targeted lever lost to the blunt one.** The stranding penalty,
   aimed at the actual failure mode, reached 0.853-0.860; simply making charge
   detours cheaper (`d_tm10_sb6000`, no stranding term at all) reached 0.880 —
   but at 124.1 h, so the band-based rule preferred `c_tm15`.

### 6.4 Selection

Pre-declared rule, applied to all 16 runs:

> Re-score every run on 150 validation scenarios disjoint from those used during
> training. Take every run within **0.03 absolute success** of the best — one
> standard error of a proportion at n=150, p≈0.85. Among those, the lowest fleet
> travel time wins.

`d_tm10_sb6000` led on feasibility at 0.880; `c_tm15` at 0.853 is inside the band
and 4.4 hours shorter, so the objective decided. Frozen as
`results/canonical/selected_travel_d.json`. **No test scenario was read before
this point.**

Two selection defects were found and fixed during the campaign, both worth
recording because both silently favoured the wrong checkpoint:

* The shortlist that feeds the re-scoring round ranked strictly by success
  first, so it cut `b_tm20` — the best run on the campaign objective — before it
  was ever re-scored. The shortlist now uses the same tolerance band as the
  final choice (`_banded_key`).
* A run of the pipeline stalled for 50 minutes because its "is training still
  going?" check matched its own monitoring shell's command line. It now matches
  the interpreter invocation, not the bare script name.

## 7. Final result: test split, 300 scenarios

Selected on validation only: **`c_tm15`** — GraphPPO (`hetero_graph` +
`complete_gcn`), trained from random initialisation with **no imitation
learning**, curriculum to the target configuration, then three refinement stages
against fleet travel hours.

### 7.1 Feasibility and cost

| Method | Success | Wilson 95% | Completed | Travel h* | Makespan h* | Operating h | s/decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `graphppo_makespan` | 0.863 | [0.82, 0.90] | 0.992 | 152.1 | 92.2 | 163.8 | 0.0063 |
| **GraphPPO** | **0.857** | [0.81, 0.89] | 0.991 | **119.9** | 79.1 | **132.4** | 0.0111 |
| `cpsat_plan` | 0.607 | [0.55, 0.66] | 0.930 | 120.5 | **74.8** | 189.1 | 0.0039 |
| `greedy_heuristic` | 0.543 | [0.49, 0.60] | 0.917 | 142.7 | 84.7 | 225.2 | **0.0008** |
| `rolling_horizon_mpc` | 0.520 | [0.46, 0.58] | 0.919 | 142.2 | 85.1 | 220.6 | 0.0016 |
| `random_feasible` | 0.000 | [0.00, 0.01] | 0.660 | n/a | n/a | 246.9 | 0.0008 |

\* conditioned on each method's own successful episodes.

`graphppo_makespan` is the document-09 policy, carried in from
`campaign_final/test`. It was scored on these same 300 seeds and the simulator
dynamics are unchanged — replaying the frozen heuristic reproduces its stored
rows exactly — so the comparison is direct even though its narrower observation
means it cannot be re-run under the current schema.

### 7.2 Paired differences on jointly solved scenarios

Methods decline different instances, so their own conditional means are not
comparable. Pairing by scenario seed is the only like-for-like reading.

| vs | Jointly solved | Travel hours | Success |
| --- | --- | --- | --- |
| `cpsat` | 169 | **-1.0 [-5.5, +3.7]** | **+0.250 [+0.190, +0.310]** |
| `mpc` | 151 | **-25.0 [-29.9, -19.8]** | +0.337 [+0.280, +0.397] |
| `heuristic` | 154 | **-21.7 [-26.9, -16.5]** | +0.313 [+0.253, +0.373] |

GraphPPO beats the heuristic and MPC decisively on travel hours — intervals
nowhere near zero — and is statistically indistinguishable from the CP-SAT
planner while solving 25 percentage points more scenarios. It wins the
head-to-head against CP-SAT on **54%** of the 169 jointly solved scenarios.

Against the same reference, the makespan-era policy was **+31.7 h [+26.5, +37.1]**
and won 19%. That gap is what this campaign closed.

### 7.3 Optimality against the best-known travel reference

| Method | Solves | Mean ratio | Median | >=99.9% of best | Owns best plan |
| --- | --- | --- | --- | --- | --- |
| **GraphPPO** | **257/300** | **0.9412** | **1.0000** | **0.630** | **161** |
| `cpsat_plan` | 182/300 | 0.9148 | 0.9590 | 0.385 | 70 |
| `greedy_heuristic` | 163/300 | 0.7903 | 0.7711 | 0.110 | 18 |
| `rolling_horizon_mpc` | 156/300 | 0.7785 | 0.7760 | 0.083 | 13 |
| `graphppo_makespan` | 259/300 | 0.7525 | 0.7637 | 0.046 | 11 |

27 of 300 scenarios were solved by no method.

GraphPPO leads every column. Its **median ratio is exactly 1.0000** — on more
than half the scenarios it solves, it finds the best plan any method found. It
owns 161 of the 273 best-known plans, more than every other method combined.
The self-referential caveat that weakened document 09's version of this table
now cuts the other way: the reference is largely defined *by* the learned
policy, and CP-SAT still scores 0.9148 against it.

### 7.4 Where the hours went

On the 169 scenarios GraphPPO and CP-SAT both solve:

| | GraphPPO | CP-SAT | makespan-era |
| --- | --- | --- | --- |
| Travel hours | 119.6 | 120.6 | 151.9 |
| Distance (km) | 4758 | 4786 | 6029 |
| Charging sessions | 8.8 | 8.3 | 10.3 |
| Energy charged (kWh) | 2211 | 2154 | 2857 |

(the makespan-era column is over its own 169 scenarios jointly solved with
CP-SAT, which happens to be the same count)

Every method realises the same ~39.8 km/h average speed, so travel hours track
kilometres almost exactly. The makespan-era policy drove **27% further** with
**17% more charging stops**; the travel-time policy now matches the CP-SAT
planner's distance to within 0.6%. The diagnosis in section 1 — that this was a
routing-and-detour problem, not a scheduling one — is confirmed by the fix.

## 8. Honest limits

**The 0.863 feasibility target was not strictly cleared.** GraphPPO reaches
0.857 against the makespan-era policy's 0.863 — a two-scenario difference on
300, with Wilson intervals [0.81, 0.89] and [0.82, 0.90] that almost entirely
overlap. Feasibility is *held*, not improved. What improved is everything else:
travel hours 152.1 → 119.9 (-21%), operating hours 163.8 → 132.4 (-19%),
optimality ratio 0.7525 → 0.9412.

Read the two policies as points on one frontier. Stage D's `d_tm10_sb6000` shows
the other end: 0.880 validation success at 124.1 h. Nothing in this campaign
found a checkpoint that is simultaneously above 0.86 success and below 119 h, and
the stage-D sweep — four different feasibility levers — moved success by at most
0.027 while costing 2-5 h of travel. That is the shape of a frontier, not of an
undertrained model.

**Success is still capped below 1.0 by the instance distribution.** 27 of 300
test scenarios are solved by no method, and document 09's nominal reference
proves at least one generated instance infeasible outright.

**GraphPPO is ~3x slower per decision than CP-SAT** (11.1 ms against 3.9 ms),
both trivially real-time. The heuristic is 13x faster than GraphPPO and much
worse on every quality measure.

**The baselines were re-tuned for this objective, which raised the bar.**
Letting CP-SAT minimize total route time rather than makespan improved it on
both axes (success 0.470 → 0.607, travel 113.1 → 120.5 on its own larger solved
set). The result above is measured against that stronger planner, not against
document 09's makespan-tuned one.

## 9. Reproduction

```bash
# baselines re-tuned for the travel-time objective, CP-SAT objective included
.venv/bin/python scripts/evaluation/tune_heuristics.py \
    --scenarios 40 --methods heuristic mpc cpsat --objective travel_time \
    --workers 16 --shards 8 --output results/canonical/tuning_travel

# stage A (from scratch, curriculum, no imitation) -> B -> C -> D
bash scripts/runners/run_graphppo_v2_campaign.sh
INIT_FROM=results/canonical/graphppo_v2/v2_base        bash scripts/runners/run_graphppo_v2_stageb.sh
INIT_FROM=results/canonical/graphppo_v2_stageb/b_tm20  bash scripts/runners/run_graphppo_v2_stagec.sh
INIT_FROM=results/canonical/graphppo_v2_stagec/c_tm15  bash scripts/runners/run_graphppo_v2_staged.sh

# selection on validation only, then the test split
TRAINING_ROOT=results/canonical/graphppo_travel_all3 \
  RESCORE_TOP=16 RESCORE_SCENARIOS=150 WORKERS=16 \
  bash scripts/runners/run_travel_final_campaign.sh
```

Artifacts: `results/canonical/campaign_travel_final/test/` (episode rows,
summaries, paired comparisons, best-known reference),
`results/canonical/selected_travel_d.json` (frozen selection),
`results/canonical/frozen_baselines_travel.json` (tuned baselines).
