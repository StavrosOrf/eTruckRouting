# GraphPPO From Scratch: Campaign and Findings

Updated: 2026-08-12

This document covers the campaign to train **GraphPPO with no imitation
learning** — classic PPO from random initialisation — and to measure how close
it gets to optimality.

`08_canonical_learning_and_baselines.md` covers the earlier warm-started work
and the baselines. Those baselines are unchanged and remain the comparison.

## 1. What GraphPPO is here

Grounded in `latex/main.tex` §`sec:ppo`: the simulator returns the state graph
and the feasible action set, from which the action graph is built; the actor
scores action nodes with probability mass only on feasible actions; training is
standard PPO with GAE, a clipped surrogate, a value loss, and an entropy bonus.

In the canonical stack that is:

| Component | Choice |
| --- | --- |
| State encoder | `hetero_graph` — the typed state graph |
| Action head | `complete_gcn` — the action graph over the feasible set |
| Algorithm | `CanonicalPPO`, matching Algorithm 1 of the manuscript |

Note this is *not* the pair validation preferred in the earlier sweep
(`self_attention`, 0.860 vs 0.853 held-out). The proposed algorithm is the
specification; that gap was inside one standard error in any case.

## 2. The exploration wall is real and was measured

Pure PPO from random initialisation does not merely learn slowly here — it
actively degrades.

| Steps | Reward | Entropy | Train success | Completion |
| --- | --- | --- | --- | --- |
| 2k | 3069 | 2.28 | 0.000 | — |
| 256k | 1313-2486 | 0.49-0.62 | 0.000 | 0.26-0.33 |

Four entropy settings (0.005, 0.02, 0.05, plus a second seed) all collapsed the
same way. The mechanism: an episode needs roughly a hundred consecutive
well-chosen actions before it succeeds, so the terminal success bonus is never
sampled. With success unreachable but the stranding penalty live, the best
available policy is to *do as little as possible* — which is why completion fell
below the 0.69 a random policy achieves.

### Two dead ends, both measured rather than assumed

1. **Bigger battery does not make the task easier.** The first curriculum gave
   stage 1 a 1600 kWh battery. Charging is proportional to capacity, so a single
   session took ~10 h and the stage burned its 400-hour budget in 67 decision
   steps. Capacity was returned to near-target and the ramp moved to distances.
2. **Shorter hops do not fix exploration.** At every hop range tried, including
   1-5 km, a random policy strands in `no_feasible_action` on 28 of 30 episodes,
   because 25 of the 42 actions are "drive to some charging station" and most of
   those are far away. Physical difficulty was never the binding constraint.

The easiest reachable stage still gives a random policy only ~3% success, which
is why an intervention was needed rather than more compute.

## 3. A reward-design bug found in this campaign

The first milestone arm paid `all_served_bonus = 4000` against
`success_bonus = 3000`. Serving every customer and **failing** the depot return
therefore paid 33% more than succeeding: the shaping made abandoning the return
leg optimal. Both arms carrying it sat at exactly 0.000 with entropy collapsed to
0.43-0.48.

`RewardShaping.__post_init__` now refuses any configuration where an
intermediate milestone reaches or exceeds the goal it leads to. The corrected
arm uses 1500 against 3000.

This is worth recording because the failure looked exactly like an exploration
failure. It was a specification error.

## 4. Curriculum is the intervention that works

Stages vary physical difficulty only — battery, hop distance, horizon, step
budget, and uncertainty — never the number of trucks, customers, or chargers, so
the observation width and action space are identical throughout and one policy
trains across all of them. **The final stage is byte-identical to the target
configuration**, and validation always scores the target configuration, never
the stage in progress.

Ablation at matched budget:

| Arm | Train success | Entropy | Target-config validation |
| --- | --- | --- | --- |
| `curriculum` | 0.51 | 1.92 | climbing (see below) |
| `control` (none) | 0.000 | 0.48 | 0.000 |
| `milestone` alone (buggy bonus) | 0.000 | 0.43 | 0.000 |

Progression through all five stages, scored on the **target** configuration
throughout:

| Steps | Success | Completion |
| --- | --- | --- |
| 102k | 0.000 | 0.570 |
| 205k | 0.025 | 0.725 |
| 307k | 0.050 | 0.725 |
| 410k | 0.075 | 0.765 |
| 512k | 0.175 | 0.882 |
| 614k | 0.175 | 0.957 |

Monotone improvement from a genuine zero. Training continued to 4M steps; the
final policies reached 0.833 and 0.847 on 150 held-out validation scenarios,
i.e. two independent seeds within 0.014 of each other, while both
non-curriculum arms finished at essentially zero on the identical budget.

## 5. Measuring optimality: what worked and what did not

### The exact reference does not work

`EVRoutingEnv/baselines/optimality_reference.py` implements a position-indexed
CP-SAT model with explicit battery state, charging, payload, and depot return.
On the primary configuration it is **not usable as an optimality denominator**:

* 9 FEASIBLE, 14 UNKNOWN, 1 INFEASIBLE out of 24 instances at 150 s each;
* optimality proven on **zero** instances;
* incumbent-to-bound gap ~85% even with the station set restricted.

Restricting stations makes the model a restriction rather than a relaxation, so
its INFEASIBLE verdicts stop meaning anything. Reporting a "gap to optimal" from
this would be false. Closing that gap needs a different formulation entirely —
branch-and-price, or a labelling DP over customer subsets and battery levels.

What it does establish: at least one generated instance is **provably
infeasible**, so 100% success is impossible by construction on this
distribution.

### Best-known solution is the denominator actually used

`scripts/evaluation/build_best_known.py` takes, per scenario, the best objective
any method achieved. On the 300-scenario test split:

| Method | Success | Scored | Mean ratio | ≥99.9% of best |
| --- | --- | --- | --- | --- |
| `cpsat` | 0.483 | 145 | 0.9849 | 0.807 |
| `heuristic` | 0.543 | 163 | 0.8812 | 0.202 |
| `mpc` | 0.520 | 156 | 0.8780 | 0.250 |
| `rl` (warm-started) | 0.850 | 255 | 0.8732 | 0.322 |

30 of 300 scenarios were solved by **no** method.

The reference is partly self-referential — CP-SAT owns 117 of the 270 best-known
plans — so a leave-one-out check was run: against the *other* methods only,
CP-SAT is 14.6% faster and the learned policy 15.4% slower. The ordering is
real, not an artefact of CP-SAT defining its own denominator.

**Read the two columns together.** A method that fails a scenario contributes no
makespan, so ratios are computed only over what each method solved. CP-SAT looks
best on quality precisely because it declines the hard half.

## 6. Final result (test split, 300 scenarios)

Selected on validation only: `speed1500` — GraphPPO (`hetero_graph` state graph
+ `complete_gcn` action graph), trained from random initialisation with **no
imitation learning**, curriculum to the target configuration, then refined with
a makespan bonus of 1500 on top of the success bonus.

### Feasibility

| Method | Success | Wilson 95% | Completed | Makespan* | Operating h | s/decision |
| --- | --- | --- | --- | --- | --- | --- |
| **GraphPPO** | **0.863** | [0.820, 0.898] | **0.992** | 92.2 | **163.8** | 0.0063 |
| `greedy_heuristic` | 0.543 | [0.487, 0.599] | 0.917 | 84.7 | 225.2 | 0.0006 |
| `rolling_horizon_mpc` | 0.520 | [0.464, 0.576] | 0.919 | 85.1 | 220.6 | 0.0014 |
| `cpsat_plan` | 0.470 | [0.414, 0.526] | 0.896 | **70.5** | 191.2 | 0.0133 |
| `random_feasible` | 0.000 | [0.000, 0.013] | 0.660 | n/a | 246.9 | 0.0006 |

\* conditioned on each method's own successful episodes.

Paired against MPC: success **+0.343 [+0.287, +0.400]**, operating hours
**-56.8 [-67.2, -46.4]**, makespan on the 152 jointly solved scenarios
**+5.9 h [+3.1, +8.6]**.

### Optimality against best-known

| Method | Solves | Mean ratio | Median | >=99.9% of best |
| --- | --- | --- | --- | --- |
| `cpsat_plan` | 141/300 | **0.9871** | 1.0000 | **0.780** |
| `greedy_heuristic` | 163/300 | 0.8830 | 0.8977 | 0.215 |
| `rolling_horizon_mpc` | 156/300 | 0.8725 | 0.8767 | 0.250 |
| **GraphPPO** | **259/300** | 0.8507 | 0.8863 | 0.328 |

### What the makespan objective bought

Refining the frozen success-first policy with a makespan bonus improved every
column at once, which is unusual and worth recording:

| | success-first | + makespan bonus |
| --- | --- | --- |
| Test success | 0.823 | **0.863** |
| Makespan | 100.7 | **92.2** |
| Optimality ratio | 0.8158 | **0.8507** |
| Within 0.1% of best | 0.263 | **0.328** |
| Makespan gap vs MPC | +14.5 h | **+5.9 h** |

Note this contradicts the earlier finding (section 4 of document 08) that the
speed bonus was inert. That result held for the *imitation-warm-started* policy,
which had inherited its demonstrator's routing habits; a policy that discovered
its own routes had slack to trade.

## 7. Status against the 99.9% target

"99.9% optimality" is interpreted as matching the best-known plan within 0.1%.
Neither axis is near it:

**The target was not reached.** GraphPPO matches the best-known plan within 0.1%
on **32.8%** of the scenarios it solves, not 99.9%. The best any method achieves
on that measure is CP-SAT's 78.0%, and it does so by declining 53% of the
instances.

Two limits are structural rather than a matter of more training:

1. **99.9% success is impossible on this distribution.** 32 of 300 test
   scenarios are solved by no available method, and the nominal reference proves
   at least one instance infeasible outright. No policy can exceed that ceiling.
2. **99.9% of *optimal* is not measurable here.** The exact reference proved
   optimality on zero instances with an ~85% bound gap, so only "99.9% of
   best-known" can be quoted — a weaker and partly self-referential statement,
   since CP-SAT owns 117 of the 268 best-known plans.

What would be required to go further:

* a stronger exact formulation (branch-and-price, or a labelling DP over
  customer subsets and battery levels) to make optimality measurable at all;
* an instance generator that guarantees feasibility, so success has a reachable
  ceiling;
* for plan quality specifically, a stronger makespan weighting or a local-search
  post-step on the learned route -- the makespan refinement above moved the
  ratio 0.8158 -> 0.8507 and had not saturated.

## 8. Reproduction

```bash
# curriculum ablation, GraphPPO, no imitation
TIMESTEPS=4000000 bash scripts/runners/run_graphppo_campaign.sh

# what fraction of instances admit any complete plan (inconclusive, see §5)
.venv/bin/python scripts/evaluation/measure_ceiling.py --split validation \
    --scenarios 24 --time-limit 150 --workers 12

# optimality denominator actually used
.venv/bin/python scripts/evaluation/build_best_known.py \
    --campaign results/canonical/campaign/test
```
