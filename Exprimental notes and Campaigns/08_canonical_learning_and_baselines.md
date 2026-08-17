# Canonical Learning Stack and Baseline Campaign

Updated: 2026-08-11

This document continues `07_implementation_handoff.md`. It records the work that
closed pairwise information parity, built the canonical learning stack, and
produced the tuned non-learning baselines the RL method is measured against.

## 1. Pairwise information parity (closed)

The gap named as the first task in `07_implementation_handoff.md` is closed.

- `EVRoutingEnv/state/features.py` now owns every typed source-target value.
  `RELATION_TYPES` fixes a stable order over the nine truck/customer/charger
  relations, and `extract_pairwise_relations` resolves values on the *unique*
  underlying network nodes before gathering them per relation. Identical node
  pairs therefore receive identical energy, travel-time, and reachability values
  regardless of which typed relation reads them.
- `CanonicalFleetFeatures.pairwise_features` is the single canonical source.
  `validate()` rejects a missing relation, a wrong shape, or a non-finite value.
- `EVRoutingEnv/state/representations.py` pads each relation to its configured
  source/target maxima, with a mask that must equal the outer product of the two
  entity masks, and requires padded cells to be exactly zero.
- The flat contract appends the padded pairwise block and its masks in
  `RELATION_TYPES` order; `CanonicalShapeSpec.pairwise_size` and `flat_size`
  account for it, and the Gymnasium observation space follows automatically.
- `canonical_graph_features` now consumes those exact tensors instead of
  querying the transport graph, so the graph adapter cannot observe an edge the
  other adapters miss.

Evidence in `tests/unit/test_canonical_features.py`: cross-adapter equality,
identical values for repeated node pairs, entity-mask outer products, flat-size
accounting, independent source/target permutation covariance, padding overflow
refusal, and finite masked handling of unreachable pairs.

The schema version moved to `joint-fleet-v2`, and later to `joint-fleet-v3`
when depot visibility was added (section 7).

### Cost control

`TransportationGraph.dense_transport_matrix()` builds the all-pairs table once
per environment and caches it. Canonical extraction dropped from 2.34 ms to
0.65 ms per observation.

## 2. Canonical learning stack

| Module | Responsibility |
| --- | --- |
| `algo/canonical_state.py` | Torch view of the flat observation; ragged feasible-action extraction; logit scattering |
| `algo/canonical_encoders.py` | `flat`, `deep_sets`, `hetero_graph` encoders plus the shared normalizer |
| `algo/canonical_policy.py` | Actor-critic joining any encoder to any approved action head |
| `algo/canonical_ppo.py` | One masked PPO shared by every variant |
| `algo/behavior_cloning.py` | Demonstration pretraining |

Every policy consumes the **same flat observation vector** and unpacks it
locally. Representation choice therefore changes inductive bias only, never
information. A single `CanonicalNormalizer` is shared by all encoders so that
feature scaling cannot become a confound.

### Correctness properties under test

- `ragged_actions` refuses an empty feasible set and refuses any override mask
  that would *widen* the simulator's hard mask.
- `evaluate_actions` refuses a stored action outside the feasible set.
- All nine encoder x head combinations produce finite forward values and
  gradients.
- The set and graph encoders are invariant to truck, customer, and charger
  permutation.
- The graph layer's folded aggregation is proven **exactly** equal to explicit
  per-edge scoring, not an approximation (`test_folded_message_passing_equals_dense_edge_scoring`).
- PPO is validated on a synthetic task with a known optimum
  (`test_ppo_learns_the_optimal_action_on_a_known_task`).

### Two defects found and fixed during bring-up

1. **Value loss swamped the policy gradient.** Raw returns reach the hundreds of
   thousands because the shaped reward pays 500 per delivery. Under a shared
   `max_grad_norm`, global clipping crushed the policy gradient: measured
   `approx_kl` was ~5e-4 and the policy did not move. `RunningReturnScale`
   normalizes rewards by the running discounted-return scale. Value loss fell
   from ~2.5e5 to ~5e-2 and the policy began learning.
2. **Dense pairwise encoders were 35-56x slower than needed.** Both slow
   encoders materialized a `[batch, sources, targets, hidden]` tensor. The graph
   message is affine in the source embedding and edge features and aggregation
   is a masked sum, so the sum is now folded through the linear map
   (algebraically identical). The set aggregator projects its three terms
   separately and runs at a reduced relation width. Forward+backward at batch
   256 went from 3638 ms to 453 ms (DeepSets) and 2233 ms to 235 ms (graph).

The self-attention action head was also rewritten from a per-state Python loop
to one padded batch with `key_padding_mask`, and the complete-graph edge index
from a per-segment loop to a masked grid. Both preserve their existing tests.

## 3. Baselines

All baselines obey the same hard feasibility mask as the learned policies and
are scored through the same runner. They read the simulator directly, which
gives them at least as much information as the canonical observation exposes.

| Baseline | Description |
| --- | --- |
| `random_feasible` | Uniform over the hard-feasible set; weakest reference |
| `greedy_heuristic` | Goal-directed nearest-neighbour with en-route recharging |
| `rolling_horizon_mpc` | Bounded nominal rollout selects the next stop |
| `cpsat_plan` | CP-SAT nominal makespan model executed with energy-safe repair |

### Why the heuristic needed three rounds of work

The naive versions were not weak baselines by accident; each failure exposed a
real property of the problem, and fixing it is what makes the comparison honest:

1. **Charger choice ignoring the goal** made trucks shuttle between two adjacent
   stations forever. Some customers sit further than one full battery away, so
   the charger rule must close distance to the goal. 0% -> 20% success.
2. **No continuation constraint.** Serving a customer the truck cannot then
   *leave* strands it, because the model requires a depot return. Requiring the
   onward leg raised success to 42.5% and removed all strandings.
3. **Targeting customers claimed by the other truck**, which are permanently
   infeasible for this one, plus a charger rule that minimized total instead of
   remaining distance. Only 6.5% of steps were deliveries and 73% were charging
   hops. Fixing both reached 44%.

### Baselines plan with the simulator's own charging physics

`07_implementation_handoff.md` lists "the same nonlinear charging integrator is
not yet shared by every heuristic and optimization baseline" as open. It is now
closed for the canonical baselines. `nominal_charge_hours` prices a recharge
with `ChargingCurveModel.calculate_charge_to_target`, the exact routine the
simulator executes, with the station's own power and the global realistic-curve
flag injected the same way the environment does it. Results are memoized on a
coarse state-of-charge grid because a receding-horizon rollout queries this
constantly.

Two findings from wiring this up:

- Charging a 750 kW DCFast station from 0.8 to 1.0 SoC takes 0.166 h under the
  constant-voltage taper versus 0.107 h under the naive energy-over-power model,
  a 55% underestimate. Planners that ignore this systematically over-value
  topping up a nearly full battery.
- `ChargingCurveModel.estimate_charge_time` is a binary search that **does not
  converge at a target of 1.0**, where the taper approaches the target
  asymptotically; it silently returns its search midpoint of 10 h. The baselines
  therefore use `calculate_charge_to_target` instead. This is worth fixing or
  documenting in the simulator itself.

Making the planners charging-aware made the MPC baseline *stronger* (0.64 ->
0.68 on the 25 tuning scenarios), which is the right direction: a baseline that
is easy to beat proves nothing.

### Fair execution layer

The MPC controller and the CP-SAT planner both execute through the tuned
heuristic's `navigate_toward`. An earlier MPC that planned over raw actions
scored 4% because it stranded trucks: that measured the execution layer, not the
lookahead. With a shared execution layer the three baselines differ only in
*which* stop they choose.

### Frozen baseline settings (grid searched on 40 validation scenarios)

These are the settings carried into the headline campaign. All three plan with
the exact charging integrator.

| Method | Success | Completed | Makespan (successful) | Decisions/episode |
| --- | --- | --- | --- | --- |
| `rolling_horizon_mpc` (h=6, b=2, esf=1.15, soc=0.8) | **0.600** | 0.935 | 82.8 | 90 |
| `cpsat_plan` (esf=1.05, mean power 300 kW) | 0.475 | 0.878 | 70.4 | 98 |
| `greedy_heuristic` (esf, soc, dw from grid) | 0.450 | 0.897 | 80.2 | 114 |
| `random_feasible` | 0.000 | ~0.66 | n/a | ~180 |

CP-SAT proves optimality of the nominal model in ~0.4 s per instance and reports
its own bound, so its gap is attributable to nominal-versus-stochastic mismatch
rather than solver weakness. It also produces the fastest routes when it does
succeed, which is what an optimal nominal plan should do.

Note how much the tuning sample matters: MPC scored 0.64 on the 25 scenarios of
the first grid and 0.60 on 40. Small-sample tuning flatters whichever
configuration happens to suit those instances, for baselines and learned
policies alike.

### The same methods on 100 validation scenarios

Grid search selected on 25 scenarios, which is optimistic: re-scoring the frozen
settings on 100 validation scenarios moves MPC from 0.64 to 0.50. This is
exactly why the held-out test split exists, and why the tuning number must not
be quoted as the baseline's performance.

| Method | Success | Completed | Makespan (successful) | Decisions/episode |
| --- | --- | --- | --- | --- |
| Canonical RL (flat + independent, MPC-taught) | **0.670** | **0.986** | 93.2 | **38** |
| `rolling_horizon_mpc` | 0.500 | 0.919 | 78.9 | 101 |
| `greedy_heuristic` | 0.460 | 0.924 | 78.4 | 95 |

The learned policy leads on feasibility and completion and needs 2.7x fewer
decisions per episode, but its successful routes finish later in simulated
hours. Two caveats matter when reading that last column:

1. Makespan is conditioned on each method's *own* successes, and those sets
   differ whenever success rates differ. `compare_campaign.py` therefore also
   reports makespan restricted to **jointly solved** scenarios, which is the
   only like-for-like speed comparison.
2. The learned policy is selected success-first by the declared rule, so
   trading simulated hours for feasibility is the intended behaviour, not an
   accident.

`RewardShaping.speed_bonus` was added to revisit that trade and then **found to
be inert on validation**. Training an otherwise identical `flat + independent`
run with a 2000-unit makespan-proportional bonus on top of the success bonus
gave mean makespan 91.1 h against 91.2 h without it, and identical peak success
(0.825 both). The knob is kept, defaults to disabled, and the standard shaping
is used, because the evidence does not support the extra term.

The plausible reason is that makespan is dominated by charging and travel legs
the policy cannot avoid once it commits to serving every customer, so the
remaining slack is small relative to the success bonus.

## 4. Exploration and the demonstration warm start

Success requires roughly a hundred consecutive well-chosen decisions. A
randomly initialised policy never observed a successful episode: after 200k
PPO steps validation completion reached 0.82 but success stayed at exactly 0.
The terminal success bonus was therefore never sampled.

`scripts/training/collect_demonstrations.py` caches demonstrations once (4000
train scenarios, 2133 successful, 72966 transitions, 67 s across 20 workers) and
every run reuses that archive. Behaviour cloning immediately produced 40%
validation success where pure PPO produced 0%.

Demonstration seeds are drawn from an offset region of the **train** namespace so
they cannot overlap either the PPO rollout stream or any validation or test
scenario.

**This must be reported plainly in the manuscript:** the learned policy is
initialised from the tuned MPC controller and improved with closed-loop PPO. The
claim is that closed-loop learning improves on its demonstrator, not that RL
solves the task from scratch.

### What the learned policy actually does

Tracing a successful validation episode shows a sensible operator rather than a
simulator exploit. Both trucks depart straight to customers; each recharges only
after its state of charge falls (typically to ~0.3) and then refills to full;
distant customers are reached through deliberate charger staging; and both
trucks return to the depot at the end. The action mix over that episode was 10
customer visits, 17 charger moves, 16 charge actions, and 2 depot returns in 45
decisions, with a minimum state of charge of 0.12.

Compare the tuned heuristic, which spent only 6.5% of its decisions on
deliveries and 73% on charging hops. The learned policy raises that to 22%,
which is where its decision-count advantage comes from.

## 5. Frozen architecture

Selected on validation only, at the shared final budget of 798,720 interaction
steps, with all six runs re-scored on 150 validation scenarios disjoint from the
40 that chose their checkpoints.

| Run | Held-out 150 | Checkpoint-selection 40 | Makespan |
| --- | --- | --- | --- |
| **`hetero_graph` + `self_attention`** | **0.860** | 0.875 | **91.0** |
| `hetero_graph` + `independent` | 0.860 | 0.875 | 96.6 |
| `hetero_graph` + `complete_gcn` | 0.853 | 0.875 | 93.3 |
| `flat` + `complete_gcn` | 0.833 | 0.875 | 94.5 |
| `flat` + `self_attention` | 0.820 | 0.900 | 98.5 |
| `flat` + `independent` | 0.793 | 0.825 | 96.5 |

`hetero_graph + self_attention` is frozen: it ties on success and wins the
declared makespan tiebreak.

Three observations that matter more than the winner:

1. **Every** run scored lower on the held-out scenarios than on the ones that
   selected its checkpoint. The bias is systematic, not one unlucky run, so a
   training-time validation figure must never be quoted as performance.
2. The ordering inverted at the top. `flat + self_attention` led on the 40
   selection scenarios with 0.900 and placed fifth of six on the held-out 150
   with 0.820. Selecting on the smaller set would have frozen the wrong policy.
3. All three heterogeneous-graph runs beat all three flat runs, and also produce
   shorter routes. The margin is about one standard error (~0.03 at n=150), so
   the *individual* comparisons are not significant, but the pattern is
   consistent across all three action heads. Because both encoders receive
   byte-identical observations, this difference is attributable to inductive
   bias rather than information -- which is exactly what the pairwise-parity
   work in section 1 was for.

The action-head effect within an encoder is small and not separable at this
sample size; the honest reading is that the state encoder matters more than the
action head on this problem.

## 6. Headline result (test split, 300 scenarios)

Every method was scored on the same 300 test scenarios through the immutable
runner, after the architecture and all baseline settings were frozen on
validation. Failures are retained in every aggregate.

| Method | Success | Wilson 95% | Completed | Makespan* | Operating h | Decisions | Inference s/ep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Canonical RL** (`hetero_graph`+`self_attention`) | **0.850** | [0.805, 0.886] | **0.987** | 90.5 | **178.1** | **38.7** | 0.303 |
| `greedy_heuristic` | 0.543 | [0.487, 0.599] | 0.917 | 84.7 | 225.2 | 101.5 | 0.054 |
| `rolling_horizon_mpc` | 0.520 | [0.464, 0.576] | 0.919 | 85.1 | 220.6 | 100.4 | 0.101 |
| `cpsat_plan` | 0.483 | [0.427, 0.540] | 0.895 | **70.8** | 192.1 | 89.5 | 0.521 |
| `random_feasible` | 0.000 | [0.000, 0.013] | 0.660 | n/a | 246.9 | 68.6 | 0.031 |

\* conditioned on each method's own successful episodes.

### Paired differences versus the MPC reference

Paired by scenario seed, 10,000 bootstrap resamples:

| Candidate | Success rate | Operating hours | Makespan, jointly solved only |
| --- | --- | --- | --- |
| **RL** | **+0.330 [+0.270, +0.390]** | **-42.5 [-51.8, -33.2]** | +5.6 [+3.3, +8.1] (n=147) |
| `greedy_heuristic` | +0.023 [-0.030, +0.080] | +4.6 [-3.6, +13.1] | -0.4 [-2.3, +1.5] (n=124) |
| `cpsat_plan` | -0.037 [-0.100, +0.027] | -28.5 [-38.7, -18.8] | -13.6 [-16.0, -11.2] (n=103) |
| `random_feasible` | -0.520 [-0.577, -0.463] | +26.3 [+8.5, +44.2] | n/a |

### Reading the result honestly

**The win is decisive on feasibility.** RL's success interval [0.805, 0.886] does
not come close to overlapping the best baseline's [0.487, 0.599], and the paired
difference against MPC excludes zero by a wide margin. It also completes 98.7% of
customers against 89.5-91.9%, and its truncation rate is 0.073 against 0.44-0.45,
meaning it usually finishes rather than running out of horizon.

**The three classical baselines are statistically indistinguishable from one
another.** Heuristic, MPC, and CP-SAT all sit within one another's intervals, and
their paired differences against MPC all include zero. The gap being measured is
learned-versus-classical, not one particular baseline being weak.

**RL is slower per successful route.** On the 147 scenarios both RL and MPC
solve, RL's makespan is 5.6 h longer [+3.3, +8.1] -- a real, significant
regression, not noise. It buys feasibility with time. CP-SAT is the fastest
router by a wide margin (-13.6 h against MPC on jointly solved scenarios), which
is what an optimal nominal plan should do; it simply fails more often. If a
deployment weighted speed above completion, CP-SAT would be the better choice.

**Total operating hours still favour RL** (-42.5 h against MPC) because that
aggregate includes failed episodes, where baselines burn the full horizon before
giving up. The two time metrics point in opposite directions and both are
reported.

**Inference cost.** RL needs 2.6x fewer decisions (38.7 versus 100.4) but each
costs more, so per-episode inference is 0.303 s against MPC's 0.101 s. It is
still 1.7x cheaper than CP-SAT's 0.521 s. All figures are CPU-only and were
measured with the campaign as the only significant load.

## 7. Residual-failure analysis and the depot observability bug

The frozen policy's 45 test failures split into 26 `no_feasible_action`
(a stranded truck: a dead battery in the field, not a late delivery) and 19
truncations. Crucially, **18 of the 45 served every customer and still failed**,
which isolates the mandatory depot return as the single largest failure mode.

### Root cause: the depot was invisible

The canonical node types are truck, customer, and charger. The depot is none of
them, and in the primary configuration it never coincides with a charging node,
so **the depot appeared in none of the nine pairwise relations**. The only depot
channel was the single depot action row, and its `required_energy` was never
computed: feasibility rejects the action on `customers_remain` *before* reaching
the energy calculation, so the row reported the `-1.0` "no value" sentinel.

Measured mid-episode: a truck 239.9 kWh from home, holding 360 kWh, observing
`-1.0`. The policy was being asked to reserve energy for a destination it could
not see, until the moment it was too late to act.

### Fix (schema `joint-fleet-v3`)

- `DEPOT_FEATURES` -- energy, travel hours, reachability to the depot -- added to
  every node type. Because the depot is a single distinguished node, its
  relation to each entity is a per-node feature rather than a tenth relation.
- `battery_minus_depot_energy` on truck rows: return-leg headroom directly.
- Depot-action rejections now carry the real leg energy instead of the sentinel.

**The verdicts are unchanged** -- only the diagnostic is richer. Confirmed by
re-scoring the frozen baselines: MPC 0.600 -> 0.600 and heuristic 0.450 -> 0.450,
bit-identical, so the published test numbers remain valid.

### What the fix demonstrably does, and does not, buy

Same architecture, same teacher, only the schema differs:

| | BC accuracy | BC holdout loss | post-BC validation |
| --- | --- | --- | --- |
| v2 (no depot) | 0.782 | 0.622 | 0.650 |
| v3 (depot) | 0.799 | **0.554** | 0.625 |

Imitation improves exactly as the mechanism predicts: the teacher's decisions
depend on depot distance, which was previously an unlearnable function of the
observation. But **post-BC task success did not improve** (0.650 vs 0.625, well
inside the +-0.076 standard error at 40 scenarios). Better imitation of a
74%-success teacher is not the same as better task success. Whether the added
information pays off is a question for PPO, answered by the ablation below --
not something to be inferred from the cloning stage.

### Shaping repairs under test

`RewardShaping` gained two terms, both training-only:

- `stranding_penalty`: extra penalty when the run ends in a stranding reason.
  Plain incompletion shaping priced a stranded truck and a slow one identically.
- `energy_margin_bonus`: paid on success, scaled by minimum terminal state of
  charge, so a route ending at 40% is preferred to one ending at 5%.

The ablation (`scripts/runners/run_repair_ablation.sh`) runs `depot_only`,
`+stranding`, `+margin`, and `full` from an identical cloned starting point, so
any difference is attributable to the shaping term alone.

## 8. Evidence contract

`scripts/evaluation/run_canonical_campaign.py` scores every method through
`run_evaluation_campaign`, publishing a manifest, raw failure-retaining episode
rows, an aggregate summary, and inference timing per method.
`scripts/evaluation/compare_campaign.py` verifies that all methods were scored
on exactly the same scenarios, then reports Wilson intervals and paired
bootstrap differences, pairing by scenario seed.

Makespan is always reported conditioned on successful episodes, and the
conditioning is stated in the output. Failures stay in every other aggregate.

## 9. Scope of the architecture sweep

The handoff requires the **action head** to be selected on validation and frozen
before test evaluation. The sweep is a complete factorial over the two cheaper
state encoders and all three approved action heads:

|  | `independent` | `complete_gcn` | `self_attention` |
| --- | --- | --- | --- |
| `flat` | yes | yes | yes |
| `hetero_graph` | yes | yes | yes |
| `deep_sets` | implemented, unit-tested, not swept |

The DeepSets encoder is implemented and covered by the same permutation,
masking, and gradient tests as the others, but its pairwise aggregator is the
most expensive of the three and its behaviour-cloning phase did not fit the
compute budget alongside the rest of the sweep. It was therefore dropped from
the sweep rather than given a smaller budget, because an unequal budget would
have made the comparison meaningless. This is a stated limitation, not a result
about DeepSets.

### Why selection compares at a shared budget

Runs finish at different wall-clock rates because the encoders differ in cost,
so `select_architecture.py` ranks every run at the largest interaction budget
*all* of them reached. Without that rule the cheapest encoder would win simply by
completing more updates in the same wall-clock window, which measures throughput
rather than architecture.

This is not a theoretical concern here. Ranked at a partial budget of 153.6k
steps, all three `hetero_graph` runs led all three `flat` runs (0.825-0.875
against 0.650-0.700). But `flat + self_attention`, which was *last* at that
budget with 0.650, finished at 0.900 by 800k steps. Reading a mid-training
snapshot as the architecture result would have inverted the conclusion, so the
reported ranking is taken only at the shared final budget.

Finalists are then re-scored on validation scenarios beyond the index used
during training, so the comparison between them is independent of the
checkpoint selection that produced them.

## 10. Split discipline

- `train` -- PPO rollouts and demonstration collection.
- `validation` -- all baseline tuning, all architecture selection, all
  checkpoint selection.
- `test` -- untouched until the frozen headline campaign.

`scripts/evaluation/select_architecture.py` compares runs at their largest
*common* interaction budget, so a run that completed more updates cannot win on
extra experience alone.

## 11. Environment changes

Only two files in the simulator changed, both additive:

- `EVRoutingEnv/models/core/transportation_graph.py`: cached all-pairs table.
- `EVRoutingEnv/state/features.py` and `representations.py`: canonical pairwise
  ownership.

`ortools` was installed for the CP-SAT baseline; `uv.lock` was not modified.

## 12. Reproduction

Run from the repository root with `PYTHONPATH=.` and
`MPLCONFIGDIR=/tmp/evrp-matplotlib`.

```bash
# 0. verification
PYTHONPATH=. MPLCONFIGDIR=/tmp/evrp-matplotlib .venv/bin/pytest -q

# 1. tune every non-learning baseline on validation only
.venv/bin/python scripts/evaluation/tune_heuristics.py \
    --scenarios 40 --methods heuristic mpc cpsat

# 2. cache the oracle-ensemble demonstrations from train seeds
.venv/bin/python scripts/training/collect_demonstrations.py \
    --demonstrator ensemble --scenarios 6000 --workers 24

# 3. architecture sweep: encoders x heads, identical budget and seeds
TIMESTEPS=800000 NUM_ENVS=16 PRETRAIN_EPOCHS=20 TORCH_THREADS=6 \
  ENCODERS="flat hetero_graph" \
  DEMOS=results/canonical/demonstrations/ensemble.npz \
  bash scripts/runners/run_canonical_sweep.sh

# 4. freeze on validation, then score everything on the held-out test split
RESCORE_TOP=6 RESCORE_SCENARIOS=150 SCENARIOS=300 SPLIT=test \
  bash scripts/runners/run_headline_campaign.sh
```

Step 4 performs, in order: validation-only architecture selection with a
held-out validation re-score, assembly of the frozen method set, the test-split
campaign through the immutable runner, and the paired comparison. It is the only
step that reads test scenarios, and it refuses to run if any baseline's tuning
file is missing, so a silently absent baseline cannot weaken the table.

### Sample sizes are load-bearing

Three times in this campaign a small sample pointed the wrong way:

- MPC scored 0.64 on the 25 scenarios of its own grid and 0.60 on 40.
- The frozen baselines scored 0.64/0.44/0.40 on 25 tuning scenarios but
  0.50/0.46/-- when re-scored on 100.
- `flat + self_attention` ranked last at a 153.6k-step budget (0.650) and first
  at 800k (0.900).

Hence: baselines are tuned on 40 scenarios and re-scored, architectures are
compared only at the shared final budget and re-scored on 150 held-out
validation scenarios, and the headline uses 300 test scenarios.

## 13. Still open

- **The learned policy is behaviour-cloned from the tuned MPC before PPO.** The
  supported claim is that closed-loop learning improves substantially on its
  demonstrator, not that RL solves this problem from scratch. Section 4 shows
  pure PPO reaching exactly 0 success.
- RL's makespan on jointly solved scenarios is significantly worse than every
  classical baseline. The success-first selection rule caused this deliberately,
  but a speed-weighted deployment would prefer CP-SAT.
- GPU execution is now verified: two A30s are present and `cuda:1` trains at
  ~273 steps/s against ~115 on CPU. The handoff's record of CUDA being
  unavailable is stale. All *reported test numbers* were produced on CPU.
- DeepSets was dropped from the sweep for compute, not because it lost; its
  encoder is implemented and unit-tested but never trained to completion.
- Only one seed per architecture was trained. The 0.850 test figure carries
  training-seed variance that is not quantified.
- Station-specific charging efficiency is still not loaded from empirical data.
- ALNS and constructive-attention baselines are not implemented.
- The manuscript and response letter have not been revised.
