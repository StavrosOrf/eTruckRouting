# California GraphPPO: capacity, graph messages, and objective alignment

The target remains 99% successful depot-return completion and at most 5% mean
paired excess operating hours against independent feasible references. These
changes are experiments toward that target; no final result is established.

## Evidence motivating the experiments

On the disjoint 128-scenario validation set, the three frozen 1M completion-mask
policies succeeded on 374/384 policy/scenario pairs. Their successful pairs
averaged 62.31% excess operating hours against the best of six independent
references. Travel contributed 60.35 percentage points, charging 1.55, and
queues 0.40. The policies used 7.73 trucks on average versus 4.25 for their
paired references. Cost-component sums matched operating hours to numerical
precision. This points to assignment and routing as the main cost problem.

An initial imitation-policy failure left one customer demanding 3.90 units
after every active truck's remaining capacity had fallen below 2.11. Other
trucks had already retired. The remaining trucks then circulated between
chargers. The retirement-only capacity check did not prevent the later customer
assignment that made completion impossible.

Artifacts are under `results/california_graphppo/ca10t3d_v1/evaluation/`:

- `strong_references_128/cost_components.json`
- `imitation_initial_0/initial_failure_trace.json`
- `capacity_mask_128/independent_reference_report.json`

## Capacity lookahead

The optional `completion_capacity_lookahead` checks remaining capacity after
each candidate customer assignment and retirement. It subtracts demands already
committed to trucks in transit or service and excludes retired/returning trucks.

It checks three necessary conditions:

1. Remaining total capacity covers remaining unassigned demand.
2. Every remaining customer fits at least one eligible truck, respecting owners.
3. For every positive remaining demand threshold `d`, the number of customers
   with demand at least `d` does not exceed `sum(floor(capacity_i / d))`.

These conditions relax the full packing problem. Passing them does not prove
that a feasible assignment exists. Failing one rules out completion under the
single-trip payload model. If the restricted set becomes empty, the original
fallback restores only actions allowed by the execution rules.

On identical six BC-only checkpoints and 128 validation scenarios, the option
raised pooled completion from 365/768 (47.53%) to 406/768 (52.86%). It added no
rejections along 32 successfully replayed feasibility witnesses. The gain is
insufficient for the target and requires training with the restriction.

## Nonlinear graph messages

For a dense relation, the original affine message has the form

`m_ij = W_h h_i + W_e e_ij + b`.

Summing over sources yields

`sum_i m_ij = W_h sum_i h_i + W_e sum_i e_ij + n*b`.

This step preserves separate sums of source states and transport features.
It loses their association: assigning the same edge values to different sources
leaves the sum unchanged. This matters when customer demand or service status
must be associated with proximity to a candidate target. Other policy inputs
still expose direct leg costs; this limitation concerns the graph message step.

The optional `nonlinear_edge_messages` instead pools

`ReLU(W_h h_i + W_e e_ij + b)`.

It uses the same tensors and parameters, with activation before aggregation.
It materializes messages for each pair and therefore uses more memory and
compute. The default preserves the original affine implementation and legacy
checkpoint behavior. The proposed complete variable-action GCN head is retained.

Cohort `ca10t3d_capacity_v7` compares affine and nonlinear messages with matched
seeds, demonstrations, capacity masks, model widths, and PPO settings.

## Undiscounted operating-cost objective

The benchmark has 30 customers and a 500-unit delivery reward. Successful
episodes with the campaign's other optional bonuses disabled have return

`30*500 + success_bonus - time_multiplier*total_operating_hours`

when `gamma=1`. Maximizing this return among successful episodes exactly
minimizes the reported operating-cost objective. A discount below one also
weights when delivery rewards and costs occur in the action sequence.

Cohort `ca10t3d_objective_v8` uses the nonlinear encoder and capacity mask in
both recipes:

| Recipe | Gamma | Time penalty | Success bonus | Incompletion coefficient | Stranding penalty |
| --- | --- | --- | --- | --- | --- |
| Undiscounted | 1 | 10 | 6,000 | 8,000 | 4,000 |
| Cost emphasis | 1 | 40 | 24,000 | 32,000 | 16,000 |

The second recipe increases the operating-cost signal relative to the fixed
per-delivery reward. Failure terms and success bonus scale with the time
penalty. Both use the same train-only demonstration archive as cohort v7.
Evaluation uses the same physical benchmark and operating metrics for every
policy, independent of training rewards.

## Scheduling and selection

Each new cohort starts with two recipes and three training seeds, packed as two
runs per A100 allocation. The existing 1M/4M/12M schedule expands finalists to
five seeds. Five cohorts can require at most 25 concurrent final-stage training
allocations, below the ceiling of 30.

All cohorts share six independent reference methods and one final selection
pipeline. Both target metrics must pass on 512 disjoint validation scenarios
before opening one locked 1,000-scenario test set. Failures remain in the
reports; certified optimality is not claimed from these reference costs.

## Retained expert guidance (v9)

The 30-checkpoint comparison on 128 intermediate-validation cases completed
on 2026-10-18. The best three-seed recipe is `bc_progress`: 97.92% successful
completion, 152.49 successful operating hours and 38.57% mean paired excess
against the independent six-reference envelope. The hierarchical cost interval
is [35.49%, 41.72%]. Travel still contributes 37.31 percentage points and the
policy uses 6.77 trucks versus references' 4.26 on paired successful cases.
The target remains unmet. This bank is opened development validation, not a
final test. Model variants have different accumulated training budgets.

`ca10t3d_retention_v9` compares three recipes with matched seeds and the same
progress-feature GraphPPO policy, corrected completion mask, 20 BC epochs,
22,923 successful train-only expert transitions, gamma 1, PPO learning rate
1e-4 and entropy coefficient 0.005. Their only difference is the coefficient
of an expert cross-entropy term during PPO: 0 (control), 0.1 or 1. The
coefficient decays linearly to zero at 12M decisions. Uniform expert minibatches
of 128 remain on CPU and are streamed to the GPU; they do not enter the PPO
importance ratio or observation-normalizer updates. Full checkpoints pin the
expert archive hash and guidance settings; resumption reloads the same archive
and restores RNG state. Deployment still uses only the learned graph policy.

Initial screening has nine runs in five A100 allocations (array 28267391),
followed by coordinator 28267396. At most two recipes advance to the existing
4M/12M schedule and five training seeds. All six cohorts share the final
selection gate; at most 30 packed final-stage training allocations are needed.
This is an experiment addressing loss of expert guidance after initialization,
not evidence that expert retention will close the gap.

The canonical action features already include direct travel time, depot-return
time, remaining-customer distance summaries and insertion detour. Adding those
same quantities again would not address the demonstrated weakness.

## Follow-up measurements

Failure-replay job 28267429 reproduced nine selected failures with the same
outcomes and operating costs as the frozen comparison. The leading recipe
shows both residual demands exceeding every remaining truck's payload after
retirement and repeated charger cycles despite sufficient residual payload.
One replay serves all 30 customers but strands a truck before depot return.
These observations distinguish completion from merely serving customers.

CPU job 28267495 measures the capacity-lookahead restriction on the same three
trained `bc_progress` checkpoints against their existing unmodified-mask
results and all six references. It changes no learned weights and is reported
as a controlled post-training mask ablation, not a new training result.

This comparison completed: all three seeds have identical success and
successful-case costs to their controls. A follow-up trajectory audit (job
28267550) confirmed that the restriction really changes masks and rejects
selected actions in two replayed failures. It changes those failed trajectories
without rescuing completion; the successful cases are unchanged. Thus the null
aggregate result is not evidence that the flag was ignored.

The audit exposes a concrete limit of the relaxed capacity checks. Before
action 66 of seed 1000002075, two unassigned demands are 3.6974 and 3.4502,
the only truck able to carry either has 6.9765 capacity, and the other live
truck has only 0.6772 capacity after its committed delivery. Neither customer
fits that second truck and their combined 7.1476 demand exceeds the first
truck's capacity. The state is already impossible to finish, although total
capacity, individual fit and threshold-count bounds can all pass before the
next assignment. A bounded exact packing check at earlier assignments and
retirements is therefore a justified next mask experiment. It cannot by itself
solve the observed charger-routing loops or the cost gap.

CPU collection job 28267467 is gathering 4,096 additional ALNS expert
instances exclusively from train seeds 600000 through 604095, disjoint from
both the rollout bank and the earlier demonstration bank. Only successful
mask-compliant trajectories are retained. No current training run consumes the
new data; it supports a subsequent demonstration-size comparison. The first
collection attempt failed before collecting data because the virtualenv Python
symlink was resolved to system Python; the retry preserves the virtualenv path.

## Bounded exact packing and work-escape measurements

Source `source_capacity_exact_v10` adds an optional residual packing search
for at most 12 customers and 4,096 search nodes. Fixed owners are reserved
first, equal-capacity unowned bins are symmetric, and an exhausted budget
returns unknown rather than pruning an action. The original necessary checks
and policy restrictions remain the default. No simulator transition, task,
observation, reward, or learned graph layer changes.

Job 28267599 preserved all 128 development witnesses (6,982 decisions) and
rejected the recorded infeasible two-customer packing. The exact mask averaged
0.913 ms per witness decision. Applied to the same three frozen policies, it
rescued two failures, lost no successes, and left operating costs exactly
unchanged on the 376 commonly successful policy/scenario pairs. Overall success
became 378/384 = 98.44%, successful operating hours 153.28, and mean per-seed
paired excess 38.66%. The successful-case cost mean rises because the two newly
successful cases cost 366.71 and 231.23 hours; this is not a cost reduction.

Source `source_work_escape_v11` additionally examines whether the active truck
is needed for remaining payload and whether an optimistic energy lower bound
proves it must recharge before finishing. Only then must customer escape energy
reach an available charger rather than a depot that retires the needed truck.
The lower bound uses minimum possible energy multipliers and minimum-energy
paths; no future disturbance or witness is a policy input. A truck that can
finish nearby work without recharging is not rejected merely because it is
needed. Job 28267662 preserved the same 128 witnesses but rescued no additional
failures; one commonly successful case cost 0.98 hours more. It stays
experimental rather than being promoted as an improvement.

## Cost-focused warm-start screening (v12)

Array 28267725 runs six experiments in three A100 allocations: two treatments
with three matched parent/training seeds. Each starts from its corresponding
frozen `bc_progress` checkpoint in the 30-policy comparison. Thus PPO begins
from a policy already trained on the target problem, with its observation
normalizer retained. Both treatments reset the value head, optimizer, return
scaler and interaction count, keep the same graph encoder/complete variable
action GCN, use gamma 1, learning rate 1e-4 and entropy coefficient 0.005, and
train for an additional 1M decisions without repeating behavior cloning.

| Treatment | Delivery bonus | GAE lambda |
| --- | --- | --- |
| Matched warm-start control | 500 | 0.95 |
| Cost/credit-assignment package | 0 | 1 |

Among successful episodes both maximize the same ordering of operating cost:
the per-delivery reward totals a fixed 15,000 when gamma is 1. The experiment
addresses finite-critic credit assignment and optimization, not a claimed
change to the successful-solution optimum. Lambda 1 uses undiscounted returns
within each rollout, with value bootstrapping at rollout boundaries. These two
changes form a package; if it helps, their separate effects need further
ablation. Raw logged rewards between treatments are deliberately incomparable;
success and operating hours remain the common evaluation criteria.

Live runtime settings and PPO updates were inspected for all six runs. This
does not establish held-out quality. Frozen-parent hashes, source hashes and
`effective_settings_audit.json` record the initialization and actual settings.
The stage is exploratory and has no automatic 12M promotion.

The concurrency plan permits at most 29 overlapping training allocations in
this stage: four legacy cohorts at at most five allocations, capacity/objective
cohorts at three each, and three screening allocations. Later capacity and
objective arrays are submitted with `afterany:28267725` by the workspace
launcher. A short guard job verified those declarations before this array
became eligible. SLURM denied direct dependency updates and release of the
initial held attempt; that zero-runtime attempt was cancelled and replaced by
the dependency-guarded array. Terminal dependency IDs are omitted only after
accounting confirms terminal states, so purged completed IDs cannot stall
later waves. Once screening ends, the original six-cohort ceiling is 30.

## Automatic intermediate evaluations and data audit

Jobs 28267620/28267621 freeze and submit the three-seed expert-retention
comparison after its first 1M stage. Jobs 28267747/28267748 do the same for
the two warm-start treatments. All use the same opened 128-case development
bank and six independent references, never adding learned policies to BKS.
They do not open final validation or test data. Final five-seed evaluation and
the 99% / 5% requirements remain unchanged.

Archive audit job 28267818 follows the 4,096-instance collection job. It checks
train-only seed provenance, immutable source/config hashes, archive integrity,
recorded teacher-action feasibility, schema dimensions and finite observations
while streaming the large observation array. The expanded dataset is not yet
used by any training run.

## Early screening observations

At the matched 102,400-decision checkpoint, all three seeds have selection
validation results: control completion averages 94.27%, retained expert weight
0.1 averages 68.23%, and weight 1 averages 50.00%. These use the reused 64-case
bank and do not establish independent quality, but contradict an assumption
that retaining the present imitation objective necessarily helps. Cheaper
successful-case costs in the failing treatments must not be interpreted as
better overall policies. The 1M gate and independent rescore remain scheduled.

All 25 collection batches returned 4,007 successful episodes and 183,697
transitions from the 4,096 new train scenarios. At this observation the large
archive was still being serialized; it is not ready until collection and the
dependent archive audit complete. No training run consumes a partial archive.

The collection and audit have now completed successfully. The verified archive
contains 4,007 successful episodes, 183,697 decisions and 18,945 finite float32
observation values per decision; every expert action passes its recorded mask.
SHA256 is `e380b504d1d5e75d5baf2cd0b0fbfacf00df8b2d707ba789ead51498675bc562`.
The archive is available for subsequent training, but no current run uses it.

At matched 102,400 fine-tuning decisions across three seeds, the warm control
has 96.35% selection completion and 137.89 successful operating hours; the
zero-delivery / lambda-one treatment has 93.23% and 131.98 hours. This is an
early reliability/cost trade-off, not evidence of improved constrained quality.
Neither reused selection scores nor training success replace the scheduled
independent evaluation.

## Next decisions after the 1M screening reports

- Use paired success and cost on the same 128 development cases to decide
  whether either new training treatment deserves more compute. Do not promote
  a lower successful-case mean when failures increase.
- If cost fine-tuning continues to trade reliability for travel savings,
  evaluate an explicit training-failure constraint or adaptive failure penalty.
  Scaling cost and all terminal terms together does not enforce a 99% success
  constraint. Keep its measurement on train trajectories and assess generality
  on held-out cases.
- Use the verified larger demonstration bank in a controlled comparison with
  the old bank, reporting demonstration coverage and optimizer updates. Internal
  BC holdout currently samples transitions within trajectories; use whole
  demonstration episodes for a stronger imitation-generalization measurement.
  Original and new banks together have 206,620 decisions. Any new GPU pilot
  needs a fresh resource-overlap audit; the current screen already uses 29
  allocations with its legacy cohorts.
- The new collection omitted 89 failed train instances. Supplementing those
  with a second independent teacher could improve coverage of hard cases;
  never fill that gap with validation/test demonstrations. The current archive
  lacks per-transition scenario labels, so recover episode boundaries from the
  observable episode-step counter or add labels to a subsequent collector.
- Before querying ALNS as an expert on learner-visited states, support current
  truck locations and reserved in-flight work. Its present insertion cost
  starts routes at the depot and its initial task ownership uses preassignment,
  so restarting that planner on an arbitrary live state is not automatically
  a valid recovery teacher.


## Larger expert-data campaign and recovery (2026-10-19)

The latest 42-policy, three-seed comparison uses the same opened 128-case
development bank and the six independent reference methods. The leading
`bc_nonlinear_cost40` recipe averages 97.92% completion, 145.98 successful
operating hours and 32.50% paired excess. Its hierarchical 95% cost interval
across training seeds and scenarios is [29.03%, 36.52%]. The completion interval
is [95.83%, 99.48%]. Neither target is established. Travel contributes about
31.25 percentage points of pooled successful-pair excess. The recipes were
frozen at different accumulated budgets, so this is a progress comparison.

Retaining expert loss did not improve completion. At the later frozen
checkpoints, coefficient 0.1 averages 88.80% completion versus its matched
control's 96.61%, while successful-only cost decreases. This treatment was
screened out during the interrupted 4M stage; it is not labelled converged.
Its final-stage allocation allowance is reassigned to the larger expert study.

Cohort `ca10t3d_large_expert_v14` retains the proposed heterogeneous graph
encoder, complete variable-action GCN and PPO. Its three recipes compare random
initialization, a fixed frozen trained-policy parent, and a matched warm-start
treatment with GAE lambda one and delivery bonus zero. All use identical
20-epoch BC followed by on-policy PPO. The last treatment combines two changes;
its earlier matched warm-start screen averaged 36.55% paired excess versus
40.90% for its control. This motivates a new experiment, not an assumed gain. Gamma is one, the operating-hours weight is 40,
and nonlinear messages, action/entity links and the exact capacity restriction
are enabled. No learned graph layers are frozen. Deployment uses the neural
policy and declared feasibility restrictions, with no ALNS action selector.

The train-only expert archive has 4,007 successful episodes from 4,096 cases
(seeds 600,000 through 604,095), yielding 183,697 decision labels. The BC split
is by whole episode; its normalization sees only the BC training portion
(in addition to previously acquired training statistics for warm starts).
All warm-start seeds use one seed-zero parent, so their variability is
conditional on that parent. The fresh arm varies the full model initialization.
All 183,697 expert labels survive the conservative exact-capacity remasking;
3,363 other candidate rows are removed. Float32 boundary uncertainty retains
actions rather than rejecting feasible expert decisions. The teacher and mask
archives are pinned by SHA-256. No validation/test cases enter this dataset.

The 1M evaluation explicitly compares BC-only, selection-best PPO and final
PPO policies on the same development cases with three training seeds per
recipe. This measures whether PPO improves or damages its initialization.
Its stage driver requires all nine runs to have completed at least 1M decisions
before scoring. Final selection remains the separate disjoint 512-case bank;
the locked 1,000-case test still requires both goal metrics and full independent
reference coverage. Seven cohorts are registered, one is screened out, and the
six active cohorts require at most 30 final-stage training allocations.

The long-running legacy arrays failed while saving checkpoints around 02:20.
No specific storage cause was established. Atomic compressed checkpoint I/O
was introduced through minimal pinned recovery sources, preserving all
physical transitions, encoder mathematics and optimizer updates. The audit
validated 55 full states and identified one missing full checkpoint; that
original seed repeats BC initialization. One corrupt selection-policy file
was archived and its restored current policy is re-evaluated before learning.
A real saved-state round trip preserves model, optimizer, simulator and RNG
values. Live post-resumption files for all ten capacity policies were CRC
checked and loaded after each advanced by at least 102,400 decisions.

Evidence: `recovery_20261019/checkpoint_audit.json`,
`recovery_20261019/live_checkpoint_audit.json`,
`intermediate_20261019_1020/`, and
`ca10t3d_large_expert_v14/input_preflight.json` under
`results/california_graphppo/`.


### Initial larger-data schedule and mask availability audit

The original six larger-data runs remain in array 28279916. Three additional
matched cost/credit-assignment runs have disjoint output directories and run
in supplementary array 28280299 (two packed allocations). Their union uses
five allocations; the five legacy cohorts use at most 25, giving a maximum
of 30 training allocations. The first six run definitions were preserved.
Both arrays must finish before CPU comparison 28280300; coordinator 28280303
then selects at most two recipes for promotion. Superseded pending CPU
coordinators were cancelled at zero runtime; no GPU training was restarted.

Observed resumed training throughput is approximately 200–310 decisions/s
per run, depending on formulation and validation overhead. Future submissions
request 4 hours for 1M stages, 8 hours for 4M stages and 12 hours for 12M stages,
with complete-checkpoint resumption after clean early exits. The scheduler
denied a request to shorten the already pending initial array, so those
existing allocations retain their original 12-hour requests.

CPU audit 28280487 measures completion-only, relaxed-capacity and exact-capacity
mask availability along new ALNS/CP-SAT planned-reference executions on all
128 opened development scenarios. Reference choices are executed unchanged;
the masks are queried for diagnosis. Initial physical descriptors must match
the cached reference cases. This checks whether efficient observed routes
remain expressible under the policy restrictions. These are newly solved
routes, not exact replays of the earlier time-limited solver outputs. The
comparison does not establish a restricted optimum or alter the BKS pool.


The reference-mask audit completed successfully in 7m41s. Both newly solved
reference methods completed 126/128 cases, with mean successful operating costs
111.99 hours (ALNS) and 118.68 hours (CP-SAT). Initial physical descriptors
match the cached scenarios. Completion-only, relaxed-capacity and exact-capacity
masks each rejected **zero actions across 252 successful routes and 11,920
decisions**. Combining the two methods gives an observed admissible successful
route on all 128 cases. Selecting the cheapest successful route separately for
each case averages 0.35% excess to the cached six-reference BKS, with paired
bootstrap interval [-0.61%, 1.29%]. This oracle reference selection is a route
availability result, not learned-policy performance, an optimality bound, or
a claim that a single online policy attains those costs. The cached BKS pool
is unchanged. On this development bank, efficient complete routes are
expressible under the current masks; this supports prioritizing the new
initialization and credit-assignment experiments over further mask changes.
The raw report and independent reference-pool comparison are under
`reference_mask_audit_20261019/`.


All nine larger-data runtime configurations have been inspected. They use
`hetero_graph` / `complete_gcn`, 16 environments with seven rollout workers,
nonlinear messages, entity links, the exact capacity mask, gamma one,
operating-time coefficient 40, and whole-episode BC holdout. The cost/credit
treatment resolves delivery bonus zero and GAE lambda one; its control resolves
500 and 0.95. Fresh and warm initializations match the registered recipes.
`ca10t3d_large_expert_v14/effective_settings_audit.json` records the actual
settings. Thirty training allocations are now running, including all five
allocations for the expanded-data study. This establishes startup, not quality.
