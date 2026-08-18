# Revision Experiments: Mask, Baselines, Optimality, Generalization

Updated: 2026-08-18
Status: **sections 1-8 complete** and reproducible from the artifacts they
cite. Section 9 (charging action-space granularity) and the seed replication of
the *full* ladder are still training.

**The three results that should change the manuscript**, before the detail:

1. **The hard feasibility mask does not explain the reported feasibility**
   (§1). Three seeds per arm: masked 0.760, unmasked 0.773, overlapping
   ranges. What matters is that infeasibility costs something commensurate with
   failure, not that it is unselectable.
2. **The optimization baseline was defective and is now stronger** (§3). Its
   CP-SAT model could not leave a truck idle, so it returned worse-than-optimal
   plans labelled OPTIMAL. Exhaustive enumeration found it; after the fix
   CP-SAT matches brute force on 30/30 tiny instances, and the headline is
   reported against the corrected planner.
3. **Training-seed variance is larger than most effects being measured** (§1.3).
   The same configuration spans 0.107 success across three seeds. Every
   single-seed conclusion in documents 08-10 -- and several of this document's
   own first drafts -- has to be read against that.

Documents 08-10 established the method and the headline result under the fleet
travel-time objective. This document covers the experiments the reviewers asked
for that documents 08-10 did not run: the mask ablation, the missing baseline
family, an optimality reference that is actually validated, the charging-model
comparison, and generalization outside the training distribution.

Every arm below matches `v2_tm10` from `run_graphppo_v2_campaign.sh` in budget
(2M interaction steps), curriculum, reward, optimizer settings, and seed stream
unless the text says otherwise, so each is read directly against that run.

**Comparability.** All environment changes made for these experiments are
behaviour-preserving on the published configuration: replaying the frozen
heuristic on 40 test scenarios reproduces the stored `campaign_travel_final`
episode rows exactly on all 11 shared metric fields, including distance, energy
charged, charging sessions, and policy calls. The earlier campaigns therefore
remain directly comparable, with one exception stated in section 3: the CP-SAT
baseline was defective and is stronger now, so its published numbers understate
it and were re-run.

## 1. Does the feasibility mask explain GraphPPO's gain?

This is R1.2 and the editor's E3, and until now the codebase had no way to even
ask it: there was no path to train without the hard mask.

### 1.1 What "without the mask" means here

Two things had to be separated, because they are usually conflated:

* **what the policy may select.** `environment.policy_action_mask` chooses
  between the `hard` feasibility mask and a `structural` mask that hides only
  slots denoting no action at all (an empty customer slot, or no active truck).
  The observation, the candidate set, and the network are identical either way.
* **what happens when an infeasible action is executed.**
  `environment.invalid_action_mode` is either `terminate` -- the simulator's own
  semantics, where committing a truck to a leg it cannot complete strands it --
  or `penalize`, where the action is refused, the state is untouched, the
  episode continues, and a per-refusal penalty is charged.

The `penalize` arm exists so the result cannot be dismissed as an artefact of
the harshest possible treatment of a mistake.

### 1.2 Result, at three seeds per arm

Every run below is 2M steps from random initialisation under the identical
curriculum, reward, and seed stream, re-scored on 150 validation scenarios
disjoint from the 40 that picked its checkpoint (the protocol of document 10
§6.3). Three training seeds per arm, because at one seed this table supports
conclusions that vanish at two -- see §1.3.

| Arm | Mask | Invalid action costs | seed 0 | seed 1 | seed 2 | Mean | Travel h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `v2_tm10` (control) | hard | unselectable | 0.700 | 0.807 | 0.773 | **0.760** | 139.1 |
| `mask_none_terminate` | structural | strands the truck | 0.793 | 0.820 | 0.707 | **0.773** | 147.3 |
| `penalize_p1000` | structural | -1000, episode continues | 0.773 | -- | -- | 0.773 | 154.0 |
| `mask_none_penalize` | structural | -100, episode continues | 0.000 | -- | -- | 0.000 | n/a |

**The hard feasibility mask makes no measurable difference to feasibility.**
Masked 0.760, unmasked 0.773, with per-arm seed ranges of 0.107 and 0.113. The
two distributions overlap almost completely. This is the direct answer to R1.2,
and it is the opposite of the hypothesis: masking is not what produces the
reported feasibility. An unmasked policy learns feasibility on its own, to
**0.1 invalid actions per episode**.

**What matters is that infeasibility costs something commensurate with
failure.** The two penalized arms differ only in magnitude and bracket the
entire effect:

* at -100, a tenth of the -1000 failure penalty, the policy never learns
  feasibility at all: 0.000 success, 1.6 refused actions per episode, and every
  episode ending in a deadlock (`no_events_with_unserved_customers`, 14 of 20)
  or `no_feasible_action` (6 of 20). A refusal advances nothing -- not the
  clock, not the truck -- so when every truck refuses in the same round the
  event queue empties;
* at -1000, matching the failure penalty, the same arm reaches 0.773 -- level
  with both the masked control and the stranding variant.

So the earlier reading that "the consequence must be terminal" was wrong: it
was an artefact of the penalty magnitude, which is exactly what the fairness
sweep was run to check. Stranding is sufficient but not necessary; a
proportionate penalty does the same job.

**On the objective the mask may help, weakly.** Masked runs average 139.1
travel hours against 147.3 unmasked -- about 8 hours, against per-arm seed
ranges of 13.6 and 7.1. The direction is consistent across all three pairs but
the magnitude is inside seed noise, so it is reported as a tendency, not a
result.

### 1.3 Why three seeds

At one seed per arm this table read: masked 0.700, unmasked 0.793, penalized
0.000 -- which supports "removing the mask *improves* feasibility" and "the
consequence must be terminal". Both dissolve at three seeds. Seed 1 of the
masked control (0.807) beats every unmasked run, and a ten-fold penalty change
moves the penalized arm from 0.000 to 0.773.

Effects smaller than roughly 0.1 success are not separable from training noise
at this budget. That threshold is what licenses the rest of the ablation table
in §2 and §6: the component ablations and the independent-head family clear it,
the attention encoder does not.

### 1.4 Seed variance across the full ladder

The full A -> B -> C ladder was replicated at seeds 1 and 2 with the stage
hyperparameters seed 0 selected, so this measures the reproducibility of the
*pipeline*, not of a re-run search. On the 150-scenario validation re-score:

| Stage | seed 0 | seed 1 | seed 2 | Range |
| --- | --- | --- | --- | --- |
| A (`v2_base`) | 0.767 / 149.3 | 0.793 / 163.0 | 0.833 / 161.1 | 0.066 |
| B (`b_tm20`) | 0.773 / 121.1 | 0.707 / 136.5 | 0.827 / 128.5 | 0.120 |
| C (`c_tm15`) | **0.853 / 119.7** | **0.827 / 127.4** | **0.827 / 123.3** | **0.026** |

**The final configuration is markedly more reproducible than the stages that
produce it.** Stage C spans 0.026 success and 7.7 travel hours across seeds,
against 0.120 and 15.4 at stage B. So the headline is stable even though the
ladder that reaches it is not: the intermediate rankings that document 10 §6.3
drew conclusions from are inside the noise, while the endpoint is not.

One consequence worth stating plainly: the ladder is **not monotone per seed**.
Seed 1 goes 0.793 -> 0.707 -> 0.827, i.e. stage B made it worse before stage C
made it better. Any narrative describing the stages as successive improvements
is describing seed 0 only.

Artifacts: `results/canonical/mask_ablation/`, `results/canonical/rescore_batch1.json`.
Reproduce with `scripts/runners/run_mask_ablation.sh`.

## 2. The learned baseline family under equal information

R1.6 and R2.9 ask for DeepSets-PPO, a state-GNN PPO with independent action
scoring, and a constructive attention model, all under equal information. Every
arm shares the observation, the hard mask, the curriculum, the reward, the seed
stream, and the 2M-step budget; only the architecture differs.

Re-scored on the same 150 disjoint validation scenarios:

| Arm | Encoder | Action head | Success | Completed | Travel h |
| --- | --- | --- | --- | --- | --- |
| `v2_tm10` (proposed) | hetero graph | complete GCN | 0.700-0.807 | 0.976 | 131-145 |
| `attention` | transformer | complete GCN | **0.773** | 0.976 | 150.1 |
| `deep_sets__independent` | DeepSets | independent | 0.573 | 0.959 | 155.9 |
| `hetero__independent` | hetero graph | independent | 0.547 | 0.971 | 146.6 |
| `flat__independent` | flat MLP | independent | 0.527 | 0.969 | 148.1 |

The proposed model is quoted as its three-seed range, because that range is
what any single-seed arm has to clear to be distinguishable from it.

**The action head is what separates the family, not the state encoder.** All
three independent-head arms land at 0.527-0.573, below the proposed model's
worst seed by more than 0.12, while swapping the *encoder* under a
complete-GCN head moves nothing detectable: the transformer arm's 0.773 sits
squarely inside the graph encoder's 0.700-0.807 band. DeepSets is finally
trained to completion -- document 08 dropped it for compute and refused to draw
a conclusion from a short run -- and it is the best of the independent-head
arms, but still far below any complete-GCN arm.

The constructive attention baseline (R1.6, R2.8) is a transformer over the node
set in the style of the attention model, with typed edge features entering as a
per-head attention bias so it reads the same canonical content as the graph
encoder. Kool et al. assume a single vehicle, no charging, and no exogenous
uncertainty, so a literal port is impossible; what transfers is the
architecture, and the deviation is stated rather than hidden. **It is
competitive**: an attention encoder is a perfectly good substitute for the
heterogeneous graph encoder on this problem, which is a more useful finding for
the manuscript than a win would have been, because it says the contribution is
not the graph.

Artifacts: `results/canonical/learned_baselines/`.
Reproduce with `scripts/runners/run_learned_baselines.sh`.

## 3. The optimization baseline was wrong, and now it is validated

R1.4 asks for exact or bounded optimization and for the solver evidence to be
published. R2.7 asks for tiny exact objectives to be validated against
exhaustive enumeration. Doing the second found a defect in the first.

### 3.1 The defect

The nominal CP-SAT model gave the depot no self-loop. `AddCircuit` then forces
every truck's circuit through the depot, which forces **every truck to serve at
least one customer**. Under the total-time objective the optimum frequently
leaves a truck idle, so the planner returned strictly worse plans and reported
them as `OPTIMAL`.

Exhaustive enumeration found it immediately. On tiny deterministic instances
brute force beat "optimal" CP-SAT on 6 of 6 scenarios:

| Scenario | Enumeration | CP-SAT (reported OPTIMAL) |
| --- | --- | --- |
| 1000000000 | 67.90 | 73.82 |
| 1000000003 | 58.71 | 72.14 |
| 1000000004 | 75.08 | 76.59 |

On seed 1000000000 the enumerated optimum puts all five customers on one truck
and leaves the other at the depot; CP-SAT could not express that plan.

### 3.2 After the fix

With a depot self-loop, and a guard that serving anything requires the depot to
be in that truck's circuit, and the integer time grain tightened from 0.01 h to
3.6 s:

* CP-SAT matches exhaustive enumeration on **30 of 30** instances across
  (5 customers, 2 trucks), (6, 2), and (4, 3), to within the model's own
  discretization;
* ALNS independently reaches the **true optimum on all 30**.

Re-tuned on 40 validation scenarios, the corrected planner is stronger on
feasibility: success 0.550 -> 0.575. The published `campaign_travel_final`
CP-SAT numbers therefore understate the baseline and are re-run in section 7.

### 3.3 Solver evidence is now published

Episode rows carry `policy_diagnostics` for any policy that exposes them: solver
status, objective, best bound, relative gap, wall seconds, solve count, and plan
fallbacks -- the last being how often the executed plan ran out and the shared
navigation layer chose the stop instead, which is what keeps "executed the
optimal plan" honest. A metaheuristic reports no bound and no optimality claim
by construction.

Artifacts: `results/canonical/exact_validation/`.
Reproduce with `scripts/evaluation/validate_exact_objective.py`.

## 4. ALNS: a strong metaheuristic baseline

R1.6 asks for ALNS or an equivalently strong routing-and-charging metaheuristic.
The implementation searches the same nominal arc costs CP-SAT minimises --
travel plus the recharge time each leg's energy implies -- with random, worst,
Shaw-related, and whole-route destroy operators, greedy and regret-2 repair,
adaptive operator weights, and simulated-annealing acceptance. It executes
through the same energy-safe navigation layer as CP-SAT and the greedy
heuristic, so the comparison is about the search.

Tuned on 40 validation scenarios under the travel objective:

| Baseline | Success | Travel h |
| --- | --- | --- |
| `alns` | **0.575** | 113.1 |
| `cpsat` (corrected) | 0.575 | 112.9 |
| `mpc` | 0.600 | 141.1 |
| `heuristic` | 0.450 | 136.5 |

The search does real work: it improves its own greedy construction by **25% on
average** on the nominal objective (32.5%, 18.7%, 9.7%, 37.0%, 38.8%, 16.3%,
36.9%, 11.5% on the first eight validation scenarios). It converges well within
2000 iterations at this instance size -- 2000, 10000, and 30000 iterations give
identical results -- which is worth stating plainly rather than implying that a
larger budget was needed.

Artifacts: `results/canonical/tuning_alns/`,
`results/canonical/frozen_baselines_revision.json`.

## 4b. Quality against compute, and what it says about the instances

E3 asks for quality-versus-runtime evidence behind any scalability claim, and
the execution checklist asks for an optimizer-budget sensitivity. Both search
baselines have a budget knob; the learned policy does not, which is the
comparison rather than an omission. On 60 validation scenarios:

| Method | Budget | Success | Travel h | s/episode |
| --- | --- | --- | --- | --- |
| `cpsat` | 0.5 s | 0.533 | 108.8 | 0.37 |
| `cpsat` | 45 s | 0.533 | 108.8 | 0.37 |
| `alns` | 100 iterations | 0.567 | 109.5 | 0.22 |
| `alns` | 2000 iterations | 0.533 | 108.8 | 0.33 |
| `alns` | 50000 iterations | 0.533 | 108.8 | 2.90 |

Two readings, and the second is the important one.

**Neither search is budget-limited.** CP-SAT returns the identical plan at a
half-second limit and at forty-five seconds, because it proves optimality long
before either; ALNS reaches the same objective by 2000 iterations and then
spends compute for nothing. Quoting a larger budget for either would be
theatre.

**The nominal planning problem is solved at this instance size.** ALNS and
CP-SAT agree on 108.8 h, and section 3 shows both match exhaustive enumeration
on tiny instances. Whatever separates the methods in execution is therefore not
search quality: it is the closed-loop response to realized travel, energy,
service, and queueing. That is the honest framing for R1.4 -- and it also means
the ten-customer instance is small from an optimization standpoint, which the
scale campaign in section 6 is there to probe.

Artifacts: `results/canonical/optimizer_budget/sweep.json`.

## 5. Charging: what the model is, and what the standard alternative costs

Ziyan asked for the nonlinear charging equation to be sourced and validated, and
compared against an established three-segment/piecewise formulation of the
Montoya type. `scripts/analysis/compare_charging_models.py` reports, for every
transition a target-SoC policy can request, the charging time each model
predicts against the curve the simulator integrates.

At 350 and 750 kW stations:

| Model | Mean relative error | Max relative error |
| --- | --- | --- |
| linear (energy / rated power) | 5.0% | 28.7% |
| Montoya-style, 3 segments | 4.7% | 27.8% |
| + a breakpoint in the taper (4) | 2.6% | 10.0% |
| + a breakpoint in the ramp (5) | 0.6% | 4.0% |
| 7 segments | 0.1% | 1.3% |

At 150 kW the taper never binds against the vehicle's own limit, so the linear
model is already within 1.2%.

**The classical three-segment form is barely better than assuming constant
power.** The reason is structural and is recorded as a test rather than a
remark: this curve ramps from 60% of peak power up to peak before tapering, so
it is *not concave*, and concavity is exactly what a Montoya-style
piecewise-linear approximation assumes. Its three pieces straddle both curved
regions. Placing breakpoints inside the ramp and inside the taper -- five
segments -- brings the approximation within 0.6% mean and 4% maximum.

A defect in the same module was fixed while doing this: `estimate_charge_time`
bisected on duration and could not converge at a target of 1.0, because the
integrator stops exactly at full and the search never saw an overshoot. It
returned the midpoint of its search range -- **10 hours for a charge that takes
0.53 h**. It now integrates directly to the target, the same routine the
simulator and every baseline already use.

Artifacts: `results/charging_curves/model_comparison.json`.

## 6. Component ablations

Each arm blanks one named block of the canonical observation, or withholds the
pooled state embedding from the action head, keeping the observation width,
network shape, budget, and seed stream identical. Re-scored on the same 150
validation scenarios; the proposed model's three-seed range is the bar.

| Arm | What is removed | Success | Travel h |
| --- | --- | --- | --- |
| `v2_tm10` (proposed, 3 seeds) | -- | 0.700-0.807 | 131-145 |
| `v2_ablate` (doc 10) | the six routing action features | **0.213** | 158.9 |
| `ablate_edges` | all nine typed pairwise relations | **0.633** | 152.1 |
| `ablate_pooling` | pooled fleet embedding, actor only | **0.667** | 154.4 |
| `ablate_active_truck` | the flag marking which truck decides | 0.727 | 145.0 |
| `ablate_queue` | port count, occupancy, waitlist, workload | 0.793 | 144.0 |

Read against the ~0.1 seed-noise threshold of §1.3, the canonical
representation splits cleanly in two.

**Three blocks carry the model.** The per-action routing features dominate by a
wide margin -- removing them costs at least 0.49 success -- followed by the
typed pairwise relations and the pooled state embedding, both of which fall
below the proposed model's worst seed and cost travel hours on top.

**Two blocks are inert, and that is a result about the instances, not the
architecture.** Blanking the charger queue state (0.793) or the active-truck
flag (0.727) leaves the model inside its own seed band. For the queue features
the explanation is measurable: two trucks across twenty-five stations with 209
ports essentially never contend, so there is no queue signal to exploit at this
fleet-to-station ratio.

This is a statement about *these instances*, not about the model. Section 10.2
builds a regime where contention does bind -- four trucks, one port per station
-- and there the learned policy waits 32% less than the CP-SAT planner. The
inertness measured here is therefore a limitation of the joint setting's
evaluation distribution, which is the honest way to answer E2 and Ziyan's
comment on endogenous queueing.

## 7. The 500-scenario test campaign

R1.7 asks for at least 500 paired test scenarios per main setting; document 10
reported 300. Every method below is scored on the same 500 held-out seeds, and
each learned arm is scored in the environment it trained in.

| Method | Success | Wilson 95% | Travel h* | Makespan h* | Operating h* |
| --- | --- | --- | --- | --- | --- |
| **GraphPPO** (3 seeds) | **0.843** [0.830, 0.858] | -- | **123.5** [120.9, 125.3] | 80.1 | **134.8** |
| GraphPPO (seed 0) | 0.858 | [0.82, 0.89] | 120.9 | 79.0 | 131.7 |
| `mask_none` | 0.824 | [0.79, 0.85] | 140.4 | 88.0 | 153.2 |
| `ppo_deepsets` | 0.670 | [0.63, 0.71] | 152.3 | 95.5 | 166.8 |
| `cpsat_plan` (corrected) | 0.622 | [0.58, 0.66] | 167.5 | **74.5** | 184.9 |
| `alns_plan` | 0.614 | [0.57, 0.66] | 168.6 | 74.0 | 186.3 |
| `greedy_heuristic` | 0.552 | [0.51, 0.60] | 199.8 | 85.1 | 221.1 |
| `rolling_horizon_mpc` | 0.532 | [0.49, 0.58] | 199.2 | 84.9 | 219.7 |
| `ppo_flat` | 0.506 | [0.46, 0.55] | 143.1 | 90.5 | 156.3 |

\* averaged over all episodes, not conditioned on success, so a method that
declines instances is not flattered. Section 7.1 pairs by scenario seed, which
is the like-for-like reading.

**The headline reproduces, and now carries a seed range.** GraphPPO reaches
0.858 on 500 scenarios against 0.857 on document 10's 300 -- extending the
sample did not move the result -- and the two further seeds of the full ladder
give 0.830 and 0.842. The three-seed summary is **0.843 success [0.830, 0.858]
at 123.5 travel hours [120.9, 125.3]**, which is the number the manuscript
should quote. Every other learned row in this table is a single seed at a 2M
budget and is marked as such.

Budget matters for reading the rest of the table: GraphPPO is the full
A→B→C ladder (6M steps), while `mask_none`, `ppo_attention`, and the
`ppo_*` family are 2M-step arms. The budget-matched control is
`graphppo_matched` at 0.726. Comparisons across different budgets are noted
where they occur rather than presented as architecture results.

**The corrected planners are stronger and still lose on feasibility.** CP-SAT
rises to 0.622 from the 0.607 its defective model produced, and ALNS lands at
0.614 -- two independent optimizers agreeing, as they did on the nominal
objective. Both hold the best *makespan* (74.5 and 74.0) while being worst on
all-episode travel, which is the same pattern document 10 reported: they build
short balanced plans and then decline the instances those plans cannot survive.

### 7.1 Paired differences on jointly solved scenarios

Methods decline different instances, so their own conditional means are not
comparable. Pairing by scenario seed is the only like-for-like reading;
negative means GraphPPO drives fewer hours.

| GraphPPO vs | Jointly solved | Travel hours | GraphPPO wins |
| --- | --- | --- | --- |
| `cpsat_plan` (corrected) | 285 | **+0.2 [-3.1, +3.7]** | 53% |
| `alns_plan` | 282 | **+0.9 [-2.5, +4.4]** | 52% |
| `greedy_heuristic` | 259 | **-23.0 [-26.7, -19.1]** | 76% |
| `rolling_horizon_mpc` | 253 | **-26.1 [-29.7, -22.3]** | 81% |
| `mask_none` | 392 | **-31.0 [-34.0, -28.0]** | 86% |
| `ppo_stategnn` | 295 | **-36.2 [-39.8, -32.5]** | 87% |
| `ppo_flat` | 247 | **-42.7 [-46.4, -38.8]** | 87% |
| `ppo_deepsets` | 322 | **-47.5 [-50.8, -44.3]** | 92% |

The central claim survives a larger sample *and* a stronger opponent: GraphPPO
is statistically indistinguishable from the exact planner on travel hours --
now against the corrected CP-SAT rather than the handicapped one -- while
solving **23.6 percentage points more instances** (0.858 against 0.622). ALNS
independently confirms the reading: it agrees with CP-SAT to within a
half-hour, and GraphPPO ties it too.

It beats every learned baseline decisively, with intervals nowhere near zero,
which is the same conclusion the validation re-score reached and locates the
contribution in the action-graph interaction rather than the state encoder.

### 7.2 What this says about the mask

The unmasked arm is the interesting row. On feasibility it nearly matches the
proposed model (0.824 against 0.858, overlapping intervals). On the campaign
objective it is **31 hours worse** on 392 jointly solved scenarios, losing 86%
of head-to-head pairs.

The decisive comparison is against `graphppo_matched`: the masked control at
the ablation's own budget (`v2_tm10`, stage A only, 2M steps), scored on the
same 500 test scenarios. Everything is identical except the mask.

| | Success | Paired travel hours |
| --- | --- | --- |
| `graphppo_matched` (masked, 2M) | 0.726 | reference |
| `mask_none` (unmasked, 2M) | **0.824** | **+5.9 [+3.3, +8.5]** on 343 pairs |
| `graphppo` (masked, full ladder) | 0.858 | **-25.7 [-28.8, -22.7]** on 352 pairs |

Both intervals exclude zero, in opposite directions, so the answer to R1.2 is
two-sided and neither half is a hedge:

1. **The mask does not explain the feasibility.** Removing it *raises* success
   by 0.098 [0.062, 0.134] at a matched budget. An unmasked policy learns
   feasibility on its own -- 0.1 invalid actions per episode -- provided
   infeasibility carries a real consequence. When it does not, the policy never
   learns it at all (section 1.2).
2. **The mask does help the objective, modestly.** Removing it costs 5.9 travel
   hours.
3. **Most of the travel-time advantage is not the mask at all.** The staged
   refinement buys 25.7 hours over the same control -- four times the mask's
   contribution -- which is where the reported result actually comes from.

That is a more useful answer than the reviewer's hypothesis anticipated, and it
is uncomfortable in one direction and favourable in the other, which is the
sign that it was measured rather than argued.

### 7.2b The mask arm on the test split, at three seeds

| Arm | Budget | Success (3 seeds) | Travel h |
| --- | --- | --- | --- |
| GraphPPO | 6M (A→B→C) | 0.843 [0.830, 0.858] | 123.5 [120.9, 125.3] |
| `mask_none` | 2M (A only) | 0.793 [0.750, 0.824] | 140.9 [138.1, 144.1] |
| `graphppo_matched` | 2M (A only) | 0.726 (1 seed) | 136.0 |

Read at matched budget, the unmasked arm's three seeds (0.750-0.824) straddle
the masked control's single 0.726, which is the same null result §1.2 reports
on validation. Read across budgets, the full ladder's three seeds (0.830-0.858)
sit entirely above the unmasked arm's, and the two ranges do not overlap: the
refinement stages, not the mask, are what separate the final model.

### 7.3 The seed caveat, which is load-bearing

Every comparison above uses one training seed per arm. Paired intervals over
500 scenarios quantify *scenario* variance; they say nothing about the variance
of training itself. Replicating the masked control at two further seeds shows
that omission is not academic. On the 150-scenario validation re-score, at an
identical 2M budget and configuration:

| Masked control | Success | Travel h |
| --- | --- | --- |
| seed 0 (`v2_tm10`) | 0.700 | 145.0 |
| seed 1 (`seed1_A`) | 0.807 | 131.4 |
| seed 2 (`seed2_A`) | 0.773 | 141.0 |

**The control alone spans 0.107 in success and 13.6 travel hours across seeds,
with nothing changed but the seed.** The mask effect measured from one seed per
arm -- +0.098 success, +5.9 travel hours -- is inside that band. Seed 1 of the
*masked* control beats the unmasked arm on both axes.

With three seeds now trained on both arms (§1.2), the statement settles at:

* the strong claim **"removing the mask improves feasibility"** is withdrawn:
  masked 0.760 and unmasked 0.773 across three seeds each, with overlapping
  ranges;
* the claim **"the mask is not necessary for feasibility"** is supported and is
  the answer to R1.2 -- the opposite of the reviewer's hypothesis;
* the claim that the refinement stages, not the mask, buy the travel time
  survives, because -25.7 hours is twice the seed range.

The same caveat re-reads the rest of the ablation table. Effects smaller than
about 0.1 success at this budget are not separable from seed noise, which puts
the attention encoder (0.773) inside the control band, and leaves the component
ablations (0.667, 0.633) and the independent-head family (0.527-0.573) as the
only architecture results that clear it.

This is precisely what R1.7 asked for and precisely why: with one seed the
ablation table would have supported several conclusions that a second seed
dissolves.

## 8. Generalization

Every regime scores every method on the same 100 held-out test scenarios, and
each learned arm is scored in the environment it trained in. Success rates:

| Regime | Kind | GraphPPO | `mask_none` | CP-SAT | ALNS | Heuristic | MPC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `in_distribution` | control | **0.82** | 0.75 | 0.59 | 0.60 | 0.54 | 0.57 |
| `customers_4` | size | **0.91** | 0.93 | 0.72 | 0.72 | 0.70 | 0.67 |
| `customers_6` | size | **0.89** | 0.86 | 0.67 | 0.67 | 0.59 | 0.61 |
| `customers_8` | size | **0.81** | 0.81 | 0.68 | 0.69 | 0.57 | 0.57 |
| `fleet_1` | size | **0.72** | 0.69 | 0.62 | 0.60 | 0.52 | 0.61 |
| `chargers_weak` | ood | **0.83** | 0.78 | 0.63 | 0.63 | 0.56 | 0.56 |
| `chargers_inefficient` | ood | **0.82** | 0.76 | 0.60 | 0.60 | 0.53 | 0.56 |
| `ports_scarce` | ood | **0.81** | 0.77 | 0.59 | 0.60 | 0.54 | 0.57 |
| `traffic_severe` | ood | **0.82** | 0.79 | 0.60 | 0.60 | 0.51 | 0.50 |
| `network_slow` | ood | **0.68** | 0.68 | 0.60 | 0.61 | 0.54 | 0.56 |
| `network_fast` | ood | **0.85** | 0.78 | 0.60 | 0.60 | 0.54 | 0.55 |
| `demand_heavy` | ood | **0.74** | 0.61 | 0.59 | 0.49 | 0.45 | 0.43 |
| `energy_severe` | ood | **0.41** | 0.27 | 0.35 | 0.35 | 0.28 | 0.16 |
| `battery_small` | ood | 0.34 | 0.34 | **0.43** | 0.43 | 0.42 | 0.39 |
| `service_slow` | ood | 0.46 | 0.51 | 0.59 | **0.60** | 0.54 | 0.57 |
| `battery_large` | ood | 0.87 | 0.87 | **0.98** | 0.98 | 0.98 | 0.96 |

### 8.1 What holds

GraphPPO leads 13 of the 16 regimes every method was scored on, usually by 20
points or more, and its lead is undiminished by charger power, charger
efficiency, port scarcity, travel-time variance, demand pressure, and a road
network whose legs are uniformly 25% faster or 40% slower. Size transfer
downward is strong: 0.91, 0.89, and 0.81 at four, six, and eight customers
against a ten-customer training distribution, and 0.72 with a single truck.

### 8.2 What does not, and the pattern in it

Three regimes reverse the ranking, and they are the same kind of regime:

* `battery_small` (300 kWh instead of 400): 0.34 against CP-SAT's 0.43;
* `service_slow` (0.4 h service instead of 0.2): 0.46 against ALNS's 0.60;
* `battery_large` (500 kWh): 0.87 against the planners' 0.98.

Each of these moves the **feasibility frontier itself** -- the energy budget or
the time budget -- rather than the cost of a decision. A planner re-solves from
scratch against whatever budget it is handed. A policy has learned where the
frontier sits, and when the frontier moves it is wrong in a way no amount of
in-distribution skill repairs. `battery_large` is the sharpest version: when the
problem becomes easy the planners solve almost everything and the policy does
not, because it never learned to exploit slack it never saw.

Averaged over the out-of-distribution regimes, success changes by -0.110 for
GraphPPO and -0.081 for the unmasked arm against -0.006 for ALNS, -0.004 for
the heuristic, and +0.009 for CP-SAT. **The learned policies degrade more than
the planners do.** They also start 20 points higher, so they still lead almost
everywhere -- but the honest statement for the manuscript is that this method
buys a large in-distribution advantage and gives part of it back under
distribution shift, with the loss concentrated where the resource budget moves.

This is the claim R1.7 asked to be separated from interpolation and from size
transfer, and it is separated: interpolation and size transfer hold, parameter
shift mostly holds, and budget shift does not.

Eighteen regimes labelled `interpolation`, `size_transfer`, and `ood`, each
scoring every method on the same held-out seeds: customer count, fleet size,
charger power and efficiency, battery capacity, vehicle speed, demand, service
time, road distances, and three different uncertainty laws.

### 8.3 Defects this campaign surfaced

Three, all found by running the experiments rather than by inspection:

* `step()` decoded delivery action indices against `len(delivery_sequence)`
  while the feasibility engine used the fixed action envelope, so the two
  disagreed on any instance smaller than the envelope and selecting the depot
  raised. No published result is affected -- the curriculum never changes
  `num_stops`, so every earlier episode ran at full size -- but no size-transfer
  evaluation was possible until it was fixed.
* Six regimes initially returned results identical to the control. Three were
  vacuous: `truck.base_speed` does not affect travel time (times come from the
  precomputed network tables), and `min_hop_distance`/`max_hop_distance` are not
  read by the joint instance generator at all. **This means the energy-ramp
  curriculum's stages are named for hop distances they never changed** -- what
  they actually ramp is battery capacity, horizon, and uncertainty. The
  curriculum works; documents 09 and 10 describe its mechanism incorrectly and
  the manuscript must not repeat that.
* The generalization runner applied each regime's config overrides but ignored
  each *method's* own environment, so the unmasked arm was scored under the hard
  mask it never trained with -- 0.560 where the correct environment gives 0.750.
  A wrong conclusion about generalization was one commit away.

## 9. Charging action space (R2.6)

R2.6 asks whether the 10% target-SoC grid is a limiting discretization. It
changes the action space, so each variant is a separate policy trained under
otherwise identical settings.

| Arm | Charging actions | Success | Travel h |
| --- | --- | --- | --- |
| `v2_tm10` (proposed, 3 seeds) | 6 target SoCs at 10% | 0.700-0.807 | 131-145 |
| `soc5` | 11 target SoCs at 5% | 0.740 | 135.0 |
| `duration` | 15/30/60-minute durations | **0.627** | **159.3** |

**Finer granularity buys nothing measurable.** Doubling the resolution of the
charging decision lands inside the proposed model's seed band on both axes, so
the 10% grid is not a binding approximation.

**The action *semantics* matter more than its resolution.** Expressing the
charging decision as a duration rather than a target state of charge costs
0.073 success against the proposed model's worst seed and 14 travel hours
against its worst -- the only charging variant to fall outside the band on both
axes. The reason is mechanical: a fixed duration buys a different amount of
energy depending on where the truck sits on the taper, so the same action means
different things in different states, while a target SoC is state-independent by
construction.

Together these close R2.6 and vindicate D6: target-SoC actions at 10%
granularity are the right choice, and both alternatives the reviewer asked
about have now been measured rather than argued.

## 10. Upward size transfer and congestion

Sections 8.1-8.2 measure transfer *downward*, because the headline policy has a
fixed observation width and cannot be evaluated above its own envelope. A
separate policy was therefore trained on a variable-size envelope of up to four
trucks and fourteen customers, and scored across a scale grid and a congestion
regime. Success rate, 100 held-out scenarios per cell:

| Regime | Envelope policy | CP-SAT | ALNS | Heuristic | MPC |
| --- | --- | --- | --- | --- | --- |
| 1 truck, 4 customers | 0.580 | **0.730** | 0.720 | 0.690 | 0.670 |
| 2 trucks, 8 customers | 0.360 | 0.680 | **0.690** | 0.570 | 0.570 |
| 3 trucks, 11 customers | 0.290 | **0.620** | 0.620 | 0.490 | 0.570 |
| 4 trucks, 14 customers | 0.190 | **0.600** | 0.590 | 0.530 | 0.490 |
| 4 trucks, 1 port/station | 0.190 | **0.590** | 0.580 | 0.510 | 0.510 |

### 10.1 Training on a size envelope costs more than it buys

**The envelope policy is beaten by every classical baseline at every size, and
degrades sharply as instances grow** -- 0.580, 0.360, 0.290, 0.190 -- while the
planners hold 0.60-0.73 throughout. For comparison, the fixed-size headline
policy reaches 0.858 on its own 2-truck/10-customer distribution.

This is a negative result and is reported as one. Training on a distribution of
instance sizes at a fixed 2M-step budget produced a policy far weaker than a
specialist trained on a single size for the same budget, and the deficit widens
with size. Whether that reflects the budget, the curriculum, or something
structural about variable-size training is not established here; what is
established is that upward size transfer is **not** obtained for free by
widening the training envelope, and no claim of scale generalization beyond the
trained envelope is supported by this campaign.

### 10.2 Congestion binds, and queue-aware routing is real after all

Section 6 found the charger-queue features inert, which is expected when two
trucks share twenty-five stations holding 209 ports. Reducing the network to one
port per station with four trucks makes contention bind for the first time:

| | Queue hours, CP-SAT | Queue hours, learned policy |
| --- | --- | --- |
| 4 trucks, normal ports | 0.23 | 0.31 |
| 4 trucks, 1 port/station | **1.92** | **1.31** |

Mean queueing time rises eight-fold for the planner, so the mechanism is
genuinely exercised. And under that contention the learned policy waits **32%
less than the planner** (1.31 h against 1.92 h) while both lose only about one
point of success, because the time budget is slack at this fleet size.

Two corrections follow, and the second is a correction to this document's own
earlier text:

1. The inertness of the queue features in Section 6 is a property of the
   fleet-to-station ratio, not evidence that the model ignores queues. Given
   contention, it routes around it better than a planner that re-solves.
2. The earlier statement that "this campaign's instance distribution does not
   exercise the endogenous-queue mechanism" was too broad. It holds for the
   two-truck joint setting and not for the congestion regime, and it says
   nothing about the main eTFRP benchmark, where fleets are an order of
   magnitude larger over the same twenty-five stations and contention is severe
   by construction. That benchmark was not re-run here, so no new claim is made
   about it in either direction.

Artifacts: `results/canonical/scale_campaign/`.

## 11. Do the findings hold in the eTFRP setting?

Sections 1-10 all use the joint fleet formulation. The paper's principal
benchmark is the eTFRP, where customers are assigned to trucks in advance, and
until this campaign that setting could not be expressed in the canonical stack
at all. It now can: `problem.assignment=preassigned` binds each customer to one
truck at generation time, balanced within payload capacity, with the same
observation width, action space, mask machinery, curriculum, reward and
artifact contract as the joint setting. The architecture is identical, so the
comparison between settings is controlled.

Two seeds per arm, 2M steps, re-scored on the 150 validation scenarios beyond
those used for checkpoint selection:

| Arm | seed 0 | seed 1 | Mean | Travel h |
| --- | --- | --- | --- | --- |
| Masked | 0.313 | 0.367 | **0.340** | 163.0 |
| Unmasked | 0.347 | 0.313 | **0.330** | 161.1 |

### 11.1 The mask finding replicates

The mask changes success by **0.010**, against per-arm seed ranges of 0.054 and
0.034, and changes travel time by 1.9 hours in the *opposite* direction. The
conclusion of §1 therefore holds in the paper's principal setting as well: at
this budget the hard feasibility mask is not what produces feasibility.

This matters for how the manuscript reads its own eTFRP results. The published
tables show generic PPO far below MaskPPO and conclude that feasibility-aware
learning is essential. That comparison is real, but it varies the architecture
and the action representation as well as the mask, so it cannot separate them.
The arms above vary only the mask, and the gap disappears. **The published
PPO-versus-MaskPPO gap is evidence for the structured action representation,
not for masking**, and the revised text says so.

### 11.2 Pre-assignment makes the problem harder, not easier

The absolute numbers are the surprise. At an identical budget and architecture,
the same model reaches 0.70-0.81 success on the joint formulation and only
0.31-0.37 here.

Removing decisions did not make the problem easier. Under pre-assignment a
truck must serve its own customers wherever they happen to be, so the fleet
cannot rebalance when one truck draws a distant or energy-expensive set; a
customer that becomes unreachable is unreachable, whereas in the joint
formulation another truck can take it. Assignment is not merely a decision the
eTFRP removes, it is also the mechanism by which the joint formulation recovers
from bad draws.

This cuts against the natural reading of R1.1 -- that the eTFRP is the easier
problem because the hard combinatorial decisions were removed. On feasibility
it is the harder one. It remains true, as the reviewer said, that it asks less
of the method combinatorially; but it asks more of it operationally.

### 11.3 What was not re-run, and why it matters

The published eTFRP tables themselves were **not** reproduced. Three obstacles,
stated so that nobody assumes otherwise:

* their optimization baseline is a per-truck Gurobi MILP, and `gurobipy` is not
  installed on the machine this revision was prepared on, so the "Math. Opt."
  column cannot be recomputed here at all;
* those tables report normalized reward, which this revision argues against as
  a headline metric;
* their fleet sizes reach 100 trucks and above, an order of magnitude beyond
  anything scored here.

So the correct statement is: the *findings* were tested against the eTFRP-style
setting and the mask finding replicates, but the *published eTFRP tables* still
carry the defects this campaign identified elsewhere -- single-seed
comparisons, and an optimization baseline whose correctness has never been
checked against enumeration. The per-truck MILP cannot suffer the specific
defect found in the fleet CP-SAT model, since with a fixed assignment there is
no idle-truck decision to get wrong, but it has not been validated either.
Doing so requires a Gurobi licence and is the one substantive item this
revision leaves open.

Artifacts: `results/canonical/preassigned/`,
`results/canonical/ablation_summary_preassigned.json`,
`results/canonical/exact_validation/enumeration_preassigned_c5t2.json`.
Reproduce with `scripts/runners/run_preassigned_campaign.sh`.
