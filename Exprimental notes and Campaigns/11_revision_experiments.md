# Revision Experiments: Mask, Baselines, Optimality, Generalization

Updated: 2026-08-17
Status: **in progress.** Sections 1-5 are complete and reproducible from the
artifacts they cite. Sections 6-8 are running.

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

### 1.2 Result

Re-scored on 150 validation scenarios disjoint from the 40 used to pick each
checkpoint, which is the protocol document 10 §6.3 used:

| Arm | Mask | Invalid action | Success | Completed | Travel h |
| --- | --- | --- | --- | --- | --- |
| `v2_tm10` (control) | hard | cannot be selected | 0.700 | 0.976 | 145.0 |
| **`mask_none_terminate`** | structural | strands the truck | **0.793** | 0.982 | 149.5 |
| `mask_none_penalize` | structural | refused, episode continues | 0.000 | 0.521 | n/a |

**The mask is not what produces the result.** Trained without it, under the
simulator's own consequence for an infeasible commitment, the same architecture
reaches 0.793 against the masked control's 0.700 at an identical budget. It
also learns feasibility almost perfectly: on 20 held-out scenarios it averages
**0.1 invalid actions per episode**, and one episode in twenty ends by selecting
an infeasible action.

**But the consequence has to be real.** The charitable variant collapses to
zero. Its failure mode is measurable rather than mysterious: 1.6 refused actions
per episode, and every episode ends either in `no_events_with_unserved_customers`
(14 of 20) or `no_feasible_action` (6 of 20). A refusal advances nothing -- not
the clock, not the truck -- so when every truck refuses in the same round the
event queue empties and the episode deadlocks. Making mistakes cheap removed the
pressure to be feasible without removing the need for it.

The honest reading for the response letter is therefore *not* "masking is
unnecessary". It is:

1. the hard mask is a convenience, not the source of the reported performance;
2. what the policy actually needs is an unambiguous, costly consequence for
   infeasibility -- which the mask provides for free, and which a penalty term
   has to be strong enough to replicate.

Caveat carried into the write-up: these are single seeds. The penalty magnitude
sweep and seed replication of both arms are in section 6.

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
| `v2_tm10` (proposed) | hetero graph | complete GCN | **0.700** | 0.976 | 145.0 |
| `deep_sets__independent` | DeepSets | independent | 0.573 | 0.959 | 155.9 |
| `hetero__independent` | hetero graph | independent | 0.547 | 0.971 | 146.6 |
| `flat__independent` | flat MLP | independent | 0.527 | 0.969 | 148.1 |

Two things follow. First, DeepSets is finally trained to completion -- document
08 dropped it for compute and explicitly refused to draw a conclusion from a
shorter run -- and it is competitive with the graph encoder when both score
actions independently. Second, the gap between the three independent-head arms
(0.527-0.573) and the proposed model (0.700) is larger than the gap between any
two encoders, which locates the contribution in the **action-graph interaction**
rather than in the state encoder.

The constructive attention baseline is a transformer over the node set in the
style of the attention model, with the typed edge features entering as a
per-head attention bias so it reads the same canonical content as the graph
encoder. Kool et al. assume a single vehicle, no charging, and no exogenous
uncertainty, so a literal port is impossible; what transfers is the
architecture, and the deviation is stated rather than hidden. Its run is in
section 6.

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

## 6. Running: seeds, ablations, attention

* seed replication of the travel ladder at seeds 1 and 2 (R1.7);
* the constructive attention baseline (R1.6);
* component ablations for state pooling and for the typed edge relations (E3);
* penalty-magnitude sweep and seed replication for the mask arms.

## 7. Running: the 500-scenario test campaign

The corrected CP-SAT planner, ALNS, the learned baseline family, and the mask
arm are scored against the frozen GraphPPO policy on 500 held-out test
scenarios, with paired bootstrap differences on jointly solved scenarios and a
best-known travel reference.

## 8. Running: generalization

Eighteen regimes labelled `interpolation`, `size_transfer`, and `ood`, each
scoring every method on the same held-out seeds: customer count, fleet size,
charger power and efficiency, battery capacity, vehicle speed, demand, service
time, road distances, and three different uncertainty laws.

A defect surfaced in building this and is worth recording: `step()` decoded
delivery action indices against `len(delivery_sequence)` while the feasibility
engine used the fixed action envelope, so the two disagreed on any instance
smaller than the envelope and selecting the depot raised. No published result is
affected -- the curriculum never changes `num_stops`, so every earlier episode
ran at full size -- but no size-transfer evaluation was possible until it was
fixed.
