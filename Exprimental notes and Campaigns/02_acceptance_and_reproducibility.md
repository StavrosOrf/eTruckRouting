# Acceptance Gates and Reproducibility Standard

## Gate A — Model correctness

- [ ] Every customer is served exactly once in successful episodes.
- [ ] Vehicle payload never exceeds capacity.
- [ ] Battery state follows travel and charging updates within numerical tolerance.
- [ ] A truck never teleports or executes an action in an invalid state.
- [ ] Depot-return rules are enforced consistently.
- [ ] FCFS queue order and station port capacity are invariant-tested.
- [ ] No hidden fallback converts an infeasible state into a nominally feasible action.
- [ ] Charging integration is monotone, capacity-bounded, and validated against reference curves.

## Gate B — Stochastic correctness

- [x] Same scenario ID and policy actions reproduce identical exogenous outcomes.
- [x] Different scenario IDs produce distinct keyed samples; distribution-level validation remains under the next item.
- [x] Policy RNG consumption is isolated from the exogenous scenario streams.
- [ ] Empirical distributions match configured moments and clipping rules.
- [ ] Scenario seed and generation version are exposed in episode `info`; persist them in every evaluation artifact.

## Gate C — Comparison fairness

- [ ] All policies receive equivalent observable information.
- [ ] All policies operate on equivalent task instances and action semantics.
- [ ] Candidate pruning and feasibility masks are identical unless they are the explicit ablation.
- [ ] Training steps, evaluation frequency, and tuning budget are documented.
- [ ] Checkpoint selection uses validation only.
- [ ] Test results cannot alter method, checkpoint, seed, or scenario selection.
- [ ] Solver assumptions and compute limits are disclosed.

## Gate D — Statistical evidence

- [ ] Independent training seeds are the unit for algorithm-training variability.
- [ ] Common test scenarios support paired comparisons.
- [ ] Success is reported with Wilson confidence intervals.
- [ ] Continuous paired differences use bootstrap confidence intervals.
- [ ] Effect sizes accompany significance tests.
- [ ] Multiple comparisons are controlled or explicitly treated as exploratory.
- [ ] Failed episodes are retained and analyzed by cause.
- [ ] Operational metrics conditional on success are not mixed silently with failures.

## Gate E — Reproducible artifacts

Every training run must save:

- immutable resolved configuration;
- git commit and dirty-worktree indicator;
- dependency/environment snapshot;
- training, validation, and test seed namespaces;
- algorithm and environment hyperparameters;
- checkpoint-selection record;
- learning curves and validation history;
- model checkpoints;
- hardware and timing data;
- failure/fallback counters.

Every evaluation must save:

- policy/checkpoint identity;
- scenario IDs and per-episode seeds;
- raw per-episode metrics;
- aggregate metrics and confidence intervals;
- failure causes;
- solver status and gaps where applicable;
- exact script/config command needed to reproduce it.

## Gate F — Manuscript consistency

- [ ] Problem name matches the implemented decisions.
- [ ] Every equation matches tested implementation behavior.
- [ ] Node, edge, action, and network dimensions match saved configurations.
- [ ] Reward coefficients and evaluation objectives are fully separated and disclosed.
- [ ] `Optimal`, `near-optimal`, and `generalization` are used only when supported by appropriate evidence.
- [ ] Scenario counts and training-seed counts are stated separately.
- [ ] Every table and figure is traceable to a final artifact manifest.
- [ ] LaTeX compiles without missing citations/references and passes the selected linter.

## Current execution constraint

At campaign creation, `nvidia-smi` could not communicate with an NVIDIA driver in the current execution environment. This does not block design and implementation, but GPU training must not be reported as launched until the target compute environment is identified and verified.

## Resubmission definition of done

The campaign is complete only when:

1. all author modeling decisions are recorded;
2. Gates A–F pass with saved evidence;
3. main, ablation, sensitivity, robustness, and generalization campaigns are complete;
4. all reviewer comments have a point-by-point response linked to new text or evidence;
5. the revised manuscript makes only claims directly supported by the final experiments;
6. code, configurations, checkpoints, raw results, and manuscript artifacts are reproducible from the final commit.
