# Acceptance Gates and Reproducibility Standard

## Gate A — Model correctness

- [x] Every customer is served exactly once in successful episodes (unit, end-to-end, atomic-claim, and 20-scenario regression tests).
- [x] Vehicle payload never exceeds capacity; generated demands have a guaranteed feasible partition.
- [x] Battery state follows travel and charging updates within numerical tolerance in randomized unit sequences and 20 seeded complete episodes.
- [ ] A truck never teleports or executes an action in an invalid state.
- [x] Depot-return rules are enforced in the primary joint model and covered across stochastic scenarios.
- [x] FCFS queue order, duplicate wakes, stale waiters, and station port capacity are invariant-tested.
- [x] Primary joint actions use explicit hard-feasibility reasons; invalid actions cannot silently reroute or charge.
- [x] Target-SoC charging integration is monotone, exact-target, and capacity-bounded; external reference-curve validation remains for the manuscript campaign.
- [x] The controlled hard-time-window variant waits on early arrival, rejects nominally impossible actions, and records realized late-arrival failure without enabling windows in the base campaign.
- [x] The primary joint, time-window, and legacy configurations satisfy Gymnasium's reset/step/space contract.

## Gate B — Stochastic correctness

- [x] Same scenario ID and policy actions reproduce identical exogenous outcomes.
- [x] Different scenario IDs produce distinct keyed samples; distribution-level validation remains under the next item.
- [x] Policy RNG consumption is isolated from the exogenous scenario streams.
- [x] Deterministic Monte Carlo checks cover clipping, approximate means, rush/business-hour variance effects, and positive travel-energy correlation.
- [x] A versioned full scenario descriptor, including config hash, instance, chargers, uncertainty configuration, and RNG version, is exposed in episode `info`.
- [ ] Wire the descriptor into every training/evaluation runner and preserve it beside raw episode rows.

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

The lightweight `.venv` also does not currently contain PyTorch, PyTorch Geometric, Stable-Baselines3, or SB3-Contrib. Environment verification is complete without them, but action-head tests and learning smoke runs require a resolved ML environment. On Python 3.13, invoke coverage with one package target and `COVERAGE_CORE=ctrace`; multiple module-level `--cov` targets trigger a NumPy double-import failure in the current coverage stack.

## Resubmission definition of done

The campaign is complete only when:

1. all author modeling decisions are recorded;
2. Gates A–F pass with saved evidence;
3. main, ablation, sensitivity, robustness, and generalization campaigns are complete;
4. all reviewer comments have a point-by-point response linked to new text or evidence;
5. the revised manuscript makes only claims directly supported by the final experiments;
6. code, configurations, checkpoints, raw results, and manuscript artifacts are reproducible from the final commit.
