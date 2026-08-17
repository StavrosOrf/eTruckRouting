# IEEE T-ITS Revision Campaign

Branch: `campaign/ieee-tits-revision`

Status (2026-08-17): **correctness complete, primary campaign complete, supporting evidence and manuscript open**

This directory is the source of truth for the top-journal revision campaign. The recommended choices in `00_modeling_validation.md` were approved by the author on 2026-08-11; correctness implementation and the primary campaign are now done.

Where the work stands, in one table — `05_execution_todo.md` carries the item-level state and `04_reviewer_response_todo.md` the reviewer-level state:

| Block | State |
| --- | --- |
| Simulator, feasibility, queues, charging semantics | complete, 281 tests pass |
| Canonical observation, encoders, action heads, artifact contract | complete |
| Primary campaign under the travel-time objective | complete (doc 10) |
| Mask, pooling, active-truck ablations | not started, needs code |
| ALNS, attention/transformer, learned-baseline family | not started |
| Generalization, multi-seed, scale grid | not started |
| Manuscript and response letter | not started |

## Documents

- `00_modeling_validation.md`: decisions requiring author validation, with recommendations and consequences.
- `01_experimental_campaign.md`: proposed implementation, baseline, ablation, sensitivity, and generalization campaigns.
- `02_acceptance_and_reproducibility.md`: correctness gates, statistical protocol, artifact requirements, and resubmission definition of done.
- `03_implementation_map.md`: file-level change map, test structure, and dependency order for post-validation implementation.
- `04_reviewer_response_todo.md`: point-by-point editor, reviewer, and coauthor response matrix aligned with D1–D10.
- `05_execution_todo.md`: prioritized implementation and experiment checklist with current evidence.
- `06_environment_verification.md`: supported environment semantics, verification commands, known constraints, and the immediate handoff checklist.
- `07_implementation_handoff.md`: exact branch/worktree snapshot, completed code, dependency state, restart commands, critical information-parity gap, and ordered resumption steps.
- `08_canonical_learning_and_baselines.md`: the canonical observation, the encoder/head sweep, the tuned baselines, and the warm-started learning results.
- `09_graphppo_from_scratch_campaign.md`: GraphPPO trained from random initialisation with no imitation, the curriculum that made it work, and the makespan-era headline result.
- `10_travel_time_objective_campaign.md`: **complete.** The objective moves from fleet makespan to **total fleet travel time**. Diagnosis of where GraphPPO lost, the three misalignments fixed (measurement, reward, per-action features), the re-tuned baselines, the four-stage campaign, and the final test-split result: GraphPPO at 119.9 fleet hours and 0.857 success, level with the CP-SAT planner on paired travel hours while solving 25 points more instances, and best on every column of the best-known travel reference.

## Current branch baseline

The branch was created from `Restructuring`. The worktree was clean at branch creation. The audit identified the following blocking issues in the inherited implementation:

1. stochastic perturbations are not keyed by episode seed;
2. flat PPO baselines receive fewer delivery slots than GraphPPO;
3. the claimed fully connected action graph is implemented as an adjacent-node chain;
4. the mathematical reference is per-truck and deterministic, and ignores charger contention;
5. reward can remain high after infeasible partial completion;
6. checkpoint and scenario selection are not sufficiently controlled;
7. the manuscript overstates both problem scope and optimality/generalization evidence.

These are treated as correctness and study-design issues, not as opportunities to tune selectively for better results.

The older `notes/reviewer_revision_plan.md` was the initial audit and recommended a narrower preassigned-route scope. D1 supersedes that recommendation: joint fleet routing is now the primary formulation, while the preassigned mode is retained as a secondary decomposition benchmark.

## Campaign principle

The target is a defensible, high-impact study. The campaign will test where GraphPPO is strong, where masking provides most of the gain, and where other architectures or solvers are superior. All headline comparisons will be feasibility-first, paired on common scenarios, and reported across independent training seeds.
