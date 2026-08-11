# IEEE T-ITS Revision Campaign

Branch: `campaign/ieee-tits-revision`

Status: **D1–D10 approved on 2026-08-11 — correctness implementation in progress**

This directory is the source of truth for the top-journal revision campaign. The recommended choices in `00_modeling_validation.md` were approved by the author on 2026-08-11. Correctness and modeling implementation may proceed, but large training runs remain gated on the acceptance tests in `02_acceptance_and_reproducibility.md`.

## Documents

- `00_modeling_validation.md`: decisions requiring author validation, with recommendations and consequences.
- `01_experimental_campaign.md`: proposed implementation, baseline, ablation, sensitivity, and generalization campaigns.
- `02_acceptance_and_reproducibility.md`: correctness gates, statistical protocol, artifact requirements, and resubmission definition of done.
- `03_implementation_map.md`: file-level change map, test structure, and dependency order for post-validation implementation.
- `04_reviewer_response_todo.md`: point-by-point editor, reviewer, and coauthor response matrix aligned with D1–D10.
- `05_execution_todo.md`: prioritized implementation and experiment checklist with current evidence.

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
