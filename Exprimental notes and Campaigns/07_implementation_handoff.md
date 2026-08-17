# Implementation Handoff

Updated: 2026-08-11

This document is the restart point for the IEEE T-ITS revision implementation. It records the exact state after the simulator-correctness, action-head, and generic evaluation-runner milestones. Read this file first, then `00_modeling_validation.md`, `05_execution_todo.md`, and `06_environment_verification.md`.

## Repository snapshot

- Repository root: `/home/sorfanoudakis/EVRP`
- Branch: `campaign/ieee-tits-revision`
- HEAD: `f212708053f4aea10a9e9be53f741a03fbb19490`
- HEAD subject: `Add comprehensive reviewer response and execution checklists for IEEE T-ITS revision`
- Worktree: intentionally dirty; all implementation described below is uncommitted.
- Remote tracking branch: `origin/campaign/ieee-tits-revision`
- No commit, push, pull request, expensive training campaign, or manuscript rewrite was performed during this implementation phase.

Do not discard, reset, or overwrite the current worktree. The modified and untracked files are the implementation under review, not disposable build output.

## Authoritative modeling decisions

The author approved the recommended path for D1–D10 on 2026-08-11. `00_modeling_validation.md` is authoritative. In particular:

- joint fleet routing is the primary problem; inherited preassigned routes are a secondary benchmark;
- the primary model uses hard feasibility only, no safety relaxation, no forced fallback, and no headline top-k pruning;
- charging actions target 50/60/70/80/90/100% SoC;
- finite-capacity FCFS queues are part of the model;
- the action architecture comparison is independent scoring versus a truly complete GCN versus self-attention;
- the winning action head must be selected on validation scenarios and frozen before test evaluation;
- failures remain in evaluation aggregates and operational outcomes are reported independently of shaped reward;
- no headline evidence may be selected using final test results.

## Completed implementation

### Joint simulator and correctness foundation

- Fleet-owned customer tasks with atomic claim, in-service, and served transitions.
- Joint assignment, sequencing, stochastic service, charging, and required depot return.
- Payload capacity, remaining payload, one-tour semantics, and partition-feasible generated demand.
- Central hard-feasibility engine with stable rejection reason codes.
- Explicit invalid-action, no-feasible-action, payload-deadlock, energy-realization, time-window-realization, and event-exhaustion failures.
- No hidden primary top-k pruning, feasibility relaxation, or forced action enabling.
- Exact nonlinear target-SoC integration.
- Preserved Level2/DCFast source types and configurable 150/350/750 kW station powers.
- Finite-port FCFS queues with duplicate-wake, stale-waiter, double-charge, cross-station occupancy, closure, and handoff protections.
- Isolated seeded instance, traffic, energy, and service random streams.
- Disabled-by-default controlled hard-time-window variant.
- Randomized energy-accounting checks and 20 seeded complete joint episodes.

Primary configurations:

- `EVRoutingEnv/config_files/config_joint.yaml`
- `EVRoutingEnv/config_files/config_joint_time_windows.yaml`

### Canonical state and evaluation primitives

- `EVRoutingEnv/state/features.py`: versioned canonical truck, customer, charger, action, and global feature rows.
- `EVRoutingEnv/state/representations.py`: padded-set, flat, and complete heterogeneous-graph adapters.
- `EVRoutingEnv/evaluation/metrics.py`: feasibility-first operational episode metrics.
- `EVRoutingEnv/evaluation/statistics.py`: failure-retaining aggregates, Wilson intervals, and paired bootstrap effects.
- `EVRoutingEnv/evaluation/artifacts.py`: disjoint seed namespaces, scenario descriptors, immutable manifests, strict JSON, and hashes.
- `EVRoutingEnv/evaluation/runner.py`: generic immutable evaluation runner with raw JSONL rows, inference timing, aggregate summary, overwrite refusal, manifest/environment identity checks, and explicit incomplete-run evidence.

The runner writes:

1. `manifest.json` before the first scenario;
2. `episode_rows.jsonl.inprogress` while running;
3. `episode_rows.jsonl` only after all requested scenarios finish;
4. `summary.json` as the completion artifact;
5. `failure.json` if execution raises, while preserving the in-progress rows.

### Approved action heads

`algo/action_heads.py` contains a common ragged-batch contract and three implementations:

- `IndependentActionHead`;
- `CompleteGraphGCNActionHead`, with every ordered within-instance action pair and no cross-instance edge;
- `SelfAttentionActionHead`, with attention isolated by `ptr` segment.

`algo/PPO_VariableActionGNN.py` now uses this factory. The training CLI accepts:

```text
--action-head {independent,complete_gcn,self_attention}
--action-head-layers N
--action-attention-heads N
--action-head-dropout P
```

The selected head and its hyperparameters are persisted in `ppo_network_config.json` and restored by `algo/policy_utils.py`.

The PPO mask path now rejects:

- empty hard masks;
- disjoint state/environment masks;
- different mask lengths;
- stored actions outside the feasible set;
- missing or malformed action feature rows.

It no longer enables all actions or silently clamps an invalid stored action.

## Verification evidence

The last complete run passed:

```text
219 passed in 14.85s
```

The suite includes 36 focused ML tests plus three seeded environment-to-action action-head checks and two campaign-runner integration checks. The action tests cover:

- action-order permutation equivariance;
- exact complete-graph edge membership;
- cross-batch isolation;
- singleton and empty pointer segments;
- finite forward values and gradients;
- malformed pointers and non-finite inputs;
- actual heterogeneous actor-critic batches;
- hard-mask refusal behavior;
- real environment state-to-feasible-action selection for every head.

Run verification from the repository root with:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/evrp-matplotlib .venv/bin/pytest -q
.venv/bin/ruff check algo/action_heads.py EVRoutingEnv/evaluation/runner.py tests/unit/test_action_heads.py tests/unit/test_variable_action_policy.py tests/integration/test_action_head_environment.py tests/integration/test_campaign_runner.py
.venv/bin/ruff format --check algo/action_heads.py EVRoutingEnv/evaluation/runner.py tests/unit/test_action_heads.py tests/unit/test_variable_action_policy.py tests/integration/test_action_head_environment.py tests/integration/test_campaign_runner.py
git diff --check
```

Use `PYTHONPATH=.` for pytest in this environment. Without it, the pytest entry point may fail to import `EVRoutingEnv` even when invoked from the repository root.

For coverage on Python 3.13, use one package target and the C trace core:

```bash
COVERAGE_CORE=ctrace PYTHONPATH=. MPLCONFIGDIR=/tmp/evrp-matplotlib .venv/bin/pytest -q --cov=EVRoutingEnv --cov-report=term tests
```

Do not pass several module-level `--cov` targets in one Python 3.13 run; that previously triggered a NumPy extension reload error.

The full test run currently emits third-party deprecation warnings from Torch JIT and PyG's Python 3.13 type inspection. They are not repository test failures, but should be rechecked when dependencies are upgraded.

## Local ML environment

Currently installed in `.venv`:

```text
torch                  2.13.0+cu130
torch_geometric        2.8.0.post1
stable_baselines3      2.9.0
sb3_contrib            2.9.0
wandb                  0.28.1
torch CUDA build       13.0
torch.cuda available   False
```

The repository's tracked `uv.lock` pins earlier compatible versions, including Torch 2.9, PyG 2.7, SB3/sb3-contrib 2.7, and W&B 0.22. The later packages above were installed directly with `uv pip install`; `uv.lock` was not modified. A clean `uv sync --extra test` may therefore restore the locked versions. Re-run the complete test suite after any sync.

There is no working GPU on this host. CPU imports, forward/backward tests, and one-step environment policy checks pass. Do not claim GPU training has been verified.

## Current critical gap: fair pairwise information

This is the first task to resume.

The canonical graph adapter receives pairwise `nominal_energy_kwh`, `nominal_travel_hours`, and `reachable` edge values for all nine truck/customer/charger source-target relations. The current canonical flat and padded-set adapters receive the same node, action, and global rows but do not receive these pairwise values. Training the baselines now would therefore give the graph policy strictly more observable information.

Recommended resolution:

1. Define a stable ordered relation schema for all nine typed relations.
2. Make the pairwise relation tensors part of the canonical semantic snapshot rather than recomputing them only inside the graph adapter.
3. Pad every relation to its configured source/target maxima with masks derived from the entity masks.
4. Include the same padded pairwise values in the flat contract.
5. Provide the same pairwise tensors to the DeepSets/set policy, using a permutation-invariant relation aggregator.
6. Make the graph adapter consume those exact canonical tensors rather than separately querying the transport graph.
7. Add equality tests showing that every valid source-target pair has identical energy, time, and reachability values in flat, set, and graph inputs.
8. Add independent source/target permutation tests and padding-overflow tests.
9. Recalculate `CanonicalShapeSpec.flat_size` and update the Gymnasium observation space and shape assertions.

Do not solve this by simply dropping the graph edges unless explicitly justified as a separate ablation. The recommended primary path is equal information with representation-specific inductive bias.

Likely files:

- `EVRoutingEnv/state/features.py`
- `EVRoutingEnv/state/representations.py`
- `EVRoutingEnv/models/environment/event_driven_env.py`
- `tests/unit/test_canonical_features.py`

Acceptance criteria for this next task:

- one canonical source owns all pairwise values;
- flat, padded-set, and graph adapters expose identical valid pairwise semantics;
- all padded values are finite and masked;
- no adapter silently truncates entities or relations;
- the full suite and Gymnasium checker pass.

## Work after information parity

Proceed in this order:

1. Implement canonical flat, DeepSets, and heterogeneous-graph state encoders.
2. Feed the same seven-dimensional canonical action rows and hard mask to all three approved action heads.
3. Add end-to-end forward/backward and permutation tests for every state-encoder/action-head combination.
4. Add canonical PPO, MaskPPO, DeepSets-PPO, state-GNN PPO, and GraphPPO training entry points with equivalent information and budgets.
5. Migrate inherited training/evaluation scripts to `run_evaluation_campaign` and the immutable artifact contract.
6. Implement tiny exact/bounded fleet optimization and exhaustive cross-checks.
7. Add ALNS and constructive attention baselines.
8. Run short CPU smoke train/evaluate cycles, then resolve GPU access and repeat smoke verification.
9. Select the action head on the predefined validation suite and freeze it.
10. Only then run headline campaigns and revise LaTeX claims, tables, figures, and reviewer responses.

## Still open

- Station-specific charging efficiency is not loaded from empirical station data.
- The same nonlinear charging integrator is not yet shared by every heuristic and optimization baseline.
- Legacy preassigned-route code still contains historical recovery behavior; the verified no-fallback guarantee applies to the primary joint hard-feasibility path and the corrected PPO mask path.
- Inherited runners are not yet migrated to the generic immutable runner.
- Exact fleet optimization, rolling horizon, ALNS, constructive attention, and CI are absent.
- Publication-grade stochastic distribution diagnostic artifacts have not been generated.
- No short train/evaluate cycle, GPU training, architecture selection, or full evidence campaign has run.
- The manuscript and response letter have not been revised against final evidence.

## Worktree map

Important untracked implementation files:

- `EVRoutingEnv/config_files/config_joint.yaml`
- `EVRoutingEnv/config_files/config_joint_time_windows.yaml`
- `EVRoutingEnv/evaluation/`
- `EVRoutingEnv/state/features.py`
- `EVRoutingEnv/state/representations.py`
- `algo/action_heads.py`
- `tests/integration/test_action_head_environment.py`
- `tests/integration/test_campaign_runner.py`
- new unit tests under `tests/unit/`
- `06_environment_verification.md` and this handoff document

Important modified integration points:

- `EVRoutingEnv/models/environment/event_driven_env.py`
- `EVRoutingEnv/models/environment/event_handlers.py`
- `EVRoutingEnv/models/environment/joint_instance.py`
- `EVRoutingEnv/models/core/customer.py`
- `EVRoutingEnv/models/core/truck.py`
- simulation charging, traffic, delivery, and station modules
- `EVRoutingEnv/state/feasibility.py`
- `algo/PPO_VariableActionGNN.py`
- `algo/policy_utils.py`
- `scripts/training/train_PPO_Variable_parallel.py`

Before editing, run `git status --short` and preserve every current change. Do not use destructive Git commands. No commit has been made because commit/push authorization was not requested.

## Training and publication gate

Do not start expensive training or change manuscript claims yet. The immediate go/no-go sequence is:

- close pairwise information parity;
- integrate canonical policies;
- migrate artifacts;
- validate exact and heuristic baselines;
- run short train/evaluate smoke tests;
- verify GPU execution;
- freeze architecture on validation only.

The detailed campaign matrix and resubmission definition of done remain in `01_experimental_campaign.md` and `02_acceptance_and_reproducibility.md`.

