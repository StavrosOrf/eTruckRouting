# Environment Verification and Immediate Handoff

Updated: 2026-08-11

## Verified environment surface

- [x] Primary joint fleet assignment, sequencing, charging, service, and depot return.
- [x] Payload-capacity feasibility and generated-demand partition feasibility.
- [x] Hard feasibility masks with stable rejection reasons and no hidden fallback.
- [x] Exact target-SoC charging with nonlinear taper and 150/350/750 kW station classes.
- [x] Finite-port FCFS queues, simultaneous handoff, stale-wake protection, and closures.
- [x] Seeded travel, energy, and service uncertainty with isolated random streams.
- [x] Disabled-by-default hard time windows with early waiting and realized-late failure.
- [x] Versioned canonical flat, padded-set, and heterogeneous-graph representations.
- [x] Feasibility-first operational metrics independent of shaped training reward.
- [x] Explicit failure for invalid actions, no feasible action, payload deadlock, energy realization, time-window realization, and unserved-customer event exhaustion.

## Reproducible verification commands

Run from the repository root:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/evrp_matplotlib .venv/bin/pytest -q
.venv/bin/ruff check EVRoutingEnv/state EVRoutingEnv/evaluation tests/unit tests/integration
COVERAGE_CORE=ctrace MPLCONFIGDIR=/tmp/evrp_matplotlib .venv/bin/python -m pytest -q --cov=EVRoutingEnv --cov-report=term tests
```

Current evidence:

- 219 tests pass.
- All three action heads pass permutation-equivariance, complete-edge, cross-batch isolation, empty/singleton-set, finite-gradient, actor-critic batch, and seeded environment-to-action checks.
- Primary joint, joint time-window, and legacy configurations pass Gymnasium's environment checker.
- Critical new modules have focused coverage: feasibility 90%, canonical extraction 94%, canonical representations 88%, artifact support 91%, campaign statistics 89%, and operational metrics 100%. Repository-wide coverage is 61% because plotting, curriculum, and inherited legacy modules are not exercised by this correctness suite.

## Known constraints—not silently treated as supported

- [x] Battery/energy conservation has deterministic randomized checks across mixed travel/charging sequences and seeded complete episodes; expand to generated mixed event schedules if failures surface during smoke runs.
- [ ] Station-specific efficiency is not yet loaded from the empirical station data; port counts are preserved.
- [x] Stochastic clipping, approximate means, rush/business-hour variance, and travel-energy correlation have deterministic Monte Carlo regression tests; save publication-grade diagnostic plots/tables during the smoke campaign.
- [ ] The inherited learning stack does not yet consume the canonical set/graph adapters end to end.
- [ ] The canonical graph adapter includes pairwise energy/time/reachability edges that the current flat and padded-set adapters do not expose. This must be equalized or declared as a separate edge-information ablation before fair baseline training.
- [x] The inherited adjacent-action chain is replaced by selectable independent, truly complete-GCN, and self-attention heads; the validation campaign still must select and freeze one head.
- [x] PyTorch 2.13, PyG 2.8, Stable-Baselines3 2.9, sb3-contrib 2.9, and W&B 0.28 import successfully in `.venv`.
- [ ] A working GPU environment is not present: the Torch build includes CUDA 13.0 support, but `torch.cuda.is_available()` is false on this host.
- [x] A generic immutable evaluation runner now connects manifests, full scenario descriptors, raw failure-inclusive rows, operational metrics, inference timing, and aggregates; interrupted runs retain `.inprogress` rows plus `failure.json`.
- [ ] Exact fleet optimization, ALNS, constructive attention, inherited-runner migration, and CI remain open.

## Next implementation sequence

1. [x] Implement the three approved permutation-equivariant action heads and cross-batch isolation tests.
2. [ ] Close pairwise information parity, then connect PPO, MaskPPO, DeepSets-PPO, state-GNN PPO, and GraphPPO to the canonical representation contract.
3. [ ] Migrate every inherited runner to the new artifact contract; the generic evaluation runner already wires manifests, scenario descriptors, raw rows, failure-retaining aggregation, Wilson summaries, and incomplete-run evidence. Paired cross-policy summaries remain campaign-level work.
4. [ ] Produce versioned uncertainty diagnostics and add generated mixed-event property tests during smoke validation.
5. [ ] Implement tiny exact/bounded fleet optimization and exhaustive cross-checks, then ALNS and constructive attention baselines.
6. [ ] Resolve GPU access and run one short train/evaluate smoke cycle per learning method; the CPU ML stack and one-step action selection are verified.
7. [ ] Freeze validation-selected action head/configurations before any headline test scenarios.
8. [ ] Run the paired evidence campaigns and only then revise manuscript claims, tables, figures, and the response letter.

## Training gate

Do not launch the expensive campaign yet. The simulator foundation and action-head implementations are usable, but training remains gated by canonical policy integration, runner/artifact wiring, baseline validation, smoke train/evaluate cycles, GPU verification, and validation-only architecture selection.
