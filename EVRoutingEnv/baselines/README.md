# Baselines

The executable baselines are grouped by role, while four selected EVRP
literature baselines are archived in self-contained folders.

## Executable implementations

| Folder | Contents |
|---|---|
| `heuristics/` | Greedy, classical VRP, energy-path, and ALNS policies |
| `optimization/` | Fleet-level CP-SAT, deterministic lower-bound, and perfect-information solvers |
| `legacy/` | Older per-truck Gurobi models retained for inherited experiments |

Import executable baselines through their grouped module paths, for example
`EVRoutingEnv.baselines.heuristics.alns` or
`EVRoutingEnv.baselines.optimization.exact_optimization`.

## Literature snapshots

| Baseline | Folder |
|---|---|
| Subpath-based column generation | `electricrouting_jl/` |
| RoutingBlocks + frvcpy | `routingblocks_frvcpy/` |
| Uncertainty-aware DRL | `uncertainty_aware_evrptw/` |
| Branch-cut-and-price for TDEVRPTW | `tdevrptw_bcp/` |

Each folder contains paper PDFs, plain-text parses, pinned source snapshots, and provenance. These are reference snapshots, not integrated dependencies. Check upstream licenses before use or redistribution.
