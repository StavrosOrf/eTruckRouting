# Scenario-generation mode ablation

This study compares the three route-assignment modes by generating certified EVRP instances, rather than evaluating a learned policy. Each episode reset must produce and replay a complete feasible fleet witness under its seeded simulator randomness.

## Experimental grid

| Factor | Values |
| --- | --- |
| Modes | `joint_fleet` (flexible ownership/order), `preassigned` (fixed ownership/flexible order), `sequential` (fixed ownership/order) |
| Trucks | 1, 10, 100 |
| Deliveries per truck | 3, 10, 20 |
| Seeds | 1 through 100, reused across modes within each size cell |
| Requested resets | 2,700 (3 modes × 3 fleet sizes × 3 workload levels × 100 seeds) |

Each cell's total delivery count is `trucks × deliveries per truck`, ranging up to 2,000. The comparison is paired by integer seed within each fleet/workload cell. A row is counted as certified only if `reset(seed=...)` returns a nonempty complete witness produced by the environment's isolated event-engine replay. A failed reset is retained in the raw file and lowers the certificate rate.

The environment requires a unique graph node for each delivery, while the bundled road graph has 258 nodes. Cells requiring more customer nodes than the map supports use a synthetic bidirectional hub-and-spoke network with 25 chargers and 40 kWh / 0.25 h hub-to-leaf arcs. The raw data records the network variant and synthetic customer-node count. Results from these cells are workload-scale feasibility scenarios, not measured geographic routes; their distance values are model proxies.

## Reproduce

From the repository root, run:

```bash
.venv/bin/python experiments/scenario_generation_ablation/run_scenario_grid.py
```

The default output is `experiments/scenario_generation_ablation/results/seeds_1_100_per_truck/`. The runner refuses to overwrite an existing output directory. Use `--seed-start` and `--seed-count` for a separate replicate, or `--output` to select a new directory.

## Outputs

- [`results/seeds_1_100_per_truck/report.md`](results/seeds_1_100_per_truck/report.md): compact tables of the headline scenario and certificate statistics, with embedded plots.
- `results/seeds_1_100_per_truck/plots/`: SVG plots for certification rate, distance proxy, and witness energy.
- `scenario_records.csv`: one row per requested reset, including seed, mode, node IDs, hidden witness ownership/routes, certificate size, and feasibility-related measurements. Failure rows include the exception and are not omitted.
- `summary.csv` and `summary.json`: mode × fleet × delivery aggregates. Continuous measures include mean, sample standard deviation, and p90 over certified scenarios. The certificate rate uses all requested seeds.
- `manifest.json`: exact seeds, resolved per-cell configs, grid, mode definitions, configuration hash, code revision/worktree status, runtime, and distance definition.

## Measures and interpretation

The runner reports graph/customer/charger counts; demand distribution and fleet payload utilization; initial state of charge; number of generator candidates tried; certificate action count; certificate trucks and trucks without assigned deliveries; navigation hops; charger visits/unique chargers/charging actions; certificate route travel time and nominal energy; final navigation-leg energy on delivery arrivals; and maximum per-truck payload utilization. Plots show certificate rates, witness distance proxy, and nominal route energy, with standard deviations across seeds.

`witness_distance_proxy_km` uses the same distance calculation as simulator actions: shortest-path travel time multiplied by configured truck speed. The bundled road graph does not contain measured road kilometres, so this field is a model-implied distance proxy, not geographic ground truth. Energy and time use the graph's shortest-path energy and travel-time weights. Witness routes are feasibility certificates. In `joint_fleet`, their ownership/order must not be interpreted as constraints imposed on the policy.

Only the scenario seed and requested workload are varied. The road graph, truck/battery capacity, charging configuration, demand range, time-window setting, traffic and service stochasticity come from `config_joint.yaml`. Payload capacity scales to four demand units per requested delivery per truck, and the study uses a generous 5,000-hour horizon so the feasibility audit does not impose the base campaign's smaller workload limit. These are documented in the resolved config in `manifest.json`. This is a scenario-generation comparison, not an ablation of policy performance or environment components.
