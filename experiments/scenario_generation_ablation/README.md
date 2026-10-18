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

## Texas reset and delivery-hop audit

For a focused audit of the current generator on Texas, run:

```bash
.venv/bin/python scripts/analysis/analyze_scenario_resets.py \
  --case 10:10 --case 100:5 --resets 100 --seed-start 0 \
  --output experiments/scenario_generation_ablation/results/texas_reset_audit
```

Each `--case` is `TRUCKS:DELIVERIES_PER_TRUCK`. The script reuses an environment per case, disables scenario disk caching, and separates initialization, reset, construction, and certification timings. Metric extraction runs outside reset timing. Payload capacity scales with deliveries per truck; other settings, including the 200-hour horizon, come from `config_joint_texas.yaml`. Use `--config` for another map or configuration.

The audit follows hidden witness routes and ownership. It records direct customer-to-customer/depot delivery hops, actual delivery travel through chargers, individual navigation legs, and final depot returns. Road kilometres from compatible stored OD lengths, nominal energy/travel time, and the simulator's distance proxy are separate fields. Outputs include raw reset/truck/hop CSVs, complete scenario/witness JSONL, distribution summaries, PNG/SVG plots, resolved configs, and a provenance manifest.

The historical `route-walk-v1` [100-reset run](results/texas_reset_audit_20261018/report.md) uses seeds 0–99. Texas supports at most 233 unique deliveries plus a depot: the 10-truck × 10-delivery case generated 100 certified scenarios, while every requested reset for 100 trucks × 5 deliveries was retained as an explicit capacity rejection. The audit never substitutes synthetic nodes or omits failed seeds.

The current `route-walk-v2-single-charger` generator permits at most one station between consecutive depot/customer stops, including the final depot return. The updated [single-station audit](results/texas_single_charger_audit_20261018_v2/report.md) again uses 100 seeds for 10 trucks × 10 deliveries and 100 trucks × 2 deliveries. All 100 resets in the first case certified, with no consecutive station visits in any of its 1,000 truck routes. The second case is rejected under this stricter rule: an independent energy-only [all-depot bound](results/texas_single_charger_audit_20261018_v2/energy_bound_notes.md) admits at most 198 distinct deliveries in pairs under the configured 1.20 energy multiplier. The audit records `max_chargers_between_stops` per truck and rejects any witness that violates the limit.

## Map capacity with one station per segment

Compute one- and two-delivery energy limits over every depot with:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/analysis/analyze_scenario_capacity.py \
  --config EVRoutingEnv/config_files/config_joint.yaml EVRoutingEnv/config_files/config_joint_texas.yaml \
  --output experiments/scenario_generation_ablation/results/map_capacity
```

The analysis assumes full initial batteries and full charging, retains the conservative traffic-energy multiplier, and solves maximum-cardinality matching for distinct two-delivery routes. It ignores queues, time and payload, so actual seeded certification is audited separately. Limits are specific to the stated deliveries per truck; longer routes can use additional customer stops on the way home.

The completed [capacity and reset comparison](results/single_charger_capacity_20261018/report.md) finds a California maximum of 112 trucks × 2 deliveries (224 customers), with 41/100 certified resets at that size and 97/100 at 100 trucks × 2 deliveries. Its one-delivery energy maximum is 209 trucks, and its raw customer-node ceiling is 231. Texas 98 trucks × 2 deliveries certified 2/100 resets: the energy limit admits this size, but the bounded random walk rarely finds the required disjoint routes near capacity. Every accepted route in these audits respects the station limit.
