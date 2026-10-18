<div align="center">

# eTruckRouting

### Learning to Route Electric Truck Fleets Under Nonlinear Models and Operational Uncertainty

**A research platform for charging-aware electric truck routing with event-driven simulation, graph reinforcement learning, and optimization baselines.**

<p>
  <img alt="Python 3.13" src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Gymnasium" src="https://img.shields.io/badge/Gymnasium-Environment-008A7A?style=for-the-badge">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-RL-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="PyTorch Geometric" src="https://img.shields.io/badge/PyG-Graph%20Networks-6C5CE7?style=for-the-badge">
  <img alt="uv" src="https://img.shields.io/badge/uv-Reproducible%20Runs-2E3440?style=for-the-badge">
  <img alt="Gurobi" src="https://img.shields.io/badge/Gurobi-Optimization-CC0000?style=for-the-badge">
</p>

<p>
  <a href="#quick-start"><strong>Quick Start</strong></a> |
  <a href="#method-ppo-over-state-and-action-graphs"><strong>Method</strong></a> |
  <a href="#evaluation"><strong>Evaluation</strong></a> |
  <a href="#citation"><strong>Citation</strong></a>
</p>

<p>
  <em>Electric fleets. Nonlinear charging. Stochastic operations. Graph-based routing intelligence.</em>
</p>

</div>

> **Paper:** [Learning to Route Electric Truck Fleets Under Nonlinear Models
> and Operational Uncertainty](https://arxiv.org/abs/2510.12335)

## At a Glance

- **Project:** **eTruckRouting** studies how electric truck fleets can route,
  recharge, and recover from operational uncertainty.
- **Problem:** **Electric Vehicle Routing Problem** with charging,
  traffic, battery limits, and delivery deadlines.
- **Core idea:** Learn policies that decide **where to drive** and
  **how long to charge** in an event-driven simulator.
- **Environment:** **Gymnasium-compatible** electric truck simulator
  with stochastic travel time, energy uncertainty, charger queues, and realistic
  charging curves.
- **Learning method:** **PPO + graph neural networks** over
  heterogeneous routing states and variable feasible-action graphs.
- **Baselines:** **Heuristics, classical VRP methods, SB3 agents, and
  Gurobi optimizers** for controlled comparisons.
- **Research use:** Designed for **paper experiments** on scalability,
  generalization, charging behavior, and policy robustness.

## What Is in This Repository?

- **Event-driven EV routing simulator** with a global clock, per-truck event
  queues, stochastic traffic, energy uncertainty, unloading-time models, charger
  queues, and route completion/failure accounting.
- **Graph-based RL policies** built around PPO and PyTorch Geometric, including
  a variable-action actor that scores feasible delivery and charging choices
  through an action graph.
- **Flexible and sequential delivery modes** for both classic EV routing and
  single-truck VRP-style experiments.
- **Curriculum learning support** for training over changing numbers of trucks
  and delivery stops.
- **Baselines** including heuristic routing, Clarke-Wright savings, nearest
  neighbor with 2-opt, Stable-Baselines3 PPO/DQN/QRDQN/MaskablePPO, and
  Gurobi-based optimal or robust planners.
- **Evaluation and analysis scripts** for policy comparison, grid
  generalization, route maps, schedule analysis, network inspection, and
  debugging.

## Research Framing

Electric vehicle routing differs from classical routing because feasibility is
not just geometric. A route can fail because a truck arrives with too little
energy, because charging takes too long, because charger queues alter the
effective schedule, or because stochastic traffic changes both time and
consumption. This repository models those interactions directly:

1. Trucks move through a transportation graph with energy and time costs.
2. Charging stations expose heterogeneous charger types and waiting behavior.
3. Each episode advances by events rather than by fixed simulation ticks.
4. The active decision maker controls the truck that is currently ready.
5. Policies receive graph-structured state and an action mask over feasible
   delivery or charging actions.

This makes the environment useful for studying RL policies that must generalize
across fleet size, number of stops, charger density, traffic uncertainty, and
delivery-order flexibility.

Routing uses an explicit minimum-energy path for each origin and destination.
Energy consumption and travel time are both summed along that same path, and
the simulator, feasibility masks, features, and planners share these costs.
Edges with negative or nonfinite energy or travel time are excluded because the
movement model requires nonnegative consumption; a valid alternate path is used
when available. Transport caches are versioned so older independent cost tables
cannot be reused.

Run the complete test suite with:

```bash
uv run pytest
```

The test session limits PyTorch CPU threads for the small neural networks used
by its learning checks, then restores the original thread setting.

Run the routing logic and scale regressions separately with:

```bash
uv run pytest tests/integration/test_routing_modes_scale.py
```

These cover joint assignment, preassigned ownership, and sequential routes with
1, 3, 10, 100, and 128 trucks and 1, 2, 5, 20, and 25 total customers, plus
128-truck/150-customer replay and observation checks. Scheduling cases isolate
routing with ample battery; separate cases require charging, send 128 trucks to
a shared station, close a station with 127 waiting trucks, and run the bundled
road network at the standard 400 kWh battery capacity. Checks include ownership,
order, unique service, payload and battery accounting, FCFS queues, exact episode
boundaries, reset, stochastic replay, malformed actions, and terminal rewards.

## Repository Layout

```text
EVRoutingEnv/
  models/
    core/                  # Truck and transportation-graph primitives
    environment/           # Event-driven and curriculum environments
    simulation/            # Traffic, charging-curve, charger, delivery models
  state/                   # Linear, heterogeneous GNN, VRP, and action-mask states
  baselines/               # Heuristic, classical VRP, and Gurobi policies
  config_files/            # Experiment configuration files
  data/case_studies/       # Case studies, one folder each: california (default), texas, pacific_northwest, seattle, chicago
  utils/                   # Plotting, statistics, config, charging logs
algo/
  PPO_VariableActionGNN.py # Variable-action PPO with action-graph head
  PPO_actionGNN.py         # PPO actor-critic over GNN state
  networks.py              # Neural-network components
scripts/
  training/                # PPO, curriculum, and SB3 training entry points
  evaluation/              # Policy comparison, parallel eval, generalization
  analysis/                # Route, network, schedule, and state inspection
  runners/                 # tmux/grid experiment launchers
```

## Installation

This project uses `uv` for reproducible Python environments.

```bash
git clone <repository-url>
cd EVRP
uv sync
```

`uv sync` installs the simulator, learning and solver dependencies (including
OR-Tools), plus the default development group with pytest, coverage, and Ruff.
Run tests with `uv run pytest`; use `uv sync --no-dev` for runtime dependencies only.
The `test` extra remains available for `uv sync --extra test`.

The package currently targets Python `>=3.13,<3.14`. Some baselines require
additional system support:

- Gurobi baselines require `gurobipy` and a valid Gurobi license.
- GPU training requires a CUDA-compatible PyTorch installation.
- Weights & Biases logging is enabled by many training scripts unless
  `--no-wandb` is provided.

## Quick Start

Run a short custom PPO-Variable training job without external logging:

```bash
uv run python scripts/training/train_PPO_Variable_parallel.py \
  --config EVRoutingEnv/config_files/config_joint.yaml \
  --gnn-state-space vrp \
  --num-trucks 1 \
  --num-stops 10 \
  --max-timesteps 100000 \
  --eval-freq 5000 \
  --num-parallel-envs 4 \
  --num-eval-envs 4 \
  --no-wandb
```

Train a Stable-Baselines3 policy:

```bash
uv run python scripts/training/train_sb3_event_driven.py \
  --algo maskppo \
  --config EVRoutingEnv/config_files/config_joint.yaml \
  --num-trucks 1 \
  --num-stops 10 \
  --steps 100000 \
  --no-wandb
```

Run curriculum learning:

```bash
uv run python scripts/training/train_curriculum.py \
  --curriculum-config scripts/training/configs/curriculum_config_staged.json \
  --exp-name curriculum_staged_seed0 \
  --seed 0 \
  --no-wandb
```

The training scripts save models and run metadata under `saved_models/`.

## Configuration

Environment behavior is controlled through YAML files in
`EVRoutingEnv/config_files/`.

- `config_joint.yaml`: `problem.mode: joint_fleet`, for one or more trucks.
  Trucks choose customer ownership and visit order. Set `problem.assignment:
  preassigned` (or use `config_joint_preassigned.yaml`) to fix ownership while
  still choosing visit order.
- `config_sequential.yaml`: `problem.mode: sequential`, with fixed customer
  ownership and visit order. Trucks choose charging stops and charging amounts.
  `config.yaml` contains the same sequential defaults.

Both modes use one fleet task registry, a shared depot, payload limits, service
completion events, charger queues, and required depot return. In all three
variants, a ready truck may return to the depot before deliveries are finished;
arrival permanently retires that truck. Selecting the depot while already there
retires it immediately. Other trucks continue, and the scenario succeeds only
if every customer is served and every truck finishes at the depot. Outstanding
preassigned deliveries stay with their original owner after retirement.
`environment.num_stops` is the **total fleet customer count** in both modes;
it is never multiplied by the truck count.
Single-truck flexible routing is covered by `joint_fleet`; its separate legacy
configuration and independent-route loader have been removed. The old
`delivery.enable_flexible_delivery_order` setting is rejected: select the mode
through `problem.mode`.

Scenarios use route-first random walks (`route-walk-v2-single-charger`): choose a shared depot,
build one closed walk per truck, and select unique delivery nodes along those
walks. Every segment between consecutive depot/customer stops visits at most
one charging station, including departure from and return to the depot. The
sampler chooses a station and the next delivery together, and reserves energy
for a direct or one-station depot return, optionally via another delivery.
Demands fit each witness truck's payload. Flexible-mode task IDs are
shuffled so they do not disclose the hidden witness order or fleet assignment;
sequential tasks retain their public prescribed order.

Map/vehicle energy lookups are prepared at environment initialization and reused
across resets. Small random proposal batches and swap-delete customer pools
avoid solving routes for randomly scattered customers. A bounded full-pool scan
is used when a proposal batch misses all feasible customers. Generation therefore
scales with workload and map size; constant millisecond latency for arbitrary
fleet/customer counts is not guaranteed. The fixed road graph must still contain
enough distinct eligible nodes for all customers and the depot.

Every accepted reset has a complete feasible fleet witness, replayed on isolated
simulator state using that scenario's stochastic seed. Certification includes
initial SoC, payload/ownership/order, configured charging actions and station
queues, hard service windows, final depot returns, and time/action limits.
Candidates without a successful replay are resampled, up to
`environment.generation_attempts` (default 64); exhaustion raises a descriptive
error without relaxing constraints. This bounded constructive search may reject
a configuration for which another feasible route exists. The guarantee is
existence of a solution for the seeded scenario, not that every policy or every
possible stochastic realization succeeds. `info["instance_feasibility"]` reports
certification; `env.feasibility_witness` contains replayable `(truck_id, action)`
pairs for debugging, outside policy observations.
Private replay uses the same action validators and event handlers, omitting
policy observations, reward accounting, and enumeration of alternative actions.
Scenario descriptors record `generator_version`; the scenario disk-cache version
is bumped so previous generators' instances are not reused for the same seed.
Measure uncached construction, certification, and complete reset separately with
`.venv/bin/python scripts/analysis/benchmark_scenario_generation.py --seeds 10`.
This compares California and Texas and reports charging sessions and route
length per truck. Texas road distances use its stored OD road-route lengths;
the simulator's time-times-speed distance proxy is labelled separately.
For an explicit workload-scaling test on a synthetic hub-and-spoke map, add
`--synthetic --trucks 1 10 100 --deliveries-per-truck 20`.

The inactive `min_hop_distance`/`max_hop_distance` keys are removed and rejected.
Use `environment.min_hop_energy_kwh` (default 0) and `max_hop_energy_kwh`
(default `null`, unbounded). These constrain nominal energy from the depot or
previous customer to the next customer in the generated witness. They do not
restrict policy reordering, charger detours, or early depot retirement. Generated
witnesses obey the one-station-per-segment limit and the configured action budget.
The limit guarantees a feasible witness without consecutive station visits;
it does not change the simulator's policy action space.
No kWh quantity is presented as measured kilometres.
The shipped `energy_ramp` curriculum keeps its actual battery/noise ramp with
unrestricted hop-energy defaults; its old, inactive distance overrides are gone.

Canonical features use schema `joint-fleet-v6`: ownership/order, remaining time
and action/invalid-action budgets, committed route ETA/energy, charge/service
timing and amounts, and FCFS queue positions/reservations are observable.
`legacy_flat` retains its historical prefix and appends the canonical state.
The legacy GNN compatibility adapter also exposes the new history and budgets
in its truck-node features, including after some trucks have retired.
Observation layouts have changed; older checkpoints require retraining, and
canonical checkpoint loading rejects missing or incompatible schema versions.
For success rates use `info["successful"]`, not `all_complete` (which only means
every truck retired). Curriculum accounting records each terminal episode once
and preserves explicit action budgets. Charger `charge_time` statistics count
elapsed port-hours, not the unexecuted remainder of scheduled sessions.

Important configuration sections include:

- `environment`: number of trucks, stops, episode limits, hop energy bounds.
- `truck`: battery capacity, speed, initial battery state.
- `charging`: charger types, charging durations, realistic tapering behavior.
- `problem`: routing mode, assignment, payload capacity, demands, time windows.
- `delivery`: unloading-time stochasticity and service-time bounds.
- `traffic`: travel-time noise, rush-hour effects, energy uncertainty.
- `rewards`: time cost, completion bonus, battery-failure penalty.
- `network`: paths to travel-time, energy, and station data.

Most training scripts also expose command-line overrides such as `--num-trucks`,
`--num-stops`, `--max-time`, `--gnn-state-space`, and action-pruning controls.

## Method: PPO over State and Action Graphs

The main learned policy is a PPO actor-critic with two graph components:

1. A heterogeneous GNN state encoder embeds trucks, deliveries, chargers, and
   transportation relations.
2. A variable-action graph head scores only the feasible actions available at
   the current decision point.

This design avoids forcing every problem instance into a fixed action template.
At each step, the environment provides an action graph containing candidate
delivery moves and charging decisions. The policy scores the candidates,
samples or selects an action, and maps the selected action back to the simulator.

Supported GNN state-space modes:

Routing constraints are selected by `problem.mode`. The GNN factory uses the
shared fleet action layout for both modes; historical `nonflex` and `detour`
CLI selections no longer change the environment's routing rules. Use `vrp` for
the shared GNN compatibility adapter, or the canonical graph/flat observations
for the complete ownership and sequence features.
`base` is another historical CLI alias for this adapter.

## Baselines

The repository includes several comparison policies:

- `heuristic`: rule-based energy-aware routing and charging.
- `savings`: Clarke-Wright savings construction adapted to EV constraints.
- `nn-2opt`: nearest-neighbor tour with lightweight 2-opt cleanup.
- `optimal`: Gurobi-based EVRP optimizer.
- `optimal-simple`: robust Gurobi variant with an energy safety margin.
- `optimal-vrp`: Gurobi-based single-truck VRP optimizer.
- `sb3-*`: Stable-Baselines3 and sb3-contrib agents.

## Evaluation

Single-setting evaluation is configured near the top of
`scripts/evaluation/eval_policies.py`. After selecting policies, config path,
number of scenarios, and seeds, run:

```bash
uv run python scripts/evaluation/eval_policies.py
```

For larger comparisons, use the parallel evaluator. It is also configured by
constants near the top of the file:

```bash
uv run python scripts/evaluation/eval_parallel_policies.py
```

To evaluate size-matched policies across the default truck/stop grid:

```bash
uv run python scripts/evaluation/eval_parallel_by_size.py
```

Grid generalization experiments are configured near the top of
`scripts/evaluation/generalization_eval.py`:

```bash
uv run python scripts/evaluation/generalization_eval.py
```

Evaluation outputs include per-episode metrics, summary CSV files, policy
comparison logs, and experiment artifacts under `results/`.

Common metrics include:

- success rate
- episode reward
- completion time
- routing, waiting, charging, and unloading time
- number of charging sessions
- total distance
- delivery completion count
- failure count
- terminal state-of-charge
- execution time

## Analysis

The current analysis utilities focus on inspecting the environment, generated
instances, learned-policy schedules, and graph state. Some scripts expose CLI
arguments; others use a configuration block near the top of the file.

```bash
uv run python scripts/analysis/visualize_network.py \
  --config EVRoutingEnv/config_files/config.yaml \
  --output results/network_visualization.png \
  --no-show
```

```bash
uv run python scripts/analysis/visualize_transport_graph.py \
  --config EVRoutingEnv/config_files/config.yaml \
  --num-trucks 1 \
  --num-stops 3 \
  --output results/visualization/transport_graph.png
```

```bash
uv run python scripts/analysis/analyze_delivery_routes.py \
  --config EVRoutingEnv/config_files/config_joint.yaml \
  --num-samples 1000 \
  --num-trucks 10 \
  --num-stops 15
```

For schedule comparisons, edit the `POLICIES`, `CONFIG_FILE`, `NUM_TRUCKS`,
`NUM_STOPS`, and `SEED` constants in `scripts/analysis/visualize_schedule.py`,
then run:

```bash
uv run python scripts/analysis/visualize_schedule.py
```

## Citation

If you use this repository in academic work, please cite the accompanying paper.

```bibtex
@misc{evrp2025,
  title        = {Learning to Route Electric Truck Fleets Under Nonlinear Models and Operational Uncertainty},
  author       = {Author Names Placeholder},
  year         = {2025},
  eprint       = {2510.12335},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2510.12335}
}
```
