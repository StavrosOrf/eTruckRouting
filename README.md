<div align="center">

# Learning to Route Electric Truck Fleets Under Nonlinear Models and Operational Uncertainty

**Event-driven graph reinforcement learning for charge-aware electric truck routing.**

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
  <em>Stochastic traffic. Realistic charging curves. Variable action spaces. Fleet-scale generalization.</em>
</p>

</div>

> **Paper:** [Learning to Route Electric Truck Fleets Under Nonlinear Models
> and Operational Uncertainty](https://arxiv.org/abs/2510.12335)

## At a Glance

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

<img width="2019" height="921" alt="image" src="https://github.com/user-attachments/assets/64b94fe5-e151-4652-93fd-5c33351c42a8" />


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
  data/                    # Network, station, travel-time, and energy data
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
cd eTruckRouting
uv sync
```

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
  --config EVRoutingEnv/config_files/config_vrp.yaml \
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
  --config EVRoutingEnv/config_files/config_vrp.yaml \
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

- `config.yaml` is the default electric truck routing setup.
- `config_vrp.yaml` enables flexible delivery order for single-truck VRP-style
  experiments.

Important configuration sections include:

- `environment`: number of trucks, stops, episode limits, hop energy bounds.
- `truck`: battery capacity, speed, initial battery state.
- `charging`: charger types, charging durations, realistic tapering behavior.
- `delivery`: flexible order, unloading-time stochasticity, service-time bounds.
- `traffic`: travel-time noise, rush-hour effects, energy uncertainty.
- `rewards`: time cost, completion bonus, battery-failure penalty.
- `network`: paths to travel-time, energy, and station data.

Most training scripts also expose command-line overrides such as `--num-trucks`,
`--num-stops`, `--max-time`, `--gnn-state-space`, and action-pruning controls.

## Method: PPO over State and Action Graphs

<img width="2439" height="722" alt="image" src="https://github.com/user-attachments/assets/1eaa7e29-266e-4352-8111-04eb5b72902a" />


The main learned policy is a PPO actor-critic with two graph components:

1. A heterogeneous GNN state encoder embeds trucks, deliveries, chargers, and
   transportation relations.
2. A variable-action graph head scores only the feasible actions available at
   the current decision point.

This design avoids forcing every problem instance into a fixed action template.
At each step, the environment provides an action graph containing candidate
delivery moves and charging decisions. The policy scores the candidates,
samples or selects an action, and maps the selected action back to the simulator.

<img width="1973" height="892" alt="image" src="https://github.com/user-attachments/assets/75acf59d-00f1-4b7f-afa7-293acd2064c7" />


Supported GNN state-space modes:

- `nonflex`: sequential delivery routing.
- `detour`: sequential routing with restricted charger detours.
- `vrp`: flexible delivery order with top-k delivery candidates.
- `base`: compatibility alias for non-flex experiments.

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
  --config EVRoutingEnv/config_files/config_vrp.yaml \
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
