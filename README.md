<div align="center">

# EVRP

### Learning to Route Electric Trucks When Time, Charge, and Traffic Refuse to Sit Still

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

`EVRP` is a research codebase for the Electric Vehicle Routing Problem with
charging decisions, stochastic travel conditions, realistic charging dynamics,
and graph-based reinforcement learning. The repository provides an event-driven
Gymnasium environment for electric truck routing, custom PPO agents with
heterogeneous GNN state encoders, variable-action policies, classical EVRP/VRP
baselines, Gurobi-based optimization baselines, and evaluation utilities for
policy comparison and generalization studies.

The central idea is simple but demanding: an electric truck should not only
choose *where* to go next, but also *when and how long to charge*, while the
road network, battery state, charger availability, and delivery obligations keep
changing around it. This codebase treats that decision process as an event-driven
control problem and exposes it through graph-structured observations that are
well suited to learning policies over changing problem sizes.

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
- **Evaluation and visualization scripts** for policy comparison, grid
  generalization, route maps, charging curves, schedule analysis, win rates, and
  optimality-gap summaries.

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
  evaluation/              # Policy comparison and generalization scripts
  analysis/                # Route, network, schedule, and state visualizers
  runners/                 # tmux/grid experiment launchers
```

## Installation

This project uses `uv` for reproducible Python environments.

```bash
git clone <repository-url>
cd EVRP
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

Grid generalization experiments are configured near the top of
`scripts/evaluation/generalization_eval.py`:

```bash
uv run python scripts/evaluation/generalization_eval.py
```

Evaluation outputs include per-episode metrics, summary CSV files, formatted
tables, route visualizations, win-rate plots, and optimality-gap summaries under
`results/`.

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

## Citation

If you use this repository in academic work, please cite the accompanying paper.
Replace the placeholder below with the final publication metadata.

```bibtex
@misc{evrp2026,
  title        = {Learning to Route Electric Trucks with Event-Driven Graph Reinforcement Learning},
  author       = {Author Names Placeholder},
  year         = {2026},
  eprint       = {arXiv:XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url          = {https://github.com/<owner>/<repo>}
}
```
