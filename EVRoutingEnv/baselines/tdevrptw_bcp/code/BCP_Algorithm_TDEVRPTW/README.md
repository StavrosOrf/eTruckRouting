# TD-EVRPTW — Branch-Cut-and-Price (BCP)

A research-grade implementation of a Branch–Cut–and–Price algorithm for the **Time-Dependent Electric Vehicle Routing Problem with Time Windows (TD-EVRPTW)**. The solver supports **time-dependent travel speeds**, **piecewise-linear (PWL) charging**, and **time-dependent station waiting**; it uses **Gurobi** as LP/MIP backend.

This project contains the material for the final exam project of the *Mathematical Optimisation* course for the academic year 2024-25. This course is offered by UniTS (Università degli Studi di Trieste), Trieste, Italy.


## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Features & Implementation](#features--implementation)
- [Results](#results)
- [Author](#author)
- [References](#references)


## Introduction
This repository implements a BCP framework consistent with the referenced paper for TD-EVRPTW.

Key ingredients:
- **Restricted Master Problem (RMP)** with set-partitioning over routes.
- **Exact and heuristic pricing** via forward labeling with time/battery resources.
- **Valid inequalities** (Subset-Row Cuts, k=2).
- **Branching** on consecutive customers (CSB) with a fallback to physical arc branching (ABR).
- **Physics-consistent** time-dependent travel, PWL charging to full at stations, and station waiting patterns.


## Project Structure

```
├── src/
│ ├── bcp_solver.py # Main BCP loop (branch–cut–and–price)
│ ├── model.py` # Gurobi RMP wrapper
│ ├── pricing.py # Pricing (forward labeling) + heuristics
│ ├── labeling.py # Label definition and extension logic
│ ├── dominance.py # Dominance rules (scalar + partial)
│ ├── branching.py # CSB/ABR branching rules
│ ├── cutting.py # SRC (k=2) separation
│ ├── pricing_worker.py # Multiprocessing wrapper for pricing
│ └── utils.py # I/O, PWL helper, TD travel/energy, MBR
├── utils/
│ └── data_processing.ipynb # Data processing for TD-instances
│ └── test_utilities.py # External feasibility check & plotting
├── data/
│ └── TDEVRPTW/ # JSON instances (e.g., r209C15_td.json)
├── results/
│ └── scalability/ # Plots and CSV summaries
│ └── tests/ # Plots on small instances 
├── A-branch-cut-and-price_algorithm_for_the_time-dependent_eletric_vehicle_routing_problem_with_time_windows.pdf
├── requirements.txt
└── README.md
```


## Usage

### 1) Installation

Activate a virtual environment and install the requirements. Ensure to have the Gurobi licence, to fully exploit the solver capabilities.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Data

Place JSON instances under `data/TDEVRPTW/` (example: `r209C15_td.json`). Each instance defines:

**nodes** (customers, stations, depot with coordinates, TW, service),

**parameters** (planning horizon T, capacities, charging/consumption params),

**time_dependent_definitions** (speed profiles, PWL charging, waiting),

**time_dependent_assignments** (arc→speed profile, station→charging mode).

### 3) Run a single instance (Python)

```python
from src.bcp_solver import BCPSolver
res = BCPSolver("data/TDEVRPTW/r209C15_td.json", num_processes=1).solve()
print(res["best_cost"], res["nodes_explored"])
```

### 4) External checks and plotting (Notebook)

```python
from src.utils import load_instance
from utils.test_utilities import simple_external_feasibility_check, plot_instance_and_routes

inst_path = "data/TDEVRPTW/r209C15_td.json"
ext = simple_external_feasibility_check(load_instance(inst_path), res["best_solution"] or [])
print("External feasible:", ext["feasible"])
plot_instance_and_routes(load_instance(inst_path), res["best_solution"], res["best_cost"])
```


## Features & Implementation

- **Time-Dependent Travel Speeds**  
  Each arc is assigned a congestion profile (e.g., *high/normal/low*). Travel time and energy are computed by advancing across time intervals with profile-based multipliers.

- **Non-Linear Charging (PWL)**  
  Each station has a charging **mode** (e.g., *slow/medium/fast*) mapped to a **piecewise-linear** function. On arrival at a station, the vehicle recharges **to full** using the PWL inverse (consistent with the paper).

- **Labeling & Dominance**  
  Forward labeling enforces TW, horizon, battery feasibility, Rule 1 via **MBR** bounds, and includes dominance (scalar + partial) to prune labels safely.

- **Pricing**  
  Heuristics (`k-shrink`, relaxed rules) run in parallel; if none yields negative reduced-cost routes, an **exact** pricing run is attempted under a timeout.

- **Cuts (SRC k=2)**  
  Heuristic separation over small subsets of customers; most violated cuts added to the RMP, avoiding duplicates via cache.

- **Branching**  
  Primary: **Customer-Successor Branching** (Ryan–Foster on consecutive customers).  
  Fallback: **Arc Branching** on immediate $(i \to j)$.

- **Parallelism**  
  Multiprocessing for pricing; deep copies isolate worker state.


## Results

### Scalability Mini-Study (C10, C15, C20)

Replicated the baseline test on:
- **C15**: original instance,
- **C10**: -5 customers (removed from C15),
- **C25**: +10 customers (augmented from C15).

For each, has been recorded:
- `instance_name | cost | n.routes_found | feasible | n.nodes_explored`

| instance_name | cost               | n.routes_found | feasible | n.nodes_explored |
|:--------------|-------------------:|---------------:|:--------:|-----------------:|
| r209C15_TD    | 253.90599156857513 | 3              | True     | 1                |
| r209C15_TD    | 379.8127020718137  | 3              | True     | 5                |
| r209C20_TD    | 585.9096010382843  | 5              | True     | 9                |

Scalability complete results are prsented in `BCP_Algorithm_TDEVRPTW_scalability.pdf`


Artifacts:
- Plots in `results/scalability/`

> The augmentation preserves all TD components and expands `arc_speed_profiles` consistently, defaulting to `"normal"` for arcs involving newly added customers.


## Author

* Tavano Matteo <matteo.tavano@studenti.units.it>


## References

1. *A Branch-Cut-and-Price Algorithm for the Time-Dependent Electric Vehicle Routing Problem with Time Windows* (paper referenced in this project): https://www.sciencedirect.com/science/article/abs/pii/S037722172300509X

2. Gurobi Optimizer — `gurobipy` Python API. https://www.gurobi.com/

3. E-VRPTW instances: https://data.mendeley.com/datasets/h3mrm5dhxw/1

4. Exam Rules: https://sites.units.it/castelli/didattica/?file=mathopt.html



