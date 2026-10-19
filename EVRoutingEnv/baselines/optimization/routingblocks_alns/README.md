# RoutingBlocks + exact charging pricing

Near-optimal plans for the nominal joint electric-truck routing problem of
`EventDrivenTruckEnv`, from the small test instances up to the large California
ones (128 trucks, 200 customers). It adapts the RoutingBlocks + frvcpy
literature baseline (snapshot in `../../routingblocks_frvcpy/`, left
unmodified) to this environment's model. Unlike `../branch_and_price/` it is a
heuristic: it reports no proof, only the plan, and the gap to the
branch-and-price lower bound when you supply one.

## The problem it solves

Identical to the branch-and-price baseline (they share `export.py`): nominal
legs, base service, each station's own charge curve, the env's discrete charge
levels, the per-leg 1.2 energy margin, payload, optional hard windows, the
horizon, a homogeneous fleet from one depot. Objective: total fleet travel hours
(`--objective operating` adds charging and service time). Charger ports are
relaxed; `validate.nominal_replay` re-executes every plan in the env, FCFS
queues included.

## Method

| Part | What it does |
|---|---|
| **Exact route pricing** (`native/evrp_core.cpp`) | For a customer sequence, the cheapest feasible insertion of charger stops and discrete charge levels, as a Pareto label setting over (cost, time, SoC). After a charge the SoC is one of the discrete levels, so everything downstream of a charging stop depends only on (station, level) and the target. Those Pareto fronts are precomputed once per instance. This plays the role frvcpy plays for a continuous-charging model, but handles the env's discrete levels, per-station curves, margin, windows and horizon. About 0.1-0.5 ms per route. |
| **Leg surrogate** | For each depot/customer pair, the cheapest station detour when leaving full. It is the arc cost/duration of the RoutingBlocks instance, so RoutingBlocks' native operators see realistic detours. On near-optimal routes it is within 1% of the exact cost. |
| **RoutingBlocks ALNS** (`search.py`) | Customer-only routes (stock `ADPTWEvaluation`: payload, windows, horizon; battery switched off). Destroy: random, non-empty route, worst, nearest-neighbour. Repair: random, best insertion with and without blinks. Granular local search (swap 0-1, 0-2, 1-1, inter-route 2-opt) when a candidate is within `delta_local_search` of the current cost. |
| **Exact acceptance** | Every candidate route is priced exactly (cached by sequence); the incumbent and acceptance (simulated annealing on the exact cost) never use the surrogate. A candidate with an unpriceable route is rejected. |
| **Route pool + set partitioning** (`pool.py`) | Routes of near-best candidates join a pool (cheapest ordering per customer set). HiGHS solves the set-partitioning problem over it, warm-started with the incumbent, periodically and at the end. |
| **Parallel workers** (`parallel.py`) | Persistent processes, each its own ALNS; in rounds the master merges their pools, partitions, and sends the result to half of them. |

Known limitation: a station detour that passes through two or more chargers
before the first charge is not considered (it matters only when the time table
violates the triangle inequality twice). It can only cost optimality, never
feasibility.

## Setup and use

The RoutingBlocks wheel has no Python 3.13 build, so the solver runs in its own
Python 3.11 environment and the bridge launches it as a subprocess (like the
Julia baseline). One-time setup:

```bash
bash scripts/setup_rb_env.sh              # creates .venv-rb and compiles the native core
```

The bridge uses `$EVRP_RB_PYTHON` or the nearest `.venv-rb`.

```bash
export PYTHONPATH=.
# plans for evaluation seeds, validated by replay; optionally the gap to B&P bounds
python -m scripts.evaluation.solve_routingblocks --trucks 50 --customers 100 \
    --split test --count 20 --time-limit 600 --workers 16 \
    --bound-from results/branch_and_price/ca_50x100 --out results/rb_alns/ca_50x100
# Slurm, one seed per array task (18 cores each)
sbatch --array=0-19 scripts/slurm/rb_array.sbatch 128 200 test results/rb_alns/ca_128x200 1800
# validation against brute force and the env
python scripts/analysis/validate_routingblocks_alns.py --out results/rb_alns/validation.json \
    --seeds 4 --time-limit 15
```

In Python: `bridge.solve_rb_plan(env, RBParameters(time_limit_seconds=600, workers=16))`
returns the same `BPSolution` as the Julia bridge. `policy.RoutingBlocksPolicy`
plays a plan (or reads a certificate directory) with the branch-and-price
policy's execution layer. A certificate also works as `--warm-from` for
`solve_branch_and_price`, which gives the exact solver a strong incumbent.

Any `SearchParameters` field can be set with `--param KEY VALUE`.

## Validation

| Check | Result |
|---|---|
| Route DP vs independent brute force (`test_routingblocks_alns.py`, all subsets of 5 customers, default and time-window configs) | no mismatches |
| 96 small instances (default, time-window, tight-window and slow-charging configs; 2-3 trucks, 4-7 customers), 15 s each | all reach the brute-force optimum; every plan replays in the env at the same travel time |
| 10 x 30 (seed 0) | 97.7167 h = branch-and-price optimum, in about 2 s |
| 20 x 50 (seed 0) | 132.2842 h = branch-and-price optimum (500 s there), in about 20 s |
| 50 x 100 (seed 1), 16 workers, 300 s | 355.87 h; branch-and-price: lower bound about 350.9, best incumbent 354.2 |
| 128 x 200 (seed 1), 16 workers, 600 s | 878.84 h; branch-and-price lower bound about 839.3 |

The large-instance rows are heuristic results with a gap to a bound, not
optima.
