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

Every plan below was replayed in the environment (`validate.nominal_replay`)
and reproduced its reported travel time exactly. "Optimum" means proven by the
brute-force reference or by branch-and-price; "bound" is a branch-and-price lower
bound. Nothing the solver returned was ever below a proven optimum.

| Instances | Settings | Result |
|---|---|---|
| Route DP vs independent brute force (`test_routingblocks_alns.py`; every subset of 5 customers, default and time-window configs) | | no mismatches |
| 96 small instances (default, time-window, tight-window, slow-charging; 2-3 trucks, 4-7 customers), `validate_routingblocks_alns.py` | 1 worker, 15 s | all 96 reach the brute-force optimum |
| 10 x 30, seeds 0 and 3 | 1 worker, about 15 s | optimum on all 6 solver seeds each |
| 128 campaign scenarios, 10 x 30 (119 proven optimal, 9 not) | 1 worker, 30 s | 110 of 119 equal the optimum, worst miss 1.8%; 5 of the 9 unproven branch-and-price incumbents beaten; all 128 replay |
| the same 9 misses | 4 workers, 60 s | 4 now reach the optimum, the rest are within 0.5% |
| 20 x 50 (seed 0), optimum 132.2842 h (500 s in branch-and-price) | 4 workers, 60 s | optimum on 3 of 3 solver seeds |
| 50 x 100 (seed 1); bound about 350.9 h, best branch-and-price incumbent 354.2 h | 16 workers, 600 s, pre-column-cap fix | 354.89 h (1.1% above the bound) |
| 128 x 200 (seed 1); bound about 839.3 h | 16 workers, 1200 s, pre-column-cap fix | 859.49 h (2.4% above the bound); wall time 2348 s (the uncapped partitioning pool stalled the master — the bug the column cap below fixes) |
| 50 x 100 (seed 1), same instance, **after** the column-cap fix | 8 workers, 250 s | 355.77 h; wall time 272 s, each partitioning solve bounded at ~20 s as designed (no stall) |
| 128 x 200 (seed 1), same instance, **after** the column-cap fix | 8 workers, 400 s | 868.63 h; wall time 422 s, bounded the same way even as the pool passed 23,000 columns |

The 50 x 100 and 128 x 200 rows are heuristic results with a gap to a bound,
not optima: branch-and-price could not close them, and this baseline cannot
produce or improve a certified bound by itself. The post-fix rows above used
fewer workers and less time than the pre-fix rows (8 workers/cbuild's shared
CPUs rather than 16, because the cluster's GPU partitions were saturated by a
large concurrent branch-and-price campaign when this was run), so they are not
a clean apples-to-apples comparison of solution quality; what they confirm is
that the column cap removes the multi-thousand-second stall without degrading
the result. Reproduce the campaign row with
`scripts/evaluation/solve_routingblocks.py ... --bound-from <bnp dir>` and
`scripts/analysis/compare_routingblocks_bnp.py`.

### Tuning notes

* Local search is what matters. Running it on nearly every iteration
  (`delta_local_search 0.5`) with the segment-swap operators beat a cheaper
  schedule by about 3% on 20 x 50 and 2% on 128 x 200; with it off the result is
  10% worse. Restricting it to the routes the destroy/repair changed
  (`local_search_scope changed`) gives the same quality much faster than a sweep
  of all routes.
* The annealing temperature scales with the cost per customer (`start_temperature`
  0.5 of it): a fixed fraction of the total cost was too hot at 100 customers and
  too cold at 50.
* HiGHS does not honour its time limit inside presolve and the root LP, so the
  set-partitioning columns are capped (about 40,000, chosen per customer by
  cost per customer served). An uncapped pool of 370,000 columns stalled the
  master for minutes.
* A few 10 x 30 instances stay 0.3-0.4% above the optimum however the
  parameters are set. In one case the optimum uses a different route count (5
  routes, one with two customers) than the fully loaded 4-route solution the
  search settles on; forcing extra routes did not help, so it is left as is.
  Extra workers and the pool recover some of them.
