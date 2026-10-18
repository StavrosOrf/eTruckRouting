# Exact branch-and-price baseline

Proven-optimal (or certified-gap) solutions of the nominal joint
electric-truck routing problem of `EventDrivenTruckEnv`, adapted from
ElectricRouting.jl (Jacquillat & Lo, arXiv:2407.02640; snapshot in
`../../electricrouting_jl/`, left unmodified).

## What is optimised

The deterministic equivalent of an episode, exported by `export.py` right
after `env.reset(seed)`:

- legs at their nominal table time and energy;
- base service times;
- each station's own charging curve (power class, efficiency, CC-CV taper);
- the env's discrete target-SoC charge levels;
- the per-leg 1.2 energy margin of the action mask;
- payload, optional hard time windows, the horizon and the fleet size.

The objective is total fleet travel hours (`OperationalMetrics.total_travel_time`).

Charger ports are relaxed. `validate.nominal_replay` re-executes every plan
through the env's public `step` with noise off, including its FCFS queues, and
reports any waiting. Waiting cannot change travel time; it only matters if it
breaks a window or the horizon, in which case the replay fails.

## Method (`julia/EVRPBranchPrice`)

| Part | Implementation |
|---|---|
| Master | Set partitioning (`== 1` per customer; the tables break the triangle inequality and the env refuses second visits), route-count rows, one column per node sequence |
| Pricing | Label setting with ng-route memory, discrete charging at chargers, per-leg margin, payload, windows and horizon, plus a q-route completion bound with 2-cycle elimination. This is upstream's "benchmark" (Desaulniers et al. 2016) pricing; upstream's subpath stitching was dropped because clustered California customers give over a million three-customer subpaths per charger at 200 customers |
| Relaxation phases | Phase 1 prices with charging time ignored (and, without windows, the horizon dropped too). That is a valid relaxation, so its optimum is a lower bound; every priced route is re-timed exactly (`schedule_route`), and a relaxed optimum that is truly feasible is optimal. Otherwise phase 2 repeats the search with full time-aware pricing over truly feasible columns, starting from that bound |
| Pricing speed | Without time windows, phase 1 prices bidirectionally (forward and backward labels meet past half the payload; both halves run on separate threads when Julia has more than one). Heuristic rounds (nearest customers, capped buckets) run before every exact round |
| Tightening | Root fleet bound ceil(sum d / payload); adaptive ng-routes (as upstream); rounded capacity cuts on customer successions, with exact dominance correction for the last-customer-dependent cut duals; limited subset-row cuts on customer triples (upstream's SR3), tracked by one parity bit per cut. Capacity-cut rounds run on heuristically converged LPs; subset-row rounds run on certified ones |
| Bounds | Lagrangian bound `LP + hi * min(0, min rc)` after every exact pricing round; Farkas (phase-one) pricing proves infeasible nodes |
| Branching | Route count, then the most fractional customer succession `y_ij`; complete, because integral successions fix the customer sequences and the cheapest column per sequence is then integral |
| Primal | Warm-start witness, single-customer seed routes, restricted-master MIP, diving (fixing the largest fractional column), integral nodes |
| Branching choice | Strong branching on the most fractional successions, ranked by child LPs over the current columns |

Status `OPTIMAL` means the incumbent's travel equals the certified lower bound
within `gap_tol` (relative 1e-6). Any other status reports the certified gap
`(objective - bound) / objective`.

## Use

```bash
source /home/sorfanoudakis/EVRP/.venv/bin/activate
export PYTHONPATH=.
# certificates for evaluation seeds (writes OUT/seed_<seed>.json, validated by replay)
python -m scripts.evaluation.solve_branch_and_price --trucks 50 --customers 100 \
    --split test --count 20 --time-limit 3600 --threads 8 --out results/branch_and_price/ca_50x100
# Slurm: one seed per array task
sbatch scripts/slurm/bnp_precompile.sbatch
sbatch --array=0-19 scripts/slurm/bnp_array.sbatch 50 100 test results/branch_and_price/ca_50x100
# validation against brute force and the env
python scripts/analysis/validate_branch_and_price.py --out results/branch_and_price/validation.json
```

In campaigns, method `bnp` (`run_canonical_campaign.build_policy`) builds
`policy.BranchAndPricePolicy`; set `certificate_dir` to reuse offline
certificates. The policy plays the planned steps and falls back to the greedy
heuristic's energy-safe navigation only when a planned step is masked. Each
fallback is counted in `diagnostics()`.

Toolchain: Julia 1.10.4 and Gurobi through `julia/run_bp.sh`, which loads the
Snellius modules and defaults `GRB_LICENSE_FILE` to `~/gurobi.lic`. Set
`EVRP_BP_THREADS=2` (or more) for the threaded bidirectional pricing. On
Snellius the account has CPU budget only on the `cbuild` partition, which the
Slurm scripts use.
`julia/upstream_repro/` reproduces the unmodified upstream results on its
bundled instances.

## Licence

Upstream has no licence file. This package is derived from its design, so keep
it private until the authors grant permission.
