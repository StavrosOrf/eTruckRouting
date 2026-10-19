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
| Pricing speed | Without time windows, phase 1 prices bidirectionally (forward and backward labels meet past half the payload; both halves run on separate threads when Julia has more than one). Heuristic rounds (nearest customers, capped buckets) run before every exact round. Each node's label bucket keeps the loads and ng memories of its labels in contiguous arrays beside their costs, so dominance scans skip most pairs without touching a label (6.6x faster exact rounds at 50 x 100) |
| Tightening | Root fleet bound ceil(sum d / payload); adaptive ng-routes (as upstream); rounded capacity cuts on customer successions, with exact dominance correction for the last-customer-dependent cut duals; limited-memory subset-row cuts on customer triples (upstream's lm-SR3, up to 256 cuts), tracked by one parity bit per cut. Capacity-cut rounds run on heuristically converged LPs; subset-row rounds run on certified ones |
| Bounds | Lagrangian bound `LP + hi * min(0, min rc)` after every exact pricing round; Farkas (phase-one) pricing proves infeasible nodes; reduced-cost arc fixing with the certified root duals (an arc whose cheapest route, by q-route bounds, has reduced cost above incumbent minus root bound leaves the graph), repeated whenever the incumbent improves |
| Tree | Best-bound search; nodes at depth <= `lazy_depth` (3) are always certified so the global bound moves; deeper, lazy certification: a node converged under heuristic pricing is branched on with its inherited (valid) bound, and exact pricing runs only where a certified LP decides something, that is, where it can prune the node or the LP is integral |
| Branching | Route count, then the most fractional customer succession `y_ij`; complete, because integral successions fix the customer sequences and the cheapest column per sequence is then integral |
| Primal | Warm-start witness, single-customer seed routes, optional ALNS routes (`--alns-seconds`; the solver adds their optimal charging stops), restricted-master MIP, diving (fixing the largest fractional column), route re-optimisation (a few neighbouring incumbent routes re-solved exactly as a sub-instance by this same solver), integral nodes |
| Closing by enumeration | At certified nodes within `enum_gap` (relative) of the cutoff, enumerate every elementary route with reduced cost below cutoff minus node LP (after Baldacci, Christofides & Mingozzi), and solve the set-partitioning MIP over them. Its optimum, or its infeasibility, settles the whole subtree. Gives up at `enum_route_cap` routes or `enum_time` seconds, and is not retried at thresholds above half of one that failed. With q-route completion bounds it rarely finishes beyond about 30 customers; pass `--enum_gap 0` there |
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

### Parallel subtrees (large instances)

One process explores the tree serially, and at 50 x 100 a node takes about
a minute. A split search spreads the tree over many processes. Within one
allocation, `bnp_parallel.sbatch` runs the whole pipeline: the split run, then
all subtrees side by side, then the combined certificate.

```bash
sbatch --array=0-7 scripts/slurm/bnp_parallel.sbatch 50 100 test results/branch_and_price/ca_50x100_par \
    84000 16 EVRoutingEnv/config_files/config_joint.yaml --alns-seconds 120
```

A solver process keeps about one core busy, so an allocation that is already
running a solve can host the pipeline as an extra step:
`srun --jobid=<job> --overlap -n1 -c16 bash scripts/slurm/bnp_parallel.sbatch ...`.
Give it a time limit that ends before that job's own script does.

To spread the subtrees over several allocations instead:

```bash
OUT=results/branch_and_price/ca_50x100_split
# 1. root, then branching until 32 nodes are open; writes OUT/split_1.json
sbatch scripts/slurm/bnp_array.sbatch 50 100 seeds=1 $OUT 84000 EVRoutingEnv/config_files/config_joint.yaml \
    --alns-seconds 120 --split-nodes 32 --solver-arg lp_solver highs
# 2. all 32 subtrees at once: 4 tasks x 8 solver processes
sbatch --array=0-3 scripts/slurm/bnp_subtree.sbatch $OUT 1 82800 8 --lns_time 300
# 3. combine, at any time; writes OUT/seed_1.json after the nominal replay
python -m scripts.evaluation.combine_bnp_subtrees --out $OUT --seed 1 --trucks 50 --customers 100
```

A solver process keeps only one or two cores busy and needs 2-4 GB at
50 x 100, so each task runs several subtrees side by side. Each subtree run
takes its node's branching restrictions as its root. The split file also
carries the split run's master: its columns, capacity and subset-row cuts and
adapted ng neighbourhoods. A subtree run therefore starts from that LP, adds
cuts of its own and searches its subtree.

Runs publish every incumbent to `OUT/incumbents_<seed>/incumbent_<tag>.json`.
Before each node, and between route re-optimisation steps, they adopt any
better incumbent another run has published. To seed the search with an
incumbent found elsewhere, copy that run's certificate into the directory
under an `incumbent_` name. Spare cores can feed it with primal-only runs:
`run_bp.sh OUT/instances/seed_<seed>.json OUT/primal.json --primal_only true
--incumbent_dir OUT/incumbents_<seed> --incumbent_tag primal --lp_solver highs`.

The subtrees cover exactly what the split run left open, so the combined
lower bound is the least subtree bound. A subtree with no result yet counts
at the bound it was split with. A subtree that finishes proves that it holds
nothing better than its final incumbent. The combined status is `OPTIMAL`
once every subtree has finished and the bounds meet.

In campaigns, method `bnp` (`run_canonical_campaign.build_policy`) builds
`policy.BranchAndPricePolicy`; set `certificate_dir` to reuse offline
certificates. The policy plays the planned steps and falls back to the greedy
heuristic's energy-safe navigation only when a planned step is masked. Each
fallback is counted in `diagnostics()`.

Toolchain: Julia 1.10.4 and Gurobi through `julia/run_bp.sh`, which loads the
Snellius modules and defaults `GRB_LICENSE_FILE` to `~/gurobi.lic`. Set
`EVRP_BP_THREADS=2` (or more) for the threaded bidirectional pricing. On
Snellius the Slurm scripts run on `gpu_a100`: one GPU share gives 18 cores and
about 120 GB, and GPU allocations are plentiful. The solver itself is CPU-only;
`cbuild` is the fallback CPU partition. The Gurobi WLS licence admits only two concurrent sessions,
and every solver process holds one, including local test runs. So throttle
arrays with `%2`, or pass `--solver-arg lp_solver highs` (or set
`EVRP_BP_LP_SOLVER=highs` for the Python bridge) to solve the master with
HiGHS. HiGHS needs no licence and gives the same certificates; its master is
somewhat slower. Any `BPParameters` field can be passed through the driver as
`--solver-arg KEY VALUE`, and through Slurm as trailing arguments of
`bnp_array.sbatch` after CONFIG. Snellius jobs do not inherit the submitting
shell's environment.
`julia/upstream_repro/` reproduces the unmodified upstream results on its
bundled instances.

## Licence

Upstream has no licence file. This package is derived from its design, so keep
it private until the authors grant permission.

## Results so far (2026-10-19)

| Instances | Outcome |
|---|---|
| 90 small instances (2-3 trucks, 4-7 customers; default, time-window, tight-window, slow-charging and squeezed-horizon variants) | all proven optimal and equal to an independent brute force; all replay in the env at the certified travel time |
| 10 x 30 (seed 0) | proven optimal at the root in about 4 s |
| 20 x 50 (seed 0) | proven optimal: 132.284230 h, at the root with lm-SR3 cuts in about 500 s |
| campaign test set, 300 scenarios at 2 x 10 | all proven optimal in 5 minutes |
| 10T3D GraphPPO campaign, 128 validation scenarios at 10 x 30 | 119 proven optimal, 8 within 0.15-4.4%, 1 without a root within 900 s. Executed plans: 127/128 success, 4.7% below the campaign's best-known reference in operating hours (`results/branch_and_price/ca10t3d_validation_2000/comparison_graphppo.json`) |
| 50 x 100 (seed 1) | root bound 350.74 h, incumbent 354.16 h after 110 nodes (0.96% gap; run continuing) |
| 128 x 200 (seed 1) | the first exact root pricing round exceeds 20 minutes; run continuing |
