# Gate reports: shared public chargers, hours of service and dynamic requests

Reports for gates 0, 1, A, B, C and Final of `docs/campaigns/dynamic_stochastic_etruck_model_plan.md`, in that plan's order. The formal model is `docs/models/dynamic_stochastic_etruck_model.md`.

Branch `NewFormulation`. Phase 0 to C was committed by the user as `e59574a New formulation`. The fixes listed under "Fixes after the first test pass" (Final gate) are uncommitted, in the working tree on top of it.

**Each gate needs your review before it counts as passed.** Only you can launch `/code-review ultra`. The checklist answers below are the implementer's, for the reviewer to verify.

## How to reproduce the numbers

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit tests/integration -m "not statistical" \
  --ignore=tests/unit/test_contract_chains.py --ignore=tests/unit/test_od_tree_sums.py \
  --ignore=tests/unit/test_road_graph_speed_cap.py
PYTHONPATH=. .venv/bin/python -m pytest tests -m statistical          # nightly / before a gate
PYTHONPATH=. .venv/bin/python scripts/analysis/report_component_oracles.py
PYTHONPATH=. .venv/bin/python scripts/analysis/benchmark_public_queues.py --seeds 5
```

The three ignored modules need `osmium`, which the project dependencies do not include.

## Pre-existing failures (not caused by this work)

Each of these failed before any change in this branch, or fails on files this work did not touch.

| Test | Cause |
|---|---|
| `tests/unit/test_contract_chains.py`, `test_od_tree_sums.py`, `test_road_graph_speed_cap.py` | `ModuleNotFoundError: osmium` at collection |
| `tests/unit/test_case_studies.py` (`china_jinghu`, `europe_10k*`) | Matrix data files (`od_energy_kwh.npy`, ...) are absent from the checkout |
| `tests/unit/test_completion_shield.py` (2 tests) | The test's stub tasks lack `node_id`; `action_mask.py` and the test are unchanged from HEAD |
| `tests/integration/test_joint_environment.py::test_joint_reset_creates_one_shared_fleet_instance` | Expects power classes {150, 350, 750}; the unchanged `config_joint.yaml` sets [150] |
| `tests/integration/test_perfect_information_cli.py::test_seeded_cli_solve_and_cross_process_replay` | 60 s subprocess timeout; failed identically on the pre-change baseline run |

---

## Gate 0: quota removed, legacy frozen

**What changed**

- The per-truck quota was removed everywhere. That covers the env parsing and size check, the registry quota API, both `DELIVERY_QUOTA_*` reasons, the quota rules in ALNS, the canonical baselines and CP-SAT, and `test_delivery_quota.py`.
  - `problem.deliveries_per_truck` now raises the removed-key `ValueError` (`test_environment_validation.py::test_delivery_quota_key_is_rejected`).
  - The quota-only files are byte-identical to HEAD.
  - `git log -S DELIVERY_QUOTA` finds no commit.
  - The depot-energy ring is kept.
- Configs: `deliveries_per_truck` is removed from the long-haul configs, and the headers say "10 trucks, 30 customers, free assignment". `config_joint_california_10t3d.yaml` is renamed to `config_joint_california_10t30c.yaml`.
- `generalist_graphppo_dayahead_plan.md` §1b: the quota section is marked withdrawn, with the reason, and the free-workload rows are restored.
- A local variable named `quota` in `route_walk.py`, which is the witness's even split of customers, is renamed to `share`, so `grep -rn quota EVRoutingEnv tests` is clean. Behaviour is unchanged; the golden traces confirm it.

**Golden traces (§3.2)**

- `scripts/analysis/record_golden_traces.py` and `tests/integration/test_golden_traces.py` cover 5 configs × {greedy, seeded random} × 5 seeds = **50 episodes**.
- Hashed per step: actions, active truck, clock, mask bit strings and termination reason. Hashed per truck at the end: the full `export_event_log()`, which is append-only and timestamped. Seed 0 is stored in full under `tests/data/golden/full/`.
- Every episode is replayed at record time, and recording refuses to store a trajectory whose replay differs.
- Recorded at `eecdaab` (quota removed, no Phase 1 code). Re-recording later gave identical hashes.
- **v6 feature baseline:** `golden_features_v6.json` freezes per-decision hashes of every v6 feature block along the seed-0 trajectories. It was recorded before the schema changed.

**Checklist**

| Question | Answer |
|---|---|
| Is every trace of the quota gone? | Yes, except three intended hits: the removed-key guard message, the guard test's name, and an unrelated per-customer column budget in `routingblocks_alns/pool.py`. `scripts/analysis/*` still use a `deliveries_per_truck` sizing knob (customers = trucks × x) for generation studies, which is not a constraint. |
| Do the golden traces cover all five configs, recorded before Phase 1? | Yes, as above. |

---

## Gate 1: shared infrastructure

**What changed**

- **System events.**
  - `EventType.SYSTEM` is appended; system events use `truck_id = -1`, priority −1, and a sequence tie-breaker that is 0 for every legacy event.
  - The event loop handles system events before it indexes `self.trucks`.
  - Same-time order is SYSTEM < arrival < completion < other ready event.
  - System events at or past the horizon are dropped and never truncate (the plan says "past"; an event exactly at the horizon cannot inform any decision).
  - Handlers are registered as method names, so environment copies dispatch to themselves.
- **Named streams.** `ScenarioRandomStreams.uniforms`, `exponential` and `keyed_generator` are keyed by (stream, key) and policy-independent. One or two uniforms come straight from the BLAKE2b digest.
- **Requests.**
  - `models/core/requests.py` adds `Stop`, `Request`, `RequestKind` and `RequestStatus` (lifecycle as in §4.3, plus `SERVED_LATE`), and a `RequestRegistry`.
  - The registry holds the node → live-requests multimap, allocates ids in reveal order, enforces the envelope, and raises `RuntimeError` on any illegal transition.
  - A `RequestTaskView` gives generic code the old task interface and raises if a node hosts more than one live request.
- **Actions.** `QUEUE`, `REST_BREAK`, `REST_DAILY`, `WAIT` and `END_SHIFT` are appended to `ActionKind`. They enter the envelope only when their component is enabled, so legacy envelopes keep their size.
- **Reasons.** All twelve reasons from §6.6 are appended. Four more are added where no listed reason fits:
  - `PORTS_BUSY`: charging at a busy station's queue-or-leave epoch;
  - `NOT_AT_DEPOT`: `END_SHIFT` away from the depot;
  - `DRAYAGE_ORIGIN_FIRST`: Xiao's rule that the first stop after the depot is an origin;
  - `STATION_HOP_LIMIT`: Kullman's optional limit on stations between stops.
- **Information options.**
  - `observation.in_flight_information: realized | nominal_eta`.
  - Schema `joint-fleet-v7` appends columns for HOS, cargo, station information (including an `occupancy_indicator` column for Dastpak et al.), requests and request counts.
- **Configuration.** All blocks are parsed and validated in `models/environment/extension_config.py`. Unknown keys and unsupported combinations raise. Resolved service presets are written back into the config for the artifact hash.

**Documented deviation.** The legacy static mode keeps `FleetTaskRegistry`, which is exactly the `static_delivery` case (no pickup, starts open, one task per node). Re-plumbing it through `RequestRegistry` would put the golden traces at risk for no modelling gain. The request engine is used by `same_day`, `drayage` and the new `problem.loading: at_depot`.

**Checklist**

| Question | Answer |
|---|---|
| Can a system event touch truck state? | No. `test_system_events.py::test_system_event_at_the_head_never_reads_or_changes_the_last_truck` runs three trucks with the last one complete: the event is processed and the truck's `__dict__` is unchanged. |
| Are same-time priorities as in §4.1? | Yes; `test_same_time_order_is_system_routing_completion_other` covers every pair. |
| Can a fleet action shift a random draw? | No. Every new draw is keyed (`test_common_random_numbers_across_policies`: greedy and random see identical outside arrivals, port timelines and request streams). |
| Does an illegal registry transition pass silently? | No (`test_request_registry.py`). |
| Do the v6 columns match exactly? | Yes. `test_v6_columns_of_current_schema_match_frozen_extraction` passes on all 10 recorded trajectories. |

---

## Gate A: public stations with outside traffic

**What changed**

- `models/simulation/queueing.py`: Erlang C, the stationary distribution with exact sampling, Keskin's stationary wait, and the transient expected wait computed from the symmetrised tridiagonal birth–death generator.
- `models/simulation/public_station_queue.py`: a lazy `StationQueue`, with exact reservation prediction on a private copy.
- `models/simulation/public_charging.py`:
  - `PublicChargingStation` (`dynamic_mmc`): Kullman's queue-or-leave epoch, commitment at the head, release on case A5, the three information models, shared or per-truck memory, and the eager debug reference.
  - `StationaryWaitChargingStation` (`iid_stationary_wait`).
- Depot charging via `ChargingStation.add_private_charger`: fleet-only, with no outside traffic or outages.
- **Baselines:**
  - the greedy navigator always queues at a busy station;
  - `KullmanMyopicPolicy` implements §5.2.3 rule by rule;
  - ALNS and CP-SAT accept `queue_aware=True`, which adds the mean long-run Erlang C wait per battery's worth of energy;
  - the perfect-information search is clairvoyant over the same keyed realization.

**Oracle results** (`report_component_oracles.py`)

| Outside-only station | Customers | Simulated $W_q$ ± 99% CI (h) | Erlang C (h) |
|---|---|---|---|
| c = 1, u = 0.40 | 104,690 | 0.3346 ± 0.0139 | 0.3333 |
| c = 1, u = 0.65 | 104,374 | 0.9059 ± 0.0437 | 0.9286 |
| c = 1, u = 0.90 | 315,557 | 4.4967 ± 0.5965 | 4.5000 |
| c = 2, u = 0.40 | 105,171 | 0.0995 ± 0.0067 | 0.0952 |
| c = 2, u = 0.65 | 105,041 | 0.3674 ± 0.0268 | 0.3658 |
| c = 2, u = 0.90 | 314,839 | 2.0268 ± 0.2099 | 2.1316 |
| c = 4, u = 0.40 | 106,046 | 0.0185 ± 0.0022 | 0.0189 |
| c = 4, u = 0.65 | 105,892 | 0.1281 ± 0.0113 | 0.1266 |
| c = 4, u = 0.90 | 315,285 | 0.9513 ± 0.0803 | 0.9847 |
| c = 10, u = 0.40 | 105,296 | 0.0009 ± 0.0004 | 0.0007 |
| c = 10, u = 0.65 | 106,308 | 0.0215 ± 0.0032 | 0.0220 |
| c = 10, u = 0.90 | 317,414 | 0.3729 ± 0.0565 | 0.3344 |
| c = 20, u = 0.40 | 107,231 | 0.0000 ± 0.0000 | 0.0000 |
| c = 20, u = 0.65 | 107,540 | 0.0036 ± 0.0008 | 0.0036 |
| c = 20, u = 0.90 | 317,858 | 0.1348 ± 0.0200 | 0.1377 |

All 15 cells fall inside their 99 % batch-means intervals (30 batches; t quantile 2.756).

| Check | Result |
|---|---|
| Shipped `waiting_time_lookup.json` vs Erlang C | Equal to 7e-15 relative in every cell of every case study |
| Little's law (c = 3, u = 0.65) | $L_q$ = 0.7917 vs $\lambda W_q$ = 0.7893, relative error 0.30 % |
| Stationary start (c = 2, u = 0.65, 10⁴ resets) | Chi-square p = 0.40 |
| Port availability (a = 0.733, 53,422 cycles) | Measured 0.7346 |
| Transient expected wait vs `scipy.linalg.expm` | Max abs difference 4.7e-14 h |
| Transient wait at 400 h vs Erlang C | Max abs difference 2.7e-11 h |
| Kullman single-truck exactness: realized wait vs an independent Kiefer–Wolfowitz recursion on the outside-only realization (200 seeds × 5 stations × u ∈ {0.40, 0.65, 0.90}) | 3,000 cases, max difference **0.0** |
| Keskin stationary waits (u = 0.8) | c = 1: P(W = 0) 0.1985 vs 0.2000, KS p = 0.70. c = 3: 0.3522 vs 0.3528, KS p = 0.58 |
| Lazy vs eager reference | Identical actions, clocks and event logs. Default suite: 3 configs × {greedy, random} × 3 seeds. Statistical: 50 seeds. |
| Reservation prediction vs lazy simulation (300 random stations, half with outages) | 0 mismatches |

**Benchmark** (§5.4; 10 trucks, Texas long-haul, u = 0.90, all 432 stations, greedy, 5 seeds)

| Variant | Median episode (s) | Ratio | Note |
|---|---|---|---|
| No outside traffic | 7.5 | 1.00 | |
| Kullman, `on_arrival` (default) | 10.3 | **1.38** | **meets the ≤ 1.5× target**, with about 20 % more decisions (queueing) |
| Kullman, `on_arrival` with outages | 12.4 | 1.66 | episodes run longer (outages) |
| Kullman, `realtime_queue` | 33.3 | 4.47 | advances all 432 stations at every observation; not part of the target |

Profiling also removed a full scan over the 432 stations on every commitment check, replacing it with a truck → station index that the invariant checker verifies after every step. Before that fix the ratio was 1.81.

**Bugs found and fixed during the gate.**

- `predict_reservation` could return a time while another truck ahead held an undecided reservation; such a truck may release at once (A5), so the answer is now "unknown".
- The eager reference advanced the clock after the fleet had retired.
- When dropping outside-traffic events past the horizon, the eager reference did not truncate the way the lazy simulator does.

**Checklist**

| Question | Answer |
|---|---|
| Does the queue reduce exactly to M/M/c with no fleet trucks? | Yes (table above; `test_outside_only_station_matches_erlang_c`). |
| Does a single truck's realized wait equal Kullman's clairvoyant wait function? | Yes, exactly, 3,000 cases (`test_single_truck_wait_equals_kullman_clairvoyant_wait`). |
| Is lazy exactly equal to eager? | Yes (`test_lazy_queues_equal_the_eager_reference*`). |
| Can the live station state advance past the global clock? | No. `queue_at` only advances to the caller's clock; predictions run on copies; `test_A13` checks `station.clock <= max_time`. |
| Can hidden queue state reach features, masks or baselines under `on_arrival`? | No. A queued truck's scheduled reservation time is removed from its features. `test_mask_never_reads_hidden_queue_state` perturbs every station and the mask is unchanged. The invariant 8 test resamples all outside traffic after t and observations up to t are bit-identical. Kullman's myopic rule reads the live queue only at the station the truck is standing at. |
| Is the queue-or-leave epoch present and the commitment enforced? | Yes (`test_A3`, `test_A6`, `test_A8`; invariant 11 in `InvariantChecker`). |
| Does the Kullman myopic baseline follow §5.2.3 rule by rule? | Yes, one test per rule in `test_component_baselines.py`. |

**Open for you.**

- Outside-traffic service time for the headline runs: proposed `keskin_truck` (1.63 h), with `kullman_fast` as the replication check (plan §13 item 2).
- `queue_policy` default: `kullman2021` is the default whenever the block is enabled (item 3).

---

## Gate B: hours of service

**What changed**

- `models/core/hours_of_service.py`: `DriverClock` and pure functions for the US, EU and Bai regimes, including the EU split break, extended days, and reduced rests counted under Art. 8(2).
- `models/core/hos_audit.py`: an independent audit. It rebuilds the duty timeline from `export_event_log()`, re-encodes the activity table, and checks with a run-based sweep.
- `models/environment/hos_tracker.py`: per-truck clocks and leg planning. Legs use Goel rest insertion with `anywhere`, or a counted unplanned roadside break with `stops_only` + `realized`. Rests are logged as `HOS_REST` events.
- Env hooks: activities are applied as they end; rests have their own completion event and a `resting` state.
- The certificate inserts rests greedily before a witness leg and rejects the instance if none fits.
- The greedy navigator rests when the mask allows no driving, and charges instead where it can (charging counts as break time).
- `baselines/optimization/bai2023_dp.py`: an exact single-truck DP, with brute force as its oracle.

**Oracle results**

| Check | Result |
|---|---|
| Hand-computed `DriverClock` tables | 27 US, 29 EU and 25 Bai sequences, all exact |
| `DriverClock` vs `hos_audit`, 10⁴ random schedules over 6 regime configurations | 10,000 / 10,000 agree; 4,577 contain violations |
| Bai clock vs Bai's recursion (eq. 4, constraints 7–9) | 2,000 random sequences, step-by-step equal |
| Bai DP vs exhaustive enumeration | 300 / 300 instances agree (290 feasible) |
| `insert_rests` on a 25 h US leg | 9 segments, arrival at exactly 46 h |

**Bugs found by the clock/audit cross-check.**

- A completed 45-minute EU break streak was later counted as the first 15-minute part of a split break.
- The EU reduced-rest rule was first written as "a rest under 11 h is reduced", then corrected to the Art. 8(2) wording (the part within 24 h after the previous rest).
- The audit's first-shift window start ignored `initial_state: fresh`.

**Primary-text confirmations.**

- Reg. 561/2006 Art. 6(1) and Art. 8(2), 8(4) were checked against the text on legislation.gov.uk (these paragraphs were unchanged by the 2020 Mobility Package): 9 h, 10 h at most twice a week; the portion within 24 h decides a reduced rest; at most three reduced rests.
- **Still to confirm:** Goel's "rest anywhere" placement rule against the full texts of Goel (2009, 2010) and Goel & Kok (2012).
- **To flag in the paper:** "EU charging counts as a break" (CORTE trade-press guidance) as an interpretation.

**Model notes.**

- California and Texas legs exceed the EU 4.5 h allowance, so `stops_only` instances cannot be certified there. The generator rejects them, which is case B8. Statistical runs on those maps use `anywhere`.
- For EU rules, the window constrains driving only.

**Checklist**

| Question | Answer |
|---|---|
| Do `DriverClock` and `hos_audit` agree everywhere? | On every one of 10⁴ random schedules, and on every audited rollout (invariant 7). |
| Is every row of §6.3 implemented as written? | Yes; `HosRegime.earns_break_credit`, and separately `hos_audit._credit`. |
| Is contiguity enforced? | Yes (`US_TABLE` row "only contiguous time counts"; B5). |
| Can HOS alone cause `no_feasible_action`? | No. Rests stay feasible whenever driving is blocked, unless the truck is already fully rested (`REST_NOT_REQUIRED`), in which case driving is allowed. Under Bai's per-trip cap, the cap itself is the published model. |
| Is the mid-leg arrival time exact? | Yes (`test_B7_long_legs_get_their_rests_inside_and_arrive_exactly`). |

**Open for you.** HOS regime per case study. Proposed: US for Texas, Pacific Northwest and California; EU for `de_nl_be` (plan §13 item 1).

---

## Gate C: dynamic requests

**What changed**

- `models/environment/request_engine.py`:
  - the static context; EURO epochs and Poisson arrivals;
  - the EURO filter plus a witness check (energy, at most one station per segment, HOS, horizon), with filter drops counted only for epochs already revealed;
  - same-day and drayage requests; depot loading of a static instance;
  - Kancharla & Ramadurai load-dependent energy;
  - terminal resolution at init, and a regional `drayage.service_radius_hours`;
  - feature rows, metrics, and `hindsight_problem`.
- **Env:** request slots, the dynamic depot action, `WAIT` and `END_SHIFT`, reveal/release/expire system events, pickup and delivery windows, loading, and termination reasons (`success`, `requests_lost`, `requests_late`, plus the existing ones).
- **Baselines:**
  - `GreedyDispatchPolicy` (EURO greedy);
  - `WaitAtDepotPolicy`;
  - `RollingHorizonALNSPolicy`, which re-plans every decision with on-board requests fixed;
  - day-ahead ALNS, CP-SAT, MPC and the branch-and-price / RoutingBlocks export raise `NotImplementedError` naming the block;
  - `solve_request_hindsight` gives the clairvoyant bound.

**Oracle results**

- **EURO filter.** The quickstart filter, ported verbatim into the test with attribution, keeps exactly the same requests, release times and windows as the engine, epoch by epoch, on 3 seeds with energy, HOS and loading time neutralised (`test_request_process.py`).
- **Li et al. (2022), formulation (1).** On tiny one-truck instances (2–6 requests, energy off, windows open), an independent enumeration of trips with release dates gives the same number served as the clairvoyant search, on 6 instances.
- **Poisson arrivals.** The mean count is within the 99 % interval and the gaps pass KS (statistical).

**Non-anticipativity found and fixed.** Two places looked at the hidden stream:
- `WAIT` feasibility checked whether any future request existed;
- `info` drop counters counted drops in unrevealed epochs.

Now `WAIT` uses only known information (the epoch schedule, the Poisson cutoff, or revealed but unreleased requests), and drop counts include only revealed epochs. The invariant 8 test changes every request revealed after t, together with all later draws, and checks that observations and masks up to t are bit-identical.

**Terminals checked against the node lists** (limit 5 km to the nearest non-charger node). Please approve before Gate C.

| Case | Terminal | Distance | Status |
|---|---|---|---|
| Texas | Port of Houston, Barbours Cut (29.6847, −95.0051) | 2.51 km | **proposed** |
| Texas | Port of Houston, Bayport | 5.55 km | rejected (> 5 km) |
| Pacific NW / Seattle | Port of Seattle, Terminal 18 | 3.38 / 2.35 km | **proposed** |
| Pacific NW / Seattle | Port of Tacoma, Husky Terminal | 3.09 / 1.83 km | **proposed** |
| Pacific NW | Port of Portland, Terminal 6 | 6.35 km | rejected |
| California | Port of Long Beach, Pier T | 2.78 km | **proposed** |
| California | Port of Los Angeles, TraPac | 2.52 km | **proposed** |
| California | Port of Oakland, Outer Harbor | 9.17 km | rejected |

**Benchmark configs** (§12): 15 files, written by `scripts/campaigns/write_dynamic_benchmark_configs.py` and tested to be current.

- Bases: `config_dynamic_texas_sameday`, `config_dynamic_texas_drayage` (Houston, 4 h service radius) and `config_dynamic_pnw_drayage` (Seattle and Tacoma), each with `_hos_us` and `_public_u{040,065,090}` variants.
- Greedy dispatch on seed 0: about 11 Texas same-day requests, about 60 drayage requests per day.
- In the drayage configs, trucks charge at the depot and legs are short, so outside traffic hardly changes the outcome (the public-variant results are almost identical). **If the paper needs congestion to bind in drayage, it needs scarcer ports or a larger fleet.**

**Checklist**

| Question | Answer |
|---|---|
| Does the request process match EURO line by line? | Yes, apart from the documented witness extension (EURO oracle test). |
| Are unrevealed requests invisible everywhere? | Features and the action envelope show only revealed rows (`REQUEST_HIDDEN` slots are padding and get zero probability under GraphPPO). `info` drop counts are limited to revealed epochs. `WAIT` uses known information. Invariant 8 test. |
| Are manifests fixed at depot departure, with preemptive returns only when enabled? | Yes (`MANIFEST_LOCKED`, `REQUESTS_ON_BOARD`; `test_C11`). |
| Are hard-deadline late deliveries counted as unserved? | Yes (`test_C5_C6_late_deliveries`). |
| Do drayage routes obey Xiao's arc rules? | Yes (`test_C15_C16`, `test_C17`). |
| Can `WAIT` loop at zero time? | No. It always ends strictly later (`test_C9`). |
| Is the hindsight problem complete? | Yes, including hidden requests (`test_hindsight_problem_holds_every_request_with_its_times`). |

**Still to confirm:** Voccia et al. (2019) arrival-time distribution and Jia et al. (2022) dynamic drayage arrivals, both from the primary texts (optional, plan §13 item 6).

---

## Final gate

- **Formal model:** `docs/models/dynamic_stochastic_etruck_model.md`, covering epochs, state, actions, exogenous information, transition, objective, source mapping and departures.
- **Invariants (§8.5),** checked after every step by `InvariantChecker`:
  - default suite on Seattle, each feature alone and all together, greedy and random;
  - statistical: 200 seeds × 5 feature sets × {Seattle, California, Texas long-haul}, all passing.
- **Scale (§8.6).**
  - 128 trucks share one public station at u = 0.90 with outages: no deadlock, invariants hold.
  - The request envelope can be filled exactly, and one fewer raises.
  - All three components on Texas long-haul with 10 trucks, US and EU (X4); the 50-seed version is statistical.
- **GraphPPO (§8.7).** The canonical graph builds with 0, 1 and a full envelope of revealed requests. A PPO run completes with no refused action: 512 steps in the default suite, 20,000 steps statistical.

### Fixes after the first test pass

Three rounds of local `/code-review` on the components, then one on the request engine, the metrics and the dynamic baselines. Each fix has a regression test.

**Public stations (Gate A)**

- A port reservation that the drain had not yet scheduled could leak when the truck left or the station closed. Reservations are now woken at once (`_absorb`), freed on departure (`_collect`), and a missed one raises.
- Unscheduling a reservation now invalidates its wake token, so a stale `port_reserved` event is ignored.
- The reservation prediction returned a time while another fleet truck ahead still held an undecided reservation; it now returns none until that truck decides.
- At equal times, queries now apply the simulator's own order (a fleet truck before an outside vehicle).
- The eager reference truncates at the horizon exactly as the lazy simulator does.
- The first 50-seed lazy/eager run, on code from before these fixes, had one failure: `random-legacy`, a clock reversal at 20.8 h. All 50 seeds of that cell pass on the current code. I did not isolate which of the fixes above removed it.
- A station closure logs the fleet trucks it sends away, and can no longer close the private depot charger.
- `KullmanMyopicPolicy` counted stations since the last *customer*. It now counts since the last arrival anywhere other than a station, as the environment's `stations_since_stop` does (asserted equal at every step).
- A truck → station index replaced a scan of all 432 stations (on-arrival slowdown 1.81× → 1.38×).

**Hours of service (Gate B)**

- `required_rest` asked for a break when the 14 h (or 13 h) window would close during that break. It now asks for the daily rest.
- The HOS features use the extended EU daily limit when it is enabled, and the window column is clipped at 0.
- `HosTracker.plan_leg` advances the clock to the departure time before planning, catches only `HosInfeasible`, and keeps rests already placed when Bai's trip cap is hit.

**Dynamic requests (Gate C)**

- The perfect-information bound now scales the realized leg energy (Kancharla) and checks pickup deadlines against the realized leg. Canonical loading is refused unless loading time is zero and preemptive returns are off.
- `all_served()` (and the `all_customers_served` feature) counted hidden requests as unserved. It now counts revealed requests only. Dynamic modes have no sequential routing and no automatic completion at the depot, so this changes no trajectory; a truck still ends only by `END_SHIFT`.
- Mid-episode metric snapshots now include elapsed loading time and elapsed waits for request windows.
- Drayage terminals are excluded from the customer pool, which is computed once at init. Init raises if the service radius leaves fewer nodes than the static context plus the depot. Each customer is paired only with a terminal within the radius.
- `RequestRegistry.get(-1)` raised nothing (Python indexing); negative ids now raise `KeyError`. A service start at t = 0.0 was treated as missing when computing the delay.

**Second local review** (`/code-review high` on the uncommitted changes). It made 10 findings; 9 were acted on and 1 was a false positive.

- **Station reservations.** A closure or departure dropped a truck's scheduled wake-up but not one still waiting to be scheduled, so the next drain woke the truck anyway. Depending on event order, this either raised in `confirm_reservation` or left a stale entry that kept the truck from being woken at its next station. `_unschedule` now drops both (`test_closure_drops_a_reservation_not_yet_scheduled`).
- **Kancharla energy on charger legs.** Customer and depot legs were scaled, but a loaded drayage truck's leg to a charger was priced and drawn as if empty. The mask (`extension_route_check`) and the realized draw (`_execute_navigation_action`) now both apply the load factor (`test_C19_a_loaded_leg_to_a_charger_is_scaled_too`).
- **`wait_only_at_depot`.** It was exact only without time windows, but this was never checked. `solve_request_hindsight` now raises if windows are enabled.
- **Kullman's queue peek.** It advanced the live station without the fleet-first tie rule. It now uses `fleet_arrival=True`, like every other observation.
- **Eager horizon truncation.** It fired for a station event at exactly the horizon, while the lazy simulator truncates only past it. Both now use `> max_time`.
- **Rest insertion.** `HosTracker.plan_leg` had its own copy of the insertion loop, without the non-convergence guard. It now calls `insert_rests`, whose `HosInfeasible` carries the rests placed so far (Bai's trip cap).
- **`set_station_available`.** It now takes an optional `clock` on every station class, so the environment always passes it instead of checking `hasattr`.
- **Station-hop counter.** The environment now keeps `stations_since_stop` in every mode. Kullman's policy reads it instead of rebuilding it from the event log, and `InvariantChecker` checks it against the arrival log after every step.
- **Changed draws.** Removing terminals from the drayage customer pool (first review round) changes the depot and static-context draws of drayage configs without a service radius. Both benchmark drayage configs set a 4 h radius, so this does not affect them. Pairing customers only with terminals within the radius (this round) can change which terminal a request uses in the two-terminal Pacific Northwest configs.
- **False positive.** The review suggested that depot-loading certification ignores the loading time. Depot pickups are built with zero service time (`_build_at_depot`), so the legacy replay does certify them. The comment now says so.

### Known limitations

- **EU daily-rest window.** The audit flags driving that ends after the 13 h window (15 h with a split rest). It does not flag non-driving work (charging, service, queueing) past the window, which Art. 8(2) also forbids because the 11 h rest no longer fits in the 24 h period. The clock and the masks follow the same rule, so the simulator and the audit agree; both are more lenient than the regulation in this one respect.
- **Drayage congestion.** See Gate C: outside traffic hardly binds in the drayage benchmarks.

### Edge cases → tests

| Case | Test |
|---|---|
| A1, A2 | `tests/unit/test_public_station_queue.py::test_A1_*`, `::test_A2_*` |
| A3–A9, A13–A16 | `tests/integration/test_public_charging_env.py::test_A3_*` … `::test_A16_*` |
| A10 | `test_extension_config.py::test_A10_*`, `test_public_charging_env.py::test_A10_*` |
| A11, A19 | `tests/unit/test_extension_config.py::test_A11_*`, `::test_A19_*` |
| A12 | `test_public_station_queue.py::test_A12_*`, `test_public_charging_env.py::test_A12_*` |
| A17 | `tests/integration/test_dynamic_requests_env.py::test_A17_*` |
| A18 | `tests/unit/test_queueing.py::test_A18_*` |
| B1–B5, B11, B14 (clock) | `tests/unit/test_hours_of_service.py` tables (rows annotated B1–B5, B11, B14) |
| B6–B14 (env) | `tests/integration/test_hos_env.py::test_B6_*` … `::test_B14_*` |
| B7 (exact split) | `test_hours_of_service.py::test_B7_*` |
| B15 | `test_extension_config.py::test_B15_*`, `test_hos_env.py::test_B15_*` |
| C1–C20 | `tests/integration/test_dynamic_requests_env.py::test_C1_*` … `::test_C20_*` (C5/C6, C15/C16 combined) |
| C14 | also `tests/unit/test_request_registry.py::test_C14_*` |
| C2 | also `test_request_registry.py::test_C2_*` |
| C21 | `tests/unit/test_extension_config.py::test_C21_*` |
| C7, C22 | `tests/integration/test_golden_traces.py` (static mode bit-identical), and `test_C7_*` |
| X1–X4 | `tests/integration/test_combined_extensions.py::test_X1_*` … `::test_X4_*` |

### Departures from the plan

These are recorded here and in the formal model, §8.

- The four extra feasibility reasons (Gate 1).
- `FleetTaskRegistry` kept for static mode.
- `SERVED_LATE` also marks hard-deadline late deliveries.
- System events at the horizon are dropped.
- The plan's "reveal at release" for EURO is read as EURO's own process: revealed at the epoch start, released after the margin.
- `outages.interrupt_sessions: true` and `depot_charging.parallel_with_loading: true` raise instead of being modelled (no silent relaxation).
- `drayage.service_radius_hours` is new: drayage is regional, and without it Texas drayage generated no feasible request.

### Test counts

Final runs on the working tree after both review rounds (2026-10-21), with the commands under "How to reproduce the numbers". Every row below ran on the same final code.

| Suite | Passed | Failed | Notes |
|---|---|---|---|
| Unit, not statistical | 658 | 8 (+22 errors) | All pre-existing: `test_case_studies` data files (6 failed, 22 errors) and `test_completion_shield` (2) |
| Integration, not statistical | 996 | 2 | Both pre-existing: `test_joint_reset_creates_one_shared_fleet_instance` and the perfect-information CLI timeout |
| Statistical | 50 | 0 | See below |

`test_sequential_environment.py::test_nominal_planners_preserve_prescribed_routes[cpsat]` is load-sensitive. It gives CP-SAT 2 s on one worker, and it failed in the full run while about 30 test processes shared the machine. Rerun on its own, it passes (counted above). It touches no component.

Statistical cells (all `-m statistical`):

| Cells | Tests | Result |
|---|---|---|
| Queue and station oracles (`test_queueing.py`, `test_public_station_queue.py`) | 23 | all pass |
| Invariants after every step, 200 seeds: {`public`, `hos_us`, `hos_eu`, `same_day`, `all`} × {Texas long-haul, California, Seattle} | 15 | all pass |
| Lazy queues = eager reference, 50 seeds: {Kullman, legacy, outages + real-time} × {greedy, random} | 6 | all pass |
| Mixed components on Texas long-haul, 50 seeds × {US, EU} × {greedy, random} | 4 | all pass |
| GraphPPO, 20,000 steps on a three-truck dynamic config | 1 | passes |
| Poisson arrival counts and gaps | 1 | passes |
