# Audit of the environment logic: public chargers, hours of service, dynamic requests

Audit of branch `NewFormulation` at `cb90934`, 2026-10-21, against the formal model `docs/models/dynamic_stochastic_etruck_model.md` and the plan `docs/campaigns/dynamic_stochastic_etruck_model_plan.md`. The fixes below are uncommitted, in the working tree.

## Method

1. **Code reading against the model.** Event loop and same-time order, the feasibility engine, the public-station queue and its facade, the queueing formulas, the driver clock and tracker, the request engine and registry, the v7 features, metrics, rewards and configuration validation.
2. **An independent accounting audit.** Each finished episode is rebuilt from `Truck.export_event_log()`, the request registry and the station logs, without the environment's own bookkeeping. Checked per episode:
   - every truck's activities never overlap and cover its whole life, and each activity's logged total equals the truck's counter;
   - the battery changes only by logged discharges and charges, energy balances, every charge stays within the station's power, travel and energy draws stay within their clip bounds, and each leg's distance is nominal time times speed;
   - every request is loaded by its owner at its pickup inside the pickup window, served at its delivery no earlier than the window opens, with the late flag and delay consistent; payload never exceeds capacity; a loaded same-day truck drives only to its own deliveries or stations; a drayage truck carries one container and goes from the depot to an origin first;
   - every fleet queue wait ends at the logged port start, no outside vehicle that arrived later starts first, and no two sessions overlap on a port;
   - the episode reward equals the sum of step rewards and `success` agrees with the termination reason;
   - the repository's `InvariantChecker` after every step, and `hos_audit` at the end.

   The sweep covered 23 configurations on California, Seattle and Texas long-haul: every public-station option, US, EU (extended and reduced days) and Bai rules with both rest placements and the worst-case time guard, same-day with hard, soft and Poisson arrivals, preemptive returns and initial requests, drayage with network and Kancharla energy, depot manifests, static time windows, two combined cases (one under `nominal_eta`). Each ran with its baseline and a random feasible policy (and Kullman's policy for public stations), 20 seeds each. Injecting one fault per check family (a counter drift, a wrong discharge, an overtaking outside vehicle, a late delivery marked on time, an exceeded payload) made the matching check fail, so a clean result is meaningful.
3. **The test suites**, before and after the fixes.

## Results

| Suite | Before | After |
|---|---|---|
| Unit, not statistical | 658 passed, 8 failed, 22 errors | 679 passed |
| Integration, not statistical | 996 passed, 2 failed | 1,003 passed |
| Statistical | 50 passed (gate report) | 50 passed |
| Accounting sweep (940 episodes, 81,715 steps) | One-seed smoke run: 397 distance mismatches (finding 4) | 0 problems |

The three modules that needed `osmium` (`test_contract_chains.py`, `test_od_tree_sums.py`, `test_road_graph_speed_cap.py`) now run in the default suite.

## Findings and fixes

Each fix has a regression test that fails on the code before the fix.

### 1. Hard time windows ignored the rests placed inside a leg (hours of service)

**Symptom.** With `rest_locations: anywhere` and hard customer windows, trucks failed with `time_window_violation_after_realization` although travel and service were deterministic: 5 and 8 trucks in 40 episodes on California with traffic and unloading noise off.

**Cause.** The window check used `now + nominal travel time`. The rests that Goel placement inserts inside the leg are fixed at departure and were not counted. In one case a truck left at 34.4 h on a 3.6 h leg (window closing 47.4 h); its shift ran out mid-leg and the 10 h daily rest put the arrival at 48.0 h.

**Fix.** `feasibility.hos_leg_hours` returns a leg's elapsed hours with the rests `HosTracker.plan_leg` will insert. Both window checks use it: static customers (`_hos_window_check`, inside `extension_route_check`, so `step` and the mask agree) and request pickups. With the fix the same episodes have no window failures. Test: `test_hos_env.py::test_a_window_check_counts_the_rests_inserted_in_the_leg`.

### 2. A same-day pickup window left no time for loading

**Symptom.** Under hard deadlines a truck could start loading up to the latest pickup, `ℓ − d(depot, c)`, and then arrive `loading_service_hours` (0.2 h) late, nominally certain. Such a delivery counts as not served, and it also avoided the lost-request penalty.

**Cause.** Plan §7.2 item 3 sets the latest pickup to `ℓ − d(depot, c)`; item 4 says loads "whose window can no longer be met nominally" are masked. The code followed item 3. The request generator's witness already required on-time delivery after loading, so the two disagreed.

**Fix.** Latest pickup is now `ℓ − d(depot, c) − loading_service_hours`. A candidate left with no pickup window is dropped by the extended filter and counted as `dropped_by_witness`, since EURO's own filter has no loading time. The formal model §3.3 now states the window. Test: `test_dynamic_requests_env.py::test_C23_the_last_hard_pickup_still_meets_the_window_nominally`.

### 3. `nominal_eta` exposed the realized unloading draw of a truck in service

**Cause.** Mid-leg trucks showed nominal values, but a truck already unloading (or waiting for a window with a pre-drawn service time) showed the realized end of its stochastic unloading in `next_event_at`, `next_event_remaining_hours` and the service-duration column. The non-anticipativity test could not see it, because the draw is made before the cutoff.

**Fix.** Under `nominal_eta` such a truck shows the nominal service time and the matching end time. Formal model §2 updated. Test: `test_combined_extensions.py::test_nominal_eta_hides_the_unloading_draw_of_a_truck_in_service`.

### 4. Request-mode legs reported a traffic-inflated distance

`_launch_leg` computed distance as realized time × speed, while static navigation uses nominal time × speed; a delayed leg was reported as longer. Both now use nominal time. Only the distance metric changes. Test: `test_dynamic_requests_env.py::test_request_legs_report_the_nominal_distance`.

### 5. The greedy baselines rested whenever nothing could be driven to

**Symptom.** In every drayage + hours-of-service run all ten drivers took a 10 h daily rest at t = 0: no container was released yet and stations are barred before the first origin.

**Cause.** `component_action` took a rest whenever no navigation action was feasible, and fell back to any rest when the clock required none. Plan §6.7 says the navigator "rests when the mask requires it".

**Fix.** It rests only when a driving action is refused for an hours-of-service reason (`HOS_DRIVING_LIMIT`, `HOS_SHIFT_WINDOW`); otherwise the caller decides (the dispatcher waits). The environment is unchanged: a daily rest at t = 0 stays feasible, because a fresh driver is on duty and the rest moves the 14 h window. Test: `test_hos_env.py::test_greedy_rests_only_when_the_driving_rules_block_driving`.

### 6. `requests.time_windows` was not validated

Unknown keys (a typo such as `window_width`) silently fell back to defaults, and reversed bounds were accepted. It now raises like every other block, with the static windows' bounds rules. Test: `test_extension_config.py::test_request_windows_reject_unknown_keys_and_reversed_bounds`.

### Failing tests

| Test | Cause | Fix |
|---|---|---|
| `test_contract_chains.py`, `test_od_tree_sums.py`, `test_road_graph_speed_cap.py` | `road_graph.py` imported `osmium` at module level | Imported inside the two PBF readers, as `chargers.py` already does |
| `test_completion_shield.py` (2) | The shield looks tasks up by `node_id` since `dcfc3a2`; the stub task had none | Stub task given `node_id=1` |
| `test_joint_environment.py::test_joint_reset_creates_one_shared_fleet_instance` | Expected 150/350/750 kW; `866708a` moved every config to one 150 kW class | The test sets the three classes itself, so it still checks assignment |
| `test_perfect_information_cli.py::test_seeded_cli_solve_and_cross_process_replay` | 60 s subprocess limit; a cold start spends ~50 s importing on the network filesystem (warm: 4 s) | Limit 600 s, as a hang guard |
| `test_case_studies.py` (6 failed, 22 errors) | `europe_10k_c300/c500` metadata names configs that were never committed; the 600 MB `od_*.npy` matrices are gitignored | Added the two configs (`config_joint_europe_10k.yaml` with the data path changed); the matrix checks skip with the rebuild command when the matrices are absent |

## Checked and correct

- Event loop: system events before arrivals, completions and other ready events at equal times; trucks in id order; no clock reversal; exogenous events at the horizon dropped; requests still revealed and expired after the fleet retires.
- Mask and execution use one engine; clairvoyant legs only on private perfect-information states.
- Public stations: pooled FCFS M/M/c with stationary start, keyed (common-random-number) arrivals, services and port timelines; queue-or-leave at a busy arrival, commitment at the head, release when nothing can be charged; exact reservation prediction; fleet before outside vehicles at equal times.
- Queueing formulas: Erlang C through the Erlang B recursion, `W_q = C / (cμ − λ)`, exact stationary sampling with a geometric tail, newcomer wait `(n − c + 1)/(cμ)`, and the spectral transient wait (off-diagonals `√(λ min(n+1, c) μ)` from detailed balance).
- Driver clock: 49 CFR 395.3 (8 h before a 30 min break with any non-driving time counting, 11 h in a 14 h window, 10 h reset) and Reg. 561/2006 Arts. 6–8 (split break, extended days, reduced rests counted under Art. 8(2)); Goel placement; stops-only masking.
- Requests: the EURO epoch range and filter, per-field sampling with replacement, same-day and drayage stops, legal lifecycle transitions, manifests fixed at departure, preemptive returns, Xiao's arc rules, payload including requests being loaded, `WAIT` that never loops at zero time and wakes only on information that is then revealed.
- Keyed streams: draws depend only on scenario, stream and key; the station, port and request keys never collide.
- Private depot charger: fleet-only, no outside traffic, and removed cleanly at reset.

## Open points for your decision

1. **Dead ends are left to the policy.** The mask checks the energy of the next leg only, so a truck can deliver into a place from which no charger or depot is reachable, and then fails with `no_feasible_action`. Measured after the fixes, with 20 seeds on California, the repository baselines strand a truck in 14 runs with depot manifests, 9 in LA drayage, 6 in same-day with epochs, 5 in same-day with Poisson arrivals, 5 under Kullman's rule and 1 with US rests at stops. Houston drayage strands a truck in 9 of 10 seeds. With manifests a loaded truck may have only that leg left. Options: keep it (learning to avoid dead ends is part of the task), add an opt-in continuation guard to the mask, or set `require_continuation` in `GreedyDispatchPolicy`'s navigator.
2. **Hard late deliveries are not penalized.** A late delivery under hard deadlines gets no bonus but also no penalty, while a lost request costs `lost_request_penalty_hours`. Both count as not served. Fix 2 removes the nominally certain cases; traffic can still make a load late. Option: charge the lost-request penalty for hard late deliveries too.
3. **`termination_reason` reports the last cause.** A truck stranded mid-episode sets `no_feasible_action`, but reaching the horizon later overwrites it with `time_limit_exhausted`. `info["failure_causes"]` keeps every failure.
4. **Station memory is written on arrival** (and when leaving a queue), not at the end of charging. This follows the plan's reading of Kullman §2.2 ("on arrival it observes the queue"). If Kullman's memory updates at every epoch at a station, add an observation in `finish_charging` and at the head of the queue.
5. **With outages, the stationary initial count assumes every port is up.** Ports start in their stationary up/down states independently. This is an approximation; document it or condition the count on the up ports.
6. **Kullman's myopic baseline can loop.** On California seed 0, a truck that cannot reach the depot directly shuttles between two single-port stations until the 200 h horizon, queueing 27–32 h each time. This is the rule-by-rule baseline; its fallback (cheapest action) has no progress rule.
7. **The pickup-window check is necessary, not sufficient.** It does not foresee a charging stop or a rest on the later delivery leg; such deliveries are carried out and counted late, as the model states.
8. **A fresh driver is on duty from t = 0.** Waiting at the depot for the first release is on-duty time (49 CFR 395.2), so the 14 h window starts at t = 0 although the first containers are released after about 2 h. With hours of service, drayage is 21–24% on time with the greedy baseline (Houston, Seattle–Tacoma, LA). Before fix 5 the baseline rested from t = 0 to 10 h by accident, which moved the window onto the release period and gave 39% on California. Option: an `initial_state` in which the shift starts at the first dispatch.

## Effect of the fixes on baseline results

The fixes change baseline trajectories only in component configurations; the golden traces of the legacy mode are unchanged. Measured on California (20 seeds, greedy baselines, mean of per-run on-time shares), before → after:

| Scenario | On-time share | Why |
|---|---|---|
| LA drayage + public chargers + US hours | 39% → 24% | Fix 5: no more accidental rest at t = 0 (open point 8) |
| Same-day, EURO epochs, hard deadlines | 84% → 84% | Fix 2 rarely binds: the greedy dispatcher loads as soon as a request is released |
| Every other California scenario (static, depot manifests, public stations, hours of service, same-day with Poisson arrivals, LA drayage alone, same-day with all components) | unchanged | |

## How to reproduce

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit tests/integration -m "not statistical"
PYTHONPATH=. .venv/bin/python -m pytest tests -m statistical
```

The accounting harness and the sweep runner used here are not part of the repository.
