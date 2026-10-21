# Implementation plan: shared public chargers, hours-of-service, and dynamic requests

Plan date: 2026-10-20. Branch `campaign/ieee-tits-revision` at `eecdaab`, with the uncommitted depot-ring and delivery-quota work listed in `git status`. No code was changed while preparing this plan.

This plan is written for the coding agent that will implement it and for the reviewers who audit each phase. It extends the event-driven simulator with three established model components:

- **A. Public DC-fast stations shared with outside traffic**, including charger outages and an explicit information model. No slot reservations.
- **B. Driver hours-of-service (HOS) rules**, with charging synchronized to mandatory breaks.
- **C. Dynamic requests**, covering same-day freight and electric drayage.

It also removes the per-truck delivery quota, which the classic E-VRP literature does not have.

Each component must reproduce a published model **one-to-one**. Where this repository has to differ (fixed fleet, event-driven decisions, nonlinear charging), the difference is written down in Section 2 and tested. The combination of the three components is the new contribution. No component is invented here.

---

## 0. Ground rules for the implementing agent

1. **Opt-in only.** Every new behaviour sits behind a new config block that is off by default: `public_charging`, `depot_charging`, `hours_of_service`, `requests`. With all blocks absent, the simulator must produce **identical trajectories** to the Phase 0 golden traces (Section 3.2). Any intended change to legacy behaviour is its own reviewed commit with its own test.
2. **One feasibility engine.** All new constraints (HOS limits, port availability, request status, payload, manifests) live in `EVRoutingEnv/state/feasibility.py`. The policy mask, `step`, the greedy navigator, ALNS execution, CP-SAT execution and the perfect-information search must all go through it. Do not add a second copy of any rule.
3. **No silent relaxation.** If an instance, request or regime cannot be modelled as specified, raise `ValueError` with the reason, following the existing `_positive_integer` / `_nonnegative_finite` idiom and the "removed key" pattern used for `min_hop_distance`.
4. **Reproducibility.** All new randomness comes from named `ScenarioRandomStreams` streams keyed so that **fleet actions never change exogenous draws** (common random numbers). Two different policies on the same seed must see the same outside-vehicle arrival times, the same outage timeline and the same request stream.
5. **Non-anticipativity.** The observation and mask at time `t` may depend only on information revealed by `t` under the configured information model. Every component has an explicit test for this (Section 8.5, invariant 8).
6. **Append-only enums.** New `ActionKind`, `FeasibilityReason`, `TaskStatus` and `EventType` members are appended last. `features.py` derives `kind_code` from `enumerate(ActionKind)`, so inserting members in the middle silently renumbers every existing code.
7. **Match the surrounding code.** Use the existing comment density, validation helpers, `heapq` event idiom and test layout (`tests/unit`, `tests/integration`, `pytestmark = pytest.mark.integration`).
8. **Stop at every review gate** (Section 10) and ask the user to run the review. Only the user can launch `/code-review ultra`; do not try to start it yourself.

---

## 1. What the repository does today

Facts from the code, verified on 2026-10-20. Function names are stable; line numbers will drift.

| Area | Current behaviour | Where |
|---|---|---|
| Event engine | Two event types, `TRUCK_READY` and `TRUCK_ROUTING`. Ordered by `(time, priority, truck_id)`. `Event.__post_init__` rejects `truck_id < 0`. | `models/environment/event_handlers.py` |
| Event loop | Its first statement indexes `self.trucks[self.event_queue[0].truck_id]`. **Trap:** a system event with `truck_id = -1` would silently read the *last* truck. | `EventDrivenTruckEnv._advance_to_next_decision` |
| Charging stations | Fleet-only FCFS pooled waitlist per station with `c_s` ports. `check_charger_gating`, `wake_waiting_trucks` and `pending_wake_trucks` implement strict FCFS. `set_station_available` closes a whole station. **No outside traffic.** A truck that finds no free port joins the waitlist automatically, with no decision on arrival, and may leave when woken at the head. Kullman et al. (2021) instead give the vehicle a queue-or-leave decision on arrival and then commit it (Section 5.2). | `models/simulation/charging_station.py` |
| Queue lookup | `waiting_time_lookup.json` is loaded and hashed but **never used by the simulator**. It is exactly the M/M/c (Erlang C) mean wait `W_q`, with mean service 0.5 h (DCFast) and 2 h (Level2), for utilizations 0.05–0.95. Verified to 0.0 error for every capacity in the file. The same file is copied into every case study. | `ChargingStation.get_waiting_time`, `data/case_studies/*/waiting_time_lookup.json` |
| Charging | Nonlinear CC-CV curve with taper. Target-SoC actions. Station-specific power and efficiency. | `models/simulation/charging_curve.py`, `_execute_charge_action` |
| Travel | The whole leg (time and energy) is realized **at departure**. Time is Gaussian, clipped to `[0.85, 2.5] × nominal`. Energy is correlated with traffic, multiplier in `[min, max]`. | `TrafficSimulator.apply_traffic`, `apply_energy_uncertainty` |
| Tasks | `CustomerTask` has one node, demand, service time and a time window. `FleetTaskRegistry` maps **node → task one-to-one** (`_task_id_by_node`). | `models/core/customer.py` |
| Payload | Trucks start with full payload. It is deducted on service. A truck away from the depot can claim **any** unclaimed customer (see Section 1.1). | `Truck.can_accept_demand`, `complete_customer_service` |
| Depot | Selecting the depot closes the truck's route (retire). There is no depot charging. | `_execute_navigation_action`, `_complete_joint_customer_service` |
| Time windows | Hard. A realized late arrival fails the truck (`time_window_violation_after_realization`). | `EventHandler.handle_truck_routing` |
| Action envelope | `Discrete(num_chargers + num_stops + 1 + num_charge_values)`. Slot `i` in the customer range means "node of task `i`". | `__init__`, `joint_action_feasibility` |
| Features | `SCHEMA_VERSION = "joint-fleet-v6"`. Node types are truck, customer and charger. | `state/features.py` |
| Instance certification | Every generated instance comes with a witness route set, replayed on a shadow copy of the environment, with at most one charger between stops. | `joint_instance.py`, `route_walk.py`, `_certify_instance` |
| Delivery quota | Uncommitted. `problem.deliveries_per_truck` with `DELIVERY_QUOTA_REACHED/UNMET`, plus hooks in ALNS, the canonical baselines and CP-SAT. | see Phase 0 |

### 1.1 Two modelling issues reviewers should rule on

These exist today and affect how the paper can describe the model. Phase 1 adds options for both. It does not change the legacy defaults without the user's decision.

1. **Implicit loading in the static joint mode.** A truck that has already left the depot can claim any unclaimed customer, as if it carried every customer's goods. That is physically valid for *service* or *pickup* tasks. For *deliveries of goods*, assignment must be fixed when the truck loads at the depot. The new request model (Phase C) fixes this with depot manifests. For the legacy static mode, add `problem.loading: implicit | at_depot`. `implicit` is the default for bit-identity. Report both in the paper, or state the service-task interpretation explicitly.
2. **In-flight realizations are observable.** The truck-history features `route_arrival_at`, `next_event_at`, `next_event_remaining_hours` and `pending_discharge_kwh` expose the *realized* arrival time and energy of legs still in progress. The docstring treats them as "committed". A real dispatcher sees at most a telematics ETA. Add `observation.in_flight_information: realized | nominal_eta`. `realized` is the default for bit-identity. In `nominal_eta`, in-flight rows use departure time plus nominal time and nominal energy until arrival. All new components must be tested under `nominal_eta`.

---

## 2. Source models and the one-to-one mapping

"Verified" means the detail was read in the source. "Verified (survey)" means it was read in a detailed secondary source: Li, Archetti & Ljubić (2024), *Emerging optimization problems for distribution in same-day delivery*, arXiv:2405.05620v2. "To confirm" means only an abstract was available; **the implementing agent must settle those rows before the gate of that phase.**

Full texts read on 2026-10-20 (supplied by the user where marked): Kullman, Goodson & Mendoza (2021), HAL v3 (user); Xiao et al. (2025) (user); Keskin, Akhavan-Tabatabaei & Çatay (2019, WSC) (user); Li, Archetti & Ljubić (2024) (user); Bai et al. (2023); Kahlert et al. (2026); the EURO-NeurIPS 2022 environment code; 49 CFR 395.3; Regulation (EC) 561/2006 Art. 7.

### 2.1 Component A: public stations with outside traffic

| Element | Source | Reproduced one-to-one | Status |
|---|---|---|---|
| Problem class | Kullman, Goodson & Mendoza (2021), *Electric Vehicle Routing with Public Charging Stations*, Transportation Science 55(3):637–659 (HAL hal-01928730v3), §2.1: one EV visits every customer and returns to the depot; travel time and energy deterministic and known; triangle inequality; objective = expected total time (travel + charging + queueing) | Objective and cost accounting. The fleet and the stochastic travel are documented extensions | Verified (full text) |
| Depot | §2.1: the depot is private, so the EV can always start charging at once; in the instances (§7.1) the depot has fast chargers | `depot_charging`: no queue, no outside traffic | Verified |
| Public station queue | §2.1: "pooled first-come-first-served systems with ψc identical chargers, infinite capacity, and exponential inter-arrival and service times with known rate parameters (p_c,arrive and p_c,depart)", i.e. M/M/ψc/∞ | Poisson outside arrivals, Exp service, pooled FCFS, unlimited waiting room. Outside vehicles never balk or renege (nothing in the model lets them) | Verified |
| EV at a busy station | §2.1: "If all chargers are occupied when the EV arrives, it must either queue or leave. If a vehicle queues at a CS it must remain in queue until a charger becomes available, after which it must charge." | New `QUEUE` action on arrival (any routing action is a balk); no reneging once queued; at the head only charge actions are feasible (`queue_policy: kullman2021`) | Verified |
| Decision epochs | §2.2: an epoch starts when the vehicle arrives at a new location, reaches the front of a queue, or finishes charging | Arrival at a busy station becomes a decision epoch (today the truck joins the waitlist automatically) | Verified |
| Information, natural filtration F | §2.1–2.2: the EV knows nothing about station demand before it arrives; on arrival it observes the queue length and keeps a memory `M_k` of (count observed, time observed) per station, `(c, ∅, ∅)` if never observed; its own queue position `z_k` is part of the state. §4.3.2 and Fig. 4: the expected wait given a memory comes from the transient M/M/c solution (software by Andersen 2020) and tends to the long-run wait | `information: on_arrival` (default), with per-station memory features and an `expected_wait_given_memory` feature | Verified |
| Real-time information, filtration Q | §7.4.2: the decision maker sees the number of vehicles at every station in real time | `information: realtime_queue`, a published variant, not an oracle | Verified |
| Binary busy indicator | Dastpak, Errico, Jabali & Malucelli (2024), Transportation Research Part C 158:104411 | `information: occupancy_indicator` | Verified (abstract) |
| Charging | §2.1: concave piecewise-linear charging function (Montoya et al. 2017) with discrete charge levels `Q` (every 10 % plus the breakpoints). §2.2, below eq. (5): at most one station between two customers | The repository's CC-CV curve with discrete target-SoC levels (compatible). Optional `max_stations_between_stops: 1`; the instance certificate already assumes it | Verified |
| Calibration | §7.1: mean service time 26.62 min (fast) and 128.78 min (moderate), from Morrissey et al. (2016); utilization `u ∈ {0.40, 0.65, 0.90}`; `p_arrive = u · ψ_c · p_depart`; infinite buffer | Presets `kullman_fast` and `kullman_moderate`; regime grid `u ∈ {0.40, 0.65, 0.90}` | Verified |
| Scenario realization | §7.1: a queue simulator generates a day of queue events per station from `(p_arrive, p_depart, ψ_c)`, and the EV observes that realization when it visits | Per-station keyed random streams (Section 4.2) | Verified |
| Dual bound | §6.2: perfect-information relaxation (Brown, Smith & Sun 2010). The clairvoyant wait is a piecewise function of the arrival time, with jumps at outside arrivals (Fig. 4, right) | Hindsight reference, and a test oracle (Section 8.2) | Verified |
| Stationary i.i.d. waits (simpler alternative) | Keskin, Akhavan-Tabatabaei & Çatay (2019), *EVRPTW and stochastic waiting times at recharging stations*, Winter Simulation Conference, §2.2: one charger per station (M/M/1); Poisson `λ`; Exp service with `μ = 1/E[recharge]` and `E[recharge] = 0.55·Q·g` (recharge amount uniform on 10–100 % of capacity); `W = 0` with probability `1 − ρ`, otherwise Exp(`μ(1 − ρ)`); `E[W] = λ/(μ(μ − λ))`; `ρ ∈ {0.1, 0.4, 0.5, 0.8, 0.9}`; plans with `E[W]` and measures reliability with 10,000 sampled waits per visit | `queue_model: iid_stationary_wait`, generalized to M/M/c: `P(W = 0) = 1 − C(c, a)` and `W \| W > 0 ~ Exp(cμ − λ)`. The `0.55·Q·g` rule becomes the outside-truck service-time preset | Verified (full text) |
| Time-of-day waits | Keskin, Laporte & Çatay (2019), C&OR 107:77–94: waits change over the day and are estimated with M/G/1 equations (as described in the WSC paper above) | `arrival_model: piecewise_poisson`, sampled exactly by thinning (Lewis & Shedler 1979) | Abstract + secondary |
| Station availability | Sweda, Dolinskaya & Klabjan (2017), Transportation Science 51(4):1326–1348: each station is available with a given probability (Kullman §3 classes it as path finding) | Port-level two-state availability | Verified (abstract) |
| Outage calibration | Rempel et al. (2023), Human Factors: 73.3 % of 655 DC-fast ports functional | Default availability 0.733 in the "unreliable" regime | Verified (abstract) |
| Fleet contention for ports | Froger, Jabali, Mendoza & Laporte (2022), Transportation Science 56(2):460–482 | Already in the simulator; fleet trucks now share the FCFS queue with outside vehicles | Verified |

**Documented differences from Kullman et al.**

- **A fleet instead of one EV.** With one truck the extension is exact. Under FCFS a truck's wait depends only on the entries ahead of it, which it cannot influence, so its realized wait equals Kullman's clairvoyant wait function computed from the outside-only realization. Section 8.2 tests that equality. With several trucks, a charging truck delays the outside vehicles behind it, and they in turn delay later trucks. That feedback is the new modelling content.
- **Stochastic travel time and energy**, from the repository's traffic model. Kullman's travel is deterministic.
- **Fleet trucks' service time** is the deterministic CC-CV time for the chosen target, as for Kullman's EV, whose own charging follows the charging function rather than an exponential. Outside vehicles keep Exp(`μ`).
- **Utilization wording.** Kullman §7.1 describes `u = 0.90` as "the probability of all chargers being occupied when a vehicle arrives is 90 %". For `ψ_c > 1` that probability is the Erlang C value, not `u`. The plan follows their formula `p_arrive = u·ψ_c·p_depart` and reports the implied Erlang C values next to `u`.
- **Charger outages** are an addition (Sweda, Rempel), off by default.

### 2.2 Component B: hours-of-service

| Element | Source | Reproduced one-to-one | Status |
|---|---|---|---|
| US property-carrying rules | 49 CFR §395.3: no driving without first 10 consecutive hours off duty; no driving after 14 consecutive hours since coming on duty; at most 11 h driving in that period; no driving once 8 h of driving have passed without a consecutive 30-minute interruption, which may be off-duty, sleeper berth or on-duty not driving; 60 h/7 days or 70 h/8 days; 34 h restart | 10 h reset, 14 h window, 11 h driving, 30-minute break after 8 h driving. Weekly limits are not binding within the allowed horizon (validated below). | Verified (eCFR text) |
| EU rules | Regulation (EC) 561/2006, Art. 7: after 4.5 h driving, an uninterrupted break of at least 45 min unless a rest period is taken; the break may be split into at least 15 min followed by at least 30 min. Art. 6: daily driving at most 9 h (10 h at most twice a week). Art. 8: daily rest of 11 h (reduced to 9 h at most three times between weekly rests), within 24 h of the previous rest. | 4.5 h / 45 min (15 + 30 split), 9 h daily driving, 11 h daily rest within 24 h. Extended 10 h days and reduced 9 h rests are options, off by default. | Art. 7 verified; **confirm** Art. 6 and Art. 8 wording from EUR-Lex |
| Scheduling methodology: breaks and rests anywhere along a driving period | Goel (2009), *Vehicle scheduling and routing with drivers' working hours*, Transportation Science 43(1):17–26; Goel (2010), *Truck driver scheduling in the European Union*, Transportation Science 44(4):429–441; Goel & Kok (2012), *Truck driver scheduling in the United States*, Transportation Science 46(3) | `rest_locations: anywhere` (default) | **Confirm** the "anywhere" assumption and the schedule-construction rule in the full texts |
| E-truck charging aligned with rest, rest only at stations | Bai et al. (2023), *Rollout-based charging strategy for electric trucks with hours-of-service regulations*, arXiv:2303.08895: consecutive driving `T_d` = 4.5 h, rest `T_r` = 45 min, daily driving 9 h. A stay of at least `T_r` at a station resets consecutive driving; shorter charging does not (their constraints 4, 7–9). | `regime: bai2023_simplified` (test oracle) and `rest_locations: stops_only` | Verified (full text read) |
| Fleet charging coordination with HOS and congestion | Kahlert, Wang & Mårtensson (2026), *En-route charging coordination for electric trucks*, arXiv:2605.27094: rest `h` = 45 min; charger occupation `max(charging time, h)`; station overheads 5 and 7 min | Port occupancy while charging-as-break; default handling overhead 5 min | Verified (full text read) |
| Does charging count as an EU break? | Trade press on CORTE enforcement guidance (trans.info, 2025): charging counts as break or rest if the driver does no work and is free to use the time; moving the charged truck off the charger may interrupt the rest | `charging_counts_as_break` (EU, default true) with handling overhead counted as work | Secondary source; **flag in the paper as an interpretation** |

**Documented differences.** Single driver per truck. No team driving, no sleeper-berth split (US) and no 3 + 9 daily-rest split (EU). The validator caps `max_time` at 96 h so that weekly limits (US 60 h/70 h on duty; EU 56 h weekly and 90 h fortnightly driving, weekly rest) cannot bind. Working-time rules outside 561/2006 (EU Directive 2002/15) are out of scope.

### 2.3 Component C: dynamic requests

| Element | Source | Reproduced one-to-one | Status |
|---|---|---|---|
| Unified request model | Savelsbergh & Sol (1995), Transportation Science 29(1):17–29; Berbeglia, Cordeau & Laporte (2010), EJOR 202(1):8–15 | One `Request` with an optional pickup stop and a delivery stop; static, same-day and drayage are special cases | Standard |
| Release dates and multiple trips | Li, Archetti & Ljubić (2024), §3.1, formulation (1), the orienteering problem with release dates of Li et al. (2022): every trip starts and ends at the depot and carries only parcels whose release date is no later than the trip's start; the vehicle makes several trips before the deadline `T_E`; capacity constraints extend it to several vehicles | Depot manifests (pickup at the depot no earlier than the release); hindsight model (7.4) | Verified (survey) |
| Same-day information process | EURO Meets NeurIPS 2022 DVRPTW, `VRPEnvironment` in `ortec/euro-neurips-vrp-2022-quickstart/environment.py` (read in full): 3600 s epochs; 3600 s dispatch margin; up to 100 requests per epoch; customer, time window, demand and service time sampled independently with replacement from a static instance; requests that cannot be reached from the depot at planning start, or cannot return before the depot closes, are dropped; release = epoch start + margin; hindsight problem = VRPTW with release times | `arrival_process: euro_neurips_epochs`, all parameters configurable, with the EURO defaults recorded | Verified (code) |
| Same-day objective and structure | Voccia, Campbell & Thomas (2019), Transportation Science 53(1):167–184, as summarized in Li et al. (2024), §3.2: customer locations known in advance; release dates uncertain, with a known arrival rate and release-date distribution; a deterministic deadline or time window per order; uncapacitated fleet; **no preemptive returns**; maximize the expected number of requests served on time (hard deadlines); MDP; an analytical result on when waiting at the depot pays; scenario sampling with a consensus function. Instances: http://ir.uiowa.edu/tippie_pubs/65/ (also used by Côté et al. 2023) | Served-on-time objective; explicit `WAIT`; `allow_preemptive_returns: false` and `deadline_type: hard` by default | Verified (survey). The arrival-time distribution still needs the primary text |
| Preemptive depot returns | Ulmer, Thomas & Mattfeld (2019), EURO J. Transp. Logist. 8:327–361 (survey §3.2): the vehicle may return to the depot before delivering everything on board; most useful late in the horizon | `allow_preemptive_returns: true` | Verified (survey) |
| Soft deadlines | Ulmer (2017), Logistics Research 10 (survey §3.2): late delivery allowed; expected delay minimized | `deadline_type: soft` | Verified (survey) |
| Continuous-time arrivals and degree of dynamism | Larsen, Madsen & Solomon (2002), JORS 53:637–646; Poisson arrival times are the common choice (survey §2.1.2) | `arrival_process: poisson`; `initial_requests` sets the degree of dynamism | Standard |
| Electric drayage task structure | Xiao, Chen, Sun, Laporte & Baldacci (2025), *The container drayage problem for electric trucks with charging resource constraints*, Transportation Research Part C 174:105100 (full text), §3.1: homogeneous trucks leave depot 0 fully charged; tasks are O–D pairs, export (customer → terminal) and import (terminal → customer); the trailer stays attached; **one container at a time**; between a task's origin and its destination only stations may be visited; from a destination the truck goes to the next origin, a station or the depot; the first stop after the depot is an origin and the last stop before it is a destination; earliest pickup time at the origin (wait if early); the destination deadline is **soft**, with a delay penalty per minute; private or contracted stations with full occupancy information; limited chargers with queueing; processing time 2 min at origins and 4 min at destinations (§5.2.1) | Request kinds `drayage_import` and `drayage_export`, these routing rules in the feasibility engine, and these time-window semantics | Verified (full text) |
| Drayage energy | Xiao §3.1: energy proportional to distance, independent of load (main model). §6.1 extension: load- and speed-dependent energy from Kancharla & Ramadurai (2020), eq. (62), parameters in Table 10 (empty truck 7000 kg, container 10–20 t, `C_r = 0.01`, `C_d = 0.8`, `ρ = 1.2041`, `A = 7 m²`, `ε = 0.89`) | Default: the network energy table for loaded and empty legs. Option `drayage.energy_model: kancharla_ramadurai_2020` | Verified |
| Dynamic drayage arrivals | Jia, Cui, Chen & Meng (2022), *Dynamic container drayage with uncertain request arrival times and service time windows*, Transportation Research Part B 166:237–258, cited in Xiao §2.1 as an MDP model | The same arrival processes as same-day (EURO epochs or Poisson) until the primary text is read | To confirm |
| Decision model | Ulmer, Goodson, Mattfeld & Thomas (2020), EURO J. Transp. Logist. 9(2):100008 (route-based MDP); Psaraftis, Wen & Kontovas (2016) classification (survey §2.2) | Event-driven decisions whenever a truck becomes free; the EURO epoch information process is kept | Standard |

**Documented differences.**

- **From EURO.** EURO has an unlimited fleet and decides a whole epoch's routes at once. Here the fleet is fixed and decisions are taken per truck at events. The feasibility filter adds energy, charging and HOS feasibility through the existing route certificate.
- **From Voccia.** Their fleet is uncapacitated; ours has payload capacity (Li et al. note that formulation (1) extends with capacity constraints).
- **From Xiao.** Xiao is static and deterministic, with battery swapping, two charging modes and costs in yuan. This plan keeps the repository's stations and power classes, adds the dynamic arrival process, and leaves out battery swapping and the two-container extension (§6.2). It reports time and penalty metrics instead of yuan.
- **Objective.** With a fixed fleet, serving every request cannot be guaranteed, so the objective follows Voccia: requests served on time first, then cost.

---

## 3. Phase 0: remove the per-truck quota and freeze legacy behaviour

### 3.1 Remove the quota completely

The quota is uncommitted, so remove it rather than deprecate it. Keep the depot-energy ring, which is a separate, valid instance generator.

| File | Change |
|---|---|
| `EVRoutingEnv/models/environment/event_driven_env.py` | Delete the `deliveries_per_truck` parsing block in `__init__`, the instance-size check in `_install_joint_instance` and the `delivery_quota=` argument. Add a removed-key guard: if `"deliveries_per_truck" in problem_config`, raise `ValueError("problem.deliveries_per_truck was removed: the E-VRP has no per-truck customer quota; bound routes by payload, battery, duration or time windows")`. |
| `EVRoutingEnv/models/core/customer.py` | Remove `delivery_quota`, `truck_deliveries` and `remaining_quota`, and restore the `route_complete_for` docstring. |
| `EVRoutingEnv/models/environment/joint_instance.py` | Remove the `delivery_quota` parameter of `create_registry`. Keep the ring parameters. |
| `EVRoutingEnv/state/feasibility.py` | Remove the quota block and the `DELIVERY_QUOTA_REACHED/UNMET` reasons. They are uncommitted, so no stored artifact carries their codes. Confirm with `git log -S DELIVERY_QUOTA` that nothing committed references them. |
| `EVRoutingEnv/baselines/heuristics/alns.py` | Remove `slots` and `fallback_truck`, and restore the original fallback (owner, else emptiest route). |
| `EVRoutingEnv/baselines/heuristics/canonical_baselines.py` | Remove the quota filter in `_claimable_customer_nodes`. |
| `EVRoutingEnv/baselines/optimization/exact_optimization.py` | Remove the quota cardinality constraint. |
| `tests/integration/test_delivery_quota.py` | Delete. Add `tests/unit/test_environment_validation.py::test_delivery_quota_key_is_rejected`. |
| `config_joint_texas_longhaul.yaml`, `config_joint_pacific_northwest_longhaul.yaml`, `config_joint_california_10t3d.yaml` | Remove `deliveries_per_truck` and fix the header comments ("exactly 3 deliveries per truck" becomes "10 trucks, 30 customers, free assignment"). Rename the California file to `config_joint_california_10t30c.yaml`; only the campaign plan references it. |
| `docs/campaigns/generalist_graphppo_dayahead_plan.md` | In Section 1b, mark the quota sub-section as withdrawn, with the reason (not in the classic E-VRP), and restore the free-workload benchmark rows. |

### 3.2 Golden legacy traces

Before any Phase 1 code changes, add `scripts/analysis/record_golden_traces.py` and `tests/integration/test_golden_traces.py`.

- Configs: `config_joint.yaml`, `config_joint_texas_longhaul.yaml` (quota removed), `config_joint_time_windows.yaml`, `config_joint_preassigned.yaml`, `config_sequential.yaml`.
- Five seeds each. The greedy canonical baseline plus a seeded uniform-random policy over feasible actions (rejecting nothing).
- Record per step: action index, `global_clock`, active truck, the mask as a bit string, `termination_reason`, and the per-truck `export_event_log()`. Store a SHA-256 of each, plus the full JSON for the first seed, under `tests/data/golden/`.
- The test replays the recorded actions and asserts identical hashes. This is the bit-identity contract for rule 1. Observation vectors are *not* hashed, because the schema bump changes them. Section 4.5 adds a separate column-subset test for features.

**Gate 0** (Section 10) follows.

---

## 4. Phase 1: shared infrastructure

Phases A, B and C all depend on these changes. Land them first, behind flags, with the golden traces still passing.

### 4.1 System events

- Add `EventType.SYSTEM` (appended). The payload `data["kind"]` is one of `request_release`, `request_expire`, `port_state_change`, `epoch_boundary`.
- Allow `truck_id = -1` **only** for `SYSTEM` events. Validate this in `Event.__post_init__`.
- Fix the event loop's first statement so it never indexes `self.trucks` with a system event's id. Add a test with a system event at the head of the queue and three trucks: the last truck's state must not be read or changed.
- Same-time ordering, highest first: `SYSTEM` (priority -1) → `TRUCK_ROUTING` (0) → completions (1) → other ready events (2). A request released at `t` is therefore visible to a truck that becomes ready at `t`. This matches EURO, where an epoch's requests are visible at that epoch. Test every pair.
- System events past `max_time` are dropped, not left pending, and never trigger truncation by themselves.

### 4.2 Named random streams (common random numbers)

Add these `ScenarioRandomStreams` streams. Draw on demand by key; never use a shared sequential generator.

| Stream | Key | Used for |
|---|---|---|
| `station_arrivals` | `(station, k)` | k-th outside arrival time at the station (homogeneous: exponential gaps; piecewise: thinning candidates and acceptance draws) |
| `station_service` | `(station, k)` | Service time of the k-th outside vehicle |
| `station_initial` | `station` | Stationary initial state at t = 0 |
| `port_state` | `(station, port, j)` | j-th up/down duration of a port |
| `request_static_context` | `field` | Static request pool (EURO "static instance") |
| `request_epoch` | `(epoch, field)` | EURO per-epoch samples of customer, time window, demand and service time |
| `request_poisson` | `k` | Poisson arrival times and attributes |

Test: run two different policies (greedy and random) on the same seed, and assert identical outside arrival and service sequences per station, identical port timelines and identical request streams.

### 4.3 Request model: identity separate from node (needed by C; the flag also makes A and B testable on it)

Replace `CustomerTask` and the node → task map with:

```text
Stop(node, earliest, latest, service_hours)
Request(request_id, kind, pickup: Stop | None, delivery: Stop, demand,
        release_time, status, owner, timestamps...)
kind   ∈ {static_delivery, same_day, drayage_import, drayage_export}
status ∈ HIDDEN → OPEN → CLAIMED_PICKUP → LOADING → ON_BOARD → CLAIMED_DELIVERY
         → IN_SERVICE → SERVED, plus LOST (window passed while OPEN)
         and SERVED_LATE (only with deadline_type: soft, or in drayage)
```

- Legacy static mode: `pickup = None` and the request starts `OPEN`. Its slot means "go to its delivery node", exactly as today. A thin `CustomerTask` view keeps the existing tests and baselines working until they are migrated. The view must raise if a node hosts more than one live request.
- `FleetTaskRegistry` becomes `RequestRegistry`. Requests are keyed by `request_id`, with a node → live-requests multimap. All transitions go through registry methods that validate the source status. Illegal transitions raise `RuntimeError`, never silently pass.
- Action slot `r` is request `r`. Slots are allocated in reveal order. Ids are never reused within an episode. The envelope is `requests.max_requests`; going over it raises at generation time and never truncates silently.
- Slot meaning for active truck `k`:

| Request state | Meaning of slot `r` |
|---|---|
| `OPEN`, has pickup | Drive to the pickup node and claim the request. If `k` is already at the pickup node (same-day at the depot), load with zero travel. |
| `ON_BOARD` on `k` | Drive to the delivery node |
| `OPEN`, no pickup (static) | Drive to the delivery node (legacy) |
| anything else | Infeasible, with a specific reason |

- Payload is checked at **pickup**: on-board demand plus the new demand must not exceed capacity. Payload is released at delivery. In legacy static mode the current check-at-service behaviour is kept.

### 4.4 New actions

Append to `ActionKind`: `QUEUE`, `REST_BREAK`, `REST_DAILY`, `WAIT`, `END_SHIFT`. Add them to the action envelope only when their feature is enabled, so legacy envelopes keep the same size.

- `QUEUE` (Phase A, `queue_policy: kullman2021`): join the FCFS queue of the busy station the truck has just reached. Feasible only at the `arrived_at_busy_charger` epoch. Section 5.2.

- `WAIT` (Phase C): idle at the current node until the next `SYSTEM` request release or for `requests.wait_hours`, whichever comes first. Infeasible if no further release can occur, which prevents zero-time loops. Without this action a dynamic fleet cannot follow Voccia's wait-at-depot strategy.
- `END_SHIFT` (Phase C): retire at the depot. Feasible only at the depot with nothing on board. In dynamic modes the `DEPOT` action becomes "drive to the depot" (for reloading or depot charging) and does not retire. In legacy modes `DEPOT` keeps its retire semantics.
- `REST_BREAK` and `REST_DAILY` (Phase B): Section 6.4.

### 4.5 Information options

- `observation.in_flight_information: realized | nominal_eta` (Section 1.1).
- Each station's feature row gets `observed_busy_ports`, `observed_queue_length`, `observation_age_hours`, `observed_available_ports`, `expected_wait_given_memory` and `known_utilization_prior` (Section 5.3). The utilization `u` is a published model parameter, so exposing it is legitimate.
- Bump `SCHEMA_VERSION` to `joint-fleet-v7`. Add a test: with all new blocks disabled, the v6 columns of a v7 observation equal the v6 extraction on the golden trajectories.

**Gate 1** follows.

---

## 5. Phase A: public stations with outside traffic and outages

### 5.1 Configuration

```yaml
public_charging:
  enabled: true
  queue_model: dynamic_mmc         # dynamic_mmc (Kullman et al. 2021) | iid_stationary_wait (Keskin et al. 2019, WSC)
  queue_policy: kullman2021        # kullman2021: queue or leave on arrival, no reneging, must charge at the head | legacy: today's behaviour
  information: on_arrival          # on_arrival (Kullman F) | realtime_queue (Kullman Q, §7.4) | occupancy_indicator (Dastpak et al. 2024)
  memory: shared                   # shared: one fleet memory per station | per_truck (ablation)
  max_stations_between_stops: null # 1 reproduces the restriction below Kullman eq. (5); null = no limit
  exogenous_traffic:
    arrival_model: homogeneous_poisson   # Kullman 2021 | piecewise_poisson (Keskin, Laporte & Çatay 2019)
    utilization: 0.65              # u; Kullman regimes 0.40 / 0.65 / 0.90; must be < 1
    service_preset: kullman_fast   # see the table below; or give mean_service_hours explicitly
    mean_service_hours: null
    hourly_profile: null           # 24 nonnegative multipliers with mean 1 (piecewise only)
    initial_state: stationary      # stationary M/M/c distribution | empty
  outages:
    enabled: false
    port_availability: 0.733       # Rempel et al. 2023
    mean_repair_hours: 24.0        # MTTR; MTBF = MTTR * a / (1 - a)
    interrupt_sessions: false
depot_charging:
  enabled: true                    # Kullman's private depot: no queue, no outside traffic
  ports: null                      # null = num_trucks
  power_kw: 150.0
```

Service-time presets. The resolved value goes into the config and the artifact hash.

| Preset | Mean service | Source |
|---|---|---|
| `kullman_fast` | 26.62 min | Kullman §7.1 (Morrissey et al. 2016) |
| `kullman_moderate` | 128.78 min | Kullman §7.1 |
| `lookup_dcfast` | 30 min | The repository's `waiting_time_lookup.json` |
| `keskin_truck` | `0.55 · Q / (P · η)` for the outside trucks' battery `Q`, charger power `P` and efficiency `η` | Keskin et al. (2019, WSC) §2.2 rule applied to trucks. For 400 kWh at 150 kW and η = 0.9 this is 1.63 h |

Validation: `0 < utilization < 1`; a profile of length 24 with mean 1 within 1e-9; `0 < port_availability ≤ 1`; positive, finite times; exactly one of `service_preset` and `mean_service_hours`; `queue_policy: kullman2021` requires `queue_model: dynamic_mmc`; `iid_stationary_wait` requires `information: on_arrival`; `depot_charging` is rejected in legacy modes, where the depot retires the truck.

### 5.2 Queue semantics (`dynamic_mmc`; the specification reviewers check against)

1. Station `s` has `c_s` ports (data × `port_capacity_scale`), of which `u_s(t)` are up.
2. Outside vehicles arrive as a Poisson process with rate `λ_s(t) = u · c_s · μ · profile(hour(t))`, which is Kullman's `p_arrive = u·ψ_c·p_depart`. Each draws an Exp(`μ`) service time from `station_service[(s, k)]` when it arrives. Outside vehicles never balk or renege (M/M/c/∞).
3. One pooled FCFS queue per station holds outside vehicles and fleet trucks, ordered by `(arrival time, class priority, sequence)`. On exact ties a departure is processed before an arrival, and a fleet truck before an outside vehicle. These rules are fixed and tested.
4. **Arrival (`kullman2021`).** If an up port is free and nobody is queued, the truck gets a normal decision epoch: charge, or leave. Otherwise it gets a decision epoch with reason `arrived_at_busy_charger` and observes the queue length (Kullman eq. 8). Its feasible actions are `QUEUE` (a new `ActionKind`, appended) or any feasible routing action, which is a balk. If it queues, its FCFS position is its arrival time.
5. **Queued.** A queued truck has no decision epoch until a port is reserved for it, and it cannot leave ("must remain in queue until a charger becomes available"). Time in the queue is waiting time.
6. **Head of the queue.** When a port frees and the truck is first, the port is reserved and the truck gets an epoch in which only charge actions are feasible ("after which it must charge"; reason `COMMITTED_TO_CHARGE` masks the rest). If no charge level is feasible, because the battery is already above every target, the truck releases the port at once and becomes free to leave. This case is logged and tested.
7. **`queue_policy: legacy`** keeps today's behaviour for ablations: the truck joins the waitlist automatically on arrival and may leave when woken at the head.
8. **Initial state, `stationary`.** The number in system is drawn from the M/M/c stationary distribution `π_n`. Residual services are fresh exponential draws, which is exact by memorylessness. Requires `u < 1`.
9. **Outages.** Each port alternates up Exp(MTBF) and down Exp(MTTR), starting from its stationary state. A port that fails while busy finishes its session (`interrupt_sessions: false`). Whole-station closure (`set_charger_available`) keeps its current meaning.
10. **Depot.** `depot_charging.ports` ports, fleet only: no outside traffic, no outages (Kullman §2.1).

**`iid_stationary_wait`** (Keskin et al. 2019, WSC, generalized to `c` ports). On each visit, the truck's outside-traffic wait is drawn independently from the stationary M/M/c waiting distribution: `P(W = 0) = 1 − C(c, a)` and `W | W > 0 ~ Exp(cμ − λ)`. For `c = 1` this is Keskin's `1 − ρ` and Exp(`μ(1 − ρ)`). After the drawn wait, fleet trucks go through today's fleet FCFS gating. There is no queue memory; the observation carries only the drawn wait. This is the model the queue-aware day-ahead planner assumes, which makes it the natural ablation.

### 5.3 Information model

| Mode | What the policy sees for station `s` at time `t` | Source |
|---|---|---|
| `on_arrival` | Memory `(count observed, time observed)` from the last fleet visit to `s`, with sentinels if never visited (Kullman's `(c, ∅, ∅)`); the live count while a fleet truck is at `s`; the active truck's own queue position; the published prior `u`; and `expected_wait_given_memory`, the transient M/M/c expected wait given the memory and its age, which tends to the Erlang C long-run wait | Kullman §2.2, §4.3.2, Fig. 4 |
| `realtime_queue` | The number of vehicles at every station, in real time | Kullman §7.4 (filtration Q) |
| `occupancy_indicator` | A binary "all up ports busy" flag for every station, in real time; the queue length only on arrival | Dastpak et al. (2024) |

`memory: shared` (one memory per station, updated by any fleet truck's visit) is the fleet analogue of Kullman's single-vehicle memory and is the default; `per_truck` is an ablation. The mask never uses hidden queue state, because energy and time feasibility do not depend on it. Add a test asserting this.

`expected_wait_given_memory`: build the birth–death generator of the M/M/c queue truncated at `N`, with `N` chosen so that the stationary tail mass is below 1e-9. Propagate the observed count over the memory's age with `scipy.linalg.expm` (or uniformization), then convert the resulting distribution into the expected FCFS wait of a newcomer. Kullman used Andersen's (2020) `mc_math` package; reimplement it and test it against:
- the Erlang C wait as the age grows;
- the exact value at age 0: a newcomer behind `n ≥ c` vehicles waits for `n − c + 1` completions at rate `cμ`;
- Monte Carlo.

### 5.4 Implementation design (exactness is mandatory; this structure is recommended)

- New module `EVRoutingEnv/models/simulation/public_station_queue.py` with one `StationQueue` per station. It keeps a local clock and advances **lazily**: `advance(t)` processes its own outside arrivals, starts and departures up to `t`. Stations interact only through fleet trucks, so lazy advancement is exact, and it avoids tens of thousands of global events (432 Texas stations × 48 h).
- When a fleet truck queues, compute when a port will be reserved for it from the entries ahead. Every entry ahead is already known, later arrivals cannot overtake under FCFS, and under `kullman2021` nobody ahead can renege. Do this on a **private copy**, so the live state never advances past the global clock and observations stay non-anticipative. Push one `TRUCK_READY` event with a version token. Any change ahead (a port failing or recovering, or, under `legacy`, a fleet truck ahead leaving at the head) recomputes the time and invalidates the old event, which the stale-event guard then skips.
- `ChargingStation` becomes the fleet-facing façade, with `free_ports = up_ports − fleet_busy − outside_busy − reserved`. Keep `check_charger_gating` and `wake_waiting_trucks` as the only entry points from the env.
- Add an **eager reference simulator** (`debug_eager_public_queues: true`) that pushes every outside event onto the global heap. The lazy-vs-eager equivalence test (Section 8.2) is the main correctness check of this phase.
- Performance target: with 10 trucks, Texas long-haul, `u = 0.90` and all 432 stations, the median episode time under the greedy policy is at most 1.5× the same episode without outside traffic. Benchmark with `scripts/analysis/benchmark_stochastic_simulator.py` and record the numbers in the Gate A report.

### 5.5 Baselines under A

| Baseline | Required behaviour |
|---|---|
| Greedy navigator | Unchanged decision rule. At a busy station it always chooses `QUEUE` unless the mask forbids it |
| **Kullman myopic** | One-to-one with Kullman §5.2.3: the cheapest immediate action, with their added rules. Return to the depot once every customer is served and the depot is reachable. At an available station, charge to full. At a busy station, queue only if the observed queue is shorter than the expected queue. Prefer customers over other nodes whenever a customer is reachable. Visit at most two public stations between customers |
| Day-ahead ALNS / CP-SAT | Plan on the nominal network, ignoring queues (the current conservative baseline) |
| **Queue-aware day-ahead** | Add the expected wait per charging stop to the planner's charging cost: the Erlang C long-run wait at plan time (Keskin's average-wait planning; the shipped lookup), and `expected_wait_given_memory` when re-planning (Kullman's FRVCP-N) |
| Perfect information | Clairvoyant over outside arrivals (Kullman's PI bound), well defined thanks to the common random numbers. A bound, not a policy |
| Unsupported combinations | Raise `NotImplementedError` naming the block, with one test per refusal |

**Gate A** follows.

---

## 6. Phase B: hours-of-service

### 6.1 Configuration

```yaml
hours_of_service:
  enabled: true
  regime: us_fmcsa_property        # us_fmcsa_property | eu_561_2006 | bai2023_simplified
  rest_locations: anywhere         # anywhere (Goel 2009/2010) | stops_only (Bai 2023; Kahlert 2026)
  initial_state: fresh             # on duty at t=0 after a full daily rest
  charging_counts_as_break: true   # EU only; under US rules on-duty-not-driving always qualifies
  queue_wait_counts_as_break: false   # EU only
  handling_overhead_hours: 0.0833  # 5 min of work per charging session (Kahlert 2026), excluded from break credit
  eu_split_break: true             # 15 min then 30 min (Art. 7)
  eu_extended_daily_driving: false # 10 h at most twice a week (Art. 6)
  eu_reduced_daily_rest: false     # 9 h (Art. 8)
  time_guard: realized             # stops_only: realized | worst_case; see 6.5
```

Validation: `max_time ≤ 96` whenever HOS is enabled, with an error message explaining that weekly limits are not modelled. The regime must be a known string. `bai2023_simplified` requires `rest_locations: stops_only`.

### 6.2 Regime parameters

| Quantity | US (49 CFR 395.3) | EU (561/2006) | Bai 2023 |
|---|---|---|---|
| Driving before a break is required | 8 h cumulative driving | 4.5 h | 4.5 h |
| Qualifying break | 30 consecutive min not driving (off duty, sleeper or on-duty not driving) | 45 min uninterrupted, or 15 min then 30 min; only recovery, no work | stay of ≥ 45 min at a station |
| Driving per shift | 11 h | 9 h (10 h option) | 9 h per trip |
| Shift window | 14 h from coming on duty | daily rest must be completed within 24 h of the previous rest's end, so 13 h (15 h with reduced rest) | none |
| Daily reset | 10 consecutive h off duty | 11 h (9 h option) | none |

### 6.3 Activity classification

| Activity | US | EU |
|---|---|---|
| Driving | driving | driving |
| Charging session | on-duty not driving; counts toward the 30-min break and the 14 h window | break if `charging_counts_as_break`, minus `handling_overhead_hours`, which counts as other work |
| Waiting in a charger queue | on-duty not driving; counts toward the break | not a break unless `queue_wait_counts_as_break` |
| Loading or unloading service | on-duty not driving; counts toward the break | other work, not a break |
| Waiting for a time window | on-duty not driving; counts toward the break | not a break (default) |
| `REST_BREAK` / `REST_DAILY` | off duty | break / daily rest |

Only **contiguous** non-driving time counts. A 20-min charge, a 5-min drive and a 10-min charge give no break under either regime.

### 6.4 Rest actions and mid-leg insertion

- `REST_BREAK` lasts exactly the regime's break length (US 0.5 h, EU 0.75 h, or the remaining 0.5 h after a 15-min first part). `REST_DAILY` lasts exactly the daily reset (US 10 h, EU 11 h). Both are feasible at any node where the truck is `ready`.
- A truck resting at a station does **not** hold a port. If it was charging, it must leave the port first (the existing `must_leave_charger` rule). Parking is unlimited.
- `rest_locations: anywhere` (Goel): when a leg's realized driving time exceeds the remaining allowance, the simulator inserts the required break, or a daily rest, at the latest legal point inside the leg ("as late as possible"), possibly several times on long legs. Arrival time = departure + driving + inserted rests. Energy is unchanged. The traffic draw stays the one realized at departure. That draw was made for the uninterrupted leg, which is a documented approximation.
- `rest_locations: stops_only` (Bai): the mask rejects a leg whose driving would exceed the allowance (Section 6.5), so breaks must happen at stops. This is what makes charging-break synchronization a real decision.
- A truck may not start driving while it is outside its shift window or over its driving limit. If it is ready in that state, its only feasible actions are rests (and `END_SHIFT` at the depot). The mask must offer them, so that `no_feasible_action` can never come from HOS alone.

### 6.5 HOS under stochastic travel time (`stops_only`)

- `time_guard: worst_case`: the mask uses the traffic clip upper bound, `2.5 × nominal`. Compliance is guaranteed, but this is very conservative.
- `time_guard: realized` (default): the mask uses nominal time. If the realized driving still overruns, the simulator inserts an **unplanned roadside break** at the legal point, as in `anywhere` mode, and counts it in `info["hos_unplanned_breaks"]`. The schedule therefore stays legal, and the metric shows how often the plan was too optimistic. Report it.

### 6.6 Implementation

- New module `EVRoutingEnv/models/core/hours_of_service.py` with a frozen `DriverClock` dataclass and pure functions: `apply(clock, activity, hours) -> DriverClock`, `driving_allowance(clock) -> hours`, `required_rest(clock) -> (kind, hours)`, `insert_rests(clock, driving_hours) -> list[segments]`. It has no environment dependencies, so it can be unit tested exhaustively.
- Add a **second, independent implementation**, `hos_audit(activity_log, regime) -> list[violations]`, that recomputes compliance from `Truck.export_event_log()` with a sweep over the timeline. Every integration test calls it. The two implementations must agree. This is the main defence against bugs in the HOS state machine.
- Hook points: `_execute_navigation_action` (allowance check, mid-leg insertion), `_execute_charge_action` and charge completion (break credit), queue waiting, the service handlers, and new rest event handling in `_advance_to_next_decision`.
- Certification: witness replay must stay valid with HOS on. The witness generator inserts `REST_*` actions greedily, as late as possible, at the last stop before a violation. If none fits, the instance is rejected with a reason (no relaxation).
- New truck features: `drive_left_before_break`, `drive_left_in_shift`, `window_left_hours`, `nondriving_streak_hours`, `eu_split_first_part_taken`, `rest_required_now`.
- New `FeasibilityReason` members, appended, across all phases: `QUEUE_NOT_NEEDED`, `COMMITTED_TO_CHARGE` (A); `HOS_DRIVING_LIMIT`, `HOS_SHIFT_WINDOW`, `REST_NOT_REQUIRED` (B); `REQUEST_HIDDEN`, `REQUEST_LOST`, `NOT_ON_BOARD`, `MANIFEST_LOCKED`, `REQUESTS_ON_BOARD`, `NO_FUTURE_RELEASE`, `CONTAINER_ON_BOARD` (C). Each reason is produced by exactly one check and has a test that triggers it.

### 6.7 Baselines under B

| Baseline | Required behaviour |
|---|---|
| Greedy navigator | Rests when the mask requires it; prefers charging-as-break when charging anyway (simple rule, documented). |
| Day-ahead planners | HOS-unaware planning, HOS-compliant execution through the navigator. State this honestly in the paper. |
| Bai 2023 single-truck oracle | For `bai2023_simplified` on single-truck, fixed-route instances, an exact DP over (station, rest or not, charge level) reproduces their problem. GraphPPO and the navigator are compared with it. |

**Gate B** follows.

---

## 7. Phase C: dynamic requests (same-day freight and electric drayage)

### 7.1 Configuration

```yaml
requests:
  mode: static                     # static (legacy) | same_day | drayage
  arrival_process: euro_neurips_epochs   # euro_neurips_epochs | poisson
  epoch_hours: 1.0                 # EURO EPOCH_DURATION = 3600 s
  dispatch_margin_hours: 1.0       # EURO MARGIN_DISPATCH = 3600 s
  max_requests_per_epoch: 100      # EURO MAX_REQUESTS_PER_EPOCH; scale down for 10-truck fleets (e.g. 3-6)
  poisson_rate_per_hour: null      # for arrival_process: poisson
  initial_requests: 0              # known at t = 0; reported as the degree of dynamism (Larsen et al. 2002)
  request_cutoff_hours: null       # no releases after this time (default: max_time - dispatch margin)
  max_requests: 200                # action/observation envelope; exceeding it raises
  static_context_size: 230         # EURO's static instance, Voccia's known customer locations
  time_windows: {...}              # distribution of windows in the static context
  deadline_type: hard              # hard (Voccia et al. 2019) | soft (Ulmer 2017)
  allow_preemptive_returns: false  # false: Voccia; Li et al. 2022 | true: Ulmer, Thomas & Mattfeld 2019
  loading_service_hours: 0.2
  lost_request_penalty_hours: 24.0 # training reward only
  wait_hours: 0.5
drayage:
  terminals:                       # resolved to the nearest network node by lat/lon from vis_data/nodes.json
    - {name: "Port of Houston (Barbours Cut)", lat: 29.68, lon: -95.00}
  import_share: 0.5                # import = terminal -> customer, export = customer -> terminal
  origin_service_hours: 0.0333     # 2 min (Xiao §5.2.1)
  destination_service_hours: 0.0667  # 4 min (Xiao §5.2.1)
  delay_penalty_per_hour: null     # Xiao's soft destination deadline; null = record the delay only
  energy_model: network            # network | kancharla_ramadurai_2020 (Xiao §6.1)
```

Validation: `same_day` and `drayage` require `problem.mode: joint_fleet`, fleet assignment and a `payload_capacity` (drayage: exactly one container, demand 1). `depot_charging.enabled` must be set explicitly. Terminals must resolve to nodes within 5 km, else raise. The coordinates above are placeholders that the agent must check against the case-study node list. Propose terminals for Texas (Houston), Pacific Northwest (Seattle/Tacoma, Portland) and California (Los Angeles/Long Beach, Oakland) to the user before Gate C. Drayage always uses Xiao's soft destination deadline, whatever `deadline_type` says.

### 7.2 Same-day freight (EURO information process, Voccia objective)

1. At reset, draw a static context of `static_context_size` customers with windows, demands and service times from the configured distributions (stream `request_static_context`). This is EURO's static instance and Voccia's set of known customer locations.
2. At each epoch start `e·Δ`, sample up to `max_requests_per_epoch` requests. Customer, window, demand and service time are each drawn **independently with replacement** from the context, as in EURO. Keep a request only if a fresh full-battery truck at the depot at planning start `e·Δ + margin` has a certified witness depot → customer → depot, with at most one charger per segment, that meets the window, the horizon and HOS. EURO checks only travel time; energy, charging and HOS are the documented extension.
3. A kept request becomes `same_day` with `pickup = Stop(depot, earliest = release_time, latest = delivery.latest − nominal depot→customer time, loading_service_hours)`. A `SYSTEM request_release` event fires at the release time.
4. **Loading** follows formulation (1) of Li et al. (2022): a trip carries only requests released by the time it leaves the depot. Trucks load through slot actions at the depot, and leaving the depot fixes the manifest. Away from the depot a truck can only drive to deliveries it carries, and it cannot drop them.
5. **Depot returns.** Driving to the depot with requests on board is infeasible unless `allow_preemptive_returns: true` (reason `REQUESTS_ON_BOARD`). When returns are allowed, the truck may reload and keeps what it carries.
6. **Deadlines.**
   - `hard` (Voccia): the mask forbids loading, or driving to, a request whose window can no longer be met nominally. A delivery that turns out late is still carried out but counts as **not served**. An `OPEN` request becomes `LOST` at its latest pickup time (`SYSTEM request_expire`).
   - `soft` (Ulmer 2017): late delivery is allowed; the delay is recorded and penalized.
   - The legacy `fail_truck` behaviour stays available only in `static` mode.
7. The episode ends when every truck has done `END_SHIFT`, or at `max_time`. Termination reasons: `success` (every released request served on time and every truck retired), `requests_lost`, `requests_late`, plus the existing ones.
8. **Metrics:** served-on-time count and fraction first (Voccia's objective); then lost and late counts; then operating hours split into driving, charging, queueing, breaks and waiting.

### 7.3 Electric drayage (Xiao et al. 2025 task structure)

1. Requests are full-truckload O–D moves: `drayage_import` (terminal → customer) and `drayage_export` (customer → terminal), mixed by `import_share`. The trailer stays attached and there is no separate empty-container repositioning task (Xiao §3.1). The leg from one destination to the next origin is an empty leg.
2. **Routing rules**, enforced in the feasibility engine (Xiao's arc set `A`):
   - after the depot, the first stop is an origin;
   - between a task's origin and its destination only stations may be visited (reason `CONTAINER_ON_BOARD`);
   - from a destination the truck may go to another origin, a station or the depot;
   - one container at a time.
3. **Time windows** (Xiao §3.1). The earliest service time at the origin is hard: the truck waits. The destination deadline is soft: the delay is recorded and penalized at `delay_penalty_per_hour`.
4. Requests are released by the same `arrival_process` as same-day (EURO epochs or Poisson), with the feasibility filter of 7.2.2 applied to the witness depot → origin → destination → depot.
5. **Energy.** By default every leg, loaded or empty, uses the network table (Xiao's main model is load-independent). `energy_model: kancharla_ramadurai_2020` scales loaded legs by the ratio of eq. (62) power with and without the container mass (Table 10 parameters) at the leg's nominal speed. Apply it in both the feasibility engine and `realized_route_costs`, and test that empty legs are unchanged.
6. Out of scope, documented: battery swapping stations, the choice between normal and fast charging modes, and the two-container extension (Xiao §6.2).

### 7.4 Hindsight reference

Implement `hindsight_problem(env)`, analogous to EURO's `get_hindsight_problem` and to the deterministic release-date formulation (1) of Li et al. (2024) extended to several capacitated vehicles. It holds every request with its release time and windows, plus the realized outside-traffic and travel draws. The perfect-information search consumes it. It is a bound, not a policy.

### 7.5 Baselines under C

| Baseline | Required behaviour |
|---|---|
| Greedy dispatch (EURO baseline "greedy") | At the depot, load every feasible released request, nearest first; wait if there is none |
| Wait-at-depot threshold | A parametric policy motivated by Voccia's analytical result on waiting (not a reproduction of it): wait at the depot while fewer than `k` released requests are loadable and the earliest deadline allows it; `k` is tuned on validation seeds |
| Rolling-horizon ALNS | Re-plan at every release and every truck decision, with on-board requests fixed. This is the standard dynamic baseline; it also replaces the broken re-planning ALNS noted in the campaign plan |
| Day-ahead planners | Not applicable to unrevealed requests. They raise `NotImplementedError` in `same_day` and `drayage` modes; test this |
| Perfect information | Hindsight reference (7.4) |

**Gate C** follows.

---

## 8. Tests

The edge-case list is Section 9. This section gives the test structure.

### 8.1 Layout and markers

- `tests/unit/test_public_station_queue.py`, `test_hours_of_service.py`, `test_hos_audit.py`, `test_request_registry.py`, `test_request_process.py`, `test_system_events.py`.
- `tests/integration/test_public_charging_env.py`, `test_hos_env.py`, `test_dynamic_requests_env.py`, `test_combined_extensions.py`, `test_golden_traces.py`.
- Register a `statistical` pytest marker in `pyproject.toml` for long Monte-Carlo checks. They run in CI nightly and before each gate, not on every commit.

### 8.2 Analytic oracles (A)

| Test | Oracle | Acceptance |
|---|---|---|
| Outside-only station, homogeneous | `waiting_time_lookup.json` (Erlang C) for c ∈ {1, 2, 4, 10, 20}, u ∈ {0.40, 0.65, 0.90}; Erlang C computed directly for other `c` and service times | The batch-means 99 % CI on `W_q` contains the oracle value; ≥ 10⁵ customers per cell, fixed seeds |
| Little's law | `L_q = λ W_q` | Relative error < 2 % |
| Stationary start | M/M/c `π_n` | Chi-square goodness of fit, p > 0.01 over 10⁴ resets |
| Piecewise Poisson | Hourly counts | Mean and variance per hour match `∫λ` within a 99 % CI |
| Port outages | Availability `a` | Long-run up fraction within a 99 % CI |
| Lazy vs eager | Eager reference simulator | **Identical** event logs and fleet trajectories for 50 seeds, three configs, greedy and random policies |
| FCFS with fleet | Hand-built 2-port example: 3 outside vehicles and 2 trucks with scripted times | Exact start times match a hand-computed table |
| **Single-truck exactness (Kullman)** | With one fleet truck that always queues, its realized wait equals the clairvoyant wait function `w(t_arrival)` computed from the eager, outside-only realization of the same seed (Kullman Fig. 4, right) | Equal to 1e-9 on 200 seeds × 5 stations, for `u` ∈ {0.40, 0.65, 0.90} |
| i.i.d. stationary waits (Keskin) | `c = 1`: `P(W = 0) = 1 − ρ` and `W \| W > 0 ~ Exp(μ(1 − ρ))`; for `c > 1`, the Erlang C generalization | `P(W = 0)` within a 99 % CI; KS test on the positive part, p > 0.01 |
| Transient expected wait | `expected_wait_given_memory` vs Erlang C as the age grows, the exact age-0 value, and Monte Carlo from a fixed initial count | Within 1e-6 (analytic cases) and within a 99 % CI (Monte Carlo) |
| Queue commitment | Scripted `kullman2021` episode: the truck queues, the ports free in a known order | No routing action is feasible from `QUEUE` until the truck has charged; the head epoch offers charge actions only |

### 8.3 Oracles (B)

- Table-driven `DriverClock` tests, at least 25 hand-computed activity sequences per regime, covering every row of Section 9.2.
- Bai 2023 cross-check: on random sequences, the `bai2023_simplified` clock equals Bai's recursion (their equation 4 with constraints 7–9) step by step.
- `hos_audit` agrees with `DriverClock` on 10⁴ random legal and illegal sequences. Both must flag the same violations.

### 8.4 Oracles (C)

- EURO filter: port the EURO `_next_observation` feasibility filter (MIT-licensed quickstart; keep attribution) into the test as an oracle. Feed both implementations the same drawn indices on a toy duration matrix, with energy and HOS disabled, and assert the same kept set, release times and windows.
- Poisson arrivals: count and gap distribution (KS test).
- Hindsight problem: contains exactly the released requests, with correct release times.
- Li et al. (2022) formulation (1) as an oracle on tiny instances (one truck, energy and HOS off, ≤ 6 requests): solve the OP with release dates by enumeration and check that the perfect-information search on the same realization serves the same number of requests.

### 8.5 Invariants checked after every step of random-policy rollouts

200 seeds × {each feature alone, all features} × {Texas long-haul, California, Seattle}:

1. The clock never decreases. Every event time ≥ the clock when it was pushed.
2. Battery stays in `[0, capacity]`. Energy drawn equals the realized leg energy.
3. Per station, `fleet_busy + outside_busy + reserved ≤ up_ports`. A truck charges at most one session at a time.
4. FCFS: no queue entry starts before an entry that arrived earlier at the same station, unless that entry left.
5. Requests: one owner at a time; pickup before delivery by the same truck; served at most once; `HIDDEN` requests never appear in features, masks or baseline inputs.
6. Payload: `0 ≤ on-board ≤ capacity`. Drayage: at most one container.
7. HOS: `hos_audit` returns no violations (in `stops_only + realized`, unplanned breaks are allowed but counted).
8. Non-anticipativity: re-run the episode with every *future* request, outside arrival and port change perturbed (all draws after time `t` resampled). Observations and masks up to `t` must be bit-identical.
9. Common random numbers (Section 4.2) and determinism: same seed and same actions give the same trajectory, and `reset` clears every new structure.
10. The mask agrees with execution: every masked-feasible action executes without `invalid_action:*`; every masked-infeasible action is rejected with the same reason code.
11. Under `queue_policy: kullman2021`, a truck that chose `QUEUE` charges at that station before any routing action, unless A5 applies.

### 8.6 Scale and stress

- 128 trucks sharing one public station with u = 0.90, plus outages: no deadlock, invariants hold.
- 432 Texas stations, 48 h, u = 0.90: the performance target in 5.4.
- `max_requests` reached exactly; one more request raises.
- Mixed: all three components on Texas long-haul with 10 trucks, EU and US regimes, 50 seeds, greedy and random policies.

### 8.7 GraphPPO integration smoke tests

- The canonical graph builds with variable numbers of revealed requests (0, 1, `max_requests`). Hidden rows are padding, and the action head never scores them.
- A 20k-step PPO run on a 3-truck dynamic config completes, and every rollout satisfies invariant 10.

---

## 9. Edge cases

Each row needs at least one test. The reviewer checklist (Section 10) asks for the test name of each.

### 9.1 Public stations (A)

| # | Case | Expected |
|---|---|---|
| A1 | Truck arrives exactly when an outside vehicle departs | Departures before arrivals at equal times (5.2.3): the truck sees the freed port |
| A2 | Truck and outside vehicle arrive at exactly the same time | Fleet first (fixed rule), tested |
| A3 | All ports down when a truck arrives | `arrived_at_busy_charger` epoch; `QUEUE` or balk; `observed_available_ports = 0`; when a port recovers, FCFS resumes |
| A4 | Port fails while a fleet truck charges | The session finishes (`interrupt_sessions: false`); the port goes down when released |
| A5 | Truck at the head has no feasible charge level | The port is released at once, the next entry starts at the same instant, and the truck is free to leave; logged |
| A6 | `kullman2021`: a queued truck | No decision epoch until its port is reserved; every routing action masked (`COMMITTED_TO_CHARGE`) at the head |
| A7 | `legacy`: a truck at the head leaves instead of charging | The reservation passes to the next entry (outside or fleet) at the same instant; no port idles; times behind are recomputed |
| A8 | Arrival with a free up port and an empty queue | No `QUEUE` action (reason `QUEUE_NOT_NEEDED`); charge or leave |
| A9 | u → 0 (λ tiny) | Matches the fleet-only legacy behaviour on the same seed (no outside arrivals within the horizon), up to the arrival epoch added by `kullman2021` |
| A10 | `utilization: 0.99`, homogeneous | Allowed; the statistical test is skipped above 0.95, but the invariants hold |
| A11 | `utilization: 1.0` | `ValueError` |
| A12 | Station closed via `set_charger_available` with outside vehicles queued | Outside vehicles leave with the closure, as fleet trucks are released; documented and tested |
| A13 | Horizon reached while outside vehicles are still queued | No effect on termination; queue state is not reported past `max_time` |
| A14 | `on_arrival`: station never visited | Only the prior `u` is visible; observed columns at their sentinels; `expected_wait_given_memory` equals the Erlang C long-run wait |
| A15 | `memory: shared`: truck B reads a station that truck A visited | B sees A's (count, time) memory; under `per_truck`, B sees sentinels |
| A16 | `realtime_queue` | The feature equals the true number in system at every station at every epoch |
| A17 | Depot charging with `ports < num_trucks` | Fleet FCFS at the depot, no outside traffic |
| A18 | Station with `c_s` above the lookup's largest capacity (50) | The oracle computes Erlang C directly instead of reading the table |
| A19 | `iid_stationary_wait` with `information: realtime_queue` | `ValueError` |

### 9.2 Hours-of-service (B)

| # | Case | Expected |
|---|---|---|
| B1 | Driving equals the limit exactly (4.5 h EU / 8 h US) | Legal (≤ with 1e-9 tolerance); the next driving second requires a break |
| B2 | EU split 15 then 30 | Qualifies |
| B3 | EU split 30 then 15 | Does **not** qualify (Art. 7 order) |
| B4 | EU 44-min charge + 1-min queue wait | Not a break by default; a break if `queue_wait_counts_as_break` |
| B5 | US 30-min mix of queue wait, charge and unloading, contiguous | Qualifies |
| B6 | US 14 h window expires while waiting in a queue | Truck may charge or unload but cannot drive until after a 10 h reset; the mask offers `REST_DAILY` |
| B7 | Single leg longer than a whole shift (`anywhere`) | Several breaks and a daily rest inserted inside the leg; arrival time is correct to 1e-9 |
| B8 | Same leg in `stops_only` | Masked infeasible (`HOS_DRIVING_LIMIT`, new reason); instance generation must still find a witness, or reject |
| B9 | Overrun under `stops_only + realized` | One unplanned roadside break, counted; the audit is clean |
| B10 | Break straddles `max_time` | Truncation semantics unchanged; the break is not counted as completed |
| B11 | Charging-as-break with `handling_overhead_hours` larger than the session | No break credit; no negative time |
| B12 | Rest action at a charger port right after charging | Truck releases the port first; the next queued entry starts at the same instant |
| B13 | Truck ready, over its limits, at the depot with `END_SHIFT` available | `END_SHIFT` and rests are both feasible; no `no_feasible_action` |
| B14 | Time-window wait that completes a break (US) | Break credited; tested against the audit |
| B15 | `max_time: 120` with HOS on | `ValueError` (weekly limits not modelled) |

### 9.3 Dynamic requests (C)

| # | Case | Expected |
|---|---|---|
| C1 | Release at the same instant a truck becomes ready at the depot | Visible (SYSTEM before ready, 4.1) |
| C2 | Two requests at the same customer node, overlapping windows | Distinct slots and ids; both servable; node multimap correct |
| C3 | Request becomes `LOST` while a truck is driving to its pickup | The truck arrives to a lost request; the claim is released; the truck becomes ready with no service and no crash |
| C4 | Truck loads two requests and the second makes it overweight | Second load masked (`PAYLOAD_EXCEEDED`) |
| C5 | `deadline_type: hard`, delivery turns out late | Delivered; counted as not served (`requests_late`); the truck continues |
| C6 | `deadline_type: soft`, delivery turns out late | Served late; delay recorded and penalized |
| C7 | `static` mode, late arrival | Legacy `fail_truck` behaviour, bit-identical to the golden traces |
| C8 | `WAIT` when no future release is possible | Infeasible |
| C9 | `WAIT` with a release 1e-12 h ahead | Wakes at that release; no zero-length loop |
| C10 | `END_SHIFT` with requests on board | Infeasible |
| C11 | Drive to the depot with requests on board, `allow_preemptive_returns: false` | Infeasible (`REQUESTS_ON_BOARD`); with `true`, feasible, the truck reloads and keeps its cargo |
| C12 | Every truck has retired and a request is released later | It becomes `LOST` at its expiry; the episode ends at the last event or `max_time`; reason `requests_lost` |
| C13 | Epoch sampling draws zero feasible requests | No events; nothing breaks |
| C14 | `max_requests` exceeded during generation | `ValueError` at reset, with a message |
| C15 | Drayage: loaded truck tries to load a second container | Masked |
| C16 | Drayage: loaded truck tries a customer, the depot or another origin | Masked (`CONTAINER_ON_BOARD`); only stations and its own destination are feasible |
| C17 | Drayage: first stop after leaving the depot | Must be an origin; destinations and stations masked from the depot |
| C18 | Drayage: early arrival at an origin / late arrival at a destination | Waits until the earliest service time / served late with the delay recorded (soft) |
| C19 | Drayage: `energy_model: kancharla_ramadurai_2020` | Loaded legs scaled; empty legs identical to the network table |
| C20 | Drayage: terminal outside the map | `ValueError` at init |
| C21 | Dynamic mode plus `problem.assignment: preassigned` | `ValueError` |
| C22 | Legacy static mode through the new registry | Golden traces identical |

### 9.4 Combinations

| # | Case | Expected |
|---|---|---|
| X1 | A + B: break taken while queued at a public station | EU: queue wait does not count by default; the charge that follows counts; the audit agrees |
| X2 | A + C: depot charging and loading at the same visit | Loading service and charging overlap only if `depot_charging.parallel_with_loading` (default false, sequential) |
| X3 | B + C: request feasibility filter with HOS | The witness includes rests; a request infeasible only because of HOS is dropped at release, and the drop is counted in `info` |
| X4 | A + B + C, 10 trucks, Texas long-haul, US and EU | Invariants hold; performance within budget |

---

## 10. Review gates

Each gate has the same procedure:

1. The implementing agent finishes the phase, runs the full suite plus the `statistical` marker, and writes a short gate report in `docs/campaigns/` covering: what changed, test counts, oracle results with numbers, benchmark timings, and the "To confirm" rows resolved or still open.
2. The agent **stops and asks the user** to run `/code-review ultra` on the branch. Only the user can trigger it, and it is billed. A local `/code-review xhigh` pass is a cheaper intermediate option.
3. The agent fixes every confirmed finding, adds a regression test for each, and re-runs the suite.
4. Then the next phase starts.

Reviewer checklist (paste it into the review request):

| Gate | Questions the review must answer |
|---|---|
| 0 | Is every trace of the quota gone (`grep -rn "quota" EVRoutingEnv tests`)? Do the golden traces cover all five configs, and were they recorded **before** any Phase 1 code? |
| 1 | Can a system event touch truck state? Are same-time priorities as in 4.1? Is any new randomness drawn from a sequential stream that a fleet action can shift? Does any illegal registry transition pass silently? Do the v6 columns match exactly? |
| A | Does the queue reduce exactly to M/M/c with no fleet trucks (oracle numbers in the report)? Does a single truck's realized wait equal Kullman's clairvoyant wait function? Is lazy exactly equal to eager? Can the live station state ever advance past the global clock? Can hidden queue state reach features, masks or baselines under `on_arrival`? Under `kullman2021`, is the queue-or-leave epoch on arrival present, and is the commitment (no reneging, must charge at the head) enforced? Does the Kullman myopic baseline follow §5.2.3 rule by rule? |
| B | Do `DriverClock` and `hos_audit` agree everywhere? Is every row of 6.3 implemented as written? Is contiguity enforced for break credit? Can HOS alone ever cause `no_feasible_action`? Is the mid-leg insertion arrival time exact? |
| C | Does the request process match EURO line by line, apart from the documented extensions? Are unrevealed requests invisible everywhere (features, masks, baseline inputs, `info`)? Are manifests fixed at depot departure, with preemptive returns only when enabled? Are hard-deadline late deliveries counted as unserved? Do drayage routes obey Xiao's arc rules (origin first, only stations between origin and destination, one container)? Can `WAIT` loop at zero time? Is the hindsight problem complete? |
| Final | Does the formal model document (Section 11) match the code? Is every edge case in Section 9 mapped to a named test? |

---

## 11. Formal model document

Before the final gate, write `docs/models/dynamic_stochastic_etruck_model.md` as a sequential decision model (Ulmer et al. 2020 route-based MDP, Powell notation). It should cover:

- **Decision epochs:** a truck becomes ready.
- **State:** trucks (node, SoC, payload or manifest, `DriverClock`, activity), requests by status, station information per the information model, clock.
- **Actions:** the feasible set from the engine.
- **Exogenous information:** travel and energy draws, outside arrivals and services, port changes, request releases.
- **Transition and cost:** as implemented, with the lexicographic evaluation objective.
- **Source mapping:** a table tying each element to its source (Section 2) and to the code module that implements it.

This document becomes the paper's problem-formulation section.

---

## 12. Order of work and estimates

| Step | Content | Depends on | Rough size |
|---|---|---|---|
| 0 | Quota removal, golden traces | — | small |
| 1 | System events, streams, request registry, new actions, information options, schema v7 | 0 | large |
| A | Public station queues, outages, depot charging, queue-aware baseline | 1 | large |
| B | HOS clock, audit, rests, mid-leg insertion, certification with rests | 1 (independent of A) | large |
| C | Same-day, drayage, hindsight, rolling ALNS | 1; B for the HOS-aware filter | large |
| Final | Combined tests, formal model document, benchmark configs | A, B, C | medium |

Benchmark configs to add at the end: `config_dynamic_texas_sameday.yaml`, `config_dynamic_texas_drayage.yaml` (Houston), `config_dynamic_pnw_drayage.yaml` (Seattle/Tacoma), and the corresponding `*_hos_us.yaml` and `*_public_u{040,065,090}.yaml` variants (Kullman's three demand levels). The regime grid is frozen before any training, as the day-ahead campaign plan already requires.

---

## 13. Open items for the user

Resolved by the supplied papers: Kullman's queue, balking and commitment rules, information model and calibration (Section 2.1); Xiao's time windows, routing rules, empty legs and energy model (Section 2.3); Keskin's stationary-wait model; Voccia's objective and structure via the survey.

Still open:

1. **HOS regime per case study.** Proposed: US rules for Texas, Pacific Northwest and California; EU rules for `de_nl_be`.
2. **Outside-traffic service time.** Kullman's 26.62 min is for cars at French fast chargers in 2016. For trucks, the `keskin_truck` preset (0.55·Q/(P·η), 1.63 h at 400 kWh and 150 kW) is more plausible. Proposed: `keskin_truck` for the headline runs and `kullman_fast` as the replication check, both over `u ∈ {0.40, 0.65, 0.90}`.
3. **`queue_policy` default.** Proposed: `kullman2021` (queue or leave on arrival, then committed) whenever `public_charging` is enabled; `legacy` only for ablations.
4. **Legacy implicit loading (1.1.1).** Keep it as the default for comparability with the paper's earlier results, or switch the static joint benchmark to `loading: at_depot`?
5. **Drayage terminals.** Approve the terminal list before Gate C.
6. **Optional primary texts.** Voccia et al. (2019), to confirm the arrival-time distribution; Jia et al. (2022, TR-B), for a dynamic drayage arrival model; EUR-Lex text of Regulation 561/2006 Art. 6 and 8.
