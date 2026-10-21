# Dynamic and stochastic electric-truck fleet routing: the sequential decision model

This document states the model the simulator implements, as a route-based Markov decision process (Ulmer, Goodson, Mattfeld & Thomas 2020) in the notation of Powell's sequential decision framework. It is the basis of the paper's problem formulation. Every element names its source and the code that implements it. Where the repository departs from a source, the departure is stated and tested.

The model combines three published components on top of the event-driven electric-truck simulator:

- **A.** public DC-fast stations shared with outside traffic (Kullman, Goodson & Mendoza 2021);
- **B.** driver hours-of-service rules (49 CFR 395.3; Regulation (EC) 561/2006; Bai et al. 2023);
- **C.** dynamic requests: same-day freight (EURO Meets NeurIPS 2022; Voccia, Campbell & Thomas 2019; Li, Archetti & Ljubić 2024) and electric drayage (Xiao et al. 2025).

Each component is off unless its configuration block is present. With every block absent, the simulator reproduces the frozen legacy trajectories exactly (`tests/integration/test_golden_traces.py`).

## 1. Decision epochs

A decision epoch $k$ occurs at time $t_k$ when a truck becomes ready. Only one truck acts per epoch; trucks that become ready at the same instant act in increasing truck id. A truck becomes ready when it:

- starts the episode or arrives at a node;
- finishes charging, a service, a rest, a wait or a loading;
- reaches the head of a public-station queue, at which point a port has been reserved for it (Kullman §2.2: the three epoch types are arrival, front of the queue and end of charging).

Exogenous events (request reveals and releases, request expiries) are processed between epochs. They never produce an epoch for the fleet by themselves.

## 2. State

$S_k = (t_k, \kappa_k, \{s^v_k\}_{v \in V}, R_k, I_k)$ where:

| Component | Content | Code |
|---|---|---|
| Clock $t_k$ | Simulation time | `EventDrivenTruckEnv.global_clock` |
| Active truck $\kappa_k$ | The truck that decides | `active_truck_id` |
| Truck $v$ | Node, state of charge, payload or manifest, activity (routing, charging, queued, resting, waiting, loading, unloading, complete, failed), next committed event, driver clock | `models/core/truck.py`, `env.truck_states` |
| Driver clock $c^v_k$ | Driving since the last qualifying break, driving in the shift, shift start, contiguous break credit, contiguous off-duty time, EU split flag, extended and reduced counters | `models/core/hours_of_service.py::DriverClock` |
| Requests $R_k$ | Every revealed request with status, owner, stops, windows, demand, release time; unrevealed requests are not part of the state | `models/core/requests.py::RequestRegistry` |
| Station information $I_k$ | Per public station, under the configured information model (§3.2) | `models/simulation/public_charging.py` |

The observation is the versioned feature snapshot `joint-fleet-v7` (`state/features.py`). Its v6 columns are unchanged when the new blocks are disabled.

**In-flight information.** With `observation.in_flight_information: realized` (the legacy default), a truck that is mid-leg exposes its realized arrival time and energy. With `nominal_eta`, it exposes departure plus nominal time, nominal energy and nominal service time, as a telematics ETA would. All new components are tested for non-anticipativity under `nominal_eta`.

## 3. Exogenous information $W_{k+1}$

### 3.1 Travel, energy and service

The time and energy of a leg are realized at departure: travel time is Gaussian, clipped to $[0.85, 2.5]$ times nominal, and energy is correlated with the traffic multiplier. Service times are stochastic around the base value. Under hours of service, a leg's realized driving time is split by rests (§4.3). These draws are keyed by event identity, so policies see common random numbers (`models/simulation/scenario.py`).

### 3.2 Public stations (component A)

Station $s$ has $c_s$ ports. Outside vehicles arrive as a Poisson process with rate $\lambda_s(t) = u\, c_s\, \mu\, p(\lfloor t \rfloor \bmod 24)$ and need Exp($\mu$) service. They form a pooled FCFS M/M/$c$/∞ queue in which outside vehicles never balk or renege (Kullman §2.1 and §7.1). Here $p \equiv 1$ in the homogeneous model; with `piecewise_poisson`, $p$ is a 24-hour profile with mean 1, sampled exactly by thinning (Lewis & Shedler 1979). The service-time presets are $26.62$ min (`kullman_fast`), $128.78$ min (`kullman_moderate`), $30$ min (`lookup_dcfast`), and $0.55\,Q/(P\eta)$ (`keskin_truck`, after Keskin, Akhavan-Tabatabaei & Çatay 2019). The initial state is stationary: the count is drawn exactly from $\pi_n$, and residual services are fresh exponentials (memorylessness).

When outages are enabled, each port alternates between up periods Exp(MTBF) and down periods Exp(MTTR), with MTBF $=$ MTTR $\cdot a/(1-a)$ and default $a = 0.733$ (Rempel et al. 2023). A port that fails while busy finishes its session.

Fleet trucks join the same queue. Same-time ties are resolved in a fixed order: departures, then port changes, then arrivals; a fleet truck before an outside vehicle.

**Information models.**

| Mode | The truck sees, for station $s$ | Source |
|---|---|---|
| `on_arrival` | The memory (number in system, time observed) from the last fleet visit, or sentinels; the live count while a fleet truck is present; its own queue position; the prior $u$; and $E[W \mid \text{memory}]$, the transient M/M/$c$ expected wait propagated over the memory's age | Kullman §2.2, §4.3.2, Fig. 4 (filtration F) |
| `realtime_queue` | The number in system at every station | Kullman §7.4 (filtration Q) |
| `occupancy_indicator` | A real-time "all up ports busy" flag at every station; queue lengths only on arrival | Dastpak, Errico, Jabali & Malucelli (2024) |

`memory: shared` gives one memory per station for the whole fleet (the default); `per_truck` is an ablation. $E[W \mid n, \tau]$ uses the symmetrised tridiagonal generator of the birth–death chain, truncated where the stationary tail mass is below $10^{-9}$. It is exact at age 0, equals the Erlang C wait as the age grows, and is tested against `scipy.linalg.expm` and against Monte Carlo.

The alternative queue model `iid_stationary_wait` draws, on each visit, $W = 0$ with probability $1 - C(c, a)$ and otherwise $W \sim$ Exp$(c\mu - \lambda)$. This is the c-port generalisation of Keskin et al. (2019, WSC). Fleet FCFS applies after the drawn wait.

### 3.3 Requests (component C)

**Static context.** At reset, a static context of customers with windows, demands and service times is drawn. This is EURO's static instance and Voccia's set of known locations.

**EURO epochs** (`euro_neurips_epochs`). The epochs run from $\lfloor(\min \text{open} - m)/\Delta\rfloor$ to $\lfloor(\max \text{open} - m)/\Delta\rfloor$, where $m$ is the dispatch margin and $\Delta$ the epoch length. At each epoch, up to $N$ candidates draw a customer, a window, a demand and a service time, each independently and with replacement from the context. A candidate is kept if

$$\max(e\Delta + m + d_{0c},\; o) \le \ell \quad\text{and}\quad \max(e\Delta + m + d_{0c},\; o) + \sigma + d_{c0} \le T,$$

where $e$ is the epoch index, $d_{0c}$ and $d_{c0}$ the nominal travel times from the depot to customer $c$ and back, $[o, \ell]$ the drawn window, $\sigma$ the drawn service time and $T$ the depot closing time. This is the filter of `VRPEnvironment._next_observation` in `ortec/euro-neurips-vrp-2022-quickstart`. A kept request is revealed at $e\Delta$ and released at $e\Delta + m$.

**Poisson arrivals** (`poisson`). Arrival times follow a Poisson process up to the cutoff; each arrival draws its attributes the same way.

**Extended filter (documented departure).** A candidate is also kept only if a fresh, full-battery truck at the depot has a witness route with:

- at most one station per segment;
- worst-case energy;
- the hours-of-service rules;
- the horizon.

Drops caused only by hours of service are counted.

**Same-day requests.** A same-day request picks up at the depot, with pickup window [release, $\ell - d_{0c}$] and the configured loading time, then delivers to $c$ within its window.

**Drayage requests.** A drayage request moves one container: import (terminal → customer) or export (customer → terminal). It has a hard earliest pickup at the origin and a soft deadline at the destination; origin and destination processing take 2 and 4 min (Xiao §3.1, §5.2.1).

## 4. Actions and the feasible set $\mathcal{X}(S_k)$

The action of truck $\kappa_k$ is one index of the fixed envelope:

| Slots | Meaning |
|---|---|
| Chargers | Drive to public station $s$ |
| Request slots $0..R_{\max}-1$ | Slot $r$ is request $r$ (ids in reveal order). Open with a pickup: drive to the pickup, or load at once if already there. On board this truck: drive to its destination. Static mode: drive to customer $r$ (legacy) |
| Depot | Legacy: return and retire. Request modes: drive to the depot |
| Charge levels | Charge to the target state of charge on the nonlinear CC-CV curve |
| `QUEUE` | Join the queue of a busy public station (Kullman) |
| `REST_BREAK`, `REST_DAILY` | Take the regime's break or daily rest |
| `WAIT` | Idle until the next request information or `wait_hours` |
| `END_SHIFT` | Retire at the depot |

The appended kinds exist only when their component is enabled. The feasible set comes from a single engine, `state/feasibility.py`. The policy mask, `step`, the navigator, the baselines and the certificate replay all call it. Every rejection carries a reason code.

### 4.1 Public-station rules (Kullman §2.1)

- **Arrival at a busy station** (no free up port, or a non-empty queue): the epoch offers `QUEUE` or any routing action (a balk). Charging is masked (`PORTS_BUSY`).
- **Queued truck**: it has no epoch until a port is reserved for it, and it may not leave.
- **Head of the queue**: only charge actions are feasible (`COMMITTED_TO_CHARGE`). A truck with no feasible charge level releases the port at once and is free to leave.
- **`queue_policy: legacy`** keeps the previous automatic waitlist for ablations.
- **Optional:** `max_stations_between_stops` limits consecutive station visits (`STATION_HOP_LIMIT`).

The mask never reads hidden queue state.

### 4.2 Request rules

- **Payload** is checked at pickup: the on-board demand plus the new demand must not exceed capacity (`PAYLOAD_EXCEEDED`).
- **Hard deadlines**: loading, or driving to a pickup, whose window can no longer be met nominally is masked (`TIME_WINDOW_EXPIRED`). An open request is lost at its latest pickup time. A late delivery is still carried out but counts as not served (Voccia).
- **Soft deadlines** (Ulmer 2017, and all drayage): late delivery is allowed; the delay is recorded and penalised.
- **Manifests are fixed at departure** (Li et al. 2022, formulation (1)). Away from the depot, a truck may only drive to deliveries it carries. Returning with cargo needs `allow_preemptive_returns` (`REQUESTS_ON_BOARD`, `MANIFEST_LOCKED`).
- **Drayage arc rules** (Xiao §3.1):
  - the first stop after the depot is an origin (`DRAYAGE_ORIGIN_FIRST`);
  - between an origin and its destination, only stations are allowed (`CONTAINER_ON_BOARD`);
  - one container at a time.
- **`WAIT`** is feasible only if a later release can still occur according to known information: the epoch schedule, the Poisson cutoff, or revealed but unreleased requests (`NO_FUTURE_RELEASE`).
- **`END_SHIFT`** is feasible only at the depot with nothing on board.

### 4.3 Hours-of-service rules

| Quantity | US (49 CFR 395.3) | EU (561/2006) | Bai et al. (2023) |
|---|---|---|---|
| Driving before a break | 8 h | 4.5 h | 4.5 h |
| Qualifying break | 30 consecutive min not driving | 45 min, or 15 then 30 (Art. 7) | stay ≥ 45 min |
| Driving per shift | 11 h | 9 h (10 h at most twice) | 9 h per trip |
| Window | 14 h from coming on duty | 13 h (15 h before a reduced rest), from the end of the previous rest | none |
| Daily reset | 10 h off duty | 11 h (9 h reduced, at most 3 times; Art. 8(2): reduced if under 11 h of it falls within 24 h) | none |

**Activity classification.**

- Under US rules, charging, queueing, service and window waits are on duty but not driving, and qualify for the 30-minute break.
- Under EU rules, charging counts as a break if `charging_counts_as_break`, minus a handling overhead that counts as work. Queue waiting counts only if `queue_wait_counts_as_break`. Service and window waits are work.
- Only contiguous time earns credit.

**Placement of rests.**

- `rest_locations: anywhere` (Goel 2009/2010): a leg that exceeds the allowance has the required rests inserted inside it, as late as possible. The arrival time includes them.
- `stops_only` (Bai; Kahlert, Wang & Mårtensson 2026): the mask rejects a leg whose nominal driving (or 2.5× nominal with `time_guard: worst_case`) exceeds the allowance (`HOS_DRIVING_LIMIT`, `HOS_SHIFT_WINDOW`). A realized overrun gets one counted roadside break.

A truck over its limits can still rest, charge, or end its shift at the depot, so hours of service alone never empty the feasible set.

## 5. Transition $S_{k+1} = S^M(S_k, x_k, W_{k+1})$

The action is executed and then events are processed in time order. Same-time order is: system events, arrivals, completions, then other ready events. Processing continues until a truck is ready.

**Public stations** advance lazily. Each station processes its own events up to the global clock and never beyond. When a fleet truck queues, the time a port will be reserved for it is computed on a private copy of the station. This is exact under FCFS, because later arrivals cannot overtake and no one ahead reneges; it is recomputed when a fleet truck ahead decides. An eager reference simulator that pushes every outside event onto the global heap produces identical trajectories.

**Driver clocks** are updated as each activity ends. Rests inside a leg are committed on arrival.

**Requests** move through `HIDDEN → OPEN → CLAIMED_PICKUP → LOADING → ON_BOARD → CLAIMED_DELIVERY → IN_SERVICE → SERVED | SERVED_LATE`, or `→ LOST`. Every transition is validated, and an illegal one raises an error.

**Termination.** The episode ends when every truck has retired or failed and no relevant event remains, or at the horizon. Requests released after the whole fleet has retired still expire.

## 6. Objective and reward

**Evaluation is lexicographic.** First, requests served on time (Voccia's objective; in static mode, all customers served and every truck home, `success`). Then lost and late requests. Then operating hours, split into travel, charging, queueing, rest, window waits, request waits and service (`evaluation/metrics.py`; `info["request_metrics"]`).

**The training reward** is shaped and separate from evaluation: minus elapsed operating hours, a delivery bonus on on-time service, minus `lost_request_penalty_hours` per lost request, and minus the delay of a soft late delivery (or Xiao's per-hour penalty for drayage).

**References.** `hindsight_problem(env)` (EURO's `get_hindsight_problem`; Li et al. 2024) holds every request with its release time and windows, and the scenario seed that fixes all draws. `solve_request_hindsight` searches it with full information. Both are bounds, not policies. For outside traffic, the perfect-information search is clairvoyant over the same keyed realization, which is Kullman's PI relaxation.

## 7. Source mapping

| Element | Source | Code |
|---|---|---|
| Pooled FCFS M/M/c/∞ public stations; queue or leave; commitment | Kullman et al. (2021) §2.1–2.2, §7.1 | `models/simulation/public_station_queue.py`, `public_charging.py`, `state/feasibility.py` |
| Transient expected wait; Erlang C | Kullman §4.3.2; Erlang C | `models/simulation/queueing.py` |
| Stationary i.i.d. waits | Keskin, Akhavan-Tabatabaei & Çatay (2019, WSC) | `StationaryWaitChargingStation` |
| Time-of-day arrivals | Keskin, Laporte & Çatay (2019); thinning (Lewis & Shedler 1979) | `StationQueue._schedule_next_arrival` |
| Port outages | Sweda, Dolinskaya & Klabjan (2017); Rempel et al. (2023) | `StationQueue._change_port` |
| Occupancy indicator | Dastpak et al. (2024) | `PublicChargingStation.feature_row` |
| Private depot charging | Kullman §2.1 | `ChargingStation.add_private_charger` |
| US and EU driving rules | 49 CFR 395.3; Reg. 561/2006 Art. 6–8 | `models/core/hours_of_service.py` |
| Rest placement anywhere | Goel (2009, 2010) | `insert_rests` |
| Rest at stations; charging as rest | Bai et al. (2023); Kahlert et al. (2026) | `rest_locations: stops_only`, `handling_overhead_hours` |
| Independent compliance audit | — | `models/core/hos_audit.py` |
| Request with pickup and delivery | Savelsbergh & Sol (1995); Berbeglia et al. (2010) | `models/core/requests.py` |
| Epoch information process and filter | EURO Meets NeurIPS 2022 quickstart | `models/environment/request_engine.py` |
| Release dates, trips, manifests | Li, Archetti & Ljubić (2024), formulation (1) | `request_engine.py`, `feasibility.py` |
| Served-on-time objective; waiting; no preemptive returns | Voccia et al. (2019) | `RequestEngine.metrics`, `WAIT` |
| Preemptive returns | Ulmer, Thomas & Mattfeld (2019) | `allow_preemptive_returns` |
| Soft deadlines | Ulmer (2017) | `deadline_type: soft` |
| Drayage tasks, arcs, windows, processing | Xiao et al. (2025) §3.1, §5.2.1 | `request_engine.py`, `feasibility.py` |
| Load-dependent drayage energy | Kancharla & Ramadurai (2020) eq. (62); Xiao Table 10 | `kancharla_power_ratio` |
| Hindsight problem | EURO `get_hindsight_problem`; Li et al. (2024) | `hindsight_problem`, `solve_request_hindsight` |
| Single-truck HOS oracle | Bai et al. (2023) | `baselines/optimization/bai2023_dp.py` |
| Kullman myopic baseline | Kullman §5.2.3 | `baselines/heuristics/component_baselines.py` |

## 8. Documented departures

- **A fleet instead of one EV.** With one truck the queue model is Kullman's exactly; the test equates its realized wait with the clairvoyant wait function. With several trucks, a charging truck delays outside vehicles, which delay later trucks.
- **Stochastic travel and service**, where Kullman's travel is deterministic.
- **Fleet charging time** is the deterministic CC-CV time of the chosen target; outside vehicles keep Exp($\mu$).
- **Utilization wording.** Kullman's "u = 0.90 means a 90 % chance that all chargers are busy" equals the Erlang C value only for one charger. We use their formula $\lambda = u c \mu$ and report Erlang C next to $u$.
- **EURO has an unlimited fleet** and plans whole epochs. Here the fleet is fixed and each truck decides at its own epochs. The filter additionally requires an energy, charging and HOS witness.
- **Voccia's fleet is uncapacitated**; ours has payload capacity.
- **Xiao's model is static** and includes battery swapping, two charging modes and a two-container extension; those are left out. Time and penalty metrics replace costs in yuan.
- **Hours of service.** One driver per truck. No team driving, sleeper-berth split, or EU 3+9 rest split. Horizons are capped at 96 h, so weekly limits cannot bind. EU charging as a break follows trade-press reports of CORTE enforcement guidance, and is flagged as an interpretation. For EU rules, the window constrains driving only.
- **The legacy static mode keeps `FleetTaskRegistry`.** This is the `static_delivery` special case (no pickup, starts open, one task per node), so the golden traces stay exact. `problem.loading: at_depot` runs a static instance through the request engine with depot manifests.
- **`SERVED_LATE`** also marks hard-deadline late deliveries, which count as not served.
- **System events at or past the horizon are dropped** (the plan says "past"). An event exactly at the horizon cannot inform any decision.
- **Rests at the arrival epoch at a busy station** are allowed; after the rest the truck re-observes the station.
