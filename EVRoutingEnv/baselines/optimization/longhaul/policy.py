"""Closed-loop executors for the static long-haul problems.

``LonghaulPlanPolicy`` plans the customer routes at time 0 (ALNS of
:mod:`planner`, verified with the exact continuation search), then at every
decision of a truck re-solves the :mod:`continuation` search from that
truck's realized state (node, battery, clock, driver clock) over the
customers its plan still owes, and takes the first action. It never falls
back to greedy customer selection.

Re-planning (``replan``):

* ``"deviation"``: the plan is reconciled with the state at every decision. A
  customer claimed by a truck other than its planned one, a customer no live
  truck owns, a truck that retired with work planned, or a route over the
  truck's remaining payload triggers an ALNS re-plan of all unclaimed
  customers from every live truck's current start (warm-started from the
  reconciled plan). This makes the executor answer from arbitrary states, as
  a DAgger expert must.
* ``"events"``: as ``"deviation"``, and also re-plans each time a truck
  finishes a customer (a rolling-horizon re-optimizing executor that absorbs
  realized travel, energy and service times).

A truck's start is its current node, or for a truck driving to a stop, that
stop at its nominal ETA with the nominal leg energy spent (the observation's
in-flight information); a truck driving to a customer keeps it.

Every search is iteration-bounded and seeded by the scenario, so a run is
reproducible whatever the machine load. ``plan_cache`` maps a scenario seed
to a precomputed time-0 plan of the same settings (saves the first search).

Public stations (Kullman et al. 2021 queue-or-leave epochs): plans price each
charging stop at a quantile of the stations' expected waits; on a busy
arrival the executor compares queueing (the observed queue's expected
newcomer wait over the ports that are up) with the best completion that
charges elsewhere. Hours of service with rests at stops only: the
continuation search places daily rests and 30-minute breaks at stops through
the simulator's own ``hours_of_service`` functions.

Only information every policy has is used: plan-time expected waits, the
mask, the trucks' states with nominal in-flight ETAs and, at a station the
active truck stands at, that station's queue. The policy never changes the
environment (the queue read is the one the mask itself performs).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from EVRoutingEnv.baselines.optimization.longhaul.continuation import (
    ContinuationSearch,
    ContinuationSettings,
)
from EVRoutingEnv.baselines.optimization.longhaul.network import (
    charge_curve,
    network_for,
    station_waits,
)
from EVRoutingEnv.baselines.optimization.longhaul.planner import (
    LonghaulPlanner,
    PlannerSettings,
    PlanProblem,
    TruckStart,
)
from EVRoutingEnv.models.core.customer import TaskStatus
from EVRoutingEnv.models.core.hours_of_service import (
    HosActivity,
    advance_to,
    window_left,
)
from EVRoutingEnv.models.simulation import queueing
from EVRoutingEnv.state.feasibility import ActionKind, joint_action_feasibility


@dataclass(frozen=True)
class LonghaulSettings:
    name: str = "lh_plan"
    planner: PlannerSettings = field(default_factory=PlannerSettings)
    continuation: ContinuationSettings = field(default_factory=ContinuationSettings)
    restarts: int = 3
    # Exact (continuation) route durations must end this far before the horizon.
    execution_margin: float = 3.0
    verify_rounds: int = 3
    replan: str = "none"  # "none" | "deviation" | "events"
    replan_iterations: int = 400
    replan_descent_rounds: int = 3
    # Congested problems: quantile of the station waits used as the planned
    # per-stop wait (the executor prefers stations below it).
    wait_quantile: float = 0.25

    def __post_init__(self) -> None:
        if self.replan not in ("none", "deviation", "events"):
            raise ValueError("replan must be 'none', 'deviation' or 'events'")

    def plan_tag(self) -> str:
        """Hash of everything the time-0 plan depends on (for plan caches)."""
        payload = {
            "planner": asdict(self.planner), "continuation": asdict(self.continuation),
            "restarts": self.restarts, "execution_margin": self.execution_margin,
            "verify_rounds": self.verify_rounds, "wait_quantile": self.wait_quantile, "version": 2,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


class LonghaulPlanPolicy:
    """Plan (ALNS), execute with the continuation search, re-plan as configured."""

    def __init__(self, settings: LonghaulSettings | None = None, plan_cache: dict | None = None):
        self.settings = settings or LonghaulSettings()
        self.name = self.settings.name
        self.plan_cache = plan_cache
        self._key = None
        self.stats: dict = {}

    # ------------------------------------------------------------------
    def reset(self, env=None, seed=None) -> None:
        self._key = None

    def act(self, env) -> int:
        return self(env)

    def _setup(self, env) -> None:
        s = self.settings
        hos = getattr(env, "hos", None)
        max_hop = float(hos.regime.driving_before_break) if hos is not None else None
        self.net = network_for(env, max_hop_hours=max_hop)
        self.curve = charge_curve(env)
        self.waits = station_waits(env, self.net)
        self.regime = hos.regime if hos is not None else None
        handling = float(hos.config.handling_overhead_hours) if hos is not None else 0.0
        self.search = ContinuationSearch(
            self.net, self.curve, regime=self.regime, waits=self.waits,
            settings=s.continuation, handling_hours=handling,
            levels=[float(v) for v in env.charge_action_values],
        )
        self.depot_node = int(env.joint_instance.depot_node)
        self.depot = self.net.position[self.depot_node]
        self.tasks = list(env.joint_instance.tasks)
        self.slot_of_node = {int(task.node_id): slot for slot, task in enumerate(self.tasks)}
        self.n_chargers = len(env.charging_nodes)
        self.charger_slot = {int(node): index for index, node in enumerate(env.charging_nodes)}
        self.depot_action = self.n_chargers + env.fixed_num_stops
        self.charge_offset = self.depot_action + 1
        self.levels = [float(v) for v in env.charge_action_values]
        extra = list(getattr(env, "extra_action_kinds", ()))
        base = self.charge_offset + len(self.levels)
        self.extra_index = {kind: base + position for position, kind in enumerate(extra)}
        self.stats = {"decisions": 0, "fallbacks": 0, "replans": 0, "replans_improved": 0,
                      "deviations": 0, "searches": 0, "queue_joined": 0, "queue_left": 0,
                      "plan": None, "plan_cached": False}
        self.replanned_at: set = set()
        self.duration_margin = s.planner.duration_margin
        self._plan_initial(env)

    # ------------------------------------------------------------------
    def _planner_settings(self, seed_offset: int, **changes) -> PlannerSettings:
        s = self.settings
        wait = 0.0
        if self.waits is not None:
            wait = float(np.quantile(self.waits[self.net.chargers], s.wait_quantile))
        return replace(s.planner, seed=s.planner.seed + seed_offset, wait_per_stop=wait, **changes)

    def _problem(self, env, starts, slots) -> PlanProblem:
        return PlanProblem(
            depot=self.depot,
            customers=[self.net.position[int(self.tasks[slot].node_id)] for slot in slots],
            demand=[float(self.tasks[slot].demand) for slot in slots],
            service=[float(self.tasks[slot].base_service_time) for slot in slots],
            starts=starts, horizon=float(env.max_time),
            capacity=float(self.net.capacity), hos=self.regime is not None,
        )

    def _plan_initial(self, env) -> None:
        s = self.settings
        cached = None
        if self.plan_cache is not None:
            cached = self.plan_cache.get(str(int(env.scenario_seed)))
        if cached is not None and cached.get("tag") == s.plan_tag():
            self.routes = {int(k): [int(n) for n in v] for k, v in cached["routes"].items()}
            self.duration_margin = float(cached.get("duration_margin", self.duration_margin))
            self.stats["plan"] = {k: v for k, v in cached.items() if k != "tag"}
            self.stats["plan_cached"] = True
            return
        starts = [
            TruckStart(truck_id=int(truck.truck_id), node=self.depot, time=0.0,
                       battery=float(truck.current_battery),
                       payload=float(truck.remaining_payload if truck.remaining_payload is not None
                                     else truck.payload_capacity))
            for truck in env.trucks
        ]
        slots = list(range(len(self.tasks)))
        problem = self._problem(env, starts, slots)
        margin = s.planner.duration_margin
        best = None
        for _ in range(max(1, s.verify_rounds)):
            candidates = []
            for restart in range(max(1, s.restarts)):
                planner = LonghaulPlanner(
                    self.net, problem,
                    self._planner_settings(restart * 7919 + int(env.scenario_seed) % 100000,
                                           duration_margin=margin),
                )
                routes, cost, info = planner.solve()
                candidates.append((cost, restart, routes, info, planner))
            cost, _, routes, info, planner = min(candidates, key=lambda item: (item[0], item[1]))
            exact = self._exact_durations(env, routes, problem)
            cheap = [planner.route_cost(t, r)[0] for t, r in enumerate(routes)]
            excess = [exact[t] - (env.max_time - s.execution_margin) for t in range(len(routes)) if routes[t]]
            worst = max(excess, default=-1.0)
            if best is None or worst <= 0.0:
                best = (routes, cost, exact, margin, info, cheap)
            if worst <= 0.0:
                break
            margin += min(worst, 6.0) + 0.5
        routes, cost, exact, margin, info, cheap = best
        self.duration_margin = margin
        self.routes = {int(env.trucks[t].truck_id): [int(self.tasks[c].node_id) for c in route]
                       for t, route in enumerate(routes)}
        self.stats["plan"] = {
            "planned_hours": float(cost),
            "exact_hours": float(sum(exact)),
            "exact_route_hours": [round(float(x), 3) for x in exact],
            "planned_route_hours": [round(float(x), 3) for x in cheap],
            "routes": {str(k): [int(n) for n in v] for k, v in self.routes.items()},
            "duration_margin": float(margin),
            "iterations": int(info["iterations"]),
        }

    def plan_record(self) -> dict:
        """The time-0 plan, as stored in a plan cache."""
        return {"tag": self.settings.plan_tag(), **(self.stats.get("plan") or {})}

    def _exact_durations(self, env, routes, problem) -> list[float]:
        out = []
        hos = getattr(env, "hos", None)
        for truck, route in enumerate(routes):
            if not route:
                out.append(0.0)
                continue
            start = problem.starts[truck]
            stops = [(problem.customers[c], problem.service[c], False) for c in route]
            stops.append((self.depot, 0.0, True))
            clock = hos.clock(int(start.truck_id)) if hos is not None else None
            result = self.search.solve(start.node, start.battery, start.time, clock, stops,
                                       full_stages=len(stops))
            out.append(math.inf if result is None else float(result.end_time) - start.time)
        return out

    # ------------------------------------------------------------------
    def remaining(self, env, truck) -> list[int]:
        out = []
        payload = float(truck.remaining_payload) if truck.remaining_payload is not None else math.inf
        for node in self.routes.get(int(truck.truck_id), []):
            task = env.task_registry.task_for_node(int(node))
            if task.status in (TaskStatus.SERVED, TaskStatus.IN_SERVICE):
                continue
            if task.status is TaskStatus.CLAIMED and task.claimed_by != truck.truck_id:
                continue
            if float(task.demand) > payload + 1e-9:
                continue
            out.append(int(node))
            payload -= float(task.demand)
        return out

    def _stops(self, nodes):
        stops = [(self.net.position[n], float(self.tasks[self.slot_of_node[n]].base_service_time), False)
                 for n in nodes]
        stops.append((self.depot, 0.0, True))
        return stops

    def _action_index(self, env, action) -> int | None:
        kind = action[0]
        if kind == "go":
            node = int(self.net.nodes[action[1]])
            if node == self.depot_node:
                return self.depot_action
            if node in self.charger_slot:
                return self.charger_slot[node]
            if node in self.slot_of_node:
                return self.n_chargers + self.slot_of_node[node]
            return None
        if kind == "charge":
            return self.charge_offset + self.levels.index(float(action[1]))
        if kind == "rest":
            wanted = ActionKind.REST_DAILY if action[1] == "daily" else ActionKind.REST_BREAK
            return self.extra_index.get(wanted)
        return None

    # ------------------------------------------------------------------
    def __call__(self, env, observation=None, info=None) -> int:
        key = (id(env), int(env.scenario_seed))
        if key != self._key:
            self._key = key
            self._setup(env)
        self.stats["decisions"] += 1
        truck = env.trucks[env.active_truck_id]
        if self.settings.replan != "none":
            self._reconcile(env, truck)
        decisions = joint_action_feasibility(env)
        feasible = [d.feasible for d in decisions]
        nodes = self.remaining(env, truck)
        here = self.net.position[int(truck.current_node)]
        if not nodes and here == self.depot and feasible[self.depot_action]:
            return self.depot_action
        stops = self._stops(nodes)
        unavailable = frozenset(
            self.net.position[int(node)]
            for node, available in env.charging_station.station_available.items()
            if not available and int(node) in self.net.position
        )
        charge_ok = any(feasible[self.charge_offset + i] for i in range(len(self.levels)))
        queue_index = self.extra_index.get(ActionKind.QUEUE)
        queue_ok = queue_index is not None and feasible[queue_index]
        navigation_ok = any(feasible[:self.depot_action + 1])
        hos = getattr(env, "hos", None)
        clock = hos.clock(int(truck.truck_id)) if hos is not None else None
        state = (here, float(truck.current_battery), float(env.global_clock), clock, stops)

        if queue_ok:
            wait = self._observed_wait(env, truck)
            stay = self.search.solve(*state, forced_charge=True, start_wait=wait, unavailable=unavailable)
            leave = self.search.solve(*state, can_charge_here=False, unavailable=unavailable)
            self.stats["searches"] += 2
            if stay is not None and (leave is None or stay.end_time <= leave.end_time):
                self.stats["queue_joined"] += 1
                return queue_index
            self.stats["queue_left"] += 1
            result = leave
        elif charge_ok and not navigation_ok:
            # At the head of a public queue (committed) or otherwise charge-only.
            result = self.search.solve(*state, forced_charge=True, start_wait=0.0, unavailable=unavailable)
            self.stats["searches"] += 1
        else:
            result = self.search.solve(*state, can_charge_here=charge_ok,
                                       start_wait=0.0 if charge_ok else None, unavailable=unavailable)
            self.stats["searches"] += 1

        excluded = set(unavailable)
        for _ in range(4):
            if result is None or result.first is None:
                break
            index = self._action_index(env, result.first)
            if index is not None and feasible[index]:
                return index
            if result.first[0] != "go":
                break
            excluded.add(result.first[1])
            result = self.search.solve(*state, can_charge_here=charge_ok and not queue_ok,
                                       start_wait=0.0 if charge_ok else None,
                                       unavailable=frozenset(excluded))
            self.stats["searches"] += 1
        return self._fallback(env, truck, nodes)

    def _observed_wait(self, env, truck) -> float:
        stations = env.public_stations
        node = int(truck.current_node)
        station = stations.queue_at(node, env.global_clock, fleet_arrival=True)
        up = sum(port.timeline_up for port in station.ports)
        params = stations.station_parameters(node)
        if up <= 0:
            return 24.0
        return queueing.newcomer_wait(up, params.service_rate, station.in_system())

    def _fallback(self, env, truck, nodes) -> int:
        """Energy-safe navigation toward the next planned stop (counted)."""
        from EVRoutingEnv.baselines.heuristics.canonical_baselines import (
            GreedyHeuristicPolicy,
            HeuristicParameters,
            decode_feasible_actions,
        )

        self.stats["fallbacks"] += 1
        navigator = GreedyHeuristicPolicy(HeuristicParameters(energy_safety_factor=1.2, target_soc=1.0))
        candidates = decode_feasible_actions(env)
        goal = nodes[0] if nodes else self.depot_node
        return navigator.navigate_toward(env, truck, goal, candidates)

    # ------------------------------------------------------------------
    def _reconcile(self, env, truck) -> None:
        """Re-plan when the state left the plan (or, for "events", after a service)."""
        alive = {int(t.truck_id) for t in env.trucks if not (t.is_complete or t.failed)}
        claimed_by: dict[int, int] = {}
        open_nodes: set[int] = set()
        for task in env.task_registry.tasks():
            if task.status is TaskStatus.CLAIMED:
                claimed_by[int(task.node_id)] = int(task.claimed_by)
            elif task.status is TaskStatus.UNASSIGNED:
                open_nodes.add(int(task.node_id))
        deviation = False
        planned_owner: dict[int, int] = {}
        for tid, route in self.routes.items():
            load = 0.0
            payload_left = None
            for t in env.trucks:
                if int(t.truck_id) == tid:
                    payload_left = float(t.remaining_payload) if t.remaining_payload is not None else math.inf
            for node in route:
                if node in open_nodes or node in claimed_by:
                    planned_owner[node] = tid
                    if node in open_nodes:
                        if tid not in alive:
                            deviation = True
                        load += float(self.tasks[self.slot_of_node[node]].demand)
            if payload_left is not None and load > payload_left + 1e-9:
                deviation = True
        for node in open_nodes:
            if node not in planned_owner:
                deviation = True
        for node, owner in claimed_by.items():
            if planned_owner.get(node) != owner:
                deviation = True
        event = False
        if self.settings.replan == "events":
            node = int(truck.current_node)
            if node in self.slot_of_node:
                task = self.tasks[self.slot_of_node[node]]
                key = (int(truck.truck_id), node)
                if task.status is TaskStatus.SERVED and key not in self.replanned_at:
                    self.replanned_at.add(key)
                    event = bool(open_nodes)
        if deviation:
            self.stats["deviations"] += 1
        if deviation or event:
            self._replan(env, claimed_by, open_nodes, alive, force=deviation)

    def _truck_start(self, env, truck, claimed_by) -> TruckStart:
        tid = int(truck.truck_id)
        state = env.truck_states[tid]
        payload = float(truck.remaining_payload) if truck.remaining_payload is not None else math.inf
        now = float(env.global_clock)
        node = int(truck.current_node)
        battery = float(truck.current_battery)
        time = now
        extra_drive = 0.0
        destination = getattr(truck, "route_destination", None)
        if state == "routing" and destination is not None:
            origin = self.net.position[node]
            target = self.net.position[int(destination)]
            departure = float(getattr(truck, "routing_start_time", now) or now)
            extra_drive = float(self.net.hours[origin, target])
            time = max(now, departure + extra_drive)
            battery = max(0.0, battery - float(self.net.energy[origin, target]) * self.settings.planner.energy_expected)
            node = int(destination)
            if node in self.slot_of_node and claimed_by.get(node) == tid:
                time += float(self.tasks[self.slot_of_node[node]].base_service_time)
                task = self.tasks[self.slot_of_node[node]]
                payload = max(0.0, payload - float(task.demand))
        window_used = driving_used = 0.0
        hos = getattr(env, "hos", None)
        if hos is not None:
            clock = advance_to(hos.regime, hos.clock(tid), max(now, hos.clock(tid).time), HosActivity.IDLE)
            window_used = max(0.0, float(hos.regime.shift_window_hours) - window_left(hos.regime, clock))
            driving_used = float(clock.driving_in_shift) + extra_drive
            window_used = min(float(hos.regime.shift_window_hours), window_used + (time - now))
        return TruckStart(truck_id=tid, node=self.net.position[node], time=time, battery=battery,
                          payload=payload, window_used=window_used, driving_used=driving_used)

    def _replan(self, env, claimed_by, open_nodes, alive, force: bool) -> None:
        s = self.settings
        trucks = list(env.trucks)
        starts = []
        for t in trucks:
            if int(t.truck_id) in alive:
                starts.append(self._truck_start(env, t, claimed_by))
            else:
                starts.append(TruckStart(int(t.truck_id), self.depot, float(env.global_clock), 0.0, 0.0))
        slots = sorted(self.slot_of_node[n] for n in open_nodes)
        if not slots:
            # Nothing left to assign: every truck finishes what it has claimed.
            self.routes = {int(t.truck_id): [n for n, o in claimed_by.items() if o == int(t.truck_id)]
                           for t in trucks}
            return
        index_of = {slot: position for position, slot in enumerate(slots)}
        problem = self._problem(env, starts, slots)
        # Dead trucks cannot take work: zero payload keeps them empty.
        problem.starts = [
            start if int(start.truck_id) in alive else replace(start, payload=0.0)
            for start in problem.starts
        ]
        current = []
        for position, t in enumerate(trucks):
            tid = int(t.truck_id)
            route = [index_of[self.slot_of_node[n]] for n in self.routes.get(tid, [])
                     if n in open_nodes and tid in alive]
            current.append(route)
        seen, cleaned = set(), []
        for route in current:
            kept = [c for c in route if c not in seen]
            seen.update(kept)
            cleaned.append(kept)
        planner = LonghaulPlanner(
            self.net, problem,
            self._planner_settings(int(env.scenario_seed) % 100000 + 104729 * (self.stats["replans"] + 1),
                                   iterations=s.replan_iterations, duration_margin=self.duration_margin),
        )
        missing = [c for c in range(len(slots)) if c not in seen]
        if missing:
            import random as _random

            planner._repair(cleaned, missing, _random.Random(self.stats["replans"]), regret_k=2)
        before = planner.total(cleaned)
        routes, cost, _ = planner.solve(initial=cleaned, descent_rounds=s.replan_descent_rounds)
        chosen = routes if cost < before - 1e-6 else cleaned
        if cost < before - 1e-6:
            self.stats["replans_improved"] += 1
        new_routes = {}
        for position, t in enumerate(trucks):
            tid = int(t.truck_id)
            own_claim = [n for n, o in claimed_by.items() if o == tid]
            new_routes[tid] = own_claim + [int(self.tasks[slots[c]].node_id) for c in chosen[position]]
        self.routes = new_routes
        self.stats["replans"] += 1
