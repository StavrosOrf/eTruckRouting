"""Decision-time additions to a trained generalist policy (track A, user decisions 2026-10-27).

Nothing here changes training; every piece is switched on for evaluation only.

* **Feasibility mask** (all five problems). The simulator's hard mask looks one
  leg ahead: it allows a move the truck has the energy for. The policy's
  failures on validation were the next leg: a truck at a customer with 13–150
  kWh left and no charger and no depot in reach (``no_feasible_action``), or a
  truck that charges late and is still driving home at the horizon
  (``time_limit_exhausted``). :func:`safe_actions` keeps only the feasible
  actions after which

  - a customer move leaves enough energy to reach a charger or the depot
    (nominal path energy times the configured worst-case multiplier, as the
    simulator's own mask), and
  - a move or a charge still leaves time to deliver what the truck carries
    (and, after a pickup, the new request) and reach the depot before the
    horizon: nominal travel (the mean of the traffic model), the charge's own
    duration from the station curve, and, with nothing on board and the depot
    out of reach on the battery left, one stop at the best charger charging
    at full power. With loads on board the bound is the farthest delivery and
    the way home from it. Service, waiting and rest times are left out, so the
    rule only removes moves that cannot make it home even when nothing else
    delays them.

  When the time rule would remove every action the energy rule leaves, the
  energy rule alone stands (dropping both stranded trucks on same-day, where
  a truck out at the horizon is no failure but a stranded one is); when that
  too would leave nothing, the hard mask stands. ``EVRP_SAFETY_MASK=on``
  switches it on and ``EVRP_SAFETY_MARGIN`` (hours, default 0) asks the truck
  home that much before the horizon, against the travel-time noise; forked
  workers inherit both.

* **Retirement guard** (static problems, with the mask). The depot action
  retires a truck for good, and once a truck the plan still counts on has
  retired the others may lack the payload for its customers. With
  ``EVRP_RETIRE_GUARD=on`` :func:`retire_guarded` hides the depot action from
  a truck whose planned route is not empty, while any other action is left.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from EVRoutingEnv.state.feasibility import ActionKind, _energy_multiplier

_TOLERANCE = 1e-9
_KINDS = tuple(ActionKind)


def safety_mask_enabled() -> bool:
    mode = os.environ.get("EVRP_SAFETY_MASK", "off") or "off"
    if mode not in ("on", "off"):
        raise ValueError(f"EVRP_SAFETY_MASK must be 'on' or 'off', not {mode!r}")
    return mode == "on"


def safety_margin() -> float:
    """Hours a truck must be home before the horizon (``EVRP_SAFETY_MARGIN``, default 0), against travel-time noise."""
    value = float(os.environ.get("EVRP_SAFETY_MARGIN", "0") or 0.0)
    if not 0.0 <= value < 24.0:
        raise ValueError(f"EVRP_SAFETY_MARGIN must lie in [0, 24) hours, not {value}")
    return value


def retire_guard_enabled() -> bool:
    mode = os.environ.get("EVRP_RETIRE_GUARD", "off") or "off"
    if mode not in ("on", "off"):
        raise ValueError(f"EVRP_RETIRE_GUARD must be 'on' or 'off', not {mode!r}")
    return mode == "on"


def retire_guarded(kinds, feasible, route) -> np.ndarray:
    """``feasible`` without the depot action when the truck's planned ``route`` is not empty and anything else is left."""
    if not route:
        return feasible
    keep = feasible & (np.asarray(kinds) != _KINDS.index(ActionKind.DEPOT))
    return keep if keep.any() else feasible


def safe_actions(env, kinds, targets, values, energies, feasible, index, energy, hours) -> np.ndarray:
    """``feasible`` minus the actions after which the active truck strands or cannot get home in time.

    ``kinds``/``targets``/``values``/``energies`` are the action rows' kind
    codes (into ``ActionKind``), target nodes, charge values and required
    energies; ``index``/``energy``/``hours`` the all-pairs nominal matrices.
    """
    if env.active_truck_id is None or not feasible.any():
        return feasible
    truck = env.trucks[env.active_truck_id]
    here = int(truck.current_node)
    battery = float(truck.current_battery)
    now, horizon = float(env.global_clock), float(env.max_time) - safety_margin()
    road = _Road(env, index, energy, hours)
    engine = getattr(env, "request_engine", None) if getattr(env, "dynamic_requests", False) else None
    on_board = [int(request.delivery.node) for request in engine.registry.on_board(truck.truck_id)] if engine else []

    def finish(node, left: float, stops: list[int]) -> float:
        return road.via(node, stops) if stops else road.home(node, left)

    keep = feasible.copy()
    no_strand = feasible.copy()  # the energy rule alone
    for action in np.flatnonzero(feasible):
        kind = _KINDS[int(kinds[action])]
        target = int(targets[action])
        if kind in (ActionKind.CUSTOMER, ActionKind.CHARGER):
            required = float(energies[action])
            left = battery - required if required >= 0.0 else battery - road.energy(here, target)
            if kind is ActionKind.CUSTOMER and left + _TOLERANCE < road.reserve(target):
                keep[action] = no_strand[action] = False
                continue
            stops = list(on_board)
            if kind is ActionKind.CUSTOMER and engine is not None:
                delivery = int(engine.registry.get(action - env.num_charging_nodes).delivery.node)
                if delivery == target and delivery in stops:
                    stops.remove(delivery)  # delivering it now
                elif delivery != target:
                    stops.append(delivery)  # a pickup: its delivery follows
            if now + road.hours(here, target) + finish(target, left, stops) > horizon + _TOLERANCE:
                keep[action] = False
        elif kind is ActionKind.CHARGE:
            try:
                after, charge_hours, _ = env._certificate_charge_transition(here, battery, float(values[action]))
            except (KeyError, ValueError):
                continue  # a charger the station curve does not know (the depot's): no time rule
            if now + float(charge_hours) + finish(here, float(after), on_board) > horizon + _TOLERANCE:
                keep[action] = False
    if keep.any():
        return keep
    return no_strand if no_strand.any() else feasible


class _Road:
    """Nominal energy and hours between nodes, the nearest refuge, and the fastest way home."""

    def __init__(self, env, index, energy, hours):
        self.index, self._energy, self._hours = index, energy, hours
        self.multiplier = _energy_multiplier(env)
        self.depot = int(env.joint_instance.depot_node)
        powers = env.charging_station.charger_power_kw
        chargers = [int(node) for node in env.charging_nodes if int(node) in index and powers.get(int(node), 0.0) > 0.0]
        self.chargers = np.asarray([index[node] for node in chargers], dtype=np.int64)
        self.powers = np.asarray([float(powers[node]) for node in chargers])
        refuges = {*map(int, env.charging_nodes), self.depot}
        self.refuges = np.asarray([index[node] for node in refuges if node in index], dtype=np.int64)

    def _row(self, node) -> int:
        return self.index.get(int(node), -1)

    def energy(self, source, target) -> float:
        if int(source) == int(target):
            return 0.0
        a, b = self._row(source), self._row(target)
        return float(self._energy[a, b]) * self.multiplier if a >= 0 and b >= 0 else np.inf

    def hours(self, source, target) -> float:
        if int(source) == int(target):
            return 0.0
        a, b = self._row(source), self._row(target)
        return float(self._hours[a, b]) if a >= 0 and b >= 0 else np.inf

    def reserve(self, node) -> float:
        """Energy to the nearest charger or the depot."""
        row = self._row(node)
        if row < 0 or not self.refuges.size:
            return 0.0
        return float(np.min(self._energy[row, self.refuges])) * self.multiplier

    def via(self, node, stops: list[int]) -> float:
        """Least hours from ``node`` through every stop to the depot: at least the farthest stop and home from it."""
        return max(self.hours(node, stop) + self.hours(stop, self.depot) for stop in stops)

    def home(self, node, battery: float) -> float:
        """Least hours from ``node`` to the depot: direct when the battery covers it, else through one charger."""
        if int(node) == self.depot:
            return 0.0
        if self.energy(node, self.depot) <= battery + _TOLERANCE:
            return self.hours(node, self.depot)
        row, depot = self._row(node), self._row(self.depot)
        if row < 0 or depot < 0 or not self.chargers.size:
            return np.inf
        # A truck standing at a charger charges where it is (the matrices' diagonal may read unreachable).
        here = self.chargers == row
        reach = np.where(here, 0.0, self._energy[row, self.chargers] * self.multiplier)
        reachable = reach <= battery + _TOLERANCE
        if not reachable.any():
            return np.inf
        onward = self._energy[self.chargers, depot] * self.multiplier
        charge = np.maximum(0.0, onward - (battery - reach)) / self.powers
        total = np.where(here, 0.0, self._hours[row, self.chargers]) + charge + self._hours[self.chargers, depot]
        return float(np.min(np.where(reachable, total, np.inf)))


# ---------------------------------------------------------------------------
# Decision-time search on the static problems (track A)
# ---------------------------------------------------------------------------
#
# Both searches roll the policy itself forward in copies of the simulator whose
# future is drawn afresh (:func:`fresh_clone`): what has happened stays, every
# draw still to come (travel times, energy, service, station traffic) comes
# from a new seed, never from the episode's own. A copy carries a copy of the
# planner behind the plan input, so it keeps re-planning on deviations exactly
# as the episode does (a plan frozen at the copy sent the policy into charging
# loops under fresh draws).
#
# * ``lookahead``: at every decision with more than one candidate, the
#   policy's ``k`` most likely actions are each played in a copy (the same seed
#   for all ``k``, so they meet the same traffic) and the policy finishes the
#   episode greedily; the action with the best outcome is taken.
# * ``best_of_n``: at the first decision, ``n`` sampled policy rollouts; the
#   routes of the best become the planner's starting plan for the episode (in
#   place of the cached BKS plan) when that rollout served every customer;
#   the policy then plays greedily while the planner re-plans from those
#   routes on deviations.
#
# Outcomes are ranked as the gate reads them: no failure, then success, then
# fewer operating hours.

SEARCHES = ("lookahead", "best_of_n")


def outcome_key(summary: dict) -> tuple:
    return (bool(summary["failed"]), not bool(summary["success"]), float(summary["hours"]))


def search_seed(*parts: int) -> int:
    """A rollout seed from the episode and decision, far from the scenario seeds."""
    return int(np.random.default_rng([*(int(part) % (2**32) for part in parts), 0x53524348]).integers(2**40, 2**62))


def _copy_provider(provider, shadow):
    """The plan provider for a copy: its planner state, re-keyed to the copy, sharing read-only network data."""
    import copy

    planner = getattr(provider, "planner", None)
    memo = {}
    for name in ("net", "curve", "waits", "search", "tasks"):
        value = getattr(planner, name, None)
        if value is not None:
            memo[id(value)] = value
    copied = copy.deepcopy(provider, memo)
    inner = getattr(copied, "planner", None)
    if inner is not None and getattr(inner, "_key", None) is not None:
        inner._key = (id(shadow), int(shadow.scenario_seed))
    return copied


def fresh_clone(env, seed: int):
    """A copy of a static ``env`` (and of its plan provider) whose future draws come from ``seed``."""
    from EVRoutingEnv.baselines.optimization.perfect_information import _clone
    from EVRoutingEnv.models.simulation.scenario import ScenarioRandomStreams

    if getattr(env, "dynamic_requests", False):
        raise ValueError("decision-time search covers the static problems; a copy would carry hidden requests")
    provider = getattr(env, "_plan_provider", None)
    env._plan_provider = None  # copied on its own below, sharing the planner's network
    try:
        shadow = _clone(env)
    finally:
        env._plan_provider = provider
    if provider is not None:
        shadow._plan_provider = _copy_provider(provider, shadow)
    streams = ScenarioRandomStreams(seed)
    shadow.scenario_random_streams = streams
    for simulator in (shadow.traffic_simulator, shadow.delivery_simulator):
        simulator.seed = seed
        simulator.random_streams = streams
        simulator._uncertainty_cache.clear()
    stations = getattr(shadow, "public_stations", None)
    if stations is not None:
        _refresh_stations(shadow, stations, streams)
    return shadow


def _refresh_stations(env, stations, streams) -> None:
    """Redraw every pending station draw from now (all exponential, so memoryless).

    Stations advance lazily, so first each is brought to the current clock with
    the episode's own draws: that stretch has happened, and a reservation it
    made was promised to a truck (redrawing it from a lagging station clock
    once moved a promised reservation into the past). Then everything still
    to come is redrawn, and only reservations predicted for later are
    predicted again.
    """
    now = float(env.global_clock)
    if not stations.eager:
        for node in stations.queues:
            stations.queue_at(node, now)
    stations.streams = streams
    for queue in stations.queues.values():
        queue.streams = streams
        clock, parameters = float(queue.clock), queue.p
        for port in queue.ports:
            if math.isfinite(port.next_change):
                up_mean = parameters.mean_repair_hours * parameters.availability / (1.0 - parameters.availability)
                mean = up_mean if port.timeline_up else parameters.mean_repair_hours
                port.next_change = clock + queue._exp("port_state", ("fresh", port.index, port.change_index), mean)
            if port.occupant is not None and port.occupant.kind == "outside" and port.busy_until is not None:
                port.busy_until = clock + queue._exp(
                    "station_service", ("fresh", port.occupant.key), parameters.service_mean_hours,
                )
        for entry in queue.queue:
            if entry.kind == "outside":
                entry.service = queue._exp("station_service", ("fresh", entry.key), parameters.service_mean_hours)
        if queue._next_arrival is not None:
            queue._next_arrival = (clock, queue._arrivals_k)
            queue._schedule_next_arrival()
    for truck, (time, _) in list(stations.scheduled.items()):
        if time > now + 1e-9:  # a wake-up still to come; one due now stays as promised
            stations._unschedule(truck)
            stations.dirty_nodes.add(stations.truck_station[truck])
    env._sync_components()


def _logits(policy, observation):
    import torch

    from algo.generalist_observation import collate

    with torch.no_grad():
        return policy(collate([observation], "cpu")).logits[0, : observation.num_actions].numpy()


def rollout(env, policy, observation=None, *, deterministic: bool = True, rng=None, served=None) -> dict:
    """Play ``env`` to its end with ``policy``; ``served`` collects ``(truck, slot)`` of customer moves."""
    from algo.generalist_ppo import episode_summary

    observation = env._get_observation() if observation is None else observation
    while env.active_truck_id is not None and not env._episode_ended:
        logits = _logits(policy, observation)
        if deterministic:
            position = int(np.argmax(logits))
        else:
            probabilities = np.exp(logits - logits.max())
            position = int(rng.choice(len(logits), p=probabilities / probabilities.sum()))
        action = int(observation.action_index[position])
        if served is not None and env.num_charging_nodes <= action < env.num_charging_nodes + env.fixed_num_stops:
            served.append((int(env.active_truck_id), action - env.num_charging_nodes))
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return episode_summary(env)


def lookahead_action(env, observation, policy, k: int, seed: int) -> tuple[int, bool]:
    """The best of the policy's ``k`` likeliest actions, each finished greedily in a fresh copy.

    Returns the action and whether the search fell back to the policy's own
    choice because a copy broke (a simulator inconsistency in the copy, never
    in the episode itself).
    """
    from algo.generalist_ppo import episode_summary

    logits = _logits(policy, observation)
    order = np.argsort(-logits, kind="stable")[: max(1, k)]
    if len(order) == 1:
        return int(observation.action_index[order[0]]), False
    outcomes = []
    try:
        for position in order:
            shadow = fresh_clone(env, seed)
            following, _, terminated, truncated, _ = shadow.step(int(observation.action_index[position]))
            summary = episode_summary(shadow) if terminated or truncated else rollout(shadow, policy, following)
            outcomes.append(outcome_key(summary))
    except RuntimeError:
        return int(observation.action_index[order[0]]), True
    best = min(range(len(order)), key=lambda index: (outcomes[index], index))
    return int(observation.action_index[order[best]]), False


def best_of_n_routes(env, policy, n: int, seed: int) -> tuple[tuple, dict[int, list[int]]]:
    """The best of ``n`` sampled policy rollouts: its outcome key and customer nodes ``{truck: [node, ...]}`` in service order."""
    best, best_routes = None, None
    for index in range(n):
        served: list[tuple[int, int]] = []
        rng = np.random.default_rng(search_seed(seed, index, 1))
        try:
            shadow = fresh_clone(env, search_seed(seed, index))
            key = outcome_key(rollout(shadow, policy, deterministic=False, rng=rng, served=served))
        except RuntimeError:
            continue  # a copy that broke counts for nothing
        if best is None or key < best:
            routes: dict[int, list[int]] = {int(truck.truck_id): [] for truck in env.trucks}
            for truck, slot in served:
                routes.setdefault(truck, []).append(int(env.joint_instance.tasks[slot].node_id))
            best, best_routes = key, routes
    return best, best_routes


def _start_from(env, routes: dict[int, list[int]]) -> None:
    """Make ``routes`` the time-0 plan of ``env``'s plan provider (through its plan cache)."""
    from algo.generalist_plans import PlanProvider

    provider = PlanProvider(env.problem_name, env)
    planner = provider.planner
    planner.plan_cache = {str(int(env.scenario_seed)): {
        "tag": planner.settings.plan_tag(),
        "routes": {str(truck): nodes for truck, nodes in routes.items()},
        "duration_margin": planner.settings.planner.duration_margin,
    }}
    env._plan_withheld, env._plan_provider = False, provider


def search_episode(env, policy, seed: int, search: str, *, k: int = 4, n: int = 16) -> dict:
    """One evaluation episode of ``policy`` on ``env`` (reset to ``seed``) with a decision-time search."""
    from algo.generalist_ppo import episode_summary

    if search not in SEARCHES:
        raise ValueError(f"unknown search {search!r}; expected one of {SEARCHES}")
    observation, _ = env.reset(seed=seed)
    if search == "best_of_n":
        if env.active_truck_id is not None and getattr(env.observation_spec, "plan_conditioned", False):
            key, routes = best_of_n_routes(env, policy, n, search_seed(seed))
            if key is not None and not key[0] and not key[1]:  # only a rollout that served everyone is a plan
                _start_from(env, routes)
                observation = env._get_observation()
        return rollout(env, policy, observation)
    step = fallbacks = 0
    while env.active_truck_id is not None and not env._episode_ended:
        action, fell_back = lookahead_action(env, observation, policy, k, search_seed(seed, step))
        fallbacks += int(fell_back)
        observation, _, terminated, truncated, _ = env.step(action)
        step += 1
        if terminated or truncated:
            break
    return {**episode_summary(env), "search_fallbacks": fallbacks}


_POLICIES: dict = {}
_ENVS: dict = {}


def _search_job(job):
    """Forked worker: one searched episode of one run, policy on the CPU."""
    import torch

    from algo.generalist_observation import GeneralistEnv
    from algo.generalist_policy import GeneralistActorCritic

    run, checkpoint, problem, config, seed, scale, search, k, n, reward = job
    torch.set_num_threads(1)
    if run not in _POLICIES:
        _POLICIES[run] = GeneralistActorCritic.load(checkpoint, prefix="best", map_location="cpu").eval()
    policy = _POLICIES[run]
    spec = policy.config.observation_spec
    if (problem, spec) not in _ENVS:
        _ENVS[(problem, spec)] = GeneralistEnv(config, spec, problem=problem)
    started = time.perf_counter()
    summary = search_episode(_ENVS[(problem, spec)], policy, seed, search, k=k, n=n)
    seconds = time.perf_counter() - started  # one CPU core: the decision latency the gate reports
    return run, {
        "problem": problem, "seed": int(seed), "score": reward.score(summary, scale), **summary,
        "search_seconds": seconds, "search_seconds_per_decision": seconds / max(1, summary["steps"]),
    }


def evaluate_search(jobs: list[tuple], workers: int, partial=None) -> dict[str, list[dict]]:
    """Run ``(run, checkpoint, problem, config, seed, scale, search, k, n, reward)`` jobs; rows per run.

    Each finished episode is appended to ``partial`` (JSON lines) when given, and
    episodes already there are not played again, so a crashed or timed-out
    evaluation resumes. Fork before any CUDA use: the workers run the policy on
    the CPU.
    """
    import json
    import multiprocessing as mp
    from pathlib import Path

    rows: dict[str, list[dict]] = {}
    done_keys = set()
    if partial is not None and Path(partial).exists():
        for line in Path(partial).read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                rows.setdefault(item["run"], []).append(item["row"])
                done_keys.add((item["run"], item["row"]["problem"], int(item["row"]["seed"])))
    pending = [job for job in jobs if (job[0], job[2], int(job[4])) not in done_keys]
    if done_keys:
        print(f"resuming: {len(done_keys)} searched episodes on file, {len(pending)} to go", flush=True)
    if not pending:
        return rows
    handle = open(partial, "a") if partial is not None else None
    try:
        with mp.get_context("fork").Pool(max(1, min(workers, len(pending)))) as pool:
            for done, (run, record) in enumerate(pool.imap_unordered(_search_job, pending, chunksize=1), start=1):
                rows.setdefault(run, []).append(record)
                if handle is not None:
                    handle.write(json.dumps({"run": run, "row": record}) + "\n")
                    handle.flush()
                if done % 32 == 0 or done == len(pending):
                    print(f"searched {done}/{len(pending)} episodes", flush=True)
    finally:
        if handle is not None:
            handle.close()
    return rows
