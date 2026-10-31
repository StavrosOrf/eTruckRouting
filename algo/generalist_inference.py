"""Inference-time improvements for a trained generalist policy (plan 9.6).

Wrappers change only how actions are chosen at evaluation time; training, the
observation, the reward and the gate are untouched. Every wrapper measures its
own decision latency, because a near-optimal claim that uses one must state it.

Two kinds:

* Batched actors (:class:`TemperaturePolicy`, :class:`EnsemblePolicy`) are
  drop-in replacements for the policy inside
  :func:`algo.generalist_ppo.evaluate_policy_suite`.
* Sequential controllers (:class:`LookaheadController`,
  :class:`PlannerRepairController`) need the live simulator at each decision,
  so :func:`evaluate_wrapped_problem` runs the episodes one by one in forked
  workers and writes the same record schema as the campaign evaluator, plus
  per-episode latency fields.

Wrapper specifications are strings, ``name`` or ``name:key=value,key=value``:
``temperature:t=0.7``, ``lookahead:k=4,horizon=12,bootstrap=0``,
``repair:iterations=1500,time_limit=8``, ``ensemble:runs=full_s0|full_s1``.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.distributions import Categorical

from algo.generalist_observation import GeneralistEnv, GraphObservation, collate
from algo.generalist_ppo import (
    ReferenceTable,
    SuiteProblem,
    SuiteReward,
    episode_summary,
    tally,
)


# ---------------------------------------------------------------------------
# Environment cloning
# ---------------------------------------------------------------------------


def clone_environment(env):
    """A stepable deep copy of a live environment for simulated lookahead.

    The transport graph is immutable at episode time and dominates the
    simulator's size, so it is shared (as ``_certify_instance`` does); every
    piece of mutable state is copied, so stepping the clone never touches the
    real episode. The clone keeps its own copy of the random state: rolling it
    forward consumes the same draws the real episode would see next.
    """
    memo = {id(env.transport_graph): env.transport_graph}
    clone = copy.deepcopy(env, memo)
    clone.verbose = clone.enable_plotting = False
    return clone


# ---------------------------------------------------------------------------
# Batched actors (drop into evaluate_policy_suite)
# ---------------------------------------------------------------------------


class _WrappedActor:
    """Duck-typed actor: config, eval/train and act, plus latency counters."""

    def __init__(self, policy):
        self.policy = policy
        self.decisions = 0
        self.act_seconds = 0.0

    @property
    def config(self):
        return self.policy.config

    @property
    def training(self) -> bool:
        return self.policy.training

    def eval(self):
        self.policy.eval()
        return self

    def train(self, mode: bool = True):
        self.policy.train(mode)
        return self

    def latency(self) -> dict:
        """Mean seconds per decision over every ``act`` call so far."""
        mean = self.act_seconds / self.decisions if self.decisions else 0.0
        return {"basis": "batched", "decisions": self.decisions, "mean_seconds": mean}


# ``now / horizon`` in the context row (algo/generalist_observation.py,
# ``_context_row``); the schema test pins it so drift breaks loudly.
CONTEXT_PROGRESS_INDEX = 25


class TemperaturePolicy(_WrappedActor):
    """Sample the masked categorical at temperature ``t``; ``t = 0`` is argmax.

    With ``t_end`` set, the temperature is annealed per decision by episode
    progress (the context's ``now / horizon``): hot sampling early, when
    committing to requests matters, and near-greedy at the horizon, where
    sampled over-commitment strands trucks. ``t_end`` defaults to ``t``.

    The ``deterministic`` flag of ``act`` is ignored on purpose: the wrapper
    exists to replace greedy decoding, and ``t = 0`` recovers it exactly.
    """

    def __init__(self, policy, temperature: float, t_end: float | None = None, top_p: float = 1.0):
        super().__init__(policy)
        if temperature < 0 or not math.isfinite(temperature):
            raise ValueError("temperature must be finite and nonnegative")
        if t_end is not None and (t_end < 0 or not math.isfinite(t_end)):
            raise ValueError("t_end must be finite and nonnegative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        self.temperature = float(temperature)
        self.t_end = self.temperature if t_end is None else float(t_end)
        self.top_p = float(top_p)

    def _nucleus(self, logits):
        """Keep the smallest probability mass >= ``top_p`` per row (the greedy
        action always survives), so sampling never overrides a confident row."""
        if self.top_p >= 1.0:
            return logits
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        probabilities = sorted_logits.softmax(dim=-1)
        before = probabilities.cumsum(dim=-1) - probabilities
        dropped = before >= self.top_p  # mass before the token already covers p
        mask = torch.zeros_like(dropped).scatter(-1, order, dropped)
        return logits.masked_fill(mask, float("-inf"))

    @torch.no_grad()
    def act(self, batch, deterministic: bool = False):
        started = time.perf_counter()
        output = self.policy(batch)
        progress = batch.context[:, CONTEXT_PROGRESS_INDEX].clamp(0.0, 1.0)
        tau = self.temperature + (self.t_end - self.temperature) * progress
        greedy = output.logits.argmax(dim=-1)
        if float(tau.max()) == 0.0:
            actions = greedy
            log_probs = torch.zeros_like(actions, dtype=output.logits.dtype)
        else:
            scaled = self._nucleus(output.logits) / tau.clamp(min=1e-6).unsqueeze(-1)
            distribution = Categorical(logits=scaled)
            actions = torch.where(tau > 0, distribution.sample(), greedy)
            log_probs = distribution.log_prob(actions)
        self.act_seconds += time.perf_counter() - started
        self.decisions += int(actions.shape[0])
        return actions, log_probs, output.values


class EnsemblePolicy(_WrappedActor):
    """Average the member policies' log-probabilities over the shared candidates.

    Members must share one observation spec (same campaign, same flags); the
    first member provides ``config`` and the value estimate.
    """

    def __init__(self, policies: Sequence, temperature: float = 0.0):
        if not policies:
            raise ValueError("an ensemble needs at least one policy")
        specs = {repr(policy.config.observation_spec) for policy in policies}
        if len(specs) != 1:
            raise ValueError("ensemble members must share one observation spec")
        super().__init__(policies[0])
        self.policies = list(policies)
        self.temperature = float(temperature)

    @property
    def training(self) -> bool:
        return any(policy.training for policy in self.policies)

    def eval(self):
        for policy in self.policies:
            policy.eval()
        return self

    def train(self, mode: bool = True):
        for policy in self.policies:
            policy.train(mode)
        return self

    @torch.no_grad()
    def act(self, batch, deterministic: bool = False):
        started = time.perf_counter()
        outputs = [policy(batch) for policy in self.policies]
        stacked = torch.stack(
            [torch.log_softmax(output.logits, dim=-1) for output in outputs]
        )
        logits = stacked.mean(dim=0)
        logits = logits.masked_fill(~outputs[0].feasible_mask, float("-inf"))
        if self.temperature == 0.0:
            actions = logits.argmax(dim=-1)
            log_probs = torch.zeros_like(actions, dtype=logits.dtype)
        else:
            distribution = Categorical(logits=logits / self.temperature)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
        self.act_seconds += time.perf_counter() - started
        self.decisions += int(actions.shape[0])
        return actions, log_probs, outputs[0].values


# ---------------------------------------------------------------------------
# Sequential controllers (one live environment per episode)
# ---------------------------------------------------------------------------


class LookaheadController:
    """Choose among the policy's top-``k`` candidates by simulated rollouts.

    At each decision the environment is cloned once per considered candidate;
    the candidate is applied and the policy then acts greedily for up to
    ``horizon`` further decisions (or to the end of the episode). Candidates
    are compared by the reward accumulated from the decision point under the
    episode's own :class:`SuiteReward` scale — the shared history cancels —
    plus the terminal reward when a rollout finishes, plus (optionally) the
    policy's value estimate at the cut-off. Ties keep the policy's preference.
    """

    def __init__(
        self,
        policy,
        reward: SuiteReward,
        scale,
        *,
        k: int = 4,
        horizon: int = 12,
        bootstrap: bool = False,
        device: str = "cpu",
    ):
        if k < 1 or horizon < 0:
            raise ValueError("k must be >= 1 and horizon >= 0")
        self.policy = policy
        self.reward = reward
        self.scale = scale
        self.k = int(k)
        self.horizon = int(horizon)
        self.bootstrap = bool(bootstrap)
        self.device = device

    @torch.no_grad()
    def _positions(self, observation: GraphObservation) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.policy(collate([observation], self.device))
        logits = output.logits[0]
        order = torch.argsort(logits, descending=True)
        keep = int(torch.isfinite(logits).sum().item())
        return order[: max(1, min(self.k, keep))], logits

    @torch.no_grad()
    def _rollout(self, env, observation: GraphObservation | None) -> float:
        """Reward accumulated on ``env`` for up to ``horizon`` more decisions."""
        total, before, done = 0.0, None, observation is None
        for _ in range(self.horizon):
            if done:
                break
            output = self.policy(collate([observation], self.device))
            position = int(output.logits[0].argmax().item())
            action = int(observation.action_index[position])
            observation, _, terminated, truncated, info = env.step(action)
            after = tally(info, env.dynamic_requests)
            if before is not None:
                total += self.reward.step(before, after, self.scale)
            before = after
            done = bool(terminated or truncated)
        if done:
            total += self.reward.terminal(episode_summary(env), self.scale)
        elif self.bootstrap and observation is not None:
            output = self.policy(collate([observation], self.device))
            total += float(output.values[0].item())
        return total

    @torch.no_grad()
    def __call__(self, env, observation: GraphObservation) -> int:
        positions, _ = self._positions(observation)
        if len(positions) == 1 or self.horizon == 0:
            return int(observation.action_index[int(positions[0].item())])
        best_position, best_value = None, None
        for rank, position in enumerate(positions.tolist()):
            action = int(observation.action_index[position])
            shadow = clone_environment(env)
            next_observation, _, terminated, truncated, info = shadow.step(action)
            value = self.reward.step((0.0, 0.0, 0.0, 0.0), tally(info, shadow.dynamic_requests), self.scale)
            if terminated or truncated:
                value += self.reward.terminal(episode_summary(shadow), self.scale)
            else:
                remaining = self._rollout(shadow, next_observation)
                # _rollout measures increments from its own first tally, so the
                # first-step increment above and its tail never double-count.
                value += remaining
            if best_value is None or value > best_value + 1e-12:
                best_position, best_value = position, value
        return int(observation.action_index[best_position])


class PlannerRepairController:
    """Plan 9.6: the policy as the long-haul planner's warm start.

    At reset, the base policy runs one full episode on a clone of the fresh
    environment; the per-truck order in which it visits customers seeds the
    ALNS as one more restart candidate (:class:`_WarmStartPlanPolicy`). The
    repaired plan is then executed by the standard ``lh_plan`` machinery
    (continuation search for charging and rests). Static problems only.
    """

    def __init__(self, policy, *, device: str = "cpu", max_steps: int = 5000, **settings):
        self.policy = policy
        self.device = device
        self.max_steps = int(max_steps)
        self.settings = settings
        self._executor = None

    @torch.no_grad()
    def _warm_routes(self, env) -> dict[int, list[int]]:
        """Per-truck customer nodes, in the base policy's own visit order."""
        shadow = clone_environment(env)
        observation = shadow._get_observation()
        slot_nodes = [int(task.node_id) for task in shadow.joint_instance.tasks]
        offset = len(shadow.charging_nodes)
        routes: dict[int, list[int]] = {}
        seen: set[int] = set()
        for _ in range(self.max_steps):
            output = self.policy(collate([observation], self.device))
            position = int(output.logits[0].argmax().item())
            action = int(observation.action_index[position])
            truck = shadow.active_truck_id
            slot = action - offset
            if truck is not None and 0 <= slot < len(slot_nodes):
                node = slot_nodes[slot]
                if node not in seen:
                    routes.setdefault(int(truck), []).append(node)
                    seen.add(node)
            observation, _, terminated, truncated, _ = shadow.step(action)
            if terminated or truncated:
                break
        return routes

    def __call__(self, env, observation: GraphObservation | None = None) -> int:
        if getattr(env, "dynamic_requests", False):
            raise ValueError("planner repair is defined for static problems only")
        if self._executor is None:
            self._executor = _WarmStartPlanPolicy(self._warm_routes(env), **self.settings)
        return self._executor(env)


def _warm_start_executor_settings(**overrides):
    from EVRoutingEnv.baselines.optimization.longhaul.policy import LonghaulSettings
    from EVRoutingEnv.baselines.optimization.longhaul.planner import PlannerSettings

    planner_keys = {"iterations", "time_limit_seconds", "seed"}
    planner = {key: overrides.pop(key) for key in list(overrides) if key in planner_keys}
    if "time_limit" in overrides:
        planner["time_limit_seconds"] = overrides.pop("time_limit")
    settings = LonghaulSettings(name="policy_repair", **overrides)
    if planner:
        from dataclasses import replace

        settings = replace(settings, planner=replace(settings.planner, **planner))
    return settings


class _WarmStartPlanPolicy:
    """``lh_plan`` whose initial ALNS also considers the base policy's routes."""

    def __init__(self, warm_routes: dict[int, list[int]], **overrides):
        from EVRoutingEnv.baselines.optimization.longhaul.policy import LonghaulPlanPolicy

        self.warm_routes = {int(truck): [int(node) for node in nodes] for truck, nodes in warm_routes.items()}
        settings = _warm_start_executor_settings(**overrides)

        warm = self.warm_routes

        class _Policy(LonghaulPlanPolicy):
            def _plan_initial(self, env) -> None:  # noqa: D401 - parent contract
                super()._plan_initial(env)
                _consider_warm_start(self, env, warm)

        self._policy = _Policy(settings)
        self.stats = self._policy.stats

    def __call__(self, env, observation=None, info=None) -> int:
        action = self._policy(env)
        self.stats = self._policy.stats
        return int(action)


def _consider_warm_start(plan_policy, env, warm_routes: dict[int, list[int]]) -> None:
    """One more ALNS candidate, seeded with the policy's routes; kept only if
    its planned cost beats the parent's and its exact durations fit."""
    import random

    from EVRoutingEnv.baselines.optimization.longhaul.planner import LonghaulPlanner
    from EVRoutingEnv.baselines.optimization.longhaul.policy import TruckStart

    plan = plan_policy.stats.get("plan") or {}
    if plan.get("plan_cached"):
        return
    settings = plan_policy.settings
    starts = [
        TruckStart(truck_id=int(truck.truck_id), node=plan_policy.depot, time=0.0,
                   battery=float(truck.current_battery),
                   payload=float(truck.remaining_payload if truck.remaining_payload is not None
                                 else truck.payload_capacity))
        for truck in env.trucks
    ]
    slots = list(range(len(plan_policy.tasks)))
    problem = plan_policy._problem(env, starts, slots)
    planner = LonghaulPlanner(
        plan_policy.net, problem,
        plan_policy._planner_settings(999_983, duration_margin=plan_policy.duration_margin),
    )
    slot_of_node = plan_policy.slot_of_node
    by_position = []
    seen: set[int] = set()
    for truck in env.trucks:
        route = []
        for node in warm_routes.get(int(truck.truck_id), []):
            slot = slot_of_node.get(int(node))
            if slot is not None and slot not in seen:
                route.append(slot)
                seen.add(slot)
        by_position.append(route)
    missing = [slot for slot in slots if slot not in seen]
    if missing:
        planner._repair(by_position, missing, random.Random(0), regret_k=2)
    routes, cost, info = planner.solve(initial=by_position)
    previous = plan.get("planned_hours")
    if previous is not None and cost >= previous - 1e-6:
        plan_policy.stats.setdefault("warm_start", {"improved": False, "planned_hours": float(cost)})
        return
    exact = plan_policy._exact_durations(env, routes, problem)
    cheap = [planner.route_cost(t, r)[0] for t, r in enumerate(routes)]
    limit = env.max_time - settings.execution_margin
    if any(routes[t] and exact[t] > limit for t in range(len(routes))):
        plan_policy.stats.setdefault("warm_start", {"improved": False, "reason": "exact durations exceed the horizon"})
        return
    plan_policy.routes = {
        int(env.trucks[t].truck_id): [int(plan_policy.tasks[c].node_id) for c in route]
        for t, route in enumerate(routes)
    }
    plan_policy.stats["plan"] = {
        "planned_hours": float(cost),
        "exact_hours": float(sum(value for value in exact if math.isfinite(value))),
        "exact_route_hours": [round(float(x), 3) for x in exact],
        "planned_route_hours": [round(float(x), 3) for x in cheap],
        "routes": {str(k): [int(n) for n in v] for k, v in plan_policy.routes.items()},
        "duration_margin": float(plan_policy.duration_margin),
        "iterations": int(info["iterations"]),
    }
    plan_policy.stats["warm_start"] = {
        "improved": True,
        "planned_hours": float(cost),
        "previous_hours": float(previous) if previous is not None else None,
    }


# ---------------------------------------------------------------------------
# Sequential evaluation (same record schema as the campaign evaluator)
# ---------------------------------------------------------------------------


def run_wrapped_episode(
    env: GeneralistEnv,
    seed: int,
    controller: Callable,
    problem: SuiteProblem,
    reward: SuiteReward,
    scale,
) -> dict:
    """One episode driven by ``controller(env, observation)``; the campaign
    record plus decision-latency fields."""
    observation, _ = env.reset(seed=int(seed))
    seconds: list[float] = []
    while observation is not None:
        started = time.perf_counter()
        action = int(controller(env, observation))
        seconds.append(time.perf_counter() - started)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    summary = episode_summary(env)
    ordered = sorted(seconds)
    index = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1) if ordered else 0
    return {
        "problem": problem.name,
        "seed": int(seed),
        "score": reward.score(summary, scale),
        **summary,
        "decisions": len(seconds),
        "decision_seconds_mean": float(sum(seconds) / len(seconds)) if seconds else 0.0,
        "decision_seconds_p95": float(ordered[index]) if ordered else 0.0,
    }


def _wrapped_job(arguments):
    problem, spec, seeds, reward, references, factory = arguments
    env = GeneralistEnv(problem.config, spec)
    rows = []
    try:
        for seed in seeds:
            scale, _ = references.scale(problem.name, int(seed), reward.reference)
            controller = factory(problem, scale)
            rows.append(run_wrapped_episode(env, seed, controller, problem, reward, scale))
    finally:
        env.close()
    return rows


def evaluate_wrapped_problem(
    problem: SuiteProblem,
    seeds: Sequence[int],
    spec,
    reward: SuiteReward,
    references: ReferenceTable,
    factory: Callable[[SuiteProblem, object], Callable],
    workers: int = 1,
) -> list[dict]:
    """Every seed once through ``factory(problem, scale)``'s controller.

    ``factory`` runs inside forked workers, so it must be picklable-by-fork
    (plain function or functools.partial over module-level objects); the
    policy it closes over stays on the CPU.
    """
    import multiprocessing as mp

    seeds = [int(seed) for seed in seeds]
    chunks = [seeds[index::max(1, workers)] for index in range(max(1, workers))]
    jobs = [(problem, spec, chunk, reward, references, factory) for chunk in chunks if chunk]
    if workers <= 1:
        parts = [_wrapped_job(job) for job in jobs]
    else:
        with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            parts = pool.map(_wrapped_job, jobs)
    order = {seed: position for position, seed in enumerate(seeds)}
    return sorted((row for part in parts for row in part), key=lambda row: order[row["seed"]])


# ---------------------------------------------------------------------------
# Wrapper specifications
# ---------------------------------------------------------------------------


BATCHED = ("temperature", "ensemble")
SEQUENTIAL = ("lookahead", "repair")


@dataclass(frozen=True)
class WrapperSpec:
    """A parsed ``name:key=value,...`` wrapper specification."""

    name: str
    options: dict

    @property
    def batched(self) -> bool:
        return self.name in BATCHED


def parse_wrapper(text: str) -> WrapperSpec:
    name, _, tail = text.partition(":")
    name = name.strip()
    if name not in (*BATCHED, *SEQUENTIAL):
        raise ValueError(f"unknown wrapper {name!r}; expected one of {(*BATCHED, *SEQUENTIAL)}")
    options: dict = {}
    for part in filter(None, (piece.strip() for piece in tail.split(","))):
        key, equals, value = part.partition("=")
        if not equals:
            raise ValueError(f"wrapper option {part!r} is not key=value")
        key = key.strip()
        value = value.strip()
        if key == "runs":
            options[key] = [item for item in value.split("|") if item]
        else:
            try:
                options[key] = int(value)
            except ValueError:
                try:
                    options[key] = float(value)
                except ValueError:
                    options[key] = value
        if key in ("t", "temperature"):
            options["temperature"] = float(options.pop(key))
    return WrapperSpec(name=name, options=options)
