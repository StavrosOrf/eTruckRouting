"""Masked PPO for one GraphPPO policy trained on several problem families.

Differences from :mod:`algo.canonical_ppo`, which trains on one configuration:

* **Mixed problems.** Sub-environment ``i`` runs problem ``i mod P`` with its
  own disjoint slice of that problem's train seeds, so every rollout holds the
  same number of decisions from each problem.
* **Variable observations.** Workers return :class:`GraphObservation` rows and
  the trainer pads each batch to its own largest graph. Actions are candidate
  positions; the worker maps them back to environment actions.
* **Reference-normalized reward** (plan section 4.3). Hours are divided by a
  reference policy's operating hours and services by its task count, so
  returns are comparable across problems of very different size. With
  ``reference="scenario"`` the reference is the same seed's run (a paired
  per-scenario baseline the policy never observes); with ``"problem"`` it is
  the problem's median, the control arm.
* **Checkpoints** hold the policy, optimizer, normalizer, return scale, RNG
  streams and seed-stream positions, but not live simulators (a Texas
  simulator pickles to about 75 MB). On resume, episodes in flight restart from
  their own seed, so a resumed run is reproducible but not bit-identical to an
  uninterrupted one.
* **Imitation** beyond the warm start (:mod:`algo.generalist_imitation`), off
  by default: a retained cross-entropy term on labelled states in every PPO
  minibatch, and DAgger rounds whose sub-environments live in the same forked
  workers.
* **Seed-paired baseline**, off by default: with ``group_size`` K > 1 every
  train seed runs in K sub-environments at once (common random numbers), and
  ``advantage_mode="group"`` replaces the policy's GAE advantage by the
  episode's return minus its partners' mean (:func:`group_advantages`); the
  critic still trains on its GAE targets. ``policy_gradient_weight=0`` turns
  the surrogate off for the imitation-only control. ``rollout_on_cpu`` keeps
  the collated rollout on the CPU so long rollouts fit the GPU, and
  ``accumulate_minibatches`` takes one gradient step per K minibatches, the
  step of a minibatch K times the size.
"""

from __future__ import annotations

import gc
import math
import multiprocessing.connection
import random
import statistics
import os
import shutil
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from algo.canonical_ppo import PPOConfig, RunningReturnScale
from algo.generalist_imitation import (
    STORE_META,
    DaggerSlots,
    ImitationConfig,
    ImitationData,
    ObservationStore,
    ShardWriter,
    gather_observations,
    store_fingerprint,
    write_store,
    check_experts,
    dagger_slots,
    episode_counts,
    expert_reference_rows,
    replay_dagger_archive,
    summarize_dagger_round,
    trainable_archive,
)
from algo.generalist_observation import (
    GeneralistEnv,
    GraphObservation,
    ObservationSpec,
    budget_chunks,
    collate,
    node_counts,
    padded_pairs,
)
from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig
from EVRoutingEnv.evaluation.metrics import extract_operational_metrics


# Observations per transfer when observations are folded into the input statistics.
OBSERVE_CHUNK = 4096
# Padded node pairs one forward pass may hold: rows x (trucks + tasks + chargers)^2, each node
# type padded to the widest member of the chunk. The encoder keeps a hidden vector per pair and
# layer, about 3 kB per pair with gradients (suite_v5c: two 256-row graphs of ~3M pairs peaked at
# 18 GB), so this is about 12 GB on the device. The suite's 256-row minibatches (10 trucks, at
# most 64 chargers) stay one chunk and run as they always did. On the broad set a 256-row
# minibatch padded to 100 trucks x 294 open requests x 64 chargers asked for 10.5 GiB per layer
# and relation and killed the suite_v6 pilot on a 40 GB A100 (2026-11-10).
PAIR_BUDGET = 4_000_000
# The input statistics hold no activations: their chunks may be this many times larger.
OBSERVE_FACTOR = 8
# Padding a chunk may add this fraction to the pairs its rows really have (see ``budget_chunks``):
# on broad-set minibatches 0.75 gives about 16 chunks holding 1.7 times the real pairs, where
# filling chunks by size alone gives 5 chunks holding 2.8 times.
CHUNK_SLACK = 0.75
_OutOfMemory = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


class PairBudget:
    """The padded node pairs one forward pass may hold; halved when the device runs out of memory."""

    def __init__(self, pairs: int | None = PAIR_BUDGET):
        if pairs is not None and pairs < 1:
            raise ValueError("the pair budget must be positive")
        self.pairs = pairs
        self.reductions = 0

    def chunks(self, counts, factor: int = 1, slack: float | None = CHUNK_SLACK) -> list[np.ndarray]:
        return budget_chunks(counts, None if self.pairs is None else self.pairs * factor, slack)

    def reduce(self, failed: int) -> None:
        """A chunk of ``failed`` padded pairs exhausted the device: half the budget from now on."""
        self.pairs = max(1, (failed if self.pairs is None else min(self.pairs, failed)) // 2)
        self.reductions += 1
        print(f"[memory] a chunk of {failed} padded node pairs exhausted the device; "
              f"pair budget now {self.pairs} (reduction {self.reductions})", flush=True)


def run_chunks(budget: PairBudget, counts, run: Callable[[np.ndarray], object], factor: int = 1):
    """Yield ``(rows, run(rows))`` over budgeted chunks of the rows described by ``counts``.

    ``run`` must leave no partial effect when it raises: a chunk that exhausts
    the device's memory is split in two and retried, and the budget is halved
    for every later call, so one misjudged size costs a retry, not the run.
    """
    counts = np.asarray(counts, dtype=np.int64).reshape(-1, 3)
    queue = deque(budget.chunks(counts, factor))
    while queue:
        rows = queue.popleft()
        failed = False
        try:
            result = run(rows)
        except _OutOfMemory:
            if len(rows) < 2:
                raise
            failed = True  # handled outside the handler, so the failed graph is released first
        if failed:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            budget.reduce(padded_pairs(counts[rows]))
            ordered = rows[np.argsort(-counts[rows].sum(axis=1), kind="stable")]
            half = len(ordered) // 2
            queue.appendleft(ordered[half:])
            queue.appendleft(ordered[:half])
            continue
        yield rows, result


def act_in_chunks(
    policy: GeneralistActorCritic,
    observations: Sequence[GraphObservation],
    device: str | torch.device,
    deterministic: bool = False,
    budget: PairBudget | None = None,
    uniform: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``policy.act`` on ``observations`` in chunks the device can hold; results in the order given.

    ``uniform`` holds one number in [0, 1) per observation: its sampled position then depends on
    the observation and that number only, not on the chunk or the call it is acted on in.
    """
    budget = budget if budget is not None else PairBudget()
    device = torch.device(device)
    size = len(observations)
    positions = torch.zeros(size, dtype=torch.long, device=device)
    logprobs = torch.zeros(size, device=device)
    values = torch.zeros(size, device=device)
    noise = None if uniform is None else torch.as_tensor(np.asarray(uniform, dtype=np.float64), device=device)

    def run(rows):
        batch = collate([observations[int(row)] for row in rows], device)
        chosen = None if noise is None else noise[torch.as_tensor(np.asarray(rows), device=device)]
        return policy.act(batch, deterministic, chosen)

    for rows, (chosen, logprob, value) in run_chunks(budget, node_counts(observations), run):
        index = torch.as_tensor(rows, device=device)
        positions[index], logprobs[index], values[index] = chosen, logprob, value
    return positions, logprobs, values


def values_in_chunks(
    policy: GeneralistActorCritic,
    observations: Sequence[GraphObservation],
    device: str | torch.device,
    budget: PairBudget | None = None,
) -> torch.Tensor:
    """``policy.predict_values`` on ``observations`` in chunks the device can hold."""
    budget = budget if budget is not None else PairBudget()
    device = torch.device(device)
    values = torch.zeros(len(observations), device=device)

    def run(rows):
        with torch.no_grad():
            return policy.predict_values(collate([observations[int(row)] for row in rows], device))

    for rows, value in run_chunks(budget, node_counts(observations), run):
        values[torch.as_tensor(rows, device=device)] = value
    return values

# Terminations that mean the fleet broke down, not that the horizon ended.
FAILURE_REASONS = frozenset({
    "no_feasible_action",
    "payload_capacity_deadlock",
    "no_events_with_unserved_customers",
    "invalid_action_budget_exhausted",
    "all_trucks_failed",
    "fleet_stopped_with_unserved_customers",
})
TEACHERS = ("auto", "greedy_heuristic", "greedy_dispatch", "wait_at_depot", "kullman_myopic")


@dataclass(frozen=True)
class SuiteProblem:
    """One problem of the suite: a configuration and its reference policy."""

    name: str
    config: str
    teacher: str = "auto"

    def __post_init__(self) -> None:
        if self.teacher not in TEACHERS:
            raise ValueError(f"teacher must be one of {TEACHERS}, got {self.teacher!r}")


def build_teacher(name: str, env) -> Callable:
    """The reference policy: greedy dispatch for requests, else the greedy navigator."""
    from EVRoutingEnv.baselines.heuristics.canonical_baselines import (
        GreedyHeuristicPolicy,
    )
    from EVRoutingEnv.baselines.heuristics.component_baselines import (
        GreedyDispatchPolicy,
        KullmanMyopicPolicy,
        WaitAtDepotPolicy,
    )

    if name == "auto":
        name = "greedy_dispatch" if getattr(env, "dynamic_requests", False) else "greedy_heuristic"
    return {
        "greedy_heuristic": GreedyHeuristicPolicy,
        "greedy_dispatch": GreedyDispatchPolicy,
        "wait_at_depot": WaitAtDepotPolicy,
        "kullman_myopic": KullmanMyopicPolicy,
    }[name]()


# ---------------------------------------------------------------------------
# Outcomes, references and the reward
# ---------------------------------------------------------------------------


def tally(info: dict, dynamic: bool) -> tuple[float, float, float, float]:
    """Running (operating hours, on-time services, late services, lost requests)."""
    hours = float(info["operational_metrics"]["total_operating_time"])
    if dynamic:
        requests = info["request_metrics"]
        return (
            hours,
            float(requests["requests_served_on_time"]),
            float(requests["requests_served_late"]),
            float(requests["requests_lost"]),
        )
    return hours, float(info["operational_metrics"]["customers_served"]), 0.0, 0.0


def episode_summary(env) -> dict:
    """Simulator-side outcome of a finished episode; never read from reward."""
    metrics = extract_operational_metrics(env).as_dict()
    dynamic = bool(getattr(env, "dynamic_requests", False))
    reason = env.termination_reason
    summary = {
        "dynamic": dynamic,
        "success": bool(metrics["success"]),
        "termination_reason": reason,
        "failed": reason in FAILURE_REASONS or any(truck.failed for truck in env.trucks),
        "hours": float(metrics["total_operating_time"]),
        "travel_hours": float(metrics["total_travel_time"]),
        "charging_sessions": int(metrics["charging_sessions"]),
        "completed_fraction": float(metrics["completed_fraction"]),
        "steps": int(env.episode_steps),
    }
    if dynamic:
        requests = env.request_engine.metrics(env.global_clock)
        summary.update(
            tasks=int(requests["requests_released"]),
            served=int(requests["requests_served_on_time"]),
            late=int(requests["requests_served_late"]),
            lost=int(requests["requests_lost"]),
            unserved=int(requests["requests_unserved"]),
        )
    else:
        summary.update(
            tasks=int(metrics["customers_total"]),
            served=int(metrics["customers_served"]),
            late=0,
            lost=0,
            unserved=int(metrics["customers_total"]) - int(metrics["customers_served"]),
        )
    return summary


@dataclass(frozen=True)
class ReferenceScale:
    """Per-episode units of the reward: reference hours and reference tasks."""

    hours: float
    tasks: float


class ReferenceTable:
    """Reference-policy outcomes per (problem, seed), from the references stage."""

    def __init__(self, rows: dict[str, dict[int, dict]]):
        self.rows = {problem: {int(seed): row for seed, row in values.items()} for problem, values in rows.items()}
        self.medians = {}
        for problem, values in self.rows.items():
            if values:
                self.medians[problem] = ReferenceScale(
                    hours=max(statistics.median(row["hours"] for row in values.values()), 1e-6),
                    tasks=max(statistics.median(_reference_tasks(row) for row in values.values()), 1.0),
                )

    @classmethod
    def load(cls, directory: str | Path, problems: Sequence[SuiteProblem]) -> ReferenceTable:
        """``<directory>/<problem>.json`` (merged references), or else a
        ``<directory>/<problem>/`` folder of expert outcomes (the BKS layout:
        ``train_demonstrations.json`` and ``validation.json``)."""
        import json

        rows = {}
        for problem in problems:
            path = Path(directory) / f"{problem.name}.json"
            folder = Path(directory) / problem.name
            if path.exists():
                rows[problem.name] = json.loads(path.read_text())
            elif folder.is_dir():
                rows[problem.name] = expert_reference_rows(folder)
            else:
                rows[problem.name] = {}
        return cls(rows)

    def scale(self, problem: str, seed: int, mode: str) -> tuple[ReferenceScale, bool]:
        """The episode's units, and whether the problem median stood in."""
        if problem not in self.medians:
            raise KeyError(f"no reference outcomes for problem {problem!r}")
        row = self.rows[problem].get(int(seed))
        if mode == "problem" or row is None:
            return self.medians[problem], mode == "scenario"
        return ReferenceScale(hours=max(float(row["hours"]), 1e-6), tasks=_reference_tasks(row)), False


def _reference_tasks(row: dict) -> float:
    """Tasks a reference episode is measured against.

    Static: every customer. Dynamic: the reference's on-time services, since
    the released count includes requests no policy could have served.
    """
    return float(max(1, row["served"] if row["dynamic"] else row["tasks"]))


@dataclass(frozen=True)
class SuiteReward:
    """Reference-normalized training signal shared by every problem.

    Per step: ``-dhours / H_ref + (w_s dserved - w_late dlate - w_lost dlost) / N_ref``.
    At the end: ``+1`` (so the hour terms sum to ``(H_ref - H) / H_ref``, plan
    4.3), a failure penalty, and for static problems a success bonus or an
    incompletion penalty; requests still unserved at the horizon count as lost.
    The undiscounted return equals :meth:`score` of the episode's summary.
    """

    reference: str = "problem"
    service_weight: float = 2.0
    lost_weight: float = 1.0
    late_weight: float = 0.5
    success_bonus: float = 1.0
    failure_penalty: float = 2.0
    incompletion_penalty: float = 2.0

    def __post_init__(self) -> None:
        if self.reference not in ("problem", "scenario"):
            raise ValueError("reference must be 'problem' or 'scenario'")
        weights = [value for name, value in asdict(self).items() if name != "reference"]
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("reward weights must be finite and nonnegative")

    def step(self, before, after, scale: ReferenceScale) -> float:
        hours, served, late, lost = (now - then for now, then in zip(after, before, strict=True))
        return -hours / scale.hours + (
            self.service_weight * served - self.late_weight * late - self.lost_weight * lost
        ) / scale.tasks

    def terminal(self, summary: dict, scale: ReferenceScale) -> float:
        value = 1.0
        if summary["failed"]:
            value -= self.failure_penalty
        if summary["dynamic"]:
            value -= self.lost_weight * summary["unserved"] / scale.tasks
        elif summary["success"]:
            value += self.success_bonus
        else:
            value -= self.incompletion_penalty * (1.0 - summary["completed_fraction"])
        return value

    def score(self, summary: dict, scale: ReferenceScale) -> float:
        final = (summary["hours"], summary["served"], summary["late"], summary["lost"])
        return self.step((0.0, 0.0, 0.0, 0.0), final, scale) + self.terminal(summary, scale)


# ---------------------------------------------------------------------------
# Vectorized environments
# ---------------------------------------------------------------------------


class _SuiteSlots:
    """The sub-environments one worker owns, with auto-reset and reward."""

    def __init__(self, problems, slots, spec, reward, references, cycle, dagger=()):
        self.problems = list(problems)
        # DAgger sub-environments (simulator and expert only), idle between rounds.
        self.dagger = DaggerSlots(self.problems, dagger, spec)
        self.problem_of = [problem for problem, _ in slots]
        self.seeds = [list(seeds) for _, seeds in slots]
        self.envs = [
            GeneralistEnv(self.problems[problem].config, spec, problem=self.problems[problem].name)
            for problem in self.problem_of
        ]
        self.position = [0] * len(self.envs)
        self.reward = reward
        self.references = references
        self.cycle = cycle
        self.current: list[tuple | None] = [None] * len(self.envs)
        self.fallbacks = 0

    def _start(self, index: int) -> GraphObservation | None:
        stream = self.seeds[index]
        if not stream or (not self.cycle and self.position[index] >= len(stream)):
            self.current[index] = None
            return None
        seed = int(stream[self.position[index] % len(stream)])
        self.position[index] += 1
        env = self.envs[index]
        observation, _ = env.reset(seed=seed)
        name = self.problems[self.problem_of[index]].name
        scale, fallback = self.references.scale(name, seed, self.reward.reference)
        self.fallbacks += int(fallback)
        # Hours spent before the first decision count too, so the return equals
        # SuiteReward.score of the finished episode exactly.
        self.current[index] = (seed, scale, (0.0, 0.0, 0.0, 0.0))
        return observation

    def reset(self) -> list[GraphObservation | None]:
        return [self._start(index) for index in range(len(self.envs))]

    def step(self, actions) -> list[tuple]:
        results = []
        for index, action in enumerate(actions):
            if action is None or self.current[index] is None:
                results.append((None, 0.0, False, None))
                continue
            env = self.envs[index]
            seed, scale, before = self.current[index]
            observation, _, terminated, truncated, info = env.step(int(action))
            after = tally(info, env.dynamic_requests)
            reward = self.reward.step(before, after, scale)
            done = bool(terminated or truncated)
            record = None
            if done:
                summary = episode_summary(env)
                reward += self.reward.terminal(summary, scale)
                record = {
                    "problem": self.problems[self.problem_of[index]].name,
                    "seed": seed,
                    "score": self.reward.score(summary, scale),
                    **summary,
                }
                observation = self._start(index)
            else:
                self.current[index] = (seed, scale, after)
            results.append((observation, reward, done, record))
        return results

    def dagger_call(self, method, values, shared):
        return getattr(self.dagger, method)(*([] if values is None else [values]), *shared)

    def close(self) -> None:
        for env in self.envs:
            env.close()
        self.dagger.close()


def _slots_worker(connection, arguments) -> None:
    """Forked worker: simulator and numpy only, never torch or CUDA."""
    slots = _SuiteSlots(*arguments)
    try:
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                connection.send(slots.reset())
            elif command == "step":
                connection.send(slots.step(payload))
            elif command == "positions":
                connection.send((list(slots.position), slots.fallbacks))
            elif command == "set_positions":
                slots.position = list(payload)
                connection.send(True)
            elif command == "dagger":
                connection.send(slots.dagger_call(*payload))
            else:
                raise ValueError(f"unknown command {command!r}")
    finally:
        slots.close()
        connection.close()


def pack_slots(weights: Sequence[float] | None, count: int, workers: int, loads: list[float] | None = None) -> list[list[int]]:
    """Deal ``count`` slots to ``workers``: in listing order, or by ``weights`` (expected seconds of work).

    With weights the heaviest slot goes first, each to the worker with the least work so far
    (``loads`` carries work dealt earlier and is updated). Workers advance at their own pace
    (:class:`Pace`), so a phase lasts as long as its busiest worker: dealing the 50 training
    slots of the 25 lighter broad problems by cost shortens a rollout from 7.5 to 4.8 minutes.
    Which worker runs a slot changes nothing else.
    """
    if weights is None:
        return [list(range(worker, count, workers)) for worker in range(workers)]
    if len(weights) != count:
        raise ValueError("one weight per slot is required")
    loads = loads if loads is not None else [0.0] * workers
    members: list[list[int]] = [[] for _ in range(workers)]
    for index in sorted(range(count), key=lambda index: (-float(weights[index]), index)):
        worker = min(range(workers), key=lambda worker: (loads[worker], len(members[worker]), worker))
        members[worker].append(index)
        loads[worker] += max(float(weights[index]), 1e-9)
    return [sorted(group) for group in members]


def slot_weights(costs: dict | None, names: Sequence[str], key: str, scale: Sequence[float] | None = None) -> list[float] | None:
    """Expected work of slots running the problems ``names``: ``costs[name][key]`` (the median for unknown names)."""
    if not costs:
        return None
    known = sorted(float(row[key]) for row in costs.values() if key in row)
    if not known:
        return None
    fallback = known[len(known) // 2]
    weights = [float(costs.get(name, {}).get(key, fallback)) for name in names]
    return weights if scale is None else [weight * float(factor) for weight, factor in zip(weights, scale, strict=True)]


class Pace:
    """Advances the slots of a :class:`SuiteVecEnv` group by group.

    Every worker's slots form a group that steps as soon as its own answer is back, so a slow
    environment holds up its worker and no other. Stepping all slots in lockstep costs the
    slowest environment at every step: with plan-conditioned observations one re-plan takes
    seconds (suite_v6: a 512-step rollout of 50 environments measured 17 min in lockstep, 7 min
    worker by worker). In-process environments, and ``together=True``, form one group of all
    slots, which is the lockstep.
    """

    def __init__(self, envs: SuiteVecEnv, *, dagger: bool = False, together: bool = False):
        self.envs = envs
        self.dagger = dagger
        size = envs.num_dagger if dagger else envs.num_envs
        self.alone = envs._local is None and not together
        if self.alone:
            self.groups = [list(span) for span in (envs._dagger_spans if dagger else envs._spans)]
        else:
            self.groups = [list(range(size))]
        self._answers: dict[int, list] = {}

    def send(self, group: int, command: str, values: Sequence, shared: Sequence = ()) -> None:
        """``command`` for the slots of ``group`` (``step``, or a ``DaggerSlots`` method); one value per slot."""
        values = list(values)
        if self.alone:
            payload = (command, values, tuple(shared)) if self.dagger else values
            self.envs._connections[group].send(("dagger" if self.dagger else command, payload))
        elif self.dagger:
            self._answers[group] = self.envs.dagger(command, values, shared)
        else:
            self._answers[group] = self.envs._scatter(command, values)

    def receive(self, pending: set[int]) -> dict[int, list]:
        """The answers that are in, by group; blocks until one of ``pending`` has answered."""
        if not self.alone:
            return {group: self._answers.pop(group) for group in sorted(pending)}
        connections = {self.envs._connections[group]: group for group in pending}
        ready = multiprocessing.connection.wait(list(connections))
        return {connections[connection]: connection.recv() for connection in ready}


class SuiteVecEnv:
    """Sub-environments over several problems, in-process or in forked workers.

    ``slots`` lists ``(problem index, seed stream)`` per sub-environment. With
    ``cycle`` the streams repeat (training); without it a sub-environment
    falls idle after its last seed and reports ``None`` (evaluation).
    ``dagger_slots`` (:class:`~algo.generalist_imitation.DaggerSlot`) are
    spread over the same workers and driven by :meth:`dagger`.
    """

    def __init__(
        self, problems, slots, spec, reward, references, *, workers=1, cycle=True, dagger_slots=(),
        weights=None, dagger_weights=None,
    ):
        import multiprocessing as mp

        self.num_envs = len(slots)
        if self.num_envs == 0:
            raise ValueError("at least one sub-environment is required")
        dagger_slots = list(dagger_slots or ())
        self.num_dagger = len(dagger_slots)
        workers = max(1, min(int(workers), self.num_envs))
        self._local = None
        self._connections, self._processes, self._spans, self._dagger_spans = [], [], [], []
        if workers == 1:
            self._local = _SuiteSlots(problems, slots, spec, reward, references, cycle, dagger_slots)
            return
        context = mp.get_context("fork")
        # A round gives its episodes to the first slots of each problem (episode_counts). Dealt
        # out in listing order, the slots of a one-episode round all sat on every fourth worker
        # (suite_v6 pilot: 39 episodes on 4 of 16 workers, every step the sum of ten planners).
        # So the first slot of every problem is dealt first, then the second, and so on.
        seen: dict[int, int] = {}
        rank = []
        for slot in dagger_slots:
            rank.append(seen.get(slot.problem, 0))
            seen[slot.problem] = rank[-1] + 1
        dagger_order = sorted(range(self.num_dagger), key=lambda index: (rank[index], index))
        spans = pack_slots(weights, self.num_envs, workers)
        if dagger_weights is None:
            dagger_spans = [dagger_order[worker::workers] for worker in range(workers)]
        else:
            # By cost within each round of slots (first slots of every problem, then second, ...).
            dagger_spans, loads = [[] for _ in range(workers)], [0.0] * workers
            for level in sorted(set(rank)):
                chosen = [index for index in dagger_order if rank[index] == level]
                dealt = pack_slots([dagger_weights[index] for index in chosen], len(chosen), workers, loads)
                for worker, group in enumerate(dealt):
                    dagger_spans[worker].extend(chosen[position] for position in group)
            dagger_spans = [sorted(group) for group in dagger_spans]
        # Keep the children's collections off the parent's objects (for
        # example replayed demonstrations), so their pages stay shared.
        gc.freeze()
        try:
            for worker in range(workers):
                members = spans[worker]
                dagger_members = dagger_spans[worker]
                parent, child = context.Pipe()
                process = context.Process(
                    target=_slots_worker,
                    args=(child, (
                        problems, [slots[i] for i in members], spec, reward, references, cycle,
                        [dagger_slots[i] for i in dagger_members],
                    )),
                    daemon=True,
                )
                process.start()
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
                self._spans.append(members)
                self._dagger_spans.append(dagger_members)
        finally:
            gc.unfreeze()

    def _scatter(self, command, payload=None) -> list:
        if self._local is not None:
            return getattr(self._local, command)(*([] if payload is None else [payload]))
        for connection, members in zip(self._connections, self._spans, strict=True):
            connection.send((command, None if payload is None else [payload[i] for i in members]))
        merged = [None] * self.num_envs
        for connection, members in zip(self._connections, self._spans, strict=True):
            for index, value in zip(members, connection.recv(), strict=True):
                merged[index] = value
        return merged

    def reset(self) -> list[GraphObservation | None]:
        return self._scatter("reset")

    def step(self, actions: Sequence[int | None]):
        results = self._scatter("step", list(actions))
        observations = [result[0] for result in results]
        rewards = np.asarray([result[1] for result in results], dtype=np.float32)
        dones = np.asarray([result[2] for result in results], dtype=bool)
        return observations, rewards, dones, [result[3] for result in results]

    def positions(self) -> tuple[list[int], int]:
        if self._local is not None:
            return list(self._local.position), self._local.fallbacks
        merged, fallbacks = [0] * self.num_envs, 0
        for connection, members in zip(self._connections, self._spans, strict=True):
            connection.send(("positions", None))
            values, count = connection.recv()
            fallbacks += count
            for index, value in zip(members, values, strict=True):
                merged[index] = value
        return merged, fallbacks

    def set_positions(self, positions: Sequence[int]) -> None:
        if self._local is not None:
            self._local.position = list(positions)
            return
        for connection, members in zip(self._connections, self._spans, strict=True):
            connection.send(("set_positions", [int(positions[i]) for i in members]))
        for connection in self._connections:
            connection.recv()

    def dagger(self, method: str, values: Sequence | None = None, shared: Sequence = ()) -> list:
        """Call ``DaggerSlots.method`` everywhere; ``values`` is per DAgger slot, ``shared`` is broadcast."""
        if not self.num_dagger:
            return []
        if self._local is not None:
            return self._local.dagger_call(method, None if values is None else list(values), tuple(shared))
        for connection, members in zip(self._connections, self._dagger_spans, strict=True):
            connection.send(("dagger", (method, None if values is None else [values[i] for i in members], tuple(shared))))
        merged: list = [None] * self.num_dagger
        for connection, members in zip(self._connections, self._dagger_spans, strict=True):
            for index, value in zip(members, connection.recv(), strict=True):
                merged[index] = value
        return merged

    def close(self) -> None:
        if self._local is not None:
            self._local.close()
            return
        for connection, process in zip(self._connections, self._processes, strict=True):
            try:
                connection.send(("close", None))
                connection.close()
            except (BrokenPipeError, OSError):
                pass
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()


def balanced_slots(
    num_problems: int, num_envs: int, seeds: Sequence[Sequence[int]], group_size: int = 1,
) -> list[tuple[int, list[int]]]:
    """Sub-environment ``i`` runs problem ``i mod P`` on a disjoint seed slice.

    With ``group_size`` K > 1 each slice is run by K consecutive sub-environments
    at once (common random numbers): slots ``Kj .. Kj+K-1`` run problem ``j mod P``.
    """
    if num_envs % (num_problems * group_size):
        raise ValueError(
            f"num_envs ({num_envs}) must be a multiple of the {num_problems} problems"
            + (f" times the group size {group_size}" if group_size > 1 else "")
        )
    width = num_envs // (num_problems * group_size)
    if any(len(stream) < width for stream in seeds):
        raise ValueError(f"every problem needs at least {width} train seeds for its {width} sub-environments")
    slots = []
    for slot in range(num_envs // group_size):
        problem = slot % num_problems
        slots.extend((problem, list(seeds[problem][slot // num_problems :: width])) for _ in range(group_size))
    return slots


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


class GeneralistPPO:
    """PPO over the candidate sets of mixed-problem graph observations."""

    def __init__(
        self,
        policy_config: GeneralistPolicyConfig,
        ppo_config: PPOConfig,
        problems: Sequence[SuiteProblem],
        references: ReferenceTable,
        reward: SuiteReward,
        train_seeds: Sequence[Sequence[int]],
        device: str | torch.device = "cpu",
        rollout_workers: int = 1,
        imitation: ImitationConfig | None = None,
        pair_budget: int | None = PAIR_BUDGET,
        slot_costs: dict | None = None,
    ):
        """``slot_costs`` maps a problem to its measured ``step`` (seconds per decision) and
        ``dagger`` (seconds per DAgger episode); slots are then dealt to the workers by cost
        (:func:`pack_slots`). It changes how long a phase takes, not what it computes."""
        self.budget = PairBudget(pair_budget)
        self.policy_config = policy_config
        self.config = ppo_config
        self.problems = list(problems)
        self.reward = reward
        self.device = torch.device(device)
        self.imitation = imitation
        torch.manual_seed(ppo_config.seed)
        np.random.seed(ppo_config.seed)
        random.seed(ppo_config.seed)
        slots_for_dagger = []
        if imitation is not None and imitation.dagger:
            check_experts(imitation, self.problems)
            slots_for_dagger = dagger_slots(self.problems, train_seeds, imitation)
        # Every worker, DAgger sub-environments included, is forked here,
        # before the policy touches the device.
        slots = balanced_slots(len(self.problems), ppo_config.num_envs, train_seeds, ppo_config.group_size)
        self.envs = SuiteVecEnv(
            self.problems,
            slots,
            policy_config.observation_spec,
            reward,
            references,
            workers=rollout_workers,
            dagger_slots=slots_for_dagger,
            weights=slot_weights(slot_costs, [self.problems[problem].name for problem, _ in slots], "step"),
            dagger_weights=slot_weights(
                slot_costs, [self.problems[slot.problem].name for slot in slots_for_dagger], "dagger",
            ),
        )
        self.dagger_slots = slots_for_dagger
        self.imitation_data = ImitationData(imitation.dagger_max_states if imitation is not None else 1)
        # A private stream, so retained imitation never shifts the PPO shuffles.
        self._bc_rng = np.random.default_rng([ppo_config.seed, 0xB1C])
        self.dagger_rounds = 0
        self.policy = GeneralistActorCritic(policy_config).to(self.device)
        self._parameters = list(self.policy.parameters())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=ppo_config.learning_rate, eps=1e-5)
        self.reward_scale = (
            RunningReturnScale(ppo_config.num_envs, ppo_config.gamma) if ppo_config.normalize_reward else None
        )
        self.history: list[dict] = []
        self.episodes = {problem.name: deque(maxlen=50) for problem in self.problems}
        self.advantage_groups = None
        if ppo_config.advantage_by_problem or ppo_config.advantage_mode == "group":
            self.advantage_groups = torch.as_tensor([problem for problem, _ in slots], device=self.device)
        self.total_steps = 0
        self.completed_updates = 0
        self._observations: list[GraphObservation] | None = None
        # Whether each sub-environment's current observation opens an episode.
        self._fresh = np.ones(ppo_config.num_envs, dtype=bool)

    # -- interaction -------------------------------------------------------

    def learn(self, progress_callback: Callable[[dict], bool] | None = None) -> list[dict]:
        config = self.config
        if self._observations is None:
            self._observations = self.envs.reset()
            self._observe(self._observations)
        started = time.perf_counter()
        for update in range(self.completed_updates, config.num_updates):
            if config.anneal_learning_rate:
                horizon = config.schedule_timesteps or config.total_timesteps
                fraction = max(0.0, 1.0 - self.total_steps / horizon)
                for group in self.optimizer.param_groups:
                    group["lr"] = fraction * config.learning_rate
            dagger = {}
            if self.envs.num_dagger and update % self.imitation.dagger_every == 0:
                dagger = self.dagger_round()
            rollout_started = time.perf_counter()
            rollout = self._collect_rollout()
            optimize_started = time.perf_counter()
            statistics_ = self._optimize(rollout)
            optimized = time.perf_counter()
            # Collection and every epoch above used the same statistics.
            self._observe(rollout["observations"], rollout["counts"])
            self.completed_updates = update + 1
            record = {
                "update": update + 1,
                "num_updates": config.num_updates,
                "timesteps": self.total_steps,
                "elapsed_seconds": time.perf_counter() - started,
                "rollout_seconds": optimize_started - rollout_started,
                "optimize_seconds": optimized - optimize_started,
                **statistics_,
                **rollout["statistics"],
                **self.episode_statistics(),
                **dagger,
            }
            self.history.append(record)
            if progress_callback is not None and progress_callback(record):
                break
        return self.history

    def episode_statistics(self) -> dict:
        record = {}
        scores = []
        for name, window in self.episodes.items():
            if not window:
                continue
            record[f"{name}/score"] = float(np.mean([row["score"] for row in window]))
            record[f"{name}/served_fraction"] = float(
                np.mean([row["served"] / max(1, row["tasks"]) for row in window])
            )
            record[f"{name}/success"] = float(np.mean([row["success"] for row in window]))
            record[f"{name}/hours"] = float(np.mean([row["hours"] for row in window]))
            scores.append(record[f"{name}/score"])
        record["train_score"] = float(np.mean(scores)) if scores else float("nan")
        return record

    def _collect_rollout(self) -> dict:
        config = self.config
        steps, num_envs = config.rollout_steps, config.num_envs
        # One number per decision, drawn before the rollout: an environment's sampled actions do
        # not depend on which environments share its forward pass, so collecting worker by worker
        # (see ``Pace``) and in lockstep give the same rollout.
        noise = np.random.default_rng([config.seed, 0x524F4C4C, self.completed_updates]).random((steps, num_envs))
        grid: list[list[GraphObservation | None]] = [[None] * num_envs for _ in range(steps)]
        actions = np.zeros((steps, num_envs), dtype=np.int64)
        logprobs = np.zeros((steps, num_envs), dtype=np.float32)
        values = np.zeros((steps, num_envs), dtype=np.float32)
        rewards = np.zeros((steps, num_envs), dtype=np.float32)
        dones = np.zeros((steps, num_envs), dtype=bool)
        finished: list[tuple[int, int, dict]] = []
        current = list(self._observations)
        pace = Pace(self.envs, together=config.step_together)
        clock = [0] * len(pace.groups)
        waiting = {group for group, members in enumerate(pace.groups) if members}
        pending: set[int] = set()
        while waiting or pending:
            if waiting:
                order = sorted(waiting)
                members = [index for group in order for index in pace.groups[group]]
                at = [clock[group] for group in order for _ in pace.groups[group]]
                observations = [current[index] for index in members]
                chosen, logprob, value = act_in_chunks(
                    self.policy, observations, self.device, budget=self.budget, uniform=noise[at, members],
                )
                chosen = chosen.cpu().numpy()
                actions[at, members] = chosen
                logprobs[at, members] = logprob.cpu().numpy()
                values[at, members] = value.cpu().numpy()
                env_actions = {}
                for index, step, observation, position in zip(members, at, observations, chosen, strict=True):
                    grid[step][index] = observation
                    env_actions[index] = int(observation.action_index[position])
                for group in order:
                    pace.send(group, "step", [env_actions[index] for index in pace.groups[group]])
                pending |= waiting
                waiting = set()
            for group, results in pace.receive(pending).items():
                pending.discard(group)
                step = clock[group]
                for index, (observation, reward, done, record) in zip(pace.groups[group], results, strict=True):
                    current[index], rewards[step, index], dones[step, index] = observation, reward, done
                    if record is not None:
                        finished.append((step, index, record))
                clock[group] += 1
                self.total_steps += len(pace.groups[group])
                if clock[group] < steps:
                    waiting.add(group)
        # Everything below is in step order, whatever order the workers answered in.
        finished.sort(key=lambda item: item[:2])
        for _, _, record in finished:
            self.episodes[record["problem"]].append(record)
        if self.reward_scale is not None:
            for step in range(steps):
                self.reward_scale.update(rewards[step], dones[step])
                rewards[step] = self.reward_scale.normalize(rewards[step])
        self._observations = current
        stored = [observation for row in grid for observation in row]
        actions, logprobs, values, rewards = (
            torch.as_tensor(array, device=self.device) for array in (actions, logprobs, values, rewards)
        )
        step_dones = dones[-1].copy()
        dones = torch.as_tensor(dones.astype(np.float32), device=self.device)
        last_value = values_in_chunks(self.policy, current, self.device, self.budget)
        advantages = generalized_advantages(rewards, values, dones, last_value, config.gamma, config.gae_lambda)
        returns = advantages + values
        statistics_: dict = {}
        if config.advantage_mode == "group":
            advantages, statistics_ = group_advantages(advantages, rewards, finished, self._fresh, config.group_size)
        self._fresh = step_dones
        if self.advantage_groups is not None:
            advantages = normalized_by_group(advantages, self.advantage_groups)
        # The rollout stays a list (``rollout_on_cpu`` or not): one batch over it would be padded
        # to its widest observation, 77 GB for a broad-set update. Minibatches are collated from
        # the list in chunks the device can hold.
        return {
            "observations": stored,
            "counts": node_counts(stored),
            "actions": actions.reshape(-1),
            "logprobs": logprobs.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "values": values.reshape(-1),
            "statistics": statistics_,
        }

    def _optimize(self, rollout: dict) -> dict:
        config = self.config
        stored, counts = rollout["observations"], rollout["counts"]
        indices = np.arange(config.batch_size)
        losses = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
        bc_weight = self.bc_weight()
        if bc_weight > 0:
            losses.update(bc_loss=[], bc_accuracy=[])
        accumulate = config.accumulate_minibatches
        for _ in range(config.epochs):
            np.random.shuffle(indices)
            starts = [s for s in range(0, config.batch_size, config.minibatch_size) if min(config.minibatch_size, config.batch_size - s) >= 2]
            for number, start in enumerate(starts):
                rows = indices[start : start + config.minibatch_size]
                size = len(rows)
                batch_index = torch.as_tensor(rows, device=self.device)
                actions = rollout["actions"][batch_index]
                old_logprobs = rollout["logprobs"][batch_index]
                advantages = rollout["advantages"][batch_index]
                if config.normalize_advantage and self.advantage_groups is None:
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
                returns = rollout["returns"][batch_index]
                old_values = rollout["values"][batch_index]

                def run(chunk, rows=rows, size=size, actions=actions, old_logprobs=old_logprobs,
                        advantages=advantages, returns=returns, old_values=old_values):
                    # The minibatch loss is a mean over its rows, so the sums of its chunks, each
                    # divided by the minibatch size, add up to the same gradient.
                    at = torch.as_tensor(chunk, device=self.device)
                    minibatch = collate([stored[int(rows[position])] for position in chunk], self.device)
                    logprob, entropy, value = self.policy.evaluate_actions(minibatch, actions[at])
                    log_ratio = logprob - old_logprobs[at]
                    ratio = log_ratio.exp()
                    advantage = advantages[at]
                    policy_sum = -torch.min(
                        ratio * advantage,
                        torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * advantage,
                    ).sum()
                    target = returns[at]
                    if config.value_clip_range > 0.0:
                        old = old_values[at]
                        clipped = old + torch.clamp(value - old, -config.value_clip_range, config.value_clip_range)
                        value_sum = 0.5 * torch.max((value - target).pow(2), (clipped - target).pow(2)).sum()
                    else:
                        value_sum = 0.5 * (value - target).pow(2).sum()
                    entropy_sum = entropy.sum()
                    loss = (
                        config.policy_gradient_weight * policy_sum
                        + config.value_coefficient * value_sum
                        - config.entropy_coefficient * entropy_sum
                    ) / (size * accumulate)
                    gradients = torch.autograd.grad(loss, self._parameters, allow_unused=True)
                    return gradients, {
                        "policy": float(policy_sum.item()), "value": float(value_sum.item()),
                        "entropy": float(entropy_sum.item()),
                        "kl": float(((ratio - 1.0) - log_ratio).sum().item()),
                    }

                # With accumulation the optimizer steps once per ``accumulate`` minibatches on
                # the mean of their losses, the gradient of one minibatch that size.
                if number % accumulate == 0:
                    self.optimizer.zero_grad()
                sums = self._accumulate(counts[rows], run)
                if bc_weight > 0:
                    bc_loss, bc_accuracy = self.imitation_gradient(bc_weight / accumulate)
                    losses["bc_loss"].append(bc_loss)
                    losses["bc_accuracy"].append(bc_accuracy)
                if (number + 1) % accumulate == 0 or number + 1 == len(starts):
                    nn.utils.clip_grad_norm_(self.policy.parameters(), config.max_grad_norm)
                    self.optimizer.step()
                losses["approx_kl"].append(sums["kl"] / size)
                losses["policy_loss"].append(sums["policy"] / size)
                losses["value_loss"].append(sums["value"] / size)
                losses["entropy"].append(sums["entropy"] / size)
            if config.target_kl is not None and losses["approx_kl"] and np.mean(losses["approx_kl"][-8:]) > config.target_kl:
                break
        statistics_ = {key: float(np.mean(values)) if values else 0.0 for key, values in losses.items()}
        if self.imitation is not None:
            statistics_.update(bc_coefficient=bc_weight, imitation_states=len(self.imitation_data))
        if self.budget.reductions:
            statistics_.update(pair_budget=self.budget.pairs, pair_budget_reductions=self.budget.reductions)
        return statistics_

    def _accumulate(self, counts, run: Callable[[np.ndarray], tuple]) -> dict:
        """Add the gradients of ``run(chunk) -> (gradients, sums)`` over budgeted chunks to ``.grad``; total the sums.

        ``run`` returns gradients from ``torch.autograd.grad``, which touches no
        ``.grad``, so a chunk that ran out of memory can be split and retried.
        """
        totals: dict[str, float] = {}
        for _, (gradients, sums) in run_chunks(self.budget, counts, run):
            for parameter, gradient in zip(self._parameters, gradients, strict=True):
                if gradient is None:
                    continue
                if parameter.grad is None:
                    parameter.grad = gradient
                else:
                    parameter.grad.add_(gradient)
            for key, value in sums.items():
                totals[key] = totals.get(key, 0.0) + value
        return totals

    # -- retained imitation and DAgger ---------------------------------------

    def bc_weight(self) -> float:
        """The current retained-imitation coefficient; zero without labelled states."""
        if self.imitation is None or not len(self.imitation_data):
            return 0.0
        return self.imitation.bc_weight(self.total_steps)

    def set_demonstrations(self, observations: list[GraphObservation], labels: list[int]) -> None:
        """Keep replayed demonstrations (on the CPU) for the retained-imitation term."""
        self.imitation_data.set_demonstrations(observations, labels)

    def imitation_gradient(self, scale: float = 1.0) -> tuple[float, float]:
        """Add ``scale`` times the cross-entropy gradient of one sampled minibatch of labelled states to ``.grad``.

        Returns the minibatch's mean loss and accuracy. The labelled states are
        scored in chunks the device can hold, like every other minibatch.
        """
        observations, labels = self.imitation_data.sample(self._bc_rng, self.imitation.bc_batch_size)
        size = len(labels)
        sums = self._labelled_sums(observations, labels, scale / size)
        return sums["loss"] / size, sums["correct"] / size

    def _labelled_sums(self, observations: Sequence[GraphObservation], labels, scale: float | None) -> dict:
        """Cross-entropy sum and correct count of labelled observations, scored in chunks the device can hold.

        With ``scale`` the gradient of ``scale`` times the cross-entropy sum is
        added to ``.grad``; with ``None`` nothing is differentiated.
        """
        target = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=self.device)

        def run(chunk):
            at = torch.as_tensor(chunk, device=self.device)
            logits = self.policy(collate([observations[int(position)] for position in chunk], self.device)).logits
            loss = F.cross_entropy(logits, target[at], reduction="sum")
            sums = {"loss": float(loss.item()), "correct": float((logits.argmax(dim=-1) == target[at]).sum().item())}
            if scale is None:
                return (), sums
            return torch.autograd.grad(loss * scale, self._parameters, allow_unused=True), sums

        counts = node_counts(observations)
        if scale is not None:
            return self._accumulate(counts, run)
        sums = {"loss": 0.0, "correct": 0.0}
        with torch.no_grad():
            for _, (_, part) in run_chunks(self.budget, counts, run):
                sums = {key: sums[key] + part[key] for key in sums}
        return sums

    def _observe(self, observations: Sequence[GraphObservation], counts=None) -> None:
        """Fold observations into the input statistics, in chunks the device can hold."""
        if not len(observations):
            return
        counts = node_counts(observations) if counts is None else counts
        for chunk in self.budget.chunks(counts, OBSERVE_FACTOR, slack=None):
            for start in range(0, len(chunk), OBSERVE_CHUNK):
                rows = chunk[start : start + OBSERVE_CHUNK]
                self.policy.observe(collate(gather_observations(observations, rows), self.device))

    def dagger_round(self, beta: float | None = None, observe: bool = False) -> dict:
        """One DAgger round in the workers' DAgger sub-environments.

        The learner acts here (sampled unless ``dagger_deterministic``); each
        worker queries the expert at every visited state and executes the
        expert's action with probability ``beta``. Labelled states join the
        retained-imitation set; with ``observe`` they also update the input
        normalizer (rounds before any rollout).
        """
        if not self.envs.num_dagger:
            return {}
        imitation = self.imitation
        beta = imitation.beta(self.total_steps) if beta is None else float(beta)
        started = time.perf_counter()
        slots = self.envs.num_dagger
        counts = episode_counts(self.dagger_slots, imitation.dagger_episodes)
        pace = Pace(self.envs, dagger=True, together=self.config.step_together)
        live: list = [None] * slots
        states: list[tuple[list, list]] = [([], []) for _ in range(slots)]
        # The learner's sampling noise, one stream per slot: a slot's episode is the same whichever
        # slots share its forward passes.
        draws = [np.random.default_rng([self.config.seed, 0x44414747, self.dagger_rounds, index]) for index in range(slots)]
        agreement: dict[str, list[int]] = {}
        episodes: list[tuple[int, dict, list, list]] = []
        records, added = [], []
        begun: set[int] = set()
        waiting: set[int] = set()
        pending: set[int] = set()
        for group, members in enumerate(pace.groups):
            if members:
                pace.send(group, "start", [counts[index] for index in members], (beta, (self.config.seed, self.dagger_rounds)))
                pending.add(group)
        was_training = self.policy.training
        self.policy.eval()
        try:
            while waiting or pending:
                if waiting:
                    order = sorted(waiting)
                    indices = [index for group in order for index in pace.groups[group] if live[index] is not None]
                    observations = [live[index][0] for index in indices]
                    uniform = None if imitation.dagger_deterministic else [draws[index].random() for index in indices]
                    positions, _, _ = act_in_chunks(
                        self.policy, observations, self.device, imitation.dagger_deterministic, self.budget, uniform=uniform,
                    )
                    actions: dict[int, int] = {}
                    for index, observation, position in zip(indices, observations, positions.cpu().numpy(), strict=True):
                        actions[index] = int(observation.action_index[position])
                        label = live[index][1]
                        if label is not None:
                            states[index][0].append(observation)
                            states[index][1].append(label)
                            name = self.problems[self.dagger_slots[index].problem].name
                            tally_ = agreement.setdefault(name, [0, 0])
                            tally_[0] += int(position == label)
                            tally_[1] += 1
                    for group in order:
                        pace.send(group, "step", [actions.get(index) for index in pace.groups[group]])
                    pending |= waiting
                    waiting = set()
                for group, results in pace.receive(pending).items():
                    pending.discard(group)
                    for index, result in zip(pace.groups[group], results, strict=True):
                        if group not in begun:
                            live[index] = result  # the answer to "start": the first (observation, label), or None
                        elif result is None:
                            live[index] = None
                        else:
                            live[index], record = result
                            if record is not None:
                                episodes.append((index, record, *states[index]))
                                states[index] = ([], [])
                    begun.add(group)
                    if any(live[index] is not None for index in pace.groups[group]):
                        waiting.add(group)
        finally:
            self.policy.train(was_training)
        # By slot, each slot's episodes in the order it played them, whatever order they ended in.
        episodes.sort(key=lambda item: item[0])
        for _, record, observed, labels in episodes:
            row = {key: record[key] for key in ("problem", "seed", "actions", "labels")}
            self.imitation_data.add_episode(row, observed, labels)
            added.extend(observed)
            records.append(record)
        if observe:
            self._observe(added)
        self.dagger_rounds += 1
        summary = summarize_dagger_round(records, agreement, beta)
        summary.update({
            "dagger/round": self.dagger_rounds,
            "dagger/seconds": time.perf_counter() - started,
            "dagger/set_states": self.imitation_data.dagger_states,
            "dagger/evicted_episodes": self.imitation_data.evicted_episodes,
        })
        return summary

    # -- imitation ---------------------------------------------------------

    def pretrain(
        self,
        observations: Sequence[GraphObservation],
        labels: Sequence[int],
        *,
        epochs: int,
        learning_rate: float,
        batch_size: int = 256,
        holdout_fraction: float = 0.1,
        progress: Callable[[dict], None] | None = None,
        observe: bool = True,
    ) -> list[dict]:
        """Behaviour cloning on teacher candidate positions, with a holdout.

        ``observe=False`` leaves the input normalizer alone (states it has seen).
        """
        if len(observations) != len(labels) or not observations:
            raise ValueError("demonstrations need one label per observation")
        if observe:
            self._observe(observations)
        generator = np.random.default_rng(self.config.seed)
        order = generator.permutation(len(observations))
        cut = int(len(order) * (1.0 - holdout_fraction)) if len(order) > 10 else len(order)
        train, holdout = order[:cut], order[cut:]
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        history = []

        def score(rows, train_step: bool) -> tuple[float, float]:
            """Mean cross-entropy and accuracy of one minibatch; with ``train_step`` its gradient lands in ``.grad``."""
            size = len(rows)
            sums = self._labelled_sums(
                gather_observations(observations, rows), [int(labels[row]) for row in rows],
                1.0 / size if train_step else None,
            )
            return sums["loss"] / size, sums["correct"] / size

        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            generator.shuffle(train)
            totals = []
            for start in range(0, len(train), batch_size):
                optimizer.zero_grad()
                loss, _ = score(train[start : start + batch_size], True)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                optimizer.step()
                totals.append(loss)
            record = {"epoch": epoch, "train_loss": float(np.mean(totals))}
            if len(holdout):
                parts = [score(holdout[start : start + batch_size], False) for start in range(0, len(holdout), batch_size)]
                record["holdout_loss"] = float(np.mean([loss for loss, _ in parts]))
                record["holdout_accuracy"] = float(np.mean([accuracy for _, accuracy in parts]))
            record["seconds"] = time.perf_counter() - epoch_started
            history.append(record)
            if progress is not None:
                progress(record)
        return history

    # -- persistence -------------------------------------------------------

    def save(self, directory: str | Path, prefix: str = "policy") -> None:
        self.policy.save(directory, prefix=prefix)

    def save_training(self, path: str | Path, metadata: dict | None = None) -> None:
        """Atomic trusted checkpoint without live simulators (see module docstring)."""
        from algo.checkpoint_io import save_torch_atomic

        positions, fallbacks = self.envs.positions()
        # Episodes in flight restart from their own seed on resume.
        restart = [max(0, value - 1) for value in positions]
        save_torch_atomic(
            {
                "version": 1,
                "policy_config": asdict(self.policy_config),
                "ppo_config": asdict(self.config),
                "reward": asdict(self.reward),
                "problems": [asdict(problem) for problem in self.problems],
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "reward_scale": self.reward_scale,
                "history": self.history,
                "episodes": self.episodes,
                "total_steps": self.total_steps,
                "completed_updates": self.completed_updates,
                "seed_positions": restart,
                "reference_fallbacks": fallbacks,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
                "metadata": metadata or {},
                **({"imitation": self._imitation_state()} if self.imitation is not None else {}),
            },
            path,
            compress=True,
        )

    def _imitation_state(self) -> dict:
        """Compact: DAgger states are stored as replayable episodes, not observations."""
        return {
            "config": asdict(self.imitation),
            "archive": self.imitation_data.archive(),
            "evicted_episodes": self.imitation_data.evicted_episodes,
            "dagger_positions": self.envs.dagger("positions"),
            "dagger_rounds": self.dagger_rounds,
            "bc_rng": self._bc_rng.bit_generator.state,
        }

    def load_training(self, path: str | Path, dagger_replay: list | None = None) -> dict:
        """Restore a training checkpoint.

        The DAgger set is rebuilt from its archive: pass ``dagger_replay``
        from :func:`~algo.generalist_imitation.replay_dagger_archive`, run
        before the policy touched CUDA; otherwise it is replayed here, in
        process and without forking.
        """
        from algo.checkpoint_io import load_trusted_training

        state = load_trusted_training(path, map_location="cpu")
        if state["policy_config"] != asdict(self.policy_config):
            raise ValueError("checkpoint policy configuration differs")
        if state["reward"] != asdict(self.reward):
            raise ValueError("checkpoint reward differs")
        if state["problems"] != [asdict(problem) for problem in self.problems]:
            raise ValueError("checkpoint problem suite differs")
        for key in ("num_envs", "rollout_steps", "gamma", "gae_lambda", "schedule_timesteps"):
            if state["ppo_config"][key] != asdict(self.config)[key]:
                raise ValueError(f"checkpoint PPO configuration differs: {key}")
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.reward_scale = state["reward_scale"]
        self.history = state["history"]
        self.episodes = state["episodes"]
        self.total_steps = state["total_steps"]
        self.completed_updates = state["completed_updates"]
        self.envs.set_positions(state["seed_positions"])
        self._observations = self.envs.reset()
        self._fresh[:] = True
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        saved = state.get("imitation")
        if saved is not None and self.imitation is not None:
            self._bc_rng.bit_generator.state = saved["bc_rng"]
            self.dagger_rounds = int(saved["dagger_rounds"])
            positions = saved["dagger_positions"]
            if self.envs.num_dagger and len(positions) == self.envs.num_dagger:
                self.envs.dagger("set_positions", positions)
            archive = trainable_archive(saved, self.problems)
            if archive:
                if dagger_replay is None:
                    dagger_replay = replay_dagger_archive(self.problems, self.policy_config.observation_spec, archive)
                self.imitation_data.restore(archive, dagger_replay)
            self.imitation_data.evicted_episodes = int(saved.get("evicted_episodes", 0))
        return state["metadata"]

    def close(self) -> None:
        self.envs.close()


def normalized_by_group(advantages: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Advantages ``(steps, envs)`` normalized within each group of environments.

    Across a mixture of problems, a batch-wide scale is set by the problem with
    the largest returns and hides the small advantages of the others.
    """
    normalized = advantages.clone()
    for group in groups.unique():
        rows = groups == group
        part = advantages[:, rows]
        normalized[:, rows] = (part - part.mean()) / (part.std(unbiased=False) + 1e-8)
    return normalized


def group_advantages(
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    finished: Sequence[tuple[int, int, dict]],
    fresh: Sequence[bool],
    group_size: int,
) -> tuple[torch.Tensor, dict]:
    """Leave-one-out return baselines over ``(steps, envs)`` for the seed-paired groups.

    ``finished`` lists ``(step, env, record)`` for every episode that ended in
    the rollout and ``fresh`` marks the sub-environments whose step-0 state
    opened an episode. An episode is complete when it both started and ended
    inside the rollout; complete episodes of the same (problem, seed) form a
    group. In a group of at least ``group_size`` episodes every step of each
    gets the episode's undiscounted return (the buffer's rewards) minus the
    mean return of the group's other episodes. Every other step keeps the GAE
    advantage passed in: an episode the rollout boundary cuts, or one whose
    partner episodes have not finished, falls back. Returns the advantages and
    the counts to log.
    """
    started: list[int | None] = [0 if fresh[env] else None for env in range(rewards.shape[1])]
    episodes = []
    for step, env, record in sorted(finished, key=lambda item: item[:2]):
        if started[env] is not None:
            total = float(rewards[started[env] : step + 1, env].sum().item())
            episodes.append((env, started[env], step, (record["problem"], int(record["seed"])), total))
        started[env] = step + 1
    returns_of: dict[tuple, list[float]] = {}
    for _, _, _, key, total in episodes:
        returns_of.setdefault(key, []).append(total)
    grouped = advantages.clone()
    grouped_steps = grouped_episodes = 0
    for env, start, end, key, total in episodes:
        partners = returns_of[key]
        if len(partners) < group_size:
            continue
        grouped[start : end + 1, env] = total - (sum(partners) - total) / (len(partners) - 1)
        grouped_steps += end + 1 - start
        grouped_episodes += 1
    fallback = rewards.numel() - grouped_steps
    return grouped, {
        "group/episodes": grouped_episodes,
        "group/fallback_steps": fallback,
        "group/fallback_fraction": fallback / rewards.numel(),
    }


def generalized_advantages(rewards, values, dones, last_value, gamma: float, gae_lambda: float) -> torch.Tensor:
    """GAE over ``[steps, envs]`` buffers; ``dones[t]`` ends the episode after step ``t``."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for step in reversed(range(rewards.shape[0])):
        continues = 1.0 - dones[step]
        next_value = last_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_value * continues - values[step]
        running = delta + gamma * gae_lambda * continues * running
        advantages[step] = running
    return advantages


# ---------------------------------------------------------------------------
# Reference episodes, demonstrations and evaluation
# ---------------------------------------------------------------------------


def run_reference_episode(problem: SuiteProblem, seed: int, policy: str = "teacher", env=None) -> dict:
    """One baseline episode with its summary and the environment actions it took."""
    from EVRoutingEnv.baselines.heuristics.canonical_baselines import (
        RandomFeasiblePolicy,
    )

    env = env or GeneralistEnv(problem.config)
    if policy == "teacher":
        controller = build_teacher(problem.teacher, env)
    elif policy == "random":
        controller = RandomFeasiblePolicy(seed)
    else:
        controller = build_teacher(policy, env)
    env.reset(seed=int(seed))
    actions, started, done = [], time.perf_counter(), False
    while not done:
        try:
            action = int(controller(env))
        except (RuntimeError, ValueError):
            break
        actions.append(action)
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return {
        "problem": problem.name,
        "seed": int(seed),
        "policy": policy,
        "seconds": time.perf_counter() - started,
        "actions": actions,
        **episode_summary(env),
    }


def _reference_job(arguments):
    problem, seeds, policy = arguments
    env = GeneralistEnv(problem.config)
    try:
        return [run_reference_episode(problem, seed, policy, env) for seed in seeds]
    finally:
        env.close()


def run_reference_episodes(problem: SuiteProblem, seeds: Sequence[int], policy: str = "teacher", workers: int = 1) -> list[dict]:
    """Baseline episodes over ``seeds``, split into one chunk per worker."""
    import multiprocessing as mp

    seeds = [int(seed) for seed in seeds]
    chunks = [seeds[index::max(1, workers)] for index in range(max(1, workers))]
    jobs = [(problem, chunk, policy) for chunk in chunks if chunk]
    if workers <= 1:
        rows = [row for job in jobs for row in _reference_job(job)]
    else:
        with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            rows = [row for part in pool.map(_reference_job, jobs) for row in part]
    order = {seed: position for position, seed in enumerate(seeds)}
    return sorted(rows, key=lambda row: order[row["seed"]])


def _replay_job(arguments):
    problem, spec, episodes, shard = arguments
    env = GeneralistEnv(problem.config, spec, problem=problem.name)
    # With a shard path the observations go to disk as they are rebuilt (ObservationStore); the
    # worker then holds none of them and the parent receives only offsets.
    writer = ShardWriter(shard) if shard is not None else None
    observations, labels, misses, steps = [], [], {}, 0
    try:
        for seed, actions in episodes:
            observation, _ = env.reset(seed=int(seed))
            for index, action in enumerate(actions):
                if not 0 <= int(action) < int(env.action_space.n):
                    # The archive was recorded on another configuration of the
                    # problem (suite_v6 pilot, 2026-11-08: a held-out and a broad
                    # problem shared a name); a silent IndexError hid which one.
                    raise ValueError(
                        f"{problem.name}: demonstration seed {seed} step {index} uses action {action} outside the "
                        f"action space of {env.action_space.n}; it was recorded on another configuration of the problem"
                    )
                position = observation.candidate(action)
                steps += 1
                if position is None:
                    kind = _action_kind(env, action)
                    misses[kind] = misses.get(kind, 0) + 1
                else:
                    if writer is not None:
                        writer.append(observation)
                    else:
                        observations.append(observation)
                    labels.append(position)
                observation, _, terminated, truncated, _ = env.step(int(action))
                if terminated or truncated:
                    break
    finally:
        env.close()
    return (writer.close() if writer is not None else observations), labels, misses, steps


def _action_kind(env, action: int) -> str:
    from EVRoutingEnv.state.feasibility import joint_action_feasibility

    return joint_action_feasibility(env)[int(action)].action_kind.value


def replay_demonstrations(
    problems: Sequence[SuiteProblem],
    spec: ObservationSpec,
    episodes: Sequence[tuple[int, int, Sequence[int]]],
    workers: int = 1,
    store: str | Path | None = None,
) -> tuple[Sequence[GraphObservation], list[int], dict]:
    """Rebuild teacher demonstrations by replaying recorded actions.

    The archive stores only ``(problem, seed, actions)``: the simulator is
    deterministic given both, so observations are rebuilt with the current
    view instead of being stored. Teacher actions that fall outside the
    candidate set (a charger outside the subgraph) are counted, not imitated.
    With ``store`` the observations are written to shard files in that
    directory as they are rebuilt and an :class:`ObservationStore` over them
    is returned in place of the list, so no process ever holds the whole set.
    """
    import multiprocessing as mp

    store_dir = Path(store) if store is not None else None
    if store_dir is not None:
        store_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[int, list] = {}
    for problem, seed, actions in episodes:
        groups.setdefault(int(problem), []).append((int(seed), list(actions)))
    jobs = []
    for problem, items in groups.items():
        width = max(1, min(workers, len(items)))
        for index in range(width):
            shard = store_dir / f"shard_{len(jobs):04d}.bin" if store_dir is not None else None
            jobs.append((problems[problem], spec, items[index::width], shard))
    if workers <= 1:
        parts = [_replay_job(job) for job in jobs]
    else:
        with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            parts = pool.map(_replay_job, jobs)
    labels = [item for part in parts for item in part[1]]
    misses: dict[str, int] = {}
    for part in parts:
        for kind, count in part[2].items():
            misses[kind] = misses.get(kind, 0) + count
    steps = sum(part[3] for part in parts)
    coverage = {"steps": steps, "kept": len(labels), "misses": misses}
    if store_dir is not None:
        write_store(store_dir, [part[0] for part in parts], labels, store_fingerprint(problems, spec, episodes), coverage)
        archive = ObservationStore(store_dir)
        return archive, [int(label) for label in archive.labels], coverage
    observations = [item for part in parts for item in part[0]]
    return observations, labels, coverage


def load_or_replay_store(
    root: str | Path,
    problems: Sequence[SuiteProblem],
    spec: ObservationSpec,
    episodes: Sequence[tuple[int, int, Sequence[int]]],
    workers: int = 1,
) -> tuple[ObservationStore, list[int], dict]:
    """The replayed demonstrations of ``episodes`` under ``root``, built there when missing.

    The store lives in ``root/<fingerprint prefix>``, the fingerprint covering
    the problems' configurations, the view and the episodes, so the runs and
    resumes of a campaign share one store per distinct replay and a changed
    recipe gets its own. A run that finds the store loads it; otherwise it
    replays into a private directory and renames it into place, keeping a
    store that another run finished meanwhile.
    """
    root = Path(root)
    fingerprint = store_fingerprint(problems, spec, episodes)
    directory = root / fingerprint[:16]

    def load() -> tuple[ObservationStore, list[int], dict]:
        archive = ObservationStore(directory)
        if archive.fingerprint != fingerprint or len(archive) != archive.coverage["kept"]:
            raise SystemExit(f"{directory} is not the store of these demonstrations; remove it to rebuild")
        return archive, [int(label) for label in archive.labels], archive.coverage

    if (directory / STORE_META).exists():
        return load()
    building = root / f"{directory.name}.building-{os.getpid()}"
    if building.exists():
        shutil.rmtree(building)
    archive, _, _ = replay_demonstrations(problems, spec, episodes, workers=workers, store=building)
    archive.close()
    try:
        building.rename(directory)
    except OSError:
        if not (directory / STORE_META).exists():
            raise
        shutil.rmtree(building)  # another run finished first; its store is the same replay
    return load()


def evaluate_policy_suite(
    policy: GeneralistActorCritic,
    problems: Sequence[SuiteProblem],
    seeds: Sequence[Sequence[int]],
    references: ReferenceTable,
    reward: SuiteReward,
    *,
    device: str | torch.device = "cpu",
    workers: int = 1,
    envs_per_problem: int = 4,
    deterministic: bool = True,
    envs: SuiteVecEnv | None = None,
    budget: PairBudget | None = None,
    together: bool = False,
) -> list[dict]:
    """Run ``policy`` once on every listed seed of every problem, batched.

    ``envs`` from :func:`evaluation_envs` is rewound and reused, never closed;
    build it before the policy touches CUDA, since workers are forked. The
    live sub-environments are scored in chunks within ``budget`` (the default
    :data:`PAIR_BUDGET` when none is given). Workers advance at their own pace
    (:class:`Pace`) unless ``together``; the rows come back by problem and seed.
    """
    budget = budget if budget is not None else PairBudget()
    owned = envs is None
    if owned:
        envs = evaluation_envs(
            problems, seeds, policy.config.observation_spec, reward, references,
            workers=workers, envs_per_problem=envs_per_problem,
        )
    else:
        envs.set_positions([0] * envs.num_envs)
    rows = []
    was_training = policy.training
    policy.eval()
    try:
        observations = envs.reset()
        pace = Pace(envs, together=together)
        waiting = {
            group for group, members in enumerate(pace.groups)
            if any(observations[index] is not None for index in members)
        }
        pending: set[int] = set()
        while waiting or pending:
            if waiting:
                order = sorted(waiting)
                live = [index for group in order for index in pace.groups[group] if observations[index] is not None]
                positions, _, _ = act_in_chunks(policy, [observations[index] for index in live], device, deterministic, budget)
                actions = {
                    index: int(observations[index].action_index[position])
                    for index, position in zip(live, positions.cpu().numpy(), strict=True)
                }
                for group in order:
                    pace.send(group, "step", [actions.get(index) for index in pace.groups[group]])
                pending |= waiting
                waiting = set()
            for group, results in pace.receive(pending).items():
                pending.discard(group)
                for index, (observation, _, _, record) in zip(pace.groups[group], results, strict=True):
                    observations[index] = observation
                    if record is not None:
                        rows.append(record)
                if any(observations[index] is not None for index in pace.groups[group]):
                    waiting.add(group)
        listing = {problem.name: position for position, problem in enumerate(problems)}
        rows.sort(key=lambda row: (listing[row["problem"]], row["seed"]))
    finally:
        policy.train(was_training)
        if owned:
            envs.close()
    return rows


def evaluation_envs(problems, seeds, spec, reward, references, *, workers=1, envs_per_problem=4, costs=None) -> SuiteVecEnv:
    """Finite, rewindable sub-environments covering every listed seed once.

    ``costs`` (problem -> ``episode`` seconds) deals the slots to the workers by the work they hold.
    """
    slots = []
    for problem, stream in enumerate(seeds):
        width = max(1, min(envs_per_problem, len(stream)))
        slots.extend((problem, list(stream[index::width])) for index in range(width) if stream[index::width])
    weights = slot_weights(
        costs, [problems[problem].name for problem, _ in slots], "episode", [len(stream) for _, stream in slots],
    )
    return SuiteVecEnv(problems, slots, spec, reward, references, workers=workers, cycle=False, weights=weights)


def summarize_suite(rows: Sequence[dict], problems: Sequence[str]) -> dict:
    """Per-problem outcomes and the equal-weight suite score."""
    from EVRoutingEnv.evaluation.statistics import wilson_interval

    summary = {}
    for name in problems:
        group = [row for row in rows if row["problem"] == name]
        if not group:
            continue
        successes = sum(bool(row["success"]) for row in group)
        summary[name] = {
            "episodes": len(group),
            "score": float(np.mean([row["score"] for row in group])),
            "success_rate": successes / len(group),
            "success_wilson95": wilson_interval(successes, len(group)),
            "served_fraction": float(np.mean([row["served"] / max(1, row["tasks"]) for row in group])),
            "served": float(np.mean([row["served"] for row in group])),
            "lost": float(np.mean([row["lost"] + row["unserved"] for row in group if row["dynamic"]] or [0.0])),
            "failure_rate": float(np.mean([row["failed"] for row in group])),
            "hours": float(np.mean([row["hours"] for row in group])),
            "hours_successful": (
                float(np.mean([row["hours"] for row in group if row["success"]])) if successes else None
            ),
        }
    scores = [values["score"] for values in summary.values()]
    summary["suite_score"] = float(np.mean(scores)) if scores else float("nan")
    return summary
