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
"""

from __future__ import annotations

import gc
import math
import random
import statistics
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
    DaggerSlots,
    ImitationConfig,
    ImitationData,
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
    collate,
)
from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig
from EVRoutingEnv.evaluation.metrics import extract_operational_metrics


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
        self.envs = [GeneralistEnv(self.problems[problem].config, spec) for problem in self.problem_of]
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


class SuiteVecEnv:
    """Sub-environments over several problems, in-process or in forked workers.

    ``slots`` lists ``(problem index, seed stream)`` per sub-environment. With
    ``cycle`` the streams repeat (training); without it a sub-environment
    falls idle after its last seed and reports ``None`` (evaluation).
    ``dagger_slots`` (:class:`~algo.generalist_imitation.DaggerSlot`) are
    spread over the same workers and driven by :meth:`dagger`.
    """

    def __init__(self, problems, slots, spec, reward, references, *, workers=1, cycle=True, dagger_slots=()):
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
        # Keep the children's collections off the parent's objects (for
        # example replayed demonstrations), so their pages stay shared.
        gc.freeze()
        try:
            for worker in range(workers):
                members = list(range(worker, self.num_envs, workers))
                dagger_members = list(range(worker, self.num_dagger, workers))
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


def balanced_slots(num_problems: int, num_envs: int, seeds: Sequence[Sequence[int]]) -> list[tuple[int, list[int]]]:
    """Sub-environment ``i`` runs problem ``i mod P`` on a disjoint seed slice."""
    if num_envs % num_problems:
        raise ValueError(f"num_envs ({num_envs}) must be a multiple of the {num_problems} problems")
    width = num_envs // num_problems
    if any(len(stream) < width for stream in seeds):
        raise ValueError(f"every problem needs at least {width} train seeds for its {width} sub-environments")
    return [
        (slot % num_problems, list(seeds[slot % num_problems][slot // num_problems :: width]))
        for slot in range(num_envs)
    ]


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
    ):
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
        self.envs = SuiteVecEnv(
            self.problems,
            balanced_slots(len(self.problems), ppo_config.num_envs, train_seeds),
            policy_config.observation_spec,
            reward,
            references,
            workers=rollout_workers,
            dagger_slots=slots_for_dagger,
        )
        self.dagger_slots = slots_for_dagger
        self.imitation_data = ImitationData(imitation.dagger_max_states if imitation is not None else 1)
        # A private stream, so retained imitation never shifts the PPO shuffles.
        self._bc_rng = np.random.default_rng([ppo_config.seed, 0xB1C])
        self.dagger_rounds = 0
        self.policy = GeneralistActorCritic(policy_config).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=ppo_config.learning_rate, eps=1e-5)
        self.reward_scale = (
            RunningReturnScale(ppo_config.num_envs, ppo_config.gamma) if ppo_config.normalize_reward else None
        )
        self.history: list[dict] = []
        self.episodes = {problem.name: deque(maxlen=50) for problem in self.problems}
        self.total_steps = 0
        self.completed_updates = 0
        self._observations: list[GraphObservation] | None = None

    # -- interaction -------------------------------------------------------

    def learn(self, progress_callback: Callable[[dict], bool] | None = None) -> list[dict]:
        config = self.config
        if self._observations is None:
            self._observations = self.envs.reset()
            self.policy.observe(collate(self._observations, self.device))
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
            rollout = self._collect_rollout()
            statistics_ = self._optimize(rollout)
            # Collection and every epoch above used the same statistics.
            self.policy.observe(rollout["batch"])
            self.completed_updates = update + 1
            record = {
                "update": update + 1,
                "num_updates": config.num_updates,
                "timesteps": self.total_steps,
                "elapsed_seconds": time.perf_counter() - started,
                **statistics_,
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
        stored: list[GraphObservation] = []
        actions = torch.zeros((steps, num_envs), dtype=torch.long, device=self.device)
        logprobs = torch.zeros((steps, num_envs), device=self.device)
        values = torch.zeros((steps, num_envs), device=self.device)
        rewards = torch.zeros((steps, num_envs), device=self.device)
        dones = torch.zeros((steps, num_envs), device=self.device)
        observations = self._observations
        for step in range(steps):
            stored.extend(observations)
            with torch.no_grad():
                chosen, logprob, value = self.policy.act(collate(observations, self.device))
            actions[step], logprobs[step], values[step] = chosen, logprob, value
            positions = chosen.cpu().numpy()
            env_actions = [
                int(observation.action_index[position])
                for observation, position in zip(observations, positions, strict=True)
            ]
            observations, step_rewards, step_dones, records = self.envs.step(env_actions)
            self.total_steps += num_envs
            if self.reward_scale is not None:
                self.reward_scale.update(step_rewards, step_dones)
                step_rewards = self.reward_scale.normalize(step_rewards)
            rewards[step] = torch.as_tensor(step_rewards, device=self.device)
            dones[step] = torch.as_tensor(step_dones.astype(np.float32), device=self.device)
            for record in records:
                if record is not None:
                    self.episodes[record["problem"]].append(record)
        self._observations = observations
        with torch.no_grad():
            last_value = self.policy.predict_values(collate(observations, self.device))
        advantages = generalized_advantages(rewards, values, dones, last_value, config.gamma, config.gae_lambda)
        return {
            "batch": collate(stored, self.device),
            "actions": actions.reshape(-1),
            "logprobs": logprobs.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": (advantages + values).reshape(-1),
            "values": values.reshape(-1),
        }

    def _optimize(self, rollout: dict) -> dict:
        config = self.config
        indices = np.arange(config.batch_size)
        losses = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
        bc_weight = self.bc_weight()
        if bc_weight > 0:
            losses.update(bc_loss=[], bc_accuracy=[])
        for _ in range(config.epochs):
            np.random.shuffle(indices)
            for start in range(0, config.batch_size, config.minibatch_size):
                batch_index = torch.as_tensor(indices[start : start + config.minibatch_size], device=self.device)
                if batch_index.numel() < 2:
                    continue
                logprob, entropy, value = self.policy.evaluate_actions(
                    rollout["batch"].select(batch_index), rollout["actions"][batch_index]
                )
                log_ratio = logprob - rollout["logprobs"][batch_index]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    losses["approx_kl"].append(float(((ratio - 1.0) - log_ratio).mean().item()))
                advantages = rollout["advantages"][batch_index]
                if config.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
                policy_loss = -torch.min(
                    ratio * advantages,
                    torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * advantages,
                ).mean()
                returns = rollout["returns"][batch_index]
                if config.value_clip_range > 0.0:
                    old = rollout["values"][batch_index]
                    clipped = old + torch.clamp(value - old, -config.value_clip_range, config.value_clip_range)
                    value_loss = 0.5 * torch.max((value - returns).pow(2), (clipped - returns).pow(2)).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(value, returns)
                loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy.mean()
                if bc_weight > 0:
                    bc_loss, bc_accuracy = self.imitation_loss()
                    loss = loss + bc_weight * bc_loss
                    losses["bc_loss"].append(float(bc_loss.item()))
                    losses["bc_accuracy"].append(float(bc_accuracy.item()))
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), config.max_grad_norm)
                self.optimizer.step()
                losses["policy_loss"].append(float(policy_loss.item()))
                losses["value_loss"].append(float(value_loss.item()))
                losses["entropy"].append(float(entropy.mean().item()))
            if config.target_kl is not None and losses["approx_kl"] and np.mean(losses["approx_kl"][-8:]) > config.target_kl:
                break
        statistics_ = {key: float(np.mean(values)) if values else 0.0 for key, values in losses.items()}
        if self.imitation is not None:
            statistics_.update(bc_coefficient=bc_weight, imitation_states=len(self.imitation_data))
        return statistics_

    # -- retained imitation and DAgger ---------------------------------------

    def bc_weight(self) -> float:
        """The current retained-imitation coefficient; zero without labelled states."""
        if self.imitation is None or not len(self.imitation_data):
            return 0.0
        return self.imitation.bc_weight(self.total_steps)

    def set_demonstrations(self, observations: list[GraphObservation], labels: list[int]) -> None:
        """Keep replayed demonstrations (on the CPU) for the retained-imitation term."""
        self.imitation_data.set_demonstrations(observations, labels)

    def imitation_loss(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-entropy and accuracy on one sampled minibatch of labelled states."""
        observations, labels = self.imitation_data.sample(self._bc_rng, self.imitation.bc_batch_size)
        logits = self.policy(collate(observations, self.device)).logits
        target = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, target), (logits.argmax(dim=-1) == target).float().mean().detach()

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
        counts = episode_counts(self.dagger_slots, imitation.dagger_episodes)
        live = self.envs.dagger("start", counts, (beta, (self.config.seed, self.dagger_rounds)))
        pending: list[tuple[list, list]] = [([], []) for _ in range(self.envs.num_dagger)]
        agreement: dict[str, list[int]] = {}
        records, added = [], []
        was_training = self.policy.training
        self.policy.eval()
        try:
            while any(item is not None for item in live):
                indices = [index for index, item in enumerate(live) if item is not None]
                observations = [live[index][0] for index in indices]
                positions, _, _ = self.policy.act(collate(observations, self.device), imitation.dagger_deterministic)
                actions: list[int | None] = [None] * self.envs.num_dagger
                for index, observation, position in zip(indices, observations, positions.cpu().numpy(), strict=True):
                    actions[index] = int(observation.action_index[position])
                    label = live[index][1]
                    if label is not None:
                        pending[index][0].append(observation)
                        pending[index][1].append(label)
                        name = self.problems[self.dagger_slots[index].problem].name
                        tally_ = agreement.setdefault(name, [0, 0])
                        tally_[0] += int(position == label)
                        tally_[1] += 1
                results = self.envs.dagger("step", actions)
                live = [None] * self.envs.num_dagger
                for index, result in enumerate(results):
                    if result is None:
                        continue
                    following, record = result
                    if record is not None:
                        states, labels = pending[index]
                        pending[index] = ([], [])
                        row = {key: record[key] for key in ("problem", "seed", "actions", "labels")}
                        self.imitation_data.add_episode(row, states, labels)
                        added.extend(states)
                        records.append(record)
                    live[index] = following
        finally:
            self.policy.train(was_training)
        if observe:
            for start in range(0, len(added), 4096):
                self.policy.observe(collate(added[start : start + 4096], self.device))
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
        observations: list[GraphObservation],
        labels: list[int],
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
        for start in range(0, len(observations) if observe else 0, 4096):
            self.policy.observe(collate(observations[start : start + 4096], self.device))
        generator = np.random.default_rng(self.config.seed)
        order = generator.permutation(len(observations))
        cut = int(len(order) * (1.0 - holdout_fraction)) if len(order) > 10 else len(order)
        train, holdout = order[:cut], order[cut:]
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        history = []

        def loss_on(rows):
            batch = collate([observations[row] for row in rows], self.device)
            target = torch.as_tensor([labels[row] for row in rows], device=self.device)
            logits = self.policy(batch).logits
            return F.cross_entropy(logits, target), (logits.argmax(dim=-1) == target).float().mean()

        for epoch in range(1, epochs + 1):
            generator.shuffle(train)
            totals = []
            for start in range(0, len(train), batch_size):
                loss, _ = loss_on(train[start : start + batch_size])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                optimizer.step()
                totals.append(float(loss.item()))
            record = {"epoch": epoch, "train_loss": float(np.mean(totals))}
            if len(holdout):
                with torch.no_grad():
                    parts = [loss_on(holdout[start : start + batch_size]) for start in range(0, len(holdout), batch_size)]
                record["holdout_loss"] = float(np.mean([float(loss) for loss, _ in parts]))
                record["holdout_accuracy"] = float(np.mean([float(accuracy) for _, accuracy in parts]))
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
    problem, spec, episodes = arguments
    env = GeneralistEnv(problem.config, spec)
    observations, labels, misses, steps = [], [], {}, 0
    try:
        for seed, actions in episodes:
            observation, _ = env.reset(seed=int(seed))
            for action in actions:
                position = observation.candidate(action)
                steps += 1
                if position is None:
                    kind = _action_kind(env, action)
                    misses[kind] = misses.get(kind, 0) + 1
                else:
                    observations.append(observation)
                    labels.append(position)
                observation, _, terminated, truncated, _ = env.step(int(action))
                if terminated or truncated:
                    break
    finally:
        env.close()
    return observations, labels, misses, steps


def _action_kind(env, action: int) -> str:
    from EVRoutingEnv.state.feasibility import joint_action_feasibility

    return joint_action_feasibility(env)[int(action)].action_kind.value


def replay_demonstrations(
    problems: Sequence[SuiteProblem],
    spec: ObservationSpec,
    episodes: Sequence[tuple[int, int, Sequence[int]]],
    workers: int = 1,
) -> tuple[list[GraphObservation], list[int], dict]:
    """Rebuild teacher demonstrations by replaying recorded actions.

    The archive stores only ``(problem, seed, actions)``: the simulator is
    deterministic given both, so observations are rebuilt with the current
    view instead of being stored. Teacher actions that fall outside the
    candidate set (a charger outside the subgraph) are counted, not imitated.
    """
    import multiprocessing as mp

    groups: dict[int, list] = {}
    for problem, seed, actions in episodes:
        groups.setdefault(int(problem), []).append((int(seed), list(actions)))
    jobs = []
    for problem, items in groups.items():
        width = max(1, min(workers, len(items)))
        jobs.extend((problems[problem], spec, items[index::width]) for index in range(width))
    if workers <= 1:
        parts = [_replay_job(job) for job in jobs]
    else:
        with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            parts = pool.map(_replay_job, jobs)
    observations = [item for part in parts for item in part[0]]
    labels = [item for part in parts for item in part[1]]
    misses: dict[str, int] = {}
    for part in parts:
        for kind, count in part[2].items():
            misses[kind] = misses.get(kind, 0) + count
    steps = sum(part[3] for part in parts)
    return observations, labels, {"steps": steps, "kept": len(labels), "misses": misses}


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
) -> list[dict]:
    """Run ``policy`` once on every listed seed of every problem, batched.

    ``envs`` from :func:`evaluation_envs` is rewound and reused, never closed;
    build it before the policy touches CUDA, since workers are forked.
    """
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
        while any(observation is not None for observation in observations):
            live = [index for index, observation in enumerate(observations) if observation is not None]
            positions, _, _ = policy.act(collate([observations[index] for index in live], device), deterministic)
            actions: list[int | None] = [None] * len(observations)
            for index, position in zip(live, positions.cpu().numpy(), strict=True):
                actions[index] = int(observations[index].action_index[position])
            observations, _, _, records = envs.step(actions)
            rows.extend(record for record in records if record is not None)
    finally:
        policy.train(was_training)
        if owned:
            envs.close()
    return rows


def evaluation_envs(problems, seeds, spec, reward, references, *, workers=1, envs_per_problem=4) -> SuiteVecEnv:
    """Finite, rewindable sub-environments covering every listed seed once."""
    slots = []
    for problem, stream in enumerate(seeds):
        width = max(1, min(envs_per_problem, len(stream)))
        slots.extend((problem, list(stream[index::width])) for index in range(width) if stream[index::width])
    return SuiteVecEnv(problems, slots, spec, reward, references, workers=workers, cycle=False)


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
