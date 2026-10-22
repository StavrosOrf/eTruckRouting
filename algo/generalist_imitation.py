"""Imitation for the generalist GraphPPO policy beyond the greedy warm start.

Three pieces, all off by default (the trainer then behaves exactly as before):

* **Demonstrations from stronger experts** (:func:`demonstration_episodes`).
  Behaviour cloning reads ``DIR/<problem>/train_demonstrations.json``, written
  by the best-known-solution agents, and falls back to the reference rows (the
  greedy teacher) for a problem without one. A row carries ``seed`` and
  ``actions``, the ``(problem, seed, actions)`` format of the reference rows,
  plus the :func:`~algo.generalist_ppo.episode_summary` fields. Rows are
  replayed through the current observation, so a schema change never
  invalidates them. Only train-namespace seeds are accepted.
* **Retained imitation** (``ImitationConfig.bc_*``). Every PPO minibatch is
  paired with a minibatch of labelled states, and ``lambda_t`` times their
  cross-entropy is added to the PPO loss; ``lambda_t`` moves linearly from
  ``bc_coefficient`` to ``bc_coefficient_final`` over ``bc_anneal_timesteps``.
  The labelled states stay on the CPU and are collated per minibatch.
* **DAgger** (``ImitationConfig.dagger_*``). Every ``dagger_every`` updates,
  extra sub-environments inside the forked rollout workers run episodes in
  which, at every visited state, the expert is queried; its action is executed
  with probability ``beta`` and the learner's otherwise. Every visited state is
  labelled with the expert's action and aggregated into the retained-imitation
  set. The learner acts in the trainer process; the workers run the simulator
  and the expert only, never torch, so they are forked before the policy
  touches CUDA like the rollout workers. The set is checkpointed as
  ``(problem, seed, executed actions, expert labels)`` and rebuilt by replay
  on resume, as the demonstrations are.

Experts (:func:`resolve_expert`) answer ``act(env) -> environment action
index`` from any live state, and must not change ``env``. ``"teacher"`` is the
problem's greedy teacher (greedy navigator, or greedy dispatch for requests); a
name in :data:`algo.generalist_ppo.TEACHERS` picks that baseline; and
``"package.module:factory"`` imports ``factory(problem, env)``, which returns an
object with ``act(env)`` (or a plain callable ``env -> action``) and,
optionally, ``reset(env, seed)``. The factory is called afresh after every
episode reset (then ``reset``), as a per-episode controller expects. Raising
``RuntimeError`` or ``ValueError`` declines a state: it is not labelled and the
learner's action runs. Experts run in the forked workers and must not use
CUDA. A plan-following executor must replan from the state it is handed:
under DAgger the learner, not the expert, chose some of the earlier actions.
"""

from __future__ import annotations

import importlib
import json
import math
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from algo.generalist_observation import GeneralistEnv, GraphObservation, ObservationSpec


DEMONSTRATIONS_FILE = "train_demonstrations.json"
VALIDATION_FILE = "validation.json"
NO_EXPERT = "none"
_TRAIN_NAMESPACE = 1_000_000_000  # EVRoutingEnv.evaluation.artifacts.SEED_SPLIT_OFFSETS["validation"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImitationConfig:
    """Retained imitation and DAgger settings; the defaults change nothing."""

    bc_coefficient: float = 0.0
    bc_coefficient_final: float = 0.0
    bc_anneal_timesteps: int = 0
    bc_batch_size: int = 256
    # Entries "SPEC" (every problem) or "PROBLEM=SPEC"; SPEC is "teacher", a
    # teacher name, "package.module:factory", or "none".
    dagger_expert: tuple[str, ...] = ()
    dagger_every: int = 10
    dagger_episodes: int = 8
    dagger_envs_per_problem: int = 4
    dagger_beta: float = 0.5
    dagger_beta_final: float = 0.0
    dagger_beta_anneal_timesteps: int = 0
    dagger_max_states: int = 200_000
    dagger_deterministic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dagger_expert", tuple(str(entry) for entry in self.dagger_expert))
        for name in ("bc_coefficient", "bc_coefficient_final"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("dagger_beta", "dagger_beta_final"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("bc_batch_size", "dagger_every", "dagger_episodes", "dagger_envs_per_problem", "dagger_max_states"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("bc_anneal_timesteps", "dagger_beta_anneal_timesteps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.expert_table()

    def expert_table(self) -> dict[str | None, str]:
        """``{problem or None (the default): expert spec}``, validated."""
        from algo.generalist_ppo import TEACHERS

        table: dict[str | None, str] = {}
        for entry in self.dagger_expert:
            problem, separator, spec = entry.partition("=")
            key = problem.strip() if separator else None
            spec = (spec if separator else entry).strip()
            if key in table:
                raise ValueError(f"dagger_expert names {key or 'the default expert'!r} twice")
            module, colon, factory = spec.partition(":")
            if spec not in ("teacher", NO_EXPERT, *TEACHERS) and not (colon and module and factory):
                raise ValueError(
                    f"expert {spec!r} must be 'teacher', 'none', a teacher in {TEACHERS}, or 'module:factory'"
                )
            table[key] = spec
        return table

    def expert_for(self, problem: str) -> str | None:
        """The expert queried on ``problem``, or ``None`` when DAgger skips it."""
        table = self.expert_table()
        spec = table.get(problem, table.get(None))
        return None if spec in (None, NO_EXPERT) else spec

    @property
    def dagger(self) -> bool:
        return bool(self.dagger_expert)

    def bc_weight(self, timesteps: int) -> float:
        return _linear(self.bc_coefficient, self.bc_coefficient_final, self.bc_anneal_timesteps, timesteps)

    def beta(self, timesteps: int) -> float:
        return _linear(self.dagger_beta, self.dagger_beta_final, self.dagger_beta_anneal_timesteps, timesteps)


def _linear(start: float, end: float, horizon: int, timesteps: int) -> float:
    """``start`` moving linearly to ``end`` over ``horizon`` steps; constant if ``horizon`` is 0."""
    if horizon <= 0:
        return float(start)
    return float(start + (end - start) * min(1.0, max(0, timesteps) / horizon))


# ---------------------------------------------------------------------------
# Demonstration and expert-outcome files
# ---------------------------------------------------------------------------


def read_rows(path: str | Path) -> list[dict]:
    """Episode rows from a list, a seed-keyed dict, or ``{"rows"|"episodes"|"demonstrations": ...}``.

    A row needs a ``seed`` (or its key) and may nest its fields under ``best``.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        for key in ("rows", "episodes", "demonstrations"):
            if isinstance(data.get(key), (list, dict)):
                data = data[key]
                break
    items = data.items() if isinstance(data, dict) else ((None, row) for row in data)
    rows = []
    for key, row in items:
        if not isinstance(row, dict):
            raise TypeError(f"{path}: every row must be an object")
        if "actions" not in row and isinstance(row.get("best"), dict):
            row = {**row["best"], **{k: v for k, v in row.items() if k != "best"}}
        seed = row.get("seed", key)
        if seed is None:
            raise ValueError(f"{path}: a row has no seed")
        rows.append({**row, "seed": int(seed)})
    return rows


def expert_reference_rows(folder: str | Path) -> dict[int, dict]:
    """Expert outcomes per seed from a ``bks/<problem>`` folder, for paired rewards.

    Train seeds come from the realized demonstrations, validation seeds from
    the ``best`` entry of ``validation.json``; both carry the
    :func:`~algo.generalist_ppo.episode_summary` fields the reward reads.
    """
    folder = Path(folder)
    rows: dict[int, dict] = {}
    if (folder / DEMONSTRATIONS_FILE).exists():
        rows.update({row["seed"]: row for row in read_rows(folder / DEMONSTRATIONS_FILE)})
    if (folder / VALIDATION_FILE).exists():
        for seed, entry in json.loads((folder / VALIDATION_FILE).read_text()).items():
            best = entry.get("best", entry) if isinstance(entry, dict) else None
            if isinstance(best, dict) and "hours" in best:
                rows[int(seed)] = {**best, "seed": int(seed)}
    required = ("hours", "served", "tasks", "dynamic")
    incomplete = sorted(seed for seed, row in rows.items() if any(key not in row for key in required))
    if incomplete:
        raise ValueError(f"{folder}: rows lack one of {required}, e.g. seed {incomplete[0]}")
    return rows


def demonstration_episodes(
    problems: Sequence,
    train_seeds: Sequence[Sequence[int]],
    count: int,
    *,
    demonstrations: str | Path | None = None,
    references=None,
) -> tuple[list[tuple[int, int, list[int]]], dict]:
    """Up to ``count`` non-failed episodes per problem as ``(problem index, seed, actions)``.

    Rows come from ``demonstrations/<problem>/train_demonstrations.json`` when
    it exists, else from the reference table. Only seeds in that problem's
    ``train_seeds`` are used, in that order; a demonstrations file holding a
    validation or test seed is refused outright.
    """
    episodes, sources = [], {}
    for position, problem in enumerate(problems):
        path = Path(demonstrations) / problem.name / DEMONSTRATIONS_FILE if demonstrations else None
        if path is not None and path.exists():
            rows = read_rows(path)
            foreign = sorted({row["seed"] for row in rows if not 0 <= row["seed"] < _TRAIN_NAMESPACE})
            if foreign:
                raise ValueError(f"{path} holds seeds outside the train namespace, e.g. {foreign[:3]}")
            mismatched = {row.get("problem") for row in rows} - {None, problem.name}
            if mismatched:
                raise ValueError(f"{path} holds rows of other problems: {sorted(mismatched)}")
            by_seed, source = {row["seed"]: row for row in rows}, str(path)
        else:
            by_seed = dict(references.rows.get(problem.name, {})) if references is not None else {}
            source = "references"
        stream = [int(seed) for seed in train_seeds[position]]
        candidates = [by_seed[seed] for seed in stream if seed in by_seed]
        usable = [row for row in candidates if not row.get("failed", False) and row.get("actions")]
        chosen = usable[:count]
        episodes.extend((position, row["seed"], [int(action) for action in row["actions"]]) for row in chosen)
        sources[problem.name] = {
            "source": source,
            "rows": len(by_seed),
            "train_rows": len(candidates),
            "usable": len(usable),
            "used": len(chosen),
            "failed_skipped": sum(bool(row.get("failed", False)) for row in candidates),
        }
    return episodes, sources


# ---------------------------------------------------------------------------
# Experts
# ---------------------------------------------------------------------------


class TeacherExpert:
    """A greedy baseline from :func:`algo.generalist_ppo.build_teacher`, rebuilt per episode."""

    def __init__(self, name: str):
        self.name = name
        self.policy = None

    def reset(self, env, seed: int) -> None:
        from algo.generalist_ppo import build_teacher

        self.policy = build_teacher(self.name, env)

    def act(self, env) -> int:
        if self.policy is None:
            self.reset(env, getattr(env, "scenario_seed", 0))
        return int(self.policy(env))


class _CallableExpert:
    def __init__(self, function):
        self.function = function

    def reset(self, env, seed: int) -> None:
        reset = getattr(self.function, "reset", None)
        if callable(reset):
            reset(env, seed)

    def act(self, env) -> int:
        return int(self.function(env))


def import_factory(spec: str):
    module, _, name = spec.partition(":")
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise TypeError(f"{spec} is not callable")
    return factory


def resolve_expert(spec: str, problem, env):
    """Build the expert named by ``spec`` for ``problem`` (see the module docstring)."""
    from algo.generalist_ppo import TEACHERS

    if spec == "teacher":
        return TeacherExpert(problem.teacher)
    if spec in TEACHERS:
        return TeacherExpert(spec)
    expert = import_factory(spec)(problem, env)
    if callable(getattr(expert, "act", None)):
        return expert
    if callable(expert):
        return _CallableExpert(expert)
    raise TypeError(f"{spec} returned {type(expert).__name__}, which has no act(env)")


def check_experts(config: ImitationConfig, problems: Sequence) -> dict[str, str | None]:
    """Import every ``module:factory`` once, in the parent, so a typo fails before any fork."""
    experts = {problem.name: config.expert_for(problem.name) for problem in problems}
    for spec in {spec for spec in experts.values() if spec and ":" in spec}:
        import_factory(spec)
    unknown = {key for key in config.expert_table() if key is not None} - set(experts)
    if unknown:
        raise ValueError(f"dagger_expert names problems outside training: {sorted(unknown)}")
    return experts


def _feasibility(env, action: int) -> tuple[str, bool]:
    from EVRoutingEnv.state.feasibility import joint_action_feasibility

    results = joint_action_feasibility(env)
    if not 0 <= int(action) < len(results):
        return "out_of_range", False
    result = results[int(action)]
    return result.action_kind.value, bool(result.feasible)


# ---------------------------------------------------------------------------
# DAgger: worker side
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DaggerSlot:
    """One DAgger sub-environment: a problem, a train-seed stream and an expert."""

    problem: int
    seeds: tuple[int, ...]
    expert: str
    index: int


def dagger_slots(problems: Sequence, train_seeds: Sequence[Sequence[int]], config: ImitationConfig) -> list[DaggerSlot]:
    """``dagger_envs_per_problem`` slots per problem with an expert.

    Seeds are taken from the end of the train stream, so they rarely coincide
    with the demonstration seeds, which come from its start.
    """
    slots = []
    width = config.dagger_envs_per_problem
    for position, problem in enumerate(problems):
        spec = config.expert_for(problem.name)
        if spec is None:
            continue
        stream = [int(seed) for seed in reversed(list(train_seeds[position]))]
        if any(not 0 <= seed < _TRAIN_NAMESPACE for seed in stream):
            raise ValueError("DAgger seeds must come from the train namespace")
        for offset in range(width):
            seeds = tuple(stream[offset::width])
            if seeds:
                slots.append(DaggerSlot(position, seeds, spec, len(slots)))
    return slots


def episode_counts(slots: Sequence[DaggerSlot], episodes: int) -> list[int]:
    """Spread ``episodes`` per problem over that problem's slots."""
    by_problem: dict[int, list[int]] = {}
    for index, slot in enumerate(slots):
        by_problem.setdefault(slot.problem, []).append(index)
    counts = [0] * len(slots)
    for members in by_problem.values():
        for rank, index in enumerate(members):
            counts[index] = episodes // len(members) + int(rank < episodes % len(members))
    return counts


class DaggerSlots:
    """The DAgger sub-environments one rollout worker owns: simulator and expert only."""

    def __init__(self, problems: Sequence, slots: Sequence[DaggerSlot], spec: ObservationSpec):
        self.problems = list(problems)
        self.slots = list(slots)
        self.envs = [GeneralistEnv(self.problems[slot.problem].config, spec) for slot in self.slots]
        self.experts: list = [None] * len(self.slots)
        self.position = [0] * len(self.slots)
        self.remaining = [0] * len(self.slots)
        self.current: list[dict | None] = [None] * len(self.slots)
        self.pending: list[tuple] = [(None, None, False)] * len(self.slots)
        self.generators: list = [None] * len(self.slots)
        self.beta = 0.0

    def start(self, counts: Sequence[int], beta: float, entropy: Sequence[int]) -> list:
        """Begin a round: slot ``i`` runs ``counts[i]`` episodes; returns ``(observation, label)`` or ``None``."""
        self.beta = float(beta)
        results = []
        for index, slot in enumerate(self.slots):
            self.remaining[index] = int(counts[index])
            self.generators[index] = np.random.default_rng([*map(int, entropy), slot.index])
            results.append(self._begin(index))
        return results

    def _begin(self, index: int):
        slot = self.slots[index]
        if self.remaining[index] <= 0 or not slot.seeds:
            self.current[index] = None
            return None
        seed = int(slot.seeds[self.position[index] % len(slot.seeds)])
        self.position[index] += 1
        self.remaining[index] -= 1
        env = self.envs[index]
        observation, _ = env.reset(seed=seed)
        # A fresh expert per episode, as a per-episode controller expects.
        self.experts[index] = resolve_expert(slot.expert, self.problems[slot.problem], env)
        reset = getattr(self.experts[index], "reset", None)
        if callable(reset):
            reset(env, seed)
        self.current[index] = {
            "seed": seed, "actions": [], "labels": [], "expert_executed": 0, "expert_errors": 0,
            "misses": {}, "expert_seconds": 0.0, "started": time.perf_counter(),
        }
        return self._label(index, observation)

    def _label(self, index: int, observation: GraphObservation):
        """Query the expert at the current state; the label is its candidate position."""
        env, episode = self.envs[index], self.current[index]
        started = time.perf_counter()
        try:
            action = int(self.experts[index].act(env))
        except (RuntimeError, ValueError):
            action = None
        episode["expert_seconds"] += time.perf_counter() - started
        if action is None:
            episode["expert_errors"] += 1
            self.pending[index] = (None, None, False)
            return observation, None
        position = observation.candidate(action)
        executable = position is not None
        if position is None:
            kind, executable = _feasibility(env, action)
            episode["misses"][kind] = episode["misses"].get(kind, 0) + 1
        self.pending[index] = (action, position, executable)
        return observation, position

    def step(self, actions: Sequence[int | None]) -> list:
        """Execute the expert's action with probability beta, else the learner's.

        Returns per slot ``None`` (idle) or ``(next (observation, label) or
        None, finished episode record or None)``.
        """
        results = []
        for index, learner in enumerate(actions):
            episode = self.current[index]
            if episode is None or learner is None:
                results.append(None)
                continue
            expert, position, executable = self.pending[index]
            draw = self.generators[index].random()
            use_expert = expert is not None and executable and draw < self.beta
            executed = int(expert) if use_expert else int(learner)
            episode["actions"].append(executed)
            episode["labels"].append(int(expert) if position is not None else -1)
            episode["expert_executed"] += int(use_expert)
            observation, _, terminated, truncated, _ = self.envs[index].step(executed)
            if terminated or truncated:
                record = self._finish(index)
                results.append((self._begin(index), record))
            else:
                results.append((self._label(index, observation), None))
        return results

    def _finish(self, index: int) -> dict:
        from algo.generalist_ppo import episode_summary

        slot, episode = self.slots[index], self.current[index]
        self.current[index] = None
        started = episode.pop("started")
        return {
            "problem": self.problems[slot.problem].name,
            "slot": slot.index,
            "decisions": len(episode["actions"]),
            "seconds": time.perf_counter() - started,
            **episode,
            **episode_summary(self.envs[index]),
        }

    def positions(self) -> list[int]:
        return list(self.position)

    def set_positions(self, positions: Sequence[int]) -> list[bool]:
        self.position = [int(value) for value in positions]
        return [True] * len(self.slots)

    def close(self) -> None:
        for env in self.envs:
            env.close()


# ---------------------------------------------------------------------------
# The labelled set used by retained imitation
# ---------------------------------------------------------------------------


class ImitationData:
    """Fixed demonstrations plus DAgger episodes, oldest episodes evicted first."""

    def __init__(self, max_dagger_states: int = 200_000):
        self.max_dagger_states = int(max_dagger_states)
        self.demonstrations: tuple[list[GraphObservation], list[int]] = ([], [])
        self.episodes: deque[tuple[dict, list[GraphObservation], list[int]]] = deque()
        self.dagger_states = 0
        self.evicted_episodes = 0
        self._flat: tuple[list, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.demonstrations[1]) + self.dagger_states

    def set_demonstrations(self, observations: Sequence[GraphObservation], labels: Sequence[int]) -> None:
        if len(observations) != len(labels):
            raise ValueError("demonstrations need one label per observation")
        self.demonstrations = (list(observations), [int(label) for label in labels])
        self._flat = None

    def add_episode(self, row: dict, observations: Sequence[GraphObservation], labels: Sequence[int]) -> None:
        """Aggregate one DAgger episode; ``row`` is its compact archive entry."""
        if len(observations) != len(labels):
            raise ValueError("a DAgger episode needs one label per observation")
        self.episodes.append((row, list(observations), [int(label) for label in labels]))
        self.dagger_states += len(labels)
        while self.dagger_states > self.max_dagger_states and len(self.episodes) > 1:
            _, _, dropped = self.episodes.popleft()
            self.dagger_states -= len(dropped)
            self.evicted_episodes += 1
        self._flat = None

    def archive(self) -> list[dict]:
        """``(problem, seed, executed actions, expert labels)`` of every retained episode."""
        return [row for row, _, _ in self.episodes]

    def restore(self, archive: Sequence[dict], replayed: Sequence[tuple[list, list]]) -> None:
        """Rebuild the DAgger part from :func:`replay_dagger_archive` output."""
        if len(archive) != len(replayed):
            raise ValueError("the replayed DAgger set does not match its archive")
        self.episodes.clear()
        self.dagger_states = 0
        for row, (observations, labels) in zip(archive, replayed, strict=True):
            self.episodes.append((dict(row), list(observations), [int(label) for label in labels]))
            self.dagger_states += len(labels)
        self._flat = None

    def flat(self) -> tuple[list[GraphObservation], np.ndarray]:
        if self._flat is None:
            observations = list(self.demonstrations[0])
            labels = list(self.demonstrations[1])
            for _, states, targets in self.episodes:
                observations.extend(states)
                labels.extend(targets)
            self._flat = (observations, np.asarray(labels, dtype=np.int64))
        return self._flat

    def sample(self, generator: np.random.Generator, size: int) -> tuple[list[GraphObservation], np.ndarray]:
        """A uniform minibatch, with replacement, over every labelled state."""
        observations, labels = self.flat()
        if not len(labels):
            raise ValueError("the imitation set is empty")
        rows = generator.integers(0, len(labels), size=int(size))
        return [observations[row] for row in rows], labels[rows]


# ---------------------------------------------------------------------------
# Rebuilding the DAgger set on resume
# ---------------------------------------------------------------------------


def _replay_labelled_job(arguments):
    problem, spec, items = arguments
    env = GeneralistEnv(problem.config, spec)
    results = []
    try:
        for position, seed, actions, labels in items:
            observation, _ = env.reset(seed=int(seed))
            states, targets = [], []
            for action, label in zip(actions, labels, strict=True):
                if int(label) >= 0:
                    candidate = observation.candidate(int(label))
                    if candidate is not None:
                        states.append(observation)
                        targets.append(candidate)
                observation, _, terminated, truncated, _ = env.step(int(action))
                if terminated or truncated:
                    break
            results.append((position, states, targets))
    finally:
        env.close()
    return results


def replay_dagger_archive(
    problems: Sequence, spec: ObservationSpec, archive: Sequence[dict], workers: int = 1,
) -> list[tuple[list[GraphObservation], list[int]]]:
    """Replay archived DAgger episodes to their labelled states, in archive order.

    The simulator is deterministic given the seed and the executed actions,
    so this reproduces the live states. With ``workers > 1`` it forks: call it
    before the policy touches CUDA.
    """
    import multiprocessing as mp

    index = {problem.name: position for position, problem in enumerate(problems)}
    missing = sorted({row["problem"] for row in archive} - set(index))
    if missing:
        raise ValueError(f"the DAgger archive holds problems outside training: {missing}")
    groups: dict[int, list] = {}
    for position, row in enumerate(archive):
        groups.setdefault(index[row["problem"]], []).append((position, row["seed"], row["actions"], row["labels"]))
    jobs = []
    for problem, items in groups.items():
        width = max(1, min(workers, len(items)))
        jobs.extend((problems[problem], spec, items[offset::width]) for offset in range(width))
    if workers <= 1 or len(jobs) <= 1:
        parts = [_replay_labelled_job(job) for job in jobs]
    else:
        with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            parts = pool.map(_replay_labelled_job, jobs)
    ordered: list = [None] * len(archive)
    for part in parts:
        for position, states, targets in part:
            ordered[position] = (states, targets)
    return ordered


def trainable_archive(saved: dict | None, problems: Sequence) -> list[dict]:
    """The archived DAgger episodes of problems that are still trained."""
    names = {problem.name for problem in problems}
    return [row for row in (saved or {}).get("archive", []) if row["problem"] in names]


def peek_imitation_state(path: str | Path) -> dict | None:
    """The imitation block of a training checkpoint, read on the CPU before any fork."""
    from algo.checkpoint_io import load_trusted_training

    return load_trusted_training(path, map_location="cpu").get("imitation")


def summarize_dagger_round(records: Sequence[dict], agreement: dict[str, list[int]], beta: float) -> dict:
    """Flat history fields for one DAgger round: coverage, mixing, and per-problem outcomes."""
    record: dict = {"dagger/beta": float(beta), "dagger/episodes": len(records)}
    decisions = sum(row["decisions"] for row in records)
    labelled = sum(sum(label >= 0 for label in row["labels"]) for row in records)
    misses: dict[str, int] = {}
    for row in records:
        for kind, count in row["misses"].items():
            misses[kind] = misses.get(kind, 0) + count
    record.update({
        "dagger/decisions": decisions,
        "dagger/labelled": labelled,
        "dagger/misses": misses,
        "dagger/expert_errors": sum(row["expert_errors"] for row in records),
        "dagger/expert_executed_fraction": sum(row["expert_executed"] for row in records) / max(1, decisions),
        "dagger/expert_seconds_per_query": sum(row["expert_seconds"] for row in records) / max(1, decisions),
    })
    for name in sorted({row["problem"] for row in records}):
        group = [row for row in records if row["problem"] == name]
        record[f"dagger/{name}/served_fraction"] = float(np.mean([row["served"] / max(1, row["tasks"]) for row in group]))
        record[f"dagger/{name}/success"] = float(np.mean([row["success"] for row in group]))
        record[f"dagger/{name}/failed"] = float(np.mean([row["failed"] for row in group]))
        record[f"dagger/{name}/hours"] = float(np.mean([row["hours"] for row in group]))
    for name, (agree, total) in sorted(agreement.items()):
        record[f"dagger/{name}/agreement"] = agree / max(1, total)
    return record
