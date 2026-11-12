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

import hashlib
import importlib
import json
import math
import os
import pickle
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from algo.generalist_observation import GeneralistEnv, GraphObservation, ObservationSpec, node_counts


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
    max_decisions: int | None = None,
) -> tuple[list[tuple[int, int, list[int]]], dict]:
    """Up to ``count`` non-failed episodes per problem as ``(problem index, seed, actions)``.

    Rows come from ``demonstrations/<problem>/train_demonstrations.json`` when
    it exists, else from the reference table. Only seeds in that problem's
    ``train_seeds`` are used, in that order; a demonstrations file holding a
    validation or test seed is refused outright. With ``max_decisions`` the
    episodes of a problem stop (whole episodes, at least one) once their
    recorded decisions reach that many, so a 100-truck problem with thousands
    of decisions per episode cannot swamp the replayed set that behaviour
    cloning holds in memory.
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
        if max_decisions is not None and chosen:
            kept, decisions = [], 0
            for row in chosen:
                if kept and decisions + len(row["actions"]) > max_decisions:
                    break
                kept.append(row)
                decisions += len(row["actions"])
            chosen = kept
        episodes.extend((position, row["seed"], [int(action) for action in row["actions"]]) for row in chosen)
        sources[problem.name] = {
            "source": source,
            "rows": len(by_seed),
            "train_rows": len(candidates),
            "usable": len(usable),
            "used": len(chosen),
            "decisions": sum(len(row["actions"]) for row in chosen),
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
        self.spec = spec
        # Built at a slot's first episode. With fewer episodes per round than slots per problem most
        # slots never run one, and an idle long-haul environment holds 0.2 to 0.4 GB of route tables.
        self.envs: list[GeneralistEnv | None] = [None] * len(self.slots)
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
        if env is None:
            problem = self.problems[slot.problem]
            env = self.envs[index] = GeneralistEnv(problem.config, self.spec, problem=problem.name)
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
            if env is not None:
                env.close()


# ---------------------------------------------------------------------------
# The labelled set used by retained imitation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Replayed observations on disk, shared by the runs and resumes of a campaign
# ---------------------------------------------------------------------------

STORE_META = "meta.json"
STORE_INDEX = "index.npy"
STORE_LABELS = "labels.npy"
STORE_COUNTS = "counts.npy"


def store_fingerprint(problems: Sequence, spec: ObservationSpec, episodes: Sequence[tuple[int, int, Sequence[int]]]) -> str:
    """What a replay depends on: the problems' configurations, the view, and the episodes' seeds and actions."""
    digest = hashlib.sha256()
    for problem in problems:
        digest.update(problem.name.encode())
        digest.update(hashlib.sha256(Path(problem.config).read_bytes()).hexdigest().encode())
    digest.update(json.dumps(asdict(spec), sort_keys=True).encode())
    for problem, seed, actions in episodes:
        digest.update(json.dumps([int(problem), int(seed), [int(action) for action in actions]]).encode())
    return digest.hexdigest()


class ShardWriter:
    """Appends pickled observations to one shard file, recording offsets and lengths."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = self.path.open("wb")
        self.offsets: list[int] = []
        self.lengths: list[int] = []
        self.counts: list[tuple[int, int, int]] = []

    def append(self, observation: GraphObservation) -> None:
        data = pickle.dumps(observation, protocol=5)
        self.offsets.append(self._handle.tell())
        self.lengths.append(len(data))
        self.counts.append((len(observation.trucks), len(observation.tasks), len(observation.chargers)))
        self._handle.write(data)

    def close(self) -> dict:
        self._handle.close()
        return {"file": self.path.name, "offsets": self.offsets, "lengths": self.lengths, "counts": self.counts}


def write_store(directory: Path, shards: Sequence[dict], labels: Sequence[int], fingerprint: str, coverage: dict) -> None:
    """Index the shard files of ``directory`` (the ``ShardWriter.close`` records, in order)."""
    rows = [
        (number, offset, length)
        for number, shard in enumerate(shards)
        for offset, length in zip(shard["offsets"], shard["lengths"], strict=True)
    ]
    index = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    if len(index) != len(labels):
        raise ValueError("the store needs one label per stored observation")
    np.save(directory / STORE_INDEX, index)
    np.save(directory / STORE_LABELS, np.asarray([int(label) for label in labels], dtype=np.int64))
    if all("counts" in shard for shard in shards):
        counts = [row for shard in shards for row in shard["counts"]]
        np.save(directory / STORE_COUNTS, np.asarray(counts, dtype=np.int64).reshape(-1, 3))
    (directory / STORE_META).write_text(json.dumps({
        "fingerprint": fingerprint, "count": int(len(index)), "files": [shard["file"] for shard in shards],
        "coverage": coverage, "bytes": int(index[:, 2].sum()) if len(index) else 0,
    }, indent=2))


class ObservationStore(Sequence):
    """Replayed observations in shard files, read one at a time with ``os.pread``.

    Only the index (three integers per observation) and the labels live in
    memory; behaviour cloning loads each minibatch from disk, in threads. File
    descriptors are opened per process, so forked workers share nothing. An
    observation of a 60-truck static problem weighs 1.5 MB (its dense edge
    block grows with the square of trucks + tasks + chargers), so the broad
    set's 92k replayed decisions do not fit beside the trainer (2026-11-09).
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        meta = json.loads((self.directory / STORE_META).read_text())
        self.fingerprint: str = meta["fingerprint"]
        self.coverage: dict = meta["coverage"]
        self.files: list[str] = list(meta["files"])
        self.index = np.load(self.directory / STORE_INDEX)
        self.labels = np.load(self.directory / STORE_LABELS)
        if len(self.index) != len(self.labels) or len(self.index) != int(meta["count"]):
            raise ValueError(f"{self.directory} is inconsistent")
        self._counts: np.ndarray | None = None
        self._descriptors: dict[int, int] = {}
        self._pid: int | None = None

    @property
    def counts(self) -> np.ndarray:
        """``(trucks, tasks, chargers)`` per observation, so batches are sized without reading them.

        A store written before the counts were kept computes them once, reading
        every observation, and leaves them beside the index for the next reader.
        """
        if self._counts is None:
            path = self.directory / STORE_COUNTS
            if path.exists():
                self._counts = np.load(path).reshape(-1, 3)
            else:
                rows: list[tuple[int, int, int]] = []
                for start in range(0, len(self), 512):
                    loaded = self.read_many(range(start, min(start + 512, len(self))))
                    rows.extend((len(item.trucks), len(item.tasks), len(item.chargers)) for item in loaded)
                self._counts = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
                try:
                    temporary = self.directory / f"{STORE_COUNTS}.{os.getpid()}.tmp"
                    with temporary.open("wb") as handle:
                        np.save(handle, self._counts)
                    os.replace(temporary, path)
                except OSError:
                    pass  # a read-only store: the counts live in this process only
            if len(self._counts) != len(self.index):
                raise ValueError(f"{self.directory} holds counts for another number of observations")
        return self._counts

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self._load(row) for row in range(*item.indices(len(self)))]
        row = int(item)
        if row < 0:
            row += len(self)
        if not 0 <= row < len(self):
            raise IndexError(row)
        return self._load(row)

    def read_many(self, rows: Sequence[int], threads: int = 8) -> list[GraphObservation]:
        """The observations at ``rows``, read concurrently (positional reads, one descriptor per shard)."""
        rows = [int(row) for row in rows]
        for shard in {int(self.index[row, 0]) for row in rows}:
            self._descriptor(shard)
        if len(rows) < 2 or threads < 2:
            return [self._load(row) for row in rows]
        with ThreadPoolExecutor(min(threads, len(rows))) as pool:
            return list(pool.map(self._load, rows))

    def _load(self, row: int) -> GraphObservation:
        shard, offset, length = (int(value) for value in self.index[row])
        return pickle.loads(os.pread(self._descriptor(shard), length, offset))

    def _descriptor(self, shard: int) -> int:
        if self._pid != os.getpid():  # fresh descriptors after a fork
            self._descriptors, self._pid = {}, os.getpid()
        descriptor = self._descriptors.get(shard)
        if descriptor is None:
            descriptor = self._descriptors[shard] = os.open(self.directory / self.files[shard], os.O_RDONLY)
        return descriptor

    def close(self) -> None:
        if self._pid == os.getpid():
            for descriptor in self._descriptors.values():
                os.close(descriptor)
        self._descriptors = {}

    def __getstate__(self):
        return {**self.__dict__, "_descriptors": {}, "_pid": None}


class _Concatenated(Sequence):
    """Several sequences read as one without copying: the demonstrations (a store or a list) and the DAgger states."""

    def __init__(self, parts: Sequence[Sequence]):
        self.parts = [part for part in parts if len(part)]
        self.bounds = np.cumsum([0, *(len(part) for part in self.parts)])
        self._counts: np.ndarray | None = None

    @property
    def counts(self) -> np.ndarray:
        """``(trucks, tasks, chargers)`` of every member (a store's are read from its index)."""
        if self._counts is None:
            parts = [node_counts(part) for part in self.parts]
            self._counts = np.concatenate(parts) if parts else np.zeros((0, 3), dtype=np.int64)
        return self._counts

    def __len__(self) -> int:
        return int(self.bounds[-1])

    def _locate(self, row: int) -> tuple[int, int]:
        part = int(np.searchsorted(self.bounds, row, side="right") - 1)
        return part, row - int(self.bounds[part])

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self[row] for row in range(*item.indices(len(self)))]
        row = int(item)
        if row < 0:
            row += len(self)
        if not 0 <= row < len(self):
            raise IndexError(row)
        part, offset = self._locate(row)
        return self.parts[part][offset]

    def read_many(self, rows: Sequence[int]) -> list[GraphObservation]:
        rows = [int(row) for row in rows]
        grouped: dict[int, list[tuple[int, int]]] = {}
        for position, row in enumerate(rows):
            part, offset = self._locate(row)
            grouped.setdefault(part, []).append((position, offset))
        out: list = [None] * len(rows)
        for part, items in grouped.items():
            loaded = gather_observations(self.parts[part], [offset for _, offset in items])
            for (position, _), observation in zip(items, loaded, strict=True):
                out[position] = observation
        return out


def gather_observations(observations: Sequence, rows: Sequence[int]) -> list[GraphObservation]:
    """The observations at ``rows``: threaded from a store, a plain lookup otherwise."""
    if hasattr(observations, "read_many"):
        return observations.read_many(rows)
    return [observations[int(row)] for row in rows]


class ImitationData:
    """Fixed demonstrations plus DAgger episodes, oldest episodes evicted first."""

    def __init__(self, max_dagger_states: int = 200_000):
        self.max_dagger_states = int(max_dagger_states)
        self.demonstrations: tuple[Sequence[GraphObservation], np.ndarray] = ([], np.zeros(0, dtype=np.int64))
        self.episodes: deque[tuple[dict, list[GraphObservation], list[int]]] = deque()
        self.dagger_states = 0
        self.evicted_episodes = 0
        self._flat: tuple[list, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.demonstrations[1]) + self.dagger_states

    def set_demonstrations(self, observations: Sequence[GraphObservation], labels: Sequence[int]) -> None:
        """Keep the replayed demonstrations: a store stays on disk, anything else is copied into a list."""
        if len(observations) != len(labels):
            raise ValueError("demonstrations need one label per observation")
        kept = observations if isinstance(observations, ObservationStore) else list(observations)
        self.demonstrations = (kept, np.asarray([int(label) for label in labels], dtype=np.int64))
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

    def flat(self) -> tuple[Sequence[GraphObservation], np.ndarray]:
        """Every labelled state as one sequence; the demonstrations are read where they are (a store is not loaded)."""
        if self._flat is None:
            parts = [self.demonstrations[0], *(states for _, states, _ in self.episodes)]
            labels = np.concatenate([
                np.asarray(self.demonstrations[1], dtype=np.int64),
                *(np.asarray(targets, dtype=np.int64) for _, _, targets in self.episodes),
            ])
            self._flat = (_Concatenated(parts), labels)
        return self._flat

    def sample(self, generator: np.random.Generator, size: int) -> tuple[list[GraphObservation], np.ndarray]:
        """A uniform minibatch, with replacement, over every labelled state."""
        observations, labels = self.flat()
        if not len(labels):
            raise ValueError("the imitation set is empty")
        rows = generator.integers(0, len(labels), size=int(size))
        return gather_observations(observations, rows), labels[rows]


# ---------------------------------------------------------------------------
# Rebuilding the DAgger set on resume
# ---------------------------------------------------------------------------


def _replay_labelled_job(arguments):
    problem, spec, items = arguments
    env = GeneralistEnv(problem.config, spec, problem=problem.name)
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
        # Where a round's wall time goes: the longest episode sets it, since the slots step together.
        record[f"dagger/{name}/decisions"] = float(np.mean([row["decisions"] for row in group]))
        record[f"dagger/{name}/seconds"] = float(np.mean([row["seconds"] for row in group]))
        record[f"dagger/{name}/expert_seconds"] = float(np.mean([row["expert_seconds"] for row in group]))
    for name, (agree, total) in sorted(agreement.items()):
        record[f"dagger/{name}/agreement"] = agree / max(1, total)
    return record
