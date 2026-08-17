"""Masked PPO shared by every canonical encoder/head combination.

One trainer serves all learning variants so that optimizer settings, rollout
length, epochs, minibatching, and the total interaction budget are identical by
construction.  Only the policy architecture differs between runs.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from algo.canonical_policy import CanonicalActorCritic, CanonicalPolicyConfig


@dataclass(frozen=True)
class PPOConfig:
    """Interaction budget and optimization settings shared by all variants."""

    total_timesteps: int = 200_000
    num_envs: int = 8
    rollout_steps: int = 256
    epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    anneal_learning_rate: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.05
    normalize_advantage: bool = True
    normalize_reward: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        positive = (
            ("total_timesteps", self.total_timesteps),
            ("num_envs", self.num_envs),
            ("rollout_steps", self.rollout_steps),
            ("epochs", self.epochs),
            ("minibatch_size", self.minibatch_size),
        )
        for name, value in positive:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 < self.learning_rate < 1.0:
            raise ValueError("learning_rate must lie in (0, 1)")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1]")
        if self.clip_range <= 0.0:
            raise ValueError("clip_range must be positive")

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    @property
    def num_updates(self) -> int:
        return max(1, self.total_timesteps // self.batch_size)


class RunningReturnScale:
    """Running scale of the discounted return, used to normalize rewards.

    The shaped reward carries a 500-per-delivery bonus and a -1000 failure
    penalty, so raw returns reach the hundreds of thousands.  Left unscaled the
    value loss dominates the shared gradient norm and global clipping crushes
    the policy gradient to nothing.  Dividing rewards by this running scale
    keeps both heads on a comparable footing without changing the optimal
    policy, since it is a positive rescaling of every return.
    """

    def __init__(self, num_envs: int, gamma: float, epsilon: float = 1e-8):
        self._returns = np.zeros(num_envs, dtype=np.float64)
        self._gamma = float(gamma)
        self._mean = 0.0
        self._variance = 1.0
        self._count = epsilon
        self._epsilon = epsilon

    @property
    def scale(self) -> float:
        return float(np.sqrt(self._variance) + self._epsilon)

    def update(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        self._returns = self._returns * self._gamma + rewards.astype(np.float64)
        batch_mean = float(self._returns.mean())
        batch_variance = float(self._returns.var())
        batch_count = float(self._returns.size)

        delta = batch_mean - self._mean
        total = self._count + batch_count
        self._mean += delta * batch_count / total
        m_a = self._variance * self._count
        m_b = batch_variance * batch_count
        self._variance = (
            m_a + m_b + delta**2 * self._count * batch_count / total
        ) / total
        self._count = total
        self._returns[dones.astype(bool)] = 0.0

    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        return np.clip(rewards / self.scale, -10.0, 10.0)


@dataclass
class TrainingHistory:
    """Interaction-indexed record of learning progress and outcomes."""

    timesteps: list[int] = field(default_factory=list)
    mean_episode_reward: list[float] = field(default_factory=list)
    success_rate: list[float] = field(default_factory=list)
    mean_makespan: list[float] = field(default_factory=list)
    mean_travel_time: list[float] = field(default_factory=list)
    policy_loss: list[float] = field(default_factory=list)
    value_loss: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    approx_kl: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "timesteps": list(self.timesteps),
            "mean_episode_reward": list(self.mean_episode_reward),
            "success_rate": list(self.success_rate),
            "mean_makespan": list(self.mean_makespan),
            "mean_travel_time": list(self.mean_travel_time),
            "policy_loss": list(self.policy_loss),
            "value_loss": list(self.value_loss),
            "entropy": list(self.entropy),
            "approx_kl": list(self.approx_kl),
        }


@dataclass(frozen=True)
class RewardShaping:
    """Terminal shaping that aligns the training signal with the report metric.

    The simulator's shaped reward pays per delivery, so serving nine customers
    and stranding a truck can out-earn a complete, feasible plan.  Evaluation
    ranks feasibility first, so training adds a terminal success bonus and an
    unserved-customer penalty.  This changes only the training signal: reported
    operational metrics are extracted from the simulator, never from reward.
    """

    enabled: bool = True
    success_bonus: float = 3000.0
    incompletion_penalty: float = 4000.0
    speed_bonus: float = 0.0
    horizon_hours: float = 200.0
    stranding_penalty: float = 0.0
    energy_margin_bonus: float = 0.0
    all_served_bonus: float = 0.0
    travel_time_bonus: float = 0.0
    travel_reference_hours: float = 400.0

    # A stranded truck is a dead battery in the field, not a late delivery.
    # Plain incompletion shaping prices the two identically, which lets the
    # policy trade strandings for coverage.
    STRANDING_REASONS = frozenset(
        {
            "no_feasible_action",
            "payload_capacity_deadlock",
            "no_events_with_unserved_customers",
        }
    )

    def __post_init__(self) -> None:
        # An intermediate milestone must never out-pay the goal it leads to.
        # Paying more for "served everyone" than for "served everyone and got
        # home" makes abandoning the depot return the optimal policy.
        if self.all_served_bonus >= self.success_bonus:
            raise ValueError(
                f"all_served_bonus ({self.all_served_bonus}) must be strictly below "
                f"success_bonus ({self.success_bonus}); otherwise failing to return "
                "to the depot pays better than succeeding"
            )

    def terminal_adjustment(self, info: dict) -> float:
        if not self.enabled:
            return 0.0
        metrics = info.get("operational_metrics") or {}
        if bool(info.get("successful", False)):
            bonus = float(self.success_bonus)
            # Optional speed term, paid only on top of a feasible plan so that
            # racing can never be preferred to completing the route.
            makespan = metrics.get("fleet_makespan")
            if self.speed_bonus > 0.0 and makespan is not None:
                remaining = 1.0 - min(
                    1.0, float(makespan) / max(self.horizon_hours, 1e-9)
                )
                bonus += float(self.speed_bonus) * max(0.0, remaining)
            # The campaign objective is fleet travel hours, which is *not*
            # makespan: a plan can finish early in wall-clock terms while both
            # trucks drive long detours to chargers. This term pays for hours
            # saved against a fleet-wide reference, and like the speed bonus it
            # is only ever paid on top of a complete plan, so a shorter route
            # can never beat a finished one.
            travel = metrics.get("total_travel_time")
            if self.travel_time_bonus > 0.0 and travel is not None:
                saved = 1.0 - min(
                    1.0, float(travel) / max(self.travel_reference_hours, 1e-9)
                )
                bonus += float(self.travel_time_bonus) * max(0.0, saved)
            # Reward finishing with charge in hand. Without this the policy has
            # no reason to prefer a route that ends at 40% over one that ends at
            # 5%, and the 5% route is one bad energy draw from stranding.
            if self.energy_margin_bonus > 0.0:
                margin = float(metrics.get("minimum_terminal_soc", 0.0))
                bonus += float(self.energy_margin_bonus) * max(0.0, min(1.0, margin))
            return bonus

        completed = float(metrics.get("completed_fraction", 0.0))
        penalty = float(self.incompletion_penalty) * (1.0 - completed)
        reason = info.get("termination_reason") or metrics.get("termination_reason")
        if self.stranding_penalty > 0.0 and reason in self.STRANDING_REASONS:
            penalty += float(self.stranding_penalty)

        # Serving every customer is the last reachable milestone before success:
        # an agent learning from scratch reaches ~0.82 completion under the
        # per-delivery bonus but essentially never closes the depot return, so
        # the two are separated here. Without this the whole last mile carries
        # no gradient and training collapses to risk-avoidance.
        if self.all_served_bonus > 0.0 and completed >= 1.0 - 1e-9:
            return float(self.all_served_bonus) - penalty
        return -penalty


@dataclass(frozen=True)
class CurriculumStage:
    """One difficulty setting plus the bar for graduating from it.

    Stages vary *physical* difficulty (battery, uncertainty, horizon) and never
    the number of trucks, customers, or chargers, so the observation width and
    action space are identical throughout and one policy trains across all of
    them without reshaping.
    """

    name: str
    overrides: dict
    advance_success: float = 0.7
    min_updates: int = 15

    def apply(self, config: dict) -> dict:
        """Return a deep copy of ``config`` with this stage's overrides merged."""
        merged = deepcopy(config)
        for section, values in self.overrides.items():
            if isinstance(values, dict):
                merged.setdefault(section, {})
                merged[section].update(values)
            else:
                merged[section] = values
        return merged


class SyncCanonicalVecEnv:
    """Minimal synchronous vector environment with automatic reset.

    Each sub-environment draws episode seeds from its own disjoint namespace so
    that training scenarios never collide with validation or test scenarios.
    """

    def __init__(
        self,
        env_factory: Callable[[], object],
        num_envs: int,
        seed_stream: Sequence[int] | None = None,
        base_seed: int = 0,
        shaping: RewardShaping | None = None,
    ):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        self.shaping = shaping or RewardShaping(enabled=False)
        self.envs = [env_factory() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.base_seed = int(base_seed)
        self._seed_stream = list(seed_stream) if seed_stream is not None else None
        self._seed_position = 0
        self.observation_size = int(self.envs[0].observation_space.shape[0])
        self.action_size = int(self.envs[0].action_space.n)

    def _next_seed(self) -> int:
        if self._seed_stream is not None:
            seed = int(self._seed_stream[self._seed_position % len(self._seed_stream)])
        else:
            seed = self.base_seed + self._seed_position
        self._seed_position += 1
        return seed

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        observations, masks = [], []
        for env in self.envs:
            observation, _ = env.reset(seed=self._next_seed())
            observations.append(observation)
            masks.append(env.mask_fn())
        return np.stack(observations), np.stack(masks)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        observations, masks = [], []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = []
        for index, env in enumerate(self.envs):
            observation, reward, terminated, truncated, info = env.step(
                int(actions[index])
            )
            rewards[index] = float(reward)
            dones[index] = bool(terminated or truncated)
            if dones[index]:
                info = dict(info)
                info["terminal_observation"] = observation
                info["truncated"] = bool(truncated and not terminated)
                rewards[index] += self.shaping.terminal_adjustment(info)
                observation, _ = env.reset(seed=self._next_seed())
            observations.append(observation)
            masks.append(env.mask_fn())
            infos.append(info)
        return np.stack(observations), rewards, dones, np.stack(masks), infos

    def close(self) -> None:
        for env in self.envs:
            env.close()


def _worker_loop(connection, env_factory, seeds, shaping) -> None:
    """Own one slice of the vector environment inside a forked process.

    Terminal shaping and auto-reset happen here rather than in the parent, so
    the only thing crossing the pipe is the small per-step tuple the trainer
    actually consumes.  The full ``info`` dict carries scenario descriptors and
    per-truck state that would dominate the transfer cost.
    """
    envs = [env_factory() for _ in range(len(seeds))]
    position = [0] * len(envs)

    def next_seed(index: int) -> int:
        stream = seeds[index]
        seed = int(stream[position[index] % len(stream)])
        position[index] += 1
        return seed

    def summarize(info: dict, observation, truncated: bool) -> dict:
        return {
            "terminal_observation": observation,
            "truncated": bool(truncated),
            "successful": bool(info.get("successful", False)),
            "termination_reason": info.get("termination_reason"),
            "episode_reward": float(info.get("episode_reward", 0.0)),
            "operational_metrics": info.get("operational_metrics") or {},
        }

    try:
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                observations, masks = [], []
                for index, env in enumerate(envs):
                    observation, _ = env.reset(seed=next_seed(index))
                    observations.append(observation)
                    masks.append(env.mask_fn())
                connection.send((np.stack(observations), np.stack(masks)))
                continue
            if command != "step":
                raise ValueError(f"unknown command {command!r}")

            observations, masks = [], []
            rewards = np.zeros(len(envs), dtype=np.float32)
            dones = np.zeros(len(envs), dtype=bool)
            infos: list[dict | None] = []
            for index, env in enumerate(envs):
                observation, reward, terminated, truncated, info = env.step(
                    int(payload[index])
                )
                rewards[index] = float(reward)
                dones[index] = bool(terminated or truncated)
                if dones[index]:
                    compact = summarize(
                        info, observation, truncated and not terminated
                    )
                    rewards[index] += shaping.terminal_adjustment(compact)
                    infos.append(compact)
                    observation, _ = env.reset(seed=next_seed(index))
                else:
                    infos.append(None)
                observations.append(observation)
                masks.append(env.mask_fn())
            connection.send(
                (np.stack(observations), rewards, dones, np.stack(masks), infos)
            )
    finally:
        for env in envs:
            env.close()
        connection.close()


class WorkerCanonicalVecEnv:
    """Vector environment whose sub-environments step in forked worker processes.

    The simulator is pure Python and costs milliseconds per step, so a
    synchronous vector environment is bound to one core no matter how many
    sub-environments it holds.  Splitting them across workers is what makes the
    interaction budget reachable in wall-clock time.

    Semantics match :class:`SyncCanonicalVecEnv` exactly, including the episode
    seed stream: worker ``w`` is handed stream positions ``w, w + workers, ...``,
    so the run consumes the same seeds from the same namespace, only in a
    different order.

    Workers are forked, so they inherit the parent's threads: they run simulator
    and numpy code only and must never call into torch or CUDA. The policy stays
    entirely in the parent, which is also why only actions travel outward.
    """

    def __init__(
        self,
        env_factory: Callable[[], object],
        num_envs: int,
        seed_stream: Sequence[int] | None = None,
        base_seed: int = 0,
        shaping: RewardShaping | None = None,
        workers: int = 4,
    ):
        import multiprocessing as mp

        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        workers = max(1, min(int(workers), num_envs))
        self.num_envs = num_envs
        self.shaping = shaping or RewardShaping(enabled=False)

        stream = (
            [int(value) for value in seed_stream]
            if seed_stream is not None
            else [base_seed + index for index in range(max(num_envs * 64, 1024))]
        )
        if len(stream) < num_envs:
            raise ValueError(
                f"seed stream of {len(stream)} scenarios cannot feed {num_envs} "
                "sub-environments; every sub-environment needs its own slice"
            )
        assignment = [
            list(range(index, num_envs, workers)) for index in range(workers)
        ]
        context = mp.get_context("fork")
        self._connections = []
        self._processes = []
        self._slices = []
        cursor = 0
        for worker in range(workers):
            width = len(assignment[worker])
            # Each sub-environment gets its own slice of the shared stream so no
            # two workers can draw the same scenario in the same episode index.
            slices = [
                stream[(cursor + offset) :: num_envs] for offset in range(width)
            ]
            cursor += width
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker_loop,
                args=(child, env_factory, slices, self.shaping),
                daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
            self._slices.append(slice(cursor - width, cursor))

        probe = env_factory()
        try:
            self.observation_size = int(probe.observation_space.shape[0])
            self.action_size = int(probe.action_space.n)
        finally:
            probe.close()

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        for connection in self._connections:
            connection.send(("reset", None))
        parts = [connection.recv() for connection in self._connections]
        return (
            np.concatenate([part[0] for part in parts]),
            np.concatenate([part[1] for part in parts]),
        )

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        for connection, span in zip(self._connections, self._slices, strict=True):
            connection.send(("step", actions[span]))
        parts = [connection.recv() for connection in self._connections]
        infos: list[dict] = []
        for part in parts:
            infos.extend({} if info is None else info for info in part[4])
        return (
            np.concatenate([part[0] for part in parts]),
            np.concatenate([part[1] for part in parts]),
            np.concatenate([part[2] for part in parts]),
            np.concatenate([part[3] for part in parts]),
            infos,
        )

    def close(self) -> None:
        for connection, process in zip(
            self._connections, self._processes, strict=True
        ):
            try:
                connection.send(("close", None))
                connection.close()
            except (BrokenPipeError, OSError):
                pass
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()


class CanonicalPPO:
    """Proximal policy optimization over the canonical hard-masked action set."""

    def __init__(
        self,
        policy_config: CanonicalPolicyConfig,
        ppo_config: PPOConfig,
        env_factory: Callable[[], object],
        device: str | torch.device = "cpu",
        seed_stream: Sequence[int] | None = None,
        shaping: RewardShaping | None = None,
        curriculum: Sequence[CurriculumStage] | None = None,
        stage_factory: Callable[[CurriculumStage], Callable[[], object]] | None = None,
        rollout_workers: int = 1,
    ):
        self.policy_config = policy_config
        self.config = ppo_config
        self.device = torch.device(device)
        torch.manual_seed(ppo_config.seed)
        np.random.seed(ppo_config.seed)

        self.curriculum = list(curriculum) if curriculum else []
        self.stage_factory = stage_factory
        self.stage_index = 0
        self.stage_updates = 0
        self.stage_log: list[dict] = []
        self._seed_stream = seed_stream
        self._shaping = shaping
        self._rollout_workers = max(1, int(rollout_workers))
        if self.curriculum and self.stage_factory is None:
            raise ValueError("a curriculum requires a stage_factory")
        if self.curriculum:
            env_factory = self.stage_factory(self.curriculum[0])

        self.envs = self._build_vec_env(
            env_factory, base_seed=ppo_config.seed * 1_000_003
        )
        if self.envs.observation_size != policy_config.shape.flat_size:
            raise ValueError(
                f"environment observation size {self.envs.observation_size} does not "
                f"match the policy shape {policy_config.shape.flat_size}"
            )
        self.policy = CanonicalActorCritic(policy_config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=ppo_config.learning_rate, eps=1e-5
        )
        self.history = TrainingHistory()
        self.reward_scale = (
            RunningReturnScale(ppo_config.num_envs, ppo_config.gamma)
            if ppo_config.normalize_reward
            else None
        )
        self._episode_rewards: deque[float] = deque(maxlen=100)
        self._episode_successes: deque[float] = deque(maxlen=100)
        self._episode_makespans: deque[float] = deque(maxlen=100)
        self._episode_travel_times: deque[float] = deque(maxlen=100)
        self.total_steps = 0

    def _build_vec_env(self, env_factory: Callable[[], object], base_seed: int):
        """Vector environment for one curriculum stage, sync or worker-backed."""
        common = {
            "seed_stream": self._seed_stream,
            "base_seed": base_seed,
            "shaping": self._shaping,
        }
        if self._rollout_workers > 1:
            return WorkerCanonicalVecEnv(
                env_factory,
                self.config.num_envs,
                workers=self._rollout_workers,
                **common,
            )
        return SyncCanonicalVecEnv(env_factory, self.config.num_envs, **common)

    def learn(
        self, progress_callback: Callable[[dict], None] | None = None
    ) -> TrainingHistory:
        """Run the configured interaction budget and return the history."""
        config = self.config
        observations, masks = self.envs.reset()
        observation_tensor = self._to_tensor(observations)
        mask_tensor = self._to_bool(masks)
        # Seed the shared normalizer before the first gradient step so that no
        # variant is penalised by cold statistics.
        self.policy.observe(observation_tensor)
        start_time = time.perf_counter()

        for update in range(config.num_updates):
            if config.anneal_learning_rate:
                fraction = 1.0 - update / config.num_updates
                for group in self.optimizer.param_groups:
                    group["lr"] = fraction * config.learning_rate

            rollout, observation_tensor, mask_tensor = self._collect_rollout(
                observation_tensor, mask_tensor
            )
            statistics = self._optimize(rollout)
            self.history.timesteps.append(self.total_steps)
            self.history.mean_episode_reward.append(_mean(self._episode_rewards))
            self.history.success_rate.append(_mean(self._episode_successes))
            self.history.mean_makespan.append(_mean(self._episode_makespans))
            self.history.mean_travel_time.append(_mean(self._episode_travel_times))
            self.history.policy_loss.append(statistics["policy_loss"])
            self.history.value_loss.append(statistics["value_loss"])
            self.history.entropy.append(statistics["entropy"])
            self.history.approx_kl.append(statistics["approx_kl"])
            if progress_callback is not None:
                progress_callback(
                    {
                        "update": update + 1,
                        "num_updates": config.num_updates,
                        "timesteps": self.total_steps,
                        "elapsed_seconds": time.perf_counter() - start_time,
                        "mean_episode_reward": self.history.mean_episode_reward[-1],
                        "success_rate": self.history.success_rate[-1],
                        "mean_makespan": self.history.mean_makespan[-1],
                        "mean_travel_time": self.history.mean_travel_time[-1],
                        "curriculum_stage": self.current_stage_name,
                        **statistics,
                    }
                )
            observation_tensor, mask_tensor = self._maybe_advance_stage(
                observation_tensor, mask_tensor
            )
        return self.history

    @property
    def current_stage_name(self) -> str:
        if not self.curriculum:
            return "none"
        return self.curriculum[self.stage_index].name

    def _maybe_advance_stage(
        self,
        observation_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Graduate to the next difficulty once the current one is mastered.

        Rebuilding the vector environment resets episode statistics, because a
        success rate carried over from an easier stage would immediately
        graduate the next one too.
        """
        self.stage_updates += 1
        if not self.curriculum or self.stage_index >= len(self.curriculum) - 1:
            return observation_tensor, mask_tensor

        stage = self.curriculum[self.stage_index]
        if self.stage_updates < stage.min_updates:
            return observation_tensor, mask_tensor
        recent = _mean(self._episode_successes)
        if not np.isfinite(recent) or recent < stage.advance_success:
            return observation_tensor, mask_tensor

        self.stage_log.append(
            {
                "from_stage": stage.name,
                "to_stage": self.curriculum[self.stage_index + 1].name,
                "timesteps": self.total_steps,
                "success_rate_at_advance": float(recent),
            }
        )
        self.stage_index += 1
        self.stage_updates = 0
        next_stage = self.curriculum[self.stage_index]

        self.envs.close()
        self.envs = self._build_vec_env(
            self.stage_factory(next_stage),
            base_seed=self.config.seed * 1_000_003 + self.stage_index * 7919,
        )
        if self.envs.observation_size != self.policy_config.shape.flat_size:
            raise ValueError(
                f"curriculum stage {next_stage.name!r} changed the observation width "
                f"to {self.envs.observation_size}; stages must preserve it"
            )
        self._episode_rewards.clear()
        self._episode_successes.clear()
        self._episode_makespans.clear()
        if self.reward_scale is not None:
            self.reward_scale = RunningReturnScale(
                self.config.num_envs, self.config.gamma
            )
        observations, masks = self.envs.reset()
        return self._to_tensor(observations), self._to_bool(masks)

    def _collect_rollout(
        self,
        observation_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        config = self.config
        steps, num_envs = config.rollout_steps, config.num_envs
        observation_buffer = torch.zeros(
            (steps, num_envs, self.envs.observation_size), device=self.device
        )
        mask_buffer = torch.zeros(
            (steps, num_envs, self.envs.action_size),
            dtype=torch.bool,
            device=self.device,
        )
        action_buffer = torch.zeros(
            (steps, num_envs), dtype=torch.long, device=self.device
        )
        logprob_buffer = torch.zeros((steps, num_envs), device=self.device)
        reward_buffer = torch.zeros((steps, num_envs), device=self.device)
        done_buffer = torch.zeros((steps, num_envs), device=self.device)
        value_buffer = torch.zeros((steps, num_envs), device=self.device)

        for step in range(steps):
            observation_buffer[step] = observation_tensor
            mask_buffer[step] = mask_tensor
            with torch.no_grad():
                actions, logprobs, values = self.policy.act(
                    observation_tensor, mask_tensor
                )
            action_buffer[step] = actions
            logprob_buffer[step] = logprobs
            value_buffer[step] = values

            observations, rewards, dones, masks, infos = self.envs.step(
                actions.cpu().numpy()
            )
            self.total_steps += num_envs
            if self.reward_scale is not None:
                self.reward_scale.update(rewards, dones)
                rewards = self.reward_scale.normalize(rewards)
            reward_buffer[step] = torch.as_tensor(rewards, device=self.device)
            done_buffer[step] = torch.as_tensor(
                dones.astype(np.float32), device=self.device
            )
            # A time-limit truncation is not a terminal state: the episode would
            # have continued. Bootstrapping its cut-off value keeps the return
            # unbiased. Truncation is common here (step and horizon limits), so
            # treating it as termination would systematically understate returns.
            truncated_indices = [
                index for index, info in enumerate(infos) if info.get("truncated")
            ]
            if truncated_indices:
                terminal_batch = self._to_tensor(
                    np.stack(
                        [
                            infos[index]["terminal_observation"]
                            for index in truncated_indices
                        ]
                    )
                )
                with torch.no_grad():
                    # The critic is fit on normalized returns, so its output is
                    # already in the same units as the normalized reward buffer.
                    terminal_values = self.policy.predict_values(terminal_batch)
                for position, index in enumerate(truncated_indices):
                    reward_buffer[step, index] += (
                        config.gamma * terminal_values[position]
                    )
            for info in infos:
                if "terminal_observation" in info:
                    self._record_episode(info)
            observation_tensor = self._to_tensor(observations)
            mask_tensor = self._to_bool(masks)

        self.policy.observe(observation_buffer.reshape(-1, self.envs.observation_size))
        with torch.no_grad():
            last_value = self.policy.predict_values(observation_tensor)
        advantages = self._compute_advantages(
            reward_buffer, value_buffer, done_buffer, last_value
        )
        returns = advantages + value_buffer
        return (
            {
                "observations": observation_buffer.reshape(
                    -1, self.envs.observation_size
                ),
                "masks": mask_buffer.reshape(-1, self.envs.action_size),
                "actions": action_buffer.reshape(-1),
                "logprobs": logprob_buffer.reshape(-1),
                "advantages": advantages.reshape(-1),
                "returns": returns.reshape(-1),
                "values": value_buffer.reshape(-1),
            },
            observation_tensor,
            mask_tensor,
        )

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        advantages = torch.zeros_like(rewards)
        running = torch.zeros_like(last_value)
        for step in reversed(range(config.rollout_steps)):
            continues = 1.0 - dones[step]
            next_value = (
                last_value if step == config.rollout_steps - 1 else values[step + 1]
            )
            delta = rewards[step] + config.gamma * next_value * continues - values[step]
            running = delta + config.gamma * config.gae_lambda * continues * running
            advantages[step] = running
        return advantages

    def _optimize(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        config = self.config
        indices = np.arange(config.batch_size)
        policy_losses, value_losses, entropies, kls = [], [], [], []
        stop = False

        for _ in range(config.epochs):
            np.random.shuffle(indices)
            for start in range(0, config.batch_size, config.minibatch_size):
                batch = indices[start : start + config.minibatch_size]
                if len(batch) < 2:
                    continue
                batch_index = torch.as_tensor(batch, device=self.device)
                logprobs, entropy, values = self.policy.evaluate_actions(
                    rollout["observations"][batch_index],
                    rollout["actions"][batch_index],
                    rollout["masks"][batch_index],
                )
                log_ratio = logprobs - rollout["logprobs"][batch_index]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                kls.append(float(approx_kl.item()))

                advantages = rollout["advantages"][batch_index]
                if config.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std(unbiased=False) + 1e-8
                    )
                policy_loss = -torch.min(
                    ratio * advantages,
                    torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
                    * advantages,
                ).mean()

                returns = rollout["returns"][batch_index]
                if config.value_clip_range > 0.0:
                    old_values = rollout["values"][batch_index]
                    clipped = old_values + torch.clamp(
                        values - old_values,
                        -config.value_clip_range,
                        config.value_clip_range,
                    )
                    value_loss = (
                        0.5
                        * torch.max(
                            (values - returns).pow(2), (clipped - returns).pow(2)
                        ).mean()
                    )
                else:
                    value_loss = 0.5 * F.mse_loss(values, returns)

                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy_mean
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), config.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_mean.item()))

            if (
                config.target_kl is not None
                and kls
                and np.mean(kls[-8:]) > config.target_kl
            ):
                stop = True
            if stop:
                break

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
        }

    def _record_episode(self, info: dict) -> None:
        self._episode_rewards.append(float(info.get("episode_reward", 0.0)))
        self._episode_successes.append(float(bool(info.get("successful", False))))
        metrics = info.get("operational_metrics") or {}
        makespan = metrics.get("fleet_makespan")
        if makespan is not None and np.isfinite(makespan):
            self._episode_makespans.append(float(makespan))
            # Only successful episodes carry a makespan, and travel hours are
            # tracked on the same episodes so the two diagnostics compare.
            travel = metrics.get("total_travel_time")
            if travel is not None and np.isfinite(travel):
                self._episode_travel_times.append(float(travel))

    def _to_tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _to_bool(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.bool, device=self.device)

    def save(self, directory: str | Path, prefix: str = "policy") -> None:
        self.policy.save(directory, prefix=prefix)

    def close(self) -> None:
        self.envs.close()


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")
