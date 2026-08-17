"""Demonstration pretraining for the canonical policy.

Success in this problem needs roughly a hundred consecutive well-chosen actions,
so a randomly initialised policy almost never observes a successful episode and
PPO gets no gradient toward feasibility.  Pretraining on demonstrations from a
tuned controller puts the policy inside the feasible regime first; PPO then
improves on the demonstrator with closed-loop experience.

The demonstrator is only ever run on ``train`` scenarios, and the cloned policy
still selects from the same hard feasibility mask, so nothing here leaks
validation or test information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from algo.canonical_policy import CanonicalActorCritic


# Chunk size for streaming the archive across the CPU/accelerator boundary.
_OBSERVE_CHUNK = 4096


@dataclass
class DemonstrationSet:
    """Observations and demonstrator actions collected from train scenarios."""

    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    episodes: int = 0
    successful_episodes: int = 0

    def __len__(self) -> int:
        return len(self.actions)

    def as_tensors(
        self, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, ...]:
        """Materialize the archive as tensors, on CPU by default.

        A demonstration set of ~130k canonical observations is several gigabytes
        at full width, so it is deliberately kept off the accelerator; training
        streams minibatches across instead. Resident copies on the GPU exhausted
        a 24 GB card with only four concurrent runs.
        """
        return (
            torch.as_tensor(
                np.stack(self.observations), dtype=torch.float32, device=device
            ),
            torch.as_tensor(np.asarray(self.actions), dtype=torch.long, device=device),
            torch.as_tensor(np.stack(self.masks), dtype=torch.bool, device=device),
        )


def _roll_episode(env, demonstrator, seed: int, max_steps: int) -> dict:
    """Run one controller on one scenario and return its trace and outcome."""
    observation, info = env.reset(seed=int(seed))
    terminated = truncated = False
    steps = 0
    observations: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    reset = getattr(demonstrator, "reset", None)
    if callable(reset):
        reset()
    while not (terminated or truncated) and steps < max_steps:
        mask = env.mask_fn()
        try:
            action = demonstrator(env, observation, info)
        except (RuntimeError, ValueError):
            break
        observations.append(np.asarray(observation, dtype=np.float32))
        actions.append(int(action))
        masks.append(np.asarray(mask, dtype=bool))
        observation, _, terminated, truncated, info = env.step(action)
        steps += 1

    metrics = info.get("operational_metrics") or {}
    makespan = metrics.get("fleet_makespan")
    return {
        "observations": observations,
        "actions": actions,
        "masks": masks,
        "success": bool(metrics.get("success", False)),
        "makespan": float(makespan) if makespan is not None else float("inf"),
    }


def collect_demonstrations(
    env,
    demonstrator,
    seeds,
    max_steps: int = 1_000,
    successful_only: bool = True,
) -> DemonstrationSet:
    """Roll the demonstrator over ``seeds`` and keep its state-action pairs.

    ``demonstrator`` may be a single controller or a sequence of them.  With a
    sequence the scenario is solved by each in turn and only the successful trace
    with the lowest makespan is kept, which yields a teacher at least as strong
    as its best member on every scenario.

    With ``successful_only`` the transitions of failed episodes are discarded, so
    the policy is not taught to reproduce the demonstrator's own failures.
    """
    controllers = (
        list(demonstrator)
        if isinstance(demonstrator, (list, tuple))
        else [demonstrator]
    )
    dataset = DemonstrationSet()
    for seed in seeds:
        traces = [
            _roll_episode(env, controller, seed, max_steps)
            for controller in controllers
        ]
        successful = [trace for trace in traces if trace["success"]]
        dataset.episodes += 1
        if successful:
            dataset.successful_episodes += 1
            best = min(successful, key=lambda trace: trace["makespan"])
        elif successful_only:
            continue
        else:
            best = traces[0]
        dataset.observations.extend(best["observations"])
        dataset.actions.extend(best["actions"])
        dataset.masks.extend(best["masks"])
    return dataset


def pretrain_policy(
    policy: CanonicalActorCritic,
    dataset: DemonstrationSet,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: torch.device | str = "cpu",
    validation_fraction: float = 0.1,
    progress=None,
) -> list[dict]:
    """Fit the policy's action distribution to the demonstrator by cross-entropy."""
    if len(dataset) == 0:
        raise ValueError("cannot pretrain on an empty demonstration set")
    device = torch.device(device)
    policy.to(device)
    # The archive stays on CPU; only minibatches cross to the accelerator.
    observations, actions, masks = dataset.as_tensors("cpu")

    count = observations.shape[0]
    for start in range(0, count, _OBSERVE_CHUNK):
        policy.observe(observations[start : start + _OBSERVE_CHUNK].to(device))

    generator = torch.Generator().manual_seed(0)
    permutation = torch.randperm(count, generator=generator)
    split = max(1, int(count * (1.0 - validation_fraction)))
    train_index, holdout_index = permutation[:split], permutation[split:]

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    history: list[dict] = []
    for epoch in range(epochs):
        policy.train()
        order = train_index[
            torch.randperm(train_index.numel(), generator=generator)
        ]
        losses = []
        for start in range(0, order.numel(), batch_size):
            batch = order[start : start + batch_size]
            if batch.numel() < 2:
                continue
            output = policy(
                observations[batch].to(device), masks[batch].to(device)
            )
            loss = F.cross_entropy(output.logits, actions[batch].to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
        }
        if holdout_index.numel() >= 2:
            policy.eval()
            with torch.no_grad():
                # Scored in chunks for the same reason the archive stays on CPU.
                total_loss = 0.0
                correct = 0
                seen = 0
                for start in range(0, holdout_index.numel(), batch_size):
                    batch = holdout_index[start : start + batch_size]
                    if batch.numel() == 0:
                        continue
                    targets = actions[batch].to(device)
                    output = policy(
                        observations[batch].to(device), masks[batch].to(device)
                    )
                    total_loss += float(
                        F.cross_entropy(output.logits, targets, reduction="sum").item()
                    )
                    correct += int(
                        (output.logits.argmax(dim=-1) == targets).sum().item()
                    )
                    seen += int(batch.numel())
                record["holdout_loss"] = total_loss / max(seen, 1)
                record["holdout_accuracy"] = correct / max(seen, 1)
        history.append(record)
        if progress is not None:
            progress(record)
    policy.train()
    return history
