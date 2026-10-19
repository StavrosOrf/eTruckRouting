"""Measure actor/value encoder gradients on a controlled train-only rollout.

This diagnostic performs no optimizer updates and publishes no learned policy.
Its rollout is controlled by the recorded seed; it is not an exact replay of
the first rollout from the running GPU training process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from algo.canonical_policy import CanonicalActorCritic
from algo.canonical_ppo import CanonicalPPO, PPOConfig, RewardShaping
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from scripts.evaluation.canonical_harness import split_seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    run = json.loads((root / "run_config.json").read_text())
    config = json.loads((root / "resolved_config.json").read_text())
    torch.set_num_threads(1)
    policy = CanonicalActorCritic.load(root, prefix="pretrained")
    expected_hash = hashlib.sha256((root / "pretrained.pt").read_bytes()).hexdigest()
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / f"evrp_gradient_{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["EVRP_SCENARIO_CACHE"] = str(scratch / "scenarios")
    os.environ["MPLCONFIGDIR"] = str(scratch / "matplotlib")

    def factory():
        return EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)

    trainer = CanonicalPPO(
        policy.config, PPOConfig(**run["ppo_config"]), factory,
        device="cpu", seed_stream=split_seeds("train", 20000),
        shaping=RewardShaping(**run["reward_shaping"]), rollout_workers=7,
    )
    trainer.policy.load_state_dict(policy.state_dict())
    torch.set_num_threads(4)
    records = []
    try:
        observations, masks = trainer.envs.reset()
        trainer.policy.observe(trainer._to_tensor(observations))
        rollout, _, _ = trainer._collect_rollout(
            trainer._to_tensor(observations), trainer._to_bool(masks)
        )
        encoder = list(trainer.policy.encoder.parameters())
        for start in range(0, trainer.config.batch_size, trainer.config.minibatch_size):
            batch = slice(start, start + trainer.config.minibatch_size)
            logprobs, entropy, values = trainer.policy.evaluate_actions(
                rollout["observations"][batch], rollout["actions"][batch], rollout["masks"][batch]
            )
            ratio = (logprobs - rollout["logprobs"][batch]).exp()
            advantages = rollout["advantages"][batch]
            if trainer.config.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            actor_loss = -torch.minimum(
                ratio * advantages,
                ratio.clamp(1 - trainer.config.clip_range, 1 + trainer.config.clip_range) * advantages,
            ).mean() - trainer.config.entropy_coefficient * entropy.mean()
            returns = rollout["returns"][batch]
            if trainer.config.value_clip_range > 0:
                old = rollout["values"][batch]
                clipped = old + (values - old).clamp(-trainer.config.value_clip_range, trainer.config.value_clip_range)
                value_loss = 0.5 * torch.maximum((values - returns).square(), (clipped - returns).square()).mean()
            else:
                value_loss = 0.5 * (values - returns).square().mean()
            weighted_value = trainer.config.value_coefficient * value_loss
            actor_grads = torch.autograd.grad(actor_loss, encoder, retain_graph=True, allow_unused=True)
            value_grads = torch.autograd.grad(weighted_value, encoder, allow_unused=True)
            actor_sq = sum(float(gradient.detach().square().sum()) for gradient in actor_grads if gradient is not None)
            value_sq = sum(float(gradient.detach().square().sum()) for gradient in value_grads if gradient is not None)
            dot = sum(float(left.detach().mul(right.detach()).sum()) for left, right in zip(actor_grads, value_grads) if left is not None and right is not None)
            records.append({
                "minibatch_start": start,
                "encoder_actor_gradient_norm": actor_sq ** 0.5,
                "encoder_weighted_value_gradient_norm": value_sq ** 0.5,
                "value_to_actor_norm_ratio": (value_sq / actor_sq) ** 0.5 if actor_sq else None,
                "actor_value_cosine": dot / (actor_sq * value_sq) ** 0.5 if actor_sq * value_sq else None,
                "value_loss": float(value_loss.detach()),
            })
            print(json.dumps(records[-1]), flush=True)
        result = {
            "status": "completed", "scope": "initial BC-only actor; controlled train-only rollout",
            "optimizer_updates": 0, "learned_policy_published": False,
            "train_rollout_decisions": trainer.total_steps,
            "reward_scale": trainer.reward_scale.scale,
            "median_value_to_actor_encoder_gradient_norm_ratio": float(np.median([row["value_to_actor_norm_ratio"] for row in records if row["value_to_actor_norm_ratio"] is not None])),
            "mean_encoder_gradient_cosine": float(np.mean([row["actor_value_cosine"] for row in records if row["actor_value_cosine"] is not None])),
            "records": records,
            "interpretation": "gradient magnitude/alignment diagnostic; causal performance effects require a training ablation",
        }
        if hashlib.sha256((root / "pretrained.pt").read_bytes()).hexdigest() != expected_hash:
            raise ValueError("diagnostic input policy changed")
        (root / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    finally:
        trainer.envs.close()


if __name__ == "__main__":
    main()
