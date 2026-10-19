"""Train one canonical (state-encoder, action-head) policy on the joint problem.

Every learning variant is launched through this single entry point, so the
interaction budget, optimizer settings, reward shaping, and seed namespaces are
identical by construction and only the architecture differs.

Training draws scenarios from the ``train`` seed namespace.  Periodic scoring
uses the ``validation`` namespace.  The ``test`` namespace is never touched
here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch

from algo.canonical_policy import CanonicalPolicyConfig
from algo.canonical_ppo import (
    CanonicalPPO,
    CurriculumStage,
    PPOConfig,
    RewardShaping,
)
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.action_mask import policy_action_mask
from EVRoutingEnv.utils.utils import load_config
from scripts.evaluation.canonical_harness import (
    evaluate_policy,
    selection_score,
    split_seeds,
    summarize,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="EVRoutingEnv/config_files/config_joint.yaml"
    )
    parser.add_argument(
        "--state-encoder",
        default="hetero_graph",
        choices=["flat", "deep_sets", "hetero_graph", "attention"],
    )
    parser.add_argument(
        "--action-head",
        default="self_attention",
        choices=["independent", "complete_gcn", "self_attention"],
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--schedule-timesteps", type=int, default=None)
    parser.add_argument("--stop-after-timesteps", type=int, default=None)
    parser.add_argument(
        "--normalize-action-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--action-entity-links", action="store_true")
    parser.add_argument("--nonlinear-edge-messages", action="store_true")
    parser.add_argument("--progress-action-features", action="store_true")
    parser.add_argument("--completion-shield", action="store_true")
    parser.add_argument("--completion-capacity-lookahead", action="store_true")
    parser.add_argument("--completion-capacity-exact-lookahead", action="store_true")
    parser.add_argument("--completion-work-escape-lookahead", action="store_true")
    parser.add_argument(
        "--resume", default=None, help="Trusted complete training checkpoint."
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-output-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--action-head-layers", type=int, default=2)
    parser.add_argument("--action-attention-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-scenarios", type=int, default=20_000)
    parser.add_argument("--validation-scenarios", type=int, default=25)
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--success-bonus", type=float, default=3000.0)
    parser.add_argument("--incompletion-penalty", type=float, default=4000.0)
    parser.add_argument("--speed-bonus", type=float, default=0.0)
    parser.add_argument("--stranding-penalty", type=float, default=0.0)
    parser.add_argument("--energy-margin-bonus", type=float, default=0.0)
    parser.add_argument("--all-served-bonus", type=float, default=0.0)
    parser.add_argument(
        "--travel-time-bonus",
        type=float,
        default=0.0,
        help="Terminal bonus for fleet travel hours saved, paid only on success.",
    )
    parser.add_argument(
        "--time-multiplier",
        type=float,
        default=None,
        help=(
            "Override rewards.time_multiplier, the dense operating-time "
            "penalty. The config default of 1.0 leaves it "
            "negligible beside the 500-per-delivery bonus."
        ),
    )
    parser.add_argument(
        "--selection-objective",
        default="travel_time",
        choices=["travel_time", "makespan", "operating_time", "distance"],
        help="Metric that breaks ties between equally feasible checkpoints.",
    )
    parser.add_argument("--delivery-bonus", type=float, default=None)
    parser.add_argument("--disable-reward-shaping", action="store_true")
    parser.add_argument(
        "--policy-action-mask",
        default="hard",
        choices=["hard", "structural"],
        help=(
            "Mask handed to the policy. 'hard' is the proposed method. "
            "'structural' removes the feasibility mask and keeps only the "
            "slots that denote an action at all, so the policy must learn "
            "feasibility; the observation and candidate set are unchanged."
        ),
    )
    parser.add_argument(
        "--invalid-action-mode",
        default="terminate",
        choices=["terminate", "penalize"],
        help=(
            "What executing an infeasible action does. 'terminate' is the "
            "simulator's own semantics (the truck strands). 'penalize' refuses "
            "the action, leaves the state untouched, and continues until the "
            "invalid-action budget is spent -- the charitable setting for the "
            "unmasked ablation."
        ),
    )
    parser.add_argument(
        "--invalid-action-penalty",
        type=float,
        default=100.0,
        help="Magnitude charged per refused action under --invalid-action-mode penalize.",
    )
    parser.add_argument(
        "--invalid-action-budget",
        type=int,
        default=64,
        help="Refused actions allowed per episode before the truck is failed.",
    )
    parser.add_argument(
        "--disable-routing-action-features",
        action="store_true",
        help=(
            "Zero the per-action leg-cost and detour columns. Keeps the "
            "observation width and network shape identical, so this isolates "
            "what those features contribute at a matched budget."
        ),
    )
    parser.add_argument(
        "--ablate-features",
        nargs="+",
        default=None,
        choices=["queue", "active_truck", "depot", "edges"],
        help=(
            "Blank named canonical feature blocks. The observation width, "
            "network shape, and budget are unchanged, so an arm differs from "
            "its control only in what the rows carry."
        ),
    )
    parser.add_argument(
        "--ablate-state-pooling",
        action="store_true",
        help=(
            "Withhold the pooled state embedding from the action head, so each "
            "candidate is scored from its own row alone. The critic keeps the "
            "embedding."
        ),
    )
    parser.add_argument("--output", default="results/canonical/training")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=1,
        help=(
            "Processes that step the vector environment. The simulator is pure "
            "Python, so one process is bound to a single core regardless of "
            "--num-envs; anything above 1 splits the sub-environments across "
            "cores. The seed stream and reward shaping are unchanged."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--init-from",
        default=None,
        help="Directory of a trained policy to initialise from (weights only).",
    )
    parser.add_argument("--reset-value-head", action="store_true")
    parser.add_argument(
        "--curriculum",
        default=None,
        help="JSON file with a list of curriculum stages (name/overrides/advance_success).",
    )
    parser.add_argument(
        "--pretrain-demonstrations",
        type=int,
        default=0,
        help="Train scenarios rolled by the demonstrator before PPO (0 disables).",
    )
    parser.add_argument(
        "--demonstrations",
        default=None,
        help="Cached demonstration .npz from collect_demonstrations.py.",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=12)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--expert-coefficient", type=float, default=0.0)
    parser.add_argument("--expert-batch-size", type=int, default=128)
    parser.add_argument("--expert-decay-timesteps", type=int, default=0)
    parser.add_argument(
        "--demonstrator",
        default="mpc",
        choices=["mpc", "heuristic"],
        help="Controller used for behaviour cloning; only ever run on train seeds.",
    )
    return parser


def _build_demonstrator(name: str):
    from EVRoutingEnv.baselines.heuristics.canonical_baselines import (
        GreedyHeuristicPolicy,
        HeuristicParameters,
        MPCParameters,
        RollingHorizonMPCPolicy,
    )

    if name == "heuristic":
        return GreedyHeuristicPolicy(
            HeuristicParameters(
                energy_safety_factor=1.15, target_soc=1.0, demand_weight=2.0
            )
        )
    return RollingHorizonMPCPolicy(
        MPCParameters(horizon=6, branching=2, energy_safety_factor=1.15, target_soc=0.8)
    )


def main() -> None:
    arguments = build_argument_parser().parse_args()
    torch.set_num_threads(max(1, arguments.torch_threads))

    run_name = (
        arguments.run_name or f"{arguments.state_encoder}__{arguments.action_head}"
    )
    destination = Path(arguments.output) / run_name
    destination.mkdir(parents=True, exist_ok=True)

    config = load_config(arguments.config)
    if arguments.resume and arguments.init_from:
        raise ValueError("choose resume or initialization, not both")
    if arguments.completion_shield:
        config["environment"]["completion_shield"] = True
    if arguments.completion_capacity_lookahead:
        config["environment"]["completion_shield"] = True
        config["environment"]["completion_capacity_lookahead"] = True
    if arguments.completion_capacity_exact_lookahead:
        config["environment"]["completion_shield"] = True
        config["environment"]["completion_capacity_lookahead"] = True
        config["environment"]["completion_capacity_exact_lookahead"] = True
    if arguments.completion_work_escape_lookahead:
        config["environment"]["completion_shield"] = True
        config["environment"]["completion_work_escape_lookahead"] = True
    if arguments.disable_routing_action_features:
        config["environment"]["routing_action_features"] = False
        print("routing action features zeroed (ablation arm)", flush=True)
    if arguments.ablate_features:
        config["environment"]["ablate_features"] = list(arguments.ablate_features)
        print(
            f"feature blocks zeroed (ablation arm): {arguments.ablate_features}",
            flush=True,
        )
    if arguments.ablate_state_pooling:
        print("state pooling withheld from the action head (ablation arm)", flush=True)
    config["environment"]["policy_action_mask"] = arguments.policy_action_mask
    config["environment"]["invalid_action_mode"] = arguments.invalid_action_mode
    config["environment"]["invalid_action_budget"] = int(
        arguments.invalid_action_budget
    )
    config["rewards"]["invalid_action_penalty"] = float(
        arguments.invalid_action_penalty
    )
    if arguments.policy_action_mask == "structural":
        print(
            "feasibility mask REMOVED (ablation arm): infeasible candidates stay "
            f"selectable and are {arguments.invalid_action_mode}d on execution",
            flush=True,
        )
    if arguments.time_multiplier is not None:
        config["rewards"]["time_multiplier"] = float(arguments.time_multiplier)
        print(
            f"dense operating-time penalty: {arguments.time_multiplier} per truck-hour",
            flush=True,
        )
    if arguments.delivery_bonus is not None:
        config["rewards"]["delivery_bonus"] = float(arguments.delivery_bonus)
        print(f"delivery bonus: {arguments.delivery_bonus}", flush=True)

    def factory():
        return EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)

    probe = factory()
    probe.reset(seed=0)
    policy_config = CanonicalPolicyConfig.from_env(
        probe,
        state_encoder=arguments.state_encoder,
        action_head=arguments.action_head,
        hidden_dim=arguments.hidden_dim,
        encoder_output_dim=arguments.encoder_output_dim,
        encoder_layers=arguments.encoder_layers,
        action_head_layers=arguments.action_head_layers,
        action_attention_heads=arguments.action_attention_heads,
        allow_infeasible_actions=arguments.policy_action_mask == "structural",
        ablate_state_pooling=arguments.ablate_state_pooling,
        normalize_action_features=arguments.normalize_action_features,
        action_entity_links=arguments.action_entity_links,
        nonlinear_edge_messages=arguments.nonlinear_edge_messages,
        progress_action_features=arguments.progress_action_features,
    )
    probe.close()

    ppo_config = PPOConfig(
        total_timesteps=arguments.total_timesteps,
        num_envs=arguments.num_envs,
        rollout_steps=arguments.rollout_steps,
        epochs=arguments.epochs,
        minibatch_size=arguments.minibatch_size,
        learning_rate=arguments.learning_rate,
        entropy_coefficient=arguments.entropy_coefficient,
        gamma=arguments.gamma,
        gae_lambda=arguments.gae_lambda,
        schedule_timesteps=arguments.schedule_timesteps,
        seed=arguments.seed,
        expert_coefficient=arguments.expert_coefficient,
        expert_batch_size=arguments.expert_batch_size,
        expert_decay_timesteps=arguments.expert_decay_timesteps,
        expert_archive_sha256=(
            hashlib.file_digest(Path(arguments.demonstrations).open("rb"), "sha256").hexdigest()
            if arguments.expert_coefficient > 0 and arguments.demonstrations else None
        ),
    )
    shaping = RewardShaping(
        enabled=not arguments.disable_reward_shaping,
        success_bonus=arguments.success_bonus,
        incompletion_penalty=arguments.incompletion_penalty,
        speed_bonus=arguments.speed_bonus,
        horizon_hours=float(config["environment"]["max_time"]),
        stranding_penalty=arguments.stranding_penalty,
        energy_margin_bonus=arguments.energy_margin_bonus,
        all_served_bonus=arguments.all_served_bonus,
        travel_time_bonus=arguments.travel_time_bonus,
        # Both trucks may drive for the whole horizon, so fleet travel hours are
        # bounded by trucks x horizon rather than by the horizon alone.
        travel_reference_hours=float(config["environment"]["max_time"])
        * float(config["environment"]["num_trucks"]),
    )

    curriculum = None
    stage_factory = None
    if arguments.curriculum:
        stages = json.loads(Path(arguments.curriculum).read_text())
        curriculum = [CurriculumStage(**stage) for stage in stages]

        def stage_factory(stage, _config=config):
            staged = stage.apply(_config)
            return lambda: EventDrivenTruckEnv(
                staged, verbose=False, enable_plotting=False
            )

        print(
            "curriculum stages: "
            + " -> ".join(f"{s.name}(>={s.advance_success})" for s in curriculum),
            flush=True,
        )

    trainer = CanonicalPPO(
        policy_config,
        ppo_config,
        factory,
        device=arguments.device,
        seed_stream=split_seeds("train", arguments.train_scenarios),
        shaping=shaping,
        curriculum=curriculum,
        stage_factory=stage_factory,
        rollout_workers=arguments.rollout_workers,
    )
    if arguments.init_from:
        # Policy weights include the observation-normalizer buffers. The
        # optimizer, return scaler, interaction count and curriculum restart.
        from algo.canonical_policy import CanonicalActorCritic

        source = CanonicalActorCritic.load(arguments.init_from, prefix="best")
        if source.config.shape.flat_size != policy_config.shape.flat_size:
            raise SystemExit(
                f"{arguments.init_from} was trained on a different observation width"
            )
        trainer.policy.load_state_dict(source.state_dict())
        trainer.policy.to(trainer.device)
        if arguments.reset_value_head:
            for module in trainer.policy.value_head.modules():
                if isinstance(module, torch.nn.Linear):
                    module.reset_parameters()
            print("reset value head for fresh-return optimization", flush=True)
        print(f"initialised weights from {arguments.init_from}", flush=True)

    validation_env = factory()
    validation_seeds = split_seeds("validation", arguments.validation_scenarios)
    validation_history: list[dict] = []
    # selection_score is "lower is better" (negated success, then the objective),
    # so the initial bar has to be worse than anything a run can produce.
    best_score = (float("inf"), float("inf"))
    if arguments.resume:
        restored = trainer.load_training(arguments.resume)
        validation_history = restored.get("validation_history", [])
        best_score = tuple(restored.get("best_score", best_score))
        print(f"resumed {trainer.total_steps} transitions", flush=True)
    started = time.perf_counter()
    pretrain_history: list[dict] = []
    dataset = None
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, request_stop)

    (destination / "resolved_config.json").write_text(json.dumps(config, indent=2))
    (destination / "run_config.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "arguments": vars(arguments),
                "policy_config": policy_config.__dict__,
                "ppo_config": ppo_config.__dict__,
                "reward_shaping": shaping.__dict__,
                "validation_seeds": validation_seeds,
            },
            indent=2,
            default=str,
        )
    )

    if not arguments.resume and (
        arguments.demonstrations or arguments.pretrain_demonstrations > 0
    ):
        from algo.behavior_cloning import (
            DemonstrationSet,
            collect_demonstrations,
            pretrain_policy,
        )

        if arguments.demonstrations:
            archive = np.load(arguments.demonstrations)
            dataset = DemonstrationSet(
                observations=list(archive["observations"]),
                actions=[int(value) for value in archive["actions"]],
                masks=list(archive["masks"]),
            )
            print(
                f"loaded {len(dataset)} cached demonstration transitions from "
                f"{arguments.demonstrations}",
                flush=True,
            )
        else:
            demonstration_env = factory()
            try:
                print(
                    f"collecting {arguments.pretrain_demonstrations} demonstrations "
                    f"from the {arguments.demonstrator} controller...",
                    flush=True,
                )
                dataset = collect_demonstrations(
                    demonstration_env,
                    _build_demonstrator(arguments.demonstrator),
                    # Demonstrations sit at the far end of the train namespace so
                    # they never overlap the PPO rollout seeds.
                    split_seeds("train", arguments.train_scenarios)[
                        -arguments.pretrain_demonstrations :
                    ],
                )
            finally:
                demonstration_env.close()
            print(
                f"  kept {len(dataset)} transitions from "
                f"{dataset.successful_episodes}/{dataset.episodes} successful episodes",
                flush=True,
            )
        if len(dataset) > 0:
            pretrain_history = pretrain_policy(
                trainer.policy,
                dataset,
                epochs=arguments.pretrain_epochs,
                learning_rate=arguments.pretrain_learning_rate,
                device=trainer.device,
                progress=lambda record: print(
                    f"  bc epoch {record['epoch']}: loss={record['train_loss']:.4f} "
                    f"holdout={record.get('holdout_loss', float('nan')):.4f} "
                    f"acc={record.get('holdout_accuracy', float('nan')):.3f}",
                    flush=True,
                ),
            )
            (destination / "pretrain_history.json").write_text(
                json.dumps(pretrain_history, indent=2, sort_keys=True)
            )
        else:
            print(
                "  no successful demonstrations; skipping behaviour cloning", flush=True
            )

    def score_validation(update: int) -> None:
        nonlocal best_score
        trainer.policy.eval()

        def policy(env, observation, info):
            batch = torch.as_tensor(
                np.expand_dims(observation, 0),
                dtype=torch.float32,
                device=trainer.device,
            )
            mask = torch.as_tensor(
                np.expand_dims(policy_action_mask(env), 0),
                dtype=torch.bool,
                device=trainer.device,
            )
            actions, _, _ = trainer.policy.act(batch, mask, deterministic=True)
            return int(actions[0].item())

        try:
            summary = summarize(
                evaluate_policy(validation_env, policy, validation_seeds)
            )
        finally:
            trainer.policy.train()

        summary["update"] = update
        summary["timesteps"] = trainer.total_steps
        summary["elapsed_seconds"] = time.perf_counter() - started
        validation_history.append(summary)
        (destination / "validation_history.json").write_text(
            json.dumps(validation_history, indent=2, sort_keys=True)
        )
        candidate = selection_score(summary, arguments.selection_objective)
        if candidate < best_score:
            best_score = candidate
            trainer.save(destination, prefix="best")
        travel = summary["mean_travel_time_successful"]
        makespan = summary["mean_makespan_successful"]
        print(
            f"[validation] update={update} steps={trainer.total_steps} "
            f"success={summary['success_rate']:.3f} "
            f"frac={summary['mean_completed_fraction']:.3f} "
            f"travel={'n/a' if travel is None else f'{travel:.1f}'} "
            f"makespan={'n/a' if makespan is None else f'{makespan:.1f}'}",
            flush=True,
        )

    def checkpoint() -> None:
        trainer.save_training(
            destination / "training.pt",
            {
                "best_score": best_score,
                "validation_history": validation_history,
            },
        )

    def progress(record: dict) -> bool:
        if record["update"] % 5 == 0 or record["update"] == 1:
            print(
                f"upd {record['update']:4d}/{record['num_updates']} "
                f"steps={record['timesteps']:8d} "
                f"reward={record['mean_episode_reward']:9.1f} "
                f"train_success={record['success_rate']:.3f} "
                f"travel={record['mean_travel_time']:6.1f} "
                f"entropy={record['entropy']:.3f} kl={record['approx_kl']:.4f} "
                f"stage={record.get('curriculum_stage', 'none')} "
                f"expert_loss={record.get('expert_loss', 0):.4f} "
                f"expert_weight={record.get('expert_coefficient', 0):.4f} "
                f"{record['elapsed_seconds']:.0f}s",
                flush=True,
            )
        stopping = stop_requested or (
            arguments.stop_after_timesteps is not None
            and trainer.total_steps >= arguments.stop_after_timesteps
        )
        if record["update"] % arguments.validate_every == 0 and not stop_requested:
            score_validation(record["update"])
        if stopping or record["update"] % arguments.checkpoint_every == 0:
            checkpoint()
        return stopping

    try:
        if arguments.expert_coefficient > 0:
            if dataset is None:
                from algo.behavior_cloning import DemonstrationSet

                with np.load(arguments.demonstrations) as archive:
                    dataset = DemonstrationSet(
                        observations=list(archive["observations"]),
                        actions=[int(value) for value in archive["actions"]],
                        masks=list(archive["masks"]),
                    )
            trainer.set_expert_demonstrations(dataset)
            print(f"retaining {len(dataset)} expert transitions during PPO", flush=True)
        if pretrain_history:
            score_validation(0)
        history = trainer.learn(progress)
        checkpoint()
        if not stop_requested:
            score_validation(trainer.completed_updates)
            checkpoint()
        trainer.save(destination, prefix="final")
        (destination / "training_history.json").write_text(
            json.dumps(history.as_dict(), indent=2, sort_keys=True)
        )
        (destination / "curriculum_log.json").write_text(
            json.dumps(trainer.stage_log, indent=2, sort_keys=True)
        )
        (destination / "run_config.json").write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "arguments": vars(arguments),
                    "policy_config": policy_config.__dict__,
                    "ppo_config": ppo_config.__dict__,
                    "reward_shaping": shaping.__dict__,
                    "validation_seeds": validation_seeds,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        print(
            f"done in {time.perf_counter() - started:.0f}s -> {destination}", flush=True
        )
    finally:
        trainer.close()
        validation_env.close()


if __name__ == "__main__":
    main()
