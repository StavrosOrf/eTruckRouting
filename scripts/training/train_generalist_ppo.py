"""Train one generalist GraphPPO policy on several problem configurations.

Every problem keeps its own configuration; one policy, one optimizer and one
interaction budget are shared. Training draws scenarios from each problem's
``train`` seed namespace and selection uses ``validation``; ``test`` is never
touched here.

    PYTHONPATH=. python scripts/training/train_generalist_ppo.py \\
        --problems results/generalist_graphppo/<campaign>/problems.json \\
        --references results/generalist_graphppo/<campaign>/references \\
        --output results/generalist_graphppo/<campaign>/training --run-name full_s0

With ``--train-problems`` the policy trains on a subset and the others are
validated zero-shot; they never enter checkpoint selection.

Imitation beyond the greedy warm start (``algo/generalist_imitation.py``; all
off by default):

* ``--demonstrations DIR`` clones ``DIR/<problem>/train_demonstrations.json``
  (strong experts' train-seed episodes), falling back to the references;
* ``--bc-coefficient`` keeps a cross-entropy term on the labelled states in
  every PPO minibatch, annealed linearly over ``--bc-anneal-timesteps``;
* ``--dagger-expert teacher`` (or ``module:factory``, or ``PROBLEM=SPEC``)
  runs DAgger rounds every ``--dagger-every`` updates and aggregates the
  expert-labelled states into that set;
* ``--reward-references DIR`` measures the paired reward against other
  outcomes, e.g. the BKS folder (``DIR/<problem>/{train_demonstrations,validation}.json``).
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import asdict
from pathlib import Path

import torch

from algo.canonical_ppo import PPOConfig
from algo.generalist_imitation import (
    ImitationConfig,
    demonstration_episodes,
    peek_imitation_state,
    replay_dagger_archive,
    trainable_archive,
)
from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig
from algo.generalist_ppo import (
    GeneralistPPO,
    ReferenceTable,
    SuiteProblem,
    SuiteReward,
    evaluate_policy_suite,
    evaluation_envs,
    replay_demonstrations,
    summarize_suite,
)
from scripts.evaluation.canonical_harness import split_seeds


SUITE = Path("EVRoutingEnv/config_files/generalist_suite")
SUITE_PROBLEMS = (
    "ca_sameday", "ca_drayage_combined", "tx_longhaul", "tx_longhaul_congested", "pnw_longhaul_hos",
)


def load_problems(path: str | None) -> list[SuiteProblem]:
    if path is None:
        return [SuiteProblem(name, str(SUITE / f"{name}.yaml")) for name in SUITE_PROBLEMS]
    return [SuiteProblem(**row) for row in json.loads(Path(path).read_text())]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", default=None, help="JSON list of {name, config, teacher}; default: the suite.")
    parser.add_argument("--train-problems", nargs="*", default=None, help="Train on these only (others zero-shot).")
    parser.add_argument("--references", required=True, help="Directory of <problem>.json reference outcomes.")
    parser.add_argument("--reward-reference", default="problem", choices=["problem", "scenario"])
    parser.add_argument("--service-weight", type=float, default=2.0)
    parser.add_argument("--lost-weight", type=float, default=1.0)
    parser.add_argument("--late-weight", type=float, default=0.5)
    parser.add_argument("--success-bonus", type=float, default=1.0)
    parser.add_argument("--failure-penalty", type=float, default=2.0)
    parser.add_argument("--incompletion-penalty", type=float, default=2.0)
    parser.add_argument("--fleet-assignment", action="store_true", help="Plan 4.2 fleet-assignment features.")
    parser.add_argument("--charger-budget", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--action-head-layers", type=int, default=2)
    parser.add_argument("--nonlinear-edge-messages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--schedule-timesteps", type=int, default=None)
    parser.add_argument("--stop-after-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=40)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--target-kl", type=float, default=0.05)
    parser.add_argument("--train-scenarios", type=int, default=2048, help="Train seeds per problem.")
    parser.add_argument("--validation-scenarios", type=int, default=16, help="Validation seeds per problem.")
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--bc-episodes", type=int, default=64, help="Teacher episodes per problem (0: no BC).")
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    imitation = parser.add_argument_group("imitation (algo/generalist_imitation.py; off by default)")
    imitation.add_argument("--demonstrations", default=None,
                           help="Folder of <problem>/train_demonstrations.json; problems without one use the references.")
    imitation.add_argument("--reward-references", default=None,
                           help="Outcomes the reward is paired against (default: --references); also a BKS folder.")
    imitation.add_argument("--bc-coefficient", type=float, default=0.0,
                           help="Retained imitation: weight of the cross-entropy term in every PPO minibatch.")
    imitation.add_argument("--bc-coefficient-final", type=float, default=0.0)
    imitation.add_argument("--bc-anneal-timesteps", type=int, default=0,
                           help="Linear anneal of the coefficient to its final value (0: constant).")
    imitation.add_argument("--bc-batch-size", type=int, default=256)
    imitation.add_argument("--dagger-expert", nargs="*", default=[],
                           help="'teacher', a teacher name, 'module:factory', or PROBLEM=SPEC ('none' skips a problem).")
    imitation.add_argument("--dagger-every", type=int, default=10, help="Updates between DAgger rounds.")
    imitation.add_argument("--dagger-episodes", type=int, default=8, help="Episodes per problem per round.")
    imitation.add_argument("--dagger-envs-per-problem", type=int, default=4)
    imitation.add_argument("--dagger-beta", type=float, default=0.5, help="Probability that the expert's action runs.")
    imitation.add_argument("--dagger-beta-final", type=float, default=0.0)
    imitation.add_argument("--dagger-beta-anneal-timesteps", type=int, default=0, help="0: constant beta.")
    imitation.add_argument("--dagger-max-states", type=int, default=200_000,
                           help="DAgger states kept; the oldest episodes go first.")
    imitation.add_argument("--dagger-deterministic", action="store_true", help="The learner acts greedily in rounds.")
    imitation.add_argument("--dagger-pretrain-rounds", type=int, default=0,
                           help="DAgger rounds, each followed by --dagger-pretrain-epochs of cloning, before PPO.")
    imitation.add_argument("--dagger-pretrain-epochs", type=int, default=4)
    parser.add_argument("--init-from", default=None, help="Policy directory with best.pt/best_config.json.")
    parser.add_argument("--resume", default=None, help="Trusted complete training checkpoint.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollout-workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", default="results/generalist_graphppo/training")
    parser.add_argument("--run-name", default=None)
    return parser


def build_imitation_config(arguments) -> ImitationConfig | None:
    """``None`` unless an imitation flag asks for something (the trainer is then unchanged)."""
    if not (arguments.bc_coefficient > 0 or arguments.bc_coefficient_final > 0 or arguments.dagger_expert):
        if arguments.dagger_pretrain_rounds:
            raise SystemExit("--dagger-pretrain-rounds needs --dagger-expert")
        return None
    config = ImitationConfig(
        bc_coefficient=arguments.bc_coefficient,
        bc_coefficient_final=arguments.bc_coefficient_final,
        bc_anneal_timesteps=arguments.bc_anneal_timesteps,
        bc_batch_size=arguments.bc_batch_size,
        dagger_expert=tuple(arguments.dagger_expert),
        dagger_every=arguments.dagger_every,
        dagger_episodes=arguments.dagger_episodes,
        dagger_envs_per_problem=arguments.dagger_envs_per_problem,
        dagger_beta=arguments.dagger_beta,
        dagger_beta_final=arguments.dagger_beta_final,
        dagger_beta_anneal_timesteps=arguments.dagger_beta_anneal_timesteps,
        dagger_max_states=arguments.dagger_max_states,
        dagger_deterministic=arguments.dagger_deterministic,
    )
    retained = config.bc_coefficient > 0 or config.bc_coefficient_final > 0
    if retained and arguments.bc_episodes <= 0 and not config.dagger:
        raise SystemExit("--bc-coefficient needs demonstrations (--bc-episodes > 0) or --dagger-expert")
    if config.dagger and not retained:
        print("warning: DAgger labels train the policy only through --bc-coefficient or "
              "--dagger-pretrain-rounds; otherwise rounds only measure expert agreement", flush=True)
    return config


def print_dagger(record: dict) -> None:
    problems = sorted({key.split("/")[1] for key in record if key.count("/") == 2})
    print(
        f"[dagger] round={record['dagger/round']} beta={record['dagger/beta']:.2f} "
        f"episodes={record['dagger/episodes']} labelled={record['dagger/labelled']}/{record['dagger/decisions']} "
        f"misses={record['dagger/misses']} errors={record['dagger/expert_errors']} "
        f"set={record['dagger/set_states']} {record['dagger/seconds']:.0f}s "
        + " ".join(
            f"{name}={record.get(f'dagger/{name}/served_fraction', float('nan')):.2f}"
            f"/agree{record.get(f'dagger/{name}/agreement', float('nan')):.2f}"
            for name in problems
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    arguments = build_argument_parser().parse_args(argv)
    torch.set_num_threads(arguments.torch_threads)
    problems = load_problems(arguments.problems)
    names = [problem.name for problem in problems]
    trained = names if arguments.train_problems is None else list(arguments.train_problems)
    if not trained or not set(trained) <= set(names) or len(set(names)) != len(names):
        raise SystemExit(f"--train-problems must name distinct problems among {names}")
    training_problems = [problem for problem in problems if problem.name in trained]
    run_name = arguments.run_name or f"generalist_{int(time.time())}"
    destination = Path(arguments.output) / run_name
    destination.mkdir(parents=True, exist_ok=True)

    references = ReferenceTable.load(arguments.references, problems)
    missing = [name for name in names if name not in references.medians]
    if missing:
        raise SystemExit(f"no reference outcomes for {missing} in {arguments.references}")
    reward_references = references
    if arguments.reward_references:
        reward_references = ReferenceTable.load(arguments.reward_references, problems)
        missing = [name for name in names if name not in reward_references.medians]
        if missing:
            raise SystemExit(f"no reward reference outcomes for {missing} in {arguments.reward_references}")
    imitation = build_imitation_config(arguments)
    reward = SuiteReward(
        reference=arguments.reward_reference,
        service_weight=arguments.service_weight,
        lost_weight=arguments.lost_weight,
        late_weight=arguments.late_weight,
        success_bonus=arguments.success_bonus,
        failure_penalty=arguments.failure_penalty,
        incompletion_penalty=arguments.incompletion_penalty,
    )
    policy_config = GeneralistPolicyConfig(
        hidden_dim=arguments.hidden_dim,
        encoder_layers=arguments.encoder_layers,
        action_head_layers=arguments.action_head_layers,
        nonlinear_edge_messages=arguments.nonlinear_edge_messages,
        charger_budget=arguments.charger_budget,
        fleet_assignment=arguments.fleet_assignment,
    )
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
        target_kl=arguments.target_kl,
        schedule_timesteps=arguments.schedule_timesteps,
        seed=arguments.seed,
    )
    train_seeds = split_seeds("train", arguments.train_scenarios)
    validation_seeds = split_seeds("validation", arguments.validation_scenarios)
    # Every worker process is forked here, before the policy touches CUDA:
    # demonstration and DAgger replays, validation workers, then the trainer's
    # rollout workers (which also hold the DAgger sub-environments).
    retained = imitation is not None and (imitation.bc_coefficient > 0 or imitation.bc_coefficient_final > 0)
    demonstrations, sources = None, {}
    if arguments.bc_episodes > 0 and (not arguments.resume or retained):
        episodes, sources = demonstration_episodes(
            training_problems, [train_seeds] * len(training_problems), arguments.bc_episodes,
            demonstrations=arguments.demonstrations, references=references,
        )
        demonstrations = replay_demonstrations(
            training_problems, policy_config.observation_spec, episodes, workers=arguments.rollout_workers,
        )
        print(f"behaviour cloning on {demonstrations[2]['kept']}/{demonstrations[2]['steps']} demonstrated decisions; "
              f"outside the candidate set: {demonstrations[2]['misses']}", flush=True)
        for name, source in sources.items():
            print(f"  {name}: {source['used']} episodes from {source['source']}", flush=True)
    dagger_replay = None
    if arguments.resume and imitation is not None:
        archive = trainable_archive(peek_imitation_state(arguments.resume), training_problems)
        if archive:
            dagger_replay = replay_dagger_archive(
                training_problems, policy_config.observation_spec, archive, workers=arguments.rollout_workers,
            )
            print(f"rebuilt {sum(len(labels) for _, labels in dagger_replay)} DAgger states "
                  f"from {len(archive)} archived episodes", flush=True)
    validation_envs = evaluation_envs(
        problems, [validation_seeds] * len(problems), policy_config.observation_spec, reward, reward_references,
        workers=arguments.rollout_workers,
    )
    trainer = GeneralistPPO(
        policy_config,
        ppo_config,
        training_problems,
        reward_references,
        reward,
        [train_seeds] * len(training_problems),
        device=arguments.device,
        rollout_workers=arguments.rollout_workers,
        imitation=imitation,
    )
    if arguments.init_from and not arguments.resume:
        source = GeneralistActorCritic.load(arguments.init_from, prefix="best")
        if asdict(source.config) != asdict(policy_config):
            raise SystemExit(f"{arguments.init_from} has a different policy configuration")
        trainer.policy.load_state_dict(source.state_dict())
        print(f"initialised weights from {arguments.init_from}", flush=True)

    run_record = {
        "run_name": run_name,
        "arguments": vars(arguments),
        "problems": [asdict(problem) for problem in problems],
        "train_problems": trained,
        "policy_config": asdict(policy_config),
        "ppo_config": asdict(ppo_config),
        "reward": asdict(reward),
        "validation_seeds": validation_seeds,
        **({"imitation": asdict(imitation), "demonstration_sources": sources} if imitation is not None else {}),
    }
    (destination / "run_config.json").write_text(json.dumps(run_record, indent=2, default=str))

    validation_history: list[dict] = []
    best_score = float("-inf")
    if arguments.resume:
        restored = trainer.load_training(arguments.resume, dagger_replay=dagger_replay)
        validation_history = restored.get("validation_history", [])
        best_score = float(restored.get("best_score", best_score))
        print(f"resumed {trainer.total_steps} transitions", flush=True)
    started = time.perf_counter()
    stop_requested = False
    # Rollout and update throughput only: the window restarts after each
    # callback, so validation and checkpoint time are excluded.
    window = {"steps": trainer.total_steps, "time": started}

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, request_stop)

    def checkpoint() -> None:
        trainer.save_training(
            destination / "training.pt",
            {"best_score": best_score, "validation_history": validation_history},
        )
        (destination / "training_history.json").write_text(json.dumps(trainer.history, indent=2, sort_keys=True))

    def score_validation(update: int) -> None:
        nonlocal best_score
        rows = evaluate_policy_suite(
            trainer.policy, problems, [validation_seeds] * len(problems), reward_references, reward,
            device=trainer.device, envs=validation_envs,
        )
        summary = summarize_suite(rows, names)
        # Held-out problems are reported but never select a checkpoint.
        summary["selection_score"] = sum(summary[name]["score"] for name in trained) / len(trained)
        summary.update(update=update, timesteps=trainer.total_steps, elapsed_seconds=time.perf_counter() - started)
        validation_history.append(summary)
        (destination / "validation_history.json").write_text(json.dumps(validation_history, indent=2, sort_keys=True))
        if summary["selection_score"] > best_score:
            best_score = summary["selection_score"]
            trainer.save(destination, prefix="best")
        print(
            f"[validation] update={update} steps={trainer.total_steps} "
            f"selection={summary['selection_score']:.3f} "
            + " ".join(
                f"{name}={summary[name]['score']:.2f}/{summary[name]['served_fraction']:.2f}"
                for name in names
            ),
            flush=True,
        )

    def progress(record: dict) -> bool:
        # The record is the entry in trainer.history, so these reach training_history.json.
        now = time.perf_counter()
        # DAgger rounds are excluded too; they report their own seconds.
        busy = now - window["time"] - record.get("dagger/seconds", 0.0)
        record["decisions_per_second"] = (record["timesteps"] - window["steps"]) / max(busy, 1e-9)
        if trainer.device.type == "cuda":
            record["gpu_peak_gb"] = torch.cuda.max_memory_allocated(trainer.device) / 2**30
        if "dagger/round" in record:
            print_dagger(record)
        if record["update"] % 5 == 0 or record["update"] == 1:
            memory = f" gpu_peak={record['gpu_peak_gb']:.1f}GB" if "gpu_peak_gb" in record else ""
            imitation_ = (
                f" bc={record['bc_coefficient']:.3f}/{record.get('bc_loss', 0.0):.3f}" if "bc_coefficient" in record else ""
            )
            print(
                f"upd {record['update']:4d}/{record['num_updates']} steps={record['timesteps']:9d} "
                f"train_score={record['train_score']:.3f} entropy={record['entropy']:.3f} "
                f"kl={record['approx_kl']:.4f}{imitation_} {record['decisions_per_second']:.0f} dec/s{memory} "
                f"{record['elapsed_seconds']:.0f}s",
                flush=True,
            )
        stopping = stop_requested or (
            arguments.stop_after_timesteps is not None and trainer.total_steps >= arguments.stop_after_timesteps
        )
        if record["update"] % arguments.validate_every == 0 and not stop_requested:
            score_validation(record["update"])
        if stopping or record["update"] % arguments.checkpoint_every == 0:
            checkpoint()
        window.update(steps=trainer.total_steps, time=time.perf_counter())
        return stopping

    def print_epoch(record: dict) -> None:
        print(
            f"  bc epoch {record['epoch']}: loss={record['train_loss']:.4f} "
            f"holdout={record.get('holdout_loss', float('nan')):.4f} "
            f"acc={record.get('holdout_accuracy', float('nan')):.3f}",
            flush=True,
        )

    try:
        pretrain_record: dict = {}
        if demonstrations is not None:
            observations, labels, coverage = demonstrations
            demonstrations = None
            if retained:
                trainer.set_demonstrations(observations, labels)
            if not arguments.resume:
                history = trainer.pretrain(
                    observations, labels, epochs=arguments.pretrain_epochs,
                    learning_rate=arguments.pretrain_learning_rate, progress=print_epoch,
                )
                pretrain_record = {"coverage": coverage, "epochs": history}
                if imitation is not None:
                    pretrain_record["sources"] = sources
            del observations, labels
        if not arguments.resume and trainer.envs.num_dagger and arguments.dagger_pretrain_rounds > 0:
            # Classic DAgger before PPO: collect with the current policy, then clone the aggregate.
            pretrain_record["dagger_rounds"] = []
            for _ in range(arguments.dagger_pretrain_rounds):
                round_record = trainer.dagger_round(observe=True)
                print_dagger(round_record)
                observations, labels = trainer.imitation_data.flat()
                round_record["epochs"] = trainer.pretrain(
                    observations, list(labels), epochs=arguments.dagger_pretrain_epochs,
                    learning_rate=arguments.pretrain_learning_rate, progress=print_epoch, observe=False,
                )
                pretrain_record["dagger_rounds"].append(round_record)
                del observations, labels
        if pretrain_record:
            (destination / "pretrain_history.json").write_text(json.dumps(pretrain_record, indent=2, sort_keys=True))
            trainer.save(destination, prefix="pretrained")
            score_validation(0)
            checkpoint()
        window.update(steps=trainer.total_steps, time=time.perf_counter())
        trainer.learn(progress)
        checkpoint()
        if not stop_requested:
            score_validation(trainer.completed_updates)
            checkpoint()
        trainer.save(destination, prefix="final")
        _, fallbacks = trainer.envs.positions()
        run_record["reference_fallbacks"] = fallbacks
        (destination / "run_config.json").write_text(json.dumps(run_record, indent=2, sort_keys=True, default=str))
        print(f"done in {time.perf_counter() - started:.0f}s -> {destination}", flush=True)
    finally:
        trainer.close()
        validation_envs.close()


if __name__ == "__main__":
    main()
