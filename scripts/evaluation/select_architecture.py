"""Freeze the winning (state encoder, action head) pair on validation evidence.

Selection reads only ``validation_history.json`` from each sweep run, ranks by
the pre-declared rule -- success rate first, then the campaign objective among
successful episodes -- and writes ``selected_architecture.json``.  No test
scenario is read here, and the rule is fixed before the numbers are inspected.

Runs are compared at their largest *common* interaction budget so a run that
happened to finish more updates cannot win on extra experience alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluation.canonical_harness import OBJECTIVE_KEYS, selection_score


def load_runs(root: Path) -> dict[str, list[dict]]:
    runs = {}
    for directory in sorted(root.iterdir()):
        history_path = directory / "validation_history.json"
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text())
        if history:
            runs[directory.name] = history
    return runs


def common_budget(runs: dict[str, list[dict]]) -> int:
    """Largest interaction budget every run actually reached."""
    return min(
        max(int(entry["timesteps"]) for entry in history) for history in runs.values()
    )


def best_at_budget(history: list[dict], budget: int, objective: str) -> dict:
    """Best validation checkpoint at or below the shared budget."""
    eligible = [entry for entry in history if int(entry["timesteps"]) <= budget]
    if not eligible:
        eligible = [min(history, key=lambda entry: int(entry["timesteps"]))]
    return min(eligible, key=lambda entry: selection_score(entry, objective))


def _architecture_of(run_directory: Path, run_name: str) -> tuple[str, str]:
    """Read the architecture from the saved policy config, not the run name.

    Sweep runs are named ``encoder__head``, but ablation runs are named after
    the thing they vary, so parsing the directory name would mislabel them.
    """
    config_path = run_directory / "best_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        encoder = config.get("state_encoder")
        head = config.get("action_head")
        if encoder and head:
            return str(encoder), str(head)
    encoder, _, head = run_name.partition("__")
    return encoder, head


def environment_overrides_for(checkpoint: Path) -> dict[str, dict]:
    """Recover the environment a run trained in, from its own run_config.json.

    Ablation arms change the environment, not only the network: an unmasked run
    needs the structural mask, and a feature-ablated run needs the same blocks
    blanked.  Re-scoring any of them under the default environment would measure
    a policy on inputs it never saw, so the environment travels with the
    checkpoint rather than being assumed.
    """
    import json

    path = Path(checkpoint) / "run_config.json"
    if not path.exists():
        return {}
    arguments = json.loads(path.read_text()).get("arguments", {})
    environment: dict[str, object] = {}
    for key in (
        "policy_action_mask",
        "invalid_action_mode",
        "invalid_action_budget",
        "ablate_features",
    ):
        value = arguments.get(key)
        if value:
            environment[key] = value
    if arguments.get("disable_routing_action_features"):
        environment["routing_action_features"] = False
    return {"environment": environment} if environment else {}


def _score_shard(job: tuple[str, str, list[int]]) -> tuple[str, list[dict]]:
    """Score one checkpoint on one shard of validation seeds, in its own process.

    Rescoring is embarrassingly parallel over (run, scenario) and single-episode
    inference is CPU-bound in the simulator, so sharding is the difference
    between minutes and an hour per selection round.
    """
    import numpy as np
    import torch

    from algo.canonical_policy import CanonicalActorCritic
    from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
    from EVRoutingEnv.state.action_mask import policy_action_mask
    from EVRoutingEnv.utils.utils import load_config
    from scripts.evaluation.canonical_harness import evaluate_policy

    checkpoint, config_path, seeds = job
    torch.set_num_threads(1)
    policy = CanonicalActorCritic.load(Path(checkpoint), prefix="best")
    policy.eval()

    def act(env, observation, info):
        batch = torch.as_tensor(np.expand_dims(observation, 0), dtype=torch.float32)
        # Whatever mask this run trained against: an unmasked arm re-scored
        # under the hard mask would be measured on a policy it never was.
        mask = torch.as_tensor(
            np.expand_dims(policy_action_mask(env), 0), dtype=torch.bool
        )
        actions, _, _ = policy.act(batch, mask, deterministic=True)
        return int(actions[0].item())

    config = load_config(config_path)
    for section, values in environment_overrides_for(Path(checkpoint)).items():
        config[section].update(values)
    environment = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        outcomes = evaluate_policy(environment, act, seeds)
    finally:
        environment.close()
    return checkpoint, [outcome.as_dict() for outcome in outcomes]


def rescore_on_validation(
    runs: list[str],
    training_root: Path,
    config_path: str,
    scenarios: int,
    offset: int = 0,
    workers: int = 1,
    objective: str = "travel_time",
) -> dict[str, dict]:
    """Re-evaluate saved checkpoints on further validation scenarios.

    Periodic scoring keeps the best of many checkpoints on a small scenario set,
    which is optimistically biased. ``offset`` skips the scenarios that were used
    to pick those checkpoints, so the finalist comparison is independent of the
    selection that produced them while still reading only the validation split.
    """
    from concurrent.futures import ProcessPoolExecutor

    from scripts.evaluation.canonical_harness import (
        EpisodeOutcome,
        split_seeds,
        summarize,
    )

    seeds = split_seeds("validation", offset + scenarios)[offset:]
    eligible = [
        name for name in runs if (training_root / name / "best.pt").exists()
    ]
    shards = max(1, min(workers, len(seeds)))
    jobs = [
        (str(training_root / name), config_path, seeds[index::shards])
        for name in eligible
        for index in range(shards)
    ]
    collected: dict[str, list[EpisodeOutcome]] = {name: [] for name in eligible}
    by_checkpoint = {str(training_root / name): name for name in eligible}

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            completed = list(pool.map(_score_shard, jobs))
    else:
        completed = [_score_shard(job) for job in jobs]
    for checkpoint, rows in completed:
        collected[by_checkpoint[checkpoint]].extend(
            EpisodeOutcome(**row) for row in rows
        )

    results: dict[str, dict] = {}
    for name in eligible:
        summary = summarize(collected[name])
        results[name] = summary
        print(
            f"  rescored {name:<30} success={summary['success_rate']:.3f} "
            f"completed={summary['mean_completed_fraction']:.3f} "
            f"travel={_format(summary['mean_travel_time_successful'])} "
            f"makespan={_format(summary['mean_makespan_successful'])}",
            flush=True,
        )
    return dict(
        sorted(results.items(), key=lambda item: selection_score(item[1], objective))
    )


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _banded_key(
    entry: dict, objective: str, best_success: float, success_tolerance: float
) -> tuple[int, float, float]:
    """Sort key that puts every feasibility-tied run in one band, best first.

    Runs within ``success_tolerance`` of the best success rate form band 0 and
    are ordered by the objective alone; everything else follows, ordered by
    success. Used for the shortlist so the re-scoring round sees the runs that
    are actually competitive on the campaign objective.
    """
    value = entry.get(OBJECTIVE_KEYS[objective])
    value = float(value) if value is not None else float("inf")
    success = float(entry["success_rate"])
    in_band = success >= best_success - success_tolerance
    return (0, value, -success) if in_band else (1, -success, value)


def choose_winner(
    summaries: dict[str, dict], objective: str, success_tolerance: float
) -> str:
    """Pick the run with the best objective among those tied on feasibility.

    A strictly lexicographic rule lets a success difference of one scenario in
    150 -- well inside sampling noise at these rates -- override an arbitrarily
    large gain on the objective being optimised. ``success_tolerance`` is the
    band, in absolute success rate, within which two runs are treated as equally
    feasible; only then does the objective decide. At 150 scenarios and a
    success rate near 0.85 one standard error is about 0.03, which is the value
    the campaign declares.

    A tolerance of 0.0 reproduces the strict lexicographic rule.
    """
    if not summaries:
        raise ValueError("cannot choose a winner from an empty set of summaries")
    key = OBJECTIVE_KEYS[objective]
    best_success = max(entry["success_rate"] for entry in summaries.values())
    eligible = [
        name
        for name, entry in summaries.items()
        if entry["success_rate"] >= best_success - success_tolerance
    ]
    return min(
        eligible,
        key=lambda name: (
            summaries[name].get(key)
            if summaries[name].get(key) is not None
            else float("inf")
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", default="results/canonical/training")
    parser.add_argument(
        "--output", default="results/canonical/selected_architecture.json"
    )
    parser.add_argument(
        "--rescore-top",
        type=int,
        default=0,
        help="Re-score this many leading runs on a larger validation sample.",
    )
    parser.add_argument("--rescore-scenarios", type=int, default=150)
    parser.add_argument(
        "--rescore-offset",
        type=int,
        default=40,
        help="Skip this many validation scenarios (those used during training).",
    )
    parser.add_argument(
        "--config", default="EVRoutingEnv/config_files/config_joint.yaml"
    )
    parser.add_argument(
        "--objective",
        default="travel_time",
        choices=sorted(OBJECTIVE_KEYS),
        help="Tie-break metric among policies with equal success (lower is better).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Processes used to re-score finalists in parallel.",
    )
    parser.add_argument(
        "--success-tolerance",
        type=float,
        default=0.0,
        help=(
            "Absolute success-rate band within which finalists count as equally "
            "feasible, leaving the objective to decide. 0.0 is strictly "
            "lexicographic."
        ),
    )
    arguments = parser.parse_args()

    root = Path(arguments.training_root)
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no validation histories under {root}")

    objective = arguments.objective
    budget = common_budget(runs)
    scored = {
        name: best_at_budget(history, budget, objective)
        for name, history in runs.items()
    }
    best_success = max(entry["success_rate"] for entry in scored.values())
    # The shortlist has to use the same tolerance band as the final choice.
    # Ranking it strictly by success first would cut a run that is tied on
    # feasibility but much better on the objective -- exactly the run the
    # campaign is looking for -- before it ever reaches the re-scoring round.
    ranking = sorted(
        scored.items(),
        key=lambda item: _banded_key(
            item[1], objective, best_success, arguments.success_tolerance
        ),
    )

    header = (
        f"{'run':<32}{'success':>9}{'completed':>11}{'travel':>10}"
        f"{'makespan':>11}{'steps':>10}"
    )
    print(f"Selection at the shared budget of {budget} interaction steps")
    print(f"objective: success first, then lower {objective}")
    print(header)
    print("-" * len(header))
    for name, entry in ranking:
        print(
            f"{name:<32}{entry['success_rate']:>9.3f}"
            f"{entry['mean_completed_fraction']:>11.3f}"
            f"{_format(entry.get('mean_travel_time_successful')):>10}"
            f"{_format(entry.get('mean_makespan_successful')):>11}"
            f"{int(entry['timesteps']):>10}"
        )

    winner_name, winner_entry = ranking[0]
    rescored: dict[str, dict] = {}
    selection_rule = (
        f"maximize validation success rate, break ties by lower {objective}"
    )
    if arguments.rescore_top > 0:
        finalists = [name for name, _ in ranking[: arguments.rescore_top]]
        print(
            f"\nRe-scoring the top {len(finalists)} runs on "
            f"{arguments.rescore_scenarios} validation scenarios..."
        )
        rescored = rescore_on_validation(
            finalists,
            root,
            arguments.config,
            arguments.rescore_scenarios,
            offset=arguments.rescore_offset,
            workers=arguments.workers,
            objective=objective,
        )
        if rescored:
            winner_name = choose_winner(
                rescored, objective, arguments.success_tolerance
            )
            winner_entry = rescored[winner_name]
            selection_rule += (
                f"; finalists re-scored on {arguments.rescore_scenarios} "
                f"validation scenarios beyond index {arguments.rescore_offset}, "
                "which are disjoint from the scenarios that chose the checkpoints"
            )
            if arguments.success_tolerance > 0.0:
                selection_rule += (
                    f"; among finalists within {arguments.success_tolerance:.3f} "
                    f"absolute success of the best, the lower {objective} wins"
                )

    encoder, head = _architecture_of(root / winner_name, winner_name)
    selection = {
        "selection_rule": selection_rule,
        "objective": objective,
        "success_tolerance": arguments.success_tolerance,
        "split_used": "validation",
        "shared_budget_timesteps": budget,
        "selected_run": winner_name,
        "state_encoder": encoder,
        "action_head": head,
        "checkpoint": str(root / winner_name),
        "validation_summary": winner_entry,
        "rescored_finalists": rescored,
        "rescore_offset": arguments.rescore_offset,
        "ranking": [
            {"run": name, "validation_summary": entry} for name, entry in ranking
        ],
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(selection, indent=2, sort_keys=True))
    print(f"\nselected {winner_name} -> {destination}")


if __name__ == "__main__":
    main()
