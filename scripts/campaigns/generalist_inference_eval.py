"""Evaluate trained generalist runs under inference-time wrappers (plan 9.6).

Rows land in the campaign's own evaluation format, so the near-optimal gate
and ``generalist_graphppo.py report`` read them as one more tag:

    PYTHONPATH=. python scripts/campaigns/generalist_inference_eval.py \
        --campaign results/generalist_graphppo/oob_v1 --split validation \
        --tag 2M_temp07 --runs full_bks_retained_s0 --wrapper temperature:t=0.7

    ... --tag 2M_la --wrapper lookahead:k=4,horizon=12 --problems ca_sameday ca_drayage_combined
    ... --tag 2M_repair --wrapper repair:time_limit=20 --problems tx_longhaul
    ... --tag 2M_ens --wrapper "ensemble:runs=full_s0|full_s1|full_s2"

Wrapped problems come from ``--problems`` (default: every suite problem; the
``repair`` wrapper defaults to the static ones). Problems outside that set are
evaluated with the plain policy, so every tag stays gate-complete. Each run's
rows go to ``evaluation/<split>/<tag>/<run>.json``; wrapper parameters and
decision-latency statistics go to ``evaluation/<split>/<tag>/wrappers/<run>.json``
(a subdirectory, which the gate's ``*.json`` glob never reads). Any gate claim
made with a wrapper must state the recorded latency.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import functools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def ensure_baselines(manifest: dict, problems, seeds, folder: Path, workers: int) -> None:
    """Teacher (and extra) baselines on exactly these seeds, as ``evaluate`` writes them."""
    from algo.generalist_ppo import run_reference_episodes
    from scripts.campaigns.generalist_graphppo import write_json

    for problem in problems:
        for policy in ["teacher", *manifest["extra_baselines"].get(problem.name, [])]:
            target = folder / "baselines" / f"{problem.name}__{policy}.json"
            if target.exists() and {row["seed"] for row in json.loads(target.read_text())} == set(seeds):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            rows = run_reference_episodes(problem, seeds, policy, workers=workers)
            for row in rows:
                row.pop("actions", None)
            partial = target.with_suffix(".partial")
            write_json(partial, rows)
            partial.replace(target)
            print(f"baseline {problem.name} {policy}: {len(rows)}", flush=True)


def _lookahead_factory(problem, scale, *, policy, reward, options):
    from algo.generalist_inference import LookaheadController

    return LookaheadController(
        policy, reward, scale,
        k=int(options.get("k", 4)),
        horizon=int(options.get("horizon", 12)),
        bootstrap=bool(options.get("bootstrap", 0)),
    )


def _repair_factory(problem, scale, *, policy, reward, options):
    from algo.generalist_inference import PlannerRepairController

    allowed = {"iterations", "time_limit", "time_limit_seconds", "seed", "restarts", "replan", "max_steps"}
    unknown = set(options) - allowed
    if unknown:
        raise SystemExit(f"repair does not take options {sorted(unknown)}")
    return PlannerRepairController(policy, **options)


def latency_summary(rows: list[dict]) -> dict:
    decisions = sum(int(row.get("decisions", 0)) for row in rows)
    if not decisions:
        return {"decisions": 0}
    mean = sum(row["decision_seconds_mean"] * row["decisions"] for row in rows if row.get("decisions")) / decisions
    return {
        "decisions": decisions,
        "mean_seconds": mean,
        "p95_seconds_max": max(row.get("decision_seconds_p95", 0.0) for row in rows),
        "episode_mean_seconds": [round(row["decision_seconds_mean"], 6) for row in rows[:8]],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--tag", required=True, help="Evaluation tag the rows are written under, e.g. 2M_temp07.")
    parser.add_argument("--runs", nargs="+", required=True, help="Manifest run names to evaluate.")
    parser.add_argument("--wrapper", required=True, help="name[:key=value,...]; see the module docstring.")
    parser.add_argument("--problems", nargs="*", default=None, help="Problems the wrapper applies to (default: all).")
    parser.add_argument("--count", type=int, default=None, help="Seeds per problem (default: manifest).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--envs-per-problem", type=int, default=16)
    parser.add_argument("--prefix", default="best", choices=["best", "final"], help="Checkpoint prefix to load.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing rows under the tag.")
    parser.add_argument("--allow-test", action="store_true", help="P5 only; the test bank is otherwise locked.")
    arguments = parser.parse_args(argv)
    if arguments.split == "test" and not arguments.allow_test:
        raise SystemExit("the test bank is locked; evaluate on validation unless the coordinator opens P5")

    import torch

    from algo.generalist_inference import (
        SEQUENTIAL,
        EnsemblePolicy,
        TemperaturePolicy,
        evaluate_wrapped_problem,
        parse_wrapper,
    )
    from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig
    from algo.generalist_ppo import (
        ReferenceTable,
        SuiteProblem,
        SuiteReward,
        evaluate_policy_suite,
        summarize_suite,
    )
    from scripts.campaigns.generalist_graphppo import read_manifest, write_json
    from scripts.evaluation.canonical_harness import split_seeds

    spec = parse_wrapper(arguments.wrapper)
    destination = arguments.campaign.resolve()
    manifest = read_manifest(destination)
    problems = [SuiteProblem(**row) for row in json.loads(Path(manifest["problems"]).read_text())]
    names = [problem.name for problem in problems]
    if arguments.problems is None:
        wrapped = list(names)
        if spec.name == "repair":
            from scripts.campaigns.generalist_graphppo import DYNAMIC

            wrapped = [p for p in names if p not in DYNAMIC]
    else:
        unknown = set(arguments.problems) - set(names)
        if unknown:
            raise SystemExit(f"unknown problems {sorted(unknown)}; the suite is {names}")
        wrapped = [p for p in names if p in set(arguments.problems)]
    if spec.name == "repair":
        from scripts.campaigns.generalist_graphppo import DYNAMIC

        offending = [p for p in wrapped if p in DYNAMIC]
        if offending:
            raise SystemExit(f"planner repair is defined for static problems only, not {offending}")

    count = arguments.count or manifest["seeds"][arguments.split]
    seeds = split_seeds(arguments.split, count)
    folder = destination / "evaluation" / arguments.split
    (folder / arguments.tag).mkdir(parents=True, exist_ok=True)
    (folder / arguments.tag / "wrappers").mkdir(exist_ok=True)
    ensure_baselines(manifest, problems, seeds, folder, arguments.workers)
    references = ReferenceTable({
        problem.name: {
            row["seed"]: row
            for row in json.loads((folder / "baselines" / f"{problem.name}__teacher.json").read_text())
        }
        for problem in problems
    })
    reward = SuiteReward(reference="scenario")
    by_name = {problem.name: problem for problem in problems}

    for run_name in arguments.runs:
        if not any(run["name"] == run_name for run in manifest["runs"]):
            raise SystemExit(f"{run_name} is not a run of this campaign")
        checkpoint = destination / "training" / run_name
        if not (checkpoint / f"{arguments.prefix}.pt").exists():
            raise SystemExit(f"{checkpoint / (arguments.prefix + '.pt')} does not exist")
        target = folder / arguments.tag / f"{run_name}.json"
        if target.exists() and not arguments.overwrite:
            print(f"{run_name}: rows exist under tag {arguments.tag}; --overwrite replaces them", flush=True)
            continue
        policy = GeneralistActorCritic.load(checkpoint, prefix=arguments.prefix, map_location=arguments.device)
        policy = policy.to(arguments.device).eval()

        rows: list[dict] = []
        sidecar = {
            "wrapper": arguments.wrapper,
            "name": spec.name,
            "options": {k: v for k, v in spec.options.items()},
            "prefix": arguments.prefix,
            "count": count,
            "created": _datetime.datetime.now().isoformat(timespec="seconds"),
            "problems": {},
        }
        plain = [by_name[name] for name in names if name not in wrapped]
        if spec.name in SEQUENTIAL:
            for name in wrapped:
                factory = functools.partial(
                    _lookahead_factory if spec.name == "lookahead" else _repair_factory,
                    policy=policy, reward=reward, options=spec.options,
                )
                with torch.no_grad():
                    part = evaluate_wrapped_problem(
                        by_name[name], seeds, policy.config.observation_spec, reward, references,
                        factory, workers=arguments.workers,
                    )
                rows.extend(part)
                sidecar["problems"][name] = {"wrapper": spec.name, **latency_summary(part)}
                print(f"{run_name} {name} ({spec.name}): {len(part)} episodes, "
                      f"mean decision {sidecar['problems'][name].get('mean_seconds', 0.0):.3f}s", flush=True)
        else:
            if spec.name == "temperature":
                t_end = spec.options.get("t_end")
                actor = TemperaturePolicy(
                    policy, float(spec.options.get("temperature", 1.0)),
                    t_end=None if t_end is None else float(t_end),
                    top_p=float(spec.options.get("top_p", 1.0)),
                )
            else:
                members = []
                for member in spec.options.get("runs", []) or [run_name]:
                    directory = destination / "training" / member
                    members.append(
                        GeneralistActorCritic.load(directory, prefix=arguments.prefix, map_location=arguments.device)
                        .to(arguments.device).eval()
                    )
                actor = EnsemblePolicy(members, temperature=float(spec.options.get("temperature", 0.0)))
            with torch.no_grad():
                part = evaluate_policy_suite(
                    actor, [by_name[name] for name in wrapped], [seeds] * len(wrapped), references, reward,
                    device=arguments.device, workers=arguments.workers,
                    envs_per_problem=arguments.envs_per_problem,
                )
            rows.extend(part)
            for name in wrapped:
                sidecar["problems"][name] = {"wrapper": spec.name, **actor.latency()}
            print(f"{run_name} ({spec.name}): {len(part)} episodes over {wrapped}", flush=True)
        if plain:
            with torch.no_grad():
                part = evaluate_policy_suite(
                    policy, plain, [seeds] * len(plain), references, reward,
                    device=arguments.device, workers=arguments.workers,
                    envs_per_problem=arguments.envs_per_problem,
                )
            rows.extend(part)
            for problem in plain:
                sidecar["problems"][problem.name] = {"wrapper": "base"}
            print(f"{run_name} (base): {len(part)} episodes over {[p.name for p in plain]}", flush=True)

        write_json(target, rows)
        write_json(folder / arguments.tag / "wrappers" / f"{run_name}.json", sidecar)
        summary = summarize_suite(rows, names)
        for name in names:
            entry = summary.get(name)
            if entry:
                print(f"  {name}: score {entry['score']:.3f} served {entry['served_fraction']:.3f} "
                      f"success {entry['success_rate']:.3f} failures {entry['failure_rate']:.3f} "
                      f"hours {entry['hours']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
