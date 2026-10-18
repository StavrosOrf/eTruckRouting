"""Solve a seed's clairvoyant simulator problem and verify its saved replay.

Example:
    python -m scripts.evaluation.solve_perfect_information --seed 7 \
        --customers 3 --trucks 1 --output results/perfect_information/seed7.json
    python -m scripts.evaluation.solve_perfect_information \
        --replay results/perfect_information/seed7.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

from EVRoutingEnv.baselines.optimization.perfect_information import (
    PerfectInformationParameters,
    PerfectInformationSolution,
    replay_perfect_information_plan,
    solve_perfect_information_plan,
)
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


ARTIFACT_VERSION = "evrp-perfect-information-v1"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="EVRoutingEnv/config_files/config_joint.yaml"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--customers", type=int)
    parser.add_argument("--trucks", type=int)
    parser.add_argument(
        "--objective", choices=("makespan", "total_time"), default="makespan"
    )
    parser.add_argument(
        "--information-mode",
        choices=("realized", "simulator"),
        default="realized",
        help="use known leg costs or the normal conservative feasibility rules",
    )
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument(
        "--no-time-limit",
        action="store_const",
        const=None,
        dest="time_limit",
        help="use only the deterministic node budget",
    )
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--replay", type=Path, help="verify a saved artifact using its config and seed"
    )
    args = parser.parse_args(argv)

    if args.replay:
        saved = json.loads(args.replay.read_text())
        if saved["schema_version"] != ARTIFACT_VERSION:
            parser.error("unsupported perfect-information artifact version")
        config = saved["resolved_config"]
        seed = saved["scenario"]["seed"]
    else:
        if args.output is None:
            parser.error("--output is required when solving")
        config = load_config(args.config)
        seed = args.seed
        if args.customers is not None:
            config["environment"].update(
                num_stops=args.customers, allow_variable_num_stops=False
            )
        if args.trucks is not None:
            config["environment"]["num_trucks"] = args.trucks

    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        env.reset(seed=seed)
        if args.replay:
            keys = {field.name for field in fields(PerfectInformationSolution)}
            values = {
                key: value for key, value in saved["solution"].items() if key in keys
            }
            values["actions"] = tuple(tuple(pair) for pair in values["actions"])
            solution = PerfectInformationSolution(**values)
            replay = replay_perfect_information_plan(env, solution)
            if replay["replay_sha256"] != saved["replay"]["replay_sha256"]:
                raise ValueError("saved and regenerated realized schedules differ")
            print(
                json.dumps({"verified": True, "replay_sha256": replay["replay_sha256"]})
            )
            return 0
        parameters = PerfectInformationParameters(
            objective=args.objective,
            information_mode=args.information_mode,
            time_limit_seconds=args.time_limit,
            max_nodes=args.max_nodes,
        )
        solution = solve_perfect_information_plan(env, parameters)
        replay = (
            replay_perfect_information_plan(env, solution)
            if solution.objective_hours is not None
            else None
        )
        payload = {
            "schema_version": ARTIFACT_VERSION,
            "command": sys.argv if argv is None else list(argv),
            "python": sys.version,
            "resolved_config": config,
            "scenario": env.scenario_descriptor,
            "parameters": asdict(parameters),
            "solution": solution.as_dict(),
            "replay": replay,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {**solution.as_dict(), "output": str(args.output)}, allow_nan=False
            )
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
