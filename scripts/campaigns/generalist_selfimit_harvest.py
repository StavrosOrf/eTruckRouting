"""Self-imitation harvest (design 9.3): the policy's own tempered wins as demos.

On TRAIN seeds only, run the champion policy once greedily and ``--samples``
times with the scheduled tempered decode; whenever the best sampled episode
strictly beats both the greedy episode and the seed's current demonstration
(by the BKS outcome order), replay-verify it and write it into a new
demonstrations directory (other problems and losing seeds keep their current
rows, hard-linked/copied). The output directory is a drop-in
``--demonstrations`` argument for a fine-tune run.

    PYTHONPATH=. python scripts/campaigns/generalist_selfimit_harvest.py \
        --policy-dir .../oob_v1/training/oob_privcritic_s0 \
        --bks-dir .../bks --out .../selfimit_demos \
        --problem ca_sameday --count 512 --samples 8 --workers 6 \
        --wrapper t=1.3,t_end=0.2,top_p=0.9
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBLEMS = ("ca_sameday", "ca_drayage_combined", "tx_longhaul", "tx_longhaul_congested", "pnw_longhaul_hos")
_WORKER: dict = {}


def _summary_row(problem, seed, policy_name, actions, summary):
    keep = ("success", "failed", "termination_reason", "hours", "travel_hours", "charging_sessions",
            "completed_fraction", "steps", "dynamic", "tasks", "served", "late", "lost", "unserved")
    return {"problem": problem, "seed": int(seed), "policy": policy_name, "actions": actions,
            **{k: summary[k] for k in keep if k in summary}}


def _episode(env, actor, collate, seed):
    import torch
    observation, _ = env.reset(seed=int(seed))
    actions = []
    from algo.generalist_ppo import episode_summary
    for _ in range(20000):
        with torch.no_grad():
            position, _, _ = actor.act(collate([observation]))
        action = int(observation.action_index[int(position[0])])
        actions.append(action)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return actions, episode_summary(env)


def _init(policy_dir, problem, options):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    torch.set_num_threads(1)
    from algo.generalist_policy import GeneralistActorCritic
    from algo.generalist_observation import collate
    from algo.generalist_inference import TemperaturePolicy
    from algo.generalist_ppo import GeneralistEnv
    from scripts.campaigns.generalist_bks_common import config_path
    policy = GeneralistActorCritic.load(policy_dir, prefix="best")
    policy.eval()
    _WORKER["env"] = GeneralistEnv(config_path(problem))
    _WORKER["collate"] = collate
    _WORKER["greedy"] = TemperaturePolicy(policy, 0.0)
    _WORKER["tempered"] = TemperaturePolicy(
        policy, float(options.get("t", 1.3)),
        t_end=float(options["t_end"]) if "t_end" in options else None,
        top_p=float(options.get("top_p", 1.0)),
    )


def _harvest_seed(job):
    problem, seed, samples, incumbent_key = job
    import torch
    from scripts.campaigns.generalist_bks_common import outcome_key
    env, collate = _WORKER["env"], _WORKER["collate"]
    torch.manual_seed(seed * 9973 + 17)
    candidates = []
    greedy_actions, greedy_summary = _episode(env, _WORKER["greedy"], collate, seed)
    for _ in range(samples):
        actions, summary = _episode(env, _WORKER["tempered"], collate, seed)
        if not summary.get("failed"):
            candidates.append((actions, summary))
    if not candidates:
        return seed, None
    best_actions, best = min(candidates, key=lambda item: outcome_key(item[1]))
    bar = [outcome_key(greedy_summary)] if not greedy_summary.get("failed") else []
    if incumbent_key is not None:
        bar.append(tuple(incumbent_key))
    if bar and not (outcome_key(best) < min(bar)):
        return seed, None
    # Replay-verify through a fresh reset before accepting the row.
    env.reset(seed=int(seed))
    from algo.generalist_ppo import episode_summary
    for action in best_actions:
        _, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break
    replay = episode_summary(env)
    same = (replay["success"] == best["success"] and replay["served"] == best["served"]
            and replay["failed"] == best["failed"] and abs(replay["hours"] - best["hours"]) < 1e-6
            and replay["steps"] == len(best_actions))
    if not same:
        return seed, "mismatch"
    return seed, _summary_row(problem, seed, "self_tempered", best_actions, best)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--bks-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--problem", default="ca_sameday", choices=PROBLEMS)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--wrapper", default="t=1.3,t_end=0.2,top_p=0.9")
    arguments = parser.parse_args()
    options = dict(part.split("=", 1) for part in arguments.wrapper.split(","))

    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.campaigns.generalist_bks_common import seeds_for, outcome_key
    seeds = seeds_for("train", arguments.count, arguments.offset)
    current = {row["seed"]: row for row in json.loads(
        (arguments.bks_dir / arguments.problem / "train_demonstrations.json").read_text())}
    jobs = [(arguments.problem, seed, arguments.samples,
             outcome_key(current[seed]) if seed in current else None) for seed in seeds]

    started = time.perf_counter()
    with mp.get_context("fork").Pool(
        arguments.workers, initializer=_init,
        initargs=(arguments.policy_dir, arguments.problem, options),
    ) as pool:
        results = dict(pool.imap_unordered(_harvest_seed, jobs, chunksize=4))
    wins = {seed: row for seed, row in results.items() if isinstance(row, dict)}
    mismatches = [seed for seed, row in results.items() if row == "mismatch"]

    merged = []
    for seed in sorted(set(current) | set(wins)):
        row = dict(wins.get(seed) or current[seed])
        row["replay_verified"] = True
        merged.append(row)
    (arguments.out / arguments.problem).mkdir(parents=True, exist_ok=True)
    (arguments.out / arguments.problem / "train_demonstrations.json").write_text(json.dumps(merged))
    for problem in PROBLEMS:
        if problem == arguments.problem:
            continue
        (arguments.out / problem).mkdir(parents=True, exist_ok=True)
        shutil.copy2(arguments.bks_dir / problem / "train_demonstrations.json",
                     arguments.out / problem / "train_demonstrations.json")
    summary = {
        "policy_dir": arguments.policy_dir, "wrapper": arguments.wrapper, "samples": arguments.samples,
        "seeds": len(seeds), "wins": len(wins), "replay_mismatches": mismatches,
        "mean_served_wins": (sum(r["served"] for r in wins.values()) / len(wins)) if wins else None,
        "mean_served_incumbent_on_win_seeds": (
            sum(current[s]["served"] for s in wins if s in current) / len(wins)) if wins else None,
        "seconds": time.perf_counter() - started,
    }
    (arguments.out / f"{arguments.problem}_harvest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
