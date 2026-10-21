"""Record the legacy golden traces that freeze simulator behaviour (Phase 0).

Run from the repository root, before any Phase 1 change:
    PYTHONPATH=. python scripts/analysis/record_golden_traces.py

Each episode stores its action sequence and a SHA-256 per recorded stream:
the active truck, clock, hard mask (as a bit string) and termination reason at
every step, plus every truck's final ``export_event_log()``. The event history
is append-only and timestamped, so the final log covers every step without a
quadratic per-step dump. The first seed of every (config, policy) pair is also
stored in full for diffing. ``tests/integration/test_golden_traces.py`` replays
the actions and compares the hashes; observations are deliberately not hashed
because the feature schema is expected to change.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
from enum import Enum
from pathlib import Path

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.baselines.heuristics.canonical_baselines import (
    GreedyHeuristicPolicy,
    RandomFeasiblePolicy,
)
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.features import SCHEMA_VERSION
from EVRoutingEnv.utils.utils import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "EVRoutingEnv" / "config_files"
GOLDEN_DIR = ROOT / "tests" / "data" / "golden"
INDEX = GOLDEN_DIR / "golden_traces.json"
FEATURE_INDEX = GOLDEN_DIR / "golden_features_v6.json"

CONFIGS = (
    "config_joint.yaml",
    "config_joint_texas_longhaul.yaml",
    "config_joint_time_windows.yaml",
    "config_joint_preassigned.yaml",
    "config_sequential.yaml",
)
POLICIES = ("greedy", "random")
SEEDS = (0, 1, 2, 3, 4)
STREAMS = ("actions", "active_truck", "clock", "mask", "termination_reason", "event_logs")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"cannot serialize {type(value).__name__} in a golden trace")


def canonical_json(value) -> str:
    # repr-exact floats and sorted keys make the hash a bit-identity check.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_jsonable)


def stream_hashes(trace: dict) -> dict[str, str]:
    return {
        stream: hashlib.sha256(canonical_json(trace[stream]).encode()).hexdigest()
        for stream in STREAMS
    }


def make_env(config_name: str) -> EventDrivenTruckEnv:
    config = load_config(str(CONFIG_DIR / config_name))
    # A cached instance equals the generated one, but keep the recording
    # independent of whatever cache a shell happens to point at.
    config["environment"].pop("scenario_cache_dir", None)
    return EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)


def make_policy(name: str, seed: int):
    if name == "greedy":
        return GreedyHeuristicPolicy()
    if name == "random":
        return RandomFeasiblePolicy(seed=seed)
    raise ValueError(f"unknown golden-trace policy {name!r}")


def run_episode(env: EventDrivenTruckEnv, seed: int, policy=None, actions=None) -> dict:
    """Roll one episode, choosing actions from ``policy`` or replaying ``actions``."""
    if (policy is None) == (actions is None):
        raise ValueError("give exactly one of policy and actions")
    observation, info = env.reset(seed=seed)
    trace = {stream: [] for stream in STREAMS}
    terminated = truncated = False
    while not (terminated or truncated):
        step = len(trace["actions"])
        if actions is not None:
            if step >= len(actions):
                raise AssertionError(f"episode outlived the {len(actions)} recorded actions")
            action = int(actions[step])
        else:
            action = int(policy(env, observation, info))
        trace["active_truck"].append(env.active_truck_id)
        trace["mask"].append("".join("1" if bit else "0" for bit in env.mask_fn()))
        observation, _, terminated, truncated, info = env.step(action)
        trace["actions"].append(action)
        trace["clock"].append(env.global_clock)
        trace["termination_reason"].append(info["termination_reason"])
    if actions is not None and len(trace["actions"]) != len(actions):
        raise AssertionError(
            f"episode ended after {len(trace['actions'])} of {len(actions)} recorded actions"
        )
    trace["event_logs"] = [truck.export_event_log() for truck in env.trucks]
    return trace


FEATURE_BLOCKS = {
    "truck": ("truck_features", "truck_feature_names"),
    "customer": ("customer_features", "customer_feature_names"),
    "charger": ("charger_features", "charger_feature_names"),
    "action": ("action_features", "action_feature_names"),
    "global": ("global_features", "global_feature_names"),
}


def feature_hashes(features, columns: dict[str, list[str]]) -> dict[str, str]:
    """Hash the named columns of every block, so a wider schema can be compared."""
    hashes = {}
    for block, (array_name, names_name) in FEATURE_BLOCKS.items():
        values = np.asarray(getattr(features, array_name))
        names = list(getattr(features, names_name))
        index = [names.index(name) for name in columns[block]]
        selected = np.ascontiguousarray(values[..., index], dtype=np.float32)
        digest = hashlib.sha256(repr(selected.shape).encode())
        digest.update(selected.tobytes())
        hashes[block] = digest.hexdigest()
    digest = hashlib.sha256()
    for relation in sorted(features.pairwise_features):
        array = np.ascontiguousarray(features.pairwise_features[relation], dtype=np.float32)
        digest.update(repr((relation, array.shape)).encode())
        digest.update(array.tobytes())
    hashes["pairwise"] = digest.hexdigest()
    return hashes


def replay_feature_hashes(env, episode: dict, columns: dict[str, list[str]]) -> list[dict]:
    """Per-decision feature hashes along a recorded golden trajectory."""
    env.reset(seed=episode["seed"])
    rows = []
    for action in episode["actions"]:
        rows.append(feature_hashes(env.get_canonical_features(), columns))
        env.step(int(action))
    return rows


def record_features() -> dict:
    """Freeze the v6 feature columns on the first seed of every golden pair."""
    golden = json.loads(INDEX.read_text())
    episodes = []
    columns = None
    for config_name in CONFIGS:
        env = make_env(config_name)
        try:
            if columns is None:
                env.reset(seed=0)
                features = env.get_canonical_features()
                columns = {
                    block: list(getattr(features, names_name))
                    for block, (_, names_name) in FEATURE_BLOCKS.items()
                }
            for episode in golden["episodes"]:
                if episode["config"] != config_name or episode["seed"] != SEEDS[0]:
                    continue
                episodes.append({
                    "config": config_name,
                    "policy": episode["policy"],
                    "seed": episode["seed"],
                    "steps": replay_feature_hashes(env, episode, columns),
                })
        finally:
            env.close()
    index = {
        "format": 1,
        "schema_version": SCHEMA_VERSION,
        "source": _source_revision(),
        "columns": columns,
        "episodes": episodes,
    }
    FEATURE_INDEX.write_text(json.dumps(index, indent=1) + "\n")
    return index


def _source_revision() -> dict:
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def record(configs=CONFIGS, policies=POLICIES, seeds=SEEDS) -> dict:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    (GOLDEN_DIR / "full").mkdir(exist_ok=True)
    episodes = []
    for config_name in configs:
        env = make_env(config_name)
        try:
            for policy_name in policies:
                for seed in seeds:
                    trace = run_episode(env, seed, policy=make_policy(policy_name, seed))
                    hashes = stream_hashes(trace)
                    # A policy that consumed simulator randomness would make
                    # the recorded actions unreplayable; refuse to store it.
                    replay = run_episode(env, seed, actions=trace["actions"])
                    if stream_hashes(replay) != hashes:
                        raise RuntimeError(
                            f"{config_name}/{policy_name}/seed {seed}: replay differs from rollout"
                        )
                    full = None
                    if seed == seeds[0]:
                        full = f"full/{Path(config_name).stem}__{policy_name}__seed{seed}.json.gz"
                        with gzip.open(GOLDEN_DIR / full, "wt", encoding="utf-8") as handle:
                            handle.write(canonical_json(trace))
                    episodes.append({
                        "config": config_name,
                        "policy": policy_name,
                        "seed": seed,
                        "steps": len(trace["actions"]),
                        "termination_reason": trace["termination_reason"][-1],
                        "actions": trace["actions"],
                        "sha256": hashes,
                        "full_trace": full,
                    })
                    print(
                        f"{config_name:40s} {policy_name:7s} seed {seed}: "
                        f"{len(trace['actions']):5d} steps, {trace['termination_reason'][-1]}"
                    )
        finally:
            env.close()
    index = {"format": 1, "source": _source_revision(), "streams": list(STREAMS), "episodes": episodes}
    INDEX.write_text(json.dumps(index, indent=1) + "\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--features-only", action="store_true",
        help="only (re)record the frozen feature columns along the stored trajectories",
    )
    args = parser.parse_args()
    if not args.features_only:
        record(tuple(args.configs), POLICIES, tuple(args.seeds))
    record_features()


if __name__ == "__main__":
    main()
