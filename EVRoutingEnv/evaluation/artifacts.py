"""Versioned scenario descriptors and immutable campaign manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


SCENARIO_DESCRIPTOR_VERSION = "evrp-scenario-v1"
RUN_MANIFEST_VERSION = "evrp-run-manifest-v1"
SEED_NAMESPACE_SIZE = 1_000_000_000
SEED_SPLIT_OFFSETS = {
    "train": 0,
    "validation": SEED_NAMESPACE_SIZE,
    "test": 2 * SEED_NAMESPACE_SIZE,
}


@dataclass(frozen=True)
class CampaignSeedPlan:
    """Provably disjoint train/validation/test integer seed namespaces."""

    base_seed: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.base_seed, int)
            or isinstance(self.base_seed, bool)
            or not 0 <= self.base_seed < SEED_NAMESPACE_SIZE
        ):
            raise ValueError(
                f"base_seed must be an integer in [0, {SEED_NAMESPACE_SIZE})"
            )

    def seed(self, split: str, index: int) -> int:
        if split not in SEED_SPLIT_OFFSETS:
            raise ValueError(f"unknown seed split {split!r}")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < SEED_NAMESPACE_SIZE
        ):
            raise ValueError(
                f"seed index must be an integer in [0, {SEED_NAMESPACE_SIZE})"
            )
        return self.base_seed + SEED_SPLIT_OFFSETS[split] + index


@dataclass(frozen=True)
class RunManifest:
    """Minimum provenance required before a campaign run may start."""

    schema_version: str
    run_id: str
    created_at_utc: str
    algorithm: str
    split: str
    command: tuple[str, ...]
    resolved_config: dict
    config_sha256: str
    scenario_seeds: tuple[int, ...]
    checkpoint: str | None
    git_commit: str | None
    git_dirty: bool | None
    python: str
    platform: str
    cpu_count: int | None
    dependencies: dict[str, str]

    def as_dict(self) -> dict:
        return asdict(self)


def build_scenario_descriptor(env) -> dict:
    """Capture the immutable inputs defining one reset scenario."""
    if env.scenario_random_streams is None or env.scenario_seed is None:
        raise RuntimeError("scenario descriptor requires a completed reset")
    rng_metadata = env.scenario_random_streams.metadata()
    customers = []
    if env.joint_instance is not None:
        customers = [
            {
                "task_id": task.task_id,
                "node_id": task.node_id,
                "demand": task.demand,
                "base_service_time": task.base_service_time,
                "earliest_service": task.earliest_service,
                "latest_service": (
                    task.latest_service
                    if math.isfinite(task.latest_service)
                    else None
                ),
            }
            for task in env.joint_instance.tasks
        ]
    descriptor = {
        "schema_version": SCENARIO_DESCRIPTOR_VERSION,
        "seed": int(env.scenario_seed),
        "rng_version": str(rng_metadata["version"]),
        "problem_mode": env.problem_mode,
        "config_sha256": canonical_json_sha256(env.config),
        "num_trucks": env.num_trucks,
        "num_customers": env.num_stops,
        "depot_node": (
            env.joint_instance.depot_node
            if env.joint_instance is not None
            else None
        ),
        "customers": customers,
        "initial_battery_kwh": [
            truck.current_battery for truck in env.trucks
        ],
        "truck_routes": [
            list(truck.delivery_sequence) for truck in env.trucks
        ],
        "chargers": [
            {
                "node_id": int(node),
                "type": env.charging_station.charger_type[node],
                "power_kw": env.charging_station.charger_power_kw[node],
                "port_capacity": int(
                    env.charging_station.charger_capacity[node]
                ),
            }
            for node in sorted(env.charging_nodes)
        ],
        "traffic": dict(env.traffic_config),
        "delivery": dict(env.delivery_config),
        "charging_action_mode": env.charging_action_mode,
        "charge_action_values": list(env.charge_action_values),
    }
    _canonical_json(descriptor)
    return descriptor


def collect_run_manifest(
    *,
    run_id: str,
    algorithm: str,
    split: str,
    command: list[str] | tuple[str, ...],
    resolved_config: dict,
    scenario_seeds: list[int] | tuple[int, ...],
    repository_root: str | Path,
    checkpoint: str | None = None,
) -> RunManifest:
    """Collect local provenance without mutating the repository or environment."""
    if not run_id.strip() or not algorithm.strip():
        raise ValueError("run_id and algorithm must be non-empty")
    if split not in SEED_SPLIT_OFFSETS:
        raise ValueError(f"unknown campaign split {split!r}")
    command_values = tuple(str(value) for value in command)
    if not command_values:
        raise ValueError("command cannot be empty")
    seeds = tuple(int(seed) for seed in scenario_seeds)
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("scenario seeds must be unique and non-negative")
    config_copy = json.loads(_canonical_json(resolved_config))

    root = Path(repository_root).resolve()
    commit = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain")
    dependencies = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    return RunManifest(
        schema_version=RUN_MANIFEST_VERSION,
        run_id=run_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        algorithm=algorithm,
        split=split,
        command=command_values,
        resolved_config=config_copy,
        config_sha256=canonical_json_sha256(config_copy),
        scenario_seeds=seeds,
        checkpoint=checkpoint,
        git_commit=commit,
        git_dirty=(bool(status) if status is not None else None),
        python=sys.version,
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        dependencies=dict(sorted(dependencies.items())),
    )


def write_immutable_manifest(manifest: RunManifest, path: str | Path) -> Path:
    """Create a strict-JSON manifest and refuse to overwrite existing evidence."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(manifest.as_dict()) + "\n"
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def canonical_json_sha256(value) -> str:
    """Hash the same strict canonical JSON representation used by artifacts."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _git_output(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
