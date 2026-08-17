"""Failure-retaining evaluation runner with immutable campaign artifacts."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    RunManifest,
    build_scenario_descriptor,
    canonical_json_sha256,
    write_immutable_manifest,
)
from .metrics import extract_operational_metrics
from .statistics import aggregate_episode_metrics


EPISODE_ROW_VERSION = "evrp-episode-row-v1"
CAMPAIGN_SUMMARY_VERSION = "evrp-campaign-summary-v1"
CAMPAIGN_FAILURE_VERSION = "evrp-campaign-failure-v1"


PolicyCallback = Callable[[Any, Any, dict], Any]
EnvironmentFactory = Callable[[], Any]


@dataclass(frozen=True)
class CampaignArtifacts:
    """Paths published by one successfully completed evaluation campaign."""

    output_directory: Path
    manifest_path: Path
    episode_rows_path: Path
    summary_path: Path
    episode_count: int


def run_evaluation_campaign(
    *,
    environment_factory: EnvironmentFactory,
    policy: PolicyCallback,
    manifest: RunManifest,
    output_directory: str | Path,
    max_policy_steps: int | None = None,
) -> CampaignArtifacts:
    """Evaluate every manifest seed and publish strict, immutable artifacts.

    The manifest is written before the first reset. Raw rows are accumulated in
    an ``.inprogress`` file and atomically renamed only after every scenario
    finishes. If policy or environment execution raises, the exception is
    propagated and the incomplete file remains as explicit evidence.
    """
    if not callable(environment_factory):
        raise TypeError("environment_factory must be callable")
    if not callable(policy):
        raise TypeError("policy must be callable")
    if not manifest.scenario_seeds:
        raise ValueError("manifest must contain at least one scenario seed")
    if max_policy_steps is not None and (
        isinstance(max_policy_steps, bool)
        or not isinstance(max_policy_steps, int)
        or max_policy_steps <= 0
    ):
        raise ValueError("max_policy_steps must be a positive integer or None")

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    manifest_path = write_immutable_manifest(manifest, destination / "manifest.json")
    inprogress_path = destination / "episode_rows.jsonl.inprogress"
    rows_path = destination / "episode_rows.jsonl"
    summary_path = destination / "summary.json"
    rows: list[dict] = []

    try:
        with inprogress_path.open("x", encoding="utf-8") as row_file:
            for episode_index, scenario_seed in enumerate(manifest.scenario_seeds):
                row = _run_episode(
                    environment_factory=environment_factory,
                    policy=policy,
                    manifest=manifest,
                    episode_index=episode_index,
                    scenario_seed=scenario_seed,
                    max_policy_steps=max_policy_steps,
                )
                row_file.write(_strict_json(row) + "\n")
                row_file.flush()
                os.fsync(row_file.fileno())
                rows.append(row)
    except Exception as error:
        failure = {
            "schema_version": CAMPAIGN_FAILURE_VERSION,
            "run_id": manifest.run_id,
            "completed_episode_count": len(rows),
            "next_episode_index": len(rows),
            "next_scenario_seed": manifest.scenario_seeds[len(rows)],
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        try:
            _write_json_exclusive(destination / "failure.json", failure)
            _fsync_directory(destination)
        except OSError:
            pass
        raise

    summary = {
        "schema_version": CAMPAIGN_SUMMARY_VERSION,
        "run_id": manifest.run_id,
        "algorithm": manifest.algorithm,
        "split": manifest.split,
        "manifest_sha256": canonical_json_sha256(manifest.as_dict()),
        "scenario_seeds": list(manifest.scenario_seeds),
        "aggregate": aggregate_episode_metrics(rows),
        "inference": _inference_summary(rows),
    }
    if rows_path.exists():
        raise FileExistsError(f"refusing to overwrite {rows_path}")
    os.replace(inprogress_path, rows_path)
    _fsync_directory(destination)
    _write_json_exclusive(summary_path, summary)
    _fsync_directory(destination)
    return CampaignArtifacts(
        output_directory=destination,
        manifest_path=manifest_path,
        episode_rows_path=rows_path,
        summary_path=summary_path,
        episode_count=len(rows),
    )


def _run_episode(
    *,
    environment_factory: EnvironmentFactory,
    policy: PolicyCallback,
    manifest: RunManifest,
    episode_index: int,
    scenario_seed: int,
    max_policy_steps: int | None,
) -> dict:
    environment = environment_factory()
    if environment is None:
        raise TypeError("environment_factory returned None")
    started = time.perf_counter()
    inference_seconds = 0.0
    policy_calls = 0
    episode_reward = 0.0
    try:
        observation, info = environment.reset(seed=scenario_seed)
        scenario = info.get("scenario") or build_scenario_descriptor(environment)
        if int(scenario.get("seed", -1)) != scenario_seed:
            raise RuntimeError("environment scenario seed does not match the manifest")
        if scenario.get("config_sha256") != manifest.config_sha256:
            raise RuntimeError("environment configuration does not match the manifest")
        scenario_hash = canonical_json_sha256(scenario)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            if max_policy_steps is not None and policy_calls >= max_policy_steps:
                raise RuntimeError(
                    f"policy exceeded max_policy_steps={max_policy_steps} on "
                    f"scenario seed {scenario_seed}"
                )
            inference_started = time.perf_counter()
            action = policy(environment, observation, info)
            inference_seconds += time.perf_counter() - inference_started
            policy_calls += 1
            observation, reward, terminated, truncated, info = environment.step(action)
            reward_value = float(reward)
            if not math.isfinite(reward_value):
                raise ValueError("environment returned a non-finite reward")
            episode_reward += reward_value

        metrics = extract_operational_metrics(environment).as_dict()
        row = {
            "schema_version": EPISODE_ROW_VERSION,
            "run_id": manifest.run_id,
            "algorithm": manifest.algorithm,
            "split": manifest.split,
            "episode_index": episode_index,
            "scenario_seed": scenario_seed,
            "scenario_sha256": scenario_hash,
            "scenario": scenario,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "policy_calls": policy_calls,
            "inference_seconds": inference_seconds,
            "wall_seconds": time.perf_counter() - started,
            "episode_reward_diagnostic": episode_reward,
            **metrics,
        }
        # A planner may expose per-episode evidence about its own solve --
        # solver status, bound, gap, retries, fallbacks. R1.4 requires those to
        # be published rather than summarised in prose, so they ride along with
        # the episode they belong to.
        diagnostics = getattr(policy, "diagnostics", None)
        if callable(diagnostics):
            row["policy_diagnostics"] = diagnostics()
        _strict_json(row)
        return row
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()


def _inference_summary(rows: Sequence[dict]) -> dict:
    total_calls = sum(int(row["policy_calls"]) for row in rows)
    total_seconds = sum(float(row["inference_seconds"]) for row in rows)
    per_decision = [
        float(row["inference_seconds"]) / int(row["policy_calls"])
        for row in rows
        if int(row["policy_calls"]) > 0
    ]
    return {
        "total_policy_calls": total_calls,
        "total_inference_seconds": total_seconds,
        "mean_seconds_per_call": (total_seconds / total_calls if total_calls else None),
        "mean_episode_seconds_per_call": (
            sum(per_decision) / len(per_decision) if per_decision else None
        ),
    }


def _write_json_exclusive(path: Path, value: dict) -> None:
    payload = _strict_json(value) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _strict_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
