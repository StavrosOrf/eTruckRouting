import copy
import json
from pathlib import Path

import pytest

from EVRoutingEnv.evaluation.artifacts import (
    CampaignSeedPlan,
    collect_run_manifest,
)
from EVRoutingEnv.evaluation.runner import run_evaluation_campaign
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


def _config() -> dict:
    config = copy.deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["environment"]["num_trucks"] = 1
    config["environment"]["num_stops"] = 2
    config["environment"]["max_episode_steps"] = 30
    config["problem"]["payload_capacity"] = 20.0
    config["truck"]["initial_battery"] = "full"
    config["traffic"]["enable_traffic"] = False
    return config


def _factory(config):
    return lambda: EventDrivenTruckEnv(
        config=copy.deepcopy(config),
        verbose=False,
        enable_plotting=False,
        run_id="campaign_runner_test",
    )


def _customer_first_policy(environment, _observation, _info):
    truck = environment.trucks[environment.active_truck_id]
    available = environment.task_registry.available_tasks(truck.remaining_payload)
    target = (
        available[0].node_id if available else environment.joint_instance.depot_node
    )
    return target, 0.0, False


def _manifest(config, seeds):
    return collect_run_manifest(
        run_id="runner-integration",
        algorithm="customer-first",
        split="validation",
        command=("pytest", "test_campaign_runner"),
        resolved_config=config,
        scenario_seeds=seeds,
        repository_root=Path.cwd(),
    )


@pytest.mark.integration
def test_campaign_runner_publishes_manifest_raw_rows_and_summary(tmp_path):
    config = _config()
    plan = CampaignSeedPlan(base_seed=73)
    seeds = [plan.seed("validation", index) for index in range(2)]
    destination = tmp_path / "completed-run"

    artifacts = run_evaluation_campaign(
        environment_factory=_factory(config),
        policy=_customer_first_policy,
        manifest=_manifest(config, seeds),
        output_directory=destination,
        max_policy_steps=30,
    )

    assert artifacts.episode_count == 2
    assert artifacts.manifest_path.is_file()
    assert artifacts.episode_rows_path.is_file()
    assert artifacts.summary_path.is_file()
    assert not (destination / "episode_rows.jsonl.inprogress").exists()
    rows = [
        json.loads(line)
        for line in artifacts.episode_rows_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["scenario_seed"] for row in rows] == seeds
    assert all(row["scenario"]["seed"] == row["scenario_seed"] for row in rows)
    assert all(row["scenario_sha256"] for row in rows)
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["aggregate"]["episode_count"] == 2
    assert summary["aggregate"]["success_count"] == sum(
        bool(row["success"]) for row in rows
    )
    assert (
        summary["aggregate"]["success_count"] + summary["aggregate"]["failure_count"]
        == 2
    )
    assert summary["inference"]["total_policy_calls"] > 0

    with pytest.raises(FileExistsError):
        run_evaluation_campaign(
            environment_factory=_factory(config),
            policy=_customer_first_policy,
            manifest=_manifest(config, seeds),
            output_directory=destination,
        )


@pytest.mark.integration
def test_campaign_runner_leaves_explicit_incomplete_evidence_on_error(tmp_path):
    config = _config()
    seeds = [CampaignSeedPlan(base_seed=91).seed("validation", 0)]
    destination = tmp_path / "failed-run"

    def failing_policy(_environment, _observation, _info):
        raise RuntimeError("intentional policy failure")

    with pytest.raises(RuntimeError, match="intentional policy failure"):
        run_evaluation_campaign(
            environment_factory=_factory(config),
            policy=failing_policy,
            manifest=_manifest(config, seeds),
            output_directory=destination,
        )

    assert (destination / "manifest.json").is_file()
    assert (destination / "episode_rows.jsonl.inprogress").is_file()
    failure = json.loads((destination / "failure.json").read_text(encoding="utf-8"))
    assert failure["completed_episode_count"] == 0
    assert failure["next_scenario_seed"] == seeds[0]
    assert failure["error_type"] == "RuntimeError"
    assert not (destination / "episode_rows.jsonl").exists()
    assert not (destination / "summary.json").exists()
