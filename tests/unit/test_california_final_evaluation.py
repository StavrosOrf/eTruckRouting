"""The final evaluation must preserve task scope and lock selection to validation."""

import json

import pytest

from scripts.campaigns import evaluate_california_graphppo as evaluation
from scripts.campaigns import report_california_graphppo as reporting
from scripts.campaigns.evaluate_california_graphppo import (
    archive_incomplete,
    base_manifest,
    check_manifests,
    choose_architecture,
    freeze_source,
    load_rows,
    validate_task_rows,
)
from scripts.campaigns.report_california_graphppo import hierarchical_intervals


def test_configuration_parity_allows_only_declared_policy_restriction():
    config = {"environment": {"max_time": 200, "num_trucks": 10, "num_stops": 30}}
    shielded = {"environment": {**config["environment"], "completion_shield": True}}
    check_manifests(
        {
            "baseline": {"resolved_config": config},
            "graphppo": {"resolved_config": shielded},
        }
    )
    shielded["environment"]["max_time"] = 400
    with pytest.raises(ValueError, match="configuration differs"):
        check_manifests(
            {
                "baseline": {"resolved_config": config},
                "graphppo": {"resolved_config": shielded},
            }
        )


def test_selects_complete_architecture_instead_of_cherry_picking_training_seed():
    methods, rows = {}, {}
    for arm in ("reliable", "unreliable"):
        for seed in range(5):
            name = f"{arm}_s{seed}"
            methods[name] = {"arm": arm, "training_seed": seed}
            rows[name] = {
                index: {
                    "success": arm == "reliable" or seed != 4,
                    "total_operating_time": 200 if arm == "reliable" else 100,
                }
                for index in range(10)
            }
    selected, _ = choose_architecture(methods, rows)
    assert selected["arm"] == "reliable"
    assert len(selected["per_seed"]) == 5


def test_selection_does_not_unlock_test_without_reliable_validation():
    methods = {
        f"arm_s{seed}": {"arm": "arm", "training_seed": seed} for seed in range(5)
    }
    rows = {
        name: {
            index: {"success": index != 0, "total_operating_time": 100}
            for index in range(10)
        }
        for name in methods
    }
    selected, _ = choose_architecture(methods, rows)
    assert selected is None


def test_partial_artifact_is_archived_before_retry(tmp_path):
    directory = tmp_path / "test/graphppo"
    directory.mkdir(parents=True)
    (directory / "episode_rows.jsonl.inprogress").write_text("partial")
    archive_incomplete(tmp_path, "test", ["graphppo"], {1, 2})
    assert not directory.exists()
    assert (
        len(
            list(
                (tmp_path / "failed_attempts").glob(
                    "*/graphppo/episode_rows.jsonl.inprogress"
                )
            )
        )
        == 1
    )


def test_duplicate_scenarios_cannot_inflate_sample_size(tmp_path):
    path = tmp_path / "rows.jsonl"
    value = json.dumps({"scenario_seed": 1, "total_operating_time": 100})
    path.write_text(value + "\n" + value + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_rows(path)


def test_two_level_interval_keeps_common_scenario_pairing():
    methods = {
        "reference": {
            index: {"success": True, "total_operating_time": 100} for index in range(10)
        }
    }
    for seed in range(5):
        methods[f"graphppo_s{seed}"] = {
            index: {"success": True, "total_operating_time": 102} for index in range(10)
        }
    interval = hierarchical_intervals(
        methods,
        ["reference"],
        [f"graphppo_s{seed}" for seed in range(5)],
        replicates=20,
    )
    assert interval["success_bootstrap95_across_training_seeds_and_scenarios"] == [1, 1]
    assert interval[
        "paired_excess_bootstrap95_across_training_seeds_and_scenarios"
    ] == pytest.approx([2, 2])


def test_frozen_source_cannot_be_changed_between_evaluation_shards(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("pass\n")
    config = tmp_path / "config.yaml"
    config.write_text("environment: {num_trucks: 10, num_stops: 30}\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"source": str(source), "config": str(config)})
    )
    for name in ("evaluation_pipeline.py", "paired_report.py"):
        (tmp_path / name).write_text("pass\n")
    evaluation = tmp_path / "final_evaluation"
    evaluation.mkdir()
    freeze_source(tmp_path, evaluation)
    base_manifest(tmp_path)
    (source / "model.py").write_text("changed = True\n")
    with pytest.raises(ValueError, match="input changed"):
        base_manifest(tmp_path)


def test_task_scope_and_full_service_are_required():
    value = {
        "customers_total": 30,
        "customers_served": 30,
        "success": True,
        "truncated": False,
        "scenario": {"num_trucks": 10},
    }
    validate_task_rows([value], "graphppo")
    value["customers_served"] = 29
    with pytest.raises(ValueError, match="contradicts"):
        validate_task_rows([value], "graphppo")
    value["scenario"]["num_trucks"] = 2
    with pytest.raises(ValueError, match="not 10 trucks"):
        validate_task_rows([value], "graphppo")


def test_aggregate_covers_every_locked_case_and_preserves_best_known_scope(
    tmp_path, monkeypatch
):
    methods = {"reference": {}}
    methods.update({f"graphppo_s{seed}": {"checkpoint": "frozen"} for seed in range(5)})
    lock = {
        "scenario_seeds": list(range(2_000_000_000, 2_000_000_010)),
        "references": ["reference"],
    }
    monkeypatch.setattr(evaluation, "TEST_COUNT", 10)
    monkeypatch.setattr(evaluation, "TEST_SHARDS", 2)
    monkeypatch.setattr(evaluation, "check_lock", lambda root: (lock, methods))
    monkeypatch.setattr(evaluation, "reporter", lambda root: reporting)
    for shard in range(2):
        for name in methods:
            directory = (
                tmp_path / "final_evaluation/test_shards" / str(shard) / "test" / name
            )
            directory.mkdir(parents=True)
            rows = [
                {
                    "scenario_seed": 2_000_000_000 + index,
                    "success": True,
                    "customers_total": 30,
                    "customers_served": 30,
                    "truncated": False,
                    "total_operating_time": 100 if name == "reference" else 102,
                    "scenario": {"num_trucks": 10},
                }
                for index in range(5 * shard, 5 * shard + 5)
            ]
            (directory / "episode_rows.jsonl").write_text(
                "\n".join(map(json.dumps, rows)) + "\n"
            )
            (directory / "manifest.json").write_text(
                json.dumps({"resolved_config": {}})
            )
    evaluation.aggregate(tmp_path)
    result = json.loads(
        (tmp_path / "final_evaluation/goal_assessment.json").read_text()
    )
    assert result["point_targets_met"]
    assert result["status"] == "targets_met_pending_goal_audit"
    assert result["certified_optimality"] is False
    assert result["training_seed_count"] == 5
    assert result["mean_paired_excess_percent_across_five_seeds"] == pytest.approx(2)
    missing = (
        tmp_path / "final_evaluation/test_shards/1/test/graphppo_s4/episode_rows.jsonl"
    )
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        evaluation.aggregate(tmp_path)
