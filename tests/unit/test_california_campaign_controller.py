"""Campaign promotion prioritizes reliability over costs of partial success."""

import json

import pytest

from scripts.campaigns import california_graphppo_controller as controller
from scripts.campaigns.california_graphppo_controller import rank_runs


def test_promotion_prefers_reliability_over_incomplete_low_cost(tmp_path):
    runs = [
        {"name": "reliable_s0", "seed": 0, "flags": []},
        {"name": "fast_s0", "seed": 0, "flags": []},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": runs}))
    for run, success, cost in ((runs[0], 1.0, 250), (runs[1], 0.75, 100)):
        directory = tmp_path / "training" / run["name"]
        directory.mkdir(parents=True)
        (directory / "validation_history.json").write_text(
            json.dumps(
                [{"success_rate": success, "mean_operating_time_successful": cost}]
            )
        )
    ranked = rank_runs(tmp_path)
    assert ranked[0]["arm"] == "reliable"


def test_partial_clean_exit_resumes_before_promotion(tmp_path, monkeypatch):
    run = {"name": "shield_s0", "seed": 0, "flags": []}
    (tmp_path / "manifest.json").write_text(
        json.dumps({"runs": [run], "transition_budget": 1000})
    )
    directory = tmp_path / "training" / run["name"]
    directory.mkdir(parents=True)
    (directory / "training_history.json").write_text(json.dumps({"timesteps": [512]}))
    calls = []
    monkeypatch.setattr(controller, "launch_wave", lambda *args: calls.append(args))
    controller.advance(tmp_path, tmp_path, 4000)
    assert calls[0][3:5] == (1000, 4000)
    assert not (tmp_path / "training_complete.json").exists()


def test_joint_selection_hold_never_unlocks_separate_test_sets(tmp_path, monkeypatch):
    run = {"name": "arm_s0", "seed": 0, "flags": []}
    (tmp_path / "manifest.json").write_text(
        json.dumps({"runs": [run], "transition_budget": 1000})
    )
    directory = tmp_path / "training" / run["name"]
    directory.mkdir(parents=True)
    (directory / "training_history.json").write_text(json.dumps({"timesteps": [1000]}))
    (directory / "validation_history.json").write_text(
        json.dumps([{"success_rate": 1.0, "mean_operating_time_successful": 100}])
    )
    (tmp_path / "evaluation_hold.json").write_text(
        json.dumps({"reason": "joint selection"})
    )
    monkeypatch.setattr(
        controller.subprocess,
        "check_output",
        lambda *args, **kwargs: pytest.fail("test must stay locked"),
    )
    monkeypatch.setattr(controller, "maybe_start_joint_evaluation", lambda *args: None)
    controller.advance(tmp_path, tmp_path, 0)
    status = json.loads((tmp_path / "training_complete.json").read_text())
    assert status["status"] == "awaiting_joint_validation_selection"
    assert not status["goal_achieved"]


def test_joint_evaluation_waits_for_live_peer_training(tmp_path, monkeypatch):
    peer = tmp_path / "peer"
    peer.mkdir()
    monkeypatch.setattr(
        controller.subprocess, "check_output", lambda *args, **kwargs: "peer\n"
    )
    controller.maybe_start_joint_evaluation(
        tmp_path, {"campaigns": [str(tmp_path), str(peer)]}
    )
    assert not (tmp_path / "joint_final_evaluation.json").exists()


def test_joint_evaluation_collects_whole_five_seed_cohort(tmp_path, monkeypatch):
    peer = tmp_path / "peer"
    peer.mkdir()
    (tmp_path / "promotion_stopped.json").write_text("{}")
    runs = [
        {"name": f"progress_s{seed}", "seed": seed, "flags": []} for seed in range(5)
    ]
    (peer / "training_complete.json").write_text(
        json.dumps(
            {
                "status": "awaiting_joint_validation_selection",
                "ranking": [{"runs": runs}],
            }
        )
    )
    for run in runs:
        directory = peer / "training" / run["name"]
        directory.mkdir(parents=True)
        (directory / "training_history.json").write_text(
            json.dumps({"timesteps": [12_001_280]})
        )
    (tmp_path / "manifest.json").write_text(json.dumps({"python": "/tmp/python"}))
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return "" if command[0] == "squeue" else "123\n"

    monkeypatch.setattr(controller.subprocess, "check_output", execute)
    controller.maybe_start_joint_evaluation(
        peer, {"campaigns": [str(tmp_path), str(peer)]}
    )
    joined = json.loads((tmp_path / "joint_finalists/manifest.json").read_text())
    assert len(joined["runs"]) == 5
    assert all(str(peer) in run["training_directory"] for run in joined["runs"])
    assert (
        json.loads((tmp_path / "joint_final_evaluation.json").read_text())[
            "validation_job"
        ]
        == "123"
    )
