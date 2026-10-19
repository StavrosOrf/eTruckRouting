"""Promoted waves must retain pinned sources and worker layouts on resume."""

import json
from types import SimpleNamespace

from scripts.campaigns import california_graphppo as campaign
from scripts.campaigns import california_graphppo_controller as controller


def test_singleton_bundle_keeps_original_worker_partition(tmp_path, monkeypatch):
    for name in ("logs", "source", "training"):
        (tmp_path / name).mkdir()
    manifest = {
        "runs": [{"name": "arm_s0", "seed": 0, "flags": []}],
        "python": "/tmp/environment/bin/python",
        "source": str(tmp_path / "source"),
        "config": str(tmp_path / "config.yaml"),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(campaign.os, "sched_getaffinity", lambda pid: set(range(18)))
    monkeypatch.setattr(campaign.signal, "signal", lambda *args: None)
    calls = []

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(campaign.subprocess, "Popen", launch)
    campaign.train_bundle(tmp_path, 0, 1000, 2)
    command, kwargs = calls[0]
    assert command[2] == "0,1,2,3,4,5,6,7,8"
    assert command[command.index("--rollout-workers") + 1] == "7"
    assert kwargs["cwd"] == manifest["source"]
    assert kwargs["env"]["PYTHONPATH"] == manifest["source"]
    total = int(command[command.index("--total-timesteps") + 1])
    assert total >= 12_000_000 and total % (16 * 128) == 0


def test_promoted_wave_exposes_pinned_source_and_handles_odd_run_counts(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("logs", "training"):
        (tmp_path / name).mkdir()
    runs = [{"name": f"arm_s{seed}", "seed": seed, "flags": []} for seed in range(3)]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "runs": runs,
                "source": str(source),
                "python": "/tmp/environment/bin/python",
            }
        )
    )
    calls = []

    def submit(command, **kwargs):
        calls.append(command)
        return "123\n"

    monkeypatch.setattr(controller.subprocess, "check_output", submit)
    monkeypatch.setattr(controller, "schedule", lambda *args: "456")
    controller.launch_wave(tmp_path, tmp_path, runs, 4000, 12000, {})
    result = json.loads((tmp_path / "promotion_4000.json").read_text())
    wave = tmp_path / "waves" / f"{tmp_path.name}_4000"
    assert (wave / "source").resolve() == source
    assert calls[0][calls[0].index("--bundles") + 1] == "2"
    assert result["coordinator_job"] == "456"
