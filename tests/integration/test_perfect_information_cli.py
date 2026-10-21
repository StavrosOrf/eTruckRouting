"""Saved clairvoyant plans replay across independent Python processes."""

import json
import os
import subprocess
import sys

import pytest


# A warm run takes about 5 s, but a cold start on a shared network filesystem
# spends up to a minute importing; the limit only guards against a hang.
TIMEOUT_SECONDS = 600


@pytest.mark.integration
def test_seeded_cli_solve_and_cross_process_replay(tmp_path):
    artifact = tmp_path / "plan.json"
    command = [sys.executable, "-m", "scripts.evaluation.solve_perfect_information"]
    environment = {
        **os.environ,
        "MPLCONFIGDIR": "/tmp/evrp_matplotlib",
        "PYTHONHASHSEED": "1",
    }
    subprocess.run(
        [
            *command,
            "--seed",
            "7",
            "--customers",
            "1",
            "--trucks",
            "1",
            "--no-time-limit",
            "--max-nodes",
            "1",
            "--output",
            str(artifact),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=TIMEOUT_SECONDS,
    )
    payload = json.loads(artifact.read_text())
    assert payload["solution"]["status"] == "FEASIBLE"
    assert not payload["solution"]["proven_optimal"]
    assert payload["solution"]["information_mode"] == "realized"
    assert payload["replay"]["operational_metrics"]["success"]
    assert payload["replay"]["routes"]
    replay = subprocess.run(
        [*command, "--replay", str(artifact)],
        env={**environment, "PYTHONHASHSEED": "987"},
        capture_output=True,
        text=True,
        check=True,
        timeout=TIMEOUT_SECONDS,
    )
    result = json.loads(replay.stdout.splitlines()[-1])
    assert result["verified"]
    assert result["replay_sha256"] == payload["replay"]["replay_sha256"]
