"""Campaign promotion prioritizes reliability over costs of partial success."""

import json

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
