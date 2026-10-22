"""The near-optimal gate compares runs with the BKS per seed and keeps the test split locked."""

import json

import pytest

from scripts.campaigns import generalist_near_optimal_gate as gate


def _static_rows(hours: float, success: bool = True, failed: bool = False, seeds=range(4)) -> list[dict]:
    return [{"problem": "tx_longhaul", "seed": seed, "dynamic": False, "success": success, "failed": failed,
             "hours": hours, "served": 30 if success else 25, "tasks": 30} for seed in seeds]


def _static_bks(hours: float = 300.0, seeds=range(4)) -> dict:
    return {seed: {"best": {"method": "lh_plan", "success": True, "failed": False, "hours": hours, "served": 30},
                   "bound": {"kind": "nominal_lower_bound_hours", "value": 290.0, "proven": True}} for seed in seeds}


def _dynamic_rows(served: int, seeds=range(4)) -> list[dict]:
    return [{"problem": "ca_sameday", "seed": seed, "dynamic": True, "success": False, "failed": False,
             "hours": 250.0, "served": served, "tasks": 60} for seed in seeds]


def _dynamic_bks(served: int = 40, seeds=range(4)) -> dict:
    return {seed: {"best": {"method": "rolling_request", "success": False, "failed": False, "hours": 240.0,
                            "served": served},
                   "bound": {"kind": "clairvoyant_rolling_served", "value": 48, "proven": False}} for seed in seeds}


def test_static_problems_pass_within_five_percent_of_the_bks_hours() -> None:
    passed = gate.gate_problem("tx_longhaul", _static_rows(309.0), _static_bks(), resamples=200)
    assert passed["status"] == "PASS" and passed["criteria"]["hours"]["mean_excess"] == pytest.approx(0.03)
    assert passed["criteria"]["success"]["pass"] and passed["criteria"]["failure"]["pass"]
    too_slow = gate.gate_problem("tx_longhaul", _static_rows(330.0), _static_bks(), resamples=200)
    assert too_slow["status"] == "FAIL" and not too_slow["criteria"]["hours"]["pass"]
    stranded = gate.gate_problem("tx_longhaul", _static_rows(300.0, success=False, failed=True), _static_bks(), resamples=200)
    assert stranded["status"] == "FAIL" and not stranded["criteria"]["failure"]["pass"]
    assert gate.gate_problem("tx_longhaul", _static_rows(300.0), None)["status"] == "MISSING"
    assert gate.gate_problem("tx_longhaul", [], _static_bks())["status"] == "NO_ROWS"


def test_dynamic_problems_pass_at_ninety_five_percent_of_the_bks_services() -> None:
    passed = gate.gate_problem("ca_sameday", _dynamic_rows(39), _dynamic_bks(), resamples=200)
    assert passed["status"] == "PASS" and passed["criteria"]["services"]["ratio"] == pytest.approx(0.975)
    assert passed["criteria"]["hours"]["hindsight_bound_served"] == 48.0
    failed = gate.gate_problem("ca_sameday", _dynamic_rows(36), _dynamic_bks(), resamples=200)
    assert failed["status"] == "FAIL" and not failed["criteria"]["services"]["pass"]


def _campaign(tmp_path, split: str) -> tuple:
    campaign = tmp_path / "suite"
    folder = campaign / "evaluation" / split / "final"
    folder.mkdir(parents=True)
    (folder / "full_s0.json").write_text(json.dumps(_static_rows(306.0)))
    # Other files of the tag folder (promotion.json, a stray dict) are not runs.
    (folder / "promotion.json").write_text(json.dumps({"promoted": ["full"], "arms": {}}))
    (folder / "notes.json").write_text(json.dumps({"rows": "none"}))
    bks = tmp_path / "bks"
    (bks / "tx_longhaul").mkdir(parents=True)
    (bks / "tx_longhaul" / f"{split}.json").write_text(json.dumps({str(k): v for k, v in _static_bks().items()}))
    return campaign, bks, folder


def test_the_test_split_opens_only_with_allow_test(tmp_path) -> None:
    campaign, bks, folder = _campaign(tmp_path, "test")
    argv = ["--campaign", str(campaign), "--split", "test", "--tag", "final", "--bks-dir", str(bks),
            "--problems", "tx_longhaul", "--resamples", "100"]
    with pytest.raises(SystemExit, match="locked"):
        gate.main(argv)
    assert not (folder / "near_optimal_gate.json").exists()
    assert gate.main([*argv, "--allow-test"]) == 0
    report = json.loads((folder / "near_optimal_gate.json").read_text())
    assert report["passing_runs"] == ["full_s0"] and report["runs"]["full_s0"]["overall"] == "PASS"
    assert list(report["runs"]) == ["full_s0"]
    assert "(test split)" in (folder / "near_optimal_gate.md").read_text()


def test_validation_needs_no_flag(tmp_path) -> None:
    campaign, bks, folder = _campaign(tmp_path, "validation")
    assert gate.main(["--campaign", str(campaign), "--split", "validation", "--tag", "final", "--bks-dir", str(bks),
                      "--problems", "tx_longhaul", "--resamples", "100"]) == 0
    assert (folder / "near_optimal_gate.md").exists()
