"""The BKS builders share one format, keep the best outcome per seed and lock the test split."""

import json

import pytest
from generalist_harness import small_config

from scripts.campaigns import generalist_bks_common as common
from scripts.campaigns import generalist_bks_dynamic as dynamic
from scripts.campaigns import generalist_bks_static as static
from scripts.evaluation.canonical_harness import split_seeds


def _row(seed: int, method: str, *, success=True, failed=False, served=30, lost=0, late=0, hours=300.0,
         dynamic_=False) -> dict:
    return {"problem": "p", "seed": seed, "method": method, "seconds": 1.0, "actions": [1, 2], "success": success,
            "failed": failed, "dynamic": dynamic_, "tasks": 30, "served": served, "late": late, "lost": lost,
            "unserved": 30 - served, "hours": hours}


def test_outcomes_order_feasibility_first_then_services_then_hours() -> None:
    best = _row(1, "a", hours=310.0)
    faster_but_incomplete = _row(1, "b", success=False, served=29, hours=200.0)
    stranded = _row(1, "c", failed=True, success=False, served=30, hours=100.0)
    ranked = sorted([stranded, faster_but_incomplete, best], key=common.outcome_key)
    assert [row["method"] for row in ranked] == ["a", "b", "c"]
    assert common.outcome_key(_row(1, "x", hours=290.0)) < common.outcome_key(best)
    assert common.outcome_key(_row(1, "x", lost=1, served=30)) > common.outcome_key(_row(1, "y", late=3, served=30))


def test_the_test_split_is_locked_unless_opened() -> None:
    with pytest.raises(SystemExit, match="locked"):
        common.seeds_for("test", 2)
    assert common.seeds_for("test", 2, allow_test=True) == split_seeds("test", 2)
    assert common.seeds_for("validation", 3, offset=1) == split_seeds("validation", 4)[1:]


def _write_runs(root, problem: str, split: str, rows_by_method: dict, bounds: list[dict]) -> None:
    folder = root / problem / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    for method, rows in rows_by_method.items():
        (folder / f"{method}__{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (folder / f"bound__{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in bounds))


def test_merge_keeps_the_best_per_seed_with_the_static_bound(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    seeds = split_seeds("validation", 3)
    _write_runs(tmp_path, "tx", "validation", {
        "greedy_teacher": [_row(s, "greedy_teacher", hours=320.0) for s in seeds],
        "lh_plan": [_row(seeds[0], "lh_plan", hours=300.0), _row(seeds[1], "lh_plan", success=False, served=28, hours=250.0),
                    {**_row(seeds[2], "lh_plan"), "crashed": True}],
    }, [{"seed": seeds[0], "value": 290.0, "proven": True}, {"seed": seeds[1], "value": None, "status": "TIME_LIMIT"}])
    result = common.merge_split("tx", "validation", static.bound_entry, "hours", count=3)
    assert set(result) == {str(s) for s in seeds}
    assert result[str(seeds[0])]["best"]["method"] == "lh_plan"
    assert result[str(seeds[1])]["best"]["method"] == "greedy_teacher"  # a complete plan beats a faster incomplete one
    assert result[str(seeds[2])]["best"]["method"] == "greedy_teacher"  # the crashed row is skipped
    assert result[str(seeds[0])]["bound"] == {"kind": "nominal_lower_bound_hours", "value": 290.0, "proven": True}
    assert result[str(seeds[1])]["bound"] == common.NO_BOUND and result[str(seeds[2])]["bound"] == common.NO_BOUND
    assert set(result[str(seeds[0])]["best"]) == {"method", *(f for f in common.SUMMARY_FIELDS if f != "method")}
    written = json.loads((tmp_path / "tx" / "validation.json").read_text())
    assert written == result
    summary = json.loads((tmp_path / "tx" / "summary.json").read_text())
    assert summary["bks"]["best_method_counts"] == {"greedy_teacher": 2, "lh_plan": 1}
    assert summary["bound"]["seeds"] == 1 and summary["bound"]["mean_gap"] == pytest.approx(300.0 / 290.0 - 1.0)
    assert summary["methods"]["greedy_teacher"]["mean_excess_vs_bks"] == pytest.approx((320.0 / 300.0 - 1.0) / 3)


def test_merge_records_the_clairvoyant_bound_for_dynamic_problems(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    seeds = split_seeds("validation", 2)
    _write_runs(tmp_path, "ca", "validation", {
        "greedy_dispatch": [_row(s, "greedy_dispatch", success=False, served=30, hours=260.0, dynamic_=True) for s in seeds],
        "rolling_request": [_row(seeds[0], "rolling_request", success=False, served=36, hours=270.0, dynamic_=True),
                            _row(seeds[1], "rolling_request", success=False, served=30, lost=2, hours=240.0, dynamic_=True)],
    }, [{"seed": seeds[0], "kind": dynamic.BOUND_KIND, "value": 35, "proven": False},
        {"seed": seeds[1], "kind": dynamic.BOUND_KIND, "value": 40, "proven": False}])
    result = common.merge_split("ca", "validation", dynamic.bound_entry, "served", count=2)
    assert result[str(seeds[0])]["best"]["method"] == "rolling_request"
    assert result[str(seeds[1])]["best"]["method"] == "greedy_dispatch"  # fewer lost requests at equal services
    # The clairvoyant planner served 35 where the BKS served 36: the bound is floored at the BKS and marked.
    assert result[str(seeds[0])]["bound"] == {"kind": "clairvoyant_rolling_served", "value": 36.0, "proven": False,
                                              "clairvoyant": 35.0, "raised_to_bks": True}
    assert result[str(seeds[1])]["bound"] == {"kind": "clairvoyant_rolling_served", "value": 40.0, "proven": False,
                                              "clairvoyant": 40.0, "raised_to_bks": False}
    summary = json.loads((tmp_path / "ca" / "summary.json").read_text())
    assert summary["bound"]["exceeded_seeds"] == 0 and summary["bound"]["clairvoyant_below_bks_seeds"] == 1
    assert summary["bound"]["mean_ratio"] == pytest.approx((36 / 36 + 30 / 40) / 2)
    assert summary["methods"]["greedy_dispatch"]["served_ratio_vs_bks"] == pytest.approx(60 / 66)
    # The gate's reader and the paired-reward reader both see the fields they need.
    from algo.generalist_imitation import expert_reference_rows
    rows = expert_reference_rows(tmp_path / "ca")
    assert rows[seeds[0]]["served"] == 36 and rows[seeds[0]]["dynamic"] is True


def test_the_test_split_merge_keeps_its_own_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    seeds = split_seeds("test", 2)
    _write_runs(tmp_path, "tx", "test", {"greedy_teacher": [_row(s, "greedy_teacher") for s in seeds]}, [])
    with pytest.raises(SystemExit, match="locked"):
        common.merge_split("tx", "test", static.bound_entry, "hours", count=2)
    common.merge_split("tx", "test", static.bound_entry, "hours", count=2, allow_test=True)
    assert (tmp_path / "tx" / "test.json").exists() and (tmp_path / "tx" / "summary_test.json").exists()
    assert not (tmp_path / "tx" / "summary.json").exists()


def test_builders_expose_their_portfolios() -> None:
    assert static.METHODS == ("greedy_teacher", "alns_respect", "lh_plan", "lh_rolling", "lh_plan_safe")
    assert callable(static.build_method("greedy_teacher", "tx_longhaul"))
    assert dynamic.METHODS == ("greedy_dispatch", "wait_at_depot", "kullman_myopic", "rolling_request")
    assert type(dynamic.build_method("rolling_request", "ca_sameday")).__name__ == "RollingHorizonRequestPolicy"
    assert not dynamic.build_method("rolling_request", "ca_sameday").parameters.clairvoyant
    assert static.bound_entry({"value": 1.0, "proven": True}, {"hours": 2.0}) == {
        "kind": "nominal_lower_bound_hours", "value": 1.0, "proven": True}
    assert static.bound_entry({"value": None}) is None and dynamic.bound_entry({"value": None}, {"served": 3}) is None
    assert callable(dynamic.build_method("greedy_dispatch", "ca_sameday"))
    with pytest.raises(ValueError):
        dynamic.build_method("lh_plan", "ca_sameday")
    with pytest.raises(SystemExit, match="public stations"):
        dynamic.check_methods("ca_sameday", ["greedy_dispatch", "kullman_myopic"])
    dynamic.check_methods("ca_drayage_combined", ["kullman_myopic"])


def test_the_reduced_instances_keep_the_envelope_and_drop_the_draws() -> None:
    config = small_config("same_day", trucks=2, stops=4)
    config["public_charging"] = {"enabled": True}
    reduced = dynamic.reduce(config, trucks=1, per_epoch=1, max_time=8.0)
    assert reduced["environment"]["num_trucks"] == 1 and reduced["environment"]["max_time"] == 8.0
    assert reduced["requests"]["max_requests_per_epoch"] == 1
    assert reduced["requests"]["max_requests"] == config["requests"]["max_requests"]  # the envelope is never a cap
    assert reduced["public_charging"]["enabled"] is False and reduced["truck"]["battery_capacity"] == 1e6
    assert reduced["traffic"]["enable_traffic"] is False
    assert config["environment"]["num_trucks"] == 2  # the input is not touched
    kept = dynamic.reduce(config, keep_public=True, stochastic=True)
    assert kept["public_charging"]["enabled"] is True and kept["truck"]["battery_capacity"] != 1e6


def test_the_bound_planner_is_clairvoyant_at_the_raised_budget() -> None:
    from generalist_harness import small_env

    env = small_env("same_day")
    planner = dynamic.bound_planner("same_day", env)
    assert planner.parameters.clairvoyant and planner.parameters.iterations == dynamic.BOUND_ITERATIONS
    assert planner.parameters.min_iterations == dynamic.BOUND_MIN_ITERATIONS
    with pytest.raises(ValueError, match="dynamic request mode"):
        dynamic.bound_planner("static", small_env("static"))


def test_the_clairvoyant_planner_never_beats_the_exact_hindsight_search(tmp_path) -> None:
    config = dynamic.reduce(small_config("same_day", trucks=2, stops=4), trucks=1, per_epoch=1, max_time=8.0)
    summary = dynamic.run_check("same_day_small", config, split_seeds("validation", 4), max_nodes=50_000,
                                stochastic=False, out_dir=tmp_path)
    assert summary["planner_exceeds"] == 0 and summary["seeds"] == 4
    report = json.loads((tmp_path / "hindsight_check.json").read_text())
    assert report["summary"] == summary and len(report["rows"]) == 4
