"""Paired reporting cannot gain quality by discarding failed scenarios."""

import pytest

from scripts.campaigns.report_california_graphppo import summarize


def row(success=True, cost=100, customer=1):
    return {
        "success": success,
        "total_operating_time": cost,
        "scenario": {"customers": [customer], "config_sha256": "policy_specific"},
    }


def test_report_retains_failures_and_uses_independent_reference():
    report = summarize(
        {
            "reference": {1: row(), 2: row()},
            "graphppo": {1: row(cost=90), 2: row(success=False, cost=0)},
        },
        ["reference"],
    )
    result = report["methods"]["graphppo"]
    assert result["success_rate"] == 0.5
    assert result["paired_reference_coverage"] == 1
    assert result["mean_paired_excess_percent"] == pytest.approx(-10)
    assert result["success_and_within5pct_fraction_all_scenarios"] == 0.5
    assert report["proven_optimality"] is False


def test_report_refuses_different_physical_instances():
    with pytest.raises(ValueError, match="physical scenario differs"):
        summarize(
            {"reference": {1: row()}, "graphppo": {1: row(customer=2)}}, ["reference"]
        )
