"""Provenance, strict JSON, and split-seed contract tests."""

import json

import pytest

from EVRoutingEnv.evaluation.artifacts import (
    CampaignSeedPlan,
    canonical_json_sha256,
    collect_run_manifest,
    write_immutable_manifest,
)


def test_seed_namespaces_are_stable_and_disjoint() -> None:
    plan = CampaignSeedPlan(base_seed=17)
    train = {plan.seed("train", index) for index in range(100)}
    validation = {plan.seed("validation", index) for index in range(100)}
    test = {plan.seed("test", index) for index in range(100)}

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert plan.seed("test", 9) == CampaignSeedPlan(17).seed("test", 9)


@pytest.mark.parametrize(
    ("split", "index"),
    [("unknown", 0), ("train", -1), ("test", 1_000_000_000)],
)
def test_seed_plan_rejects_unknown_or_out_of_range_values(split, index) -> None:
    with pytest.raises(ValueError):
        CampaignSeedPlan().seed(split, index)


def test_canonical_config_hash_ignores_mapping_order_and_rejects_nan() -> None:
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256(
        {"b": 2, "a": 1}
    )
    with pytest.raises(ValueError):
        canonical_json_sha256({"invalid": float("nan")})


def test_manifest_is_strict_json_and_cannot_be_overwritten(tmp_path) -> None:
    config = {"environment": {"num_trucks": 2}, "algorithm": {"lr": 3e-4}}
    manifest = collect_run_manifest(
        run_id="smoke-001",
        algorithm="heuristic",
        split="validation",
        command=["python", "evaluate.py", "--seed", "7"],
        resolved_config=config,
        scenario_seeds=[1_000_000_007, 1_000_000_008],
        repository_root=".",
    )
    config["environment"]["num_trucks"] = 999
    path = tmp_path / "nested" / "manifest.json"

    write_immutable_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "evrp-run-manifest-v1"
    assert payload["resolved_config"]["environment"]["num_trucks"] == 2
    assert payload["config_sha256"] == canonical_json_sha256(
        payload["resolved_config"]
    )
    assert isinstance(payload["git_dirty"], bool)
    assert payload["dependencies"]
    with pytest.raises(FileExistsError):
        write_immutable_manifest(manifest, path)
