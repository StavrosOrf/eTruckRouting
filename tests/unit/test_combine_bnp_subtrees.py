"""Combining split branch-and-price runs: bounds over unexplored subtrees, best incumbent."""

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from scripts.evaluation.combine_bnp_subtrees import combine


def certificate(status, objective, bound, nodes=3, wall=10.0):
    return {"status": status, "objective_hours": objective, "best_bound_hours": bound, "nodes_explored": nodes,
            "nodes_open": 0 if status == "OPTIMAL" else 2, "wall_seconds": wall, "root_lp_hours": 90.0,
            "root_bound_hours": 90.0, "routes": [{"witness": [], "objective": objective}]}


def layout(tmp_path, results):
    (tmp_path / "split_7.json").write_text(json.dumps({"nodes": [
        {"label": f"node {k}", "bound": 95.0 + k, "forbidden": [], "lo": 1, "hi": 3, "depth": 2}
        for k in range(3)]}))
    (tmp_path / "seed_7.json").write_text(json.dumps(certificate("GAP", 101.0, 95.0, nodes=5, wall=50.0)))
    (tmp_path / "subtrees_7").mkdir()
    for k, cert in results.items():
        (tmp_path / "subtrees_7" / f"sub_{k}.result.json").write_text(json.dumps(cert))


def test_all_subtrees_closed_proves_the_best_incumbent(tmp_path) -> None:
    layout(tmp_path, {0: certificate("OPTIMAL", 100.0, 100.0), 1: certificate("OPTIMAL", 101.0, 101.0),
                      2: certificate("INFEASIBLE", None, None)})
    combined = combine(tmp_path, 7)
    assert combined["status"] == "OPTIMAL" and combined["proven_optimal"]
    assert combined["objective_hours"] == 100.0 == combined["best_bound_hours"]
    assert combined["nodes_explored"] == 5 + 3 + 3 + 3
    assert (tmp_path / "seed_7.split.json").exists()


def test_missing_subtree_counts_with_its_split_bound(tmp_path) -> None:
    layout(tmp_path, {0: certificate("OPTIMAL", 100.0, 100.0), 2: certificate("TIME_LIMIT", 100.5, 99.0)})
    combined = combine(tmp_path, 7)
    assert combined["status"] == "GAP"
    assert combined["best_bound_hours"] == 96.0      # subtree 1 has no result: its split bound
    assert combined["objective_hours"] == 100.0
    assert combined["parallel"]["finished"] == 2
    # combining again reads the split run's kept certificate
    assert combine(tmp_path, 7)["best_bound_hours"] == 96.0
