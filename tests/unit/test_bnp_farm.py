"""Branch-and-price farm bookkeeping: bounds over open items, orphan reaping, race-safe snapshots."""

import json
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from scripts.evaluation import bnp_farm


def make_farm(tmp_path):
    for name in ("queue", "running", "done", "work", "incumbents", "cancelled"):
        (tmp_path / name).mkdir()
    (tmp_path / "incumbents" / "incumbent_a.json").write_text(json.dumps({"objective_hours": 100.0}))
    return tmp_path


def put(farm, where, item):
    (farm / where / f"{item['id']}.json").write_text(json.dumps(item))


def test_open_items_bound_and_optimality(tmp_path) -> None:
    farm = make_farm(tmp_path)
    put(farm, "queue", {"id": "s0", "bound": 98.0})
    put(farm, "running", {"id": "s1", "bound": 99.0, "started": time.time()})
    put(farm, "done", {"id": "s2", "bound": 97.0, "outcome": "split"})
    info = bnp_farm.summary(farm)
    assert info["lower"] == 98.0 and info["upper"] == 100.0 and not info["optimal"]
    (farm / "queue" / "s0.json").unlink()
    (farm / "running" / "s1.json").unlink()
    put(farm, "done", {"id": "s0", "bound": 98.0, "outcome": "closed"})
    info = bnp_farm.summary(farm)
    assert info["optimal"] and info["lower"] == 100.0


def test_reap_requeues_only_orphans(tmp_path) -> None:
    farm = make_farm(tmp_path)
    now = time.time()
    put(farm, "running", {"id": "old", "bound": 98.0, "started": now - 7200 - 1900, "budget": 7200, "pid": 1})
    put(farm, "running", {"id": "live", "bound": 98.5, "started": now - 600, "budget": 7200})
    put(farm, "running", {"id": "legacy", "bound": 99.0, "started": now - 9100})      # no budget recorded
    assert bnp_farm.reap(farm) == 2
    assert sorted(p.stem for p in (farm / "queue").glob("*.json")) == ["legacy", "old"]
    assert [p.stem for p in (farm / "running").glob("*.json")] == ["live"]
    requeued = json.loads((farm / "queue" / "old.json").read_text())
    assert requeued["outcome"] == "retry" and "pid" not in requeued and "started" not in requeued


def test_summary_retakes_snapshot_when_an_item_moves(tmp_path, monkeypatch) -> None:
    farm = make_farm(tmp_path)
    put(farm, "queue", {"id": "s0", "bound": 98.0})
    real_read, moved = bnp_farm._read, []

    def racing_read(path):
        # the first read of a queued item finds it already claimed by another process
        if path.parent.name == "queue" and not moved:
            moved.append(path)
            target = farm / "running" / path.name
            path.rename(target)
            raise FileNotFoundError(path)
        return real_read(path)

    monkeypatch.setattr(bnp_farm, "_read", racing_read)
    info = bnp_farm.summary(farm)
    assert moved and info["running"] == 1 and info["lower"] == 98.0 and not info["optimal"]
