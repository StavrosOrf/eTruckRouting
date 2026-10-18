"""Audit metrics must follow witness ownership and conserve travelled costs."""

import io
from types import SimpleNamespace

import networkx as nx
import pytest

from scripts.analysis import analyze_scenario_resets as audit


def fixture_env():
    tasks = [SimpleNamespace(node_id=n, demand=float(n)) for n in (1, 2, 3)]
    routes = (
        (
            (9, 0.0, False),
            (9, 1.0, True),
            (1, 0.0, False),
            (8, 0.0, False),
            (8, 1.0, True),
            (2, 0.0, False),
            (0, 0.0, False),
        ),
        ((3, 0.0, False), (0, 0.0, False)),
    )
    return SimpleNamespace(
        joint_instance=SimpleNamespace(
            depot_node=0, tasks=tasks, witness_owners=(0, 0, 1), witness_routes=routes
        ),
        charging_nodes=[8, 9],
        num_navigation_actions=10,
        feasibility_witness=((0, 0), (0, 10), (0, 2), (0, 1), (0, 3), (1, 4), (1, 5)),
        trucks=[
            SimpleNamespace(payload_capacity=20.0, delivery_sequence=[0, 1, 2, 3])
            for _ in range(2)
        ],
    )


def lookup(a, b):
    energy = float(10 + abs(a - b))
    return {
        "energy_kwh": energy,
        "travel_hours": energy / 100,
        "road_km": energy / 2,
        "distance_proxy_km": energy * 0.4,
    }


lookup.road = object()


def test_hops_follow_witness_partition_and_separate_charger_detours():
    hops, trucks = audit.extract_witness(fixture_env(), 7, "case", lookup)
    direct = [h for h in hops if h["kind"] == "delivery_direct"]
    travelled = [h for h in hops if h["kind"] == "delivery_travel"]
    navigation = [h for h in hops if h["kind"] == "navigation"]
    returns = [h for h in hops if h["kind"] == "return_travel"]
    assert len(direct) == len(travelled) == 3
    assert len(navigation) == 7
    assert len(returns) == 2
    assert [row["deliveries"] for row in trucks] == [2, 1]
    assert [(h["from_node"], h["to_node"]) for h in direct] == [(0, 1), (1, 2), (0, 3)]
    assert direct[1]["energy_kwh"] == lookup(1, 2)["energy_kwh"]
    assert (
        travelled[1]["energy_kwh"]
        == lookup(1, 8)["energy_kwh"] + lookup(8, 2)["energy_kwh"]
    )
    assert travelled[1]["planned_charges"] == 1
    # One planned charge was skipped in this artificial certified trace.
    assert trucks[0]["planned_charges"] == 2
    assert trucks[0]["certified_charges"] == 1
    assert trucks[0]["max_chargers_between_stops"] == 1
    assert trucks[1]["max_chargers_between_stops"] == 0
    for key in ("energy_kwh", "travel_hours", "road_km", "distance_proxy_km"):
        assert sum(t[key] for t in trucks) == pytest.approx(
            sum(h[key] for h in navigation)
        )
        assert sum(t[key] for t in trucks) == pytest.approx(
            sum(h[key] for h in travelled + returns)
        )


def test_repeated_customer_witness_is_detected():
    env = fixture_env()
    env.joint_instance.witness_routes = (
        env.joint_instance.witness_routes[0],
        ((1, 0.0, False), (0, 0.0, False)),
    )
    with pytest.raises(ValueError, match="repeats a customer"):
        audit.extract_witness(env, 7, "case", lookup)


def test_consecutive_station_visits_are_detected_even_without_charging():
    env = fixture_env()
    env.joint_instance.witness_routes = (
        ((9, 0.0, False), (8, 0.0, False), *env.joint_instance.witness_routes[0][2:]),
        env.joint_instance.witness_routes[1],
    )
    with pytest.raises(ValueError, match="consecutive charging stations"):
        audit.extract_witness(env, 7, "case", lookup)


def test_empty_distributions_are_missing_not_zero_length_routes():
    assert audit.distribution([]) == {"count": 0}
    assert audit.distribution([None, float("inf")]) == {"count": 0}
    stats = audit.distribution([10.0, 20.0, 30.0])
    assert stats["count"] == 3 and stats["mean"] == stats["median"] == 20.0


def test_capacity_rejections_keep_each_requested_seed(tmp_path, monkeypatch):
    graph = nx.complete_graph(3, create_using=nx.DiGraph)
    env = SimpleNamespace(
        transport_graph=SimpleNamespace(graph=graph, get_all_nodes=lambda: list(graph)),
        charging_nodes=[2],
        num_charging_nodes=1,
        num_stops=500,
        config={"truck": {"base_speed": 40.0}},
        close=lambda: None,
    )

    def reset(*, seed):
        raise ValueError("road graph has too few eligible depot/customer nodes")

    env.reset = reset
    monkeypatch.setattr(
        audit.module, "EventDrivenTruckEnv", lambda *args, **kwargs: env
    )
    monkeypatch.setattr(audit, "load_road_distances", lambda *args: None)
    records = {name: [] for name in ("resets", "trucks", "hops")}
    writers = {
        name: SimpleNamespace(writerow=rows.append) for name, rows in records.items()
    }
    base = {"environment": {}, "problem": {"max_customer_demand": 4.0}}
    summary, resets, hops, trucks = audit.run_case(
        base, (100, 5), [11, 12, 13], tmp_path, writers, io.StringIO()
    )
    assert [r["seed"] for r in resets] == [11, 12, 13]
    assert summary["requested_resets"] == summary["failed_resets"] == 3
    assert summary["certified_resets"] == 0
    assert summary["statuses"] == {"insufficient_map_nodes": 3}
    assert not hops and not trucks
    assert summary["hop_statistics"]["delivery_direct"]["road_km"] == {"count": 0}
    assert len(records["resets"]) == 3
