"""Regressions for the optional policy restriction, independent of legality."""

from types import SimpleNamespace

import numpy as np

from EVRoutingEnv.state import action_mask
from EVRoutingEnv.state.feasibility import ActionKind, FeasibilityResult


def environment(other_capacity=4.0, escape_energy=8.0):
    task = SimpleNamespace(is_available=True, preassigned_to=None, demand=5.0)
    trucks = [
        SimpleNamespace(
            truck_id=0,
            remaining_payload=10.0,
            current_battery=10.0,
            failed=False,
            is_complete=False,
        ),
        SimpleNamespace(
            truck_id=1,
            remaining_payload=other_capacity,
            failed=False,
            is_complete=False,
        ),
    ]
    return SimpleNamespace(
        active_truck_id=0,
        trucks=trucks,
        traffic_config={"enable_traffic": False, "enable_energy_uncertainty": False},
        joint_instance=SimpleNamespace(depot_node=0),
        charging_nodes=[2],
        charging_station=SimpleNamespace(station_available={2: True}),
        task_registry=SimpleNamespace(tasks=lambda: [task]),
        transport_graph=SimpleNamespace(
            get_path_energy=lambda source, target: escape_energy
        ),
    )


def test_shield_prevents_stranding_and_premature_payload_retirement(monkeypatch):
    decisions = [
        FeasibilityResult.allow(ActionKind.CUSTOMER, target_node=1, required_energy=4),
        FeasibilityResult.allow(ActionKind.DEPOT, target_node=0, required_energy=1),
        FeasibilityResult.allow(ActionKind.CHARGER, target_node=2, required_energy=2),
    ]
    monkeypatch.setattr(action_mask, "joint_action_feasibility", lambda env: decisions)
    result = action_mask.completion_action_mask(environment(), np.ones(3, dtype=bool))
    np.testing.assert_array_equal(result, [False, False, True])
    result = action_mask.completion_action_mask(
        environment(other_capacity=10), np.ones(3, dtype=bool)
    )
    assert result[1], "surplus vehicles may retire when remaining payload suffices"


def test_shield_fallback_preserves_execution_legality(monkeypatch):
    decisions = [
        FeasibilityResult.allow(ActionKind.CUSTOMER, target_node=1, required_energy=4),
        FeasibilityResult.allow(ActionKind.DEPOT, target_node=0, required_energy=1),
        FeasibilityResult.reject(
            action_mask.FeasibilityReason.INSUFFICIENT_ENERGY,
            ActionKind.CHARGER,
            target_node=2,
        ),
    ]
    monkeypatch.setattr(action_mask, "joint_action_feasibility", lambda env: decisions)
    legal = np.array([True, True, False])
    result = action_mask.completion_action_mask(environment(), legal)
    np.testing.assert_array_equal(result, legal)
    assert not result[2], "fallback must never expose an execution-illegal action"
