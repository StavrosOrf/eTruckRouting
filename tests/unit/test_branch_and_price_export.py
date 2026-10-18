"""The branch-and-price exporter must hand Julia exactly the nominal problem the env executes."""

import math
import os

import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from copy import deepcopy

from EVRoutingEnv.baselines.optimization.branch_and_price.export import (
    SCHEMA,
    export_nominal_instance,
    station_charger_settings,
)
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


@pytest.fixture(scope="module")
def exported():
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["environment"].update(
        {"num_stops": 6, "num_trucks": 2, "allow_variable_num_stops": False}
    )
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    env.reset(seed=11)
    return env, export_nominal_instance(env)


def test_nodes_and_tables_match_the_environment(exported) -> None:
    env, instance = exported
    assert instance["schema"] == SCHEMA
    kinds = [node["kind"] for node in instance["nodes"]]
    assert kinds[0] == "depot" and instance["nodes"][0]["id"] == env.joint_instance.depot_node
    assert kinds.count("customer") == len(env.joint_instance.tasks)
    assert kinds.count("charger") == len(env.charging_nodes)

    ids = [node["id"] for node in instance["nodes"]]
    for i, source in enumerate(ids):
        for j, target in enumerate(ids):
            energy, hours = env.transport_graph.get_leg_costs(source, target)
            exported_energy = instance["energy_kwh"][i][j]
            if math.isfinite(energy):
                assert exported_energy == pytest.approx(energy)
                assert instance["time_h"][i][j] == pytest.approx(hours)
            else:
                assert exported_energy is None


def test_cumulative_curve_reproduces_direct_charge_times(exported) -> None:
    env, instance = exported
    step = instance["charge_curves"]["soc_step"]
    battery = instance["vehicle"]["battery_kwh"]
    for charger in instance["chargers"]:
        table = instance["charge_curves"]["cumulative_hours"][charger["curve"]]
        kind, settings = station_charger_settings(env, charger["node"])
        for initial, target in [(0.1, 0.5), (0.35, 0.8), (0.5, 0.9), (0.62, 1.0)]:
            _, details = env.charging_curve_model.calculate_charge_to_target(
                initial, target, battery, settings, kind,
            )
            estimate = table[round(target / step)] - table[round(initial / step)]
            assert estimate == pytest.approx(details["actual_charge_hours"], abs=1e-9)


def test_margin_and_levels_follow_the_action_mask(exported) -> None:
    env, instance = exported
    assert instance["energy_margin"] == pytest.approx(
        max(1.0, env.traffic_config["max_energy_multiplier"])
    )
    assert instance["charge_levels"] == list(env.charge_action_values)
    assert len(instance["warm_start"]["witness"]) == env.num_trucks
