"""The accelerated certificate must make exactly the decisions of the per-pair reference."""

import numpy as np
import pytest

from EVRoutingEnv.models.environment.route_certificate import construct_energy_routes
from EVRoutingEnv.models.simulation.charging_curve import ChargingCurveModel


@pytest.mark.parametrize("peak,floor", [(150.0, 150.0), (350.0, 150.0), (750.0, 150.0), (100.0, 50.0)])
def test_fast_charge_hours_match_adaptive_integration(peak, floor):
    model = ChargingCurveModel(verbose=False)
    settings = dict(charge_rate=peak, efficiency=0.9, use_realistic_curve=True,
                    taper_start_soc=0.8, taper_power_min=floor)
    rng = np.random.default_rng(3)
    for _ in range(60):
        soc = float(rng.uniform(0.0, 0.95))
        target = float(rng.choice([1.0, 0.9, 0.8, 0.6]))
        if target <= soc + 1e-6:
            continue
        _, details = model.calculate_charge_to_target(soc, target, 400.0, settings, "DCFast")
        fast = model.charge_hours_to_target_fast(soc, target, 400.0, peak, 0.9, 0.8, floor)
        assert fast == pytest.approx(details["actual_charge_hours"], rel=1e-7)


@pytest.mark.parametrize("peak,floor", [(150.0, 150.0), (350.0, 150.0), (750.0, 150.0)])
def test_replay_charging_without_plot_samples_matches_physical_curve(peak, floor):
    full = ChargingCurveModel()
    replay = ChargingCurveModel(record_power_curve=False)
    settings = {"charge_rate": peak, "efficiency": 0.9, "use_realistic_curve": True,
                "taper_start_soc": 0.8, "taper_power_min": floor}
    rng = np.random.default_rng(82)
    for _ in range(40):
        soc = float(rng.uniform(0.0, 0.99))
        target = float(rng.uniform(soc + 1e-6, 1.0))
        expected, details = full.calculate_charge_to_target(soc, target, 400.0, settings)
        actual, fast_details = replay.calculate_charge_to_target(soc, target, 400.0, settings)
        assert actual == expected
        assert fast_details["actual_charge_hours"] == pytest.approx(details["actual_charge_hours"], rel=1e-9)


class _Task:
    def __init__(self, node_id):
        self.node_id = node_id


class _Instance:
    def __init__(self, depot, tasks):
        self.depot_node, self.tasks, self.witness_owners = depot, tasks, tuple(0 for _ in tasks)


class _Graph:
    """Per-pair reference graph (no cost_matrices): forces the original scan."""

    def __init__(self, energy, hours):
        self.energy, self.hours = energy, hours

    def get_path_energy(self, a, b):
        return self.energy[a][b]

    def get_time_distance(self, a, b):
        return self.hours[a][b]


class _MatrixGraph(_Graph):
    def cost_matrices(self):
        n = len(self.energy)
        return ({i: i for i in range(n)}, np.array(self.energy, float), np.array(self.hours, float))


@pytest.mark.parametrize("seed", [1, 2, 3, 0, 4])
def test_vectorised_certificate_equals_reference(seed):
    rng = np.random.default_rng(seed)
    n = 26
    xy = rng.uniform(0, 300, (n, 2))
    dist = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    energy = (dist * 1.2 * (1 + 0.1 * rng.random((n, n)))).tolist()   # asymmetric, kWh
    hours = (dist / 60.0).tolist()
    chargers = list(range(1, 21))
    instance = _Instance(0, [_Task(21), _Task(23), _Task(25)])

    def charge(node, battery):
        return 120.0, 0.5 + 0.001 * (120.0 - battery), 1.0

    kwargs = dict(num_trucks=1, battery_capacity=120.0, initial_batteries=[120.0],
                  energy_multiplier=1.1, charge_transition=charge)
    try:
        reference = construct_energy_routes(instance, _Graph(energy, hours), chargers, **kwargs)
    except ValueError as expected:
        with pytest.raises(type(expected)):
            construct_energy_routes(instance, _MatrixGraph(energy, hours), chargers, **kwargs)
        return
    assert construct_energy_routes(instance, _MatrixGraph(energy, hours), chargers, **kwargs) == reference
    if seed in (1, 2, 3):  # these need several charging stops, so the label search is really exercised
        assert sum(action[2] for action in reference[0]) >= 5
