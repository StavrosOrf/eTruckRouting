"""Public station queue: tie rules, a hand-computed FCFS table, and oracles."""

import math

import numpy as np
import pytest

from EVRoutingEnv.models.simulation import queueing as q
from EVRoutingEnv.models.simulation.public_station_queue import StationParameters, StationQueue
from EVRoutingEnv.models.simulation.scenario import ScenarioRandomStreams


class ScriptedStation(StationQueue):
    """Outside arrivals and services from a script instead of keyed draws."""

    def __init__(self, ports, arrivals):
        self._script = list(arrivals)  # [(time, service)]
        super().__init__(
            StationParameters(node=1, ports=ports, utilization=0.5, service_mean_hours=1.0,
                              initial_state="empty"),
            ScenarioRandomStreams(0),
        )

    def _schedule_next_arrival(self) -> None:
        k = self._arrivals_k
        self._next_arrival = (self._script[k][0], k) if k < len(self._script) else None

    def _exp(self, stream, key, mean):
        if stream == "station_service":
            return self._script[key][1]
        return super()._exp(stream, key, mean)


def _starts(station):
    return {(kind, key): (time, port) for kind, key, time, port in station.log.starts}


def test_hand_built_two_port_fcfs_table() -> None:
    station = ScriptedStation(2, [(0.0, 2.0), (0.5, 1.0), (1.0, 3.0)])
    station.advance(1.2, fleet_arrival=True)
    station.fleet_join(1, 1.2)
    station.advance(1.4, fleet_arrival=True)
    station.fleet_join(2, 1.4)
    assert station.predict_reservation(1) == pytest.approx(2.0)
    assert station.predict_reservation(2) is None  # behind an undecided truck
    assert station.advance(2.0) == [(1, 2.0)]
    station.fleet_start(1, 2.0, 0.5)
    assert station.predict_reservation(2) == pytest.approx(2.5)
    assert station.advance(2.5) == [(2, 2.5)]
    station.fleet_start(2, 2.5, 1.0)
    station.advance(10.0)
    assert _starts(station) == {
        ("outside", 0): (0.0, 0),
        ("outside", 1): (0.5, 1),
        ("outside", 2): (1.5, 1),
        ("fleet", 1): (2.0, 0),
        ("fleet", 2): (2.5, 0),
    }


def test_A1_departure_before_arrival_at_equal_times() -> None:
    station = ScriptedStation(1, [(0.0, 1.0)])
    station.advance(1.0, fleet_arrival=True)
    assert station.has_free_port_for_newcomer()


def test_A2_fleet_before_outside_at_equal_arrival_times() -> None:
    station = ScriptedStation(1, [(0.0, 5.0), (1.0, 1.0)])
    station.advance(1.0, fleet_arrival=True)
    station.fleet_join(7, 1.0)
    station.advance(1.0)
    assert [(e.kind, e.key) for e in station.queue] == [("fleet", 7), ("outside", 1)]


def test_A12_closure_sends_queued_vehicles_away_and_sessions_finish() -> None:
    station = ScriptedStation(1, [(0.0, 5.0), (0.5, 1.0), (3.0, 1.0)])
    station.advance(1.0, fleet_arrival=True)
    station.fleet_join(4, 1.0)
    released = station.close(1.0)
    assert released == [4] and station.queue == []
    assert station.log.balked_on_closure == 1  # the queued outside vehicle
    station.advance(3.5)
    assert station.log.balked_on_closure == 2  # arrivals while closed leave
    assert station.busy_ports() == 1  # the session in progress continues


def test_prediction_equals_lazy_simulation_including_outages() -> None:
    rng = np.random.default_rng(3)
    for trial in range(300):
        c = int(rng.integers(1, 5))
        params = StationParameters(
            node=3, ports=c, utilization=float(rng.uniform(0.4, 0.95)), service_mean_hours=0.44,
            outages=bool(trial % 2), availability=0.733, mean_repair_hours=1.0,
        )
        station = StationQueue(params, ScenarioRandomStreams(trial))
        t = float(rng.uniform(0, 50))
        station.advance(t, fleet_arrival=True)
        if station.has_free_port_for_newcomer():
            continue
        station.fleet_join(99, t)
        predicted = station.predict_reservation(99)
        got = None
        while got is None:
            res = station.advance(min(station.next_event_time(), station.clock + 5.0))
            got = next((time for key, time in res if key == 99), None)
        assert got == pytest.approx(predicted, abs=1e-9)


def _kiefer_wolfowitz_start(arrivals, ports: int, t: float) -> float:
    """Independent FCFS multi-server recursion for a newcomer at ``t``."""
    free = [0.0] * ports
    for arrival, service in sorted(arrivals):
        i = int(np.argmin(free))
        free[i] = max(arrival, free[i]) + service
    return max(t, min(free))


@pytest.mark.parametrize("utilization", [0.40, 0.65, 0.90])
def test_single_truck_wait_equals_kullman_clairvoyant_wait(utilization) -> None:
    """Kullman et al. (2021) Fig. 4: with one fleet truck, its realized wait is the
    clairvoyant wait function of the outside-only realization of the same seed."""
    for seed in range(200):
        for node in range(5):
            params = StationParameters(node=node, ports=1 + node % 3, utilization=utilization,
                                       service_mean_hours=0.4437)
            t = float(np.random.default_rng((seed, node)).uniform(0.0, 24.0))
            outside_only = StationQueue(params, ScenarioRandomStreams(seed))
            outside_only.advance(t, fleet_arrival=True)
            before = [(a, s) for k, a, s in outside_only.log.arrivals if a < t]
            clairvoyant = _kiefer_wolfowitz_start(before, params.ports, t)

            with_truck = StationQueue(params, ScenarioRandomStreams(seed))
            with_truck.advance(t, fleet_arrival=True)
            if with_truck.has_free_port_for_newcomer():
                realized = t
            else:
                with_truck.fleet_join(0, t)
                realized = with_truck.predict_reservation(0)
            assert realized == pytest.approx(clairvoyant, abs=1e-9), (seed, node)


def _waits(station, after):
    started = {key: time for kind, key, time, port in station.log.starts if kind == "outside"}
    return [
        (arrival, started[k] - arrival)
        for k, arrival, service in station.log.arrivals
        if k >= 0 and k in started and arrival > after
    ]


@pytest.mark.statistical
@pytest.mark.parametrize("ports", [1, 2, 4, 10, 20])
@pytest.mark.parametrize("utilization", [0.40, 0.65, 0.90])
def test_outside_only_station_matches_erlang_c(ports, utilization) -> None:
    mu = 2.0  # 0.5 h mean service, as in the shipped DCFast lookup
    lam = utilization * ports * mu
    customers = 100_000 if utilization < 0.9 else 300_000
    horizon = 1.05 * customers / lam + 200.0
    station = StationQueue(
        StationParameters(node=2, ports=ports, utilization=utilization, service_mean_hours=0.5),
        ScenarioRandomStreams(ports * 100 + int(utilization * 100)),
    )
    station.advance(horizon)
    waits = np.array([w for _, w in _waits(station, 100.0)])
    batches = np.array_split(waits, 30)
    means = np.array([b.mean() for b in batches])
    half = 2.756 * means.std(ddof=1) / math.sqrt(len(means))  # t_{0.995, 29}
    oracle = q.mean_wait(ports, lam, mu)
    assert len(waits) >= 100_000
    assert abs(means.mean() - oracle) <= half + 1e-12


@pytest.mark.statistical
def test_littles_law_for_the_queue() -> None:
    c, u, mu = 3, 0.65, 2.0
    lam = u * c * mu
    station = StationQueue(StationParameters(node=4, ports=c, utilization=u, service_mean_hours=0.5),
                           ScenarioRandomStreams(4))
    horizon = 30_000.0
    station.advance(horizon)
    started = {key: time for kind, key, time, port in station.log.starts if kind == "outside"}
    in_queue_time = sum(
        min(started.get(k, horizon), horizon) - max(arrival, 0.0)
        for k, arrival, service in station.log.arrivals if k >= 0
    )
    mean_queue = in_queue_time / horizon
    mean_wait = np.mean([w for _, w in _waits(station, 0.0)])
    assert abs(mean_queue - lam * mean_wait) / (lam * mean_wait) < 0.02


@pytest.mark.statistical
def test_stationary_initial_state_chi_square() -> None:
    from scipy import stats

    c, u, mu = 2, 0.65, 2.0
    params = StationParameters(node=5, ports=c, utilization=u, service_mean_hours=0.5)
    counts = np.array([StationQueue(params, ScenarioRandomStreams(s)).in_system() for s in range(10_000)])
    pi = q.stationary_distribution(c, u * c * mu, mu, 200)
    cut = 8
    observed = np.append(np.bincount(np.minimum(counts, cut), minlength=cut + 1)[:cut], (counts >= cut).sum())
    expected = np.append(pi[:cut], pi[cut:].sum()) * len(counts)
    assert stats.chisquare(observed, expected).pvalue > 0.01


@pytest.mark.statistical
def test_piecewise_poisson_hourly_counts() -> None:
    profile = tuple([0.5] * 8 + [1.5] * 8 + [1.0] * 8)
    params = StationParameters(node=6, ports=4, utilization=0.5, service_mean_hours=0.5,
                               hourly_profile=profile, initial_state="empty")
    station = StationQueue(params, ScenarioRandomStreams(6))
    days = 3000
    station.advance(24.0 * days)
    counts = np.zeros((days, 24))
    for _, time, _ in station.log.arrivals:
        counts[int(time // 24), int(time % 24)] += 1
    for hour in range(24):
        expected = params.mean_arrival_rate * profile[hour]
        sample = counts[:, hour]
        assert abs(sample.mean() - expected) < 2.576 * math.sqrt(expected / days)
        assert abs(sample.var(ddof=1) - expected) < 2.576 * expected * math.sqrt(2.0 / days) + 0.05


@pytest.mark.statistical
def test_port_availability_long_run() -> None:
    a, repair = 0.733, 2.0
    params = StationParameters(node=8, ports=1, utilization=0.01, service_mean_hours=0.1,
                               outages=True, availability=a, mean_repair_hours=repair)
    station = StationQueue(params, ScenarioRandomStreams(8))
    horizon = 400_000.0
    station.advance(horizon)
    port = station.ports[0]
    up_time, last, state = 0.0, 0.0, None
    changes = [(t, up) for index, t, up in station.log.port_changes]
    # State before the first change is the opposite of the first change's new state.
    state = not changes[0][1]
    for t, up in changes:
        up_time += (t - last) if state else 0.0
        last, state = t, up
    up_time += (horizon - last) if state else 0.0
    cycles = len(changes) / 2
    # Alternating renewal: the up fraction's standard error from cycle counts.
    se = a * (1 - a) * math.sqrt(2.0 / max(cycles, 1.0))
    assert abs(up_time / horizon - a) < 2.576 * se
    assert port.change_index > 100
