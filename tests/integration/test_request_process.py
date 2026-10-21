"""The request process against the EURO Meets NeurIPS 2022 filter (plan 8.4).

``_euro_next_observation_filter`` is a port of the feasibility filter in
``VRPEnvironment._next_observation`` of ortec/euro-neurips-vrp-2022-quickstart
(environment.py, MIT licence, Copyright (c) 2022 ORTEC). It is kept here as an
oracle, written in the original's terms (indices into a static instance whose
row 0 is the depot).
"""

import copy
from collections import Counter

import numpy as np
import pytest

from extension_harness import SAME_DAY

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config

pytestmark = pytest.mark.integration


def _euro_next_observation_filter(instance, planning_starttime, cust_idx, timewi_idx,
                                  service_t_idx):
    """EURO quickstart, ``_next_observation``: which sampled requests are kept."""
    duration_matrix = instance["duration_matrix"]
    new_request_timewi = instance["time_windows"][timewi_idx]
    earliest_arrival = np.maximum(
        planning_starttime + duration_matrix[0, cust_idx], new_request_timewi[:, 0]
    )
    earliest_return_at_depot = (
        earliest_arrival + instance["service_times"][service_t_idx] + duration_matrix[cust_idx, 0]
    )
    return (earliest_arrival <= new_request_timewi[:, 1]) & (
        earliest_return_at_depot <= instance["time_windows"][0, 1]
    )


def _env(**requests):
    config = copy.deepcopy(load_config("EVRoutingEnv/config_files/config_joint_seattle.yaml"))
    config["environment"].update(num_trucks=1, max_time=24.0)
    # Energy, charging and hours of service out of the way: only EURO's rule binds.
    config["truck"]["battery_capacity"] = 1e7
    config["requests"] = {**SAME_DAY, "loading_service_hours": 0.0, "max_requests_per_epoch": 12,
                          "max_requests": 400, **requests}
    config["depot_charging"] = {"enabled": False}
    return EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)


def test_kept_requests_equal_the_euro_filter_epoch_by_epoch() -> None:
    env = _env()
    for seed in range(3):
        env.reset(seed=seed)
        engine = env.request_engine
        context = engine.context
        depot = engine.depot
        index = engine._index
        nodes = [depot, *map(int, context.customers)]
        matrix = np.array([[engine._hours[index[a], index[b]] for b in nodes] for a in nodes])
        instance = {
            "duration_matrix": matrix,
            "time_windows": np.vstack([[0.0, env.max_time],
                                       np.column_stack([context.open, context.close])]),
            "service_times": np.concatenate([[0.0], context.service]),
            "demands": np.concatenate([[0.0], context.demand]),
        }
        size = len(context.customers)
        expected = Counter()
        cfg = env.extensions.requests
        for epoch_start in engine._epoch_starts:
            epoch = int(round(epoch_start / cfg.epoch_hours))
            draws = {
                field: env.scenario_random_streams.keyed_generator(
                    "request_epoch", ("epoch", epoch, field)
                ).integers(size, size=cfg.max_requests_per_epoch) + 1  # EURO skips the depot
                for field in ("customer", "time_window", "demand", "service")
            }
            planning = epoch_start + cfg.dispatch_margin_hours
            keep = _euro_next_observation_filter(
                instance, planning, draws["customer"], draws["time_window"], draws["service"]
            )
            for position in np.flatnonzero(keep):
                window = instance["time_windows"][draws["time_window"][position]]
                expected[(
                    epoch_start, planning, nodes[draws["customer"][position]],
                    float(window[0]), float(window[1]),
                    float(instance["demands"][draws["demand"][position]]),
                    float(instance["service_times"][draws["service"][position]]),
                )] += 1
        actual = Counter(
            (r.reveal_time, r.release_time, int(r.delivery.node), r.delivery.earliest,
             r.delivery.latest, r.demand, r.delivery.service_hours)
            for r in engine.registry.requests()
        )
        assert actual == expected
        assert sum(expected.values()) > 0


@pytest.mark.statistical
def test_poisson_arrival_counts_and_gaps() -> None:
    from scipy import stats

    rate, cutoff = 2.0, 20.0
    gaps, counts = [], []
    env = _env(arrival_process="poisson", poisson_rate_per_hour=rate, request_cutoff_hours=cutoff,
               time_windows={"enabled": False}, max_requests_per_epoch=1)
    # The arrival process itself, before the feasibility filter.
    for seed in range(400):
        env.reset(seed=seed)
        streams = env.scenario_random_streams
        time, arrivals = 0.0, []
        k = 0
        while True:
            time += streams.exponential("request_poisson", k, 1.0 / rate)
            if time > cutoff:
                break
            arrivals.append(time)
            k += 1
        counts.append(len(arrivals))
        # Every drawn gap, including the one that crosses the cutoff: keeping
        # only gaps that end before it would bias the sample short.
        gaps.extend(streams.exponential("request_poisson", j, 1.0 / rate) for j in range(k + 1))
    assert abs(np.mean(counts) - rate * cutoff) < 2.576 * np.sqrt(rate * cutoff / len(counts))
    assert stats.kstest(gaps, "expon", args=(0.0, 1.0 / rate)).pvalue > 0.01
