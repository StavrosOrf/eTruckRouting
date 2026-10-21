"""Numbers behind the component oracles, for the review-gate reports.

Run from the repository root:
    PYTHONPATH=. python scripts/analysis/report_component_oracles.py > oracles.json

The tests assert these properties; this script reports the measured values so
that a reviewer can see the margins: the outside-only Erlang C check with
batch-means 99% intervals, Little's law, the stationary start, port
availability, the transient expected wait, Kullman's single-truck exactness,
Keskin's stationary waits, the HOS clock/audit agreement and the Bai DP.
"""

from __future__ import annotations

import json
import math
import random

import numpy as np
from scipy import stats

from EVRoutingEnv.baselines.optimization.bai2023_dp import FixedRoute, brute_force, linear_charge_hours, solve
from EVRoutingEnv.models.core import hours_of_service as h
from EVRoutingEnv.models.core.hos_audit import AuditSettings, hos_audit
from EVRoutingEnv.models.simulation import queueing as q
from EVRoutingEnv.models.simulation.public_station_queue import StationParameters, StationQueue
from EVRoutingEnv.models.simulation.scenario import ScenarioRandomStreams


def erlang_table() -> list[dict]:
    rows = []
    for ports in (1, 2, 4, 10, 20):
        for utilization in (0.40, 0.65, 0.90):
            mu = 2.0
            lam = utilization * ports * mu
            customers = 100_000 if utilization < 0.9 else 300_000
            station = StationQueue(
                StationParameters(node=2, ports=ports, utilization=utilization, service_mean_hours=0.5),
                ScenarioRandomStreams(ports * 100 + int(utilization * 100)),
            )
            station.advance(1.05 * customers / lam + 200.0)
            started = {k: t for kind, k, t, _ in station.log.starts if kind == "outside"}
            waits = np.array([started[k] - a for k, a, _ in station.log.arrivals
                              if k >= 0 and k in started and a > 100.0])
            means = np.array([b.mean() for b in np.array_split(waits, 30)])
            half = 2.756 * means.std(ddof=1) / math.sqrt(len(means))
            oracle = q.mean_wait(ports, lam, mu)
            rows.append({
                "ports": ports, "u": utilization, "customers": int(len(waits)),
                "simulated_wq_h": float(means.mean()), "ci99_half_width_h": float(half),
                "erlang_c_wq_h": oracle, "inside_ci": bool(abs(means.mean() - oracle) <= half),
                "erlang_c_probability": q.erlang_c(ports, lam, mu),
            })
    return rows


def littles_law() -> dict:
    c, u, mu = 3, 0.65, 2.0
    lam = u * c * mu
    station = StationQueue(StationParameters(node=4, ports=c, utilization=u, service_mean_hours=0.5),
                           ScenarioRandomStreams(4))
    horizon = 30_000.0
    station.advance(horizon)
    started = {k: t for kind, k, t, _ in station.log.starts if kind == "outside"}
    queue_time = sum(min(started.get(k, horizon), horizon) - max(a, 0.0)
                     for k, a, _ in station.log.arrivals if k >= 0)
    waits = [started[k] - a for k, a, _ in station.log.arrivals if k >= 0 and k in started]
    lq, wq = queue_time / horizon, float(np.mean(waits))
    return {"L_q": lq, "lambda_W_q": lam * wq, "relative_error": abs(lq - lam * wq) / (lam * wq)}


def stationary_start() -> dict:
    c, u, mu = 2, 0.65, 2.0
    params = StationParameters(node=5, ports=c, utilization=u, service_mean_hours=0.5)
    counts = np.array([StationQueue(params, ScenarioRandomStreams(s)).in_system() for s in range(10_000)])
    pi = q.stationary_distribution(c, u * c * mu, mu, 200)
    cut = 8
    observed = np.append(np.bincount(np.minimum(counts, cut), minlength=cut + 1)[:cut], (counts >= cut).sum())
    expected = np.append(pi[:cut], pi[cut:].sum()) * len(counts)
    return {"resets": len(counts), "chi_square_p": float(stats.chisquare(observed, expected).pvalue)}


def availability() -> dict:
    a = 0.733
    station = StationQueue(StationParameters(node=8, ports=1, utilization=0.01, service_mean_hours=0.1,
                                             outages=True, availability=a, mean_repair_hours=2.0),
                           ScenarioRandomStreams(8))
    horizon = 400_000.0
    station.advance(horizon)
    changes = [(t, up) for _, t, up in station.log.port_changes]
    up_time, last, state = 0.0, 0.0, not changes[0][1]
    for t, up in changes:
        up_time += (t - last) if state else 0.0
        last, state = t, up
    up_time += (horizon - last) if state else 0.0
    return {"target": a, "measured": up_time / horizon, "cycles": len(changes) // 2}


def transient() -> dict:
    worst = 0.0
    for c, u in ((1, 0.4), (2, 0.65), (4, 0.9), (10, 0.9)):
        mu = 1.0 / 0.4437
        lam = u * c * mu
        for n in (0, c, c + 7):
            for age in (0.05, 0.5, 2.0, 10.0):
                size = q._round_size(max(q.truncation_level(c, lam, mu), n + 1) + 1)
                worst = max(worst, abs(q.expected_wait_given_memory(c, lam, mu, n, age)
                                       - q.transient_expected_wait_expm(c, lam, mu, n, age, size)))
    c, mu = 2, 1.5
    lam = 0.65 * c * mu
    long_gap = max(abs(q.expected_wait_given_memory(c, lam, mu, n, 400.0) - q.mean_wait(c, lam, mu))
                   for n in (0, 2, 9))
    return {"max_abs_diff_vs_expm_h": worst, "max_abs_diff_to_erlang_c_at_400h": long_gap}


def kullman_single_truck() -> dict:
    worst, cases = 0.0, 0
    for utilization in (0.40, 0.65, 0.90):
        for seed in range(200):
            for node in range(5):
                params = StationParameters(node=node, ports=1 + node % 3, utilization=utilization,
                                           service_mean_hours=0.4437)
                t = float(np.random.default_rng((seed, node)).uniform(0.0, 24.0))
                outside = StationQueue(params, ScenarioRandomStreams(seed))
                outside.advance(t, fleet_arrival=True)
                free = [0.0] * params.ports
                for arrival, service in sorted((a, s) for _, a, s in outside.log.arrivals if a < t):
                    i = int(np.argmin(free))
                    free[i] = max(arrival, free[i]) + service
                clairvoyant = max(t, min(free))
                station = StationQueue(params, ScenarioRandomStreams(seed))
                station.advance(t, fleet_arrival=True)
                if station.has_free_port_for_newcomer():
                    realized = t
                else:
                    station.fleet_join(0, t)
                    realized = station.predict_reservation(0)
                worst = max(worst, abs(realized - clairvoyant))
                cases += 1
    return {"cases": cases, "max_abs_diff_h": worst}


def keskin() -> list[dict]:
    rows = []
    for c in (1, 3):
        rng = np.random.default_rng(c)
        mu, rho = 1.0 / 1.63, 0.8
        lam = rho * c * mu
        waits = np.array([q.sample_stationary_wait(c, lam, mu, tuple(rng.random(2))) for _ in range(40_000)])
        positive = waits[waits > 0]
        rows.append({
            "ports": c, "p_zero_theory": 1 - q.erlang_c(c, lam, mu), "p_zero_measured": float((waits == 0).mean()),
            "ks_p_positive_part": float(stats.kstest(positive, "expon", args=(0.0, 1 / (c * mu - lam))).pvalue),
        })
    return rows


def hos_agreement() -> dict:
    configs = [("us_fmcsa_property", {}), ("eu_561_2006", {}),
               ("eu_561_2006", {"queue_wait_counts_as_break": True, "charging_counts_as_break": False}),
               ("eu_561_2006", {"eu_extended_daily_driving": True, "eu_reduced_daily_rest": True}),
               ("eu_561_2006", {"eu_split_break": False}), ("bai2023_simplified", {})]
    rng = random.Random(2026)
    agree = illegal = 0
    A = h.HosActivity
    for index in range(10_000):
        name, options = configs[index % len(configs)]
        regime = h.HosRegime.build(name, **options)
        clock, t, intervals, flagged = h.DriverClock.fresh(), 0.0, [], set()
        for _ in range(rng.randint(1, 40)):
            activity = rng.choice(list(A))
            if activity is A.DRIVE:
                hours = rng.choice([rng.uniform(0, 3), h.driving_allowance(regime, clock), rng.uniform(0, 12)])
                if hours <= 1e-9:
                    continue
                if hours > h.driving_allowance(regime, clock) + 1e-9:
                    flagged.add(round(t + hours, 6))
            elif activity is A.REST_DAILY:
                try:
                    hours = rng.choice([h.rest_hours(regime, clock, "daily"), rng.uniform(0, 12)])
                except h.HosInfeasible:
                    continue
            elif activity is A.REST_BREAK:
                hours = rng.choice([h.rest_hours(regime, clock, "break"), rng.uniform(0, 1)])
            else:
                hours = rng.choice([rng.uniform(0, 1), rng.uniform(0, 0.3), 0.25, 0.5, 0.75])
            if hours <= 1e-9:
                continue
            intervals.append((t, t + hours, activity))
            clock = h.apply(regime, clock, activity, hours)
            t += hours
        audited = {round(v.time, 6) for v in hos_audit(intervals, AuditSettings(name, **options))}
        agree += audited == flagged
        illegal += bool(flagged)
    return {"schedules": 10_000, "agreeing": agree, "with_violations": illegal}


def bai_dp() -> dict:
    rng = random.Random(0)
    agree = feasible = 0
    charge = linear_charge_hours(150.0, 400.0)
    for _ in range(300):
        stops = rng.randint(1, 5)
        route = FixedRoute(tuple(rng.uniform(0.3, 2.6) for _ in range(stops)),
                           tuple(rng.uniform(0.1, 0.55) for _ in range(stops)))
        plan, exhaustive = solve(route, (0.5, 0.7, 0.9, 1.0), charge), brute_force(route, (0.5, 0.7, 0.9, 1.0), charge)
        same = (plan is None and exhaustive is None) or (
            plan is not None and exhaustive is not None and abs(plan.total_hours - exhaustive) <= 1e-9)
        agree += same
        feasible += plan is not None
    return {"instances": 300, "agreeing": agree, "feasible": feasible}


def main() -> None:
    report = {
        "erlang_c": erlang_table(),
        "littles_law": littles_law(),
        "stationary_start": stationary_start(),
        "port_availability": availability(),
        "transient_wait": transient(),
        "kullman_single_truck": kullman_single_truck(),
        "keskin_stationary_wait": keskin(),
        "hos_clock_vs_audit": hos_agreement(),
        "bai_dp_vs_enumeration": bai_dp(),
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
