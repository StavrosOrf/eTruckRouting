"""The rolling request planner never starts a daily rest while a known request is due before it would end."""

import copy

import pytest
from generalist_harness import small_config

from EVRoutingEnv.baselines.heuristics.rolling_request_planner import (
    REST,
    DrayageModel,
    PlannerParameters,
    RollingHorizonRequestPolicy,
    build_snapshot,
)
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.feasibility import ActionKind, joint_action_feasibility
from scripts.campaigns.generalist_bks_dynamic import reduce

TERMINAL = {"name": "Port of Seattle, Terminal 18", "lat": 47.578, "lon": -122.353}


def _drayage_config() -> dict:
    """A one-truck, one-request-per-epoch drayage day with US hours of service, 12 h long."""
    config = small_config("same_day", trucks=1, stops=4)
    config["requests"].update(mode="drayage", max_requests_per_epoch=1, max_requests=40, static_context_size=20)
    config["drayage"] = {"terminals": [TERMINAL]}
    config["hours_of_service"] = {"enabled": True, "regime": "us_fmcsa_property"}
    return reduce(config, trucks=1, per_epoch=1, max_time=12.0, keep_hos=True)


def _episode(config: dict, seed: int, clairvoyant: bool) -> tuple[list[tuple[str, float]], int]:
    env = EventDrivenTruckEnv(copy.deepcopy(config), verbose=False, enable_plotting=False)
    env.reset(seed=seed)
    policy = RollingHorizonRequestPolicy(PlannerParameters(clairvoyant=clairvoyant))
    trail, done = [], False
    while not done and len(trail) < 80:
        decisions = joint_action_feasibility(env)
        action = int(policy(env))
        trail.append((decisions[action].action_kind, float(env.global_clock)))
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    served = env.request_engine.metrics(env.global_clock)["requests_served_on_time"]
    return trail, int(served), env


def _first_release(env) -> float | None:
    releases = [float(request.release_time) for request in env.request_engine.registry.requests()]
    return min(releases) if releases else None


@pytest.mark.parametrize("clairvoyant", [False, True])
def test_no_daily_rest_starts_while_a_known_request_is_due_before_it_ends(clairvoyant) -> None:
    config = _drayage_config()
    checked = 0
    for seed in range(12):
        trail, served, env = _episode(config, seed, clairvoyant)
        release = _first_release(env)
        if release is None:
            continue
        rest = env.hos.regime.daily_rest_hours
        rests = [t for kind, t in trail if kind is ActionKind.REST_DAILY]
        # The causal planner only knows a request once revealed; the clairvoyant one from the start.
        known_from = 0.0 if clairvoyant else max(0.0, release - env.extensions.requests.dispatch_margin_hours)
        early = [t for t in rests if t >= known_from - 1e-9 and t + rest > release + 1e-9]
        assert not early, (seed, clairvoyant, trail)
        checked += 1
        if checked >= 4:
            break
    assert checked >= 2


def test_rest_blocked_reads_the_releases_of_the_pending_jobs() -> None:
    config = _drayage_config()
    env = EventDrivenTruckEnv(copy.deepcopy(config), verbose=False, enable_plotting=False)
    for seed in range(12):
        env.reset(seed=seed)
        if len(env.request_engine.registry) >= 1:
            break
    else:
        pytest.skip("no seed with a request")
    model = DrayageModel(env, PlannerParameters(clairvoyant=True))
    snapshot = build_snapshot(env, model, clairvoyant=True)
    jobs = snapshot.jobs
    rest = model.ops.daily_rest
    earliest = min(job.release for job in jobs.values())
    assert model.rest_blocked(earliest - rest + 1.0, jobs, list(jobs))  # due during the rest
    assert model.rest_blocked(earliest + 1.0, jobs, list(jobs))  # already due
    assert not model.rest_blocked(earliest - rest - 1.0, jobs, list(jobs))  # due after the rest would end
    assert not model.rest_blocked(0.0, jobs, [REST])
    assert not model.rest_blocked(0.0, jobs, [])
