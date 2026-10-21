"""Invariants, common random numbers and non-anticipativity across components."""

import numpy as np
import pytest

from extension_harness import (
    HOS_EU,
    HOS_US,
    PUBLIC,
    SAME_DAY,
    InvariantChecker,
    audit_violations,
    build,
    choose,
    mask_matches_execution,
    perturb_after,
    perturb_hidden_requests,
    rollout,
)

pytestmark = pytest.mark.integration

FEATURES = {
    "public": {"public_charging": PUBLIC},
    "hos_us": {"hours_of_service": HOS_US},
    "hos_eu": {"hours_of_service": {**HOS_EU, "rest_locations": "stops_only"}},
    "same_day": {"requests": SAME_DAY},
    "all": {"public_charging": PUBLIC, "hours_of_service": {**HOS_US, "rest_locations": "stops_only"},
            "requests": SAME_DAY},
}


def _invariant_run(map_name, feature, seed, policy):
    env = build(map_name, **FEATURES[feature])
    checker = InvariantChecker(env)
    info, steps = rollout(env, seed, choose(env, policy, seed), check=checker)
    assert info["termination_reason"] is not None
    if env.hos is not None:
        assert audit_violations(env) == []
    return env, info


@pytest.mark.parametrize("feature", sorted(FEATURES))
@pytest.mark.parametrize("policy", ["greedy", "random"])
def test_invariants_hold_on_seattle(feature, policy) -> None:
    for seed in range(3):
        _invariant_run("seattle", feature, seed, policy)


@pytest.mark.statistical
@pytest.mark.parametrize("map_name", ["texas_longhaul", "california", "seattle"])
@pytest.mark.parametrize("feature", sorted(FEATURES))
def test_invariants_hold_on_two_hundred_seeds(map_name, feature) -> None:
    for seed in range(200):
        _invariant_run(map_name, feature, seed, "random" if seed % 2 else "greedy")


def test_common_random_numbers_across_policies() -> None:
    """Section 4.2: two policies on one seed see the same exogenous realization."""
    blocks = {"public_charging": PUBLIC, "requests": SAME_DAY}
    logs = []
    for policy in ("greedy", "random"):
        env = build("seattle", **blocks)
        rollout(env, 5, choose(env, policy, 5))
        for station in env.public_stations.queues.values():
            station.advance(env.max_time)  # extend every station to the horizon
        logs.append((
            {node: list(s.log.arrivals) for node, s in env.public_stations.queues.items()},
            {node: list(s.log.port_changes) for node, s in env.public_stations.queues.items()},
            [r.as_dict() | {"status": None, "owner": None, "claimed_at": None, "loaded_at": None,
                            "service_started_at": None, "served_at": None, "lost_at": None,
                            "delay_hours": 0.0}
             for r in env.request_engine.registry.requests()],
        ))
    assert logs[0] == logs[1]


@pytest.mark.parametrize("feature", ["public", "same_day", "all"])
def test_observations_and_masks_before_t_ignore_the_future(feature) -> None:
    """Invariant 8 under nominal_eta: resample everything after t; up to t the
    observations and masks are bit-identical."""
    blocks = dict(FEATURES[feature], observation={"in_flight_information": "nominal_eta"})
    for seed in (1, 2):
        cutoff = 6.0
        reference, perturbed = [], []
        env = build("seattle", **blocks)
        rollout(env, seed, choose(env, "greedy", seed), record=reference)
        env = build("seattle", **blocks)
        original_reset = env.reset

        def reset_and_perturb(seed=None, options=None, env=env):
            result = original_reset(seed=seed, options=options)
            perturb_after(env, cutoff)
            perturb_hidden_requests(env, cutoff)
            # Rebuild the first observation: nothing after the cutoff exists yet.
            return env._get_observation(), env._get_info()

        env.reset = reset_and_perturb
        rollout(env, seed, choose(env, "greedy", seed), record=perturbed)
        before = [row for row in reference if row[0] <= cutoff]
        assert before
        for (t1, obs1, mask1), (t2, obs2, mask2) in zip(before, perturbed, strict=False):
            assert t1 == t2
            assert np.array_equal(obs1, obs2), f"observation diverged at t={t1}"
            assert np.array_equal(mask1, mask2), f"mask diverged at t={t1}"
        # The perturbation is real: the futures differ.
        after = [(t, o) for t, o, _ in reference if t > cutoff]
        later = [(t, o) for t, o, _ in perturbed if t > cutoff]
        assert len(after) != len(later) or any(
            t1 != t2 or not np.array_equal(o1, o2)
            for (t1, o1), (t2, o2) in zip(after, later, strict=False)
        )


@pytest.mark.parametrize("feature", ["public", "hos_us", "same_day", "all"])
def test_mask_agrees_with_execution(feature) -> None:
    """Invariant 10, sampled at every step of a short rollout."""
    env = build("seattle", **FEATURES[feature])
    rng = np.random.default_rng(1)
    policy = choose(env, "random", 3)
    observation, info = env.reset(seed=3)
    for _ in range(12):
        if env._episode_ended:
            break
        mask_matches_execution(env, sample=4, rng=rng)
        observation, _, terminated, truncated, info = env.step(policy(env, observation, info))


def test_determinism_and_reset_clear_every_structure() -> None:
    """Invariant 9: same seed and actions, same trajectory; reset forgets."""
    blocks = FEATURES["all"]
    env = build("seattle", **blocks)
    first = []
    rollout(env, 4, choose(env, "random", 4), record=first)
    rollout(env, 11, choose(env, "random", 11))  # pollute
    again = []
    rollout(env, 4, choose(env, "random", 4), record=again)
    assert len(first) == len(again)
    for (t1, o1, m1), (t2, o2, m2) in zip(first, again, strict=True):
        assert t1 == t2 and np.array_equal(o1, o2) and np.array_equal(m1, m2)


def test_X1_eu_break_while_queued_counts_only_the_charge() -> None:
    from EVRoutingEnv.models.core import hours_of_service as h

    regime = h.HosRegime.build("eu_561_2006")
    clock = h.apply(regime, h.DriverClock.fresh(), h.HosActivity.DRIVE, 4.5)
    clock = h.apply(regime, clock, h.HosActivity.QUEUE, 1.0)  # not a break
    assert h.driving_allowance(regime, clock) == 0.0
    clock = h.apply(regime, clock, h.HosActivity.CHARGE, 0.75)  # charging is
    assert h.driving_allowance(regime, clock) == pytest.approx(4.5)


def test_X2_loading_and_depot_charging_are_sequential() -> None:
    with pytest.raises(ValueError, match="parallel_with_loading"):
        build("seattle", requests=SAME_DAY, depot_charging={"enabled": True, "parallel_with_loading": True})


def test_X3_requests_infeasible_only_under_hours_of_service_are_dropped_and_counted() -> None:
    plain = build("seattle", requests=SAME_DAY)
    plain.reset(seed=0)
    hos = build("seattle", requests=SAME_DAY,
                hours_of_service={"enabled": True, "regime": "eu_561_2006", "rest_locations": "stops_only"})
    hos.reset(seed=0)
    dropped = hos.request_engine._dropped("hos", float("inf"))
    assert len(hos.request_engine.registry) + dropped == len(plain.request_engine.registry)


@pytest.mark.parametrize("regime", [HOS_US, {**HOS_EU, "rest_locations": "anywhere"}])
def test_X4_all_components_on_texas_longhaul(regime) -> None:
    blocks = {"public_charging": PUBLIC, "hours_of_service": regime,
              "requests": {**SAME_DAY, "static_context_size": 30}}
    env = build("texas_longhaul", trucks=10, max_time=48.0, **blocks)
    checker = InvariantChecker(env)
    info, _ = rollout(env, 0, choose(env, "greedy", 0), check=checker)
    assert info["termination_reason"] is not None
    assert audit_violations(env) == []
