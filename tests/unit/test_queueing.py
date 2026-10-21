"""M/M/c closed forms against the shipped lookup, matrix exponentials and simulation."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from EVRoutingEnv.models.simulation import queueing as q


ROOT = Path(__file__).resolve().parents[2]
LOOKUPS = sorted((ROOT / "EVRoutingEnv" / "data" / "case_studies").glob("*/waiting_time_lookup.json"))
MEAN_SERVICE_HOURS = {"DCFast": 0.5, "Level2": 2.0}


def test_shipped_lookup_is_exactly_the_erlang_c_wait() -> None:
    assert LOOKUPS
    for path in LOOKUPS:
        table = json.loads(path.read_text())
        for kind, by_capacity in table.items():
            mu = 1.0 / MEAN_SERVICE_HOURS[kind]
            for capacity, row in by_capacity.items():
                for utilization, minutes in row.items():
                    c, u = int(capacity), float(utilization)
                    expected = q.mean_wait(c, u * c * mu, mu) * 60.0
                    assert minutes == pytest.approx(expected, rel=1e-9, abs=1e-12), (path, kind, c, u)


def test_erlang_c_reduces_to_rho_for_one_server() -> None:
    for rho in (0.1, 0.4, 0.9):
        assert q.erlang_c(1, rho, 1.0) == pytest.approx(rho)
        assert q.mean_wait(1, rho, 1.0) == pytest.approx(rho / (1.0 - rho))


def test_A18_erlang_c_above_the_lookup_capacity_is_computed_directly() -> None:
    c, a = 60, 54.0
    terms = [a**k / math.factorial(k) for k in range(c)]
    tail = a**c / math.factorial(c) / (1.0 - a / c)
    assert q.erlang_c(c, a, 1.0) == pytest.approx(tail / (sum(terms) + tail), rel=1e-10)


@pytest.mark.parametrize("utilization", [1.0, 1.2])
def test_unstable_queue_is_rejected(utilization) -> None:
    with pytest.raises(ValueError, match="utilization below one"):
        q.erlang_c(2, utilization * 2.0, 1.0)


def test_truncation_tail_mass_is_below_tolerance() -> None:
    for c, u in ((1, 0.4), (4, 0.9), (20, 0.65)):
        lam = u * c
        level = q.truncation_level(c, lam, 1.0)
        log_weights = q.stationary_log_weights(c, lam, 1.0, level + 400)
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        assert weights[level + 1:].sum() < 1e-9


@pytest.mark.parametrize(("c", "u"), [(1, 0.4), (2, 0.65), (4, 0.9), (10, 0.9)])
def test_transient_wait_matches_matrix_exponential(c, u) -> None:
    mu = 1.0 / 0.4437
    lam = u * c * mu
    for observed in (0, c, c + 7):
        for age in (0.05, 0.5, 2.0, 10.0):
            size = q._round_size(max(q.truncation_level(c, lam, mu), observed + 1) + 1)
            assert q.expected_wait_given_memory(c, lam, mu, observed, age) == pytest.approx(
                q.transient_expected_wait_expm(c, lam, mu, observed, age, size), abs=1e-9
            )


def test_transient_wait_at_age_zero_is_exact() -> None:
    # A newcomer behind n >= c vehicles waits for n - c + 1 completions at rate c mu.
    c, mu = 3, 2.0
    for n in range(9):
        expected = max(0, n - c + 1) / (c * mu)
        assert q.expected_wait_given_memory(c, 4.0, mu, n, 0.0) == pytest.approx(expected)


def test_A14_transient_wait_tends_to_erlang_c_and_unobserved_is_long_run() -> None:
    c, mu = 2, 1.5
    lam = 0.65 * c * mu
    long_run = q.mean_wait(c, lam, mu)
    assert q.expected_wait_given_memory(c, lam, mu, None, None) == long_run
    for observed in (0, 2, 9):
        assert q.expected_wait_given_memory(c, lam, mu, observed, 400.0) == pytest.approx(
            long_run, abs=1e-6
        )


@pytest.mark.statistical
def test_transient_wait_monte_carlo() -> None:
    rng = np.random.default_rng(11)
    c, mu, lam, observed, age = 2, 1.0, 1.4, 5, 1.5
    waits = []
    for _ in range(20000):
        # Simulate the birth-death chain for ``age`` hours, then a newcomer.
        n, t = observed, 0.0
        while True:
            rate = lam + min(n, c) * mu
            t += rng.exponential(1.0 / rate)
            if t > age:
                break
            n += 1 if rng.random() < lam / rate else -1
        waits.append(max(0, n - c + 1) / (c * mu))
    mean, sem = np.mean(waits), np.std(waits) / math.sqrt(len(waits))
    exact = q.expected_wait_given_memory(c, lam, mu, observed, age)
    assert abs(mean - exact) < 2.576 * sem


@pytest.mark.statistical
def test_stationary_count_sampling_chi_square() -> None:
    from scipy import stats

    rng = np.random.default_rng(5)
    c, lam, mu = 3, 2.4, 1.0
    draws = np.array([q.sample_stationary_count(c, lam, mu, tuple(rng.random(2))) for _ in range(10000)])
    pi = q.stationary_distribution(c, lam, mu, 400)
    cut = 12
    observed = np.append(np.bincount(np.minimum(draws, cut), minlength=cut + 1)[:cut], (draws >= cut).sum())
    expected = np.append(pi[:cut], pi[cut:].sum()) * len(draws)
    assert stats.chisquare(observed, expected).pvalue > 0.01


@pytest.mark.statistical
@pytest.mark.parametrize("c", [1, 3])
def test_stationary_wait_law_keskin(c) -> None:
    """P(W = 0) = 1 - C(c, a); W | W > 0 ~ Exp(c mu - lambda); c = 1 gives 1 - rho."""
    from scipy import stats

    rng = np.random.default_rng(c)
    mu, rho = 1.0 / 1.63, 0.8
    lam = rho * c * mu
    waits = np.array([q.sample_stationary_wait(c, lam, mu, tuple(rng.random(2))) for _ in range(40000)])
    zero = 1.0 - q.erlang_c(c, lam, mu)
    if c == 1:
        assert zero == pytest.approx(1.0 - rho)
    share = (waits == 0.0).mean()
    assert abs(share - zero) < 2.576 * math.sqrt(zero * (1 - zero) / len(waits))
    positive = waits[waits > 0.0]
    assert stats.kstest(positive, "expon", args=(0.0, 1.0 / (c * mu - lam))).pvalue > 0.01
