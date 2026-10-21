"""M/M/c formulas for public charging stations shared with outside traffic.

Kullman, Goodson & Mendoza (2021, Transportation Science 55(3), sec. 2.1)
model each public station as a pooled FCFS M/M/c/inf queue with known arrival
and service rates. This module holds the closed forms used by the simulator,
the information model and the test oracles:

* Erlang C waiting probability and long-run mean wait W_q;
* the stationary number-in-system distribution, sampled exactly;
* the stationary waiting-time law used by Keskin, Akhavan-Tabatabaei & Catay
  (2019, WSC): W = 0 with probability 1 - C(c, a), else Exp(c mu - lambda);
* the transient expected wait of a newcomer given an observed number in system
  ``age`` hours ago (Kullman sec. 4.3.2, Fig. 4), which tends to W_q.

Rates are per hour; ``offered_load`` is a = lambda / mu and utilization is
rho = a / c, which must be below one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


TAIL_MASS = 1e-9


def _validate(servers: int, arrival_rate: float, service_rate: float) -> None:
    if isinstance(servers, bool) or not isinstance(servers, (int, np.integer)) or servers < 1:
        raise ValueError("servers must be a positive integer")
    if not math.isfinite(arrival_rate) or arrival_rate < 0.0:
        raise ValueError("arrival_rate must be finite and non-negative")
    if not math.isfinite(service_rate) or service_rate <= 0.0:
        raise ValueError("service_rate must be finite and positive")
    if arrival_rate >= servers * service_rate:
        raise ValueError("M/M/c requires utilization below one")


def erlang_c(servers: int, arrival_rate: float, service_rate: float) -> float:
    """Probability that an arriving customer waits, C(c, a)."""
    _validate(servers, arrival_rate, service_rate)
    offered = arrival_rate / service_rate
    rho = offered / servers
    # Erlang B by its stable recursion, then the standard B -> C conversion.
    blocking = 1.0
    for k in range(1, int(servers) + 1):
        blocking = offered * blocking / (k + offered * blocking)
    return blocking / (1.0 - rho * (1.0 - blocking))


def mean_wait(servers: int, arrival_rate: float, service_rate: float) -> float:
    """Long-run mean wait in queue W_q, in hours."""
    if arrival_rate == 0.0:
        _validate(servers, arrival_rate, service_rate)
        return 0.0
    return erlang_c(servers, arrival_rate, service_rate) / (
        servers * service_rate - arrival_rate
    )


def mean_queue_length(servers: int, arrival_rate: float, service_rate: float) -> float:
    """Long-run mean number waiting L_q = lambda W_q (Little's law)."""
    return arrival_rate * mean_wait(servers, arrival_rate, service_rate)


def stationary_log_weights(
    servers: int, arrival_rate: float, service_rate: float, size: int
) -> np.ndarray:
    """Unnormalized log pi_n for n = 0..size-1."""
    _validate(servers, arrival_rate, service_rate)
    offered = arrival_rate / service_rate
    n = np.arange(size, dtype=float)
    if offered == 0.0:
        weights = np.full(size, -np.inf)
        weights[0] = 0.0
        return weights
    log_offered = math.log(offered)
    below = n * log_offered - np.array([math.lgamma(k + 1.0) for k in n])
    c = int(servers)
    at_c = c * log_offered - math.lgamma(c + 1.0)
    above = at_c + (n - c) * math.log(offered / c)
    return np.where(n < c, below, above)


def truncation_level(servers: int, arrival_rate: float, service_rate: float) -> int:
    """Smallest N whose stationary tail mass beyond N is below ``TAIL_MASS``."""
    _validate(servers, arrival_rate, service_rate)
    c = int(servers)
    rho = arrival_rate / (c * service_rate)
    if rho == 0.0:
        return c
    # P(n > N) = C * rho^(N - c + 1) for N >= c - 1.
    waiting = erlang_c(c, arrival_rate, service_rate)
    if waiting <= TAIL_MASS:
        return c
    extra = math.ceil(math.log(TAIL_MASS / waiting) / math.log(rho))
    return c - 1 + max(1, extra)


def stationary_distribution(
    servers: int, arrival_rate: float, service_rate: float, size: int | None = None
) -> np.ndarray:
    """Stationary pi_n for n = 0..size-1, normalized over the truncated support."""
    if size is None:
        size = truncation_level(servers, arrival_rate, service_rate) + 1
    log_weights = stationary_log_weights(servers, arrival_rate, service_rate, size)
    weights = np.exp(log_weights - log_weights.max())
    return weights / weights.sum()


def sample_stationary_count(
    servers: int, arrival_rate: float, service_rate: float, uniforms: tuple[float, float]
) -> int:
    """Exact draw of the stationary number in system from two U(0, 1) values.

    Below c the support is finite; at or above c the excess n - c is geometric
    with success probability 1 - rho, so no truncation is involved.
    """
    _validate(servers, arrival_rate, service_rate)
    first, second = (float(value) for value in uniforms)
    c = int(servers)
    waiting = erlang_c(c, arrival_rate, service_rate) if arrival_rate > 0 else 0.0
    if first >= waiting:
        below = stationary_distribution(servers, arrival_rate, service_rate, c)
        cumulative = np.cumsum(below)
        index = int(np.searchsorted(cumulative, second * cumulative[-1], side="right"))
        return min(index, c - 1)
    rho = arrival_rate / (c * service_rate)
    # Inverse CDF of the number of failures before the first success.
    excess = math.floor(math.log1p(-second) / math.log(rho)) if rho > 0 else 0
    return c + max(0, excess)


def sample_stationary_wait(
    servers: int, arrival_rate: float, service_rate: float, uniforms: tuple[float, float]
) -> float:
    """Stationary FCFS wait: 0 w.p. 1 - C, else Exp(c mu - lambda) (Keskin 2019)."""
    _validate(servers, arrival_rate, service_rate)
    first, second = (float(value) for value in uniforms)
    if arrival_rate == 0.0:
        return 0.0
    waiting = erlang_c(servers, arrival_rate, service_rate)
    if first >= waiting:
        return 0.0
    return -math.log1p(-second) / (servers * service_rate - arrival_rate)


def newcomer_wait(servers: int, service_rate: float, in_system: int) -> float:
    """Mean FCFS wait of a newcomer behind ``in_system`` vehicles.

    With n >= c in system the newcomer waits for n - c + 1 completions, each at
    rate c mu once all ports are busy.
    """
    return max(0, int(in_system) - int(servers) + 1) / (int(servers) * service_rate)


@dataclass(frozen=True)
class _Spectrum:
    """Symmetrized birth-death generator, diagonalized once per station class."""

    size: int
    sqrt_pi: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    weighted_wait: np.ndarray  # a_k = sum_j V_jk sqrt(pi_j) w_j


def birth_death_generator(
    servers: int, arrival_rate: float, service_rate: float, size: int
) -> np.ndarray:
    """Generator of the M/M/c number-in-system chain truncated at size - 1."""
    generator = np.zeros((size, size))
    for n in range(size):
        if n + 1 < size:
            generator[n, n + 1] = arrival_rate
        if n > 0:
            generator[n, n - 1] = min(n, servers) * service_rate
        generator[n, n] = -generator[n].sum()
    return generator


@lru_cache(maxsize=256)
def _spectrum(servers: int, arrival_rate: float, service_rate: float, size: int) -> _Spectrum:
    from scipy.linalg import eigh_tridiagonal

    pi = stationary_distribution(servers, arrival_rate, service_rate, size)
    sqrt_pi = np.sqrt(pi)
    # The truncated chain is reversible, so Pi^1/2 Q Pi^-1/2 is the symmetric
    # tridiagonal matrix with off-diagonals sqrt(lambda * min(n + 1, c) mu).
    n = np.arange(size)
    births = np.where(n + 1 < size, arrival_rate, 0.0)
    deaths = np.minimum(n, servers) * service_rate
    diagonal = -(births + deaths)
    off = np.sqrt(arrival_rate * np.minimum(n[1:], servers) * service_rate)
    eigenvalues, eigenvectors = eigh_tridiagonal(diagonal, off)
    waits = np.array([newcomer_wait(servers, service_rate, k) for k in range(size)])
    weighted = eigenvectors.T @ (sqrt_pi * waits)
    return _Spectrum(size, sqrt_pi, eigenvalues, eigenvectors, weighted)


def expected_wait_given_memory(
    servers: int,
    arrival_rate: float,
    service_rate: float,
    observed_in_system: int | None,
    age_hours: float | None,
) -> float:
    """Transient expected wait of a newcomer now, given a past observation.

    ``observed_in_system`` vehicles were seen ``age_hours`` ago. With no
    observation (Kullman's memory (c, empty, empty)) this is the long-run W_q.
    The count is propagated through the birth-death chain truncated where the
    stationary tail mass is below ``TAIL_MASS`` (and above the observed count).
    """
    _validate(servers, arrival_rate, service_rate)
    if observed_in_system is None or age_hours is None:
        return mean_wait(servers, arrival_rate, service_rate)
    observed = int(observed_in_system)
    age = float(age_hours)
    if observed < 0 or not math.isfinite(age) or age < 0.0:
        raise ValueError("observation must be a non-negative count and age")
    if age == 0.0 or (arrival_rate == 0.0 and observed < servers):
        return newcomer_wait(servers, service_rate, observed) if age == 0.0 else 0.0
    size = max(truncation_level(servers, arrival_rate, service_rate), observed + 1) + 1
    size = _round_size(size)
    spectrum = _spectrum(int(servers), float(arrival_rate), float(service_rate), size)
    anchor = spectrum.sqrt_pi[observed]
    if anchor > 1e-150:
        decay = np.exp(spectrum.eigenvalues * age)
        value = float(
            spectrum.eigenvectors[observed] @ (decay * spectrum.weighted_wait) / anchor
        )
        return max(0.0, value)
    return transient_expected_wait_expm(
        servers, arrival_rate, service_rate, observed, age, size
    )


def transient_expected_wait_expm(
    servers: int,
    arrival_rate: float,
    service_rate: float,
    observed_in_system: int,
    age_hours: float,
    size: int | None = None,
) -> float:
    """Reference computation with a dense matrix exponential."""
    from scipy.linalg import expm

    if size is None:
        size = max(
            truncation_level(servers, arrival_rate, service_rate), observed_in_system + 1
        ) + 1
    generator = birth_death_generator(servers, arrival_rate, service_rate, size)
    row = expm(generator * float(age_hours))[int(observed_in_system)]
    waits = np.array([newcomer_wait(servers, service_rate, n) for n in range(size)])
    return float(row @ waits)


def _round_size(size: int) -> int:
    # Coarse sizes let observations with different counts share a spectrum.
    return int(32 * math.ceil(size / 32))
