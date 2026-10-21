"""Exact single-truck oracle for Bai et al. (2023), ``bai2023_simplified``.

Bai et al. (2023, arXiv:2303.08895) route one electric truck along a fixed
path of charging stations. At each station the truck chooses a charge level
and whether to rest; a stay of at least ``T_r`` (45 min) resets the
consecutive-driving counter (``T_d`` = 4.5 h), shorter charging does not
(their constraints 4 and 7-9), and the trip drives at most 9 h. The objective
is the total trip time.

The dynamic program is exact: the state at station ``k`` is the battery level
on the configured grid and the index of the last station where the counter was
reset, which fixes the consecutive driving exactly. Charging time comes from a
supplied ``charge_hours(soc_from, soc_to)`` function, so the simulator's own
charging curve can be used.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

T_DRIVE = 4.5
T_REST = 0.75
TRIP_DRIVING = 9.0


@dataclass(frozen=True)
class FixedRoute:
    """Segments between consecutive stops: origin, stations..., destination."""

    driving_hours: tuple[float, ...]
    energy_soc: tuple[float, ...]  # energy of each segment as a fraction of capacity
    initial_soc: float = 1.0

    def __post_init__(self) -> None:
        if len(self.driving_hours) != len(self.energy_soc) or not self.driving_hours:
            raise ValueError("a route needs one driving time and energy per segment")


@dataclass(frozen=True)
class Plan:
    total_hours: float
    charge_to: tuple[float | None, ...]  # per station
    rest: tuple[bool, ...]


def solve(route: FixedRoute, levels: tuple[float, ...],
          charge_hours: Callable[[float, float], float]) -> Plan | None:
    """Minimum total time; None if no plan satisfies battery and HOS limits."""
    if sum(route.driving_hours) > TRIP_DRIVING + 1e-9:
        return None
    stations = len(route.driving_hours) - 1
    # value[(soc, last_reset)] = (time, decisions) at the arrival to the next stop.
    states: dict[tuple[float, int], tuple[float, tuple]] = {}
    soc = route.initial_soc - route.energy_soc[0]
    drive = route.driving_hours[0]
    if soc < -1e-9 or drive > T_DRIVE + 1e-9:
        return None
    states[(round(soc, 12), -1)] = (drive, ())
    for k in range(stations):
        nxt: dict[tuple[float, int], tuple[float, tuple]] = {}
        for (soc, last_reset), (time, decisions) in states.items():
            options = [None] + [level for level in levels if level > soc + 1e-12]
            for target in options:
                charging = 0.0 if target is None else charge_hours(soc, target)
                for rest in (False, True):
                    stay = max(charging, T_REST) if rest else charging
                    reset = stay >= T_REST - 1e-12
                    after = soc if target is None else target
                    arrive = after - route.energy_soc[k + 1]
                    if arrive < -1e-9:
                        continue
                    # Segment j runs from stop j-1 to stop j; after a reset at
                    # station r the counter holds segments r+1 .. k+1.
                    origin = k if reset else last_reset
                    since = sum(route.driving_hours[origin + 1: k + 2])
                    if since > T_DRIVE + 1e-9:
                        continue
                    key = (round(arrive, 12), k if reset else last_reset)
                    value = time + stay + route.driving_hours[k + 1]
                    if key not in nxt or value < nxt[key][0] - 1e-12:
                        nxt[key] = (value, decisions + ((target, rest),))
        states = nxt
        if not states:
            return None
    time, decisions = min(states.values(), key=lambda item: item[0])
    return Plan(time, tuple(d[0] for d in decisions), tuple(d[1] for d in decisions))


def brute_force(route: FixedRoute, levels: tuple[float, ...],
                charge_hours: Callable[[float, float], float]) -> float | None:
    """Enumerate every (charge level, rest) choice; the DP's test oracle."""
    if sum(route.driving_hours) > TRIP_DRIVING + 1e-9:
        return None
    stations = len(route.driving_hours) - 1
    best = None
    per_station = [(target, rest) for target in (None, *levels) for rest in (False, True)]
    for plan in itertools.product(per_station, repeat=stations):
        soc = route.initial_soc - route.energy_soc[0]
        since = route.driving_hours[0]
        total = route.driving_hours[0]
        feasible = soc >= -1e-9 and since <= T_DRIVE + 1e-9
        for k, (target, rest) in enumerate(plan):
            if not feasible:
                break
            if target is not None and target <= soc + 1e-12:
                feasible = False
                break
            charging = 0.0 if target is None else charge_hours(soc, target)
            stay = max(charging, T_REST) if rest else charging
            if stay >= T_REST - 1e-12:
                since = 0.0
            soc = (soc if target is None else target) - route.energy_soc[k + 1]
            since += route.driving_hours[k + 1]
            total += stay + route.driving_hours[k + 1]
            feasible = soc >= -1e-9 and since <= T_DRIVE + 1e-9
        if feasible and (best is None or total < best - 1e-12):
            best = total
    return best


def linear_charge_hours(power_kw: float, capacity_kwh: float, efficiency: float = 0.9):
    """Constant-power charging, for tests and quick studies."""
    def hours(soc_from: float, soc_to: float) -> float:
        return max(0.0, soc_to - soc_from) * capacity_kwh / (power_kw * efficiency)
    return hours


def stays_reset(stay_hours: float) -> bool:
    """Bai's reset rule, exposed for the clock cross-check."""
    return stay_hours >= T_REST - 1e-12 and math.isfinite(stay_hours)
