"""Complete a fixed customer sequence from any truck state, nominally.

Given the active truck's state (node, battery, clock, driver clock) and the
customers it still has to serve, in order, followed by the depot, the search
chooses the waypoint stations between consecutive stops, where and to which
target SoC to charge, and (with hours of service) where to rest. It is a
label-setting search (Desaulniers et al. 2016 style labels, as in fixed-route
vehicle charging problems, Montoya et al. 2017) with:

* stages: one per remaining stop; within a stage the candidate stations are a
  corridor of the closure (small detour) visited in order of decreasing
  closure distance to the stop, so the search graph is acyclic;
* labels ``(time, battery, driver clock)`` with Pareto dominance;
* the action mask's rules: a leg needs ``margin x nominal kWh`` on board and,
  with stops-only rests, its nominal driving must fit the driving allowance;
* charging to one of the environment's discrete target SoCs with the
  environment's own charging curve, plus the expected station wait;
* hours of service through the same pure ``hours_of_service`` functions the
  simulator runs (US: 11 h driving, 14 h window, 30 min break after 8 h
  driving, 10 h daily rest; rests only where the truck stops).

Legs into a customer are priced at the worst-case energy multiplier, so the
truck can always leave a customer whatever the realization; legs into a
station or the depot use the expected draw, since the truck can charge on
arrival (or is done). Stages beyond ``full_stages`` are replaced by a linear
tail (closure cost minus the value of the battery carried), which is exact
whenever the remaining route recharges anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from EVRoutingEnv.models.core.hours_of_service import (
    HosActivity,
    _shift_driving_limit,
    advance_to,
    apply,
    driving_allowance,
    rest_hours,
    window_left,
)


@dataclass(frozen=True)
class ContinuationSettings:
    corridor_hours: float = 1.0
    corridor_size: int = 24
    nearest_stations: int = 6
    full_stages: int = 3
    max_labels: int = 8
    max_labels_hos: int = 12
    energy_expected: float = 1.02
    travel_factor: float = 1.0
    wait_weight: float = 1.0
    # Offer a daily rest at a stop once the window or shift driving left is
    # below this many hours (resting earlier only wastes the window).
    rest_threshold_hours: float = 8.0
    # Hours-of-service tail model: usable on-duty and driving hours per window.
    tail_window_hours: float = 13.5
    tail_driving_hours: float = 10.5


class Label:
    __slots__ = ("time", "battery", "clock", "parent", "action", "node", "charged")

    def __init__(self, time, battery, clock, parent, action, node, charged=False):
        self.time = time
        self.battery = battery
        self.clock = clock
        self.parent = parent
        self.action = action
        self.node = node
        self.charged = charged


@dataclass
class ContinuationResult:
    actions: list[tuple]
    end_time: float
    labels: list[Label]

    @property
    def first(self) -> tuple | None:
        return self.actions[0] if self.actions else None


class ContinuationSearch:
    """Shared, stateless search over one network (one per environment)."""

    def __init__(self, network, curve, *, regime=None, waits=None, settings=None,
                 handling_hours: float = 5.0 / 60.0, levels=None):
        self.network = network
        self.curve = curve
        self.regime = regime
        self.waits = waits
        self.settings = settings or ContinuationSettings()
        self.handling_hours = float(handling_hours)
        self.levels = tuple(levels if levels is not None else curve.levels)
        self.is_station = np.zeros(len(network.nodes), dtype=bool)
        self.is_station[network.chargers] = True

    # ------------------------------------------------------------------
    def corridor(self, source: int, target: int, exclude=frozenset(), extra_nearest=False):
        net, s = self.network, self.settings
        chargers = net.chargers
        cost = net.closure_cost
        detour = cost[source, chargers] + cost[chargers, target] - cost[source, target]
        order = np.argsort(detour)
        chosen = []
        for position in order[: s.corridor_size * 2]:
            if detour[position] > s.corridor_hours + 1e-9 or len(chosen) >= s.corridor_size:
                break
            node = int(chargers[position])
            if node != source and node != target and node not in exclude:
                chosen.append(node)
        if extra_nearest and s.nearest_stations:
            near = np.argsort(cost[source, chargers])
            added = 0
            for position in near:
                node = int(chargers[position])
                if node in exclude or node == source or node in chosen:
                    continue
                chosen.append(node)
                added += 1
                if added >= s.nearest_stations:
                    break
        return chosen

    # ------------------------------------------------------------------
    def _dominates(self, a: Label, b: Label) -> bool:
        if a.time > b.time + 1e-9 or a.battery < b.battery - 1e-6:
            return False
        if self.regime is None:
            return True
        ca, cb = a.clock, b.clock
        if ca.driving_since_break > cb.driving_since_break + 1e-9:
            return False
        if ca.driving_in_shift > cb.driving_in_shift + 1e-9:
            return False
        return window_left(self.regime, ca) >= window_left(self.regime, cb) - 1e-9

    def _pareto(self, labels: list[Label], rank=None) -> list[Label]:
        if len(labels) <= 1:
            return labels
        labels = sorted(labels, key=lambda label: (label.time, -label.battery))
        kept: list[Label] = []
        if self.regime is None:
            best_battery = -math.inf
            for label in labels:
                if label.battery > best_battery + 1e-6:
                    kept.append(label)
                    best_battery = label.battery
        else:
            for label in labels:
                if not any(self._dominates(other, label) for other in kept):
                    kept.append(label)
        limit = self.settings.max_labels if self.regime is None else self.settings.max_labels_hos
        if len(kept) <= limit:
            return kept
        if rank is not None:
            # Best completion estimates first, always keeping the fastest and
            # the fullest label so feasibility is never pruned away.
            ordered = sorted(kept, key=rank)
            extras = [kept[0], max(kept, key=lambda label: label.battery)]
            if self.regime is not None:
                # The freshest driver: the continuous tail cannot tell that a
                # tired driver must stop before the next long hop.
                def fresh(label):
                    return (driving_allowance(self.regime, label.clock), label.battery)

                extras.append(max(kept, key=fresh))
                extras.append(max(kept, key=lambda label: (label.battery, driving_allowance(self.regime, label.clock))))
            chosen = ordered[: max(1, limit - len(extras))]
            for extra in extras:
                if extra not in chosen:
                    chosen.append(extra)
            return chosen
        # Keep the front's spread (fastest ... fullest battery), not just the
        # fastest labels, which are the emptiest ones.
        picks = sorted({round(i * (len(kept) - 1) / (limit - 1)) for i in range(limit)})
        return [kept[i] for i in picks]

    # ------------------------------------------------------------------
    def _wait(self, node: int, start_wait):
        if start_wait is not None:
            return start_wait
        if self.waits is None:
            return 0.0
        return float(self.waits[node]) * self.settings.wait_weight

    def _charge(self, label: Label, node: int, level: float, wait: float) -> Label | None:
        target = level * self.network.capacity
        if target <= label.battery + 1.0:
            return None
        hours = self.curve.hours(label.battery, level)
        clock = label.clock
        if clock is not None:
            if wait > 0.0:
                clock = apply(self.regime, clock, HosActivity.QUEUE, wait)
            overhead = min(self.handling_hours, hours)
            if overhead > 0.0:
                clock = apply(self.regime, clock, HosActivity.HANDLING, overhead)
            clock = apply(self.regime, clock, HosActivity.CHARGE, hours - overhead)
        return Label(label.time + wait + hours, target, clock, label, ("charge", level), node, True)

    def _rests(self, label: Label, node: int) -> list[Label]:
        regime, clock = self.regime, label.clock
        if regime is None or clock is None:
            return []
        out = []
        allowance = driving_allowance(regime, clock)
        threshold = self.settings.rest_threshold_hours
        daily_allowed = regime.daily_rest_hours is not None and (
            clock.shift_start is not None or clock.driving_in_shift > 1e-9
        )
        shift_left = min(
            window_left(regime, clock), _shift_driving_limit(regime, clock) - clock.driving_in_shift
        )
        if daily_allowed and shift_left < threshold:
            hours = rest_hours(regime, clock, "daily")
            rested = apply(regime, clock, HosActivity.REST_DAILY, hours)
            out.append(Label(label.time + hours, label.battery, rested, label, ("rest", "daily"), node, label.charged))
        if (
            clock.driving_since_break > 1e-9
            and regime.driving_before_break - clock.driving_since_break < threshold
            and allowance < shift_left - 1e-9
        ):
            hours = rest_hours(regime, clock, "break")
            rested = apply(regime, clock, HosActivity.REST_BREAK, hours)
            out.append(Label(label.time + hours, label.battery, rested, label, ("rest", "break"), node, label.charged))
        return out

    def _expand(self, labels, node, *, can_charge, forced_charge, start_wait, no_charge, rank=None):
        """Departure labels at ``node``: pass through, charge, rest."""
        out = [] if forced_charge else list(labels)
        charged = []
        if can_charge and node not in no_charge:
            wait = self._wait(node, start_wait)
            for label in labels:
                if label.charged:
                    continue
                for level in self.levels:
                    new = self._charge(label, node, level, wait)
                    if new is not None:
                        charged.append(new)
        out.extend(charged)
        rested = []
        for label in out:
            rested.extend(self._rests(label, node))
        out.extend(rested)
        return self._pareto(out, rank)

    def _arrive(self, label: Label, source: int, target: int, into_customer: bool) -> Label | None:
        net = self.network
        energy = net.energy[source, target]
        hours = net.hours[source, target]
        if not (math.isfinite(energy) and math.isfinite(hours)):
            return None
        if label.battery < net.margin * energy - 1e-9:
            return None
        clock = label.clock
        if clock is not None:
            if hours > driving_allowance(self.regime, clock) + 1e-9:
                return None
        drive = hours * self.settings.travel_factor
        factor = net.margin if into_customer else self.settings.energy_expected
        battery = label.battery - energy * factor
        if into_customer and battery < -1e-9:
            return None
        if clock is not None:
            clock = apply(self.regime, clock, HosActivity.DRIVE, drive)
        return Label(label.time + drive, max(battery, 0.0), clock, label, ("go", target), target, False)

    def _service(self, label: Label, hours: float) -> Label:
        clock = label.clock
        if clock is not None and hours > 0.0:
            clock = apply(self.regime, clock, HosActivity.SERVICE, hours)
        return Label(label.time + hours, label.battery, clock, label.parent, label.action, label.node, False)

    # ------------------------------------------------------------------
    def tail_hours(self, label: Label, node: int, stops) -> float:
        """Linear completion estimate from ``node`` through ``stops``."""
        net = self.network
        if not stops:
            return 0.0
        drive = energy = cost = service = 0.0
        previous = node
        for target, hours, _ in stops:
            cost += net.closure_cost[previous, target]
            drive += net.closure_hours[previous, target]
            energy += net.closure_energy[previous, target]
            service += hours
            previous = target
        charge = max(0.0, energy * self.settings.energy_expected - label.battery) / net.rate
        total = drive + charge + service
        if self.waits is not None and charge > 0:
            stops_needed = math.ceil(charge * net.rate / (0.6 * net.capacity))
            total += stops_needed * float(np.median(self.waits[net.chargers]))
        if self.regime is not None and label.clock is not None:
            clock = label.clock
            window = window_left(self.regime, clock)
            driving = _shift_driving_limit(self.regime, clock) - clock.driving_in_shift
            work_over = total - window
            drive_over = drive - driving
            windows = max(
                math.ceil(max(0.0, work_over) / self.settings.tail_window_hours - 1e-9),
                math.ceil(max(0.0, drive_over) / self.settings.tail_driving_hours - 1e-9),
            )
            total += windows * float(self.regime.daily_rest_hours or 0.0)
        return total

    # ------------------------------------------------------------------
    def solve(
        self,
        start: int,
        battery: float,
        time: float,
        clock,
        stops: list[tuple[int, float, bool]],
        *,
        can_charge_here: bool = True,
        forced_charge: bool = False,
        start_wait: float | None = None,
        unavailable=frozenset(),
        no_charge=frozenset(),
        full_stages: int | None = None,
    ) -> ContinuationResult | None:
        """Best nominal completion; ``stops`` = [(matrix index, service h, is_depot)]."""
        if self.regime is not None and clock is not None:
            clock = advance_to(self.regime, clock, time, HosActivity.IDLE)
        root = Label(float(time), float(battery), clock, None, ("start",), int(start), False)
        frontier = [root]
        node = int(start)
        full = self.settings.full_stages if full_stages is None else int(full_stages)
        stages = min(len(stops), max(1, full))
        for stage in range(stages):
            target, service, _ = stops[stage]
            first_stage = stage == 0
            corridor = self.corridor(node, target, exclude=unavailable, extra_nearest=True)
            dist = self.network.closure_cost[:, target]
            corridor.sort(key=lambda k: -dist[k])
            arrivals: dict[int, list[Label]] = {k: [] for k in corridor}
            arrivals[target] = []
            remaining_stops = stops[stage:]
            if self.regime is not None:
                def rank(label, _stops=remaining_stops):
                    return label.time + self.tail_hours(label, label.node, _stops)
            else:
                rank = None
            # Departures from the stage's start node.
            at_start = self.is_station[node]
            departures = self._expand(
                frontier,
                node,
                can_charge=at_start and (can_charge_here or not first_stage),
                forced_charge=forced_charge and first_stage,
                start_wait=start_wait if first_stage else None,
                no_charge=no_charge,
                rank=rank,
            )
            sequence = [(node, departures, math.inf)] + [(k, None, dist[k]) for k in corridor]
            for position, (here, labels, here_dist) in enumerate(sequence):
                if labels is None:
                    labels = arrivals.get(here, [])
                    if not labels:
                        continue
                    labels = self._expand(
                        self._pareto(labels, rank), here, can_charge=True, forced_charge=False,
                        start_wait=None, no_charge=no_charge, rank=rank,
                    )
                for successor in corridor[position:] if position > 0 else corridor:
                    if successor == here or dist[successor] >= here_dist - 1e-12 and position > 0:
                        continue
                    for label in labels:
                        new = self._arrive(label, here, successor, False)
                        if new is not None:
                            arrivals[successor].append(new)
                into_customer = not stops[stage][2]
                for label in labels:
                    new = self._arrive(label, here, target, into_customer)
                    if new is not None:
                        arrivals[target].append(new)
            reached = arrivals[target]
            if not reached:
                return None
            if service > 0.0:
                reached = [self._service(label, service) for label in reached]
            node = target
            if self.regime is not None:
                next_stops = stops[stage + 1:]
                frontier = self._pareto(
                    reached, lambda label, _stops=next_stops: label.time + self.tail_hours(label, label.node, _stops)
                )
            else:
                frontier = self._pareto(reached)
        remaining = stops[stages:]
        best, best_value = None, math.inf
        for label in frontier:
            value = label.time + self.tail_hours(label, node, remaining)
            if value < best_value:
                best, best_value = label, value
        chain = []
        label = best
        while label is not None:
            chain.append(label)
            label = label.parent
        chain.reverse()
        actions = [label.action for label in chain[1:]]
        return ContinuationResult(actions=actions, end_time=best_value, labels=chain)
