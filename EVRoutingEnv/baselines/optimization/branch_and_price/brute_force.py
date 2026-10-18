"""Independent exact optimum of a small exported instance, for validating the solver.

Shares no code with the Julia solver: every elementary route is enumerated by
plain Pareto labeling over (travel, time, battery) with all discrete charge
levels and exact charging times (no ng relaxation, no charging-time
relaxation, no completion bounds), the cheapest route per customer subset is
kept, and a subset DP picks the best partition into at most ``count`` routes.
Exponential; intended for instances of up to about seven customers.
"""

from __future__ import annotations

import heapq
import math

EPS = 1e-9


def _instance_arrays(instance: dict):
    nodes = instance["nodes"]
    chargers = {c["node"]: c for c in instance["chargers"] if c["available"]}
    keep = [k for k, x in enumerate(nodes) if x["kind"] != "charger" or x["id"] in chargers]
    kinds = [nodes[k]["kind"] for k in keep]
    energy = [[math.inf if instance["energy_kwh"][a][b] is None else instance["energy_kwh"][a][b]
               for b in keep] for a in keep]
    hours = [[math.inf if instance["time_h"][a][b] is None else instance["time_h"][a][b]
              for b in keep] for a in keep]
    curves = instance["charge_curves"]
    curve_of = {}
    for i, k in enumerate(keep):
        if kinds[i] == "charger":
            curve_of[i] = curves["cumulative_hours"][chargers[nodes[k]["id"]]["curve"]]
    return [nodes[k] for k in keep], kinds, energy, hours, curve_of, curves["soc_step"]


def _cumulative(table, step, soc):
    x = min(max(soc, 0.0), 1.0) / step
    k = min(int(math.floor(x)), len(table) - 2)
    return table[k] + (x - k) * (table[k + 1] - table[k])


def best_route_per_subset(instance: dict) -> dict[int, float]:
    nodes, kinds, E, H, curve_of, step = _instance_arrays(instance)
    customers = [i for i, k in enumerate(kinds) if k == "customer"]
    bit = {c: 1 << p for p, c in enumerate(customers)}
    depot = kinds.index("depot")
    B = instance["vehicle"]["battery_kwh"]
    start = instance["vehicle"]["start_soc"][0] * B
    margin = instance["energy_margin"]
    horizon = instance["horizon_h"]
    payload = instance["vehicle"]["payload"]
    levels = sorted(instance["charge_levels"])

    def usable(a, b):
        return a != b and math.isfinite(E[a][b]) and math.isfinite(H[a][b]) and E[a][b] >= 0

    best: dict[int, float] = {}
    fronts: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    heap = [(0.0, 0, depot, 0.0, start, 0.0)]  # time, mask, node, travel, battery, load
    while heap:
        t, mask, v, travel, b, load = heapq.heappop(heap)
        if (travel, t, b) not in fronts.get((mask, v), [(travel, t, b)]) and v != depot:
            continue
        options = [(b, t)]
        if kinds[v] == "charger":
            table = curve_of[v]
            for L in levels:
                if L * B > b + EPS:
                    options.append((L * B, t + _cumulative(table, step, L) - _cumulative(table, step, b / B)))
        for w in range(len(kinds)):
            if not usable(v, w) or (w == depot and v == depot):
                continue
            if kinds[w] == "customer" and mask & bit[w]:
                continue
            need = margin * E[v][w]
            for bd, td in options:
                if need > bd + EPS:
                    continue
                arrival, nb = td + H[v][w], bd - E[v][w]
                if w == depot:
                    if mask and arrival <= horizon + EPS:
                        best[mask] = min(best.get(mask, math.inf), travel + H[v][w])
                    break
                nmask, nload, nt = mask, load, arrival
                if kinds[w] == "customer":
                    node = nodes[w]
                    close = math.inf if node["tw"][1] is None else node["tw"][1]
                    if arrival > close + EPS:
                        break
                    nload = load + node["demand"]
                    if nload > payload + EPS:
                        break
                    nmask = mask | bit[w]
                    nt = max(arrival, node["tw"][0]) + node["service_h"]
                if nt > horizon + EPS:
                    break
                state = (travel + H[v][w], nt, nb)
                front = fronts.setdefault((nmask, w), [])
                if any(a[0] <= state[0] + 1e-12 and a[1] <= state[1] + 1e-12 and a[2] >= state[2] - 1e-12
                       for a in front):
                    continue
                front[:] = [a for a in front if not (state[0] <= a[0] + 1e-12 and state[1] <= a[1] + 1e-12
                                                     and state[2] >= a[2] - 1e-12)]
                front.append(state)
                heapq.heappush(heap, (nt, nmask, w, state[0], nb, nload))
    return best


def exact_optimum(instance: dict) -> float:
    """Least total travel serving every customer exactly once with at most ``count`` routes."""
    best = best_route_per_subset(instance)
    n = sum(1 for x in instance["nodes"] if x["kind"] == "customer")
    full = (1 << n) - 1
    trucks = instance["vehicle"]["count"]
    layer = {0: 0.0}
    answer = math.inf
    for _ in range(trucks):
        nxt = {}
        for mask, cost in layer.items():
            rest = full & ~mask
            if rest == 0:
                continue
            low = rest & -rest
            sub = rest
            while sub:
                if sub & low and sub in best:
                    value = cost + best[sub]
                    if value < nxt.get(mask | sub, math.inf):
                        nxt[mask | sub] = value
                sub = (sub - 1) & rest
        layer = nxt
        answer = min(answer, layer.get(full, math.inf))
    return answer
