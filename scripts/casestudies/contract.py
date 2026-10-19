"""Exact contraction of degree-2 road nodes (series reduction) for all-pairs routing.

About four out of five OpenStreetMap road nodes only continue a road: one way in, one way out (a one-way
street) or one neighbour on each side (a two-way street). Folding such a node ``x`` into its neighbours,

    a -> x -> b   becomes the single edge   a -> b   with the summed costs,

leaves the cost of every route between the kept nodes unchanged, because all of ``x``'s traffic must pass
through it, while the graph that Dijkstra searches shrinks by about 7x. Terminals (delivery and charger nodes)
are never folded away, so every terminal-to-terminal route keeps exactly its time, energy and length.
"""
from __future__ import annotations

import numpy as np


def contract_chains(n: int, eu: np.ndarray, ev: np.ndarray, protect: np.ndarray, costs: list[np.ndarray],
                    seed: int = 0, min_removed: int = 1000, max_rounds: int = 80, log=lambda *a: None):
    """Fold non-protected nodes with one in/out edge, or two-way chains, into their neighbours.

    ``costs`` are per-edge additive quantities (time, energy, length). Each round removes an independent set
    of foldable nodes (random priorities, so a chain of k nodes disappears in about log k rounds).

    Returns ``(alive, eu, ev, costs)`` with the original node numbering; ``alive`` marks the kept nodes.
    """
    rng = np.random.default_rng(seed)
    eu = eu.astype(np.int64); ev = ev.astype(np.int64)
    costs = [np.asarray(c, np.float64) for c in costs]
    loops = eu == ev
    if loops.any():  # a self-loop never lies on a shortest route
        eu, ev, costs = eu[~loops], ev[~loops], [c[~loops] for c in costs]
    alive = np.ones(n, bool)
    pri = rng.permutation(n)
    for rnd in range(max_rounds):
        indeg = np.bincount(ev, minlength=n); outdeg = np.bincount(eu, minlength=n)
        base = alive & ~protect
        c1 = base & (indeg == 1) & (outdeg == 1)
        c2 = base & (indeg == 2) & (outdeg == 2)
        valid = np.zeros(n, bool)

        # one in, one out: a -> x -> b with a != b
        in1 = np.nonzero(c1[ev])[0]; out1 = np.nonzero(c1[eu])[0]
        in1 = in1[np.argsort(ev[in1], kind="stable")]; out1 = out1[np.argsort(eu[out1], kind="stable")]
        ok1 = eu[in1] != ev[out1]            # both lists are ordered by node, one arc each
        valid[ev[in1][ok1]] = True

        # two in, two out, from and to the same two distinct neighbours: a <-> x <-> b
        in2 = np.nonzero(c2[ev])[0]; out2 = np.nonzero(c2[eu])[0]
        in2 = in2[np.argsort(ev[in2], kind="stable")]; out2 = out2[np.argsort(eu[out2], kind="stable")]
        i0, i1, o0, o1 = in2[0::2], in2[1::2], out2[0::2], out2[1::2]
        a1, a2, b1, b2 = eu[i0], eu[i1], ev[o0], ev[o1]
        ok2 = (a1 != a2) & (((a1 == b1) & (a2 == b2)) | ((a1 == b2) & (a2 == b1)))
        valid[ev[i0][ok2]] = True

        # independent set: a node goes only if no foldable neighbour has a smaller priority
        both = valid[eu] & valid[ev]
        blocked = np.zeros(n, bool)
        blocked[np.where(pri[eu[both]] > pri[ev[both]], eu[both], ev[both])] = True
        sel = valid & ~blocked
        n_sel = int(sel.sum())
        if n_sel == 0:
            break

        new_u, new_v, new_c = [], [], [[] for _ in costs]
        s1 = sel[ev[in1]] & ok1
        a, b = in1[s1], out1[s1]
        new_u.append(eu[a]); new_v.append(ev[b])
        for k, c in enumerate(costs):
            new_c[k].append(c[a] + c[b])
        s2 = sel[ev[i0]] & ok2
        i0s, i1s, o0s, o1s = i0[s2], i1[s2], o0[s2], o1[s2]
        a1s, a2s, b1s = eu[i0s], eu[i1s], ev[o0s]
        o_to_a2 = np.where(b1s == a2s, o0s, o1s)   # leave x towards the other neighbour
        o_to_a1 = np.where(b1s == a2s, o1s, o0s)
        for i_arc, o_arc in ((i0s, o_to_a2), (i1s, o_to_a1)):
            new_u.append(eu[i_arc]); new_v.append(ev[o_arc])
            for k, c in enumerate(costs):
                new_c[k].append(c[i_arc] + c[o_arc])

        keep = ~(sel[eu] | sel[ev])
        eu = np.concatenate([eu[keep], *new_u]); ev = np.concatenate([ev[keep], *new_v])
        costs = [np.concatenate([c[keep], *new_c[k]]) for k, c in enumerate(costs)]
        alive &= ~sel
        log(f"  contraction round {rnd}: folded {n_sel} nodes, {int(alive.sum())} nodes / {len(eu)} edges left")
        if n_sel < min_removed:
            break
        loops = eu == ev   # a dead-end loop folded onto itself
        if loops.any():
            eu, ev, costs = eu[~loops], ev[~loops], [c[~loops] for c in costs]
    return alive, eu, ev, costs
