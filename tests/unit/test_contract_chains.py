"""Degree-2 contraction, leader charger merge and grid place picking used by the 10^4-node case studies."""

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "casestudies"))
import build_case_study as bcs  # noqa: E402
import chargers as chg  # noqa: E402
import contract as ctr  # noqa: E402


def _road_like_graph(rng, junctions=12, chain=(1, 6)):
    """Junctions joined by chains of plain road nodes: two-way, one-way and a dead-end spur."""
    eu, ev, n = [], [], junctions
    def add_chain(a, b, two_way):
        nonlocal n
        k = int(rng.integers(*chain))
        mids = list(range(n, n + k)); n += k
        path = [a, *mids, b]
        for x, y in zip(path[:-1], path[1:]):
            eu.append(x); ev.append(y)
            if two_way:
                eu.append(y); ev.append(x)
    for a in range(junctions):  # ring keeps it strongly connected
        add_chain(a, (a + 1) % junctions, two_way=True)
    for _ in range(junctions):
        a, b = rng.choice(junctions, 2, replace=False)
        add_chain(int(a), int(b), two_way=bool(rng.integers(0, 2)))
    return n, np.array(eu), np.array(ev)


def test_contraction_keeps_every_terminal_route_cost():
    rng = np.random.default_rng(3)
    for trial in range(5):
        n, eu, ev = _road_like_graph(rng)
        t, e, l = (rng.uniform(0.1, 3.0, len(eu)) for _ in range(3))
        terminals = rng.choice(n, 9, replace=False)
        protect = np.zeros(n, bool); protect[terminals] = True
        alive, cu, cv, (ct, ce, cl) = ctr.contract_chains(n, eu, ev, protect, [t, e, l], seed=trial, min_removed=0)
        assert alive[terminals].all() and alive.sum() < n

        # parallel edges: keep the cheapest on both graphs, like od_graph does
        def cheapest(u, v, w):
            order = np.lexsort((w, v, u))
            u, v = u[order], v[order]
            first = np.r_[True, (u[1:] != u[:-1]) | (v[1:] != v[:-1])]
            return order[first]
        i0 = cheapest(eu, ev, e)
        i1 = cheapest(cu, cv, ce)
        d0 = dijkstra(csr_matrix((e[i0], (eu[i0], ev[i0])), shape=(n, n)), directed=True, indices=terminals)[:, terminals]
        d1 = dijkstra(csr_matrix((ce[i1], (cu[i1], cv[i1])), shape=(n, n)), directed=True, indices=terminals)[:, terminals]
        np.testing.assert_allclose(d1, d0, rtol=1e-12)


def test_contraction_does_not_fold_terminals_or_junctions():
    # 0 <-> 1 <-> 2 chain with a branch 1 <-> 3: node 1 has three neighbours and must stay
    eu = np.array([0, 1, 1, 2, 1, 3]); ev = np.array([1, 0, 2, 1, 3, 1])
    c = [np.ones(6)]
    protect = np.zeros(4, bool)
    alive, *_ = ctr.contract_chains(4, eu, ev, protect, c, min_removed=0)
    assert alive[1] and alive.sum() >= 2
    protect[0] = True
    alive, *_ = ctr.contract_chains(4, eu, ev, protect, c, min_removed=0)
    assert alive[0]


def test_leader_merge_bounds_the_diameter_on_a_wide_region():
    rng = np.random.default_rng(1)
    lat = rng.uniform(42.0, 57.0, 3000); lon = rng.uniform(-4.0, 15.0, 3000)
    sites = [{"lat": a, "lon": b, "ports": int(p), "name": None, "network": None, "connectors": [], "afdc_id": f"n/{i}"}
             for i, (a, b, p) in enumerate(zip(lat, lon, rng.integers(1, 20, 3000)))]
    merged = chg.merge_sites(sites, 20000.0, "leader")
    assert sum(m["ports"] for m in merged) == sum(s["ports"] for s in sites)
    assert sum(m["n_stations"] for m in merged) == len(sites)
    assert max(m["diameter_m"] for m in merged) <= 20000.0 * (1 + 1e-6)
    assert any(m["n_stations"] > 1 for m in merged)


def test_pick_places_respects_spacing_and_distinct_nodes():
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 1, (400, 2))
    places = [{"name": f"p{i}", "lat": 48.0 + a * 0.5, "lon": 6.0 + b * 0.5, "population": 500 + 10 * i, "place": "town"}
              for i, (a, b) in enumerate(base)]
    # snapping: every place keeps its own "node" (index), 100 m away
    snap = lambda lat, lon: (np.arange(len(lat)), np.full(len(lat), 100.0))
    used = {}
    picked = bcs.pick_places(places, 150, rng, snap, used, 5000.0, 8000.0)
    assert 0 < len(picked) <= 150
    xyz = chg.sphere_xyz([p["lat"] for p in picked], [p["lon"] for p in picked])
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1) + np.eye(len(picked)) * 1e9
    assert d.min() >= 8000.0
    assert len({p["node_idx"] for p in picked}) == len(picked) and set(used) == {p["node_idx"] for p in picked}


def test_complete_merge_keeps_every_pair_within_the_radius_on_a_wide_region():
    rng = np.random.default_rng(2)
    lat = rng.uniform(42.0, 57.0, 1500); lon = rng.uniform(-4.0, 15.0, 1500)
    sites = [{"lat": a, "lon": b, "ports": 3, "name": None, "network": None, "connectors": [], "afdc_id": f"n/{i}"}
             for i, (a, b) in enumerate(zip(lat, lon))]
    merged = chg.merge_sites(sites, 20000.0, "complete")
    assert sum(m["n_stations"] for m in merged) == len(sites)
    for m in merged:
        ids = [int(a.split("/")[1]) for a in m["afdc_ids"]]
        la, lo = np.radians(lat[ids]), np.radians(lon[ids])
        a = np.sin((la[None] - la[:, None]) / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None] * np.sin((lo[None] - lo[:, None]) / 2) ** 2
        assert 12742000.0 * np.arcsin(np.sqrt(a)).max() <= 20000.0 * (1 + 1e-4)  # great-circle, not the merge's own metric
