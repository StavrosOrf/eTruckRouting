"""Path sums along the Dijkstra predecessor tree used by the case-study all-pairs routing."""

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "casestudies"))
import build_case_study as bcs  # noqa: E402


def _edges(edges, n):
    u = np.array([e[0] for e in edges]); v = np.array([e[1] for e in edges])
    order = np.lexsort((v, u))
    return u[order], v[order], order


def test_sums_follow_the_tree_and_unreachable_is_nan():
    # 0 -> 1 -> 2 and 0 -> 3 (shared trunk 0 -> 1), node 4 cannot be reached
    n = 5
    edges = [(0, 1, 1.0, 10.0, 100.0), (1, 2, 2.0, 20.0, 200.0), (0, 3, 5.0, 50.0, 500.0), (4, 0, 1.0, 1.0, 1.0)]
    u, v, order = _edges([e[:2] for e in edges], n)
    t, e, l = (np.array([edges[i][k] for i in order]) for k in (2, 3, 4))
    csr = csr_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
    _, P = dijkstra(csr, directed=True, indices=0, return_predecessors=True)
    bcs._OD.clear()
    at, ae, al = bcs._tree_sums(P, 0, np.array([0, 1, 2, 3, 4]), u.astype(np.int64) * n + v, t, e, l, n)
    np.testing.assert_allclose(at[:4], [0, 1, 3, 5])
    np.testing.assert_allclose(ae[:4], [0, 10, 30, 50])
    np.testing.assert_allclose(al[:4], [0, 100, 300, 500])
    assert np.isnan([at[4], ae[4], al[4]]).all()
    bcs._OD.clear()
