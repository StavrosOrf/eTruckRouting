"""max_speed_kph in the vehicle model caps posted speeds (speed-limited trucks in Europe)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "casestudies"))
import road_graph as rg  # noqa: E402


def _net(kph):
    z = np.zeros(2)
    return rg.RoadNetwork(
        osm_id=np.array([1, 2]), lat=z, lon=z, ctrl=np.zeros(2, np.int8),
        eu=np.array([0], np.int32), ev=np.array([1], np.int32), length=np.array([10_000.0]),
        kph=np.array([kph]), hwclass=np.array([rg.CLASS_ID["motorway"]], np.int8), span=np.array([False]),
        elev=z.copy(),
    )


def test_cap_slows_fast_roads_and_leaves_slow_ones():
    t_free, e_free, _ = rg.edge_costs(_net(130.0))
    t_cap, e_cap, _ = rg.edge_costs(_net(130.0), {"max_speed_kph": 90.0})
    assert t_cap[0] > t_free[0]
    assert e_cap[0] < e_free[0]                  # less drag at the lower speed outweighs the longer auxiliary load
    t_slow, _, _ = rg.edge_costs(_net(60.0), {"max_speed_kph": 90.0})
    assert np.isclose(t_slow[0], rg.edge_costs(_net(60.0))[0][0])
    assert np.isclose(t_cap[0], rg.edge_costs(_net(90.0))[0][0])
