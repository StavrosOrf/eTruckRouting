"""OpenStreetMap charging-station tag parsing used for case studies outside the US (no AFDC there)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "casestudies"))
import chargers as chg  # noqa: E402


@pytest.mark.parametrize("value,expected", [
    ("150 kW", 150.0), ("50000", 50.0), ("22kW", 22.0), ("43;50", 50.0), ("1,5 kW", 1.5), ("0.35 MW", 350.0),
    ("fast", None), (None, None),
])
def test_output_parsing(value, expected):
    assert chg._osm_kw(value) == expected


def test_dc_sockets_count_once_per_dispenser():
    # a Tesla site repeats its stalls under three keys and a Lidl charger has one CCS + one CHAdeMO cable
    tesla = {"socket:tesla_supercharger": "12", "socket:tesla_supercharger_ccs": "12", "socket:type2_combo": "12",
             "socket:type2_combo:output": "150 kW", "capacity": "12"}
    assert chg.osm_dcfast_ports(tesla) == (12, ["CCS", "TESLA"])
    mixed = {"socket:chademo": "1", "socket:type2_combo": "1", "socket:type2": "1", "socket:type2:output": "22 kW",
             "capacity": "2"}
    assert chg.osm_dcfast_ports(mixed)[0] == 1


def test_slow_or_ac_only_stations_are_not_dc_fast():
    assert chg.osm_dcfast_ports({"socket:type2": "4", "socket:type2:output": "22 kW"}) == (0, [])
    assert chg.osm_dcfast_ports({"socket:type2_combo": "2", "socket:type2_combo:output": "22 kW"}) == (0, [])
    assert chg.osm_dcfast_ports({"socket:type2_combo": "2", "socket:type2_combo:output": "50 kW"})[0] == 2


def test_untagged_output_is_kept_and_capacity_caps_ports():
    assert chg.osm_dcfast_ports({"socket:type2_combo": "yes"}) == (1, ["CCS"])
    assert chg.osm_dcfast_ports({"socket:type2_combo": "8", "capacity": "4"})[0] == 4
