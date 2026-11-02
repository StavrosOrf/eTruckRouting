"""Write the broad training set of the generalist campaign (suite_v6): many problems, nothing held out touched.

Run from the repository root:
    PYTHONPATH=. python scripts/campaigns/write_generalist_broad_configs.py

The suite's five problems vary nothing but the mechanism: one network per
mechanism, ten trucks everywhere. The held-out check (claim 2) failed exactly
along the axes never varied, so this set varies them, while keeping every
held-out problem out of reach:

* networks: Texas and the Pacific Northwest for long haul; California,
  Chicago, Seattle and the Pacific Northwest for same-day; California and
  Texas (Houston) for drayage. Germany + Netherlands + Belgium is never used,
  same-day never runs on Texas, drayage never on the Pacific Northwest.
* fleets: 5 to 100 trucks where the problem allows (user decision 2026-11-01):
  static stops at three per truck up to the network's certified customer cap
  (Texas 120, Pacific Northwest 160), so static fleets stop at 40 (Texas) and
  60 (Pacific Northwest) trucks; dynamic request rates scale with the fleet
  (0.4 requests per epoch per truck) up to 100 trucks. The held-out sizes 5 and 20 are
  interpolation tests; the held-out set gains larger sizes.
* mechanisms: plain long haul, congested public charging with outages, hours
  of service (rests at stops only, as in the suite), same-day freight,
  drayage with hours of service and public charging, as in the suite.

The five suite problems are included unchanged, so suite_v6 is a superset: 40 problems, one env each at 40 envs.
"""

from __future__ import annotations

import copy

import yaml

from scripts.campaigns.write_dynamic_benchmark_configs import CONFIGS, HOUSTON, base_config
from scripts.campaigns.write_generalist_suite_configs import (
    LONG_BEACH_LOS_ANGELES, OBSERVATION, US_HOURS, _longhaul, public, suite,
)

BROAD = CONFIGS / "generalist_broad"
LONGHAUL = {"tx": "config_joint_texas_longhaul.yaml", "pnw": "config_joint_pacific_northwest_longhaul.yaml"}
SAMEDAY = {
    "ca": "config_joint_california_10t30c.yaml", "chi": "config_joint_chicago.yaml",
    "sea": "config_joint_seattle.yaml", "pnw": "config_joint_pacific_northwest_longhaul.yaml",
}
DRAYAGE = {"ca": ("config_joint_california_10t30c.yaml", LONG_BEACH_LOS_ANGELES), "tx": ("config_joint_texas_longhaul.yaml", HOUSTON)}
STOPS_PER_TRUCK = 3
# The instance generator certifies every static instance solvable; above these
# customer counts it finds no energy-safe set (Texas 120 passed, 160 did not;
# the Pacific Northwest passed 160). Larger fleets share the capped stops.
MAX_STOPS = {"tx": 120, "pnw": 160}
REQUESTS_PER_EPOCH_PER_TRUCK = 0.4  # the suite's 4 per epoch at 10 trucks
REQUEST_ENVELOPE = 25  # max_requests per request-per-epoch, well above the 36 h stream

# Fleets per problem family and network (the suite's 10-truck problems are added unchanged).
# Static fleets stop where every truck still averages at least two customers
# under the cap (Texas 40 trucks, Pacific Northwest 60); larger static fleets
# would mostly idle at the depot. Dynamic fleets go to 100, their request rate
# scales with them.
FLEETS = {
    ("plain", "tx"): (5, 8, 16, 25, 40), ("plain", "pnw"): (5, 13, 25, 40, 60),
    ("congested", "tx"): (8, 25, 40), ("congested", "pnw"): (13, 40),
    ("hos", "tx"): (8, 25, 40), ("hos", "pnw"): (13, 40),
    ("sameday", "ca"): (5, 16, 40, 100), ("sameday", "chi"): (8, 25, 50), ("sameday", "sea"): (13, 60), ("sameday", "pnw"): (10,),
    ("drayage", "ca"): (16, 60), ("drayage", "tx"): (10, 25, 100),
}


def _fleet(config: dict, trucks: int, static: bool, network: str | None = None) -> dict:
    config = copy.deepcopy(config)
    config["environment"]["num_trucks"] = trucks
    if static:
        config["environment"]["num_stops"] = min(STOPS_PER_TRUCK * trucks, MAX_STOPS[network])
    else:
        per_epoch = max(1, round(REQUESTS_PER_EPOCH_PER_TRUCK * trucks))
        config["requests"].update(max_requests_per_epoch=per_epoch, max_requests=REQUEST_ENVELOPE * per_epoch)
    return config


def _static(network: str, mechanism: str) -> dict:
    config = _longhaul(LONGHAUL[network])
    if mechanism == "congested":
        config["public_charging"] = public(0.65, outages=True)
    elif mechanism == "hos":
        config["hours_of_service"] = {**US_HOURS, "rest_locations": "stops_only"}
    return config


def _sameday(network: str) -> dict:
    config = base_config(SAMEDAY[network], "same_day", None, 200)
    config["environment"].pop("min_depot_energy_kwh", None)
    return config


def _drayage(network: str) -> dict:
    source, terminals = DRAYAGE[network]
    config = base_config(source, "drayage", terminals, 36)
    config["environment"].pop("min_depot_energy_kwh", None)
    config["hours_of_service"] = copy.deepcopy(US_HOURS)
    config["public_charging"] = public(0.65)
    return config


def broad() -> dict[str, dict]:
    configs = dict(suite())
    for (family, network), fleets in FLEETS.items():
        for trucks in fleets:
            if family == "sameday":
                configs[f"{network}_sameday_t{trucks}"] = _fleet(_sameday(network), trucks, False)
            elif family == "drayage":
                configs[f"{network}_drayage_combined_t{trucks}"] = _fleet(_drayage(network), trucks, False)
            else:
                label = "longhaul" if family == "plain" else f"longhaul_{family}"
                configs[f"{network}_{label}_t{trucks}"] = _fleet(_static(network, family), trucks, True, network)
    for name, config in configs.items():
        config["observation"] = dict(OBSERVATION)
    return configs


def main() -> None:
    BROAD.mkdir(exist_ok=True)
    header = "# Generated by scripts/campaigns/write_generalist_broad_configs.py.\n# Edit the script, not this file.\n\n"
    for name, config in broad().items():
        (BROAD / f"{name}.yaml").write_text(header + yaml.safe_dump(config, sort_keys=False))
        print(name)


if __name__ == "__main__":
    main()
