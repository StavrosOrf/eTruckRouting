"""Write the held-out problems of the generalist campaign: never trained on, evaluation only.

Run from the repository root:
    PYTHONPATH=. python scripts/campaigns/write_generalist_heldout_configs.py

Three kinds, each changing one thing against a suite problem:

* unseen networks: the suite's long-haul settings on Germany + Netherlands +
  Belgium;
* unseen sizes: the four size probes of ``generalist_graphppo.PROBES``;
* unseen combinations: same-day freight on the Texas network and drayage on the
  Pacific Northwest network, each known to the policy from long haul only.
"""

from __future__ import annotations

import copy

import yaml

from EVRoutingEnv.utils.utils import load_config
from scripts.campaigns.write_dynamic_benchmark_configs import CONFIGS, PUGET_SOUND, base_config
from scripts.campaigns.write_generalist_suite_configs import OBSERVATION, US_HOURS, public, suite

HELDOUT = CONFIGS / "generalist_heldout"
MIN_DEPOT_ENERGY = 225.0  # the Texas long-haul rule: every route must charge


def _unseen_network(source: str, base: dict) -> dict:
    """``base``'s settings with only the network of ``source``."""
    config = copy.deepcopy(base)
    other = load_config(str(CONFIGS / source))
    config["network"] = copy.deepcopy(other["network"])
    config["environment"]["min_depot_energy_kwh"] = MIN_DEPOT_ENERGY
    return config


def _resized(base: dict, trucks: int, stops: int | None) -> dict:
    config = copy.deepcopy(base)
    config["environment"]["num_trucks"] = trucks
    if stops is not None:
        config["environment"]["num_stops"] = stops
    return config


def heldout() -> dict[str, dict]:
    parts = suite()
    tx, pnw, sameday = parts["tx_longhaul"], parts["pnw_longhaul_hos"], parts["ca_sameday"]
    de_hos = _unseen_network("config_joint_de_nl_be.yaml", pnw)
    configs = {
        "de_nl_be_longhaul": _unseen_network("config_joint_de_nl_be.yaml", tx),
        "de_nl_be_longhaul_hos": de_hos,
        "tx_longhaul_t5c15": _resized(tx, 5, 15),
        "tx_longhaul_t20c60": _resized(tx, 20, 60),
        "ca_sameday_t5": _resized(sameday, 5, None),
        "ca_sameday_t20": _resized(sameday, 20, None),
    }
    tx_sameday = base_config("config_joint_texas_longhaul.yaml", "same_day", None, 200)
    pnw_drayage = base_config("config_joint_pacific_northwest_longhaul.yaml", "drayage", PUGET_SOUND, 36)
    pnw_drayage["hours_of_service"] = copy.deepcopy(US_HOURS)
    pnw_drayage["public_charging"] = public(0.65)
    for config in (tx_sameday, pnw_drayage):
        config["environment"].pop("min_depot_energy_kwh", None)
        config["observation"] = dict(OBSERVATION)
    configs.update(tx_sameday=tx_sameday, pnw_drayage_combined=pnw_drayage)
    return configs


def main() -> None:
    HELDOUT.mkdir(exist_ok=True)
    for name, config in heldout().items():
        (HELDOUT / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        print(name)


if __name__ == "__main__":
    main()
