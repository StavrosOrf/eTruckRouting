"""The generalist evaluation suite matches its generator and covers every component."""

import yaml

from EVRoutingEnv.models.environment.extension_config import parse_extension_config
from scripts.campaigns.write_generalist_suite_configs import SUITE, suite


def test_suite_configs_are_current_and_valid() -> None:
    expected = suite()
    assert sorted(p.stem for p in SUITE.glob("*.yaml")) == sorted(expected)
    networks, components = [], set()
    for name, config in expected.items():
        assert yaml.safe_load((SUITE / f"{name}.yaml").read_text()) == config, name
        extensions = parse_extension_config(
            config, max_time=config["environment"]["max_time"], sequential=False, preassigned=False,
        )
        assert extensions.in_flight_information == "nominal_eta"
        networks.append(config["network"]["data_path"].rstrip("/").rsplit("/", 1)[-1])
        components |= {
            block for block, on in (
                ("public", extensions.public_charging.enabled),
                ("hours", extensions.hours_of_service.enabled),
                ("requests", extensions.requests.dynamic),
            ) if on
        }
    assert sorted(networks) == ["california", "california", "pacific_northwest", "texas", "texas"]
    assert components == {"public", "hours", "requests"}
