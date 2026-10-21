"""The dynamic benchmark configs match their generator and validate (plan section 12)."""

import yaml

from EVRoutingEnv.models.environment.extension_config import parse_extension_config
from scripts.campaigns.write_dynamic_benchmark_configs import (
    BASES,
    CONFIGS,
    base_config,
    variants,
)


def test_generated_configs_are_current_and_valid() -> None:
    seen = 0
    for name, (source, mode, terminals, context) in BASES.items():
        for suffix, expected in variants(base_config(source, mode, terminals, context)):
            path = CONFIGS / f"{name}{suffix}.yaml"
            assert yaml.safe_load(path.read_text()) == expected, path.name
            extensions = parse_extension_config(
                expected, max_time=expected["environment"]["max_time"],
                sequential=False, preassigned=False,
            )
            assert extensions.requests.mode == mode
            seen += 1
    assert seen == 15
    assert sorted(p.stem for p in CONFIGS.glob("config_dynamic_*.yaml")) == sorted(
        f"{name}{suffix}" for name in BASES
        for suffix in ("", "_hos_us", "_public_u040", "_public_u065", "_public_u090")
    )
