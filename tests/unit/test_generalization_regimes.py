"""The generalization campaign must respect both the regime and the method.

A regime says what changed about the world. A method's environment_overrides say
what environment that policy was trained in -- an unmasked arm needs the
structural mask, a feature-ablated arm needs the same blocks blanked. Applying
only the first scores a policy under conditions it never saw, which reads as a
collapse in generalization that is really an evaluation error.
"""

import os

import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from scripts.evaluation.run_generalization_campaign import REGIMES, _apply


BASE = {
    "environment": {"num_trucks": 2, "policy_action_mask": "hard"},
    "charging": {"station_power_classes_kw": [150.0, 350.0, 750.0]},
    "truck": {"battery_capacity": 400.0},
    "network": {},
    "traffic": {"std_dev_factor": 0.15},
    "problem": {"base_service_time": 0.2},
}


def test_regime_and_method_overrides_both_survive() -> None:
    regime = {"charging": {"station_power_classes_kw": [50.0, 150.0, 350.0]}}
    method = {"environment": {"policy_action_mask": "structural"}}

    config = _apply(_apply(BASE, regime), method)

    assert config["charging"]["station_power_classes_kw"] == [50.0, 150.0, 350.0]
    assert config["environment"]["policy_action_mask"] == "structural"
    # Untouched sections keep their values, and the base is not mutated.
    assert config["truck"]["battery_capacity"] == 400.0
    assert BASE["environment"]["policy_action_mask"] == "hard"


def test_every_regime_names_an_existing_section() -> None:
    for name, regime in REGIMES.items():
        for section in regime["overrides"]:
            assert section in BASE, f"{name} overrides unknown section {section}"


def test_every_regime_is_labelled_and_actually_changes_something() -> None:
    """A regime that overrides nothing is the in-distribution control, once."""
    controls = [
        name for name, regime in REGIMES.items() if not regime["overrides"]
    ]
    assert controls == ["in_distribution"]
    for name, regime in REGIMES.items():
        assert regime["kind"] in {"interpolation", "size_transfer", "ood"}, name
        assert regime["description"], name


def test_unknown_section_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown config section"):
        _apply(BASE, {"not_a_section": {"x": 1}})
