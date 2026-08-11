"""Tests for charger-type preservation and validation."""

import pytest

from EVRoutingEnv.utils.utils import map_charger_type


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Level2", "Level2"),
        ("level 2", "Level2"),
        ("L2", "Level2"),
        ("DCFast", "DCFast"),
        ("DCFC", "DCFast"),
        ("Level 3", "DCFast"),
    ],
)
def test_charger_type_is_normalized_without_collapsing_classes(
    raw: str,
    expected: str,
) -> None:
    assert map_charger_type(raw) == expected


def test_unknown_charger_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported charger type"):
        map_charger_type("mystery")
