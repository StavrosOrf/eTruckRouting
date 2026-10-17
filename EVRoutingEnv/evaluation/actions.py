"""Strict conversion at policy/environment boundaries (never modulo-wrap IDs)."""

from numbers import Integral


def validate_discrete_action(action, action_count: int) -> int:
    """Accept scalar integer IDs only, preserving an invalid policy's failure."""
    if isinstance(action, bool) or not isinstance(action, Integral):
        raise ValueError(f"policy action must be an integer ID, got {action!r}")  # noqa: TRY004 -- matches Gym env.step validation
    value = int(action)
    if not 0 <= value < action_count:
        raise ValueError(f"policy action {value} outside [0, {action_count})")
    return value
