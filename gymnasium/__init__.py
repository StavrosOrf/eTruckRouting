"""
Minimal shim of the gymnasium API needed for the EVPR project.
This avoids pulling the full dependency in environments where it's unavailable.
"""

from . import spaces  # noqa: F401


class Env:
    """Lightweight stand-in for gymnasium.Env."""

    metadata = {}

    def reset(self, *, seed=None, options=None):
        # Default stub just returns None observation/info.
        return None, {}

    def step(self, action):
        raise NotImplementedError("Step not implemented in minimal Env stub.")

    def close(self):
        pass


__all__ = ["Env", "spaces"]
