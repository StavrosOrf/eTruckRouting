"""Deterministic random streams for reproducible evaluation scenarios.

The environment uses two kinds of random values:

* sequential streams for instance generation, where call order is part of the
  generation procedure; and
* keyed values for exogenous events, where a value must depend only on the
  scenario, stream name, and event identity rather than on policy RNG usage.

Keyed values make common-random-number evaluation possible without relying on
Python's process-randomized ``hash`` implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np


SCENARIO_VERSION = "1"


def _normalise_key(value: Any) -> Any:
    """Convert event keys to a stable JSON-serialisable representation."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _normalise_key(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_key(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(frozen=True)
class ScenarioDescriptor:
    """Identity and generator version for one evaluation scenario."""

    seed: int
    version: str = SCENARIO_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "version": self.version}


class ScenarioRandomStreams:
    """Named sequential streams and deterministic keyed event samples."""

    def __init__(self, seed: int, version: str = SCENARIO_VERSION):
        self.descriptor = ScenarioDescriptor(seed=int(seed), version=str(version))
        self._generators: dict[str, np.random.Generator] = {}
        self._normal_cache: dict[tuple[str, Hashable], float] = {}

    @property
    def seed(self) -> int:
        return self.descriptor.seed

    def metadata(self) -> dict[str, Any]:
        return self.descriptor.as_dict()

    def _seed_for(self, stream: str, key: Any = None) -> int:
        return int.from_bytes(self._digest(stream, key), byteorder="little", signed=False)

    def _digest(self, stream: str, key: Any = None) -> bytes:
        payload = {
            "scenario_seed": self.descriptor.seed,
            "scenario_version": self.descriptor.version,
            "stream": str(stream),
            "key": _normalise_key(key),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.blake2b(encoded, digest_size=16).digest()

    def generator(self, stream: str) -> np.random.Generator:
        """Return a stateful generator isolated from every other named stream."""
        stream = str(stream)
        if stream not in self._generators:
            self._generators[stream] = np.random.default_rng(
                self._seed_for(stream, key="sequential")
            )
        return self._generators[stream]

    def standard_normal(self, stream: str, key: Any) -> float:
        """Return a deterministic N(0, 1) sample for a named event key."""
        normalised = _normalise_key(key)
        cache_key = (
            str(stream),
            json.dumps(normalised, sort_keys=True, separators=(",", ":")),
        )
        if cache_key not in self._normal_cache:
            rng = np.random.default_rng(self._seed_for(stream, key=normalised))
            self._normal_cache[cache_key] = float(rng.standard_normal())
        return self._normal_cache[cache_key]

    def keyed_generator(self, stream: str, key: Any) -> np.random.Generator:
        """Fresh generator for one event identity (common random numbers).

        The draws depend only on the scenario, the stream name and the key, so
        no fleet action can shift them.
        """
        return np.random.default_rng(self._seed_for(stream, key=_normalise_key(key)))

    def uniforms(self, stream: str, key: Any, size: int = 1) -> np.ndarray:
        """``size`` deterministic U[0, 1) values for one event identity.

        One or two values come straight from the 128-bit digest (53 bits each),
        which avoids building a generator for every outside-traffic event; more
        values come from a keyed generator, so a size-3 draw does not extend a
        size-2 one. These keyed draws are new with the opt-in components and
        are not used by legacy code, whose draws are unchanged.
        """
        if size <= 2:
            digest = self._digest(stream, _normalise_key(key))
            words = (
                int.from_bytes(digest[:8], "little"),
                int.from_bytes(digest[8:], "little"),
            )
            return np.asarray([(word >> 11) * 2.0**-53 for word in words[:size]])
        return self.keyed_generator(stream, key).random(size)

    def exponential(self, stream: str, key: Any, mean: float) -> float:
        """One deterministic Exp(mean) value for one event identity."""
        if not np.isfinite(mean) or mean <= 0.0:
            raise ValueError("exponential mean must be finite and positive")
        return float(-mean * np.log1p(-self.uniforms(stream, key, 1)[0]))

    def clear_cache(self) -> None:
        """Clear derived values without changing scenario identity."""
        self._normal_cache.clear()
