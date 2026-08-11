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
        digest = hashlib.blake2b(encoded, digest_size=16).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

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

    def clear_cache(self) -> None:
        """Clear derived values without changing scenario identity."""
        self._normal_cache.clear()
