"""
Minimal subset of gymnasium.spaces required for the EVPR codebase.
Provides Discrete and Box spaces with simple sampling utilities.
"""

from __future__ import annotations

import numbers
from typing import Iterable, Tuple

import numpy as np


class Space:
    """Base class for all stub spaces."""

    def sample(self):
        raise NotImplementedError

    def contains(self, x) -> bool:
        raise NotImplementedError


class Discrete(Space):
    """Simplified Discrete space."""

    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("Discrete space must have positive n.")
        self.n = int(n)

    def sample(self) -> int:
        return int(np.random.randint(self.n))

    def contains(self, x) -> bool:
        if isinstance(x, numbers.Integral):
            return 0 <= int(x) < self.n
        return False

    def __repr__(self):
        return f"Discrete({self.n})"


class Box(Space):
    """Simplified Box space supporting numeric bounds."""

    def __init__(self, low, high, shape: Tuple[int, ...] = None, dtype=np.float32):
        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        if shape is None:
            self.shape = self.low.shape
        else:
            self.shape = tuple(shape)
            if np.prod(self.shape) != np.prod(self.low.shape):
                self.low = np.full(self.shape, self.low, dtype=dtype)
                self.high = np.full(self.shape, self.high, dtype=dtype)
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high).astype(self.dtype)

    def contains(self, x) -> bool:
        arr = np.array(x, dtype=self.dtype)
        if arr.shape != self.shape:
            return False
        return np.all(arr >= self.low) and np.all(arr <= self.high)

    def __repr__(self):
        return f"Box(shape={self.shape}, dtype={self.dtype})"


__all__ = ["Space", "Discrete", "Box"]
