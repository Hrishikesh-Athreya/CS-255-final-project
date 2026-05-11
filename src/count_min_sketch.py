"""Count–Min Sketch: approximate per-item frequencies with one-sided error bounds.

Width and depth follow the standard CMS construction so that with probability
at least 1 − δ, every estimate is at most f(x) + εN for stream length N.
"""

import math

import mmh3
import numpy as np


class CountMinSketch:
    """Streaming frequency sketch: updates are increments; estimates are lower-biased minima across rows."""

    def __init__(self, epsilon: float = 0.01, delta: float = 0.05):
        """Build a sketch with target relative error ``epsilon`` and failure probability ``delta``.

        Args:
            epsilon: Upper bound on additive error εN (with high probability) for any item.
            delta: Probability that the εN guarantee fails for some item.

        Attributes:
            epsilon: Same as parameter (stored for introspection).
            delta: Same as parameter.
            width: Number of columns per row; ceil(e / epsilon).
            depth: Number of hash rows; ceil(ln(1/delta)).
            table: Integer matrix of shape (depth, width) holding per-cell counts.
            total_count: Number of ``add`` calls (stream length if only ``add`` is used).
        """
        self.epsilon = epsilon
        self.delta = delta
        self.width = math.ceil(math.e / epsilon)
        self.depth = math.ceil(math.log(1.0 / delta))
        self.table = np.zeros((self.depth, self.width), dtype=np.int64)
        self.total_count = 0

    def _hash(self, item: str, i: int) -> int:
        """Map ``item`` to a column index in row ``i`` (0 .. width-1)."""
        return mmh3.hash(item, seed=i) % self.width

    def add(self, item: str) -> None:
        """Increment the sketch by one observation of ``item``."""
        self.total_count += 1
        for i in range(self.depth):
            j = self._hash(item, i)
            self.table[i, j] += 1

    def process_stream(self, stream: list[str]) -> None:
        """Apply ``add`` to every element of ``stream`` in order."""
        for item in stream:
            self.add(item)

    def estimate(self, item: str) -> int:
        """Return the minimum across rows of the counters hit by ``item`` (CMS estimate)."""
        return int(min(self.table[i][self._hash(item, i)] for i in range(self.depth)))

    def heavy_hitters(self, stream: list[str], threshold: float) -> dict[str, int]:
        """Return candidate heavy hitters: items whose estimate ≥ threshold × stream length.

        Candidates are drawn from the distinct set of ``stream``; estimates may include
        false positives but not false negatives relative to the sketch's guarantees.

        Args:
            stream: Full stream (used for candidate set and implicit length).
            threshold: Fractional cutoff (same semantics as ``exact_heavy_hitters``).

        Returns:
            Mapping from item to CMS estimate for items at or above the cutoff.
        """
        cutoff = threshold * self.total_count
        candidates = set(stream)
        return {
            item: self.estimate(item)
            for item in candidates
            if self.estimate(item) >= cutoff
        }

    def get_dimensions(self) -> dict:
        """Return parameters and derived table shape for logging or plotting."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "width": self.width,
            "depth": self.depth,
            "memory_cells": self.width * self.depth,
        }
