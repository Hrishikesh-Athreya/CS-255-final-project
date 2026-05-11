"""Flajolet–Martin (FM): probabilistic distinct counting via trailing zeros in hashes.

Uses multiple independent hash functions grouped for a median-of-means style
stabilization (still higher variance than HLL for comparable cost in this codebase).
"""

import mmh3
import numpy as np


class FlajoletMartin:
    """Maintains per-hash maximum trailing-zero counts; estimate is median of group means."""

    def __init__(self, num_hashes: int = 64, num_groups: int = 8):
        """Allocate ``num_hashes`` parallel FM counters partitioned into ``num_groups`` groups.

        Args:
            num_hashes: Number of independent hash streams (must divide ``num_groups``).
            num_groups: Groups over which we take the median of per-group estimates.

        Raises:
            ValueError: If ``num_hashes`` is not divisible by ``num_groups``.

        Attributes:
            num_hashes: Total hash functions.
            num_groups: Number of groups for median-of-means.
            group_size: Hashes per group (num_hashes // num_groups).
            max_trailing_zeros: Per-hash maximum trailing zero count seen so far.
        """
        if num_hashes % num_groups != 0:
            raise ValueError("num_hashes must be divisible by num_groups")
        self.num_hashes = num_hashes
        self.num_groups = num_groups
        self.group_size = num_hashes // num_groups
        self.max_trailing_zeros = np.zeros(num_hashes, dtype=int)

    def _trailing_zeros(self, value: int) -> int:
        """Return the number of trailing zero bits in ``value`` (LSB side); 32 if value is 0."""
        if value == 0:
            return 32
        count = 0
        while (value & 1) == 0:
            count += 1
            value >>= 1
        return count

    def add(self, item: str) -> None:
        """Update each hash's maximum trailing-zero statistic for ``item``."""
        for i in range(self.num_hashes):
            h = mmh3.hash(item, seed=i) & 0xFFFFFFFF
            tz = self._trailing_zeros(h)
            if tz > self.max_trailing_zeros[i]:
                self.max_trailing_zeros[i] = tz

    def process_stream(self, stream: list[str]) -> None:
        """Ingest every element of ``stream``."""
        for item in stream:
            self.add(item)

    def estimate(self) -> float:
        """Return the median across groups of (2^{mean trailing zeros} / φ), φ ≈ 0.77351."""
        phi = 0.77351
        group_estimates = []
        for i in range(self.num_groups):
            chunk = self.max_trailing_zeros[i * self.group_size : (i + 1) * self.group_size]
            mean_r = float(np.mean(chunk))
            group_estimates.append((2.0**mean_r) / phi)
        return float(np.median(group_estimates))
