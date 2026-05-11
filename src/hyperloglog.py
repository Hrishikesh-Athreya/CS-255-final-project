"""HyperLogLog (HLL): approximate cardinality in sublinear memory via register maxima.

Implements the standard 32-bit hash split into bucket index and suffix, with
small/large range corrections for raw harmonic-mean estimates.
"""

import math

import mmh3


class HyperLogLog:
    """Cardinality estimator using m = 2^p registers (p is the precision parameter)."""

    def __init__(self, p: int = 14):
        """Allocate ``m = 2**p`` registers and the Flajolet–Martin bias constant α_m.

        Args:
            p: Precision; register count is 2^p (typical 4 ≤ p ≤ 16).

        Raises:
            ValueError: If ``p`` is outside the supported range.

        Attributes:
            p: Precision parameter.
            m: Number of registers (2^p).
            registers: List of length ``m`` holding maximum "observed" ρ values per bucket.
            alpha: Multiplicative bias correction for the raw HLL estimator.
        """
        if not (4 <= p <= 16):
            raise ValueError("p must be between 4 and 16")
        self.p = p
        self.m = 1 << p
        self.registers = [0] * self.m
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)

    def _leading_zeros(self, value: int, max_bits: int) -> int:
        """Count leading zero bits of ``value`` in a field of ``max_bits`` bits (MSB first)."""
        if value == 0:
            return max_bits
        count = 0
        for i in range(max_bits - 1, -1, -1):
            if value & (1 << i):
                break
            count += 1
        return count

    def add(self, item: str) -> None:
        """Ingest one element: update the register selected by the top ``p`` hash bits."""
        h = mmh3.hash(item, seed=0) & 0xFFFFFFFF
        idx = h >> (32 - self.p)
        remaining_bits = 32 - self.p
        w = h & ((1 << remaining_bits) - 1)
        rho = self._leading_zeros(w, remaining_bits) + 1
        if rho > self.registers[idx]:
            self.registers[idx] = rho

    def process_stream(self, stream: list[str]) -> None:
        """Ingest every element of ``stream``."""
        for item in stream:
            self.add(item)

    def estimate(self) -> float:
        """Return the bias-corrected cardinality estimate (float for sub-register precision)."""
        indicator = sum(2.0 ** (-r) for r in self.registers)
        raw_estimate = self.alpha * self.m * self.m / indicator

        if raw_estimate <= 2.5 * self.m:
            zeros = self.registers.count(0)
            if zeros > 0:
                return self.m * math.log(self.m / zeros)
            return raw_estimate

        if raw_estimate <= (1 << 32) / 30.0:
            return raw_estimate

        return -(1 << 32) * math.log(1 - raw_estimate / (1 << 32))
