import mmh3
import numpy as np


class FlajoletMartin:

    def __init__(self, num_hashes: int = 64, num_groups: int = 8):
        if num_hashes % num_groups != 0:
            raise ValueError("num_hashes must be divisible by num_groups")
        self.num_hashes = num_hashes
        self.num_groups = num_groups
        self.group_size = num_hashes // num_groups
        self.max_trailing_zeros = np.zeros(num_hashes, dtype=int)

    def _trailing_zeros(self, value: int) -> int:
        if value == 0:
            return 32
        count = 0
        while (value & 1) == 0:
            count += 1
            value >>= 1
        return count

    def add(self, item: str):
        for i in range(self.num_hashes):
            h = mmh3.hash(item, seed=i) & 0xFFFFFFFF
            tz = self._trailing_zeros(h)
            if tz > self.max_trailing_zeros[i]:
                self.max_trailing_zeros[i] = tz

    def process_stream(self, stream: list[str]):
        for item in stream:
            self.add(item)

    def estimate(self) -> float:
        phi = 0.77351
        group_estimates = []
        for i in range(self.num_groups):
            chunk = self.max_trailing_zeros[i * self.group_size:(i + 1) * self.group_size]
            mean_r = float(np.mean(chunk))
            group_estimates.append((2.0 ** mean_r) / phi)
        return float(np.median(group_estimates))
