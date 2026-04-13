import math
import mmh3
import numpy as np


class CountMinSketch:

    def __init__(self, epsilon: float = 0.01, delta: float = 0.05):
        self.epsilon = epsilon
        self.delta = delta
        self.width = math.ceil(math.e / epsilon)
        self.depth = math.ceil(math.log(1.0 / delta))
        self.table = np.zeros((self.depth, self.width), dtype=np.int64)
        self.total_count = 0

    def _hash(self, item: str, i: int) -> int:
        return mmh3.hash(item, seed=i) % self.width

    def add(self, item: str):
        self.total_count += 1
        for i in range(self.depth):
            j = self._hash(item, i)
            self.table[i][j] += 1

    def process_stream(self, stream: list[str]):
        for item in stream:
            self.add(item)

    def estimate(self, item: str) -> int:
        return int(min(self.table[i][self._hash(item, i)] for i in range(self.depth)))

    def heavy_hitters(self, stream: list[str], threshold: float) -> dict[str, int]:
        cutoff = threshold * self.total_count
        candidates = set(stream)
        return {
            item: self.estimate(item)
            for item in candidates
            if self.estimate(item) >= cutoff
        }

    def get_dimensions(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "width": self.width,
            "depth": self.depth,
            "memory_cells": self.width * self.depth,
        }
