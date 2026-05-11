"""Unit tests for data generation, exact baselines, and sketch implementations."""

import unittest

from src.data_pipeline import generate_uniform_stream, generate_zipf_stream, tokenize
from src.exact_counters import exact_distinct_count, exact_frequencies, exact_heavy_hitters
from src.flajolet_martin import FlajoletMartin
from src.hyperloglog import HyperLogLog
from src.count_min_sketch import CountMinSketch


class TestDataPipeline(unittest.TestCase):
    """Tests for ``tokenize`` and synthetic stream generators."""

    def test_tokenize_basic(self):
        tokens = tokenize("Hello, World! This is a TEST.")
        self.assertEqual(tokens, ["hello", "world", "this", "is", "a", "test"])

    def test_tokenize_empty(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("123 !@# $%^"), [])

    def test_uniform_stream(self):
        stream = generate_uniform_stream(1000, 50)
        self.assertEqual(len(stream), 1000)
        distinct = len(set(stream))
        self.assertGreater(distinct, 1)
        self.assertLessEqual(distinct, 50)

    def test_zipf_stream(self):
        stream = generate_zipf_stream(1000, alpha=1.5, cardinality=100)
        self.assertEqual(len(stream), 1000)
        distinct = len(set(stream))
        self.assertGreater(distinct, 1)
        self.assertLessEqual(distinct, 100)


class TestExactCounters(unittest.TestCase):
    """Tests for exact distinct count, frequencies, and heavy-hitters helpers."""

    def test_distinct_count(self):
        stream = ["a", "b", "c", "a", "b", "a"]
        self.assertEqual(exact_distinct_count(stream), 3)

    def test_frequencies(self):
        stream = ["a", "b", "c", "a", "b", "a"]
        freqs = exact_frequencies(stream)
        self.assertEqual(freqs, {"a": 3, "b": 2, "c": 1})

    def test_heavy_hitters(self):
        stream = ["a"] * 50 + ["b"] * 30 + ["c"] * 20
        hitters = exact_heavy_hitters(stream, threshold=0.25)
        self.assertIn("a", hitters)
        self.assertIn("b", hitters)
        self.assertNotIn("c", hitters)


class TestFlajoletMartin(unittest.TestCase):
    """Sanity checks for FM cardinality estimates and constructor validation."""

    def test_small_known_input(self):
        stream = [f"item_{i}" for i in range(100)]
        fm = FlajoletMartin(num_hashes=64, num_groups=8)
        fm.process_stream(stream)
        estimate = fm.estimate()
        self.assertGreater(estimate, 30)
        self.assertLess(estimate, 300)

    def test_duplicates_dont_increase_count(self):
        fm = FlajoletMartin(num_hashes=64, num_groups=8)
        stream_unique = [f"item_{i}" for i in range(50)]
        stream_with_dups = stream_unique * 10
        fm.process_stream(stream_with_dups)
        estimate = fm.estimate()
        self.assertLess(estimate, 200)

    def test_invalid_groups(self):
        with self.assertRaises(ValueError):
            FlajoletMartin(num_hashes=10, num_groups=3)


class TestHyperLogLog(unittest.TestCase):
    """Sanity checks for HLL estimates and precision bounds."""

    def test_small_known_input(self):
        stream = [f"item_{i}" for i in range(1000)]
        hll = HyperLogLog(p=10)
        hll.process_stream(stream)
        estimate = hll.estimate()
        self.assertGreater(estimate, 800)
        self.assertLess(estimate, 1200)

    def test_higher_precision_less_error(self):
        stream = [f"item_{i}" for i in range(5000)]
        hll_low = HyperLogLog(p=6)
        hll_high = HyperLogLog(p=14)
        hll_low.process_stream(stream)
        hll_high.process_stream(stream)
        error_high = abs(hll_high.estimate() - 5000) / 5000
        self.assertLess(error_high, 0.1)

    def test_invalid_precision(self):
        with self.assertRaises(ValueError):
            HyperLogLog(p=2)
        with self.assertRaises(ValueError):
            HyperLogLog(p=20)


class TestCountMinSketch(unittest.TestCase):
    """Tests for CMS frequency estimates, heavy hitters, and table dimensions."""

    def test_frequency_estimate(self):
        stream = ["a"] * 100 + ["b"] * 50 + ["c"] * 10
        cms = CountMinSketch(epsilon=0.01, delta=0.05)
        cms.process_stream(stream)
        self.assertGreaterEqual(cms.estimate("a"), 100)
        self.assertGreaterEqual(cms.estimate("b"), 50)
        self.assertGreaterEqual(cms.estimate("c"), 10)

    def test_heavy_hitters_detection(self):
        stream = ["a"] * 100 + ["b"] * 50 + ["c"] * 5
        cms = CountMinSketch(epsilon=0.01, delta=0.05)
        cms.process_stream(stream)
        hitters = cms.heavy_hitters(stream, threshold=0.3)
        self.assertIn("a", hitters)
        self.assertNotIn("c", hitters)

    def test_dimensions_from_params(self):
        cms = CountMinSketch(epsilon=0.1, delta=0.1)
        dims = cms.get_dimensions()
        self.assertEqual(dims["epsilon"], 0.1)
        self.assertEqual(dims["delta"], 0.1)
        self.assertGreater(dims["width"], 0)
        self.assertGreater(dims["depth"], 0)

    def test_smaller_epsilon_wider_table(self):
        cms_wide = CountMinSketch(epsilon=0.001, delta=0.05)
        cms_narrow = CountMinSketch(epsilon=0.1, delta=0.05)
        self.assertGreater(cms_wide.width, cms_narrow.width)

    def test_smaller_delta_deeper_table(self):
        cms_deep = CountMinSketch(epsilon=0.01, delta=0.001)
        cms_shallow = CountMinSketch(epsilon=0.01, delta=0.5)
        self.assertGreater(cms_deep.depth, cms_shallow.depth)


if __name__ == "__main__":
    unittest.main()
