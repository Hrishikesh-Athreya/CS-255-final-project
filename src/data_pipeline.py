"""Synthetic and real token streams for sketch experiments.

Provides Shakespeare download/tokenization plus reproducible uniform and Zipf draws.
"""

import os
import re
import urllib.request

import numpy as np

# Project ``data/`` directory (sibling of ``src/``).
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
# Local path for Project Gutenberg's Shakespeare plain text.
SHAKESPEARE_PATH = os.path.join(DATA_DIR, "shakespeare.txt")
# Direct HTTP URL for the same text (cached under ``SHAKESPEARE_PATH``).
SHAKESPEARE_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"


def download_shakespeare() -> str:
    """Ensure ``SHAKESPEARE_PATH`` exists, downloading from ``SHAKESPEARE_URL`` if needed.

    Returns:
        Absolute filesystem path to the cached text file.
    """
    if os.path.exists(SHAKESPEARE_PATH):
        return SHAKESPEARE_PATH
    os.makedirs(DATA_DIR, exist_ok=True)
    urllib.request.urlretrieve(SHAKESPEARE_URL, SHAKESPEARE_PATH)
    return SHAKESPEARE_PATH


def tokenize(text: str) -> list[str]:
    """Lowercase and extract contiguous alphabetic words (a–z only).

    Args:
        text: Raw document text.

    Returns:
        List of word tokens in order of appearance.
    """
    return re.findall(r"[a-z]+", text.lower())


def load_shakespeare_stream() -> list[str]:
    """Load the cached Shakespeare corpus and return it as a token stream."""
    path = download_shakespeare()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return tokenize(text)


def generate_uniform_stream(n: int, cardinality: int, seed: int = 42) -> list[str]:
    """Draw ``n`` tokens uniformly with replacement from ``cardinality`` distinct labels.

    Args:
        n: Stream length.
        cardinality: Number of distinct items ``item_0`` .. ``item_{K-1}``.
        seed: RNG seed for reproducibility.

    Returns:
        List of string tokens ``item_{j}`` with uniform independent indices.
    """
    rng = np.random.default_rng(seed)
    return [f"item_{i}" for i in rng.integers(0, cardinality, size=n)]


def generate_zipf_stream(
    n: int, alpha: float = 1.5, cardinality: int = 1000, seed: int = 42
) -> list[str]:
    """Draw ``n`` tokens from a Zipf(power-law) distribution over ``cardinality`` items.

    Args:
        n: Stream length.
        alpha: Zipf exponent (larger → steeper head; more skew).
        cardinality: Support size for ranks 1..K (weights ∝ 1/rank^alpha).
        seed: RNG seed for reproducibility.

    Returns:
        List of tokens ``item_{j}`` where ``j`` follows the Zipf-like discrete law.
    """
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, cardinality + 1, dtype=float)
    weights = 1.0 / (ranks**alpha)
    probs = weights / weights.sum()
    indices = rng.choice(cardinality, size=n, p=probs)
    return [f"item_{i}" for i in indices]
