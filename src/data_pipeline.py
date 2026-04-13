import re
import os
import urllib.request
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SHAKESPEARE_PATH = os.path.join(DATA_DIR, "shakespeare.txt")
SHAKESPEARE_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"


def download_shakespeare():
    if os.path.exists(SHAKESPEARE_PATH):
        return SHAKESPEARE_PATH
    os.makedirs(DATA_DIR, exist_ok=True)
    urllib.request.urlretrieve(SHAKESPEARE_URL, SHAKESPEARE_PATH)
    return SHAKESPEARE_PATH


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def load_shakespeare_stream() -> list[str]:
    path = download_shakespeare()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return tokenize(text)


def generate_uniform_stream(n: int, cardinality: int, seed: int = 42) -> list[str]:
    rng = np.random.default_rng(seed)
    return [f"item_{i}" for i in rng.integers(0, cardinality, size=n)]


def generate_zipf_stream(n: int, alpha: float = 1.5, cardinality: int = 1000, seed: int = 42) -> list[str]:
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, cardinality + 1, dtype=float)
    weights = 1.0 / (ranks ** alpha)
    probs = weights / weights.sum()
    indices = rng.choice(cardinality, size=n, p=probs)
    return [f"item_{i}" for i in indices]
