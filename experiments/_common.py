from __future__ import annotations

import csv
import os
import sys
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "phase2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import (
    generate_uniform_stream,
    generate_zipf_stream,
    load_shakespeare_stream,
)


def get_dataset(name: str, n: int = 100_000, seed: int = 42):
    if name == "shakespeare":
        return load_shakespeare_stream()
    if name == "uniform":
        return generate_uniform_stream(n=n, cardinality=max(2, n // 10), seed=seed)
    if name == "zipf":
        return generate_zipf_stream(n=n, alpha=1.5, cardinality=max(2, n // 20), seed=seed)
    raise ValueError(f"unknown dataset: {name}")


@contextmanager
def measure(label: str = ""):
    tracemalloc.start()
    t0 = time.perf_counter()
    state = {"label": label, "elapsed_s": 0.0, "peak_kb": 0.0}
    try:
        yield state
    finally:
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        state["elapsed_s"] = elapsed
        state["peak_kb"] = peak / 1024.0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_plot(fig, name: str) -> Path:
    out = RESULTS_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def banner(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def info(msg: str) -> None:
    print(f"  {msg}", flush=True)
