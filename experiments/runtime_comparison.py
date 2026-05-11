"""Runtime + memory comparison across exact counters and three sketches.

Output:
    results/phase2/runtime_comparison.csv
    results/phase2/runtime_bar.png
    results/phase2/peak_memory_bar.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments._common import RESULTS_DIR, banner, get_dataset, info, measure, plt, save_plot, write_csv
from src.count_min_sketch import CountMinSketch
from src.exact_counters import exact_distinct_count, exact_frequencies
from src.flajolet_martin import FlajoletMartin
from src.hyperloglog import HyperLogLog


def time_exact_distinct(stream: list[str]) -> dict:
    """Time and memory for exact distinct count on ``stream``."""
    with measure("exact_distinct") as m:
        exact_distinct_count(stream)
    return m


def time_exact_freq(stream: list[str]) -> dict:
    """Time and memory for building exact frequency table on ``stream``."""
    with measure("exact_freq") as m:
        exact_frequencies(stream)
    return m


def time_fm(stream: list[str], hashes: int, groups: int) -> dict:
    """Time and memory for Flajolet–Martin ingest + final estimate."""
    with measure("fm") as m:
        fm = FlajoletMartin(num_hashes=hashes, num_groups=groups)
        fm.process_stream(stream)
        fm.estimate()
    return m


def time_hll(stream: list[str], p: int) -> dict:
    """Time and memory for HyperLogLog ingest + final estimate."""
    with measure("hll") as m:
        hll = HyperLogLog(p=p)
        hll.process_stream(stream)
        hll.estimate()
    return m


def time_cms(stream: list[str], eps: float, delta: float) -> dict:
    """Time and memory for Count–Min Sketch ingest (no per-query phase)."""
    with measure("cms") as m:
        cms = CountMinSketch(epsilon=eps, delta=delta)
        cms.process_stream(stream)
    return m


def run_dataset(name: str, n: int, fm_hashes: int, fm_groups: int, hll_p: int, eps: float, delta: float) -> list[dict]:
    """Load dataset ``name`` and append one metrics row per algorithm."""
    info(f"loading dataset: {name} (n={n})")
    stream = get_dataset(name, n=n)
    info(f"  stream length = {len(stream):,}")
    rows = []
    for label, runner in [
        ("exact_distinct", lambda: time_exact_distinct(stream)),
        ("exact_freq", lambda: time_exact_freq(stream)),
        ("flajolet_martin", lambda: time_fm(stream, fm_hashes, fm_groups)),
        ("hyperloglog", lambda: time_hll(stream, hll_p)),
        ("count_min_sketch", lambda: time_cms(stream, eps, delta)),
    ]:
        m = runner()
        rows.append(
            {
                "dataset": name,
                "stream_n": len(stream),
                "algorithm": label,
                "elapsed_s": round(m["elapsed_s"], 4),
                "peak_kb": round(m["peak_kb"], 2),
            }
        )
        info(f"  {label:<18} time={m['elapsed_s']*1000:8.2f} ms  peak={m['peak_kb']:>10,.1f} KB")
    return rows


def plot_bars(rows: list[dict], metric: str, title: str, ylabel: str, name: str) -> Path:
    """Grouped bar chart: datasets on X, one bar series per algorithm; ``metric`` is CSV column key."""
    datasets = sorted({r["dataset"] for r in rows})
    algs = ["exact_distinct", "exact_freq", "flajolet_martin", "hyperloglog", "count_min_sketch"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_w = 0.15  # horizontal offset step between algorithm series within each dataset group
    for i, alg in enumerate(algs):
        xs: list[float] = []  # bar center x positions
        ys: list[float] = []  # bar heights (``metric`` column)
        for j, ds in enumerate(datasets):
            for r in rows:
                if r["dataset"] == ds and r["algorithm"] == alg:
                    xs.append(j + i * bar_w)
                    ys.append(r[metric])
                    break
        ax.bar(xs, ys, width=bar_w, label=alg)
    ax.set_xticks([j + (len(algs) * bar_w) / 2 - bar_w / 2 for j in range(len(datasets))])
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8)
    return save_plot(fig, name)


def main() -> None:
    """CLI entry: benchmark all algorithms on selected datasets; write CSV and two bar charts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["uniform", "zipf", "shakespeare"])
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--fm-hashes", type=int, default=128)
    ap.add_argument("--fm-groups", type=int, default=8)
    ap.add_argument("--hll-p", type=int, default=14)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--delta", type=float, default=0.05)
    args = ap.parse_args()

    banner(f"Runtime + memory comparison on datasets={args.datasets}")
    all_rows: list[dict] = []
    for ds in args.datasets:
        all_rows.extend(
            run_dataset(
                ds,
                n=args.n,
                fm_hashes=args.fm_hashes,
                fm_groups=args.fm_groups,
                hll_p=args.hll_p,
                eps=args.eps,
                delta=args.delta,
            )
        )

    csv_path = RESULTS_DIR / "runtime_comparison.csv"
    write_csv(csv_path, all_rows, ["dataset", "stream_n", "algorithm", "elapsed_s", "peak_kb"])
    info(f"wrote {csv_path}")

    info(f"wrote {plot_bars(all_rows, 'elapsed_s', 'Runtime per algorithm × dataset (log scale)', 'Seconds (log)', 'runtime_bar.png')}")
    info(f"wrote {plot_bars(all_rows, 'peak_kb', 'Peak memory per algorithm × dataset (log scale)', 'Peak KB (log)', 'peak_memory_bar.png')}")


if __name__ == "__main__":
    main()
