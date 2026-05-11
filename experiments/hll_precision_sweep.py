"""HLL precision sweep: error and memory vs p, averaged over multiple seeded streams.

Output:
    results/phase2/hll_precision_sweep.csv
    results/phase2/hll_error_vs_p.png
    results/phase2/hll_error_vs_memory.png
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from experiments._common import RESULTS_DIR, banner, info, plt, save_plot, write_csv
from src.data_pipeline import generate_uniform_stream
from src.exact_counters import exact_distinct_count
from src.hyperloglog import HyperLogLog


def run(p_values: list[int], n: int, cardinality: int, seeds: list[int]) -> list[dict]:
    """For each precision ``p``, average relative HLL error over ``seeds`` uniform streams.

    Returns:
        One result dict per ``p`` (mean/std error, theoretical 1.04/√m, metadata).
    """
    rows: list[dict] = []
    for p in p_values:
        m = 1 << p
        rel_errs = []
        for seed in seeds:
            stream = generate_uniform_stream(n=n, cardinality=cardinality, seed=seed)
            true_distinct = exact_distinct_count(stream)
            hll = HyperLogLog(p=p)
            hll.process_stream(stream)
            est = hll.estimate()
            rel = abs(est - true_distinct) / true_distinct
            rel_errs.append(rel)
        mean_err = statistics.fmean(rel_errs)
        std_err = statistics.pstdev(rel_errs) if len(rel_errs) > 1 else 0.0
        theoretical = 1.04 / (m ** 0.5)
        rows.append(
            {
                "p": p,
                "registers_m": m,
                "memory_bytes_estimate": m,
                "mean_rel_error": mean_err,
                "std_rel_error": std_err,
                "theoretical_error": theoretical,
                "n_seeds": len(seeds),
                "stream_n": n,
                "stream_cardinality": cardinality,
            }
        )
        info(f"p={p:2d} m={m:6d}  mean_err={mean_err*100:6.2f}%  theo={theoretical*100:6.2f}%")
    return rows


def plot_error_vs_p(rows: list[dict]) -> Path:
    """Plot empirical mean ± std relative error vs ``p`` with theoretical curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ps = [r["p"] for r in rows]
    means = [r["mean_rel_error"] * 100 for r in rows]
    stds = [r["std_rel_error"] * 100 for r in rows]
    theo = [r["theoretical_error"] * 100 for r in rows]
    ax.errorbar(ps, means, yerr=stds, marker="o", capsize=3, label="Empirical mean ± std")
    ax.plot(ps, theo, linestyle="--", marker="x", label="Theoretical 1.04/√m")
    ax.set_xlabel("Precision p (registers m = 2^p)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("HyperLogLog: error vs precision")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()
    return save_plot(fig, "hll_error_vs_p.png")


def plot_error_vs_memory(rows: list[dict]) -> Path:
    """Plot mean relative error vs register count ``m`` (log–log tradeoff view)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    mems = [r["memory_bytes_estimate"] for r in rows]
    means = [r["mean_rel_error"] * 100 for r in rows]
    ax.plot(mems, means, marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Registers m (proxy for memory)")
    ax.set_ylabel("Mean relative error (%)")
    ax.set_title("HLL: accuracy vs memory tradeoff")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    return save_plot(fig, "hll_error_vs_memory.png")


def main() -> None:
    """CLI entry: sweep ``p``, write CSV and two PNGs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-min", type=int, default=4)
    ap.add_argument("--p-max", type=int, default=14)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--cardinality", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    p_values = list(range(args.p_min, args.p_max + 1))
    seeds = list(range(args.seeds))

    banner(f"HLL precision sweep p={args.p_min}..{args.p_max}, n={args.n}, card={args.cardinality}, seeds={args.seeds}")
    rows = run(p_values, args.n, args.cardinality, seeds)

    csv_path = RESULTS_DIR / "hll_precision_sweep.csv"
    write_csv(
        csv_path,
        rows,
        [
            "p",
            "registers_m",
            "memory_bytes_estimate",
            "mean_rel_error",
            "std_rel_error",
            "theoretical_error",
            "n_seeds",
            "stream_n",
            "stream_cardinality",
        ],
    )
    info(f"wrote {csv_path}")

    info(f"wrote {plot_error_vs_p(rows)}")
    info(f"wrote {plot_error_vs_memory(rows)}")


if __name__ == "__main__":
    main()
