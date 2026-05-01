"""FM variance study: distribution of FM estimates over many seeds versus HLL.

Demonstrates why median-of-means + HLL exist: FM is high-variance.

Output:
    results/phase2/fm_variance.csv
    results/phase2/fm_vs_hll_distribution.png
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from experiments._common import RESULTS_DIR, banner, info, plt, save_plot, write_csv
from src.data_pipeline import generate_uniform_stream
from src.exact_counters import exact_distinct_count
from src.flajolet_martin import FlajoletMartin
from src.hyperloglog import HyperLogLog


def run(n: int, cardinality: int, n_seeds: int, num_hashes: int, num_groups: int, p: int) -> dict:
    fm_estimates: list[float] = []
    hll_estimates: list[float] = []
    truths: list[int] = []
    for seed in range(n_seeds):
        stream = generate_uniform_stream(n=n, cardinality=cardinality, seed=seed)
        true_distinct = exact_distinct_count(stream)
        truths.append(true_distinct)

        fm = FlajoletMartin(num_hashes=num_hashes, num_groups=num_groups)
        fm.process_stream(stream)
        fm_estimates.append(fm.estimate())

        hll = HyperLogLog(p=p)
        hll.process_stream(stream)
        hll_estimates.append(hll.estimate())

    truth_mean = statistics.fmean(truths)
    return {
        "truth_mean": truth_mean,
        "fm_estimates": fm_estimates,
        "hll_estimates": hll_estimates,
        "n_seeds": n_seeds,
    }


def summarize(label: str, ests: list[float], truth: float) -> dict:
    rel_errs = [abs(e - truth) / truth for e in ests]
    return {
        "algorithm": label,
        "mean": statistics.fmean(ests),
        "stdev": statistics.pstdev(ests) if len(ests) > 1 else 0.0,
        "min": min(ests),
        "max": max(ests),
        "mean_rel_error": statistics.fmean(rel_errs),
        "max_rel_error": max(rel_errs),
        "truth_mean": truth,
    }


def plot_distribution(fm: list[float], hll: list[float], truth: float) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = 30
    ax.hist(fm, bins=bins, alpha=0.55, label="Flajolet–Martin")
    ax.hist(hll, bins=bins, alpha=0.55, label="HyperLogLog")
    ax.axvline(truth, color="black", linestyle="--", label=f"Exact distinct ≈ {truth:,.0f}")
    ax.set_xlabel("Distinct-count estimate")
    ax.set_ylabel("Frequency over seeds")
    ax.set_title("Estimate distribution over independent seeds: FM vs HLL")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    return save_plot(fig, "fm_vs_hll_distribution.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--cardinality", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--fm-hashes", type=int, default=128)
    ap.add_argument("--fm-groups", type=int, default=8)
    ap.add_argument("--hll-p", type=int, default=14)
    args = ap.parse_args()

    banner(
        f"FM vs HLL variance: n={args.n} card={args.cardinality} seeds={args.seeds}"
    )
    out = run(
        n=args.n,
        cardinality=args.cardinality,
        n_seeds=args.seeds,
        num_hashes=args.fm_hashes,
        num_groups=args.fm_groups,
        p=args.hll_p,
    )

    fm_summary = summarize("flajolet_martin", out["fm_estimates"], out["truth_mean"])
    hll_summary = summarize("hyperloglog", out["hll_estimates"], out["truth_mean"])

    info(f"FM  mean={fm_summary['mean']:,.0f}  std={fm_summary['stdev']:,.0f}  meanRelErr={fm_summary['mean_rel_error']*100:.2f}%")
    info(f"HLL mean={hll_summary['mean']:,.0f}  std={hll_summary['stdev']:,.0f}  meanRelErr={hll_summary['mean_rel_error']*100:.2f}%")

    csv_path = RESULTS_DIR / "fm_variance.csv"
    write_csv(
        csv_path,
        [fm_summary, hll_summary],
        ["algorithm", "mean", "stdev", "min", "max", "mean_rel_error", "max_rel_error", "truth_mean"],
    )
    info(f"wrote {csv_path}")

    info(f"wrote {plot_distribution(out['fm_estimates'], out['hll_estimates'], out['truth_mean'])}")


if __name__ == "__main__":
    main()
