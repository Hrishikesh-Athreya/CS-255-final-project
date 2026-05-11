"""CMS epsilon/delta sweep: verify empirical max overcount stays within εN
across multiple seeded streams; also report heavy-hitter precision/recall.

Output:
    results/phase2/cms_param_sweep.csv
    results/phase2/cms_overcount_vs_epsilon.png
    results/phase2/cms_heavy_hitter_recall.png
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from experiments._common import RESULTS_DIR, banner, info, plt, save_plot, write_csv
from src.count_min_sketch import CountMinSketch
from src.data_pipeline import generate_zipf_stream
from src.exact_counters import exact_frequencies, exact_heavy_hitters


def evaluate(stream: list[str], epsilon: float, delta: float, threshold: float) -> dict:
    """Run CMS on ``stream``; compare to exact frequencies and heavy-hitters baseline.

    Returns:
        Metrics including max overcount, εN bound check, and heavy-hitter recall/precision.
    """
    cms = CountMinSketch(epsilon=epsilon, delta=delta)
    cms.process_stream(stream)
    dims = cms.get_dimensions()
    N = len(stream)  # stream length; ε·N is the CMS worst-case overcount bound scale

    truths = exact_frequencies(stream)
    overcounts: list[int] = []  # per-item (estimate − true), nonnegative for CMS
    rel_overs: list[float] = []  # overcount / N for each distinct item
    for item, truth in truths.items():
        est = cms.estimate(item)
        over = est - truth
        overcounts.append(over)
        rel_overs.append(over / N)
    max_over = max(overcounts) if overcounts else 0
    max_over_frac = max(rel_overs) if rel_overs else 0.0
    mean_over = statistics.fmean(overcounts) if overcounts else 0.0

    truth_hh = set(exact_heavy_hitters(stream, threshold).keys())
    cms_hh = set(cms.heavy_hitters(stream, threshold).keys())
    tp = len(truth_hh & cms_hh)  # true positives in heavy-hitter ID sets
    recall = tp / len(truth_hh) if truth_hh else float("nan")
    precision = tp / len(cms_hh) if cms_hh else float("nan")

    bound_holds = max_over <= epsilon * N

    return {
        "epsilon": epsilon,
        "delta": delta,
        "width": dims["width"],
        "depth": dims["depth"],
        "memory_cells": dims["memory_cells"],
        "stream_n": N,
        "max_overcount": max_over,
        "max_overcount_fraction": max_over_frac,
        "mean_overcount": mean_over,
        "epsilon_times_n": epsilon * N,
        "bound_holds": bound_holds,
        "true_heavy_hitters": len(truth_hh),
        "cms_heavy_hitters": len(cms_hh),
        "recall": recall,
        "precision": precision,
    }


def aggregate(runs: list[dict]) -> dict:
    """Average numeric fields across random seeds; replace per-run ``bound_holds`` with rate."""
    keys = runs[0].copy()
    keys["max_overcount_fraction"] = statistics.fmean(r["max_overcount_fraction"] for r in runs)
    keys["max_overcount"] = statistics.fmean(r["max_overcount"] for r in runs)
    keys["mean_overcount"] = statistics.fmean(r["mean_overcount"] for r in runs)
    keys["bound_holds_rate"] = sum(1 for r in runs if r["bound_holds"]) / len(runs)
    keys["recall"] = statistics.fmean(r["recall"] for r in runs if r["recall"] == r["recall"])
    keys["precision"] = statistics.fmean(r["precision"] for r in runs if r["precision"] == r["precision"])
    keys.pop("bound_holds", None)
    return keys


def plot_overcount(rows: list[dict]) -> Path:
    """Log–log plot: empirical max(overcount)/N vs ε, dashed y = ε reference."""
    fig, ax = plt.subplots(figsize=(8, 5))
    by_delta: dict[float, list[dict]] = {}
    for r in rows:
        by_delta.setdefault(r["delta"], []).append(r)
    for delta, items in sorted(by_delta.items()):
        items.sort(key=lambda x: x["epsilon"])
        eps = [x["epsilon"] for x in items]
        emp = [x["max_overcount_fraction"] for x in items]
        ax.plot(eps, emp, marker="o", label=f"empirical δ={delta}")
    eps_line = sorted({r["epsilon"] for r in rows})
    ax.plot(eps_line, eps_line, linestyle="--", color="black", label="theoretical bound = ε")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ε")
    ax.set_ylabel("Max overcount / N")
    ax.set_title("CMS: empirical max overcount vs theoretical bound")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    return save_plot(fig, "cms_overcount_vs_epsilon.png")


def plot_recall(rows: list[dict]) -> Path:
    """Plot heavy-hitter recall and precision vs ε for each δ curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    by_delta: dict[float, list[dict]] = {}
    for r in rows:
        by_delta.setdefault(r["delta"], []).append(r)
    for delta, items in sorted(by_delta.items()):
        items.sort(key=lambda x: x["epsilon"])
        eps = [x["epsilon"] for x in items]
        recall = [x["recall"] for x in items]
        precision = [x["precision"] for x in items]
        ax.plot(eps, recall, marker="o", label=f"recall δ={delta}")
        ax.plot(eps, precision, marker="x", linestyle=":", label=f"precision δ={delta}")
    ax.set_xscale("log")
    ax.set_xlabel("ε")
    ax.set_ylabel("Heavy-hitter recall / precision (≥1%)")
    ax.set_title("CMS heavy hitters: recall and precision vs ε")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    return save_plot(fig, "cms_heavy_hitter_recall.png")


def main() -> None:
    """Grid search ε × δ over Zipf streams; write CSV and two figures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--cardinality", type=int, default=5_000)
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.01)
    args = ap.parse_args()

    epsilons = [0.1, 0.05, 0.01, 0.005, 0.001]
    deltas = [0.1, 0.05, 0.01]
    seeds = list(range(args.seeds))

    banner(
        f"CMS sweep: ε={epsilons} δ={deltas} n={args.n} card={args.cardinality} α={args.alpha} seeds={args.seeds}"
    )
    aggregated_rows: list[dict] = []
    for eps in epsilons:
        for delta in deltas:
            runs = []
            for seed in seeds:
                stream = generate_zipf_stream(
                    n=args.n, alpha=args.alpha, cardinality=args.cardinality, seed=seed
                )
                runs.append(evaluate(stream, eps, delta, args.threshold))
            agg = aggregate(runs)
            aggregated_rows.append(agg)
            info(
                f"ε={eps:<6} δ={delta:<5} cells={agg['memory_cells']:>7,}  "
                f"maxOver/N={agg['max_overcount_fraction']*100:6.3f}%  recall={agg['recall']*100:5.1f}%  "
                f"precision={agg['precision']*100:5.1f}%  boundHoldRate={agg['bound_holds_rate']*100:.0f}%"
            )

    csv_path = RESULTS_DIR / "cms_param_sweep.csv"
    fieldnames = [
        "epsilon",
        "delta",
        "width",
        "depth",
        "memory_cells",
        "stream_n",
        "max_overcount",
        "max_overcount_fraction",
        "mean_overcount",
        "epsilon_times_n",
        "true_heavy_hitters",
        "cms_heavy_hitters",
        "recall",
        "precision",
        "bound_holds_rate",
    ]
    write_csv(csv_path, aggregated_rows, fieldnames)
    info(f"wrote {csv_path}")

    info(f"wrote {plot_overcount(aggregated_rows)}")
    info(f"wrote {plot_recall(aggregated_rows)}")


if __name__ == "__main__":
    main()
