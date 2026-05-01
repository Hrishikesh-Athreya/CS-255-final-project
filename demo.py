import time
from src.data_pipeline import load_shakespeare_stream, generate_uniform_stream, generate_zipf_stream
from src.exact_counters import exact_distinct_count, exact_frequencies, exact_heavy_hitters
from src.flajolet_martin import FlajoletMartin
from src.hyperloglog import HyperLogLog
from src.count_min_sketch import CountMinSketch


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def demo_distinct_counting(stream, label):
    print(f"\n--- {label} ---")
    exact = exact_distinct_count(stream)
    print(f"  Stream length:        {len(stream):,}")
    print(f"  Exact distinct count: {exact:,}")

    fm = FlajoletMartin(num_hashes=128, num_groups=8)
    t0 = time.perf_counter()
    fm.process_stream(stream)
    fm_time = time.perf_counter() - t0
    fm_est = fm.estimate()
    fm_err = abs(fm_est - exact) / exact * 100
    print(f"  FM estimate:          {fm_est:,.0f}  (error: {fm_err:.2f}%, time: {fm_time:.3f}s)")

    hll = HyperLogLog(p=14)
    t0 = time.perf_counter()
    hll.process_stream(stream)
    hll_time = time.perf_counter() - t0
    hll_est = hll.estimate()
    hll_err = abs(hll_est - exact) / exact * 100
    print(f"  HLL estimate (p=14):  {hll_est:,.0f}  (error: {hll_err:.2f}%, time: {hll_time:.3f}s)")


def demo_cms(stream, label):
    print(f"\n--- {label} ---")
    exact_freq = exact_frequencies(stream)
    exact_hh = exact_heavy_hitters(stream, threshold=0.01)

    cms = CountMinSketch(epsilon=0.01, delta=0.05)
    dims = cms.get_dimensions()
    print(f"  CMS config: ε={dims['epsilon']}, δ={dims['delta']} → {dims['width']}×{dims['depth']} table ({dims['memory_cells']} cells)")

    t0 = time.perf_counter()
    cms.process_stream(stream)
    cms_time = time.perf_counter() - t0
    print(f"  Processing time: {cms_time:.3f}s")

    top5 = sorted(exact_freq.items(), key=lambda x: -x[1])[:5]
    print(f"\n  {'Item':<20} {'Exact':>8} {'CMS Est':>8} {'Error':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for item, count in top5:
        est = cms.estimate(item)
        err = est - count
        print(f"  {item:<20} {count:>8,} {est:>8,} {'+' + str(err) if err > 0 else str(err):>8}")

    print(f"\n  Exact heavy hitters (≥1%): {len(exact_hh)} items")
    cms_hh = cms.heavy_hitters(stream, threshold=0.01)
    print(f"  CMS heavy hitters  (≥1%): {len(cms_hh)} items")
    if exact_hh:
        overlap = set(exact_hh.keys()) & set(cms_hh.keys())
        print(f"  Overlap (true positives): {len(overlap)} / {len(exact_hh)}")


if __name__ == "__main__":
    section("DISTINCT COUNTING")

    print("\nLoading Shakespeare dataset...")
    shakespeare = load_shakespeare_stream()
    demo_distinct_counting(shakespeare, "Shakespeare (Complete Works)")

    uniform = generate_uniform_stream(100_000, cardinality=10_000)
    demo_distinct_counting(uniform, "Synthetic Uniform (n=100k, card=10k)")

    zipf = generate_zipf_stream(100_000, alpha=1.5, cardinality=5_000)
    demo_distinct_counting(zipf, "Synthetic Zipf (n=100k, α=1.5, card=5k)")

    section("COUNT-MIN SKETCH — Frequency & Heavy Hitters")

    demo_cms(shakespeare, "Shakespeare (Complete Works)")
    demo_cms(zipf, "Synthetic Zipf (n=100k, α=1.5, card=5k)")

    # section("CMS ε/δ PARAMETER IMPACT")
    # print(f"\n  {'ε':<8} {'δ':<8} {'Width':>8} {'Depth':>5} {'Cells':>10}")
    # print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*10}")
    # for eps in [0.1, 0.01, 0.001]:
    #     for delta in [0.1, 0.05, 0.01]:
    #         c = CountMinSketch(epsilon=eps, delta=delta)
    #         d = c.get_dimensions()
    #         print(f"  {eps:<8} {delta:<8} {d['width']:>8,} {d['depth']:>5} {d['memory_cells']:>10,}")

    print(f"\n{'='*60}")
    print("  Phase 1 demo complete.")
    print(f"{'='*60}")
