"""Phase 2 orchestrator: run all experiments and write CSV + PNG to results/phase2/.

Usage:
    python3 -m experiments.run_all              # default sizes
    python3 -m experiments.run_all --quick      # smaller sizes for a fast pass
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(module: str, args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Run with smaller streams and fewer seeds")
    args = ap.parse_args()

    if args.quick:
        n = "20000"
        seeds_hll = "3"
        seeds_fm = "10"
        seeds_cms = "2"
    else:
        n = "100000"
        seeds_hll = "5"
        seeds_fm = "30"
        seeds_cms = "3"

    plan = [
        ("experiments.hll_precision_sweep", ["--n", n, "--cardinality", "10000", "--seeds", seeds_hll]),
        ("experiments.fm_variance", ["--n", n, "--cardinality", "10000", "--seeds", seeds_fm]),
        ("experiments.cms_param_sweep", ["--n", n, "--cardinality", "5000", "--seeds", seeds_cms]),
        ("experiments.runtime_comparison", ["--n", n]),
    ]
    for module, extra in plan:
        rc = run(module, extra)
        if rc != 0:
            print(f"!! {module} exited with {rc}", flush=True)
            return rc

    print("\nPhase 2 experiments complete. Artifacts under results/phase2/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
