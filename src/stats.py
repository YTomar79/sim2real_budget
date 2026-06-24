"""Inferential statistics for the budget-allocation sweep.

Reproduces every "significant"/CI claim in the paper from results/summary.csv.

For each load-bearing contrast we take the two cells' per-seed mean returns,
pair them by seed (the resampling unit), and bootstrap the mean paired
difference (default 20000 resamples) to get a 95% CI. A contrast is called
significant when the CI excludes zero. We also print a per-cell mean +/- SEM
table over seeds (the quantity shown as error bars in the figures).

Usage:
  python -m src.stats --summary results/summary.csv
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def _cell(df: pd.DataFrame, n: int, w: float) -> np.ndarray:
    """Per-seed mean returns for cell (n, w), ordered by seed (the pairing key)."""
    sub = df[(df.n == n) & (df.w == w)].sort_values("seed")
    return sub["eval_return_mean"].to_numpy()


def paired_bootstrap(a: np.ndarray, b: np.ndarray, B: int = 20000, seed: int = 0):
    """Bootstrap the mean of the seed-paired difference a - b. Returns dict."""
    assert len(a) == len(b), "cells must have the same seeds to pair"
    diff = a - b
    n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = min(1.0, 2 * min((boot >= 0).mean(), (boot <= 0).mean()))
    return {
        "mean_diff": float(diff.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "boot_p": float(p),
        "significant": bool(lo > 0 or hi < 0),
        "n_pairs": n,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=str, default="results/summary.csv")
    p.add_argument("--bootstrap", type=int, default=20000)
    args = p.parse_args()

    df = pd.read_csv(args.summary)
    ns = sorted(df.n.unique())
    ws = sorted(df.w.unique())

    g = df.groupby(["n", "w"])["eval_return_mean"].agg(["mean", "std", "count"])
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))

    print("Per-cell mean +/- SEM over seeds (rows n, cols w):")
    print("       " + "".join(f"  w={w:<9g}" for w in ws))
    for n in ns:
        cells = "  ".join(f"{g.loc[(n,w),'mean']:7.0f}+/-{g.loc[(n,w),'sem']:3.0f}" for w in ws)
        print(f"n={n:2d}: {cells}")

    def report(a_nw, b_nw, label):
        a = _cell(df, *a_nw); b = _cell(df, *b_nw)
        r = paired_bootstrap(a, b, B=args.bootstrap)
        star = "*" if r["significant"] else " "
        print(f"{star} {label:34s} d={r['mean_diff']:+7.1f}  "
              f"95%CI[{r['ci_lo']:+7.1f},{r['ci_hi']:+7.1f}]  p={r['boot_p']:.4f}")

    print("\nLoad-bearing paired contrasts (resampling unit = seed):")
    report((10, 0.0), (0, 0.0), "budget jump: n=10 vs n=0 (w=0)")
    report((0, 0.2), (0, 0.0), "n=0 best-width w=0.2 vs w=0")
    for n in ns:
        report((n, 0.0), (n, 0.5), f"widest hurts: w=0 vs w=0.5 (n={n})")
    for n in [5, 10, 25, 50]:
        report((n, 0.0), (n, 0.2), f"point est. vs narrow DR: w=0 vs w=0.2 (n={n})")

    print("\nMean identification error by budget n:",
          {int(k): round(v, 3) for k, v in df.groupby("n")["id_error"].mean().items()})


if __name__ == "__main__":
    main()
