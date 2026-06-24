"""Collect all per-run result.json files into a single tidy CSV.

Usage:
  python -m src.aggregate --output_root ./results/sweep --out ./results/summary.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    root = Path(args.output_root)
    rows = []
    for rp in sorted(root.glob("*/result.json")):
        try:
            d = json.loads(rp.read_text())
        except Exception as e:
            print(f"[warn] could not read {rp}: {e}")
            continue
        if d.get("status") != "ok":
            continue
        rows.append(
            {
                "n": d["n"],
                "w": d["w"],
                "seed": d["seed"],
                "id_error": d["id_error"],
                "eval_return_mean": d["eval_return_mean"],
                "eval_return_std": d["eval_return_std"],
                "theta_hat_mass": d["theta_hat"][0],
                "theta_hat_length": d["theta_hat"][1],
                "wall_time_s": d.get("wall_time_s"),
            }
        )

    if not rows:
        raise SystemExit(f"No completed runs found under {root}")

    df = pd.DataFrame(rows).sort_values(["n", "w", "seed"]).reset_index(drop=True)
    n_expected = config.total_jobs()
    print(f"Collected {len(df)} / {n_expected} runs.")
    missing = n_expected - len(df)
    if missing:
        print(f"[warn] {missing} runs missing or unfinished.")

    out = args.out or str(root.parent / "summary.csv")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}")

    # Quick mean-over-seeds pivot to eyeball the surface.
    pivot = df.groupby(["n", "w"])["eval_return_mean"].mean().unstack("w")
    print("\nMean zero-shot return  (rows = n budget, cols = w width):")
    print(pivot.round(1).to_string())


if __name__ == "__main__":
    main()
