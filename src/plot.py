"""Produce the workshop figures from summary.csv.

  1. heatmap.png  -- mean zero-shot return over the (n budget, w width) grid.
                     Best cell outlined in red; white cell separators; numbers
                     switch black/white for contrast.
  2. pareto.png   -- best width per budget vs pure-breadth / pure-fidelity refs.
                     w* labels placed to the side of high-error points and above
                     tight ones, so nothing overlaps.

Usage:
  python -m src.plot --summary ./results/summary.csv --outdir ./results/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update({
    "font.size": 17,
    "axes.titlesize": 19,
    "axes.labelsize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.dpi": 200,
})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./results/figures")
    args = p.parse_args()

    df = pd.read_csv(args.summary)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    g = (
        df.groupby(["n", "w"])["eval_return_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))

    ns = sorted(df["n"].unique())
    ws = sorted(df["w"].unique())

    # ---- Figure 1: heatmap ------------------------------------------------ #
    mat = np.full((len(ns), len(ws)), np.nan)
    semmat = np.full((len(ns), len(ws)), np.nan)
    for _, r in g.iterrows():
        mat[ns.index(r["n"]), ws.index(r["w"])] = r["mean"]
        semmat[ns.index(r["n"]), ws.index(r["w"])] = r["sem"]

    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis")
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)

    ax.set_xticks(range(len(ws)), [f"{w:g}" for w in ws])
    ax.set_yticks(range(len(ns)), [str(n) for n in ns])
    # white separators between cells
    ax.set_xticks(np.arange(-0.5, len(ws), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ns), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    ax.set_xlabel("domain-randomization width  w")
    ax.set_ylabel("real-data budget  n")

    for i in range(len(ns)):
        for j in range(len(ws)):
            if np.isnan(mat[i, j]):
                continue
            frac = (mat[i, j] - vmin) / (vmax - vmin + 1e-9)
            col = "black" if frac > 0.55 else "white"
            ax.text(j, i + 0.13, f"{mat[i, j]:.0f}", ha="center", va="center",
                    color=col, fontsize=15)
            ax.text(j, i - 0.17, f"$\\pm${semmat[i, j]:.0f}", ha="center",
                    va="center", color=col, fontsize=10.5)

    bi, bj = np.unravel_index(np.nanargmax(mat), mat.shape)
    ax.add_patch(patches.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                    edgecolor="red", lw=3, zorder=5))
    ax.text(0.0, 1.015, "red outline = best cell;  values: mean $\\pm$ SEM (seeds)",
            transform=ax.transAxes, color="red", fontsize=12.5, va="bottom", ha="left")

    cb = fig.colorbar(im, ax=ax, label="return", fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=15)
    fig.tight_layout()
    fig.savefig(outdir / "heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: best width per budget + reference lines ---------------- #
    best = []
    for n in ns:
        sub = g[g["n"] == n]
        row = sub.loc[sub["mean"].idxmax()]
        best.append((n, row["w"], row["mean"], row["sem"]))
    bn = pd.DataFrame(best, columns=["n", "w", "mean", "sem"])

    fig, ax = plt.subplots(figsize=(7.8, 5.3))
    ax.grid(axis="y", color="0.9", linewidth=1, zorder=0)

    pure_breadth = g[(g["n"] == min(ns)) & (g["w"] == max(ws))]
    pure_fidelity = g[(g["n"] == max(ns)) & (g["w"] == min(ws))]
    if len(pure_breadth):
        ax.axhline(pure_breadth["mean"].iloc[0], ls="--", color="gray", lw=1.5,
                   label=f"pure breadth (n={min(ns)}, w={max(ws):g})", zorder=1)
    if len(pure_fidelity):
        ax.axhline(pure_fidelity["mean"].iloc[0], ls=":", color="black", lw=1.8,
                   label=f"pure fidelity (n={max(ns)}, w=0)", zorder=1)

    ax.errorbar(bn["n"], bn["mean"], yerr=bn["sem"], marker="o", markersize=8,
                capsize=4, lw=2.2, color="#1f77b4",
                label="best width per budget", zorder=3)

    ref_vals = []
    if len(pure_breadth):
        ref_vals.append(pure_breadth["mean"].iloc[0])
    if len(pure_fidelity):
        ref_vals.append(pure_fidelity["mean"].iloc[0])
    lo = min([bn["mean"].min()] + ref_vals)
    hi = max([bn["mean"].max()] + ref_vals)
    span = hi - lo
    ax.set_ylim(lo - 0.10 * span, hi + 0.20 * span)

    # side-label points with big error bars, top-label tight ones
    for _, r in bn.iterrows():
        if r["sem"] > 15:
            ax.annotate(f"w*={r['w']:g}", (r["n"], r["mean"]),
                        textcoords="offset points", xytext=(12, 0),
                        ha="left", va="center", fontsize=15, color="#1f77b4")
        else:
            ax.annotate(f"w*={r['w']:g}", (r["n"], r["mean"]),
                        textcoords="offset points", xytext=(0, 14),
                        ha="center", va="bottom", fontsize=15, color="#1f77b4")

    ax.set_xlabel("real-data budget  n")
    ax.set_ylabel("best mean zero-shot return")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.margins(x=0.09)
    fig.tight_layout()
    fig.savefig(outdir / "pareto.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {outdir/'heatmap.png'} and {outdir/'pareto.png'}")


if __name__ == "__main__":
    main()
