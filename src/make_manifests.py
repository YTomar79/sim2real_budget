"""Generate run manifests for the robustness experiments.

Each manifest is a plain-text file with one line per run; each line is the set of
`src.train` CLI args for that run (minus --output_root, which the caller
supplies). This decouples the experiment definition from the runner: feed the
lines to any executor — a shell loop, GNU parallel, xargs, or a job scheduler —
to sweep arbitrary (n, w, seed, theta_star, id_obs_noise, dr_mode) combinations.

Usage:
  python -m src.make_manifests --outdir manifests

Run a manifest locally (one job per line):
  while read -r ARGS; do
      python -m src.train $ARGS --output_root results/<name>
  done < manifests/<name>.txt
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

# (manifest filename, suggested output_root, list-of-arg-lines)
def _lines():
    exps = {}

    # Exp 1 — prior-range DR baseline: n=0, w=0.5, dr_mode=prior, 10 seeds.
    exps["exp_priorDR"] = (
        "results/exp_priorDR",
        [f"--n 0 --w 0.5 --seed {s} --dr_mode prior --id_mode grid"
         for s in range(10)],
    )

    # Exp 2 — gap-magnitude robustness: two gaps x {0,10,50} x {0,0.2,0.5} x 5 seeds.
    for ts in ["1.5,1.2", "2.5,2.0"]:
        name = f"exp_gap_{ts.replace(',', '_')}"
        lines = [
            f"--n {n} --w {w:g} --seed {s} --theta_star {ts} --id_mode grid"
            for n, w, s in itertools.product([0, 10, 50], [0.0, 0.2, 0.5], range(5))
        ]
        exps[name] = (f"results/{name}", lines)

    # Exp 3 — noise-level robustness: two noises x {0,5,10,25,50} x {0,0.5} x 5 seeds.
    for nz in ["0.5", "2.0"]:
        name = f"exp_noise_{nz}"
        lines = [
            f"--n {n} --w {w:g} --seed {s} --id_obs_noise {nz} --id_mode grid"
            for n, w, s in itertools.product([0, 5, 10, 25, 50], [0.0, 0.5], range(5))
        ]
        exps[name] = (f"results/{name}", lines)

    return exps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="manifests")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"{'manifest':28s} {'lines':>6s}  output_root")
    print("-" * 70)
    for name, (out_root, lines) in _lines().items():
        path = outdir / f"{name}.txt"
        path.write_text("\n".join(lines) + "\n")
        print(f"{str(path):28s} {len(lines):>6d}  {out_root}")
    print("\nRun each manifest with:")
    print("  while read -r ARGS; do")
    print("      python -m src.train $ARGS --output_root results/<name>")
    print("  done < manifests/<name>.txt")


if __name__ == "__main__":
    main()
