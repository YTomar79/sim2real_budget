# How Should a Simulation-to-Reality Transfer Budget Be Spent?

**Syed Hamzah Rizvi¹**<sup>\*</sup> · **Yash Vardhan Tomar¹**<sup>\*</sup>

¹ Purdue University

[![arXiv](https://img.shields.io/badge/arXiv-2606.22062-b31b1b.svg)](https://arxiv.org/abs/2606.22062)
[![CI](https://github.com/YTomar79/sim2real_budget/actions/workflows/ci.yml/badge.svg)](https://github.com/YTomar79/sim2real_budget/actions/workflows/ci.yml)

<sub><sup>\*</sup> Co-first Authors with equal contribution</sub>

***

This repo contains the code, raw CSVs, and results for "How Should a Simulation-to-Reality Transfer Budget Be Spent?" (arXiv:2606.22062v1). 

Submitted to IEEE IROS First Workshop on Sim2Real and Classical Control 2026.

## Abstract
Simulation-to-reality transfer, often called sim-to-real transfer, is a central challenge in robot learning. Yet, the tradeoff between measuring a system more accurately and training over a broader range of simulated dynamics is still poorly understood. In this work, we focused on the allocation of real-robot measurement time between system identification and domain randomization. We studied this tradeoff in a controlled sim-to-sim pendulum setting, where a hidden-parameter model stands in for the physical robot, and the experiment sweeps identification rollouts against the width of the randomization distribution. Across the reality gaps and noise levels we tested, the measurement budget did most of the work. A small number of identification rollouts closed most of the transfer gap, and once any real data was available, policies performed best when trained at the estimated parameters rather than over a widened randomization band. Broad randomization that contained the true system still did not substitute for measurement. These results hold in a benign regime where the dynamics are identifiable and only two parameters are unknown, so structural model mismatch remains the setting where randomization breadth may become more valuable. Overall, our results suggest that sim-to-real pipelines should first measure the parameters they can and reserve randomization for the uncertainty that remains.

## Layout

```
sim2real_budget/
├── README.md                  this file
├── LICENSE                    MIT
├── requirements.txt           pip dependencies
├── environment.yml            equivalent conda environment (`sim2real`)
├── run_local.sh               quick local smoke test (tiny budget, "ideal" ID)
│
├── src/                       the experiment package
│   │
│   │   # — configuration (single source of truth) —
│   ├── config.py              sweep grid, physical constants, HParams,
│   │                          and the task_id <-> (n, w, seed) mapping
│   │
│   │   # — environment & methods —
│   ├── param_pendulum.py      Pendulum env with settable (mass, length);
│   │                          make_real_env() is the held-out "real" system
│   ├── dr_wrapper.py          domain-randomization reset wrapper (the `w` lever)
│   ├── system_id.py           estimate theta_hat from n real rollouts (the `n`
│   │                          lever): "grid" least-squares ID, or "ideal" ablation
│   │
│   │   # — run pipeline —
│   ├── train.py               one run: ID -> DR-train (SAC/PPO) -> eval -> result.json
│   ├── evaluate.py            zero-shot evaluation on the real system
│   ├── make_manifests.py      emit runner-agnostic command lists for the
│   │                          robustness experiments
│   │
│   │   # — analysis & figures —
│   ├── aggregate.py           collect per-run result.json files -> summary.csv
│   ├── stats.py               per-cell SEM + paired-bootstrap CIs for every claim
│   └── plot.py                render heatmap.png + pareto.png from a summary CSV
│
└── results/                   tracked: aggregated tables + figures
    │                          (raw per-run result.json / model.zip are gitignored)
    ├── summary.csv            tidy table for the main 250-run sweep
    ├── figures/               main-sweep figures
    │   ├── heatmap.png        mean zero-shot return over the (n, w) grid
    │   └── pareto.png         best width per budget vs. pure-breadth / -fidelity
    │
    │   # — robustness experiments (aggregated CSV + matching figures) —
    ├── exp_gap_1.5_1.2.csv    smaller reality gap, theta* = (1.5, 1.2)
    ├── exp_gap_2.5_2.0.csv    larger reality gap,  theta* = (2.5, 2.0)
    ├── fig_gap_1.5_1.2/       heatmap.png + pareto.png for the smaller gap
    ├── fig_gap_2.5_2.0/       heatmap.png + pareto.png for the larger gap
    ├── exp_noise_0.5.csv      lower sensor-noise sweep
    ├── exp_noise_2.0.csv      higher sensor-noise sweep
    ├── fig_noise_0.5/         heatmap.png + pareto.png for the lower-noise sweep
    ├── fig_noise_2.0/         heatmap.png + pareto.png for the higher-noise sweep
    └── exp_priorDR.csv        prior-range DR baseline (n=0, ignore the estimate)
```

## Replicating the experiments

Every command is run from the repository root. The full sweep is CPU-only and
embarrassingly parallel; on a single machine it runs sequentially as written.

**1. Set up the environment**

```bash
git clone https://github.com/YTomar79/sim2real_budget.git
cd sim2real_budget

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or, with conda:  conda env create -f environment.yml && conda activate sim2real
```

**2. Smoke test (optional, ~1 min)**

Verifies the pipeline end-to-end with tiny timesteps and the fast "ideal"
identifier before committing to the full sweep.

```bash
bash run_local.sh        # writes results/smoke_summary.csv + results/smoke_figures/
```

**3. Run the main 250-run sweep**

Each task decodes its own `(n, w, seed)` from the task id via
`config.decode_task_id`, so any executor (a shell loop, `xargs -P`, GNU parallel,
or a job scheduler) sweeps the grid without a separate index file. Re-running is
idempotent — a run whose `result.json` is `status: ok` is skipped, so a partial
sweep resumes safely.

```bash
N_JOBS=$(python -m src.config | sed -n 's/TOTAL_JOBS=//p')   # 250

# Sequential:
for TID in $(seq 0 $((N_JOBS - 1))); do
    python -m src.train --task_id "$TID" --output_root results/sweep
done

# ...or parallel across 8 workers:
# seq 0 $((N_JOBS - 1)) | xargs -P 8 -I{} \
#     python -m src.train --task_id {} --output_root results/sweep
```

**4. Aggregate, test, and plot**

```bash
python -m src.aggregate --output_root results/sweep --out results/summary.csv
python -m src.stats     --summary results/summary.csv                       # CIs for every claim
python -m src.plot      --summary results/summary.csv --outdir results/figures
```

`src.stats` prints the per-cell mean ± SEM table and a bootstrap CI for each
contrast quoted above; `src.plot` regenerates `heatmap.png` and `pareto.png`.

**5. Robustness experiments (optional)**

Generate the run lists, execute each, then aggregate (and plot the gap sweeps).

```bash
python -m src.make_manifests --outdir manifests        # writes manifests/*.txt

for NAME in exp_priorDR exp_gap_1.5_1.2 exp_gap_2.5_2.0 exp_noise_0.5 exp_noise_2.0; do
    while read -r ARGS; do
        python -m src.train $ARGS --output_root "results/$NAME"
    done < "manifests/$NAME.txt"
    python -m src.aggregate --output_root "results/$NAME" --out "results/$NAME.csv"
done

# Figures for the reality-gap sweeps:
python -m src.plot --summary results/exp_gap_1.5_1.2.csv --outdir results/fig_gap_1.5_1.2
python -m src.plot --summary results/exp_gap_2.5_2.0.csv --outdir results/fig_gap_2.5_2.0

# Figures for the sensor-noise sweeps:
python -m src.plot --summary results/exp_noise_0.5.csv --outdir results/fig_noise_0.5
python -m src.plot --summary results/exp_noise_2.0.csv --outdir results/fig_noise_2.0
```

## Reproducibility notes

- **Seeds.** Each run is fully seeded from its `(n, w, seed)`: the seed is
  threaded into `numpy`, `torch`, the SB3 model, environment resets, the
  domain-randomization RNG, and the system-identification rollouts. Same-seed
  reruns are bit-for-bit comparable, and re-running the sweep is idempotent.
- **Shipped data.** `results/summary.csv` holds all 250 per-run rows (one per
  `(n, w, seed)`), so the paired-bootstrap CIs in `src.stats` reproduce from the
  committed CSVs without re-running the sweep.
- **Exact environment.** `requirements.txt` / `environment.yml` use ranges for a
  fresh install. Because minor releases of the RL stack can shift numerical
  results, exact reproduction uses a pinned lockfile. Generate it once on the
  machine that ran the sweep and commit it:

  ```bash
  pip freeze > requirements.lock.txt   # then: git add requirements.lock.txt && git commit
  ```

  Others reproduce with `pip install -r requirements.lock.txt`.

## Citation

If you use this code or find this work helpful, please cite:

```bibtex
@article{rizvi-tomar2026sim2realbudget,
  title={How Should a Simulation-to-Reality Transfer Budget Be Spent?},
  author={Rizvi, Syed Hamzah and Tomar, Yash Vardhan},
  journal={arXiv preprint arXiv:2606.22062},
  year={2026}
}
```

## License

Released under the [MIT License](LICENSE).

