# How Should a Simulation-to-Reality Transfer Budget Be Spent?

**Syed Hamzah Rizvi¹**<sup>\*</sup> · **Yash Vardhan Tomar¹**<sup>\*</sup>

¹ Purdue University

[![arXiv](https://img.shields.io/badge/arXiv-2606.22062-b31b1b.svg)](https://arxiv.org/abs/2606.22062)

<sub><sup>\*</sup> Equal contribution</sub>

***

This repo contains the code, raw CSVs, and results for "How Should a Simulation-to-Reality Transfer Budget Be Spent?" (arXiv:2606.22062v1). Submitted to IEEE IROS First Workshop on Sim2Real and Classical Control 2026.


## Abstract
Simulation-to-reality transfer, often called sim-to-real transfer, is a central challenge in robot learning. Yet, the tradeoff between measuring a system more accurately and training over a broader range of simulated dynamics is still poorly understood. In this work, we focused on the allocation of real-robot measurement time between system identification and domain randomization. We studied this tradeoff in a controlled sim-to-sim pendulum setting, where a hidden-parameter model stands in for the physical robot, and the experiment sweeps identification rollouts against the width of the randomization distribution. Across the reality gaps and noise levels we tested, the measurement budget did most of the work. A small number of identification rollouts closed most of the transfer gap, and once any real data was available, policies performed best when trained at the estimated parameters rather than over a widened randomization band. Broad randomization that contained the true system still did not substitute for measurement. These results hold in a benign regime where the dynamics are identifiable and only two parameters are unknown, so structural model mismatch remains the setting where randomization breadth may become more valuable. Overall, our results suggest that sim-to-real pipelines should first measure the parameters they can and reserve randomization for the uncertainty that remains.

## Layout

```
src/
  config.py          grid + constants + task_id <-> (n, w, seed) mapping (single source of truth)
  param_pendulum.py  Pendulum env with settable (mass, length)
  dr_wrapper.py      domain-randomization reset wrapper
  system_id.py       estimate theta_hat from n real rollouts (grid ID, or "ideal" ablation)
  train.py           one run: ID -> DR-train -> zero-shot eval -> result.json
  evaluate.py        zero-shot evaluation on the real system
  aggregate.py       collect result.json files -> summary.csv
  stats.py           per-cell SEM + paired-bootstrap CIs for the reported contrasts
  plot.py            heatmap.png + pareto.png
  make_manifests.py  emit run lists for the robustness experiments
run_local.sh         quick local smoke test
results/             aggregated CSVs + figures (raw per-run artifacts are gitignored)
```

## Quick start

```bash
pip install -r requirements.txt        # or: conda env create -f environment.yml
bash run_local.sh
```

`run_local.sh` runs a handful of configs with tiny timesteps and the fast "ideal"
identifier, then writes `results/smoke_summary.csv` and `results/smoke_figures/`.

## Reproducing the full results

```bash
# Run the 250-run grid (sequential; parallelize across the task ids as you like).
for TID in $(seq 0 $(($(python -m src.config | sed -n 's/TOTAL_JOBS=//p') - 1))); do
    python -m src.train --task_id "$TID" --output_root results/sweep
done

python -m src.aggregate --output_root results/sweep --out results/summary.csv
python -m src.stats     --summary results/summary.csv      # CIs for every claim
python -m src.plot      --summary results/summary.csv --outdir results/figures
```

Each run decodes its own `(n, w, seed)` from the task id via
`config.decode_task_id`, so any executor, such as, for example, a shell loop, GNU parallel, or a job
scheduler sweeps the grid without a separate index file. Re-running is
idempotent: a run whose `result.json` is `status: ok` is skipped, so a partial
sweep can be resumed safely.


## Grid and configuration

`n ∈ {0, 5, 10, 25, 50}` × `w ∈ {0, 0.1, 0.2, 0.3, 0.5}` × `seed ∈ {0..9}` =
250 runs. The learner is SAC (`total_timesteps = 75000`); the reality gap is
`theta* = (2.0, 1.5)` with sensor noise `1.0`. Everything is defined in
`src/config.py` — edit only that file to change the sweep.


Key overrides (CLI flags on `src.train`):

- `--total_timesteps` — learner budget per run. All reported numbers use 75000.
- `--id_mode` — `grid` (least-squares identification; the default behind every
  reported result) or `ideal` (a fast ablation that skips data collection;
  not used for any reported figure).
- `--theta_star`, `--id_obs_noise`, `--dr_mode` — vary the reality gap, sensor
  noise, and randomization scheme for the robustness experiments.
  

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

