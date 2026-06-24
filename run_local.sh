#!/bin/bash
###############################################################################
# Local smoke test: run a few configs fast, then aggregate + plot.
# Uses the "ideal" system-ID mode and tiny timesteps so it finishes in a minute.
#
#   bash run_local.sh
###############################################################################
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

OUT="./results/smoke"
rm -rf "${OUT}"

# A handful of task ids spanning the grid corners + middle (grid is now 250 jobs).
for TID in 0 9 124 200 249; do
    python -m src.train \
        --task_id "${TID}" \
        --output_root "${OUT}" \
        --total_timesteps 3000 \
        --id_mode ideal \
        --n_eval_episodes 5 \
        --torch_threads 1
done

python -m src.aggregate --output_root "${OUT}" --out ./results/smoke_summary.csv
python -m src.plot --summary ./results/smoke_summary.csv --outdir ./results/smoke_figures

echo ""
echo "Smoke test OK. See results/smoke_summary.csv and results/smoke_figures/."
