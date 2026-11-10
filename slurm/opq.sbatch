#!/bin/bash
#SBATCH --job-name=ivfopq
#SBATCH --partition=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ivfopq_%j.out
# OPQ vs PQ on identical inverted lists (scripts/opq_experiment.py), M = 16/32/64,
# against the watgpu full run so the numbers compare with its error attribution.
# Usage: sbatch slurm/opq.sbatch [RUN_ID]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN=${1:-gist1m_20261110_003629}

PYDIR=/tmp/ivfopq_py_${SLURM_JOB_ID}
export TMPDIR=/tmp/ivfopq_tmp_${SLURM_JOB_ID}
trap 'rm -rf "$PYDIR" "$TMPDIR"' EXIT
mkdir -p "$PYDIR" "$TMPDIR" logs
pip3 install --quiet --target="$PYDIR" -r requirements.txt
export PYTHONPATH="$PYDIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "== node $(hostname), $(nproc) CPUs visible, $(date), RUN=$RUN"
python3 -u scripts/opq_experiment.py --run-id "$RUN"
echo "== done $(date)"
