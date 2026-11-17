#!/bin/bash
#SBATCH --job-name=ivfpq_full
#SBATCH --partition=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ivfpq_full_%j.out
# Full IVF-Flat vs IVF-PQ study on one watgpu node, so every reported number comes from
# the same machine:
#   1. validate dataset + unit tests
#   2. build indexes (fresh, shared coarse quantizer) + benchmark run 1 (5 reps) + plots
#   3. benchmark run 2 (independent repeat, cached indexes) for latency stability
#   4. Exact-1NN Hit@R, error attribution (coverage vs PQ ranking, rerank), resident memory
#   5. OPQ confirmation experiment (appendix)
#   6. report figures and tables -> results/figures/final_watgpu, results/summary
# FAISS uses 4 threads (config) inside an 8-CPU allocation, so OS/other work does not
# compete with the timed threads.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# /u801 is mounted noexec: compiled wheels must live on node-local /tmp.
PYDIR=/tmp/ivfpq_py_${SLURM_JOB_ID}
export TMPDIR=/tmp/ivfpq_tmp_${SLURM_JOB_ID}
trap 'rm -rf "$PYDIR" "$TMPDIR"' EXIT
mkdir -p "$PYDIR" "$TMPDIR" logs
pip3 install --quiet --target="$PYDIR" -r requirements.txt
export PYTHONPATH="$PYDIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="python3 -u"

echo "== node $(hostname), $(nproc) CPUs visible, $(date)"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket" || true
free -g

latest_run() { ls -t results/raw/gist1m_*_metadata.json | head -1 | xargs basename | sed 's/_metadata.json//'; }

echo "== 1. validate + tests"
$PY -m src.datasets
$PY -m pytest -q

echo "== 2. build + benchmark run 1"
$PY scripts/run_all.py --rebuild --repetitions 5
RUN1=$(latest_run); echo "RUN1=$RUN1"

echo "== 3. benchmark run 2 (repeat)"
$PY scripts/run_benchmark.py --repetitions 5
RUN2=$(latest_run); echo "RUN2=$RUN2"
[ "$RUN1" != "$RUN2" ]

echo "== 4. diagnostics on $RUN1"
$PY scripts/compute_exact1nn_hit.py --run-id "$RUN1"
$PY scripts/diagnose_error.py --run-id "$RUN1"
$PY scripts/measure_resident_memory.py --run-id "$RUN1" --nprobe 32 --rerank 100 1000

echo "== 5. OPQ confirmation experiment (appendix)"
$PY scripts/opq_experiment.py --run-id "$RUN1"

echo "== 6. final figures and tables"
$PY scripts/generate_final_figures.py --run-id "$RUN1" --comparison-run "$RUN2" \
    --out results/figures/final_watgpu
$PY scripts/analyze_attribution.py --run-id "$RUN1" --out results/figures/final_watgpu
$PY scripts/frontier.py --run-id "$RUN1" --out results/figures/final_watgpu
echo "== done $(date)  RUN1=$RUN1 RUN2=$RUN2"
