#!/usr/bin/env bash
# Reproduce the study.
#
#   scripts/reproduce.sh figures   Rebuild every report figure and table from the versioned
#                                  measurements in results/raw + results/summary (seconds;
#                                  no dataset or indexes needed).
#   scripts/reproduce.sh full      Re-run the whole experiment on this machine: download
#                                  GIST1M, validate, tests, build indexes, two benchmark runs,
#                                  diagnostics, resident memory, OPQ appendix, figures.
#                                  Needs ~15 GB disk and ~12 GB RAM; about 1-2 h on 4 cores.
#                                  On a SLURM cluster use slurm/watgpu_full.sbatch instead.
#
# PYTHON selects the interpreter (default: .venv/bin/python if present, else python3).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-figures}
PY=${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Runs reported in docs/report_notes.md
MAIN=gist1m_20261117_200743      # watgpu main run
REPEAT=gist1m_20261117_201058    # watgpu independent repeat
MAC=gist1m_20261102_151830       # earlier Mac run
MAC_REPEAT=gist1m_20261102_151037

figures() {
  local run=$1 rep=$2 out=$3
  "$PY" scripts/generate_final_figures.py --run-id "$run" --comparison-run "$rep" --out "$out"
  if [ -f "results/summary/${run}_error_attribution.csv" ]; then
    "$PY" scripts/analyze_attribution.py --run-id "$run" --out "$out"
    "$PY" scripts/frontier.py --run-id "$run" --out "$out"
  fi
}

case "$MODE" in
  figures)
    figures "$MAIN" "$REPEAT" results/figures/final_watgpu
    "$PY" scripts/generate_final_figures.py --run-id "$MAC" --comparison-run "$MAC_REPEAT" \
        --out results/figures/final
    echo "Figures in results/figures/final_watgpu (report) and results/figures/final (Mac runs)."
    ;;
  full)
    [ -f data/gist1m/gist_base.fvecs ] || bash scripts/download_gist1m.sh
    "$PY" -m src.datasets
    "$PY" -m pytest -q
    "$PY" scripts/run_all.py --rebuild --repetitions 5
    RUN1=$(ls -t results/raw/gist1m_*_metadata.json | head -1 | xargs basename | sed 's/_metadata.json//')
    "$PY" scripts/run_benchmark.py --repetitions 5
    RUN2=$(ls -t results/raw/gist1m_*_metadata.json | head -1 | xargs basename | sed 's/_metadata.json//')
    [ "$RUN1" != "$RUN2" ]
    "$PY" scripts/compute_exact1nn_hit.py --run-id "$RUN1"
    "$PY" scripts/diagnose_error.py --run-id "$RUN1"
    "$PY" scripts/measure_resident_memory.py --run-id "$RUN1" --nprobe 32 --rerank 100 1000
    "$PY" scripts/opq_experiment.py --run-id "$RUN1"
    figures "$RUN1" "$RUN2" "results/figures/$RUN1"
    echo "New runs: $RUN1 (main), $RUN2 (repeat). Figures in results/figures/$RUN1."
    ;;
  *) echo "usage: $0 [figures|full]" >&2; exit 2 ;;
esac
