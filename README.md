# Memory–Accuracy Trade-offs in Inverted-File Vector Search: IVF-Flat vs. IVF-PQ

CS 848 course project (University of Waterloo). The experiment isolates the effect of
Product Quantization **inside one IVF framework**: IVF-Flat and IVF-PQ (M = 16, 32, 64;
8-bit codes) share the same coarse quantizer, the same queries and the same search
protocol, and differ only in how vectors are stored in the inverted lists.

* **RQ1** – How much smaller is IVF-PQ than IVF-Flat for each code length?
* **RQ2** – How much Recall@K does PQ lose at the same `nprobe`?
* **RQ3** – Can a larger `nprobe` recover that recall, and at what latency cost?

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./scripts/download_gist1m.sh            # 2.74 GB from ftp.irisa.fr -> data/gist1m/
.venv/bin/python -m src.datasets        # validate files + brute-force ground-truth spot check
.venv/bin/python -m pytest              # unit + small FAISS tests (seconds)
```

## Running

```bash
.venv/bin/python scripts/run_all.py --smoke-test   # whole pipeline on a small subset (~15 s)
.venv/bin/python scripts/run_all.py                # full GIST1M experiment
```

Individual steps: `scripts/build_indexes.py`, `scripts/run_benchmark.py`,
`scripts/generate_plots.py [--run-id ID]`. Common flags: `--config`, `--smoke-test`,
`--threads`, `--nprobe`, `--pq-m`, `--repetitions`, `--rebuild`.
All experiment settings live in `config/default.yaml`.

### Final figures

The report figures in `results/figures/final/` come from run `gist1m_20261102_151830`
(5 repetitions); run `gist1m_20261102_151037` is an independent repeat used only to
check latency stability.

```bash
.venv/bin/python scripts/compute_exact1nn_hit.py --run-id gist1m_20261102_151830
.venv/bin/python scripts/generate_final_figures.py --run-id gist1m_20261102_151830 \
    --comparison-run gist1m_20261102_151037
```

`compute_exact1nn_hit.py` re-searches the cached indexes (no timing) to get Exact-1NN
Hit@R, after verifying that it reproduces the run's recorded Recall@1/10/100 exactly.
`results/figures/final/README.md` documents the setup and the latency caveat, and
`plotted_values.csv` lists every plotted value with its source column.

## Dataset (GIST1M, TEXMEX)

| file | vectors | notes |
|---|---|---|
| `gist_base.fvecs` | 1,000,000 × 960 | database |
| `gist_learn.fvecs` | 500,000 × 960 | first **100,000** rows used for training |
| `gist_query.fvecs` | **1,000** × 960 | all queries used |
| `gist_groundtruth.ivecs` | 1,000 × 100 | exact L2 neighbours (supplied; spot-checked) |

The project brief originally listed 100k learn / 10k query vectors; those are the SIFT1M
sizes. The loader validates the actual GIST1M sizes above.

## Method

* **Shared coarse quantizer.** k-means (nlist = 1024, 20 iterations, seed 1234) is
  trained once on the 100k training vectors and cached (`indexes/coarse_nlist1024.npy`).
  All four indexes are built on a flat quantizer holding exactly those centroids, so
  inverted-list assignments are identical and recall differences are due to PQ alone.
  `training_time_s` = shared k-means time + the index's own training (PQ codebooks on
  residuals; zero for IVF-Flat).
* **Index cache.** Each `indexes/<name>.faiss` has a `<name>.json` sidecar with the spec
  it was built from (dataset fingerprint, nlist, M, nbits, seed, …). A cached index is
  reused only if the spec matches; otherwise the run stops (`--rebuild` to overwrite).
* **Search.** One search with k = 100 per query; Recall@1/10/100 all come from it.
  Recall@K = |top-K returned ∩ top-K ground truth| / K, averaged over queries. FAISS
  sentinel ids (−1, when the probed lists hold < k vectors) count as misses.
* **Throughput (QPS).** All 1,000 queries in one `search` call, 4 FAISS threads.
  1 warm-up pass, then 3 timed repetitions; QPS = 1000 / wall time, median reported.
* **Latency (ms/query).** One query per `search` call (FAISS IVF then runs on one thread),
  each call timed with `perf_counter_ns`; 1 untimed warm-up pass over all queries, then 3 timed
  passes over all 1,000 queries. p50/p95/p99/mean are computed per repetition and the
  median across repetitions is reported (p95 min/max across reps kept).
  Python call overhead (a few µs) is included and identical for every index.
* **Sanity checks recorded per configuration:** results identical across repetitions;
  single-query vs batch top-100 (IVF-PQ may permute exact distance ties; recall
  is reported for both modes); fraction of queries with missing ids;
  swap-in bytes during timed search.
* **Memory.** Theoretical payload: IVF-Flat 4d + 8 = 3848 B/vector; IVF-PQ M·nbits/8 + 8
  B/vector. Measured: serialized `.faiss` file size. Compression ratio =
  size(IVF-Flat) / size(IVF-PQ); saving = 1 − size(IVF-PQ)/size(IVF-Flat).
  IVF-PQ additionally allocates residual lookup tables in RAM at load time
  (nlist·M·256·4 bytes: 16 / 32 / 64 MiB) which are not in the file; reported as
  `precomputed_table_bytes`.

## Outputs

```
results/raw/<run_id>_indexes.csv      size, build/training times, per index
results/raw/<run_id>_recall.csv       recall + sanity checks, per (index, nprobe)
results/raw/<run_id>_throughput.csv   one row per repetition
results/raw/<run_id>_latency.csv.gz   one row per (index, nprobe, rep, query)
results/raw/<run_id>_metadata.json    versions, CPU, threads, full config, seed, git hash
results/summary/<run_id>_summary.csv  one row per (index, nprobe) – main table
results/summary/<run_id>_vs_ivfflat.csv       RQ2: PQ vs Flat at equal nprobe
results/summary/<run_id>_recall_matched.csv   RQ3: smallest PQ nprobe reaching Flat recall
results/summary/<run_id>_exact1nn_hit100.csv  Exact-1NN Hit@R (compute_exact1nn_hit.py)
results/figures/<run_id>/             fig1–fig5 (PDF + PNG)
results/figures/final/                report figures (PDF + PNG), README, plotted_values.csv
```

`generate_plots.py` re-aggregates from the raw files, so summaries can be recomputed
without re-running searches.

Only `results/figures/final/` is version-controlled. Raw measurements, summaries,
indexes, logs and the dataset are generated locally and excluded by `.gitignore`.

## Layout

```
config/default.yaml   all experiment parameters
src/datasets.py       .fvecs/.ivecs loaders + validation, smoke-test data
src/indexes.py        shared quantizer, IVF-Flat / IVF-PQ build, cache + validation
src/metrics.py        Recall@K, latency stats, Pareto mask
src/memory.py         theoretical vs serialized size
src/benchmark.py      timed search runs, raw output, aggregation, comparison tables
src/plotting.py       figures
scripts/              download, build, benchmark, plots, run_all,
                      compute_exact1nn_hit, generate_final_figures
tests/                dataset, metric and small-FAISS tests
```
