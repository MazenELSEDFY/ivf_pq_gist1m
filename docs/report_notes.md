# Report notes: proposal, feedback, deviations

Checklist for the final report, written after the watgpu runs. Final numbers come from
run `gist1m_20261110_003629` (repeat: `gist1m_20261110_004019`) on watgpu208,
Intel Xeon Gold 5415+, FAISS 1.15.1, 4 OpenMP threads, SLURM job 1681750.
Do not mix in the Mac runs of 2 Nov (`results/figures/final/`): recall matches,
latency does not.

## Deviations from the proposal (state them in the report)

| Proposal | What was done | Why |
|---|---|---|
| "100,000 training vectors, 10,000 queries" (Sec. 3.1) | learn file has 500,000 rows, first 100,000 used; **1,000 queries** | Those are the SIFT1M sizes; GIST1M ships 1,000 queries |
| "full 10,000-query set" (Sec. 3.2) | all 1,000 queries | same |
| nprobe in {1, 4, 16, 32} | main grid unchanged; diagnostics sweep 1..1024 | Feedback asked for a higher-nprobe diagnostic; Sec. 4 allows experiments that explain an observed result |
| Throughput "in fixed-size batches" | one batch of all 1,000 queries | state the batch size |
| Memory: theoretical + serialized size | plus resident memory (RSS) per index | Feedback point 4 |
| OPQ is a non-goal (Sec. 2, 3.5) | one confirmation experiment, appendix only | Supports the attribution; not a third method |
| Search stays in the compressed domain | exact re-ranking used as a **diagnostic** | It is the "exact vs PQ scoring over identical candidate pools" test the feedback asked for |
| nlist sensitivity (optional) | not run | dropped first, as planned |

## Scope labels

* **Proposal scope:** IVF-Flat and plain IVF-PQ, nprobe grid {1,4,16,32}. Main tables,
  frontier, p50/p95 latency, serialized size.
* **Diagnostic:** coverage vs PQ ranking (`*_error_attribution.csv`, `error_attribution.*`),
  exact re-ranking (`scope == "diagnostic"` rows of `*_matched_recall_v2.csv`,
  `recall_vs_qps_rerank.*`). Re-ranking needs the raw vectors: resident memory 2.9-3.8 GB,
  so it is not a memory-saving operating point unless the vectors stay on SSD.
* **Appendix:** OPQ (`appendix/opq_vs_pq.*`, `*_opq_vs_pq.csv`). OPQ lists differ from
  IVF-Flat for 8-12 of 1M vectors (float rounding after rotation); max coverage change 0.0002.

## Findings to write up (where each number lives)

1. **RQ1 compactness** (`*_summary.csv`, `index_size.*`): 3,673 MiB vs 28 / 43 / 73 MiB
   serialized (133x / 86x / 50x). Resident: 3,678 vs 46 / 78 / 140 MiB (FAISS precomputed
   residual tables are allocated at load and are not in the file).
2. **RQ2 equal nprobe** (`*_vs_ivfflat.csv`): at nprobe 32 Flat 0.912, PQ64 0.362,
   PQ32 0.266, PQ16 0.177.
3. **RQ3 more nprobe** (`error_attribution.*`, `*_attribution_table.csv`): coverage of the
   true top-10 is 0.80 at nprobe 16, 0.91 at 32, 1.0 at 1024, and IVF-Flat attains it
   exactly (at most 8 of 10,000 neighbor slots differ, all exact duplicate vectors at equal distance). PQ recall is flat from nprobe 16:
   PQ64 0.354 -> 0.364 even at nprobe = nlist. Up to ~nprobe 16 loss is IVF routing;
   beyond, it is PQ ranking error that nprobe cannot recover.
4. **Proof it is ranking, not candidates:** exact re-scoring of PQ's top-1000 on the same
   pool gives 0.873 / 0.963 / 0.992 at nprobe 1024 (M = 16 / 32 / 64).
5. **Matched recall, proposal scope** (`*_matched_recall_v2.csv`, scope = proposal):
   achievable only up to Recall@10 ~0.36. At 0.2: PQ32 nprobe 4 is 2.4x Flat's QPS;
   at 0.3: PQ64 nprobe 8 is 2.0x. No plain-PQ point reaches 0.5.
6. **Frontier** (Sec. 3.4; `frontier.*`, `*_frontier.csv`, `*_memory_budget.csv`):
   with three objectives 15/16 points are non-dominated; the one dominated point is
   IVF-Flat nprobe 1 (beaten by PQ64 nprobe 4 on recall, p95 and size). On recall vs p95
   latency PQ16 is never on the frontier, PQ32 only at nprobe 1-4, PQ64 from there to
   ~2 ms, then IVF-Flat. Answer to Sec. 3.3: the 64-byte code is worth its 2x storage
   over 32 bytes; the 16-byte code is not worth its extra compression.

## Threats to validity to keep

GIST1M descriptors and L2 only; nlist fixed at 1024; 4 threads may share physical cores
via hyper-threading on watgpu; matched-recall QPS for re-rank uses a single-threaded NumPy
re-rank loop (conservative); latency on a shared cluster node (two runs agree, see
`latency_stability.*`).
