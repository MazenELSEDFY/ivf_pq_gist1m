"""Query benchmark: recall, throughput (QPS) and per-query latency for every
(index, nprobe) pair, with raw per-repetition output and a separate aggregation step.

Measurement protocol (identical for every index):
  * FAISS threads fixed to cfg.faiss.threads for the whole run.
  * One search with k = cfg.search.k per query; Recall@{1,10,100} all come from it.
  * Throughput mode: the query set is searched in batches of `throughput_batch_size`
    (null = all queries in one call). Wall time of the whole workload -> QPS.
  * Latency mode: queries are searched `latency_batch_size` (default 1) at a time,
    each call timed with perf_counter_ns -> per-query latency distribution in ms.
    With one query per call FAISS IVF search runs on a single thread.
  * `warmup_runs` untimed passes precede the timed passes of each mode.
  * `repetitions` timed passes; every pass is stored in results/raw.
Index training/construction is never inside a timed region.
"""
from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import psutil

from .datasets import Dataset
from .indexes import build_or_load, configured_indexes, index_path, set_nprobe
from .memory import add_relative_columns, memory_record
from .metrics import latency_stats_ms, pareto_mask, recalls
from .utils import resolve, run_metadata

log = logging.getLogger(__name__)

# Latency warm-up = one full pass over all queries (same lists touched as a timed pass).
LATENCY_WARMUP_QUERIES = None


def label(index_type: str, m: int | None) -> str:
    return "IVF-Flat" if index_type == "ivfflat" else f"IVF-PQ M={m}"


# --------------------------------------------------------------------------- timing kernels

def _search_batched(index: faiss.Index, xq: np.ndarray, k: int, bs: int) -> np.ndarray:
    out = np.empty((xq.shape[0], k), dtype=np.int64)
    for s in range(0, xq.shape[0], bs):
        out[s:s + bs] = index.search(xq[s:s + bs], k)[1]
    return out


def throughput_runs(index, xq, k, batch_size, warmup, reps) -> tuple[list[float], np.ndarray, bool]:
    """Return (wall seconds per repetition, ids from the first timed rep, all reps identical)."""
    bs = batch_size or xq.shape[0]
    for _ in range(warmup):
        _search_batched(index, xq, k, bs)
    times, first, identical = [], None, True
    for _ in range(reps):
        t0 = time.perf_counter()
        ids = _search_batched(index, xq, k, bs)
        times.append(time.perf_counter() - t0)
        if first is None:
            first = ids
        elif not np.array_equal(first, ids):
            identical = False
    return times, first, identical


def latency_runs(index, xq, k, batch_size, warmup, reps) -> tuple[np.ndarray, np.ndarray]:
    """Return (reps x nq latency matrix in ms/query, ids from the first timed rep)."""
    nq, bs = xq.shape[0], batch_size
    for _ in range(warmup):
        for s in range(0, min(LATENCY_WARMUP_QUERIES or nq, nq), bs):
            index.search(xq[s:s + bs], k)
    lat = np.empty((reps, nq), dtype=np.float64)
    ids = np.empty((nq, k), dtype=np.int64)
    pc = time.perf_counter_ns
    for r in range(reps):
        for s in range(0, nq, bs):
            q = xq[s:s + bs]
            t0 = pc()
            _, I = index.search(q, k)
            lat[r, s:s + bs] = (pc() - t0) / 1e6 / q.shape[0]
            if r == 0:
                ids[s:s + bs] = I
    return lat, ids


# --------------------------------------------------------------------------- main loop

def run(cfg: dict[str, Any], ds: Dataset, run_id: str, rebuild: bool = False) -> dict[str, Path]:
    s = cfg["search"]
    k, reps, warm = s["k"], s["repetitions"], s["warmup_runs"]
    tp_bs, lat_bs = s["throughput_batch_size"], s["latency_batch_size"]
    nprobes = [int(p) for p in cfg["ivf"]["nprobe"]]
    recall_ks = [int(x) for x in cfg["metrics"]["recall_k"]]
    nlist, nbits = cfg["ivf"]["nlist"], cfg["pq"]["nbits"]
    n_base, nq = ds.xb.shape[0], ds.xq.shape[0]

    faiss.omp_set_num_threads(cfg["faiss"]["threads"])
    threads = faiss.omp_get_max_threads()
    assert threads == cfg["faiss"]["threads"], (threads, cfg["faiss"]["threads"])
    assert max(recall_ks) <= k <= ds.gt.shape[1]
    assert nprobes == sorted(nprobes) and max(nprobes) <= nlist
    xq = np.ascontiguousarray(ds.xq, dtype=np.float32)

    raw_dir = resolve(cfg["paths"]["results_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    tp_rows, lat_frames, rec_rows, idx_rows = [], [], [], []

    for index_type, m in configured_indexes(cfg):
        if m is not None:
            assert ds.d % m == 0, f"{ds.d} % {m} != 0"
        index, meta = build_or_load(cfg, ds, index_type, m, rebuild=rebuild)
        name = index_path(cfg, index_type, m).stem
        mem = memory_record(index, index_path(cfg, index_type, m), index_type, n_base, ds.d,
                            nlist, m, nbits if m else None)
        idx_rows.append({"index": name, "label": label(index_type, m), "index_type": index_type,
                         "nlist": nlist, "pq_m": m, "pq_nbits": nbits if m else None, **mem,
                         "coarse_train_time_s": meta["coarse_train_time_s"],
                         "encoder_train_time_s": meta["encoder_train_time_s"],
                         "training_time_s": meta["training_time_s"],
                         "add_time_s": meta["add_time_s"], "build_time_s": meta["build_time_s"],
                         "build_faiss_threads": meta["faiss_threads"],
                         "loaded_from_cache": meta["loaded_from_cache"]})

        swap_before = psutil.swap_memory()
        for p in nprobes:
            set_nprobe(index, p)
            log.info("Evaluating %s nprobe=%d", name, p)
            times, ids, identical = throughput_runs(index, xq, k, tp_bs, warm, reps)
            if not identical:
                log.warning("%s nprobe=%d: results differ between repetitions", name, p)
            valid = (ids >= -1) & (ids < n_base)
            assert valid.all(), f"{name}: returned ids out of range"
            r = recalls(ids, ds.gt, recall_ks)
            n_missing = int((ids < 0).sum())
            lat, ids_single = latency_runs(index, xq, k, lat_bs, warm, reps)
            mismatch = float((ids_single != ids).any(axis=1).mean())
            log.info("Search completed %s nprobe=%d: R@10=%.4f QPS(med)=%.1f p50=%.3f ms",
                     name, p, r.get(10, float("nan")), nq / float(np.median(times)),
                     float(np.median(np.percentile(lat, 50, axis=1))))

            rec_rows.append({"index": name, "label": label(index_type, m), "index_type": index_type,
                             "pq_m": m, "nprobe": p,
                             **{f"recall_at_{kk}": v for kk, v in r.items()},
                             "n_missing_ids": n_missing,
                             "frac_queries_with_missing": float((ids < 0).any(axis=1).mean()),
                             "reps_identical": identical,
                             "frac_queries_single_vs_batch_differ": mismatch,
                             "recall_at_10_single_query_mode": recalls(ids_single, ds.gt, [10])[10]
                             if 10 in recall_ks else None})
            for rep, t in enumerate(times):
                tp_rows.append({"index": name, "nprobe": p, "rep": rep, "n_queries": nq,
                                "batch_size": tp_bs or nq, "wall_time_s": t, "qps": nq / t,
                                "ms_per_query": 1000 * t / nq})
            lat_frames.append(pd.DataFrame({
                "index": name, "nprobe": p,
                "rep": np.repeat(np.arange(reps), nq), "query_id": np.tile(np.arange(nq), reps),
                "latency_ms": lat.ravel()}))
        swap_after = psutil.swap_memory()
        swapped_in = swap_after.sin - swap_before.sin
        idx_rows[-1]["swap_in_bytes_during_search"] = int(swapped_in)
        idx_rows[-1]["rss_bytes_during_search"] = psutil.Process().memory_info().rss
        if swapped_in > 0:
            log.warning("%s: %d bytes swapped in during timed search - latency may be inflated",
                        name, swapped_in)
        del index
        gc.collect()

    add_relative_columns(idx_rows)
    paths = {
        "indexes": raw_dir / f"{run_id}_indexes.csv",
        "recall": raw_dir / f"{run_id}_recall.csv",
        "throughput": raw_dir / f"{run_id}_throughput.csv",
        "latency": raw_dir / f"{run_id}_latency.csv.gz",
        "metadata": raw_dir / f"{run_id}_metadata.json",
    }
    pd.DataFrame(idx_rows).to_csv(paths["indexes"], index=False)
    pd.DataFrame(rec_rows).to_csv(paths["recall"], index=False)
    pd.DataFrame(tp_rows).to_csv(paths["throughput"], index=False)
    pd.concat(lat_frames).to_csv(paths["latency"], index=False)
    meta = run_metadata(cfg, run_id, {
        "dataset": ds.describe(), "faiss_threads_effective": threads,
        "throughput_batch_size": tp_bs or nq, "latency_batch_size": lat_bs,
        "latency_warmup_queries": LATENCY_WARMUP_QUERIES or nq, "k": k, "repetitions": reps,
        "warmup_runs": warm, "seed": cfg["seed"], "nprobe": nprobes,
        "outputs": {k_: str(v) for k_, v in paths.items()}})
    paths["metadata"].write_text(json.dumps(meta, indent=2, default=str))
    log.info("Raw results saved under %s (run_id=%s)", raw_dir, run_id)
    return paths


# --------------------------------------------------------------------------- aggregation

SUMMARY_COLUMNS = [
    "index_type", "label", "nlist", "nprobe", "pq_m", "pq_nbits", "pq_code_bytes",
    "recall_at_1", "recall_at_10", "recall_at_100",
    "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "latency_mean_ms",
    "latency_p95_ms_min_rep", "latency_p95_ms_max_rep",
    "qps", "qps_min_rep", "qps_max_rep",
    "training_time_s", "coarse_train_time_s", "encoder_train_time_s", "add_time_s", "build_time_s",
    "serialized_size_bytes", "serialized_size_mib", "theoretical_bytes_per_vector",
    "theoretical_total_bytes", "precomputed_table_bytes",
    "compression_ratio_vs_ivfflat", "memory_saving_percent",
    "frac_queries_with_missing", "frac_queries_single_vs_batch_differ",
    "faiss_threads", "batch_size", "latency_batch_size", "repetitions", "seed", "run_id", "timestamp",
]


def aggregate(raw_dir: str | Path, run_id: str, summary_dir: str | Path) -> Path:
    """Combine raw per-repetition files into one summary row per (index, nprobe).

    Latency: p50/p95/p99/mean are computed per repetition over all queries, then the
    median across repetitions is reported (min/max across reps kept for p95).
    QPS: median across repetitions (min/max kept).
    """
    raw_dir, summary_dir = Path(raw_dir), Path(summary_dir)
    meta = json.loads((raw_dir / f"{run_id}_metadata.json").read_text())
    idx = pd.read_csv(raw_dir / f"{run_id}_indexes.csv")
    rec = pd.read_csv(raw_dir / f"{run_id}_recall.csv")
    tp = pd.read_csv(raw_dir / f"{run_id}_throughput.csv")
    lat = pd.read_csv(raw_dir / f"{run_id}_latency.csv.gz")

    per_rep = (lat.groupby(["index", "nprobe", "rep"])["latency_ms"]
               .agg(p50=lambda a: np.percentile(a, 50), p95=lambda a: np.percentile(a, 95),
                    p99=lambda a: np.percentile(a, 99), mean="mean").reset_index())
    lat_s = per_rep.groupby(["index", "nprobe"]).agg(
        latency_p50_ms=("p50", "median"), latency_p95_ms=("p95", "median"),
        latency_p99_ms=("p99", "median"), latency_mean_ms=("mean", "median"),
        latency_p95_ms_min_rep=("p95", "min"), latency_p95_ms_max_rep=("p95", "max")).reset_index()
    tp_s = tp.groupby(["index", "nprobe"]).agg(
        qps=("qps", "median"), qps_min_rep=("qps", "min"), qps_max_rep=("qps", "max"),
        batch_size=("batch_size", "first")).reset_index()

    df = (rec.merge(lat_s, on=["index", "nprobe"]).merge(tp_s, on=["index", "nprobe"])
          .merge(idx.drop(columns=["label", "index_type", "pq_m"]), on="index"))
    df["faiss_threads"] = meta["faiss_threads_effective"]
    df["latency_batch_size"] = meta["latency_batch_size"]
    df["repetitions"] = meta["repetitions"]
    df["seed"] = meta["seed"]
    df["run_id"] = run_id
    df["timestamp"] = meta["timestamp"]
    order = {n: i for i, n in enumerate(idx["index"])}
    df = df.sort_values(["index", "nprobe"], key=lambda c: c.map(order) if c.name == "index" else c)
    cols = [c for c in SUMMARY_COLUMNS if c in df.columns]
    out = df[cols + [c for c in df.columns if c not in cols and c != "index"]]

    # Optional Pareto flag: maximise Recall@10, minimise p95 latency and serialized size.
    out = out.assign(pareto_optimal=pareto_mask(out["recall_at_10"], out["latency_p95_ms"],
                                                out["serialized_size_bytes"]))

    summary_dir.mkdir(parents=True, exist_ok=True)
    comparisons(out).to_csv(summary_dir / f"{run_id}_vs_ivfflat.csv", index=False)
    matching(out).to_csv(summary_dir / f"{run_id}_recall_matched.csv", index=False)
    path = summary_dir / f"{run_id}_summary.csv"
    out.to_csv(path, index=False)
    out.to_csv(summary_dir / "latest_summary.csv", index=False)
    log.info("Summary saved: %s (%d rows)", path, len(out))
    return path


def comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """RQ2: every IVF-PQ config against IVF-Flat at the same nprobe."""
    flat = df[df["index_type"] == "ivfflat"].set_index("nprobe")
    rows = []
    for _, r in df[df["index_type"] == "ivfpq"].iterrows():
        f = flat.loc[r["nprobe"]]
        rows.append({"label": r["label"], "pq_m": r["pq_m"], "nprobe": r["nprobe"],
                     **{f"delta_recall_at_{k}_pts": 100 * (r[f"recall_at_{k}"] - f[f"recall_at_{k}"])
                        for k in (1, 10, 100) if f"recall_at_{k}" in r},
                     "p95_latency_ratio_vs_flat": r["latency_p95_ms"] / f["latency_p95_ms"],
                     "qps_ratio_vs_flat": r["qps"] / f["qps"],
                     "compression_ratio_vs_ivfflat": r["compression_ratio_vs_ivfflat"],
                     "memory_saving_percent": r["memory_saving_percent"]})
    return pd.DataFrame(rows)


def matching(df: pd.DataFrame, recall_col: str = "recall_at_10") -> pd.DataFrame:
    """RQ3: for each IVF-Flat operating point and each PQ index, the smallest nprobe whose
    recall reaches the Flat recall (if any), with the memory and latency of both."""
    rows = []
    flat = df[df["index_type"] == "ivfflat"].sort_values("nprobe")
    for _, f in flat.iterrows():
        for lab, g in df[df["index_type"] == "ivfpq"].groupby("label", sort=False):
            g = g.sort_values("nprobe")
            ok = g[g[recall_col] >= f[recall_col]]
            best = ok.iloc[0] if len(ok) else g.iloc[-1]
            rows.append({"flat_nprobe": f["nprobe"], f"flat_{recall_col}": f[recall_col],
                         "flat_p95_ms": f["latency_p95_ms"], "flat_size_mib": f["serialized_size_mib"],
                         "pq": lab, "pq_nprobe": best["nprobe"], f"pq_{recall_col}": best[recall_col],
                         "reaches_flat_recall": bool(len(ok)),
                         "pq_p95_ms": best["latency_p95_ms"], "pq_size_mib": best["serialized_size_mib"],
                         "p95_latency_ratio": best["latency_p95_ms"] / f["latency_p95_ms"],
                         "size_ratio_flat_over_pq": f["serialized_size_mib"] / best["serialized_size_mib"]})
    return pd.DataFrame(rows)
