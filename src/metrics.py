"""Recall@K and latency summary statistics."""
from __future__ import annotations

import numpy as np


def per_query_recall(retrieved: np.ndarray, gt: np.ndarray, k: int) -> np.ndarray:
    """|retrieved[:, :k] ∩ gt[:, :k]| / k for every query.

    Vectorised in O(nq·k·log(nq·k)): row ids are offset into disjoint integer ranges so
    one np.isin over the flattened arrays does the per-row set intersection. FAISS
    sentinel ids (-1, returned when fewer than k candidates were scanned) never count.
    """
    retrieved = np.asarray(retrieved)
    gt = np.asarray(gt)
    if retrieved.ndim != 2 or gt.ndim != 2:
        raise ValueError("retrieved and gt must be 2-D")
    if retrieved.shape[0] != gt.shape[0]:
        raise ValueError(f"query count mismatch: {retrieved.shape[0]} vs {gt.shape[0]}")
    if retrieved.shape[1] < k or gt.shape[1] < k:
        raise ValueError(f"need at least {k} columns, got {retrieved.shape[1]} / {gt.shape[1]}")
    nq = retrieved.shape[0]
    r = retrieved[:, :k].astype(np.int64)
    g = gt[:, :k].astype(np.int64)
    span = int(max(r.max(initial=0), g.max(initial=0))) + 1
    offset = (np.arange(nq, dtype=np.int64) * span)[:, None]
    hits = np.isin(r + offset, g + offset) & (r >= 0)
    return hits.sum(axis=1) / k


def recall_at_k(retrieved: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Mean Recall@K over queries."""
    return float(per_query_recall(retrieved, gt, k).mean())


def recalls(retrieved: np.ndarray, gt: np.ndarray, ks) -> dict[int, float]:
    """Recall@K for several K from a single top-max(K) result."""
    return {int(k): recall_at_k(retrieved, gt, int(k)) for k in ks}


def latency_stats_ms(latencies_ms: np.ndarray) -> dict[str, float]:
    a = np.asarray(latencies_ms, dtype=np.float64)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean()),
            "min": float(a.min()), "max": float(a.max())}


def pareto_mask(recall, latency, memory) -> np.ndarray:
    """True where no other point has >= recall, <= latency, <= memory with one strict."""
    r, l, m = (np.asarray(v, dtype=float) for v in (recall, latency, memory))
    n = len(r)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dom = (r >= r[i]) & (l <= l[i]) & (m <= m[i]) & ((r > r[i]) | (l < l[i]) | (m < m[i]))
        keep[i] = not dom.any()
    return keep
