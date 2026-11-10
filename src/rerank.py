"""Exact L2 re-ranking of an approximate shortlist (used by the error-attribution diagnostic)."""
import numpy as np


def exact_rerank(xb, xq: np.ndarray, cand: np.ndarray, k: int) -> np.ndarray:
    """Re-score candidate ids with exact L2 and return the top-k ids per query."""
    nq, r = cand.shape
    safe = np.where(cand < 0, 0, cand)
    out = np.empty((nq, k), dtype=np.int64)
    for q in range(nq):
        v = xb[safe[q]]                                   # (r, d) gather from the memmap
        d = ((v - xq[q]) ** 2).sum(axis=1)
        d[cand[q] < 0] = np.inf
        top = np.argpartition(d, k - 1)[:k] if r > k else np.arange(r)
        out[q] = cand[q, top[np.argsort(d[top], kind="stable")]]
    return out
