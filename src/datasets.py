"""Loading and validation of .fvecs/.ivecs datasets (GIST1M) plus smoke-test data.

File format (TEXMEX): each record is an int32 dimension `d` followed by `d` values
(float32 for .fvecs, int32 for .ivecs). Files are memory-mapped, so loading is O(N d)
I/O and the 3.84 GB base matrix is never copied in full: callers materialise it in
row batches with `iter_batches`.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .utils import resolve

log = logging.getLogger(__name__)


class DatasetError(ValueError):
    """Raised when a dataset file is missing, malformed or has unexpected shape."""


# --------------------------------------------------------------------------- raw I/O

def read_vecs(path: str | Path, kind: str, expected_dim: int | None = None,
              expected_n: int | None = None, check_headers: bool = True) -> np.ndarray:
    """Memory-map an .fvecs (kind='f') or .ivecs (kind='i') file and validate every record.

    Returns an (n, d) read-only view (float32 or int32). The view is not C-contiguous
    because of the per-record headers; use np.ascontiguousarray on row slices before
    passing data to FAISS.
    """
    path = Path(path)
    if kind not in ("f", "i"):
        raise ValueError("kind must be 'f' or 'i'")
    if not path.is_file():
        raise DatasetError(f"missing file: {path}")
    size = os.path.getsize(path)
    if size == 0 or size % 4:
        raise DatasetError(f"{path.name}: size {size} bytes is not a positive multiple of 4")

    raw = np.memmap(path, dtype=np.int32, mode="r")
    d = int(raw[0])
    if d <= 0:
        raise DatasetError(f"{path.name}: invalid dimension header {d}")
    rec = d + 1
    if raw.size % rec:
        raise DatasetError(f"{path.name}: {raw.size} int32 words not divisible by record "
                           f"length {rec} (truncated or malformed file)")
    n = raw.size // rec
    mat = raw.reshape(n, rec)
    # Scanning every header touches every page of the file (O(N d) I/O); callers that
    # only need shape/metadata of an already validated file can skip it.
    bad = np.flatnonzero(mat[:, 0] != d) if check_headers else np.empty(0, dtype=np.int64)
    if bad.size:
        raise DatasetError(f"{path.name}: {bad.size} records have a dimension header != {d} "
                           f"(first at row {bad[0]})")
    if expected_dim is not None and d != expected_dim:
        raise DatasetError(f"{path.name}: dimension {d}, expected {expected_dim}")
    if expected_n is not None and n != expected_n:
        raise DatasetError(f"{path.name}: {n} records, expected {expected_n}")

    data = mat[:, 1:]
    if kind == "f":
        data = data.view(np.float32)
    return data


def write_vecs(path: str | Path, arr: np.ndarray) -> None:
    """Write a 2-D float32 (.fvecs) or int32 (.ivecs) array. Used for tests."""
    arr = np.asarray(arr)
    if arr.dtype == np.float32:
        body = arr.view(np.int32)
    elif arr.dtype == np.int32:
        body = arr
    else:
        raise ValueError("array must be float32 or int32")
    n, d = arr.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = body
    out.tofile(path)


def check_finite(x: np.ndarray, name: str, batch: int = 100_000) -> None:
    """Verify all values are finite, scanning in row batches to bound memory."""
    for s in range(0, x.shape[0], batch):
        chunk = x[s:s + batch]
        if not np.isfinite(chunk).all():
            r = s + int(np.flatnonzero(~np.isfinite(chunk).all(axis=1))[0])
            raise DatasetError(f"{name}: non-finite value in row {r}")


def validate_groundtruth(gt: np.ndarray, n_queries: int, n_base: int, min_k: int) -> None:
    if gt.ndim != 2:
        raise DatasetError(f"ground truth must be 2-D, got shape {gt.shape}")
    if gt.shape[0] != n_queries:
        raise DatasetError(f"ground truth has {gt.shape[0]} rows, expected {n_queries} (one per query)")
    if gt.shape[1] < min_k:
        raise DatasetError(f"ground truth has {gt.shape[1]} neighbours per query, need >= {min_k}")
    lo, hi = int(gt.min()), int(gt.max())
    if lo < 0 or hi >= n_base:
        raise DatasetError(f"ground-truth ids out of range [0, {n_base}): min {lo}, max {hi}")
    s = np.sort(gt, axis=1)
    dup_rows = np.flatnonzero((s[:, 1:] == s[:, :-1]).any(axis=1))
    if dup_rows.size:
        raise DatasetError(f"ground truth has duplicate ids in {dup_rows.size} rows (first {dup_rows[0]})")


def iter_batches(x: np.ndarray, batch_size: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (start_row, C-contiguous float32 copy of rows) for streaming into FAISS."""
    for s in range(0, x.shape[0], batch_size):
        yield s, np.ascontiguousarray(x[s:s + batch_size], dtype=np.float32)


# --------------------------------------------------------------------------- dataset

@dataclass
class Dataset:
    xt: np.ndarray          # training vectors, contiguous float32 (n_train, d)
    xb: np.ndarray          # database vectors, possibly a memmap view (n_base, d)
    xq: np.ndarray          # queries, contiguous float32 (n_query, d)
    gt: np.ndarray          # exact neighbour ids, int64 (n_query, >=k)
    source: str             # "gist1m" | "gist1m_subset" | "synthetic"
    gt_source: str          # "supplied" | "computed_exact"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def d(self) -> int:
        return self.xb.shape[1]

    def describe(self) -> dict[str, Any]:
        return {"source": self.source, "gt_source": self.gt_source, "dim": self.d,
                "n_train": int(self.xt.shape[0]), "n_base": int(self.xb.shape[0]),
                "n_query": int(self.xq.shape[0]), "gt_k": int(self.gt.shape[1]), **self.meta}


def _paths(cfg: dict[str, Any]) -> dict[str, Path]:
    ds = cfg["dataset"]
    root = resolve(ds["dir"])
    return {k: root / v for k, v in ds["files"].items()}


def dataset_available(cfg: dict[str, Any]) -> bool:
    return all(p.is_file() for p in _paths(cfg).values())


def exact_groundtruth(xb: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    """Brute-force exact L2 k-NN (only for small smoke-test subsets)."""
    import faiss
    index = faiss.IndexFlatL2(xb.shape[1])
    for _, chunk in iter_batches(xb, 100_000):
        index.add(chunk)
    _, ids = index.search(np.ascontiguousarray(xq, dtype=np.float32), k)
    return ids.astype(np.int64)


def load_gist1m(cfg: dict[str, Any], check_values: bool = True) -> Dataset:
    """Load the full GIST1M dataset with all validation checks from the config."""
    ds = cfg["dataset"]
    exp = ds["expected"]
    d = ds["dim"]
    paths = _paths(cfg)
    log.info("Loading GIST1M from %s", resolve(ds["dir"]))

    xb = read_vecs(paths["base"], "f", d, exp["database_size"], check_headers=check_values)
    xl = read_vecs(paths["learn"], "f", d, exp["learn_size"])
    xq = read_vecs(paths["query"], "f", d, exp["query_size"])
    gt = read_vecs(paths["groundtruth"], "i", None, exp["query_size"])

    n_train = ds["train_size"]
    if n_train > xl.shape[0]:
        raise DatasetError(f"train_size {n_train} > learn set size {xl.shape[0]}")
    xt = np.ascontiguousarray(xl[:n_train], dtype=np.float32)
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    gt = np.ascontiguousarray(gt, dtype=np.int64)

    if check_values:
        for name, arr in (("train", xt), ("query", xq), ("base", xb)):
            check_finite(arr, name)
    validate_groundtruth(gt, xq.shape[0], xb.shape[0], cfg["search"]["k"])

    assert xb.shape[1] == xt.shape[1] == xq.shape[1] == d
    assert xb.dtype == xt.dtype == xq.dtype == np.float32
    out = Dataset(xt=xt, xb=xb, xq=xq, gt=gt, source="gist1m", gt_source="supplied",
                  meta={"paths": {k: str(v) for k, v in paths.items()},
                        "file_sizes": {k: v.stat().st_size for k, v in paths.items()},
                        "learn_file_rows": int(xl.shape[0]), "train_rows": f"learn[0:{n_train}]"})
    log.info("Dataset loaded: train=%s base=%s query=%s gt=%s",
             xt.shape, xb.shape, xq.shape, gt.shape)
    return out


def synthetic_clustered(n_train: int, n_base: int, n_query: int, d: int, seed: int,
                        n_clusters: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian-mixture data (non-negative-ish like GIST) for pipeline smoke tests."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d)).astype(np.float32)

    def draw(n: int) -> np.ndarray:
        lab = rng.integers(0, n_clusters, n)
        return (centers[lab] + 0.5 * rng.standard_normal((n, d))).astype(np.float32)

    return draw(n_train), draw(n_base), draw(n_query)


def load_smoke(cfg: dict[str, Any]) -> Dataset:
    """Small dataset for --smoke-test: a GIST1M prefix if available, else synthetic.

    Ground truth is always recomputed exactly here because the supplied GIST1M ground
    truth refers to the full 1M base, not to the subset.
    """
    s = cfg["smoke_test"]
    d = cfg["dataset"]["dim"]
    k = cfg["search"]["k"]
    if dataset_available(cfg):
        paths = _paths(cfg)
        xt = np.ascontiguousarray(read_vecs(paths["learn"], "f", d)[: s["train_size"]])
        xb = np.ascontiguousarray(read_vecs(paths["base"], "f", d)[: s["database_size"]])
        xq = np.ascontiguousarray(read_vecs(paths["query"], "f", d)[: s["query_size"]])
        source = "gist1m_subset"
    elif s.get("synthetic_if_missing", True):
        xt, xb, xq = synthetic_clustered(s["train_size"], s["database_size"], s["query_size"],
                                         d, cfg["seed"])
        source = "synthetic"
    else:
        raise DatasetError("GIST1M not found and synthetic fallback disabled")
    gt = exact_groundtruth(xb, xq, k)
    validate_groundtruth(gt, xq.shape[0], xb.shape[0], k)
    log.info("Smoke dataset (%s): train=%s base=%s query=%s, exact GT computed (k=%d)",
             source, xt.shape, xb.shape, xq.shape, k)
    meta: dict[str, Any] = {"train_rows": f"learn[0:{xt.shape[0]}]"}
    if source == "synthetic":
        meta["seed"] = cfg["seed"]
    else:
        meta["file_sizes"] = {k: v.stat().st_size for k, v in _paths(cfg).items()}
    return Dataset(xt=xt, xb=xb, xq=xq, gt=gt, source=source, gt_source="computed_exact", meta=meta)


def load_dataset(cfg: dict[str, Any]) -> Dataset:
    return load_smoke(cfg) if cfg.get("smoke") else load_gist1m(cfg)


def spot_check_groundtruth(ds: Dataset, n_queries: int = 20, k: int = 100, seed: int = 0) -> float:
    """Brute-force exact k-NN for a random sample of queries and compare with the supplied
    ground truth. Returns the mean overlap (1.0 = identical sets). Cost: n_queries full scans."""
    rng = np.random.default_rng(seed)
    qi = np.sort(rng.choice(ds.xq.shape[0], n_queries, replace=False))
    ids = exact_groundtruth(ds.xb, ds.xq[qi], k)
    overlap = np.mean([len(set(ids[j]) & set(ds.gt[q, :k])) / k for j, q in enumerate(qi)])
    log.info("Ground-truth spot check on %d queries: overlap %.4f, top-1 agreement %.4f",
             n_queries, overlap, float(np.mean(ids[:, 0] == ds.gt[qi, 0])))
    return float(overlap)


if __name__ == "__main__":  # python -m src.datasets  -> validate the full dataset
    from .utils import load_config, setup_logging
    setup_logging("validate_dataset")
    ds = load_gist1m(load_config())
    log.info("Validation passed: %s", ds.describe())
    assert spot_check_groundtruth(ds) > 0.99, "supplied ground truth disagrees with brute force"
