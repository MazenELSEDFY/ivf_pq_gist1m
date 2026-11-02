"""Build, cache and validate IVF-Flat / IVF-PQ indexes that share one coarse quantizer.

Design: k-means centroids (nlist of them) are trained once and cached. Every index is
built on top of a flat quantizer holding exactly those centroids, so IVF-Flat and all
IVF-PQ variants have identical inverted-list assignments and differ only in how the
vectors inside each list are stored. That isolates the effect of PQ compression.

Each index file `<name>.faiss` has a sidecar `<name>.json` with the spec it was built
from. A cached file is reused only if its spec matches the requested one exactly;
otherwise IncompatibleIndexError is raised (pass rebuild=True to overwrite).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from tqdm import tqdm

from .datasets import Dataset, iter_batches
from .utils import resolve

log = logging.getLogger(__name__)


class IncompatibleIndexError(RuntimeError):
    pass


# --------------------------------------------------------------------------- naming/specs

def index_name(index_type: str, nlist: int, m: int | None = None, nbits: int | None = None) -> str:
    if index_type == "ivfflat":
        return f"ivfflat_nlist{nlist}"
    if index_type == "ivfpq":
        return f"ivfpq_m{m}_nbits{nbits}_nlist{nlist}"
    raise ValueError(index_type)


def dataset_fingerprint(ds: Dataset) -> dict[str, Any]:
    fp = {"source": ds.source, "dim": ds.d, "n_train": int(ds.xt.shape[0]),
          "n_base": int(ds.xb.shape[0])}
    if "file_sizes" in ds.meta:
        fp["file_sizes"] = ds.meta["file_sizes"]
    if "train_rows" in ds.meta:
        fp["train_rows"] = ds.meta["train_rows"]
    if "seed" in ds.meta:
        fp["seed"] = ds.meta["seed"]
    return fp


def _metric(cfg: dict[str, Any]) -> int:
    if cfg["faiss"]["metric"].lower() != "l2":
        raise ValueError("only the L2 metric is supported in this study")
    return faiss.METRIC_L2


def coarse_spec(cfg: dict[str, Any], ds: Dataset) -> dict[str, Any]:
    return {"kind": "coarse_kmeans", "nlist": cfg["ivf"]["nlist"], "dim": ds.d,
            "metric": cfg["faiss"]["metric"], "seed": cfg["seed"],
            "kmeans_niter": cfg["ivf"]["kmeans_niter"], "dataset": dataset_fingerprint(ds)}


def index_spec(cfg: dict[str, Any], ds: Dataset, index_type: str, m: int | None) -> dict[str, Any]:
    spec = coarse_spec(cfg, ds)
    spec.update({"kind": "index", "index_type": index_type,
                 "pq_m": m if index_type == "ivfpq" else None,
                 "pq_nbits": cfg["pq"]["nbits"] if index_type == "ivfpq" else None})
    return spec


def _diff(a: dict, b: dict) -> list[str]:
    keys = sorted(set(a) | set(b))
    return [f"{k}: cached={a.get(k)!r} requested={b.get(k)!r}" for k in keys if a.get(k) != b.get(k)]


def _sha1(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- coarse quantizer

def get_coarse_centroids(cfg: dict[str, Any], ds: Dataset, rebuild: bool = False
                         ) -> tuple[np.ndarray, dict[str, Any]]:
    """Train (or load cached) k-means centroids for the shared coarse quantizer."""
    nlist = cfg["ivf"]["nlist"]
    out_dir = resolve(cfg["paths"]["indexes"])
    out_dir.mkdir(parents=True, exist_ok=True)
    npy, meta_path = out_dir / f"coarse_nlist{nlist}.npy", out_dir / f"coarse_nlist{nlist}.json"
    spec = coarse_spec(cfg, ds)

    if npy.exists() and meta_path.exists() and not rebuild:
        meta = json.loads(meta_path.read_text())
        if meta["spec"] != spec:
            raise IncompatibleIndexError("cached coarse centroids do not match config:\n  "
                                         + "\n  ".join(_diff(meta["spec"], spec)))
        cent = np.load(npy)
        assert cent.shape == (nlist, ds.d) and _sha1(cent) == meta["centroids_sha1"]
        log.info("Coarse centroids loaded from cache: %s", npy.name)
        return cent, meta

    n_train = ds.xt.shape[0]
    if n_train < 39 * nlist:
        log.warning("only %d training points for %d centroids (FAISS recommends >= %d)",
                    n_train, nlist, 39 * nlist)
    log.info("Coarse k-means training started: nlist=%d, n_train=%d, niter=%d, seed=%d",
             nlist, n_train, cfg["ivf"]["kmeans_niter"], cfg["seed"])
    clus = faiss.Clustering(ds.d, nlist)
    clus.seed = cfg["seed"]
    clus.niter = cfg["ivf"]["kmeans_niter"]
    q = faiss.IndexFlatL2(ds.d)
    t0 = time.perf_counter()
    clus.train(ds.xt, q)
    t = time.perf_counter() - t0
    cent = q.reconstruct_n(0, nlist).copy()
    log.info("Coarse k-means training completed in %.2f s", t)
    np.save(npy, cent)
    meta = {"spec": spec, "train_time_s": t, "centroids_sha1": _sha1(cent),
            "faiss_version": faiss.__version__, "faiss_threads": cfg["faiss"]["threads"],
            "created": _dt.datetime.now().isoformat(timespec="seconds")}
    _write_json(meta_path, meta)
    return cent, meta


# --------------------------------------------------------------------------- build / load

def _new_index(cfg: dict[str, Any], d: int, centroids: np.ndarray, index_type: str,
               m: int | None) -> faiss.Index:
    nlist = cfg["ivf"]["nlist"]
    quantizer = faiss.IndexFlatL2(d)
    quantizer.add(centroids)
    if index_type == "ivfflat":
        index = faiss.IndexIVFFlat(quantizer, d, nlist, _metric(cfg))
    else:
        assert d % m == 0, f"dimension {d} not divisible by M={m}"
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, cfg["pq"]["nbits"], _metric(cfg))
        index.pq.cp.seed = cfg["seed"]
    index.cp.seed = cfg["seed"]
    return index


def validate_index(index: faiss.Index, spec: dict[str, Any], coarse_sha1: str) -> None:
    """Check a built or loaded index against its spec. Raises AssertionError."""
    idx = faiss.downcast_index(index)
    n_base = spec["dataset"]["n_base"]
    assert idx.is_trained, "index is not trained"
    assert idx.d == spec["dim"], f"dim {idx.d} != {spec['dim']}"
    assert idx.nlist == spec["nlist"], f"nlist {idx.nlist} != {spec['nlist']}"
    assert idx.metric_type == faiss.METRIC_L2
    assert idx.ntotal == n_base, f"ntotal {idx.ntotal} != {n_base}"
    if spec["index_type"] == "ivfflat":
        assert isinstance(idx, faiss.IndexIVFFlat), type(idx)
    else:
        assert isinstance(idx, faiss.IndexIVFPQ), type(idx)
        assert idx.pq.M == spec["pq_m"] and idx.pq.nbits == spec["pq_nbits"]
        assert idx.code_size == spec["pq_m"] * spec["pq_nbits"] // 8
    cent = idx.quantizer.reconstruct_n(0, idx.nlist)
    assert _sha1(cent) == coarse_sha1, "index does not use the shared coarse centroids"


def build_or_load(cfg: dict[str, Any], ds: Dataset, index_type: str, m: int | None = None,
                  rebuild: bool = False) -> tuple[faiss.Index, dict[str, Any]]:
    """Return (index, metadata). Loads from cache when the stored spec matches."""
    nlist, nbits = cfg["ivf"]["nlist"], cfg["pq"]["nbits"]
    name = index_name(index_type, nlist, m, nbits)
    out_dir = resolve(cfg["paths"]["indexes"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path, meta_path = out_dir / f"{name}.faiss", out_dir / f"{name}.json"
    spec = index_spec(cfg, ds, index_type, m)

    if path.exists() and meta_path.exists() and not rebuild:
        meta = json.loads(meta_path.read_text())
        if meta["spec"] != spec:
            raise IncompatibleIndexError(f"cached {path.name} does not match config "
                                         "(use --rebuild to overwrite):\n  "
                                         + "\n  ".join(_diff(meta["spec"], spec)))
        t0 = time.perf_counter()
        index = faiss.read_index(str(path))
        validate_index(index, spec, meta["coarse_centroids_sha1"])
        log.info("Index loaded from cache: %s (%.1f s)", path.name, time.perf_counter() - t0)
        meta["loaded_from_cache"] = True
        return index, meta

    cent, cmeta = get_coarse_centroids(cfg, ds, rebuild=rebuild)
    index = _new_index(cfg, ds.d, cent, index_type, m)

    log.info("Index training started: %s", name)
    t0 = time.perf_counter()
    index.train(ds.xt)       # IVF-Flat: no-op (quantizer pre-trained); IVF-PQ: PQ codebooks
    encoder_train = time.perf_counter() - t0
    assert index.is_trained, "index must be trained before add()"
    log.info("Index training completed: %s (%.2f s, excl. shared k-means)", name, encoder_train)

    log.info("Index build (add) started: %s, %d vectors", name, ds.xb.shape[0])
    bs = cfg["build"]["add_batch_size"]
    add_time = 0.0
    for _, chunk in tqdm(iter_batches(ds.xb, bs), total=-(-ds.xb.shape[0] // bs),
                         desc=f"add {name}", leave=False):
        t0 = time.perf_counter()        # time only FAISS add, not reading the memmap
        index.add(chunk)
        add_time += time.perf_counter() - t0
    log.info("Index build completed: %s (add %.2f s), ntotal=%d", name, add_time, index.ntotal)
    validate_index(index, spec, cmeta["centroids_sha1"])

    tmp = path.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, path)
    training = cmeta["train_time_s"] + encoder_train
    meta = {"spec": spec, "name": name,
            "coarse_train_time_s": cmeta["train_time_s"],
            "encoder_train_time_s": encoder_train,
            "training_time_s": training,
            "add_time_s": add_time,
            "build_time_s": training + add_time,
            "coarse_centroids_sha1": cmeta["centroids_sha1"],
            "serialized_size_bytes": path.stat().st_size,
            "faiss_version": faiss.__version__, "faiss_threads": cfg["faiss"]["threads"],
            "add_batch_size": bs,
            "created": _dt.datetime.now().isoformat(timespec="seconds")}
    _write_json(meta_path, meta)
    log.info("Index saved: %s (%d bytes)", path.name, meta["serialized_size_bytes"])
    meta["loaded_from_cache"] = False
    return index, meta


def index_path(cfg: dict[str, Any], index_type: str, m: int | None = None) -> Path:
    name = index_name(index_type, cfg["ivf"]["nlist"], m, cfg["pq"]["nbits"])
    return resolve(cfg["paths"]["indexes"]) / f"{name}.faiss"


def configured_indexes(cfg: dict[str, Any]) -> list[tuple[str, int | None]]:
    """[(index_type, M)] in a fixed order: IVF-Flat first, then PQ by increasing M."""
    return [("ivfflat", None)] + [("ivfpq", int(m)) for m in sorted(cfg["pq"]["m"])]


def set_nprobe(index: faiss.Index, nprobe: int) -> None:
    ivf = faiss.extract_index_ivf(index)
    ivf.nprobe = int(nprobe)
    assert ivf.nprobe == nprobe
