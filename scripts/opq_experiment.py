#!/usr/bin/env python
"""OPQ vs PQ on identical inverted lists: does a learned rotation reduce PQ ranking loss?

For each M it trains an OPQ rotation R (faiss.OPQMatrix, on the same training rows as
the run's PQ codebooks) and builds IVF-PQ in the rotated space. The run's shared coarse
centroids are rotated by the same R. R is orthogonal, so L2 distances are preserved:
every database vector lands in the same inverted list and every query probes the same
lists as in IVF-Flat / IVF-PQ. Coverage is therefore unchanged and any recall difference
comes from PQ ranking alone. The script checks the list assignment against IVF-Flat.

Index: IndexPreTransform(R, IndexIVFPQ(rotated centroids, M, nbits)), searched with
original (unrotated) queries. Rows match scripts/diagnose_error.py so OPQ and PQ can be
compared on the same table.

Outputs (results/summary/):
  <run>_opq_attribution.csv   one row per (M, nprobe, rerank R), index_type "ivfopq"
  <run>_opq_build.json        training/add time, sizes, assignment check per M
Index files: indexes/ivfopq_m<M>_nbits<b>_nlist<nlist>.faiss
"""
import argparse
import json
import os
import time

import _bootstrap  # noqa: F401
import faiss
import numpy as np
import pandas as pd

from diagnose_error import list_assignment, timed
from src.datasets import iter_batches, read_vecs
from src.indexes import index_name
from src.metrics import recall_at_k
from src.rerank import exact_rerank
from src.utils import resolve, setup_logging


def build_opq(xt, xb, cent, m, nbits, seed, add_bs, log):
    d, nlist = xt.shape[1], cent.shape[0]
    opq = faiss.OPQMatrix(d, m)
    opq.verbose = False
    t0 = time.perf_counter()
    opq.train(xt)
    t_opq = time.perf_counter() - t0
    R = faiss.vector_to_array(opq.A).reshape(d, d)
    R64 = R.astype(np.float64)
    orth = float(np.abs(R64 @ R64.T - np.eye(d)).max())
    assert orth < 1e-3, f"OPQ matrix not orthogonal (max dev {orth})"
    log.info("OPQ M=%d trained in %.1f s (orthogonality dev %.1e)", m, t_opq, orth)

    quantizer = faiss.IndexFlatL2(d)
    quantizer.add(opq.apply_py(np.ascontiguousarray(cent)))
    ivf = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_L2)
    ivf.pq.cp.seed = seed
    ivf.cp.seed = seed
    ivf.own_fields = True
    quantizer.this.disown()
    t0 = time.perf_counter()
    ivf.train(opq.apply_py(xt))           # PQ codebooks on rotated residuals
    t_pq = time.perf_counter() - t0
    t_add = 0.0
    for _, chunk in iter_batches(xb, add_bs):
        t0 = time.perf_counter()
        ivf.add(opq.apply_py(np.ascontiguousarray(chunk)))
        t_add += time.perf_counter() - t0
    index = faiss.IndexPreTransform(opq, ivf)
    index.own_fields = True
    opq.this.disown()
    ivf.this.disown()
    assert index.ntotal == xb.shape[0]
    return index, {"opq_train_s": t_opq, "pq_train_s": t_pq, "add_s": t_add,
                   "orthogonality_max_dev": orth}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--m", type=int, nargs="+", default=None, help="default: the run's PQ M values")
    ap.add_argument("--nprobe", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ap.add_argument("--rerank", type=int, nargs="+", default=[100, 1000])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    log = setup_logging(f"opq_{args.run_id}")

    meta = json.loads((resolve("results/raw") / f"{args.run_id}_metadata.json").read_text())
    cfg = meta["config"]
    nlist, nbits, seed = cfg["ivf"]["nlist"], cfg["pq"]["nbits"], cfg["seed"]
    threads = int(meta["faiss_threads_effective"])
    faiss.omp_set_num_threads(threads)
    nprobes = sorted(p for p in set(args.nprobe) if p <= nlist)
    R = sorted(set(args.rerank))
    kmax = max(max(R), 10)
    ms = sorted(args.m or cfg["pq"]["m"])

    ds = cfg["dataset"]
    root = resolve(ds["dir"])
    xq = np.ascontiguousarray(read_vecs(root / ds["files"]["query"], "f", ds["dim"]))
    gt = np.asarray(read_vecs(root / ds["files"]["groundtruth"], "i"), dtype=np.int64)
    xb = read_vecs(root / ds["files"]["base"], "f", ds["dim"], check_headers=False)
    xt = np.ascontiguousarray(read_vecs(root / ds["files"]["learn"], "f", ds["dim"],
                                        check_headers=False)[:ds["train_size"]])
    n, nq = xb.shape[0], xq.shape[0]
    gt10 = gt[:, :10]
    idx_dir = resolve(cfg["paths"]["indexes"])

    # Reference: IVF-Flat of this run (shared centroids, list assignment, coverage).
    flat = faiss.read_index(str(idx_dir / f"{index_name('ivfflat', nlist)}.faiss"))
    fivf = faiss.extract_index_ivf(flat)
    cent = fivf.quantizer.reconstruct_n(0, nlist)
    assign_ref = list_assignment(flat, n)
    _, order = fivf.quantizer.search(xq, nlist)
    del flat
    list_rank = np.empty_like(order)
    list_rank[np.arange(nq)[:, None], order] = np.arange(nlist)[None, :]
    gt_rank = np.take_along_axis(list_rank, assign_ref[gt10], axis=1)
    coverage = {p: float((gt_rank < p).mean()) for p in nprobes}
    log.info("coverage@10 (IVF-Flat lists): %s", {p: round(c, 4) for p, c in coverage.items()})

    rows, build = [], {}
    for m in ms:
        name = index_name("ivfpq", nlist, m, nbits).replace("ivfpq", "ivfopq", 1)
        path = idx_dir / f"{name}.faiss"
        if path.exists() and not args.rebuild:
            index = faiss.read_index(str(path))
            binfo = json.loads(path.with_suffix(".json").read_text())
            log.info("Loaded %s from cache", name)
        else:
            log.info("Building %s", name)
            index, binfo = build_opq(xt, xb, cent, m, nbits, seed, cfg["build"]["add_batch_size"], log)
            tmp = path.with_suffix(".faiss.tmp")
            faiss.write_index(index, str(tmp))
            os.replace(tmp, path)
            binfo["serialized_size_bytes"] = path.stat().st_size
            path.with_suffix(".json").write_text(json.dumps(binfo, indent=2))
        ivf = faiss.extract_index_ivf(index)

        assign = list_assignment(index, n)
        diff = int((assign != assign_ref).sum())
        binfo["assignment_differs_from_flat"] = diff
        build[name] = binfo
        log.info("%s: %d of %d vectors in a different list than IVF-Flat", name, diff, n)
        # Orthogonal rotation preserves distances; allow float-rounding ties only.
        assert diff <= n * 1e-4, f"{name}: {diff} vectors changed list"
        rq = faiss.downcast_VectorTransform(index.chain.at(0)).apply_py(xq)
        # This index's own coverage (its lists, its probe order). Near-tied centroids can
        # swap probe rank under float rounding; coverage must still match IVF-Flat's.
        _, o2 = ivf.quantizer.search(rq, nlist)
        lr2 = np.empty_like(o2)
        lr2[np.arange(nq)[:, None], o2] = np.arange(nlist)[None, :]
        own_rank = np.take_along_axis(lr2, assign[gt10], axis=1)
        own_cov = {p: float((own_rank < p).mean()) for p in nprobes}
        gap = max(abs(own_cov[p] - coverage[p]) for p in nprobes)
        binfo["max_coverage_gap_vs_flat"] = gap
        log.info("%s: max |coverage - IVF-Flat coverage| = %.4f", name, gap)
        assert gap <= 0.002, f"{name}: coverage differs from IVF-Flat by {gap}"

        label = f"IVF-OPQ M={m}"
        for p in nprobes:
            ivf.nprobe = p
            base = {"index": name, "label": label, "index_type": "ivfopq", "pq_m": m,
                    "code_bytes": m * nbits // 8, "nprobe": p, "coverage_at_10": own_cov[p],
                    "flat_coverage_at_10": coverage[p]}
            _, ids = index.search(xq, kmax)
            pq_rec = recall_at_k(ids[:, :10], gt, 10)
            for r in [0] + R:
                if r == 0:
                    rec, t_rr, fn = pq_rec, 0.0, (lambda: index.search(xq, 10))
                else:
                    cand = ids[:, :r]
                    t_rr, top = (timed(lambda: exact_rerank(xb, xq, cand, 10), 1)
                                 if not args.no_timing else (np.nan, exact_rerank(xb, xq, cand, 10)))
                    rec = recall_at_k(top, gt, 10)
                    fn = lambda rr=r: index.search(xq, rr)
                t_s = timed(fn, args.reps)[0] if not args.no_timing else np.nan
                tot = t_s + t_rr
                rows.append({**base, "rerank_R": r, "recall_at_10": rec,
                             "pq_recall_at_10": pq_rec, "ivf_miss": 1 - own_cov[p],
                             "pq_rank_loss": own_cov[p] - pq_rec,
                             "pq_rank_loss_recovered": rec - pq_rec,
                             "search_s": t_s, "rerank_s": t_rr, "total_s": tot,
                             "qps": nq / tot if tot == tot else np.nan,
                             "serialized_size_bytes": binfo["serialized_size_bytes"]})
                log.info("%s nprobe=%d R=%d R@10=%.4f (cov %.4f) %.3fs", label, p, r, rec,
                         own_cov[p], tot)
        del index

    out = resolve("results/summary")
    df = pd.DataFrame(rows)
    df["run_id"], df["faiss_threads"], df["n_queries"] = args.run_id, threads, nq
    csv = out / f"{args.run_id}_opq_attribution.csv"
    if args.m and csv.exists():           # partial rerun: keep the other M values
        old = pd.read_csv(csv)
        df = pd.concat([old[~old.pq_m.isin(ms)], df], ignore_index=True)
    df.to_csv(csv, index=False)
    (out / f"{args.run_id}_opq_build.json").write_text(json.dumps(build, indent=2))
    log.info("Saved %s", csv)


if __name__ == "__main__":
    main()
