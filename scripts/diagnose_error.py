#!/usr/bin/env python
"""Error attribution: IVF candidate coverage vs PQ ranking error, on identical candidate pools.

Answers the question "is the IVF-PQ recall plateau caused by quantization?" directly,
instead of inferring it from the shape of the Recall-vs-nprobe curve.

For one benchmark run's cached indexes (shared coarse quantizer) it:

1. Verifies that IVF-Flat and every IVF-PQ index hold *identical inverted-list
   assignments* (same database id -> same list), not just identical centroids.
   Together with the shared quantizer this means that for a given (query, nprobe)
   every index scans exactly the same candidate pool.
2. Computes, for a wide nprobe sweep up to nlist (= scanning every list),
     coverage@10  = fraction of the true top-10 that lies in the probed lists.
   This is the best any IVF index can do at that nprobe. Exact scoring over the pool
   (IVF-Flat) achieves it exactly; the script checks that Flat Recall@10 == coverage@10.
3. For each IVF-PQ index, on the same pool:
     pq_recall@10        PQ distances rank the pool
     rerank_R recall@10  PQ shortlist of R, re-scored with exact distances
   so the recall loss splits into
     ivf_miss  = 1 - coverage          (true neighbour not in probed lists)
     pq_rank   = coverage - pq_recall  (in the pool, but PQ ranks it out of the top 10)
   and rerank_R shows how much of pq_rank a shortlist of size R recovers.
4. Times every configuration in batch mode (all queries per call, median of reps), so
   configurations can be compared at matched recall.

Outputs (results/summary/):
  <run>_error_attribution.csv   one row per (index, nprobe, rerank R)
  <run>_assignment_check.json   list-assignment identity check
"""
import argparse
import json
import time

import _bootstrap  # noqa: F401
import faiss
import numpy as np
import pandas as pd

from src.datasets import read_vecs
from src.indexes import index_name, set_nprobe
from src.metrics import recall_at_k
from src.rerank import exact_rerank
from src.utils import resolve, setup_logging


def list_assignment(index: faiss.Index, n: int) -> np.ndarray:
    """assign[id] = inverted list holding database vector `id`."""
    ivf = faiss.extract_index_ivf(index)
    inv = ivf.invlists
    assign = np.full(n, -1, dtype=np.int32)
    for lst in range(ivf.nlist):
        sz = inv.list_size(lst)
        if sz:
            ids = faiss.rev_swig_ptr(inv.get_ids(lst), sz).copy()
            assign[ids] = lst
    assert (assign >= 0).all(), "some database ids are in no list"
    return assign


def timed(fn, reps: int) -> tuple[float, object]:
    """Median wall time over `reps` calls (after one untimed warm-up) and the last result."""
    out = fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--nprobe", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ap.add_argument("--rerank", type=int, nargs="+", default=[10, 100, 1000],
                    help="PQ shortlist sizes R re-scored exactly (R=10 = no reordering beyond top 10)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args(argv)
    log = setup_logging(f"diagnose_error_{args.run_id}")

    raw = resolve("results/raw")
    meta = json.loads((raw / f"{args.run_id}_metadata.json").read_text())
    cfg = meta["config"]
    nlist, nbits = cfg["ivf"]["nlist"], cfg["pq"]["nbits"]
    threads = int(meta["faiss_threads_effective"])
    faiss.omp_set_num_threads(threads)
    nprobes = sorted(p for p in set(args.nprobe) if p <= nlist)
    R = sorted(set(args.rerank))
    kmax = max(max(R), 10)

    ds = cfg["dataset"]
    root = resolve(ds["dir"])
    xq = np.ascontiguousarray(read_vecs(root / ds["files"]["query"], "f", ds["dim"]))
    gt = np.asarray(read_vecs(root / ds["files"]["groundtruth"], "i"), dtype=np.int64)
    xb = read_vecs(root / ds["files"]["base"], "f", ds["dim"], check_headers=False)
    n, nq = xb.shape[0], xq.shape[0]
    gt10 = gt[:, :10]

    names = [("ivfflat", None)] + [("ivfpq", int(m)) for m in sorted(cfg["pq"]["m"])]
    rows, assign_ref, check = [], None, {}

    for itype, m in names:
        name = index_name(itype, nlist, m, nbits)
        label = "IVF-Flat" if itype == "ivfflat" else f"IVF-PQ M={m}"
        path = resolve(cfg["paths"]["indexes"]) / f"{name}.faiss"
        log.info("Loading %s", name)
        index = faiss.read_index(str(path))
        ivf = faiss.extract_index_ivf(index)

        assign = list_assignment(index, n)
        if assign_ref is None:
            assign_ref = assign
            # Rank of every list for every query under the shared coarse quantizer.
            _, order = ivf.quantizer.search(xq, nlist)
            list_rank = np.empty_like(order)
            list_rank[np.arange(nq)[:, None], order] = np.arange(nlist)[None, :]
            gt_list_rank = np.take_along_axis(list_rank, assign[gt10], axis=1)  # (nq, 10)
            coverage = {p: float((gt_list_rank < p).mean()) for p in nprobes}
            log.info("coverage@10: %s", {p: round(c, 4) for p, c in coverage.items()})
            check["reference"] = name
        else:
            same = int((assign == assign_ref).sum())
            check[name] = {"identical_ids": same, "n": n, "identical": same == n}
            assert same == n, f"{name}: {n - same} vectors in a different list than IVF-Flat"
            log.info("%s: list assignment identical to IVF-Flat for all %d vectors", name, n)
        # Same quantizer also means same probed lists; check it explicitly.
        _, o2 = ivf.quantizer.search(xq, min(64, nlist))
        assert np.array_equal(o2, order[:, :o2.shape[1]]), f"{name}: different probe order"

        for p in nprobes:
            set_nprobe(index, p)
            base = {"index": name, "label": label, "index_type": itype, "pq_m": m,
                    "code_bytes": 4 * ds["dim"] if m is None else m * nbits // 8,
                    "nprobe": p, "coverage_at_10": coverage[p]}
            if itype == "ivfflat":
                t, (_, ids) = (timed(lambda: index.search(xq, 10), args.reps) if not args.no_timing
                               else (np.nan, index.search(xq, 10)))
                rec = recall_at_k(ids, gt, 10)
                # Exact scoring over the pool reaches coverage except where a true
                # neighbour has an exact duplicate (equal distance) that wins the tie.
                assert rec <= coverage[p] + 1e-9 and coverage[p] - rec <= 0.002, \
                    f"Flat recall {rec} vs coverage {coverage[p]} at nprobe={p}"
                rows.append({**base, "rerank_R": 0, "recall_at_10": rec,
                             "tie_gap": coverage[p] - rec,
                             "search_s": t, "rerank_s": 0.0, "total_s": t,
                             "qps": nq / t if t == t else np.nan})
                log.info("%s nprobe=%d R@10=%.4f (coverage %.4f) %.3fs", label, p, rec, coverage[p], t)
                continue

            _, ids = index.search(xq, kmax)       # one search gives every shortlist prefix
            pq_rec = recall_at_k(ids[:, :10], gt, 10)
            for r in [0] + R:
                cand = ids[:, :max(r, 10)]
                if r == 0:      # plain PQ ranking
                    rec, t_rr, fn = pq_rec, 0.0, (lambda: index.search(xq, 10))
                else:
                    inter = (cand[:, :, None] == gt10[:, None, :]).any(axis=1).sum(axis=1)
                    rec_bound = float(inter.mean() / 10)
                    t_rr, top = (timed(lambda: exact_rerank(xb, xq, cand, 10), 1)
                                 if not args.no_timing else (np.nan, exact_rerank(xb, xq, cand, 10)))
                    rec = recall_at_k(top, gt, 10)
                    assert rec <= rec_bound + 1e-9 and rec_bound - rec <= 0.002, (rec, rec_bound)
                    fn = lambda rr=r: index.search(xq, rr)
                t_s = timed(fn, args.reps)[0] if not args.no_timing else np.nan
                tot = t_s + t_rr
                rows.append({**base, "rerank_R": r, "recall_at_10": rec,
                             "tie_gap": (rec_bound - rec) if r else 0.0,
                             "pq_recall_at_10": pq_rec,
                             "ivf_miss": 1 - coverage[p], "pq_rank_loss": coverage[p] - pq_rec,
                             "pq_rank_loss_recovered": rec - pq_rec,
                             "within_pool_recall": rec / coverage[p] if coverage[p] else np.nan,
                             "search_s": t_s, "rerank_s": t_rr, "total_s": tot,
                             "qps": nq / tot if tot == tot else np.nan})
                log.info("%s nprobe=%d R=%d R@10=%.4f (cov %.4f) %.3fs", label, p, r, rec,
                         coverage[p], tot)
        del index

    out_dir = resolve("results/summary")
    df = pd.DataFrame(rows)
    df["run_id"], df["faiss_threads"], df["n_queries"] = args.run_id, threads, nq
    out = out_dir / f"{args.run_id}_error_attribution.csv"
    df.to_csv(out, index=False)
    (out_dir / f"{args.run_id}_assignment_check.json").write_text(json.dumps(check, indent=2))
    log.info("Saved %s", out)


if __name__ == "__main__":
    main()
