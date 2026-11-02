#!/usr/bin/env python
"""Exact-1NN Hit@R for an existing benchmark run (quality only; nothing is timed or rebuilt).

Hit@R = fraction of queries whose exact #1 neighbour (ground truth column 0) appears
anywhere in the approximate top-R. This is NOT Recall@R (which is |top-R ∩ GT top-R| / R).

The benchmark does not store retrieved ids, so this script re-searches the cached
indexes of the run. Before trusting the result it verifies that
  * every index file has the serialized size recorded in <run>_indexes.csv,
  * every index sidecar spec matches the run's config (nlist, M, nbits, seed, dataset),
  * the recomputed Recall@1/10/100 equal <run>_recall.csv exactly,
so the hit rates come from exactly the searches that produced the run's recall numbers.

Output: results/summary/<run>_exact1nn_hit100.csv
"""
import argparse
import json

import _bootstrap  # noqa: F401
import faiss
import numpy as np
import pandas as pd

from src.datasets import read_vecs
from src.indexes import index_name, set_nprobe
from src.metrics import recalls
from src.utils import resolve, setup_logging


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--hit-at", type=int, nargs="+", default=[1, 10, 100])
    args = ap.parse_args(argv)
    log = setup_logging(f"exact1nn_hit_{args.run_id}")

    raw = resolve("results/raw")
    meta = json.loads((raw / f"{args.run_id}_metadata.json").read_text())
    cfg = meta["config"]
    idx_tab = pd.read_csv(raw / f"{args.run_id}_indexes.csv")
    rec_tab = pd.read_csv(raw / f"{args.run_id}_recall.csv")
    k = int(meta["k"])
    assert max(args.hit_at) <= k
    nprobes = [int(p) for p in meta["nprobe"]]
    faiss.omp_set_num_threads(int(meta["faiss_threads_effective"]))

    ds_cfg = cfg["dataset"]
    root = resolve(ds_cfg["dir"])
    xq = np.ascontiguousarray(read_vecs(root / ds_cfg["files"]["query"], "f", ds_cfg["dim"]))
    gt = np.asarray(read_vecs(root / ds_cfg["files"]["groundtruth"], "i"), dtype=np.int64)
    assert xq.shape[0] == gt.shape[0] == meta["dataset"]["n_query"]

    rows = []
    for _, ir in idx_tab.iterrows():
        m = None if pd.isna(ir["pq_m"]) else int(ir["pq_m"])
        name = index_name(ir["index_type"], int(ir["nlist"]), m, cfg["pq"]["nbits"])
        assert name == ir["index"], (name, ir["index"])
        path = resolve(cfg["paths"]["indexes"]) / f"{name}.faiss"
        size = path.stat().st_size
        assert size == ir["serialized_size_bytes"], f"{name}: file size {size} != run record"
        side = json.loads(path.with_suffix(".json").read_text())["spec"]
        expect = {k_: v for k_, v in side.items() if k_ != "dataset"}
        assert expect["nlist"] == cfg["ivf"]["nlist"] and expect["seed"] == cfg["seed"]
        assert expect["pq_m"] == m and side["dataset"]["n_base"] == meta["dataset"]["n_base"]
        log.info("Loading %s (%d bytes, matches run record)", name, size)
        index = faiss.read_index(str(path))
        for p in nprobes:
            set_nprobe(index, p)
            _, ids = index.search(xq, k)
            ref = rec_tab[(rec_tab["index"] == name) & (rec_tab["nprobe"] == p)].iloc[0]
            got = recalls(ids, gt, [1, 10, 100])
            for kk in (1, 10, 100):
                # CSV round-trip changes the float repr; recall moves in 1e-5 steps, so 1e-9 = exact
                assert abs(got[kk] - ref[f"recall_at_{kk}"]) < 1e-9, \
                    f"{name} nprobe={p}: Recall@{kk} {got[kk]} != run {ref[f'recall_at_{kk}']}"
            nn1 = gt[:, 0:1]
            row = {"index": name, "label": ir["label"], "index_type": ir["index_type"], "pq_m": m,
                   "nprobe": p, "n_queries": int(xq.shape[0]), "k_searched": k,
                   "recall_matches_run": True, "run_id": args.run_id}
            for r in args.hit_at:
                row[f"exact1nn_hit_at_{r}"] = float((ids[:, :r] == nn1).any(axis=1).mean())
            rows.append(row)
            log.info("%s nprobe=%d: %s", name, p,
                     ", ".join(f"Hit@{r}={row[f'exact1nn_hit_at_{r}']:.3f}" for r in args.hit_at))
        del index

    out = resolve("results/summary") / f"{args.run_id}_exact1nn_hit100.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("Saved %s", out)


if __name__ == "__main__":
    main()
