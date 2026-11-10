#!/usr/bin/env python
"""Resident memory (RSS) of each index, measured in a fresh process per configuration.

Serialized size (the .faiss file) is a *storage* measure. Runtime memory differs:
IVF-PQ allocates residual lookup tables at load time (not in the file), and FAISS
allocates per-query scratch during search. This script reports, per index:

  rss_baseline     after importing numpy/faiss and reading the queries
  rss_loaded       after faiss.read_index
  rss_after_search after searching all queries at the given nprobe
  peak_rss         ru_maxrss of the process
  index_rss        = rss_after_search - rss_baseline (memory attributable to the index)

Option --rerank R adds the exact re-ranking step, which needs the raw base vectors.
They are memory-mapped, so only the touched pages become resident; that measured value
and the full raw-vector size (n*d*4) are both reported.

Each configuration runs in its own subprocess so allocations cannot leak between them.
Output: results/summary/<run>_resident_memory.csv
"""
import argparse
import json
import subprocess
import sys

import _bootstrap  # noqa: F401
import pandas as pd

from src.indexes import index_name
from src.utils import resolve, setup_logging

CHILD = r"""
import json, resource, sys, numpy as np, psutil, faiss
sys.path.insert(0, {root!r})
from src.datasets import read_vecs
from src.rerank import exact_rerank
proc = psutil.Process()
faiss.omp_set_num_threads({threads})
xq = np.ascontiguousarray(read_vecs({q!r}, "f"))
rss0 = proc.memory_info().rss
index = faiss.read_index({path!r})
rss1 = proc.memory_info().rss
ivf = faiss.extract_index_ivf(index); ivf.nprobe = {nprobe}
k = max({rerank}, 10)
_, ids = index.search(xq, k)
rss2 = proc.memory_info().rss
rss3 = rss2
if {rerank}:
    xb = read_vecs({b!r}, "f", check_headers=False)
    exact_rerank(xb, xq, ids, 10)
    rss3 = proc.memory_info().rss
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform != "darwin":
    peak *= 1024            # Linux reports KiB, macOS bytes
pt = 0
idx = faiss.downcast_index(index)
if isinstance(idx, faiss.IndexIVFPQ) and idx.use_precomputed_table:
    pt = int(idx.precomputed_table.size()) * 4
print(json.dumps(dict(rss_baseline=rss0, rss_loaded=rss1, rss_after_search=rss2,
                      rss_after_rerank=rss3, peak_rss=peak, precomputed_table_bytes=pt)))
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--rerank", type=int, nargs="*", default=[100],
                    help="also measure IVF-PQ + exact rerank with these shortlist sizes")
    args = ap.parse_args(argv)
    log = setup_logging(f"resident_memory_{args.run_id}")

    meta = json.loads((resolve("results/raw") / f"{args.run_id}_metadata.json").read_text())
    cfg = meta["config"]
    ds = cfg["dataset"]
    root = resolve(ds["dir"])
    nlist, nbits = cfg["ivf"]["nlist"], cfg["pq"]["nbits"]
    n, d = meta["dataset"]["n_base"], ds["dim"]

    configs = [("ivfflat", None, 0)] + [("ivfpq", int(m), 0) for m in sorted(cfg["pq"]["m"])]
    configs += [("ivfpq", int(m), r) for m in sorted(cfg["pq"]["m"]) for r in args.rerank]
    rows = []
    for itype, m, r in configs:
        name = index_name(itype, nlist, m, nbits)
        path = resolve(cfg["paths"]["indexes"]) / f"{name}.faiss"
        code = CHILD.format(root=str(resolve(".")), threads=int(meta["faiss_threads_effective"]),
                            q=str(root / ds["files"]["query"]), b=str(root / ds["files"]["base"]),
                            path=str(path), nprobe=args.nprobe, rerank=r)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        res = json.loads(out.stdout.strip().splitlines()[-1])
        label = "IVF-Flat" if m is None else f"IVF-PQ M={m}" + (f" + rerank {r}" if r else "")
        row = {"label": label, "index": name, "pq_m": m, "rerank_R": r, "nprobe": args.nprobe,
               "serialized_size_bytes": path.stat().st_size, **res}
        row["index_rss_bytes"] = res["rss_after_search"] - res["rss_baseline"]
        row["total_rss_bytes"] = res["rss_after_rerank"] - res["rss_baseline"]
        row["raw_vectors_bytes_if_in_ram"] = n * d * 4 if r else 0
        rows.append(row)
        log.info("%s: file %.1f MiB, index RSS %.1f MiB, total RSS %.1f MiB, peak %.1f MiB",
                 label, row["serialized_size_bytes"] / 2**20, row["index_rss_bytes"] / 2**20,
                 row["total_rss_bytes"] / 2**20, res["peak_rss"] / 2**20)

    out = resolve("results/summary") / f"{args.run_id}_resident_memory.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("Saved %s", out)


if __name__ == "__main__":
    main()
