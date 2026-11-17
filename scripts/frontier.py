#!/usr/bin/env python
"""Operating-point frontier for the planned grid (proposal Sec. 3.4).

A configuration is dominated if another one has at least as much Recall@10 with no more
p95 latency and no more memory (one strictly better). With three objectives almost
every point survives (each PQ code size is the smallest at something), so this script
also reports the two 2-D frontiers and a memory-budget table, which are easier to read:

  * recall vs p95 latency (memory ignored)
  * recall vs memory (latency ignored), for serialized size and for resident memory
  * per memory budget: best Recall@10 reachable and the p95 latency it costs

Only the planned systems (IVF-Flat, plain IVF-PQ; nprobe grid of the benchmark run).
Memory: serialized .faiss size (storage) and, if available, index RSS from
scripts/measure_resident_memory.py (runtime).

Outputs:
  results/summary/<run>_frontier.csv
  results/summary/<run>_memory_budget.csv
  <out>/frontier.{pdf,png}
"""
import argparse

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import pareto_mask
from src.plotting import INK, INK2, _end_label, _readable_log, _save, styles
from src.utils import resolve

MiB = 2 ** 20


def frontier_2d(better_y, cost_x) -> np.ndarray:
    """Non-dominated mask for (maximise y, minimise x)."""
    y, x = np.asarray(better_y, float), np.asarray(cost_x, float)
    keep = np.ones(len(y), bool)
    for i in range(len(y)):
        dom = (y >= y[i]) & (x <= x[i]) & ((y > y[i]) | (x < x[i]))
        keep[i] = not dom.any()
    return keep


def dominator(df, i, cols_max, cols_min) -> str:
    r = df.iloc[i]
    m = np.ones(len(df), bool)
    strict = np.zeros(len(df), bool)
    for c in cols_max:
        m &= df[c].values >= r[c]
        strict |= df[c].values > r[c]
    for c in cols_min:
        m &= df[c].values <= r[c]
        strict |= df[c].values < r[c]
    m &= strict
    if not m.any():
        return ""
    j = df[m].sort_values("recall_at_10", ascending=False).index[0]
    return df.loc[j, "config"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default="results/figures/final_watgpu")
    ap.add_argument("--budgets-mib", type=float, nargs="+", default=[32, 48, 80, 160, 4096])
    args = ap.parse_args(argv)
    summ, out = resolve("results/summary"), resolve(args.out)

    df = pd.read_csv(summ / f"{args.run_id}_summary.csv").reset_index(drop=True)
    df["config"] = [f"{l} nprobe={p}" for l, p in zip(df.label, df.nprobe)]
    df["file_mib"] = df.serialized_size_bytes / MiB
    mem_path = summ / f"{args.run_id}_resident_memory.csv"
    if mem_path.exists():
        mem = pd.read_csv(mem_path)
        mem = mem[mem.rerank_R == 0].set_index("label")["index_rss_bytes"] / MiB
        df["rss_mib"] = df["label"].map(mem)
    else:
        df["rss_mib"] = np.nan

    p95, r10 = df.latency_p95_ms, df.recall_at_10
    df["pareto_3d_file"] = pareto_mask(r10, p95, df.file_mib)
    df["dominated_by_3d_file"] = [dominator(df, i, ["recall_at_10"], ["latency_p95_ms", "file_mib"])
                                  for i in range(len(df))]
    if df.rss_mib.notna().all():
        df["pareto_3d_rss"] = pareto_mask(r10, p95, df.rss_mib)
    df["frontier_recall_vs_p95"] = frontier_2d(r10, p95)
    df["frontier_recall_vs_file"] = frontier_2d(r10, df.file_mib)
    df["retained_vs_flat_same_nprobe"] = r10 / df.nprobe.map(
        df[df.index_type == "ivfflat"].set_index("nprobe").recall_at_10)
    cols = ["config", "label", "nprobe", "recall_at_10", "recall_at_100", "latency_p50_ms",
            "latency_p95_ms", "qps", "file_mib", "rss_mib", "retained_vs_flat_same_nprobe",
            "pareto_3d_file", "dominated_by_3d_file"] + \
           (["pareto_3d_rss"] if "pareto_3d_rss" in df else []) + \
           ["frontier_recall_vs_p95", "frontier_recall_vs_file"]
    df[cols].to_csv(summ / f"{args.run_id}_frontier.csv", index=False)

    rows = []
    for b in args.budgets_mib:
        for memcol, kind in [("file_mib", "serialized"), ("rss_mib", "resident")]:
            ok = df[df[memcol] <= b]
            if ok.empty:
                rows.append({"budget_mib": b, "memory_measure": kind, "best_config": None})
                continue
            best = ok.sort_values(["recall_at_10", "latency_p95_ms"], ascending=[False, True]).iloc[0]
            rows.append({"budget_mib": b, "memory_measure": kind, "best_config": best.config,
                         "recall_at_10": best.recall_at_10, "latency_p95_ms": best.latency_p95_ms,
                         "memory_mib": best[memcol]})
    pd.DataFrame(rows).to_csv(summ / f"{args.run_id}_memory_budget.csv", index=False)

    # ---- figure: two 2-D frontiers, one axis each --------------------------------------
    st = styles(df)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for lab, s in st.items():
        d = df[df.label == lab].sort_values("nprobe")
        for ax, x in [(a1, "latency_p95_ms"), (a2, "file_mib")]:
            ax.scatter(d[x], d.recall_at_10, color=s["color"], marker=s["marker"], s=36,
                       edgecolor="white", linewidth=0.8, zorder=3, label=lab)
        a1.plot(d.latency_p95_ms, d.recall_at_10, color=s["color"], lw=1.0, alpha=0.6, zorder=2)
        _end_label(a1, d.latency_p95_ms.iloc[-1], d.recall_at_10.iloc[-1], s["short"], INK2)
    for ax, x, fcol in [(a1, "latency_p95_ms", "frontier_recall_vs_p95"),
                        (a2, "file_mib", "frontier_recall_vs_file")]:
        f = df[df[fcol]].sort_values(x)
        ax.step(f[x], f.recall_at_10, where="post", color=INK, lw=1.0, ls="--", zorder=1,
                label="frontier")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Recall@10")
    for ax in (a1, a2):
        _readable_log(ax.xaxis)
    a1.set_xlabel("p95 latency (ms/query, single query)")
    a2.set_xlabel("Serialized index size (MiB)")
    a1.set_title("Recall vs latency", color=INK)
    a2.set_title("Recall vs storage", color=INK)
    h, lab = a1.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    _save(fig, out, "frontier")
    plt.close(fig)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[["config", "recall_at_10", "latency_p95_ms", "file_mib", "rss_mib", "pareto_3d_file",
                  "dominated_by_3d_file", "frontier_recall_vs_p95"]].round(3).to_string(index=False))
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
