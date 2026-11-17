#!/usr/bin/env python
"""Tables and figures from scripts/diagnose_error.py (+ measure_resident_memory.py).

Outputs (results/figures/<out>/ and results/summary/):
  error_attribution.{pdf,png}   per PQ M: Recall@10 vs nprobe for IVF coverage (= IVF-Flat),
                                PQ ranking, and PQ + exact rerank; shaded gap = PQ ranking loss
  recall_vs_qps_rerank.{pdf,png} Recall@10 vs batch QPS for every configuration
  <run>_attribution_table.csv   loss decomposition per (M, nprobe)
  <run>_matched_recall_v2.csv   cheapest config (max QPS) per target Recall@10, with memory.
                                Column `scope`: "proposal" = IVF-Flat / plain IVF-PQ (the planned
                                systems); "diagnostic" = IVF-PQ + exact rerank, which reads the raw
                                vectors and is outside the proposal's compressed-domain scope.
  appendix/opq_vs_pq.{pdf,png}  OPQ vs PQ on identical lists (OPQ is a proposal non-goal; shown only
                                as a confirmation that the gap is PQ ranking error)
"""
import argparse

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting import FLAT_COLOR, INK, INK2, PQ_MARKERS, PQ_RAMP, _end_label, _nprobe_axis, _save
from src.utils import resolve

TARGETS = [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def config_label(r) -> str:
    if r.index_type == "ivfflat":
        return f"IVF-Flat nprobe={r.nprobe}"
    s = f"IVF-PQ M={int(r.pq_m)} nprobe={r.nprobe}"
    return s + (f" +rerank {int(r.rerank_R)}" if r.rerank_R else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default="results/figures/final_watgpu")
    args = ap.parse_args(argv)
    summ, out = resolve("results/summary"), resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summ / f"{args.run_id}_error_attribution.csv")
    idx = pd.read_csv(resolve("results/raw") / f"{args.run_id}_indexes.csv")
    size = dict(zip(idx["index"], idx["serialized_size_bytes"]))
    df["index_file_bytes"] = df["index"].map(size)
    # Exact rerank needs the raw base vectors (n*d*4 bytes); counted separately because
    # they can stay on disk (memory-mapped) rather than in RAM.
    tb = idx[idx.index_type == "ivfflat"].iloc[0]
    n = int(round(tb.theoretical_payload_bytes / tb.theoretical_bytes_per_vector))
    d = int((tb.theoretical_bytes_per_vector - 8) // 4)      # Flat payload = 4d + 8 B/vector
    df["raw_vectors_bytes"] = np.where(df["rerank_R"] > 0, n * d * 4, 0)
    df["config"] = [config_label(r) for r in df.itertuples()]
    mem_path = summ / f"{args.run_id}_resident_memory.csv"
    mem = pd.read_csv(mem_path) if mem_path.exists() else None

    flat = df[df.index_type == "ivfflat"].sort_values("nprobe")
    pq = df[df.index_type == "ivfpq"]
    ms = sorted(int(m) for m in pq.pq_m.unique())
    nprobes = sorted(df.nprobe.unique())
    Rs = sorted(r for r in pq.rerank_R.unique() if r > 10)

    # ---- table: loss decomposition -------------------------------------------------
    att = pq[pq.rerank_R == 0][["pq_m", "nprobe", "coverage_at_10", "pq_recall_at_10",
                                "ivf_miss", "pq_rank_loss"]].copy()
    for r in Rs:
        rr = pq[pq.rerank_R == r].set_index(["pq_m", "nprobe"])["recall_at_10"]
        att[f"rerank{r}_recall_at_10"] = [rr[(m, p)] for m, p in zip(att.pq_m, att.nprobe)]
        att[f"rerank{r}_share_of_pq_loss_recovered"] = np.where(
            att.pq_rank_loss > 0, (att[f"rerank{r}_recall_at_10"] - att.pq_recall_at_10) / att.pq_rank_loss, np.nan)
    att.to_csv(summ / f"{args.run_id}_attribution_table.csv", index=False)

    # ---- table: matched recall ------------------------------------------------------
    rows = []
    families = [("IVF-Flat", df.index_type == "ivfflat", "proposal")]
    families += [(f"IVF-PQ M={m}", (df.pq_m == m) & (df.rerank_R == 0), "proposal") for m in ms]
    families += [(f"IVF-PQ M={m} + rerank", (df.pq_m == m) & (df.rerank_R > 0), "diagnostic")
                 for m in ms]
    for t in TARGETS:
        for fam, mask, scope in families:
            ok = df[mask & (df.recall_at_10 >= t)]
            if ok.empty:
                rows.append({"target_recall_at_10": t, "family": fam, "scope": scope, "reached": False,
                             "best_recall_in_family": df[mask].recall_at_10.max()})
                continue
            b = ok.sort_values("qps", ascending=False).iloc[0]
            rows.append({"target_recall_at_10": t, "family": fam, "scope": scope, "reached": True,
                         "config": b.config, "nprobe": b.nprobe, "rerank_R": b.rerank_R,
                         "recall_at_10": b.recall_at_10, "qps": b.qps,
                         "ms_per_query_batch": 1000 / b.qps, "index_file_bytes": b.index_file_bytes,
                         "raw_vectors_bytes": b.raw_vectors_bytes})
    mr = pd.DataFrame(rows)
    fq = mr[mr.family == "IVF-Flat"].set_index("target_recall_at_10")["qps"]
    mr["speedup_vs_flat"] = mr.qps / mr.target_recall_at_10.map(fq)
    mr.to_csv(summ / f"{args.run_id}_matched_recall_v2.csv", index=False)

    # ---- figure: error attribution (small multiples, shared y) -----------------------
    fig, axes = plt.subplots(1, len(ms), figsize=(3.1 * len(ms), 3.0), sharey=True)
    ls_r = dict(zip(Rs, ["--", ":", "-."]))
    for ax, m, col, mk in zip(np.atleast_1d(axes), ms, PQ_RAMP, PQ_MARKERS):
        sub = pq[pq.pq_m == m]
        base = sub[sub.rerank_R == 0].sort_values("nprobe")
        ax.fill_between(base.nprobe, base.pq_recall_at_10, base.coverage_at_10,
                        color=INK2, alpha=0.10, lw=0, label="PQ ranking loss")
        ax.plot(flat.nprobe, flat.coverage_at_10, color=FLAT_COLOR, marker="s",
                markeredgecolor="white", markeredgewidth=0.8, label="IVF coverage (= IVF-Flat)")
        ax.plot(base.nprobe, base.pq_recall_at_10, color=col, marker=mk,
                markeredgecolor="white", markeredgewidth=0.8, label="PQ ranking")
        for r in Rs:
            s = sub[sub.rerank_R == r].sort_values("nprobe")
            ax.plot(s.nprobe, s.recall_at_10, color=col, ls=ls_r.get(r, "-."), lw=1.6,
                    label=f"PQ + exact rerank of top {r}")
            _end_label(ax, s.nprobe.iloc[-1], s.recall_at_10.iloc[-1], f"R={r}", INK2)
        _end_label(ax, base.nprobe.iloc[-1], base.pq_recall_at_10.iloc[-1], "PQ", INK2)
        ax.set_title(f"IVF-PQ M={m} ({m} B/vector)", color=INK)
        _nprobe_axis(ax, nprobes)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_ylim(0, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("Recall@10")
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    _save(fig, out, "error_attribution")
    plt.close(fig)

    # ---- figure: recall vs QPS (matched-recall view) ---------------------------------
    if df.qps.notna().any():
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot(flat.qps, flat.recall_at_10, color=FLAT_COLOR, marker="s", markeredgecolor="white",
                label="IVF-Flat")
        _end_label(ax, flat.qps.iloc[-1], flat.recall_at_10.iloc[-1], "Flat", INK2)
        for m, col, mk in zip(ms, PQ_RAMP, PQ_MARKERS):
            sub = pq[pq.pq_m == m]
            b = sub[sub.rerank_R == 0].sort_values("nprobe")
            ax.plot(b.qps, b.recall_at_10, color=col, marker=mk, markeredgecolor="white",
                    lw=1.2, label=f"PQ{m}")
            for r in Rs:
                s = sub[sub.rerank_R == r].sort_values("nprobe")
                ax.plot(s.qps, s.recall_at_10, color=col, ls=ls_r.get(r, "-."), marker=mk, ms=4,
                        lw=1.2, label=f"PQ{m} + rerank {r}")
        ax.set_xscale("log")
        ax.set_xlabel("Throughput (queries/s, batch, 4 threads)")
        ax.set_ylabel("Recall@10")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        _save(fig, out, "recall_vs_qps_rerank")
        plt.close(fig)

    # ---- figure + table: OPQ vs PQ on identical lists (scripts/opq_experiment.py) ------
    opq_path = summ / f"{args.run_id}_opq_attribution.csv"
    if opq_path.exists():
        opq = pd.read_csv(opq_path)
        k = ["pq_m", "nprobe", "rerank_R"]
        cmp = (pq[k + ["coverage_at_10", "recall_at_10"]]
               .merge(opq[k + ["coverage_at_10", "recall_at_10"]], on=k, suffixes=("_pq", "_opq"),
                      validate="one_to_one"))
        cmp["gain"] = cmp.recall_at_10_opq - cmp.recall_at_10_pq
        cmp.to_csv(summ / f"{args.run_id}_opq_vs_pq.csv", index=False)

        fig, axes = plt.subplots(1, len(ms), figsize=(3.1 * len(ms), 3.0), sharey=True)
        for ax, m, col, mk in zip(np.atleast_1d(axes), ms, PQ_RAMP, PQ_MARKERS):
            ax.plot(flat.nprobe, flat.coverage_at_10, color=FLAT_COLOR, marker="s",
                    markeredgecolor="white", markeredgewidth=0.8, label="IVF coverage (= IVF-Flat)")
            for src, ls, name in [(pq, ":", "PQ"), (opq, "-", "OPQ")]:
                s = src[(src.pq_m == m) & (src.rerank_R == 0)].sort_values("nprobe")
                ax.plot(s.nprobe, s.recall_at_10, color=col, ls=ls, marker=mk,
                        markeredgecolor="white" if name == "OPQ" else col, markeredgewidth=1.0,
                        mfc=col if name == "OPQ" else "white", label=name)
                _end_label(ax, s.nprobe.iloc[-1], s.recall_at_10.iloc[-1], name, INK2)
            for src, ls, name in [(pq, ":", "PQ"), (opq, "-", "OPQ")]:
                s = src[(src.pq_m == m) & (src.rerank_R == 100)].sort_values("nprobe")
                if len(s):
                    ax.plot(s.nprobe, s.recall_at_10, color=INK2, ls=ls, lw=1.2,
                            label=f"{name} + exact rerank of top 100")
            ax.set_title(f"M={m} ({m} B/vector)", color=INK)
            _nprobe_axis(ax, nprobes)
            ax.tick_params(axis="x", labelrotation=45)
            ax.set_ylim(0, 1.02)
        np.atleast_1d(axes)[0].set_ylabel("Recall@10")
        h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False,
                   bbox_to_anchor=(0.5, -0.08))
        fig.tight_layout()
        (out / "appendix").mkdir(exist_ok=True)
        _save(fig, out / "appendix", "opq_vs_pq")
        plt.close(fig)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(att.round(3).to_string(index=False))
        print(mr[["target_recall_at_10", "family", "reached", "config", "qps", "speedup_vs_flat"]]
              .round(3).to_string(index=False))
        if mem is not None:
            print(mem[["label", "serialized_size_bytes", "index_rss_bytes", "total_rss_bytes",
                       "peak_rss", "precomputed_table_bytes"]].to_string(index=False))


if __name__ == "__main__":
    main()
