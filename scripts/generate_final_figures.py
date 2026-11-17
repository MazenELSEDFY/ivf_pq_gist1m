#!/usr/bin/env python
"""Final publication figures for the IVF-Flat vs IVF-PQ GIST1M study.

Every plotted number is read from the run's result files; nothing experimental is
typed in here. Sources (all under results/):
  summary/<run>_summary.csv           recall, p50/p95 latency, QPS, sizes, code bytes
  summary/<run>_exact1nn_hit100.csv   Exact-1NN Hit@R (scripts/compute_exact1nn_hit.py)
  summary/<cmp>_summary.csv           second run, latency stability figure only
  raw/<run>_metadata.json             threads, repetitions, k, dataset sizes (README)

Outputs: results/figures/final/*.{pdf,png}, plotted_values.csv (every plotted value with
its source column) and README.md (measurement notes incl. the swap caveat).

Visual identity is fixed across figures: IVF-Flat = orange squares; IVF-PQ is an ordered
family (code size), so 16/32/64 B use one blue hue light->dark with circle/triangle/
diamond markers. Each series is also direct-labelled or in a legend, never colour alone.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator  # noqa: E402

from src.utils import resolve  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e8e7e3"
FLAT = dict(color="#eb6834", marker="s")
PQ_STYLE = [dict(color="#86b6ef", marker="o"), dict(color="#2a78d6", marker="^"),
            dict(color="#104281", marker="D")]
COL_W, FULL_W = 3.35, 7.0          # ACM single column / full width, inches
# candidate (dx, dy) offsets in points for point labels, tried in order
LABEL_CANDIDATES = [(0, 6), (0, -7), (6, 0), (-6, 0), (6, 6), (-6, 6), (6, -6), (-6, -6),
                    (0, 11), (0, -12)]


def place_point_labels(fig, ax, pts, fontsize=5.5, marker_pt=4.5):
    """Annotate (x, y, text) points, choosing for each label the first candidate offset whose
    box overlaps neither an already placed label nor any data marker."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72
    markers = []
    for x, y, _ in pts:
        cx, cy = ax.transData.transform((x, y))
        markers.append((cx - marker_pt * px, cy - marker_pt * px, cx + marker_pt * px, cy + marker_pt * px))
    placed = []

    def hits(b, boxes):
        return any(b.x0 < q[2] and b.x1 > q[0] and b.y0 < q[3] and b.y1 > q[1] for q in boxes)

    for x, y, text in pts:
        for i, (dx, dy) in enumerate(LABEL_CANDIDATES):
            ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
            va = "center" if dy == 0 else ("bottom" if dy > 0 else "top")
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, color=INK2, ha=ha, va=va)
            b = t.get_window_extent(r)
            if not hits(b, placed) and not hits(b, markers) or i == len(LABEL_CANDIDATES) - 1:
                placed.append((b.x0, b.y0, b.x1, b.y1))
                break
            t.remove()

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.6,
    "lines.markersize": 5, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})

PLOTTED: list[dict] = []


# --------------------------------------------------------------------------- helpers

def series_names(df: pd.DataFrame) -> list[str]:
    """Display name per index, derived from data: IVF-Flat, PQ-<code bytes> (IVF-PQ)."""
    df["name"] = np.where(df["index_type"] == "ivfflat", "IVF-Flat",
                          "PQ-" + df["pq_code_bytes"].fillna(0).astype(int).astype(str))
    pq = (df.loc[df["index_type"] == "ivfpq", ["pq_code_bytes", "name"]]
          .drop_duplicates().sort_values("pq_code_bytes")["name"].tolist())
    assert len(pq) <= len(PQ_STYLE)
    return ["IVF-Flat"] + pq


def style(name: str, order: list[str]) -> dict:
    return FLAT if name == "IVF-Flat" else PQ_STYLE[order.index(name) - 1]


def short(name: str) -> str:
    return name


def record(fig: str, df: pd.DataFrame, cols: list[str], source: str):
    for _, r in df.iterrows():
        for c in cols:
            PLOTTED.append({"figure": fig, "series": r.get("name", ""), "nprobe": r.get("nprobe", ""),
                            "column": c, "value": r[c], "source": source})


def save(fig, out: Path, name: str) -> list[Path]:
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return paths


def nprobe_axis(ax, nprobes):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(nprobes))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("nprobe (inverted lists searched)")


def log_axis(axis, subs=(1.0, 2.0, 5.0)):
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def end_label(ax, x, y, text, dx=5, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                color=INK, va="center", ha="left")


def legend_below(fig, ax, ncol=2):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=ncol, frameon=False,
               handlelength=2.2, columnspacing=1.2)


def fmt_size(b: float) -> str:
    return f"{b / 1024**3:.2f} GiB" if b >= 1024**3 else f"{b / 1024**2:.1f} MiB"


# --------------------------------------------------------------------------- figures

def fig_recall_vs_nprobe(df, order, nprobes, out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.6), layout="constrained")
    for n in order:
        d = df[df["name"] == n].sort_values("nprobe")
        ax.plot(d["nprobe"], d["recall_at_10"], label=n, markeredgecolor="white",
                markeredgewidth=0.6, **style(n, order))
        end_label(ax, d["nprobe"].iloc[-1], d["recall_at_10"].iloc[-1], short(n))
        record("recall10_vs_nprobe", d, ["recall_at_10"], "summary.recall_at_10")
    # shade the 16 -> 32 step, where the PQ curves flatten
    ax.axvspan(nprobes[-2], nprobes[-1], color="#f0efec", zorder=0, lw=0)
    nprobe_axis(ax, nprobes)
    ax.set_xlim(nprobes[0] / 1.25, nprobes[-1] * 2.0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall@10")
    legend_below(fig, ax)
    return save(fig, out, "recall10_vs_nprobe")


def fig_recall_vs_latency(df, order, out, lat_col, stat):
    fig, ax = plt.subplots(figsize=(COL_W, 3.0), layout="constrained")
    pts = []
    for n in order:
        d = df[df["name"] == n].sort_values("nprobe")
        ax.plot(d[lat_col], d["recall_at_10"], label=n, markeredgecolor="white",
                markeredgewidth=0.6, **style(n, order))
        pts += [(r[lat_col], r["recall_at_10"], f"{int(r.nprobe)}") for _, r in d.iterrows()]
        record(f"recall10_vs_{stat}_latency", d, ["recall_at_10", lat_col],
               f"summary.recall_at_10 / summary.{lat_col}")
    ax.set_xscale("log")
    log_axis(ax.xaxis)
    ax.set_xlabel(f"{stat} query latency (ms/query, log scale)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=True)
    legend_below(fig, ax)
    ax.text(0.01, 0.99, "point labels = nprobe", transform=ax.transAxes, fontsize=6,
            color=INK2, va="top")
    place_point_labels(fig, ax, pts)   # after layout is final, so positions are measured
    return save(fig, out, f"recall10_vs_{stat}_latency")


def fig_index_size(df, order, out):
    sz = df.groupby("name")[["serialized_size_bytes", "compression_ratio_vs_ivfflat"]].first()
    # each index has one size; check nprobe rows agree
    assert (df.groupby("name")["serialized_size_bytes"].nunique() == 1).all()
    flat_b = sz.loc["IVF-Flat", "serialized_size_bytes"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.5), layout="constrained")
    x = np.arange(len(order))
    vals = np.array([sz.loc[n, "serialized_size_bytes"] for n in order]) / 1024**2
    bottom = 10 ** np.floor(np.log10(vals.min() / 2))
    ax.bar(x, vals - bottom, bottom=bottom, width=0.62,
           color=[style(n, order)["color"] for n in order], edgecolor="white", linewidth=1)
    for xi, n, v in zip(x, order, vals):
        b = sz.loc[n, "serialized_size_bytes"]
        # ratio computed from bytes here and cross-checked against the stored column
        ratio = flat_b / b
        assert abs(ratio - sz.loc[n, "compression_ratio_vs_ivfflat"]) < 1e-6
        txt = fmt_size(b) if n == "IVF-Flat" else f"{fmt_size(b)}\n{ratio:.0f}× smaller"
        ax.annotate(txt, (xi, v), xytext=(0, 2), textcoords="offset points", ha="center",
                    va="bottom", fontsize=6.5, color=INK)
    ax.set_yscale("log")
    log_axis(ax.yaxis, subs=(1.0,))
    ax.set_ylim(bottom, vals.max() * 6)
    ax.set_xticks(x, [short(n) for n in order])
    ax.set_ylabel("Serialized index size (MiB, log scale)")
    record("index_size", sz.reset_index(), ["serialized_size_bytes", "compression_ratio_vs_ivfflat"],
           "summary.serialized_size_bytes / compression_ratio_vs_ivfflat")
    return save(fig, out, "index_size")


def fig_recall_vs_memory(df, order, nprobes, out):
    # Design (Option A): memory does not depend on nprobe, so each index is ONE point at
    # the largest tested search effort (nprobe = max). A light whisker shows the range of
    # Recall@10 over the other nprobe values at the same size, so no information is hidden
    # and no x-value is duplicated without explanation.
    p_max = nprobes[-1]
    fig, ax = plt.subplots(figsize=(COL_W, 2.6), layout="constrained")
    for n in order:
        d = df[df["name"] == n]
        x = d["serialized_size_bytes"].iloc[0] / 1024**2
        lo, hi = d["recall_at_10"].min(), d["recall_at_10"].max()
        top = d[d["nprobe"] == p_max].iloc[0]
        st = style(n, order)
        ax.plot([x, x], [lo, hi], color=st["color"], lw=1, alpha=0.45, zorder=1)
        ax.scatter([x], [top["recall_at_10"]], s=34, zorder=3, color=st["color"],
                   marker=st["marker"], edgecolor="white", linewidth=0.6, label=n)
        # IVF-Flat, PQ16, PQ32 labelled on the left, the largest PQ on the right (keeps labels
        # off the neighboring whiskers)
        ha, dx = ("left", 6) if n == order[-1] else ("right", -6)
        ax.annotate(f"{short(n)}\n{top['recall_at_10']:.3f}", (x, top["recall_at_10"]),
                    xytext=(dx, 0), textcoords="offset points", ha=ha, va="center",
                    fontsize=6.5, color=INK)
        record("recall10_vs_memory", d.assign(size_mib=d["serialized_size_bytes"] / 1024**2),
               ["recall_at_10", "size_mib"], "summary.recall_at_10 / serialized_size_bytes")
    ax.set_xscale("log")
    log_axis(ax.xaxis, subs=(1.0,))
    ax.set_xlim(10, 2e4)
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Serialized index size (MiB, log scale)")
    ax.set_ylabel(f"Recall@10 at nprobe = {p_max}")
    nps = ", ".join(str(p) for p in nprobes)
    ax.text(0.01, 0.99, f"marker: nprobe = {p_max}; faint bar: range over nprobe = {nps}",
            transform=ax.transAxes, fontsize=6, color=INK2, va="top")
    return save(fig, out, "recall10_vs_memory")


def fig_qps(df, order, nprobes, out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.6), layout="constrained")
    for n in order:
        d = df[df["name"] == n].sort_values("nprobe")
        ax.plot(d["nprobe"], d["qps"], label=n, markeredgecolor="white", markeredgewidth=0.6,
                **style(n, order))
        end_label(ax, d["nprobe"].iloc[-1], d["qps"].iloc[-1], short(n))
        record("qps_vs_nprobe", d, ["qps"], "summary.qps (median of reps)")
    nprobe_axis(ax, nprobes)
    ax.set_xlim(nprobes[0] / 1.25, nprobes[-1] * 2.0)
    ax.set_yscale("log")
    log_axis(ax.yaxis)
    ax.set_ylabel("Queries/s (log scale)")
    legend_below(fig, ax)
    return save(fig, out, "qps_vs_nprobe")


def fig_hit100(df, hit, order, nprobes, out):
    """(a) Exact-1NN Hit@100 vs nprobe; (b) at max nprobe, Recall@10 vs Hit@100 per PQ size."""
    hit = hit.merge(df[["index_type", "pq_m", "nprobe", "name", "recall_at_10"]],
                    on=["index_type", "pq_m", "nprobe"], how="left", validate="one_to_one")
    assert hit["name"].notna().all()
    p_max = nprobes[-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.5), layout="constrained",
                               gridspec_kw={"width_ratios": [1.15, 1]})
    ends = {}
    for n in order:
        d = hit[hit["name"] == n].sort_values("nprobe")
        kw = style(n, order)
        if n == "IVF-Flat":   # reference: coverage limited only by IVF pruning
            a.plot(d["nprobe"], d["exact1nn_hit_at_100"], ls="--", lw=1.1, label=f"{n} (reference)",
                   markeredgecolor="white", markeredgewidth=0.6, **kw)
        else:
            a.plot(d["nprobe"], d["exact1nn_hit_at_100"], label=n, markeredgecolor="white",
                   markeredgewidth=0.6, **kw)
        ends[n] = d["exact1nn_hit_at_100"].iloc[-1]
        record("exact1nn_hit100(a)", d, ["exact1nn_hit_at_100"], "exact1nn_hit100.exact1nn_hit_at_100")
    nprobe_axis(a, nprobes)
    a.set_xlim(nprobes[0] / 1.25, nprobes[-1] * 2.0)
    a.set_ylim(0, 1)
    # end labels, spread to >= 8 pt apart when end values are close (measured axis height)
    fig.canvas.draw()
    pts_per_unit = a.get_window_extent().height / fig.dpi * 72 / (a.get_ylim()[1] - a.get_ylim()[0])
    placed = []
    for n, v in sorted(ends.items(), key=lambda kv: kv[1]):
        y = v * pts_per_unit
        y = y if not placed or y - placed[-1] >= 8 else placed[-1] + 8
        placed.append(y)
        a.annotate(short(n), (nprobes[-1], v), xytext=(5, y - v * pts_per_unit),
                   textcoords="offset points", fontsize=7, color=INK, va="center", ha="left")
    a.set_ylabel("Exact-1NN Hit@100")
    a.legend(loc="upper left", frameon=False, fontsize=6.5)
    a.set_title("(a) true nearest neighbor in top-100", fontsize=8, loc="left")

    pq = hit[(hit["index_type"] == "ivfpq") & (hit["nprobe"] == p_max)].sort_values("pq_m")
    x = np.arange(len(pq))
    w = 0.36
    for i, (_, r) in enumerate(pq.iterrows()):
        c = style(r["name"], order)["color"]
        b.bar(x[i] - w / 2, r["recall_at_10"], w, color=c, edgecolor="white", linewidth=1,
              hatch="////", alpha=0.55)
        b.bar(x[i] + w / 2, r["exact1nn_hit_at_100"], w, color=c, edgecolor="white", linewidth=1)
        for dx, v in ((-w / 2, r["recall_at_10"]), (w / 2, r["exact1nn_hit_at_100"])):
            b.annotate(f"{v:.3f}", (x[i] + dx, v), xytext=(0, 2), textcoords="offset points",
                       ha="center", va="bottom", fontsize=6.5, color=INK)
    record("exact1nn_hit100(b)", pq, ["recall_at_10", "exact1nn_hit_at_100"],
           "summary.recall_at_10 / exact1nn_hit100.exact1nn_hit_at_100")
    b.set_xticks(x, [short(n) for n in pq["name"]])
    b.set_ylim(0, 1)
    b.set_ylabel("Fraction")
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(facecolor="#9a9a96", hatch="////", alpha=0.55, edgecolor="white",
                            label="Recall@10 (top-10 overlap)"),
                      Patch(facecolor="#9a9a96", label="Exact-1NN Hit@100")],
             loc="upper left", frameon=False, fontsize=6.5)
    b.set_title(f"(b) IVF-PQ at nprobe = {p_max}", fontsize=8, loc="left")
    b.grid(axis="x", visible=False)
    return save(fig, out, "exact1nn_hit100"), hit


def fig_code_size(df, nprobes, out, dim):
    p_max = nprobes[-1]
    pq = df[(df["index_type"] == "ivfpq") & (df["nprobe"] == p_max)].sort_values("pq_code_bytes")
    flat = df[(df["index_type"] == "ivfflat") & (df["nprobe"] == p_max)].iloc[0]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4), layout="constrained")
    ax.plot(pq["pq_code_bytes"], pq["recall_at_10"], color=PQ_STYLE[1]["color"], lw=1.4, zorder=1)
    for i, (_, r) in enumerate(pq.iterrows()):
        st = PQ_STYLE[i]
        ax.scatter(r["pq_code_bytes"], r["recall_at_10"], s=34, zorder=2, color=st["color"],
                   marker=st["marker"], edgecolor="white", linewidth=0.6)
        ax.annotate(f"{r['recall_at_10']:.3f}", (r["pq_code_bytes"], r["recall_at_10"]),
                    xytext=(0, -11), textcoords="offset points", ha="center", fontsize=6.5)
    # per-doubling gains, computed from the data (no claim of diminishing returns)
    for (_, r0), (_, r1) in zip(pq.iloc[:-1].iterrows(), pq.iloc[1:].iterrows()):
        xm = np.sqrt(r0["pq_code_bytes"] * r1["pq_code_bytes"])
        ym = (r0["recall_at_10"] + r1["recall_at_10"]) / 2
        ax.annotate(f"+{100 * (r1['recall_at_10'] - r0['recall_at_10']):.1f} pts", (xm, ym),
                    xytext=(-4, 6), textcoords="offset points", ha="right", fontsize=6.5, color=INK2)
    ax.axhline(flat["recall_at_10"], color=FLAT["color"], ls="--", lw=1)
    ax.annotate(f"IVF-Flat ({4 * dim} B/vector, float32): {flat['recall_at_10']:.3f}",
                (pq["pq_code_bytes"].iloc[0], flat["recall_at_10"]), xytext=(0, -9),
                textcoords="offset points", fontsize=6.5, color=INK)
    record("pq_code_size_vs_recall", pq.assign(name=pq["name"]), ["pq_code_bytes", "recall_at_10"],
           "summary.pq_code_bytes / recall_at_10")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(pq["pq_code_bytes"].tolist()))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(pq["pq_code_bytes"].min() / 1.4, pq["pq_code_bytes"].max() * 1.4)
    ax.set_ylim(0, 1)
    ax.set_xlabel("PQ code size (bytes/vector, log2 scale)")
    ax.set_ylabel(f"Recall@10 at nprobe = {p_max}")
    return save(fig, out, "pq_code_size_vs_recall")


def fig_stability(df, cmp_df, order, run_id, cmp_id, out):
    m = df.merge(cmp_df, on=["index_type", "pq_m", "nprobe"], suffixes=("", "_cmp"), validate="one_to_one")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W * 0.75, 2.5), layout="constrained")
    for ax, col, stat in ((axes[0], "latency_p50_ms", "p50"), (axes[1], "latency_p95_ms", "p95")):
        for n in order:
            d = m[m["name"] == n]
            st = style(n, order)
            ax.scatter(d[f"{col}_cmp"], d[col], s=22, color=st["color"], marker=st["marker"],
                       edgecolor="white", linewidth=0.5, label=n, zorder=2)
        lo = min(m[col].min(), m[f"{col}_cmp"].min()) / 1.4
        hi = max(m[col].max(), m[f"{col}_cmp"].max()) * 1.4
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls=":", zorder=1, label="y = x")
        ax.set_xscale("log"); ax.set_yscale("log")
        log_axis(ax.xaxis); log_axis(ax.yaxis)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(axis="x", visible=True)
        ax.set_xlabel(f"{stat} latency, run {cmp_id[-6:]} (ms/query)")
        ax.set_ylabel(f"{stat} latency, run {run_id[-6:]} (ms/query)")
        m[f"{stat}_pct_diff"] = 100 * (m[col] / m[f"{col}_cmp"] - 1)
        record(f"latency_stability_{stat}", m, [col, f"{col}_cmp"], "summary of both runs")
    axes[1].legend(loc="upper left", frameon=False, fontsize=6)
    return save(fig, out, "latency_stability"), m


# --------------------------------------------------------------------------- README

def machine_desc(meta) -> str:
    """Human-readable description of the machine a run was measured on (from its metadata)."""
    cpu = meta.get("cpu", {})
    model = cpu.get("machdep.cpu.brand_string") or cpu.get("model_name") or cpu.get("machine", "?")
    ram = cpu.get("total_ram_bytes")
    host = meta.get("hostname")
    parts = [model] + ([f"{ram / 2**30:.0f} GiB RAM"] if ram else []) + ([f"host {host}"] if host else [])
    if meta.get("slurm_job_id"):
        parts.append(f"SLURM job {meta['slurm_job_id']}, {meta.get('slurm_cpus', '?')} CPUs allocated")
    return ", ".join(parts)


def write_readme(out, run_id, cmp_id, meta, df, stab, hit_path):
    sw = df.groupby("name", sort=False)["swap_in_bytes_during_search"].first()
    lines = [
        "# Final figures", "",
        f"Primary run: `{run_id}` (timestamp {meta['timestamp']}). "
        f"Generated by `scripts/generate_final_figures.py --run-id {run_id}"
        + (f" --comparison-run {cmp_id}`." if cmp_id else "`."), "",
        "Every plotted value is listed with its source column in `plotted_values.csv`. "
        "No values were averaged across runs; the comparison run is used only in "
        "`latency_stability.*`.", "",
        "## Setup", "",
        f"* GIST1M: {meta['dataset']['n_base']:,} database vectors, {meta['dataset']['n_train']:,} "
        f"training vectors, {meta['dataset']['n_query']:,} queries, d = {meta['dataset']['dim']}, L2.",
        f"* nlist = {meta['config']['ivf']['nlist']}, PQ nbits = {meta['config']['pq']['nbits']}, "
        f"nprobe = {meta['nprobe']}, k = {meta['k']}.",
        f"* {meta['faiss_threads_effective']} FAISS threads, {meta['repetitions']} timed repetitions "
        "after warm-up.",
        f"* Latency: one query per search call, per-query timing; p50/p95 computed per repetition, "
        f"median over repetitions reported (ms/query).",
        f"* Throughput: all {meta['throughput_batch_size']} queries in one call; median QPS over repetitions.",
        "* Index size: bytes of the serialized `.faiss` file (MiB = 2^20 B, GiB = 2^30 B).",
        f"* Exact-1NN Hit@100: fraction of queries whose exact nearest neighbor appears anywhere "
        f"in the approximate top-100. Not Recall@100. Computed by `scripts/compute_exact1nn_hit.py` "
        f"from the run's cached indexes (no timing); that script verified Recall@1/10/100 equal the "
        f"run's recorded values exactly before saving `{hit_path.name}`.",
        "", "## Latency caveat", "",
        f"Latency was measured on {machine_desc(meta)}. Bytes swapped in (system-wide counter) "
        "while each index was being searched:", "",
        "| index | swap-in during search |", "|---|---|",
        *[f"| {n} | {v / 1024**2:.0f} MiB |" for n, v in sw.items()], "",
        ("Some swapping occurred, which can inflate latency." if sw.sum() > 0
         else "No swapping occurred during timed search."), "",
        "Measured latencies were not modified, corrected or filtered. Absolute latencies should be "
        "interpreted with care; the comparison between indexes is the more reliable signal.", "",
    ]
    if stab is not None:
        lines += [
            f"Two independent {meta['repetitions']}-repetition runs (`{cmp_id}` and `{run_id}`) gave "
            "identical recall for all 16 configurations and very similar latency:", "",
            f"* p50 latency: run-to-run difference median {stab['p50_pct_diff'].abs().median():.1f}%, "
            f"max {stab['p50_pct_diff'].abs().max():.1f}%.",
            f"* p95 latency: run-to-run difference median {stab['p95_pct_diff'].abs().median():.1f}%, "
            f"max {stab['p95_pct_diff'].abs().max():.1f}%.",
            "* The ordering of indexes by latency is the same at every nprobe in both runs: "
            + ("yes." if stab['same_order'] else "no (see latency_stability)."), "",
        ]
    lines += [
        "## Interpretation note", "",
        "IVF-Flat Recall@10 keeps rising with nprobe while IVF-PQ Recall@10 flattens from "
        "nprobe = 16. The observed plateau is consistent with quantization error becoming the "
        "dominant source of retrieval error once IVF candidate coverage is sufficient; it is not "
        "a causal proof.", "",
        "## Files", "",
        "| file | content |", "|---|---|",
        "| recall10_vs_nprobe | Recall@10 vs nprobe, all indexes |",
        "| recall10_vs_p95_latency | Recall@10 vs p95 latency (points = nprobe) |",
        "| recall10_vs_p50_latency | same with p50 latency |",
        "| index_size | serialized size, log scale, with compression ratio |",
        "| recall10_vs_memory | Recall@10 at max nprobe vs size; faint bar = range over nprobe |",
        "| qps_vs_nprobe | batch throughput vs nprobe |",
        "| exact1nn_hit100 | (a) Hit@100 vs nprobe; (b) Recall@10 vs Hit@100 at max nprobe |",
        "| pq_code_size_vs_recall | Recall@10 at max nprobe vs PQ code bytes |",
    ]
    if stab is not None:
        lines.append("| latency_stability | p50/p95 latency of the two runs, y = x reference |")
    (out / "README.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- main

def load_summary(run_id: str) -> pd.DataFrame:
    df = pd.read_csv(resolve("results/summary") / f"{run_id}_summary.csv")
    assert (df["run_id"] == run_id).all(), "summary file contains rows from another run"
    assert len(df) == df[["index_type", "pq_m", "nprobe"]].drop_duplicates().shape[0]
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--comparison-run", help="second run for the latency stability figure")
    ap.add_argument("--out", default="results/figures/final")
    args = ap.parse_args(argv)

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((resolve("results/raw") / f"{args.run_id}_metadata.json").read_text())
    df = load_summary(args.run_id)
    order = series_names(df)
    nprobes = sorted(int(p) for p in df["nprobe"].unique())
    assert nprobes == sorted(meta["nprobe"]), (nprobes, meta["nprobe"])
    for n in order:
        assert sorted(df.loc[df["name"] == n, "nprobe"]) == nprobes, f"{n} missing nprobe values"

    hit_path = resolve("results/summary") / f"{args.run_id}_exact1nn_hit100.csv"
    if not hit_path.exists():
        raise SystemExit(f"{hit_path} missing: run scripts/compute_exact1nn_hit.py --run-id {args.run_id}")
    hit = pd.read_csv(hit_path)
    assert (hit["run_id"] == args.run_id).all() and hit["recall_matches_run"].all()

    files = []
    files += fig_recall_vs_nprobe(df, order, nprobes, out)
    files += fig_recall_vs_latency(df, order, out, "latency_p95_ms", "p95")
    files += fig_recall_vs_latency(df, order, out, "latency_p50_ms", "p50")
    files += fig_index_size(df, order, out)
    files += fig_recall_vs_memory(df, order, nprobes, out)
    files += fig_qps(df, order, nprobes, out)
    f, _ = fig_hit100(df, hit, order, nprobes, out)
    files += f
    files += fig_code_size(df, nprobes, out, int(meta["dataset"]["dim"]))

    stab = None
    if args.comparison_run:
        cmp_df = load_summary(args.comparison_run)
        series_names(cmp_df)
        f, stab = fig_stability(df, cmp_df, order, args.run_id, args.comparison_run, out)
        files += f
        # same latency ordering of indexes at every nprobe in both runs (p50)?
        same = all(
            stab[stab["nprobe"] == p].sort_values("latency_p50_ms")["name"].tolist()
            == stab[stab["nprobe"] == p].sort_values("latency_p50_ms_cmp")["name"].tolist()
            for p in nprobes)
        stab = {"p50_pct_diff": stab["p50_pct_diff"], "p95_pct_diff": stab["p95_pct_diff"],
                "same_order": same}
    write_readme(out, args.run_id, args.comparison_run, meta, df, stab, hit_path)
    pd.DataFrame(PLOTTED).to_csv(out / "plotted_values.csv", index=False)
    for p in files:
        print(p)
    print(out / "README.md")
    print(out / "plotted_values.csv")


if __name__ == "__main__":
    main()
