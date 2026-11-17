"""Publication figures from a summary CSV (matplotlib only).

Encoding: IVF-Flat (baseline) = orange, squares. IVF-PQ is an ordered family, so
M=16/32/64 use one blue hue light->dark (validated ordinal ramp) with distinct markers.
Every curve is also direct-labelled, so identity never relies on colour alone.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator  # noqa: E402

log = logging.getLogger(__name__)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FLAT_COLOR = "#eb6834"
PQ_RAMP = ["#86b6ef", "#2a78d6", "#104281"]          # blue steps 250 / 450 / 650
PQ_MARKERS = ["o", "^", "D"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 6, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def styles(df: pd.DataFrame) -> dict[str, dict]:
    """label -> plotting style, in fixed order (Flat, PQ by increasing M)."""
    out: dict[str, dict] = {}
    if (df["index_type"] == "ivfflat").any():
        out["IVF-Flat"] = dict(color=FLAT_COLOR, marker="s", short="IVF-Flat")
    ms = sorted(int(m) for m in df.loc[df["index_type"] == "ivfpq", "pq_m"].dropna().unique())
    for i, m in enumerate(ms):
        j = min(i, len(PQ_RAMP) - 1) if len(ms) <= 3 else int(round(i * 2 / (len(ms) - 1)))
        out[f"IVF-PQ M={m}"] = dict(color=PQ_RAMP[j], marker=PQ_MARKERS[j % 3], short=f"PQ-{m}")
    return out


def _nprobe_axis(ax, nprobes):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(nprobes))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(p) for p in nprobes])
    ax.set_xlabel("nprobe (inverted lists searched)")


def _readable_log(axis, subs=(1.0, 2.0, 5.0)):
    """Label log axes at 1-2-5 steps with plain numbers instead of 10^x."""
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _end_label(ax, x, y, text, color, dx=4, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                color=INK, va="center", ha="left")


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    log.info("Figure saved: %s.{pdf,png}", out_dir / name)
    return paths


def _fmt_bytes(b: float) -> str:
    return f"{b / 1024**3:.2f} GiB" if b >= 1024**3 else f"{b / 1024**2:.1f} MiB"


# --------------------------------------------------------------------------- figures

def plot_recall_vs_latency(df, out_dir, latency_col="latency_p95_ms", recall_cols=("recall_at_10", "recall_at_100")):
    st = styles(df)
    fig, axes = plt.subplots(1, len(recall_cols), figsize=(3.4 * len(recall_cols), 3.0), sharex=True)
    axes = np.atleast_1d(axes)
    stat = latency_col.split("_")[1]
    for ax, rc in zip(axes, recall_cols):
        for lab, s in st.items():
            d = df[df["label"] == lab].sort_values("nprobe")
            ax.plot(d[latency_col], d[rc], color=s["color"], marker=s["marker"], label=lab,
                    markeredgecolor="white", markeredgewidth=0.8)
            _end_label(ax, d[latency_col].iloc[-1], d[rc].iloc[-1], s["short"], s["color"])
        flat = df[df["label"] == "IVF-Flat"].sort_values("nprobe")
        for _, r in flat.iterrows():
            ax.annotate(f"{int(r.nprobe)}", (r[latency_col], r[rc]), xytext=(-3, 6),
                        textcoords="offset points", fontsize=7, color=INK2, ha="right")
        ax.set_xscale("log")
        _readable_log(ax.xaxis)
        ax.set_xlabel(f"{stat} query latency (ms/query, log scale)")
        k = rc.split("_")[-1]
        ax.set_ylabel(f"Recall@{k}")
        ax.set_title(f"Recall@{k} vs {stat} latency")
        ax.margins(x=0.15)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, 0.03))
    nps = ", ".join(str(int(p)) for p in sorted(df["nprobe"].unique()))
    fig.text(0.5, -0.01, f"Points along each curve: nprobe = {nps} (left to right; labelled on IVF-Flat).",
             fontsize=7, color=INK2, ha="center")
    return _save(fig, out_dir, f"fig1_recall_vs_{stat}_latency")


def plot_recall_vs_memory(df, out_dir, recall_col="recall_at_10"):
    """Each index has ONE size; its nprobe operating points stack vertically at that size."""
    st = styles(df)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for lab, s in st.items():
        d = df[df["label"] == lab].sort_values("nprobe")
        x = d["serialized_size_bytes"].iloc[0] / 1024**2
        assert d["serialized_size_bytes"].nunique() == 1
        ax.plot([x] * len(d), d[recall_col], color=s["color"], lw=1.2, zorder=1)
        ax.scatter([x] * len(d), d[recall_col], color=s["color"], marker=s["marker"], s=36,
                   edgecolor="white", linewidth=0.8, zorder=2, label=lab)
        if lab == "IVF-Flat":   # nprobe labels on one series only (order is monotone for all)
            for _, r in d.iterrows():
                ax.annotate(f"np={int(r.nprobe)}", (x, r[recall_col]), xytext=(6, 0),
                            textcoords="offset points", fontsize=7, color=INK2, va="center")
            ax.annotate(s["short"], (x, d[recall_col].max()), xytext=(0, 9),
                        textcoords="offset points", fontsize=8, color=INK, ha="center")
        # PQ series: identified by marker + legend (too close on the log axis for names)
    ax.set_xscale("log")
    _readable_log(ax.xaxis, subs=(1.0,))
    ax.set_xlabel("Serialized index size (MiB, log scale)")
    k = recall_col.split("_")[-1]
    ax.set_ylabel(f"Recall@{k}")
    ax.set_title(f"Recall@{k} vs index size")
    nps = ", ".join(str(int(p)) for p in sorted(df["nprobe"].unique()))
    fig.text(0.5, -0.02, f"Each index has one size; its dots (bottom to top) are nprobe = {nps}.",
             fontsize=7, color=INK2, ha="center")
    ax.margins(x=0.3, y=0.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=4, frameon=False,
              handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, f"fig2_recall{k}_vs_memory")


def plot_index_size(df, out_dir):
    """Linear bars (honest lengths). Right panel zooms into the PQ indexes, which are
    too small to see next to IVF-Flat."""
    st = styles(df)
    sizes = df.groupby("label", sort=False)["serialized_size_bytes"].first()
    labels = [lab for lab in st if lab in sizes.index]
    flat = sizes.get("IVF-Flat", np.nan)
    pq_labels = [l for l in labels if l != "IVF-Flat"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw={"width_ratios": [4, 3]})
    for ax, labs, title in ((axes[0], labels, "All indexes"), (axes[1], pq_labels, "IVF-PQ only (zoom)")):
        vals = np.array([sizes[l] for l in labs], dtype=float) / 1024**2
        x = np.arange(len(labs))
        ax.bar(x, vals, width=0.6, color=[st[l]["color"] for l in labs], edgecolor="white", linewidth=2)
        for xi, v, lab in zip(x, vals, labs):
            b = v * 1024**2
            txt = _fmt_bytes(b) if lab == "IVF-Flat" else f"{_fmt_bytes(b)}\n{flat / b:.0f}× smaller"
            ax.annotate(txt, (xi, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(x, [st[l]["short"] for l in labs])
        ax.set_ylim(0, vals.max() * 1.3)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Serialized index size (MiB)")
    fig.suptitle("Index size on disk", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_index_size")


def plot_vs_nprobe(df, out_dir, ycol, ylabel, title, name, logy=False):
    st = styles(df)
    nprobes = sorted(df["nprobe"].unique())
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for lab, s in st.items():
        d = df[df["label"] == lab].sort_values("nprobe")
        ax.plot(d["nprobe"], d[ycol], color=s["color"], marker=s["marker"], label=lab,
                markeredgecolor="white", markeredgewidth=0.8)
        _end_label(ax, d["nprobe"].iloc[-1], d[ycol].iloc[-1], s["short"], s["color"], dx=6)
    _nprobe_axis(ax, nprobes)
    ax.set_xlim(nprobes[0] / 1.3, nprobes[-1] * 2.2)
    if logy:
        ax.set_yscale("log")
        _readable_log(ax.yaxis)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def make_all(summary_csv: str | Path, out_dir: str | Path) -> list[Path]:
    df = pd.read_csv(summary_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    paths += plot_recall_vs_latency(df, out_dir, "latency_p95_ms")
    paths += plot_recall_vs_latency(df, out_dir, "latency_p50_ms")
    paths += plot_recall_vs_memory(df, out_dir, "recall_at_10")
    paths += plot_index_size(df, out_dir)
    paths += plot_vs_nprobe(df, out_dir, "recall_at_10", "Recall@10", "Recall@10 vs nprobe",
                            "fig4_recall10_vs_nprobe")
    paths += plot_vs_nprobe(df, out_dir, "recall_at_100", "Recall@100", "Recall@100 vs nprobe",
                            "fig4b_recall100_vs_nprobe")
    paths += plot_vs_nprobe(df, out_dir, "latency_p95_ms", "p95 latency (ms/query, log scale)",
                            "p95 query latency vs nprobe", "fig5_p95_latency_vs_nprobe", logy=True)
    paths += plot_vs_nprobe(df, out_dir, "qps", "Throughput (queries/s, log scale)",
                            f"Batch throughput vs nprobe", "fig5b_qps_vs_nprobe", logy=True)
    return paths
