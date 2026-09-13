"""
Compute latency and cost statistics (avg, p50, p90, p99) for TPC-H and TPC-DS,
plot 2x2: top row TPC-H (Latency | Cost), bottom row TPC-DS (Latency | Cost).
Cost is execution_time * thread_count (not read from *_cost columns).
TPC-H: linear y. TPC-DS: y from 0 with ticks 0,1,2,4,8,...; consecutive ticks are equally spaced
visually (piecewise-linear map between tick values; not a raw log axis, so 0 is included).
Hatch style from draw/multi_stack_bar.py.

Usage: python scripts/stats_latency_cost.py
Output PDF: output/evaluations/stats_latency_cost.pdf
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

TPCH_CSV = "output/evaluations/tpch_graph.csv"
TPCDS_CSV = "output/evaluations/tpcds_graph_corrected.csv"

# Single font size for all text (ticks, axis labels, legend, row captions).
FONT_SIZE = 15

pylab.rcParams.update({
    # Latin: Times New Roman; CJK: Song (SimSun) via per-glyph fallback (matplotlib 3.6+)
    "font.family": ["Times New Roman", "SimSun", "NSimSun", "STSong"],
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "axes.unicode_minus": False,
})
plt.rcParams["pdf.fonttype"] = 42

# Style: 3 methods (no Base) with colors + hatches (aligned with run_separate_graph_exp1.py)
COLORS = ["C3", "C1", "C0"]
HATCHES = ["xx", "..", "++"]
# Same display names as run_separate_graph_exp1.py (CSV columns remain PPM_/FGO_/OPRO_)
METHOD_LABELS = ["Query-Uniform", "Stage-Independent", "RePro"]
STAT_LABELS = ["平均", "p50", "p90", "p99"]
BAR_WIDTH = 0.2
EDGE_LW = 0.7
# Drop samples with execution time above this threshold (ms); cost uses the same rows (time * threads).
MAX_LATENCY_MS = 1000000

# Fewer y-ticks avoids crowding when the dynamic range is large (e.g. TPC-DS).
Y_TICK_NBINS = 6


def _format_tick_plain_number(v):
    """Plain tick labels; integers shown without trailing .0"""
    if not np.isfinite(v):
        return ""
    r = round(float(v))
    if abs(v - r) < 1e-5 * max(1.0, abs(v)):
        return str(int(r))
    av = abs(v)
    if av >= 1000:
        return f"{v:.4g}"
    return f"{v:.2g}"


def _tick_levels_zero_then_pow2(ymax, max_ticks=7):
    """
    Tick values: 0, 1, then 2, 4, 8, ...; subsample when long (fewer labels).
    """
    ymax = max(float(ymax), 1e-30)
    out = [0.0, 1.0]
    if ymax <= 1.0:
        out.append(min(ymax * 1.05, 1.0))
        return np.unique(np.sort(np.asarray(out, dtype=float)))
    p = 2.0
    while p <= ymax * 1.08:
        out.append(p)
        p *= 2.0
    out = np.asarray(out, dtype=float)
    if len(out) > max_ticks:
        head = out[:2]
        rest = out[2:]
        step = int(np.ceil(len(rest) / float(max_ticks - 2)))
        rest = rest[::step]
        if rest[-1] < out[-1]:
            rest = np.append(rest, out[-1])
        out = np.unique(np.concatenate([head, rest]))
    return out


def _map_y_to_equal_tick_spacing(y, tick_levels):
    """
    Map nonnegative data y to [0,1] so tick_levels are uniformly spaced on the axis.
    Linear interpolation between consecutive tick_levels.
    """
    t = np.asarray(tick_levels, dtype=float)
    n = len(t)
    if n < 2:
        return np.zeros_like(y, dtype=float)
    disp = np.linspace(0.0, 1.0, n)
    y = np.maximum(np.asarray(y, dtype=float), 0.0)
    idx = np.searchsorted(t, y, side="right") - 1
    idx = np.clip(idx, 0, n - 2)
    t0, t1 = t[idx], t[idx + 1]
    d0, d1 = disp[idx], disp[idx + 1]
    denom = t1 - t0
    frac = np.where(denom > 1e-30, (y - t0) / denom, 0.0)
    return d0 + frac * (d1 - d0)


def compute_stats_numeric(series):
    """Return (avg, p50, p90, p99) as floats."""
    return (
        float(series.mean()),
        float(series.quantile(0.50)),
        float(series.quantile(0.90)),
        float(series.quantile(0.99)),
    )


def repro_reduction_vs_baselines(time_m, cost_m, dataset_label):
    """
    Print and save RePro (OPRO, index 2) reduction vs Query-Uniform / Stage-Independent.
    Reduction % = (baseline - RePro) / baseline * 100 (positive => RePro lower).
    """
    baselines = (("Query-Uniform", 0), ("Stage-Independent", 1))
    repro_i = 2
    stat_names = STAT_LABELS
    lines = [
        f"RePro (OPRO) vs baselines — {dataset_label}",
        "(Positive % means RePro is lower than the baseline.)",
        "",
    ]
    for metric_name, mat in (("延迟", time_m), ("成本", cost_m)):
        lines.append(f"  [{metric_name}]")
        for bname, bidx in baselines:
            row_parts = []
            for j, sn in enumerate(stat_names):
                b = float(mat[bidx, j])
                r = float(mat[repro_i, j])
                if b > 0 and np.isfinite(b) and np.isfinite(r):
                    pct = (b - r) / b * 100.0
                    row_parts.append(f"{sn} {pct:+.2f}%")
                else:
                    row_parts.append(f"{sn} n/a")
            lines.append(f"    vs {bname}:  " + " | ".join(row_parts))
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    print(text)
    return text


def build_stats_matrix(df, methods, time_cols, thread_cols):
    """
    Return (time_matrix, cost_matrix), each shape (n_methods, 4) for avg/p50/p90/p99.
    Cost per row = execution_time * thread_count (ms * threads, same product unit as typical CSV cost).
    """
    time_matrix = np.zeros((len(methods), 4))
    cost_matrix = np.zeros((len(methods), 4))
    for i, (meth, tcol, thcol) in enumerate(zip(methods, time_cols, thread_cols)):
        t = df[tcol]
        th = df[thcol]
        valid = t.notna() & th.notna() & (t <= MAX_LATENCY_MS)
        time_matrix[i, :] = compute_stats_numeric(df.loc[valid, tcol])
        cost = df.loc[valid, tcol].astype(float) * df.loc[valid, thcol].astype(float)
        cost_matrix[i, :] = compute_stats_numeric(cost)
    return time_matrix, cost_matrix


def plot_bar_subplot(
    ax,
    matrix,
    ylabel,
    value_scale=1.0,
    use_log_scale=False,
    y_tick_nbins=Y_TICK_NBINS,
    y_display="linear",
):
    """
    Vertical bar chart. matrix: (n_methods, 4) for avg, p50, p90, p99.
    value_scale: divide values by this for display (e.g. 1e6 for cost).
    use_log_scale: log10 y-axis (legacy; prefer y_display).
    y_display: "linear" or "log2" (ticks 0,1,2,4,...; equal spacing in display; piecewise linear map).
    y_tick_nbins: max tick intervals on linear y.
    Style: hatch + white edge, then black border overlay (multi_stack_bar.py).
    """
    n_methods, n_stats = matrix.shape
    x = np.arange(n_stats)
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * BAR_WIDTH
    vals_scaled = matrix / value_scale
    use_log2 = y_display == "log2" and not use_log_scale

    pos = vals_scaled[np.isfinite(vals_scaled) & (vals_scaled > 0)]
    # Tiny floor only for exact zeros on legacy log10 axis.
    eps0 = max(float(np.nanmax(vals_scaled)) * 1e-15, 1e-30) if vals_scaled.size else 1e-30

    tick_levels = None
    y_disp_for_ticks = None
    if use_log2:
        ymax = max(float(np.nanmax(vals_scaled)), 0.0) * 1.12
        tick_levels = _tick_levels_zero_then_pow2(ymax, max_ticks=7)
        if tick_levels[-1] < ymax:
            tick_levels = np.append(tick_levels, ymax)
        tick_levels = np.unique(np.sort(tick_levels))
        y_disp_for_ticks = np.linspace(0.0, 1.0, len(tick_levels))

    for method_idx in range(n_methods):
        vals = matrix[method_idx, :] / value_scale
        if use_log2:
            plot_y = _map_y_to_equal_tick_spacing(vals, tick_levels)
        else:
            plot_y = vals
        ax.bar(
            x + offsets[method_idx],
            plot_y,
            BAR_WIDTH,
            color=COLORS[method_idx],
            hatch=HATCHES[method_idx],
            edgecolor="white",
            linewidth=1,
        )
        ax.bar(
            x + offsets[method_idx],
            plot_y,
            BAR_WIDTH,
            color="none",
            edgecolor="black",
            linewidth=EDGE_LW,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(STAT_LABELS, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, labelpad=4, fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

    if use_log_scale:
        ax.set_yscale("log", base=10)
        min_val = np.min(vals_scaled[vals_scaled > 0]) if np.any(vals_scaled > 0) else 1.0
        ax.set_ylim(bottom=max(min_val * 0.1, 1e-6))
    elif use_log2:
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(y_disp_for_ticks)
        ax.set_yticklabels([_format_tick_plain_number(t) for t in tick_levels])
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax.set_ylim(0, None)
        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=y_tick_nbins, min_n_ticks=4, prune=None)
        )
        ymin, ymax = ax.get_ylim()
        ticks = [t for t in ax.get_yticks() if ymin - 1e-12 <= t <= ymax + 1e-12]
        if not any(abs(t) < 1e-9 for t in ticks):
            ticks = [0.0] + ticks
        ticks = sorted({float(t) for t in ticks if t >= -1e-12})
        ax.set_yticks(ticks)
    ax.grid(False)


def make_legend_handle(method_idx):
    """
    Match plot_bar_subplot: colored fill + white-edged hatch, then black border overlay.
    Hatch line color follows edgecolor; a black-edged legend patch looks unlike the bars.
    """
    return (
        Patch(
            facecolor=COLORS[method_idx],
            edgecolor="white",
            linewidth=1,
            hatch=HATCHES[method_idx],
        ),
        Patch(facecolor="none", edgecolor="black", linewidth=EDGE_LW),
    )


def add_row_title_centered(fig, ax_left, ax_right, text, y_below_axes=0.042, fontsize=FONT_SIZE):
    """Place a caption centered under two axes in the same row (figure coordinates)."""
    p0 = ax_left.get_position()
    p1 = ax_right.get_position()
    cx = (p0.x0 + p1.x1) / 2
    y = min(p0.y0, p1.y0) - y_below_axes
    fig.text(
        cx,
        y,
        text,
        ha="center",
        va="top",
        fontsize=fontsize,
        transform=fig.transFigure,
    )


def main():
    project = Path(__file__).resolve().parent.parent
    tpch_path = project / TPCH_CSV
    tpcds_path = project / TPCDS_CSV
    for p, name in [(tpch_path, "TPC-H"), (tpcds_path, "TPC-DS")]:
        if not p.exists():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)

    df_tpch = pd.read_csv(tpch_path, sep=";").dropna(how="all")
    df_tpcds = pd.read_csv(tpcds_path, sep=";").dropna(how="all")

    methods = ["PPM", "FGO", "OPRO"]
    time_cols = [f"{m}_execution_time" for m in methods]
    thread_cols = [f"{m}_thread_count" for m in methods]

    for c in time_cols + thread_cols:
        if c not in df_tpch.columns or c not in df_tpcds.columns:
            print(f"Missing column: {c}", file=sys.stderr)
            sys.exit(1)

    tpch_time, tpch_cost = build_stats_matrix(df_tpch, methods, time_cols, thread_cols)
    tpcds_time, tpcds_cost = build_stats_matrix(df_tpcds, methods, time_cols, thread_cols)

    reduction_chunks = [
        repro_reduction_vs_baselines(tpch_time, tpch_cost, "TPC-H"),
        repro_reduction_vs_baselines(tpcds_time, tpcds_cost, "TPC-DS"),
    ]
    reduction_path = project / "output" / "evaluations" / "repro_vs_ppm_fgo_reduction.txt"
    reduction_path.parent.mkdir(parents=True, exist_ok=True)
    reduction_path.write_text("\n".join(reduction_chunks), encoding="utf-8")
    print(f"Wrote reduction summary: {reduction_path}")

    def cost_scale_and_label(cost_matrix):
        cmax = cost_matrix.max()
        if cmax >= 1e6:
            return 1e6, "成本（×10⁶）"
        return 1.0, "成本"

    tpch_cost_scale, tpch_cost_ylabel = cost_scale_and_label(tpch_cost)
    tpcds_cost_scale, tpcds_cost_ylabel = cost_scale_and_label(tpcds_cost)

    # 2x2: top row TPC-H, bottom row TPC-DS; each row (Latency | Cost); taller figure for readability
    # Shorter height for paper layout (was 9.5; ~20% less vertical space)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 7.4), sharex=False)

    # CSV execution_time is in ms; divide by 1000 to show as seconds
    plot_bar_subplot(ax1, tpch_time, "延迟（秒）", value_scale=1000.0, use_log_scale=False)
    plot_bar_subplot(ax2, tpch_cost, tpch_cost_ylabel, value_scale=tpch_cost_scale, use_log_scale=False)
    plot_bar_subplot(ax3, tpcds_time, "延迟（秒）", value_scale=1000.0, y_display="log2")
    plot_bar_subplot(
        ax4,
        tpcds_cost,
        tpcds_cost_ylabel,
        value_scale=tpcds_cost_scale,
        y_display="log2",
    )

    legend_handles = [make_legend_handle(i) for i in range(len(METHOD_LABELS))]

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.13, left=0.1, right=0.96, wspace=0.28, hspace=0.34)

    # Legend anchored just above the top row (closer to plots than a fixed bbox at y≈1)
    y_top_row = max(ax1.get_position().y1, ax2.get_position().y1)
    fig.legend(
        legend_handles,
        METHOD_LABELS,
        loc="lower center",
        bbox_to_anchor=(0.5, y_top_row + 0.008),
        bbox_transform=fig.transFigure,
        ncol=3,
        fontsize=FONT_SIZE,
        frameon=False,
        borderaxespad=0.2,
    )

    add_row_title_centered(fig, ax1, ax2, "（a）TPC-H", y_below_axes=0.042)
    add_row_title_centered(fig, ax3, ax4, "（b）TPC-DS", y_below_axes=0.042)

    out_path = project / "output" / "evaluations" / "stats_latency_cost.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved bar chart: {out_path}")
    print(f"TPC-H: {len(df_tpch)} queries, TPC-DS: {len(df_tpcds)} queries")


if __name__ == "__main__":
    main()
