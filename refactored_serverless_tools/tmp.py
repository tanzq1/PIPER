import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import make_interp_spline

# ── CJK in SimSun, Latin in Times New Roman (matplotlib 3.6+ per-glyph fallback) ──
plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "NSimSun", "STSong"]
plt.rcParams["axes.unicode_minus"] = False

mpl.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.6,
    "lines.markersize": 8,
    "legend.frameon": True,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Data ──
parallelism = np.array([8, 16, 32, 64, 96, 128])

MS = 1000.0  # ms → s conversion

data = {
    "Cstore Scan": {
        "real": np.array([11986.2, 6703, 5982, 6317, 7024, 7530]) / MS,
        "pred": np.array([10646, 7533, 6327, 6424, 7079, 7860]) / MS,
    },
    "Vector Hash Aggregation": {
        "real": np.array([40329.9, 21704.4, 12787.31, 9317, 7724, 6317]) / MS,
        "pred": np.array([38409.51, 23010.28, 15411.00, 11812.05, 10790.78, 8811]) / MS,
    },
    "Streaming": {
        "real": np.array([41352, 21522, 10595, 7005.18, 6034.32, 5151.7]) / MS,
        "pred": np.array([39646.86, 21537.52, 12593.64, 8343.27, 7123.44, 6360]) / MS,
    },
}

COLORS = {
    "real": "#2166AC",
    "pred": "#D6604D",
}
MARKERS = {"real": "o", "pred": "s"}

# ── Smooth interpolation helper ──
def smooth_curve(x, y, num=200):
    x_log = np.log2(x)
    x_smooth = np.linspace(x_log.min(), x_log.max(), num)
    spline = make_interp_spline(x_log, y, k=3)
    return 2 ** x_smooth, spline(x_smooth)


# ── Plot ──
fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

subtitles = ["(a) Cstore Scan", "(b) Vector Hash Aggregation", "(c) Streaming"]

for idx, (op_name, vals) in enumerate(data.items()):
    ax = axes[idx]

    xs_real, ys_real = smooth_curve(parallelism, vals["real"])
    xs_pred, ys_pred = smooth_curve(parallelism, vals["pred"])

    ax.plot(xs_real, ys_real, color=COLORS["real"], linestyle="-", linewidth=1.5,
            zorder=2)
    ax.plot(xs_pred, ys_pred, color=COLORS["pred"], linestyle="--", linewidth=1.5,
            zorder=2)

    # scatter points (no label — legend handled at figure level)
    ax.plot(parallelism, vals["real"], color=COLORS["real"], marker=MARKERS["real"],
            linestyle="None", markersize=7, markeredgecolor="white",
            markeredgewidth=0.8, zorder=3)
    ax.plot(parallelism, vals["pred"], color=COLORS["pred"], marker=MARKERS["pred"],
            linestyle="None", markersize=6.5, markeredgecolor="white",
            markeredgewidth=0.8, zorder=3)

    ax.set_xlabel("并行度")
    if idx == 0:
        ax.set_ylabel("执行时间 (s)")

    ax.set_title(subtitles[idx], y=-0.34, fontsize=17)

    ax.set_xticks(parallelism)
    ax.set_xticklabels(parallelism)

    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}")
    )

    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Shared figure-level legend (lines only, no markers) ──
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color=COLORS["real"], linestyle="-",  linewidth=1.8, label="实际值"),
    Line2D([0], [0], color=COLORS["pred"], linestyle="--", linewidth=1.8, label="预测值"),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.02),
    frameon=False,
    edgecolor="0.8",
    fancybox=False,
    fontsize=15,
    handlelength=2.2,
    handletextpad=0.5,
    columnspacing=1.5,
    borderpad=0.4,
)

fig.subplots_adjust(wspace=0.31, bottom=0.28, top=0.86, left=0.07, right=0.985)

plt.savefig("op_scalability.pdf")
plt.savefig("op_scalability.png")
plt.show()
print("Saved: op_scalability.pdf / .png")
