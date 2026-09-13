"""
Approximate reproduction of a plan-aware resource allocation figure (matplotlib).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import dates as mdates

mpl.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    # Latin: Times New Roman; CJK: SimSun (matplotlib 3.6+ per-glyph fallback)
    "font.family": ["Times New Roman", "SimSun", "NSimSun", "STSong"],
    "axes.unicode_minus": False,
})
from matplotlib.transforms import blended_transform_factory
from datetime import datetime, timedelta

# Time range ~2m38s as in the reference
T_END = 158.0
BASE = datetime(2000, 1, 1, 0, 0, 0)

# --- Plan-aware stepped allocation (seconds from start, resource level) ---
step_t = np.array([0.0, 1.5, 23.0, 65.0, 75.0, 85.0, 120.0, 135.0, T_END])
step_y = np.array([125.0, 78.0, 68.0, 55.0, 45.0, 35.0, 15.0, 4.0, 4.0])


def plan_cap(ts: np.ndarray) -> np.ndarray:
    """Same envelope as ax.step(..., where='post'): usage must stay at or below this."""
    out = np.empty_like(ts, dtype=float)
    for i in range(len(step_t) - 1):
        lo, hi = step_t[i], step_t[i + 1]
        m = (ts >= lo) & (ts < hi)
        out[m] = step_y[i]
    out[ts >= step_t[-1]] = step_y[-1]
    return out


# --- Actual usage: keyframes + noise ---
np.random.seed(7)
t_fine = np.linspace(0.0, T_END, 900)


def _interp_usage(ts: np.ndarray) -> np.ndarray:
    """Piecewise shape matching spikes/valleys from the reference."""
    kx = np.array(
        [0, 8, 20, 38, 45, 52, 60, 72, 88, 105, 112, 115, 128, 140, T_END]
    )
    ky = np.array(
        [58, 62, 75, 40, 3, 8, 65, 48, 25, 4, 10, 35, 12, 3, 0.5]
    )
    return np.interp(ts, kx, ky)


usage = _interp_usage(t_fine)
# Jagged noise (scaled down over time)
usage += np.random.randn(len(t_fine)) * (3.2 * np.exp(-t_fine / 90) + 0.8)
cap = plan_cap(t_fine)
# Keep actual usage strictly below the plan-aware step (small gap so the dashed line stays on top)
margin = np.clip(0.06 * cap, 0.25, 1.8)
usage = np.minimum(usage, cap - margin)
usage = np.clip(usage, 0, None)

# Datetime x for nice hh:mm:ss labels
t_dt = np.array([BASE + timedelta(seconds=float(s)) for s in t_fine])
step_dt = np.array([BASE + timedelta(seconds=float(s)) for s in step_t])

fig, ax = plt.subplots(figsize=(9, 4.2), dpi=120)

# Shaded step area (plan-aware)
ax.fill_between(
    step_dt,
    0,
    step_y,
    step="post",
    color="#f5b8b8",
    alpha=0.85,
    linewidth=0,
)
ax.step(
    step_dt,
    step_y,
    where="post",
    color="#c62828",
    linestyle="--",
    linewidth=1.6,
    label="_nolegend_",
)

# Default allocation
ax.axhline(125, color="black", linestyle="--", linewidth=1.4, label="_nolegend_")

# Actual usage
ax.plot(t_dt, usage, color="#6d1a1a", linewidth=1.2, label="_nolegend_")

ax.set_ylim(0, 130)
ax.set_xlim(BASE, BASE + timedelta(seconds=T_END))
ax.set_ylabel("资源使用")
ax.set_xlabel("时间")
ax.set_yticks(np.arange(0, 131, 10))
ax.grid(True, axis="y", linestyle="-", alpha=0.35)
ax.grid(True, axis="x", linestyle="-", alpha=0.2)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

# Match reference tick spacing (~every 22s from first tick)
tick_secs = [23, 45, 68, 90, 113, 136, 158]
tick_dt = [BASE + timedelta(seconds=s) for s in tick_secs]
ax.set_xticks(tick_dt)

plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

# Text labels: avoid overlap with the y=125 dashed line; keep vertical gap between labels
trans_axes_x_data_y = blended_transform_factory(ax.transAxes, ax.transData)
ax.text(
    0.78,
    0.38,
    "计划感知的资源分配",
    transform=ax.transAxes,
    color="#b71c1c",
    fontsize=14,
    fontweight="bold",
    ha="center",
    va="top",
)
ax.text(
    0.96,
    112.0,
    "默认分配",
    transform=trans_axes_x_data_y,
    color="black",
    fontsize=14,
    ha="right",
    va="bottom",
)

fig.tight_layout(rect=[0, 0, 1, 0.93])

plt.savefig("figure3_plan_aware_allocation.pdf", bbox_inches="tight")
plt.show()
