import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', 'NSimSun', 'STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASETS = ['tpch', 'tpcds']
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output/tru_cdf", "exp3_mapping_accuracy.pdf")

METHOD_COLUMNS = [
    ('Ours', 'q_error_mapping_time'),
    ('SUM', 'q_error_thread_wise_time'),
]

CUSTOM_TICKS_TPCDS = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
CUSTOM_POSITIONS_TPCDS = np.arange(len(CUSTOM_TICKS_TPCDS))

SMOOTH_RATIO = 0.06   # ~6% rolling window smoothing


fig, axs = plt.subplots(1, 2, figsize=(12, 5))
legend_labels = []


for col, dataset in enumerate(DATASETS):
    df = pd.read_csv(
        os.path.join(PROJECT_ROOT, "input/tru_cdf", dataset, "q_error_comparison.csv"),
        sep=';'
    )

    ax = axs[col]

    # compute TPCH numeric range automatically
    vals = []
    for _, cname in METHOD_COLUMNS:
        if cname in df.columns:
            vals.extend(list(df[cname].astype(float) + 1))
    vals = [v for v in vals if 1 <= v <= 2000]

    if dataset == "tpch":
        upper = np.ceil(max(vals) * 2) / 2 if vals else 2.0
        x_max = upper
    else:
        x_max = CUSTOM_POSITIONS_TPCDS[-1]

    for label, cname in METHOD_COLUMNS:
        if cname not in df.columns:
            continue

        q = (df[cname].astype(float) + 1).clip(lower=1).dropna()
        q = q[q <= 2000]
        if len(q) == 0:
            continue

        q = np.sort(q.to_numpy())
        n = len(q)

        # empirical CDF
        y = np.linspace(0, 1, n)

        # ✅ rolling smoothing on x (unified for both datasets)
        window = max(3, int(len(q) * SMOOTH_RATIO))
        if window % 2 == 0:
            window += 1

        xs = pd.Series(q).rolling(window=window, min_periods=1, center=True).mean().to_numpy()

        # enforce monotonicity
        xs = np.maximum.accumulate(xs)

        # ✅ TPC-DS axis compression mapping
        if dataset == "tpcds":
            xs = np.interp(xs, CUSTOM_TICKS_TPCDS, CUSTOM_POSITIONS_TPCDS)

        # extend last point to full axis
        if xs[-1] < x_max:
            xs = np.append(xs, x_max)
            y = np.append(y, y[-1])

        ax.plot(xs, y, linewidth=2.5, label=label)

        if label not in legend_labels:
            legend_labels.append(label)

    ax.set_xlabel("执行时间 Q-Error", fontsize=18)
    ax.set_ylabel("CDF", fontsize=18)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

    if dataset == "tpcds":
        ax.set_xlim(0, CUSTOM_POSITIONS_TPCDS[-1])
        ax.set_xticks(CUSTOM_POSITIONS_TPCDS)
        ax.set_xticklabels([str(v) for v in CUSTOM_TICKS_TPCDS])
    else:
        ax.set_xlim(1.0, upper)
        ax.set_xticks(np.arange(1.0, upper + 1e-9, 0.5))


fig.legend(
    legend_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.97),
    ncol=len(legend_labels),
    fontsize=17,
    frameon=False,
)

plt.subplots_adjust(wspace=0.20, top=0.87, bottom=0.20)

axs[0].text(0.5, -0.28, '(a) TPC-H', transform=axs[0].transAxes, fontsize=18, ha='center')
axs[1].text(0.5, -0.28, '(b) TPC-DS', transform=axs[1].transAxes, fontsize=18, ha='center')

plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_PATH}")
