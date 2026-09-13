import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'Times New Roman'

# === 路径设置 ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TPCH_CSV = os.path.join(PROJECT_ROOT, "output/evaluations/tpch_graph.csv")
TPCDS_CSV = os.path.join(PROJECT_ROOT, "output/evaluations/tpcds_graph_corrected.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output/overall", "exp_pio_opro_comparison.pdf")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# === 颜色和hatch风格（参照draw/subplot.py） ===
COLORS = ['C0', 'C1']  # PIO和OPRO的颜色
HATCHES = ['xx', '..']  # PIO和OPRO的hatch样式
LABELS = ['PIO', 'OPRO']
BAR_LABELS = ['Execution Time', 'Cost']

def calculate_mean_metrics(df):
    mean_time = [df['PIO_time_ratio'].mean(), df['POO_time_ratio'].mean()]
    mean_cost = [df['PIO_cost_ratio'].mean(), df['POO_cost_ratio'].mean()]
    return mean_time, mean_cost

def plot_double_comparison(mean_time_tpch, mean_cost_tpch, mean_time_tpcds, mean_cost_tpcds, output_path):
    bar_width = 0.18
    group_x = np.arange(2)  # 0: Execution Time, 1: Cost
    bar_offsets = [-bar_width / 2, bar_width / 2]  # PIO, OPRO

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    # 1. TPC-H
    ax = axes[0]
    tpc_h_vals = np.array([
        [mean_time_tpch[0], mean_time_tpch[1]],  # Execution Time: PIO, OPRO
        [mean_cost_tpch[0], mean_cost_tpch[1]],  # Cost: PIO, OPRO
    ])
    ymax = tpc_h_vals.max() * 1.13
    ax.set_ylim(0, ymax)

    for i in range(2):  # 0: Execution Time, 1: Cost
        for j in range(2):  # 0: PIO, 1: OPRO
            xpos = group_x[i] + bar_offsets[j]
            height = tpc_h_vals[i, j]
            ax.bar(xpos, height, bar_width,
                   color=COLORS[j], hatch=HATCHES[j], alpha=0.8,
                   edgecolor='white', linewidth=1)
            ax.bar(xpos, height, bar_width,
                   color='none', edgecolor='black', linewidth=1.2)
            label_y = min(height + ymax * 0.02, ymax * 0.98)
            ax.text(xpos, label_y, f'{height:.2f}', ha='center', va='bottom', fontsize=14)

    ax.set_xticks(group_x)
    ax.set_xticklabels(BAR_LABELS, fontsize=15)
    ax.set_ylabel('Mean Normalized Value', fontsize=18)
    ax.annotate('(a) TPC-H', xy=(0.5, -0.22), xycoords='axes fraction',
                ha='center', va='center', fontsize=17)
    ax.tick_params(axis='y', labelsize=15)
    ax.grid(True, axis='y', alpha=0.25)

    # 2. TPC-DS
    ax2 = axes[1]
    tpc_ds_vals = np.array([
        [mean_time_tpcds[0], mean_time_tpcds[1]],
        [mean_cost_tpcds[0], mean_cost_tpcds[1]],
    ])
    ymax2 = tpc_ds_vals.max() * 1.13
    ax2.set_ylim(0, ymax2)

    for i in range(2):
        for j in range(2):
            xpos = group_x[i] + bar_offsets[j]
            height = tpc_ds_vals[i, j]
            ax2.bar(xpos, height, bar_width,
                    color=COLORS[j], hatch=HATCHES[j], alpha=0.8,
                    edgecolor='white', linewidth=1)
            ax2.bar(xpos, height, bar_width,
                    color='none', edgecolor='black', linewidth=1.2)
            label_y = min(height + ymax2 * 0.02, ymax2 * 0.98)
            ax2.text(xpos, label_y, f'{height:.2f}', ha='center', va='bottom', fontsize=14)

    ax2.set_xticks(group_x)
    ax2.set_xticklabels(BAR_LABELS, fontsize=15)
    ax2.set_ylabel('Mean Normalized Value', fontsize=18)
    ax2.annotate('(b) TPC-DS', xy=(0.5, -0.22), xycoords='axes fraction',
                 ha='center', va='center', fontsize=17)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.grid(True, axis='y', alpha=0.25)

    # 图例
    legend_handles = [
        Patch(facecolor=COLORS[0], edgecolor='white', hatch=HATCHES[0], alpha=0.8, linewidth=1, label='PIO'),
        Patch(facecolor=COLORS[1], edgecolor='white', hatch=HATCHES[1], alpha=0.8, linewidth=1, label='OPRO'),
    ]
    fig.legend(
        legend_handles, LABELS,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=18,
        frameon=False,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"✅ 成功保存: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("加载TPC-H数据...")
    df_tpch = pd.read_csv(TPCH_CSV, sep=';')
    print("加载TPC-DS数据...")
    df_tpcds = pd.read_csv(TPCDS_CSV, sep=';')

    mean_time_tpch, mean_cost_tpch = calculate_mean_metrics(df_tpch)
    mean_time_tpcds, mean_cost_tpcds = calculate_mean_metrics(df_tpcds)

    plot_double_comparison(mean_time_tpch, mean_cost_tpch, mean_time_tpcds, mean_cost_tpcds, OUTPUT_PATH)

    # 打印统计
    print("\n=== 统计信息 ===")
    print(f"TPC-H 平均执行时间: PIO={mean_time_tpch[0]:.2f}, OPRO={mean_time_tpch[1]:.2f}")
    print(f"TPC-DS 平均执行时间: PIO={mean_time_tpcds[0]:.2f}, OPRO={mean_time_tpcds[1]:.2f}")
    print(f"TPC-H 平均Cost: PIO={mean_cost_tpch[0]:.2f}, OPRO={mean_cost_tpch[1]:.2f}")
    print(f"TPC-DS 平均Cost: PIO={mean_cost_tpcds[0]:.2f}, OPRO={mean_cost_tpcds[1]:.2f}") 