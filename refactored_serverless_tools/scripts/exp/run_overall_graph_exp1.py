import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

plt.rcParams['font.family'] = 'Times New Roman'

# === 路径设置 ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TPCH_CSV = os.path.join(PROJECT_ROOT, "../output/evaluations/tpch_graph.csv")
TPCDS_CSV = os.path.join(PROJECT_ROOT, "../output/evaluations/tpcds_graph_corrected.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "../output/overall", "exp1_overall_performance.pdf")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# === 颜色和hatch风格（全局定义，供绘图和图例共用）===
COLORS = ['C3', 'C1', 'C0']  # 参考multi_bar.py，移除C2（LPO）
HATCHES = ['xx', '..', '++']  # 移除'**'（LPO）

def filter_best_queries(df, top_n=10):
    """
    筛选出表现最好的查询，优先选择GPO表现好的查询
    
    Args:
        df: 包含所有查询数据的DataFrame
        top_n: 返回前N个最好的查询
    
    Returns:
        筛选后的DataFrame
    """
    # 计算GPO相对于其他方法的综合优势
    df['gpo_overall_advantage'] = (
        (df['Auto_DOP_cost_ratio'] - df['POO_cost_ratio']) +  # GPO比Auto-DOP节省的cost
        (df['PPM_cost_ratio'] - df['POO_cost_ratio'])         # GPO比PPM节省的cost
    )
    
    # 计算GPO的时间劣化程度（相对于baseline）
    df['gpo_time_penalty'] = df['POO_time_ratio'] - 1.0  # 大于0表示时间劣化
    
    # 综合评分：考虑整体cost优势和时间惩罚
    df['score'] = (
        df['gpo_overall_advantage'] * 1.0 -      # 整体cost优势
        df['gpo_time_penalty'] * 0.5             # 时间惩罚
    )
    
    # 筛选条件：GPO的时间劣化不应该太大
    filtered_df = df[df['POO_time_ratio'] <= 1.3].copy()
    
    # 按综合评分排序，取前top_n个
    if len(filtered_df) >= top_n:
        best_queries = filtered_df.nlargest(top_n, 'score')
    else:
        # 如果筛选后的查询数量不足，放宽条件
        print(f"警告：筛选后只有{len(filtered_df)}个查询满足条件，放宽筛选标准...")
        # 放宽时间劣化限制到50%
        filtered_df = df[df['POO_time_ratio'] <= 1.5].copy()
        
        if len(filtered_df) >= top_n:
            best_queries = filtered_df.nlargest(top_n, 'score')
        else:
            # 如果还是不够，直接按score排序取前top_n
            best_queries = df.nlargest(top_n, 'score')
    
    # 按query_id排序以保持一致性
    best_queries = best_queries.sort_values('query_id')
    
    print(f"筛选出{len(best_queries)}个最佳查询:")
    for _, row in best_queries.iterrows():
        print(f"  Query {row['query_id']}: GPO cost={row['POO_cost_ratio']:.3f}, "
              f"GPO time={row['POO_time_ratio']:.3f}, score={row['score']:.3f}")
    
    return best_queries

# === 绘图函数 ===
def plot_bar_subplot(ax, df, is_time=True, show_xlabel=True, skip_labels=False):
    query_ids = df['query_id'].astype(str).values
    x = np.arange(len(query_ids))
    bar_width = 0.22  # 增加条形图宽度，因为现在只有3个方法

    colors = COLORS
    hatches = HATCHES
    labels = ['Auto-DOP', 'PPM', 'GPO']

    if is_time:
        data = [
            (df['Auto_DOP_time_ratio'].values, 'Auto-DOP', colors[0], hatches[0]),
            (df['PPM_time_ratio'].values, 'PPM', colors[1], hatches[1]),
            (df['POO_time_ratio'].values, 'GPO', colors[2], hatches[2]),
        ]
        ax.set_ylabel('Normalized Time', fontsize=13, labelpad=13)
    else:
        data = [
            (df['Auto_DOP_cost_ratio'].values, 'Auto-DOP', colors[0], hatches[0]),
            (df['PPM_cost_ratio'].values, 'PPM', colors[1], hatches[1]),
            (df['POO_cost_ratio'].values, 'GPO', colors[2], hatches[2]),
        ]
        ax.set_ylabel('Normalized Cost', fontsize=13, labelpad=13)

    for i, (values, label, color, hatch) in enumerate(data):
        # 第一层：有颜色和花纹的条形图，白色边框
        bars1 = ax.bar(x + (i - 1.5) * bar_width, values, width=bar_width,
                      color=color, hatch=hatch, alpha=0.8, edgecolor='white',
                      linewidth=1, label=label)
        # 第二层：无颜色的条形图，黑色边框
        bars2 = ax.bar(x + (i - 1.5) * bar_width, values, width=bar_width,
                      color='none', edgecolor='black', linewidth=1.2)

    ax.set_xticks(x)
    
    # 如果skip_labels为True，则间隔显示标签
    if skip_labels:
        # 每隔2个显示一个标签
        tick_labels = []
        for i, query_id in enumerate(query_ids):
            if i % 2 == 0:
                tick_labels.append(query_id)
            else:
                tick_labels.append('')
        ax.set_xticklabels(tick_labels, fontsize=12)
    else:
        ax.set_xticklabels(query_ids, fontsize=12)
    
    ax.set_xlim([-0.5, len(x) - 0.5])
    ax.tick_params(axis='x', length=0, labelsize=12)
    ax.tick_params(axis='y', labelsize=11)
    if show_xlabel:
        ax.set_xlabel('Query ID', labelpad=4, fontsize=13)
    else:
        ax.set_xlabel('')

    # 只保留y=1的横向虚线
    ax.axhline(1, color='gray', linestyle='--', linewidth=1.2, zorder=0)
    ax.grid(False)

    ax.yaxis.set_label_coords(-0.045, 0.5)

# === 数据 ===
df_tpch = pd.read_csv(TPCH_CSV, sep=';')
df_tpcds = pd.read_csv(TPCDS_CSV, sep=';')

# 筛选每个数据集的最佳查询
print("筛选TPCH数据集的最佳查询...")
# df_tpch_best = filter_best_queries(df_tpch, top_n=10)
df_tpch_best = df_tpch  # 使用所有TPCH查询

print("\n筛选TPCDS数据集的最佳查询...")
# df_tpcds_best = filter_best_queries(df_tpcds, top_n=10)
df_tpcds_best = df_tpcds  # 使用所有TPCDS查询

# === 图形和布局 ===
fig = plt.figure(figsize=(14, 13))  # 再加高
# 嵌套GridSpec：外层2行（组间hspace大），每组内2行（组内hspace小）
outer_gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.24, figure=fig)
inner_gs_tpch = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0], hspace=0.12)
inner_gs_tpcds = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[1], hspace=0.12)

# === (a) TPCH Execution Time ===
ax1 = fig.add_subplot(inner_gs_tpch[0])
plot_bar_subplot(ax1, df_tpch_best, is_time=True, show_xlabel=False)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

# === (b) TPCH Cost ===
ax2 = fig.add_subplot(inner_gs_tpch[1])
plot_bar_subplot(ax2, df_tpch_best, is_time=False, show_xlabel=True)
ax2.annotate('(a) TPC-H', xy=(0.5, -0.32), xycoords='axes fraction',
             ha='center', va='center', fontsize=18, fontweight='bold')

# === (c) TPCDS Execution Time ===
ax3 = fig.add_subplot(inner_gs_tpcds[0])
plot_bar_subplot(ax3, df_tpcds_best, is_time=True, show_xlabel=False, skip_labels=True)

# === (d) TPCDS Cost ===
ax4 = fig.add_subplot(inner_gs_tpcds[1])
plot_bar_subplot(ax4, df_tpcds_best, is_time=False, show_xlabel=True, skip_labels=True)
ax4.annotate('(b) TPC-DS', xy=(0.5, -0.32), xycoords='axes fraction',
             ha='center', va='center', fontsize=18, fontweight='bold')

# === 图例（顶部横跨）===
# 创建白框效果的图例句柄
legend_handles = []
legend_labels = ['Auto-DOP', 'PPM', 'OPRO']

for i, (color, hatch, label) in enumerate(zip(COLORS, HATCHES, legend_labels)):
    # 创建有颜色和花纹的patch
    patch1 = Patch(facecolor=color, edgecolor='white', hatch=hatch, alpha=0.8, linewidth=1)
    # 创建黑色边框的patch
    patch2 = Patch(facecolor='none', edgecolor='black', linewidth=1.2)
    # 组合两个patch
    legend_handles.append((patch1, patch2))

fig.legend(
    legend_handles,
    legend_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.97),
    ncol=3,
    fontsize=15,
    frameon=False,
    prop={'weight': 'bold'}
)

# === 修改标题字体 ===
for ax in [ax1, ax2, ax3, ax4]:
    for text in ax.texts:
        text.set_fontsize(15)
    # 统一y轴标题间距
    ax.set_ylabel(ax.get_ylabel(), labelpad=13)

# === 保存图像 ===
plt.subplots_adjust(top=0.94, bottom=0.07)
plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight')
print(f"✅ 成功保存 SIGMOD 风格图表 PDF：{OUTPUT_PATH}")
