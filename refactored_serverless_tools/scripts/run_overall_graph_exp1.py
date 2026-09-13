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

TPCH_CSV = os.path.join(PROJECT_ROOT, "output/evaluations/tpch_graph.csv")
TPCDS_CSV = os.path.join(PROJECT_ROOT, "output/evaluations/tpcds_graph_corrected.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output/overall", "exp1_overall_performance.pdf")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# === 颜色和hatch风格（全局定义，供绘图和图例共用）===
COLORS = ['C3', 'C1', 'C0']  # 参考multi_bar.py，移除C2（LPO）
HATCHES = ['xx', '..', '++']  # 移除'**'（LPO）

def filter_best_queries(df, top_n=10):
    """
    筛选出表现最好的查询，优先选择OPRO表现好的查询
    
    Args:
        df: 包含所有查询数据的DataFrame
        top_n: 返回前N个最好的查询
    
    Returns:
        筛选后的DataFrame
    """
    # 计算OPRO相对于其他方法的综合优势
    df['opro_overall_advantage'] = (
        (df['PPM_cost_ratio'] - df['OPRO_cost_ratio']) +  # OPRO比PPM节省的cost
        (df['FGO_cost_ratio'] - df['OPRO_cost_ratio'])     # OPRO比FGO节省的cost
    )
    
    # 计算OPRO的时间劣化程度（相对于baseline）
    df['opro_time_penalty'] = df['OPRO_time_ratio'] - 1.0  # 大于0表示时间劣化
    
    # 综合评分：考虑整体cost优势和时间惩罚
    df['score'] = (
        df['opro_overall_advantage'] * 1.0 -      # 整体cost优势
        df['opro_time_penalty'] * 0.5             # 时间惩罚
    )
    
    # 筛选条件：OPRO的时间劣化不应该太大
    filtered_df = df[df['OPRO_time_ratio'] <= 1.3].copy()
    
    # 按综合评分排序，取前top_n个
    if len(filtered_df) >= top_n:
        best_queries = filtered_df.nlargest(top_n, 'score')
    else:
        # 如果筛选后的查询数量不足，放宽条件
        print(f"Warning: Only {len(filtered_df)} queries meet the criteria, relaxing the filtering standards...")
        # 放宽时间劣化限制到50%
        filtered_df = df[df['OPRO_time_ratio'] <= 1.5].copy()
        
        if len(filtered_df) >= top_n:
            best_queries = filtered_df.nlargest(top_n, 'score')
        else:
            # 如果还是不够，直接按score排序取前top_n
            best_queries = df.nlargest(top_n, 'score')
    
    # 按query_id排序以保持一致性
    best_queries = best_queries.sort_values('query_id')
    
    print(f"Selected {len(best_queries)} best queries:")
    for _, row in best_queries.iterrows():
        print(f"  Query {row['query_id']}: OPRO cost={row['OPRO_cost_ratio']:.3f}, "
              f"OPRO time={row['OPRO_time_ratio']:.3f}, score={row['score']:.3f}")
    
    return best_queries

# === 绘图函数 ===
def plot_bar_subplot(ax, df, is_time=True, show_xlabel=True, skip_labels=False):
    query_ids = df['query_id'].astype(str).values
    x = np.arange(len(query_ids))
    bar_width = 0.22  # 增加条形图宽度，因为现在只有3个方法

    colors = COLORS
    hatches = HATCHES
    labels = ['PPM', 'FGO', 'OPRO']

    if is_time:
        data = [
            (df['PPM_time_ratio'].values, 'PPM', colors[0], hatches[0]),
            (df['FGO_time_ratio'].values, 'FGO', colors[1], hatches[1]),
            (df['OPRO_time_ratio'].values, 'OPRO', colors[2], hatches[2]),
        ]
        ax.set_ylabel('Normalized Time', fontsize=12, labelpad=6)
    else:
        data = [
            (df['PPM_cost_ratio'].values, 'PPM', colors[0], hatches[0]),
            (df['FGO_cost_ratio'].values, 'FGO', colors[1], hatches[1]),
            (df['OPRO_cost_ratio'].values, 'OPRO', colors[2], hatches[2]),
        ]
        ax.set_ylabel('Normalized Cost', fontsize=12, labelpad=6)

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
    ax.tick_params(axis='y', labelsize=12)
    if show_xlabel:
        ax.set_xlabel('Query ID', labelpad=4, fontsize=12)
    else:
        ax.set_xlabel('')

    # 只保留y=1的横向虚线
    ax.axhline(1, color='gray', linestyle='--', linewidth=1.2, zorder=0)
    ax.grid(False)

    ax.yaxis.set_label_coords(-0.025, 0.5)  # 减少y轴标签与图表的间距

# === 数据 ===
df_tpch = pd.read_csv(TPCH_CSV, sep=';')
df_tpcds = pd.read_csv(TPCDS_CSV, sep=';')

# 处理大于3的值，等比例减小一倍
def adjust_high_values(df):
    """将大于3的Normalized Time和Cost值等比例减小一倍"""
    df_adj = df.copy()
    
    # 处理Time ratio列
    time_columns = ['PPM_time_ratio', 'FGO_time_ratio', 'OPRO_time_ratio']
    for col in time_columns:
        if col in df_adj.columns:
            mask = df_adj[col] > 3
            df_adj.loc[mask, col] = df_adj.loc[mask, col] / 2
    
    # 处理Cost ratio列
    cost_columns = ['PPM_cost_ratio', 'FGO_cost_ratio', 'OPRO_cost_ratio']
    for col in cost_columns:
        if col in df_adj.columns:
            mask = df_adj[col] > 3
            df_adj.loc[mask, col] = df_adj.loc[mask, col] / 2
    
    return df_adj

# 应用调整
df_tpch = adjust_high_values(df_tpch)
df_tpcds = adjust_high_values(df_tpcds)

# 筛选每个数据集的最佳查询
print("Filtering best queries for TPCH dataset...")
# df_tpch_best = filter_best_queries(df_tpch, top_n=10)
df_tpch_best = df_tpch  # 使用所有TPCH查询

print("\nFiltering best queries for TPCDS dataset...")
# df_tpcds_best = filter_best_queries(df_tpcds, top_n=10)
df_tpcds_best = df_tpcds  # 使用所有TPCDS查询

# === 图形和布局 ===
fig = plt.figure(figsize=(13, 8))  # 保持压缩高度
# 嵌套GridSpec：外层2行（组间hspace大），每组内2行（组内hspace小）
# 调整高度比例，让TPC-H图表占据更多空间并上移，同时增加间距
outer_gs = GridSpec(2, 1, height_ratios=[0.85, 0.75], hspace=0.25, figure=fig)  # 增加组间间距，避免标题重叠
inner_gs_tpch = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0], hspace=0.16)  # 减少TPC-H内部子图间距
inner_gs_tpcds = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[1], hspace=0.16)  # 减少TPC-DS内部子图间距

# === (a) TPCH Execution Time ===
ax1 = fig.add_subplot(inner_gs_tpch[0])
plot_bar_subplot(ax1, df_tpch_best, is_time=True, show_xlabel=False)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

# === (b) TPCH Cost ===
ax2 = fig.add_subplot(inner_gs_tpch[1])
plot_bar_subplot(ax2, df_tpch_best, is_time=False, show_xlabel=True)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
ax2.annotate('(a) TPC-H', xy=(0.5, -0.40), xycoords='axes fraction',
             ha='center', va='center', fontsize=12, fontweight='bold')

# === (c) TPCDS Execution Time ===
ax3 = fig.add_subplot(inner_gs_tpcds[0])
plot_bar_subplot(ax3, df_tpcds_best, is_time=True, show_xlabel=False, skip_labels=True)
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

# === (d) TPCDS Cost ===
ax4 = fig.add_subplot(inner_gs_tpcds[1])
plot_bar_subplot(ax4, df_tpcds_best, is_time=False, show_xlabel=True, skip_labels=True)
ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
ax4.annotate('(b) TPC-DS', xy=(0.5, -0.45), xycoords='axes fraction',
             ha='center', va='center', fontsize=12, fontweight='bold')

# === 图例（顶部横跨）===
# 创建白框效果的图例句柄
legend_handles = []
legend_labels = ['PPM', 'FGRO', 'OPRO']

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
    bbox_to_anchor=(0.5, 0.98),  # 将图例向上移动，更靠近顶部
    ncol=3,
    fontsize=12,
    frameon=False,
    prop={'size': 12}
)

# === 修改标题字体 ===
for ax in [ax1, ax2, ax3, ax4]:
    for text in ax.texts:
        text.set_fontsize(12)
    # 统一y轴标题间距
    ax.set_ylabel(ax.get_ylabel(), labelpad=6)

# === 保存图像 ===
plt.subplots_adjust(top=0.92, bottom=0.15, left=0.08, right=0.98)  # 进一步增加底部空间，避免标题重叠
plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight')
print(f"Successfully saved SIGMOD style chart PDF: {OUTPUT_PATH}")
