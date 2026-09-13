import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab

# 数据
labels = ['Latency', 'Total DOP']
uniform_dop = [9.6, 257]
per_pipeline_dop = [10.7, 137]

x = np.arange(len(labels))
width = 0.20  # 柱宽更小

# 配色和花纹（参考multi_bar.py）
colors = ['C3', 'C1']
hatchs = ['xx', '..']

pylab.rcParams.update({
    'axes.labelsize': 18,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 20,  # 图例字体更大
    'font.family': 'Arial',
    'pdf.fonttype': 42,
})

fig, ax1 = plt.subplots(figsize=(6, 6))  # 画布更窄

# 左Y轴：Latency
bar1 = ax1.bar(x[0] - width/2, uniform_dop[0], width, label='Uniform DOP', color=colors[0], hatch=hatchs[0], edgecolor='white', linewidth=1)
bar2 = ax1.bar(x[0] + width/2, per_pipeline_dop[0], width, label='Per-pipeline DOP', color=colors[1], hatch=hatchs[1], edgecolor='white', linewidth=1)
# 白框
ax1.bar(x[0] - width/2, uniform_dop[0], width, color='none', edgecolor='black', linewidth=1.2)
ax1.bar(x[0] + width/2, per_pipeline_dop[0], width, color='none', edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Latency (s)', labelpad=18)
ax1.set_ylim(0, max(uniform_dop[0], per_pipeline_dop[0]) * 1.25)

# 右Y轴：Total DOP
ax2 = ax1.twinx()
bar3 = ax2.bar(x[1] - width/2, uniform_dop[1], width, color=colors[0], hatch=hatchs[0], edgecolor='white', linewidth=1)
bar4 = ax2.bar(x[1] + width/2, per_pipeline_dop[1], width, color=colors[1], hatch=hatchs[1], edgecolor='white', linewidth=1)
# 白框
ax2.bar(x[1] - width/2, uniform_dop[1], width, color='none', edgecolor='black', linewidth=1.2)
ax2.bar(x[1] + width/2, per_pipeline_dop[1], width, color='none', edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Total number of threads', labelpad=18)
ax2.set_ylim(0, max(uniform_dop[1], per_pipeline_dop[1]) * 1.25)

# x轴
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlim(-0.3, 1.3)  # 让柱子更靠近

# 柱子上加数值
def autolabel(ax, rects, values):
    for rect, v in zip(rects, values):
        height = rect.get_height()
        ax.annotate(f'{v}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=17)

# 只取BarContainer中的第一个Rectangle对象
autolabel(ax1, [bar1[0], bar2[0]], [uniform_dop[0], per_pipeline_dop[0]])
autolabel(ax2, [bar3[0], bar4[0]], [uniform_dop[1], per_pipeline_dop[1]])

# 图例
ax1.legend([bar1, bar2], ['Uniform DOP', 'Per-pipeline DOP'], loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False, fontsize=18)

plt.tight_layout()
plt.subplots_adjust(left=0.18, right=0.97, top=0.85, bottom=0.18)  # 增加四周空白
plt.savefig('bar_compare.pdf', bbox_inches='tight', dpi=600)