import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

# 路径配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASETS = ['tpch', 'tpcds']
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output/tru_cdf", f"exp3_mapping_accuracy.pdf")

# Define your desired order of methods
METHOD_ORDER = {
    'ours': 0,
    'SUM': 1,
    # Add other methods here with their desired order
    # 'OTHER_METHOD': 2,
}

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

for col, data_set in enumerate(DATASETS):
    INPUT_DIR = os.path.join(PROJECT_ROOT, "input/tru_cdf", data_set)
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    # Sort files according to METHOD_ORDER
    def get_sort_key(file_path):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        return METHOD_ORDER.get(filename, 999)  # Default to high number for unspecified methods
    
    csv_files.sort(key=get_sort_key)

    # --- Execution Time Q-error ---
    ax = axs[col]
    for file_path in csv_files:
        df = pd.read_csv(file_path, sep=';')
        # 将Q-error值加上1（之前是-1的，现在需要补回来）
        if 'Execution Time Q-error' in df.columns:
            df['Execution Time Q-error'] = df['Execution Time Q-error'].astype(float) + 1.0
        df = df[df['Execution Time Q-error'] <= 2000]
        qerrors = df['Execution Time Q-error'].dropna()
        qerrors = qerrors[qerrors > 0].sort_values()
        y = np.linspace(0, 1, len(qerrors))
        label = os.path.splitext(os.path.basename(file_path))[0]
        ax.plot(qerrors, y, label=label)
    ax.set_xlim(1, 3)
    ax.set_xlabel('Execution Time Q-Error', labelpad=5)
    ax.set_ylabel('CDF')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
    # ax.legend(fontsize=13)  # 移除子图图例

# 添加统一的图例到顶部
fig.legend(['Ours', 'SUM'], loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=17, frameon=False)

plt.subplots_adjust(
    wspace=0.25,
    top=0.83,   # 图整体下移一点，图例更贴近图
    bottom=0.32 # 下方空间更大，给标签留空间
)

# 添加数据集标签到下方
axs[0].text(0.5, -0.36, '(a) TPC-H', transform=axs[0].transAxes, fontsize=18, ha='center')
axs[1].text(0.5, -0.36, '(b) TPC-DS', transform=axs[1].transAxes, fontsize=18, ha='center')

# 保存 PDF
plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight')
print(f"✅ 合并 CDF 图已保存为 PDF：{OUTPUT_PATH}")
