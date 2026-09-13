import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASETS = ['tpch', 'tpcds']
OUTPUT_PATH_TEMPLATE = os.path.join(PROJECT_ROOT, "output/cdf", "exp2_predict_accuracy_{}.pdf")

# Define your desired order of methods
METHOD_ORDER = {
    'Auto-DOP': 0,
    'PPM': 1,
    'POO': 2,
    # Add other methods here with their desired order
    # 'OTHER_METHOD': 2,
}

for data_set in DATASETS:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    INPUT_DIR = os.path.join(PROJECT_ROOT, "input/cdf", data_set)
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

    # Sort files according to METHOD_ORDER
    def get_sort_key(file_path):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        return METHOD_ORDER.get(filename, 999)  # Default to high number for unspecified methods
    csv_files.sort(key=get_sort_key)

    # --- Execution Time Q-error ---
    ax = axs[0]
    for file_path in csv_files:
        df = pd.read_csv(file_path, sep=';')
        if 'Execution Time Q-error' in df.columns:
            df['Execution Time Q-error'] = df['Execution Time Q-error'] + 1
        df = df[df['Execution Time Q-error'].le(2000)]
        qerrors = df['Execution Time Q-error'].dropna()
        qerrors = qerrors[qerrors > 0].sort_values()
        y = np.linspace(0, 1, len(qerrors))
        label = os.path.splitext(os.path.basename(file_path))[0]
        ax.plot(qerrors, y, label=label)
    ax.set_xscale('log')
    ax.set_xlim(1, 100)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(['1', '2', '5', '10', '20', '50', '100'])
    ax.set_title('')
    ax.set_ylabel('CDF', fontsize=18)
    ax.set_xlabel('Execution Time Q-Error', fontsize=18)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

    # --- Memory Q-error ---
    ax = axs[1]
    for file_path in csv_files:
        df = pd.read_csv(file_path, sep=';')
        if 'Memory Q-error' in df.columns:
            df['Memory Q-error'] = df['Memory Q-error'] + 1
        df = df[df['Memory Q-error'].le(2000)]
        qerrors = df['Memory Q-error'].dropna()
        qerrors = qerrors[qerrors > 0].sort_values()
        y = np.linspace(0, 1, len(qerrors))
        label = os.path.splitext(os.path.basename(file_path))[0]
        ax.plot(qerrors, y, label=label)
    ax.set_xscale('log')
    ax.set_xlim(1, 100)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(['1', '2', '5', '10', '20', '50', '100'])
    ax.set_title('')
    ax.set_ylabel('CDF', fontsize=18)
    ax.set_xlabel('Memory Q-Error', fontsize=18)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

    # 添加统一的图例到顶部
    fig.legend(['Auto-DOP', 'PPM', 'POO'], 
              loc='upper center', bbox_to_anchor=(0.5, 0.97), 
              ncol=3, fontsize=17, frameon=False)

    plt.subplots_adjust(
        wspace=0.20,
        top=0.87,
        bottom=0.20
    )

    # 删除下方数据集标签
    # axs[0].text(0.5, -0.22, f'({"a" if data_set=="tpch" else "b"}) {data_set.upper()}', transform=axs[0].transAxes, fontsize=18, ha='center')

    # 保存 PDF
    output_path = OUTPUT_PATH_TEMPLATE.format(data_set)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"✅ {data_set.upper()} CDF 图已保存为 PDF：{output_path}")
    plt.close(fig)