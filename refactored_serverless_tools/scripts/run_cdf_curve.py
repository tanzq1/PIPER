import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys

plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', 'NSimSun', 'STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASETS = ['tpch', 'tpcds']
OUTPUT_PATH_TEMPLATE = os.path.join(PROJECT_ROOT, "output/cdf", "exp2_predict_accuracy_{}.pdf")

# Define your desired order of methods
METHOD_ORDER = {
    'PPM': 0,
    'FGO': 1,
    'OPRO': 2,  # CSV file name is OPRO, but display as RePro
    'RePro': 2,  # Also support RePro filename
    # Add other methods here with their desired order
    # 'OTHER_METHOD': 2,
}

# Map CSV file names to display labels
LABEL_MAP = {
    'OPRO': 'RePro',
    'RePro': 'RePro',
    'PPM': 'Query-Uniform',
    'FGO': 'FGO',
    'FGRO': 'Stage-Independent',
}

# Define x-axis settings for each dataset
XAXIS_SETTINGS = {
    'tpch': {
        'xlim': (1, 30),
        'xticks': [1, 2, 5, 10, 20, 30],
        'xticklabels': ['1', '2', '5', '10', '20', '30']
    },
    'tpcds': {
        'xlim': (1, 1000),
        'xticks': [1, 2, 5, 10, 100, 1000],
        'xticklabels': ['1', '2', '5', '10', '100', '1000']
    }
}

def smooth_cdf_curve(qerrors_extended, y_extended):
    """Smooth CDF curve using dense interpolation in log space"""
    if len(qerrors_extended) <= 1:
        return qerrors_extended, y_extended
    
    # Use log space for x to handle logarithmic scale better
    log_x = np.log10(qerrors_extended)
    # Remove duplicates to avoid interpolation issues
    unique_mask = np.concatenate(([True], np.diff(log_x) > 1e-10))
    log_x_unique = log_x[unique_mask]
    y_unique = y_extended[unique_mask]
    
    if len(log_x_unique) <= 1:
        return qerrors_extended, y_extended
    
    # Generate many more smooth points in log space for better smoothness
    log_x_min, log_x_max = log_x_unique[0], log_x_unique[-1]
    # Use more points for smoother curve (1000-2000 points)
    n_smooth = min(2000, max(1000, len(qerrors_extended) * 50))
    log_x_smooth = np.linspace(log_x_min, log_x_max, n_smooth)
    
    # Interpolate y values using numpy's interp function
    y_smooth = np.interp(log_x_smooth, log_x_unique, y_unique)
    
    # Convert back from log space
    qerrors_smooth = 10 ** log_x_smooth
    
    return qerrors_smooth, y_smooth

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
        df = df[df['Execution Time Q-error'].le(500)]
        qerrors = df['Execution Time Q-error'].dropna()
        qerrors = qerrors[qerrors > 0].sort_values()
        y = np.linspace(0, 1, len(qerrors))
        # Extend CDF to the right: add a point at x-axis max with y=1
        x_max = XAXIS_SETTINGS[data_set]['xlim'][1]
        if len(qerrors) > 0:
            qerrors_array = qerrors.values
            # Filter out points beyond x_max and ensure curve extends to x_max
            mask = qerrors_array <= x_max
            qerrors_filtered = qerrors_array[mask]
            y_filtered = y[mask]
            
            # Always extend to x_max with y=1
            if len(qerrors_filtered) == 0:
                qerrors_extended = np.array([x_max])
                y_extended = np.array([1.0])
            elif qerrors_filtered[-1] < x_max:
                qerrors_extended = np.append(qerrors_filtered, x_max)
                y_extended = np.append(y_filtered, 1.0)
            else:
                # Last point equals x_max, ensure y is 1.0
                qerrors_extended = qerrors_filtered.copy()
                y_extended = y_filtered.copy()
                y_extended[-1] = 1.0
            
            # Smooth the curve for better visual appearance
            qerrors_smooth, y_smooth = smooth_cdf_curve(qerrors_extended, y_extended)
        else:
            qerrors_smooth = qerrors.values if len(qerrors) > 0 else np.array([x_max])
            y_smooth = y if len(qerrors) > 0 else np.array([1.0])
        filename = os.path.splitext(os.path.basename(file_path))[0]
        # Map filename to display label (e.g., OPRO -> RePro)
        label = LABEL_MAP.get(filename, filename)
        ax.plot(qerrors_smooth, y_smooth, label=label, linewidth=2.5)
    ax.set_xscale('log')
    ax.set_xlim(XAXIS_SETTINGS[data_set]['xlim'])
    ax.set_xticks(XAXIS_SETTINGS[data_set]['xticks'])
    ax.set_xticklabels(XAXIS_SETTINGS[data_set]['xticklabels'])
    ax.set_title('')
    ax.set_ylabel('CDF', fontsize=18)
    ax.set_xlabel('执行时间 Q-Error', fontsize=18)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

    # --- Memory Q-error ---
    ax = axs[1]
    for file_path in csv_files:
        df = pd.read_csv(file_path, sep=';')
        if 'Memory Q-error' in df.columns:
            df['Memory Q-error'] = df['Memory Q-error'] + 1
        df = df[df['Memory Q-error'].le(500)]
        qerrors = df['Memory Q-error'].dropna()
        qerrors = qerrors[qerrors > 0].sort_values()
        y = np.linspace(0, 1, len(qerrors))
        # Extend CDF to the right: add a point at x-axis max with y=1
        x_max = XAXIS_SETTINGS[data_set]['xlim'][1]
        if len(qerrors) > 0:
            qerrors_array = qerrors.values
            # Filter out points beyond x_max and ensure curve extends to x_max
            mask = qerrors_array <= x_max
            qerrors_filtered = qerrors_array[mask]
            y_filtered = y[mask]
            
            # Always extend to x_max with y=1
            if len(qerrors_filtered) == 0:
                qerrors_extended = np.array([x_max])
                y_extended = np.array([1.0])
            elif qerrors_filtered[-1] < x_max:
                qerrors_extended = np.append(qerrors_filtered, x_max)
                y_extended = np.append(y_filtered, 1.0)
            else:
                # Last point equals x_max, ensure y is 1.0
                qerrors_extended = qerrors_filtered.copy()
                y_extended = y_filtered.copy()
                y_extended[-1] = 1.0
            
            # Smooth the curve for better visual appearance
            qerrors_smooth, y_smooth = smooth_cdf_curve(qerrors_extended, y_extended)
        else:
            qerrors_smooth = qerrors.values if len(qerrors) > 0 else np.array([x_max])
            y_smooth = y if len(qerrors) > 0 else np.array([1.0])
        filename = os.path.splitext(os.path.basename(file_path))[0]
        # Map filename to display label (e.g., OPRO -> RePro)
        label = LABEL_MAP.get(filename, filename)
        ax.plot(qerrors_smooth, y_smooth, label=label, linewidth=2.5)
    ax.set_xscale('log')
    ax.set_xlim(XAXIS_SETTINGS[data_set]['xlim'])
    ax.set_xticks(XAXIS_SETTINGS[data_set]['xticks'])
    ax.set_xticklabels(XAXIS_SETTINGS[data_set]['xticklabels'])
    ax.set_title('')
    ax.set_ylabel('CDF', fontsize=18)
    ax.set_xlabel('内存 Q-Error', fontsize=18)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

    # 添加统一的图例到顶部
    fig.legend(['Query-Uniform', 'Stage-Independent', 'RePro'], 
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