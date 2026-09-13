#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import font_manager as fm
import numpy as np
import os
from matplotlib.patches import Rectangle

def load_qerror_data(file_path):
    """Load Q-Error data"""
    # 先检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path, sep=';')
        # 将Q-error值加上1（之前是-1的，现在需要补回来）
        qerror_columns = ['p50_q_error_time', 'p90_q_error_time', 'avg_q_error_time', 
                         'p50_q_error_mem', 'p90_q_error_mem', 'avg_q_error_mem']
        for col in qerror_columns:
            if col in df.columns:
                df[col] = df[col] + 1
        print(f"成功加载数据: {file_path}, 共{len(df)}行")
        return df
    except Exception as e:
        print(f"加载数据失败: {file_path}, 错误: {e}")
        return None

def filter_reasonable_data(df):
    """Filter out extremely large values, keep reasonable data"""
    # Filter out anomalous values from PPM-GNN and PPM-NN
    # Keep Auto-DOP and Query-Level data
    valid_models = ['(Operator-Level)']  # 修正模型名称
    filtered_df = df[df['model'].isin(valid_models)].copy()
    
    # Filter out infinite values
    for col in ['p50_q_error_time', 'p90_q_error_time', 'avg_q_error_time']:
        filtered_df = filtered_df[filtered_df[col] != np.inf]
        filtered_df = filtered_df[filtered_df[col] < 1000]  # Filter out extremely large values
    
    return filtered_df

def setup_sigmod_style():
    """Setup SIGMOD paper style parameters"""
    plot_params = {
        'axes.labelsize': '16',  # 轴标签字体大小 +2
        'xtick.labelsize': '16', # x轴刻度字体大小 +2
        'ytick.labelsize': '16', # y轴刻度字体大小 +2
        'lines.linewidth': 1,
        'legend.fontsize': '16',  # 图例字体大小 +2
        'figure.figsize': '14, 6',
        'legend.loc': 'upper center',
        'legend.frameon': False,
        'font.family': ['Times New Roman', 'SimSun', 'NSimSun', 'STSong'],
        'axes.unicode_minus': False,
    }
    
    bar_params = {
        'ncol': 3,  # legend columns
        'anchor': (0.5, 1.15),  # legend position
        'columnspacing': 2,  # horizontal legend spacing
        'labelspacing': 0.5,  # vertical legend spacing
        'handletextpad': 0.1,  # text distance
        'handleheight': 0.7,  # legend height
        'handlelength': 1.2,  # legend width
        'bar_width': 0.25,
        'bar_line': 0.5,
        'colors': ['C3', 'C1', 'C2'],  # 参考multi_bar.py的配色
        'hatchs': ['xx', '..', '**'],  # 参考multi_bar.py的花纹
    }
    
    return plot_params, bar_params

def create_sigmod_cardinality_plot(tpch_data, tpcds_data, output_dir):
    """Create SIGMOD-style cardinality comparison plot with time and memory"""
    
    plot_params, bar_params = setup_sigmod_style()
    pylab.rcParams.update(plot_params)
    plt.rcParams['pdf.fonttype'] = 42
    # Single FontProperties for axis titles so x/y labels share the same family/size/weight
    axis_label_font = fm.FontProperties(
        family=plot_params['font.family'],
        size=int(plot_params['axes.labelsize']),
        weight='normal',
    )
    
    # Define experiment type mapping
    experiment_mapping = {
        'exact_exact': 'Exact Train, Exact Eval.',
        'exact_estimated': 'Exact Train, Estimated Eval.',
        'estimated_estimated': 'Estimated Train, Estimated Eval.'
    }
    
    # Extract Auto-DOP data for both time and memory
    def extract_auto_dop_data(df):
        auto_dop = df[df['model'] == '(Operator-Level)']  # 修正模型名称
        data = {}
        for exp_type in ['exact_exact', 'exact_estimated', 'estimated_estimated']:
            row = auto_dop[auto_dop['experiment_type'] == exp_type]
            if not row.empty:
                data[exp_type] = {
                    'time': {
                        'p50': float(row['p50_q_error_time'].iloc[0]),
                        'p90': float(row['p90_q_error_time'].iloc[0]),
                        'avg': float(row['avg_q_error_time'].iloc[0])
                    },
                    'memory': {
                        'p50': float(row['p50_q_error_mem'].iloc[0]),
                        'p90': float(row['p90_q_error_mem'].iloc[0]),
                        'avg': float(row['avg_q_error_mem'].iloc[0])
                    }
                }
        return data
    
    # 只处理TPC-H数据
    if tpch_data is None:
        print("没有TPC-H数据进行绘图！")
        return
    
    tpch_auto_dop = extract_auto_dop_data(tpch_data)
    if not tpch_auto_dop:
        print("TPC-H数据中没有找到(Operator-Level)模型数据！")
        return
    
    # Create figure - 1行2列布局（时间和内存）
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.3)  # 适当增加子图间距以便对齐
    
    # 数据类型：时间和内存
    metric_types = [('time', '执行时间 Q-Error'), ('memory', '内存 Q-Error')]
    legend_handles = []
    legend_labels = []
    
    for idx, (metric_type, ylabel) in enumerate(metric_types):
        ax = axes[idx]
        
        # Prepare data
        metrics = ['p50', 'p90', 'avg']
        x_pos = np.arange(len(metrics))
        width = bar_params['bar_width']
        
        # Plot bars - 实现白框效果
        for i, (exp_type, label) in enumerate(experiment_mapping.items()):
            if exp_type in tpch_auto_dop:
                values = [tpch_auto_dop[exp_type][metric_type][metric] for metric in metrics]
                # 第一层：有颜色和花纹的条形图，白色边框
                bars1 = ax.bar(x_pos + i * width, values, width, 
                             color=bar_params['colors'][i], 
                             hatch=bar_params['hatchs'][i],
                             alpha=0.8, edgecolor='white',
                             linewidth=1)
                # 第二层：无颜色的条形图，黑色边框
                bars2 = ax.bar(x_pos + i * width, values, width,
                             color='none', edgecolor='black',
                             linewidth=plot_params['lines.linewidth'])
                
                # 保存图例句柄（仅第一个subplot需要）
                if idx == 0:
                    legend_handles.append((bars1[0], bars2[0]))
                    legend_labels.append(label)
                
                # Add value labels on bars
                for bar, value in zip(bars1, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=15)
        
        # Set axes labels (same font object for x/y so mixed CJK/Latin renders consistently)
        ax.set_xlabel('Q-Error 指标', fontproperties=axis_label_font)
        ax.set_ylabel(ylabel, fontproperties=axis_label_font)
        # 横坐标只保留p50/p90/Avg
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(['p50', 'p90', 'Avg'])
        
        # Set grid
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Set spines
        ax.spines[['right', 'top']].set_visible(True)
        ax.tick_params(bottom=True, left=True)
        for spine in ax.spines.values():
            spine.set_linewidth(plot_params['lines.linewidth'])
        
        # Adjust y-axis range - 从1开始
        all_values = []
        for exp_type in tpch_auto_dop.keys():
            all_values.extend(tpch_auto_dop[exp_type][metric_type].values())
        if all_values:
            max_val = max(all_values)
        y_ticks = ax.get_yticks()
        if 1.0 not in y_ticks:
            y_ticks = np.insert(y_ticks, 0, 1.0)
            ax.set_yticks(y_ticks)
        ax.axhline(1.0, color='gray', linewidth=1, linestyle='--', alpha=0.7, zorder=0)
        # 最后再设置y轴范围，确保从1开始
        if all_values:
            ax.set_ylim(1.0, max_val * 1.15)
    
    # Add legend（使用组合图例句柄实现白框效果）
    fig.legend(legend_handles, legend_labels, 
              loc='upper center', bbox_to_anchor=(0.5, 0.96), 
              ncol=bar_params['ncol'], fontsize=16, frameon=False,
              columnspacing=1.5, labelspacing=0.3, handletextpad=0.2,
              handleheight=1.0, handlelength=1.5)
    
    # Adjust layout - 删除标题后调整空间
    plt.tight_layout(rect=(0, 0.05, 1, 0.88))
    plt.subplots_adjust(bottom=0.10, top=0.82)
    
    # Save figure
    output_path = os.path.join(output_dir, 'exp4_estimate_qerror.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=1000, pad_inches=0.1)
    print(f"TPC-H Q-error对比图表已保存至: {output_path}")


def main():
    """Main function"""
    # Data file paths - 使用相对路径避免转义问题
    tpch_file = '../output/tpch/evaluations/qerror_summary_report.csv'
    output_dir = '../output/evaluations'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("正在加载TPC-H数据...")
    tpch_data = load_qerror_data(tpch_file)
    
    if tpch_data is None:
        print("加载TPC-H数据失败！文件不存在或无法读取。")
        return
    
    print(f"TPC-H数据: {len(tpch_data)} 条记录")
    
    # Create Q-error comparison plot
    print("正在创建TPC-H Q-error对比图...")
    create_sigmod_cardinality_plot(tpch_data, None, output_dir)

if __name__ == "__main__":
    main()
