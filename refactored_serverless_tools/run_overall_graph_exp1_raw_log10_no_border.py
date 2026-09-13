import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch


plt.rcParams['font.family'] = 'Arial'

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TPCH_CSV = PROJECT_ROOT / 'output' / 'evaluations' / 'tpch_graph_corrected.csv'
TPCDS_CSV = PROJECT_ROOT / 'output' / 'evaluations' / 'tpcds_graph_corrected.csv'
OUTPUT_PATH = Path(os.environ.get(
    'OVERALL_GRAPH_OUTPUT_PATH',
    PROJECT_ROOT / 'output' / 'overall' / 'exp1_overall_performance_raw_log10_no_border.pdf',
))

METHODS = [
    ('PPM', 'Query-uniform', 'C3'),
    ('FGO', 'Stage-independent', 'C1'),
    ('OPRO', 'PIPER', 'C0'),
]


def load_data(path):
    df = pd.read_csv(path, sep=';')
    required_columns = ['query_id']
    for method, _label, _color in METHODS:
        required_columns.extend([
            f'{method}_execution_time',
            f'{method}_cost',
        ])

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f'{path.name} is missing columns: {missing}')

    numeric_columns = [column for column in required_columns if column != 'query_id']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='raise')
    non_positive = [
        column for column in numeric_columns
        if (df[column] <= 0).any()
    ]
    if non_positive:
        raise ValueError(
            f'{path.name} has non-positive values in columns: {non_positive}'
        )
    return df


def format_power_tick(value, _position):
    if np.isclose(value, 0.0):
        return '1'
    exponent = f'{value:g}'
    return rf'$10^{{{exponent}}}$'


def plot_log10_subplot(
        ax, df, metric, show_xlabel=True, skip_labels=False,
        empty_labels=False):
    query_ids = df['query_id'].astype(str).values
    x = np.arange(len(query_ids))
    bar_width = 0.22

    source_suffix = 'execution_time' if metric == 'time' else 'cost'
    for index, (method, label, color) in enumerate(METHODS):
        raw_values = df[f'{method}_{source_suffix}'].to_numpy(dtype=float)
        ax.bar(
            x + (index - 1) * bar_width,
            np.log10(raw_values),
            width=bar_width,
            color=color,
            alpha=0.85,
            edgecolor='black',
            linewidth=0.6,
            label=label,
        )

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_power_tick))
    ax.set_ylabel(
        'Latency (ms)' if metric == 'time' else 'Cost ($)',
        fontsize=12,
        labelpad=6,
    )
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.set_xticks(x)
    if empty_labels:
        tick_labels = [''] * len(query_ids)
    elif skip_labels:
        tick_labels = [
            query_id if index % 2 == 0 else ''
            for index, query_id in enumerate(query_ids)
        ]
    else:
        tick_labels = query_ids
    ax.set_xticklabels(tick_labels, fontsize=12)

    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.tick_params(axis='x', length=0, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel('Query ID' if show_xlabel else '', fontsize=12, labelpad=4)
    ax.grid(False)


def main():
    df_tpch = load_data(TPCH_CSV)
    df_tpcds = load_data(TPCDS_CSV)

    fig = plt.figure(figsize=(13, 8))
    outer_gs = GridSpec(
        2, 1, height_ratios=[0.85, 0.75], hspace=0.25, figure=fig
    )
    inner_gs_tpch = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0], hspace=0.16
    )
    inner_gs_tpcds = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[1], hspace=0.16
    )

    ax1 = fig.add_subplot(inner_gs_tpch[0])
    plot_log10_subplot(
        ax1, df_tpch, 'time', show_xlabel=False, empty_labels=True
    )

    ax2 = fig.add_subplot(inner_gs_tpch[1])
    plot_log10_subplot(ax2, df_tpch, 'cost')
    ax2.annotate(
        '(a) TPC-H', xy=(0.5, -0.40), xycoords='axes fraction',
        ha='center', va='center', fontsize=12
    )

    ax3 = fig.add_subplot(inner_gs_tpcds[0])
    plot_log10_subplot(
        ax3, df_tpcds, 'time', show_xlabel=False, empty_labels=True
    )

    ax4 = fig.add_subplot(inner_gs_tpcds[1])
    plot_log10_subplot(ax4, df_tpcds, 'cost', skip_labels=True)
    ax4.annotate(
        '(b) TPC-DS', xy=(0.5, -0.45), xycoords='axes fraction',
        ha='center', va='center', fontsize=12
    )

    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor='black',
            linewidth=0.6,
            alpha=0.85,
            label=label,
        )
        for _method, label, color in METHODS
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        fontsize=12,
        frameon=False,
    )

    plt.subplots_adjust(top=0.92, bottom=0.15, left=0.08, right=0.98)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Successfully saved raw log10 chart PDF: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
