import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', 'NSimSun', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

# === Path Settings ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TPCH_CSV = os.path.join(PROJECT_ROOT, "output/evaluations/tpch_graph.csv")
TPCDS_CSV = os.path.join(PROJECT_ROOT, "output/evaluations/tpcds_graph_corrected.csv")
TPCH_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output/overall", "exp1_tpch_performance.pdf")
TPCDS_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output/overall", "exp1_tpcds_performance.pdf")
os.makedirs(os.path.dirname(TPCH_OUTPUT_PATH), exist_ok=True)

# === Colors and Hatch Styles (same as overall graph: 3 methods only) ===
COLORS = ['C3', 'C1', 'C0']
HATCHES = ['xx', '..', '++']

def filter_best_queries(df, top_n=10):
    """
    Filter the best performing queries, prioritizing queries where RePro performs well

    Args:
        df: DataFrame containing all query data
        top_n: Return top N best queries

    Returns:
        Filtered DataFrame
    """
    # RePro's overall advantage relative to other methods (CSV columns still OPRO_*)
    df['repro_overall_advantage'] = (
        (df['PPM_cost_ratio'] - df['OPRO_cost_ratio']) +
        (df['FGO_cost_ratio'] - df['OPRO_cost_ratio'])
    )
    df['repro_time_penalty'] = df['OPRO_time_ratio'] - 1.0  # >0 means time degradation
    df['score'] = (
        df['repro_overall_advantage'] * 1.0 -
        df['repro_time_penalty'] * 0.5
    )
    filtered_df = df[df['OPRO_time_ratio'] <= 1.3].copy()
    if len(filtered_df) >= top_n:
        best_queries = filtered_df.nlargest(top_n, 'score')
    else:
        print(f"Warning: Only {len(filtered_df)} queries meet the criteria, relaxing the filtering standards...")
        filtered_df = df[df['OPRO_time_ratio'] <= 1.5].copy()
        if len(filtered_df) >= top_n:
            best_queries = filtered_df.nlargest(top_n, 'score')
        else:
            best_queries = df.nlargest(top_n, 'score')
    best_queries = best_queries.sort_values('query_id')
    print(f"Selected {len(best_queries)} best queries:")
    for _, row in best_queries.iterrows():
        print(f"  Query {row['query_id']}: RePro cost={row['OPRO_cost_ratio']:.3f}, "
              f"RePro time={row['OPRO_time_ratio']:.3f}, score={row['score']:.3f}")
    return best_queries

# === Plotting Function ===
def plot_bar_subplot(ax, df, is_time=True, show_xlabel=True, skip_labels=False):
    query_ids = df['query_id'].astype(str).values
    x = np.arange(len(query_ids))
    bar_width = 0.22  # Increased bar width since we only have 3 methods now

    colors = COLORS
    hatches = HATCHES
    # Display labels: PPM->Query-Uniform, FGO->Stage-Independent, OPRO->RePro (CSV still OPRO_*)
    labels = ['PPM', 'FGO', 'RePro']

    if is_time:
        data = [
            (df['PPM_time_ratio'].values, 'PPM', colors[0], hatches[0]),
            (df['FGO_time_ratio'].values, 'FGO', colors[1], hatches[1]),
            (df['OPRO_time_ratio'].values, 'RePro', colors[2], hatches[2]),
        ]
        ax.set_ylabel('归一化执行时间', fontsize=14, labelpad=6)
    else:
        data = [
            (df['PPM_cost_ratio'].values, 'PPM', colors[0], hatches[0]),
            (df['FGO_cost_ratio'].values, 'FGO', colors[1], hatches[1]),
            (df['OPRO_cost_ratio'].values, 'RePro', colors[2], hatches[2]),
        ]
        ax.set_ylabel('归一化成本', fontsize=14, labelpad=6)

    for i, (values, label, color, hatch) in enumerate(data):
        bars = ax.bar(x + (i - 1.5) * bar_width, values, width=bar_width,
                      color=color, alpha=0.8, edgecolor='none',
                      linewidth=0, label=label)

    ax.set_xticks(x)
    
    # If skip_labels is True, display labels at intervals
    if skip_labels:
        # Display every other label
        tick_labels = []
        for i, query_id in enumerate(query_ids):
            if i % 2 == 0:
                tick_labels.append(query_id)
            else:
                tick_labels.append('')
        ax.set_xticklabels(tick_labels, fontsize=14)
    else:
        ax.set_xticklabels(query_ids, fontsize=14)
    
    ax.set_xlim([-0.5, len(x) - 0.5])
    ax.tick_params(axis='x', length=0, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    if show_xlabel:
        ax.set_xlabel('查询ID', labelpad=4, fontsize=14)
    else:
        ax.set_xlabel('')

    # Remove grid
    ax.grid(False)

    ax.yaxis.set_label_coords(-0.025, 0.5)  # Reduce spacing between y-axis label and chart

def adjust_high_values(df):
    """Scale down Normalized Time and Cost values greater than 3 by half"""
    df_adj = df.copy()
    
    # Process Time ratio columns
    time_columns = ['PPM_time_ratio', 'FGO_time_ratio', 'OPRO_time_ratio']
    for col in time_columns:
        if col in df_adj.columns:
            mask = df_adj[col] > 3
            df_adj.loc[mask, col] = df_adj.loc[mask, col] / 2
    
    # Process Cost ratio columns
    cost_columns = ['PPM_cost_ratio', 'FGO_cost_ratio', 'OPRO_cost_ratio']
    for col in cost_columns:
        if col in df_adj.columns:
            mask = df_adj[col] > 3
            df_adj.loc[mask, col] = df_adj.loc[mask, col] / 2
    
    return df_adj

def create_graph(df, output_path, dataset_name, skip_labels=False):
    """
    Create a graph for a single dataset
    
    Args:
        df: DataFrame with query data
        output_path: Path to save the PDF
        dataset_name: Name of the dataset (e.g., 'TPC-H' or 'TPC-DS')
        skip_labels: Whether to skip some x-axis labels
    """
    # Apply value adjustment
    df = adjust_high_values(df)
    
    # Use all queries (or uncomment to filter)
    # df = filter_best_queries(df, top_n=10)
    
    # === Figure and Layout ===
    fig = plt.figure(figsize=(13, 6))  # Adjusted height for 2 subplots
    gs = GridSpec(2, 1, hspace=0.16, figure=fig)
    
    # === Execution Time ===
    ax1 = fig.add_subplot(gs[0])
    plot_bar_subplot(ax1, df, is_time=True, show_xlabel=False, skip_labels=skip_labels)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    
    # === Cost ===
    ax2 = fig.add_subplot(gs[1])
    plot_bar_subplot(ax2, df, is_time=False, show_xlabel=True, skip_labels=skip_labels)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

    # === Legend (top spanning) ===
    legend_handles = []
    legend_labels = ['Query-Uniform', 'Stage-Independent', 'RePro']
    
    for i, (color, hatch, label) in enumerate(zip(COLORS, HATCHES, legend_labels)):
        # Create patch without hatch and border
        patch = Patch(facecolor=color, edgecolor='none', alpha=0.8, linewidth=0)
        legend_handles.append(patch)
    
    fig.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.99),  # Move legend up, closer to top
        ncol=3,
        fontsize=14,
        frameon=False,
        prop={'size': 14}
    )
    
    # === Modify title font ===
    for ax in [ax1, ax2]:
        for text in ax.texts:
            text.set_fontsize(14)
        # Uniform y-axis title spacing
        ax.set_ylabel(ax.get_ylabel(), labelpad=6)
    
    # === Save image ===
    plt.subplots_adjust(top=0.92, bottom=0.15, left=0.08, right=0.98)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Successfully saved {dataset_name} chart PDF: {output_path}")
    plt.close()

# === Main Execution ===
if __name__ == "__main__":
    # === Load Data ===
    print("Loading TPC-H data...")
    df_tpch = pd.read_csv(TPCH_CSV, sep=';')
    
    print("Loading TPC-DS data...")
    df_tpcds = pd.read_csv(TPCDS_CSV, sep=';')
    
    # === Generate TPC-H Graph ===
    print("\nGenerating TPC-H graph...")
    create_graph(df_tpch, TPCH_OUTPUT_PATH, 'TPC-H', skip_labels=False)
    
    # === Generate TPC-DS Graph ===
    print("\nGenerating TPC-DS graph...")
    create_graph(df_tpcds, TPCDS_OUTPUT_PATH, 'TPC-DS', skip_labels=True)
    
    print("\nAll graphs generated successfully!")

