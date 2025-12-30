#!/usr/bin/env python3
"""
Plot Results for CP-HNSW PhD Portfolio Paper

Generates publication-quality plots from experiment CSV files.

Usage:
    python plot_results.py --input results/ --output results/plots/

Requirements:
    pip install pandas matplotlib numpy
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette (colorblind-friendly)
COLORS = {
    'Faiss_HNSW': '#1f77b4',      # Blue
    'Faiss_IVFPQ_GPU': '#ff7f0e',  # Orange
    'Faiss_Flat_GPU': '#2ca02c',   # Green
    'CP_HNSW_CPU': '#d62728',      # Red
    'CP_HNSW_Rerank': '#9467bd',   # Purple
}

MARKERS = {
    'Faiss_HNSW': 'o',
    'Faiss_IVFPQ_GPU': 's',
    'Faiss_Flat_GPU': '^',
    'CP_HNSW_CPU': 'D',
    'CP_HNSW_Rerank': '*',
}


def plot_exp1_money_plot(input_dir, output_dir):
    """Plot Experiment 1: Recall vs QPS (Money Plot)."""
    print("Generating: Money Plot (Recall vs QPS)...")

    cphnsw_path = os.path.join(input_dir, 'exp1_recall_qps', 'cphnsw_results.csv')
    faiss_path = os.path.join(input_dir, 'exp1_recall_qps', 'faiss_results.csv')

    fig, ax = plt.subplots(figsize=(8, 6))

    # Load and plot CP-HNSW results
    if os.path.exists(cphnsw_path):
        df = pd.read_csv(cphnsw_path)

        for system in df['system'].unique():
            subset = df[df['system'] == system]
            color = COLORS.get(system, '#333333')
            marker = MARKERS.get(system, 'o')
            label = system.replace('_', ' ')

            ax.scatter(subset['qps'], subset['recall_10'],
                      c=color, marker=marker, s=80, label=label, alpha=0.8)

            # Connect points with lines
            sorted_subset = subset.sort_values('recall_10')
            ax.plot(sorted_subset['qps'], sorted_subset['recall_10'],
                   c=color, alpha=0.3, linestyle='--')

    # Load and plot Faiss results
    if os.path.exists(faiss_path):
        df = pd.read_csv(faiss_path)

        for system in df['system'].unique():
            subset = df[df['system'] == system]
            color = COLORS.get(system, '#333333')
            marker = MARKERS.get(system, 'o')
            label = system.replace('_', ' ')

            ax.scatter(subset['qps'], subset['recall_10'],
                      c=color, marker=marker, s=80, label=label, alpha=0.8)

            sorted_subset = subset.sort_values('recall_10')
            ax.plot(sorted_subset['qps'], sorted_subset['recall_10'],
                   c=color, alpha=0.3, linestyle='--')

    ax.set_xlabel('Queries Per Second (QPS)')
    ax.set_ylabel('Recall@10')
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(100, None)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Recall vs Throughput (SIFT-1M)')

    output_path = os.path.join(output_dir, 'exp1_money_plot.pdf')
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.pdf', '.png'))
    plt.close()
    print(f"  Saved: {output_path}")


def plot_exp2_scalability(input_dir, output_dir):
    """Plot Experiment 2: Build Scalability."""
    print("Generating: Thread Scaling Plot...")

    csv_path = os.path.join(input_dir, 'exp2_scalability', 'thread_scaling.csv')
    if not os.path.exists(csv_path):
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Throughput vs Threads
    ax1.plot(df['threads'], df['throughput_vps'], 'o-', color='#1f77b4',
             markersize=8, linewidth=2)
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Throughput (vectors/s)')
    ax1.set_title('Build Throughput vs Threads')
    ax1.grid(True, alpha=0.3)

    # Speedup vs Threads
    ax2.plot(df['threads'], df['speedup'], 's-', color='#2ca02c',
             markersize=8, linewidth=2, label='Actual')
    ax2.plot(df['threads'], df['threads'], '--', color='gray',
             alpha=0.5, label='Linear (ideal)')
    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Parallel Speedup')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'exp2_scalability.pdf')
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.pdf', '.png'))
    plt.close()
    print(f"  Saved: {output_path}")


def plot_exp3_ablation(input_dir, output_dir):
    """Plot Experiment 3: Topology Ablation."""
    print("Generating: Ablation Bar Chart...")

    csv_path = os.path.join(input_dir, 'exp3_ablation', 'topology_comparison.csv')
    if not os.path.exists(csv_path):
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(df))
    width = 0.6

    # Connectivity
    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
    bars1 = ax1.bar(x, df['connectivity_pct'], width, color=colors)
    ax1.set_ylabel('Connectivity (%)')
    ax1.set_title('Graph Connectivity')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['edge_selection'], rotation=15)
    ax1.set_ylim(0, 110)
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # Graph Recall
    bars2 = ax2.bar(x, df['graph_recall_10'], width, color=colors)
    ax2.set_ylabel('Graph Recall@10')
    ax2.set_title('Search Quality (no reranking)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['edge_selection'], rotation=15)
    ax2.set_ylim(0, 0.35)

    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'exp3_ablation.pdf')
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.pdf', '.png'))
    plt.close()
    print(f"  Saved: {output_path}")


def plot_exp4_correlation(input_dir, output_dir):
    """Plot Experiment 4: Estimator Correlation."""
    print("Generating: Correlation Bar Chart...")

    csv_path = os.path.join(input_dir, 'exp4_correlation', 'correlation_data.csv')
    if not os.path.exists(csv_path):
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(df))
    width = 0.6

    labels = [f"K={row['K']}" for _, row in df.iterrows()]
    bars = ax.bar(x, df['pearson_r'], width, color='#1f77b4')

    ax.set_ylabel('Pearson Correlation (r)')
    ax.set_title('CP Estimator Correlation with True Cosine')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Usability threshold (r=0.7)')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'exp4_correlation.pdf')
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.pdf', '.png'))
    plt.close()
    print(f"  Saved: {output_path}")


def plot_exp5_memory(input_dir, output_dir):
    """Plot Experiment 5: Memory Footprint."""
    print("Generating: Memory Comparison Chart...")

    csv_path = os.path.join(input_dir, 'exp5_memory', 'memory_breakdown.csv')
    if not os.path.exists(csv_path):
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # Filter for total values
    cp_total = df[(df['system'] == 'CP_HNSW') & (df['component'] == 'Index Only')]['MB'].values
    faiss_total = df[(df['system'] == 'Faiss_HNSW') & (df['component'] == 'Total')]['MB'].values

    if len(cp_total) == 0 or len(faiss_total) == 0:
        print("  Skipping: insufficient data")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    systems = ['Faiss HNSW\n(Float32)', 'CP-HNSW\n(32 bytes)']
    values = [faiss_total[0], cp_total[0]]
    colors = ['#1f77b4', '#d62728']

    bars = ax.bar(systems, values, color=colors, width=0.5)

    ax.set_ylabel('Index Memory (MB)')
    ax.set_title('Memory Footprint Comparison (1M vectors)')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f} MB',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    compression = faiss_total[0] / cp_total[0]
    ax.annotate(f'{compression:.1f}x smaller',
                xy=(1, cp_total[0] / 2),
                fontsize=12, ha='center', color='#d62728', fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'exp5_memory.pdf')
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.pdf', '.png'))
    plt.close()
    print(f"  Saved: {output_path}")


def generate_latex_tables(input_dir, output_dir):
    """Generate LaTeX tables from CSV results."""
    print("Generating: LaTeX Tables...")

    # Table 2: Scalability
    csv_path = os.path.join(input_dir, 'exp2_scalability', 'thread_scaling.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        latex = df.to_latex(index=False, float_format='%.2f',
                           column_format='rrrrl')
        output_path = os.path.join(output_dir, 'table2_scalability.tex')
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"  Saved: {output_path}")

    # Table 3: Ablation
    csv_path = os.path.join(input_dir, 'exp3_ablation', 'topology_comparison.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        latex = df.to_latex(index=False, float_format='%.2f')
        output_path = os.path.join(output_dir, 'table3_ablation.tex')
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate plots from CP-HNSW results')
    parser.add_argument('--input', type=str, default='results', help='Input directory with CSV files')
    parser.add_argument('--output', type=str, default='results/plots', help='Output directory for plots')
    args = parser.parse_args()

    print("=== CP-HNSW Results Plotter ===\n")

    os.makedirs(args.output, exist_ok=True)

    # Generate all plots
    plot_exp1_money_plot(args.input, args.output)
    plot_exp2_scalability(args.input, args.output)
    plot_exp3_ablation(args.input, args.output)
    plot_exp4_correlation(args.input, args.output)
    plot_exp5_memory(args.input, args.output)
    generate_latex_tables(args.input, args.output)

    print(f"\n=== All plots saved to: {args.output} ===")


if __name__ == '__main__':
    main()
