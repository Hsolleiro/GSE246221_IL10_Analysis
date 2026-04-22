"""
make_figureS1_QC.py — Supplementary Figure S1: Quality Control

Four panels:
  A) Library size distribution per sample (colored by stage)
  B) Number of detected genes (log2-CPM > 1) per sample
  C) PCA of log2-CPM values (PC1 vs PC2)
  D) Sample-sample distance heatmap (hierarchical clustering)

Inputs:
  data/counts_final.csv
  data/log_cpm_final.csv
  data/cohort_final.csv

Output:
  results/figures/figureS1_QC.{png,pdf}
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STAGE_ORDER = [
    'S1_Control_07w', 'S2a_EarlyMASLD_14w', 'S3_MASH_20w',
    'S4_Fibrosis_32w', 'S2b_ChronicNT_56w', 'S5_HCC'
]
STAGE_LABELS = ['Ctrl', 'Early MASLD', 'MASH', 'Fibrosis',
                'Chronic NT', 'HCC']
STAGE_COLORS = ['#88B7B5', '#A0C4A0', '#F4A261', '#E76F51',
                '#9B5DE5', '#4A4E69']


def load_inputs():
    counts = pd.read_csv(DATA_DIR / 'counts_final.csv', index_col=0)
    log_cpm = pd.read_csv(DATA_DIR / 'log_cpm_final.csv', index_col=0)
    cohort = pd.read_csv(DATA_DIR / 'cohort_final.csv')
    cohort = cohort.set_index('column_name').loc[log_cpm.columns]
    cohort = cohort.reset_index().rename(columns={'index': 'column_name'})
    return counts, log_cpm, cohort


def panel_a_library_sizes(ax, counts, cohort):
    """Panel A: library size per sample, bars colored by stage."""
    lib_sizes = counts.sum(axis=0)  # per sample
    # Order samples by stage
    order = []
    for stg in STAGE_ORDER:
        smps = cohort[cohort['group'] == stg]['column_name'].tolist()
        order.extend(smps)
    
    lib_sizes = lib_sizes.loc[order]
    
    # Assign color
    colors = []
    for sample in order:
        stg = cohort[cohort['column_name'] == sample]['group'].iloc[0]
        colors.append(STAGE_COLORS[STAGE_ORDER.index(stg)])
    
    x = np.arange(len(order))
    ax.bar(x, lib_sizes.values / 1e6, color=colors, edgecolor='black', lw=0.3)
    ax.set_xticks([])
    ax.set_ylabel('Library size (millions of reads)', fontsize=9)
    ax.set_title('A. Library size per sample',
                 fontsize=10, loc='left', fontweight='bold')
    ax.axhline(lib_sizes.median() / 1e6, color='black',
               linestyle='--', lw=0.6, alpha=0.7,
               label=f'Median: {lib_sizes.median()/1e6:.1f}M')
    ax.legend(fontsize=7, loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def panel_b_genes_detected(ax, log_cpm, cohort):
    """Panel B: number of genes detected per sample."""
    # Gene is "detected" if log2-CPM > 1 in that sample
    genes_detected = (log_cpm > 1).sum(axis=0)
    
    order = []
    for stg in STAGE_ORDER:
        smps = cohort[cohort['group'] == stg]['column_name'].tolist()
        order.extend(smps)
    
    genes_detected = genes_detected.loc[order]
    colors = []
    for sample in order:
        stg = cohort[cohort['column_name'] == sample]['group'].iloc[0]
        colors.append(STAGE_COLORS[STAGE_ORDER.index(stg)])
    
    x = np.arange(len(order))
    ax.bar(x, genes_detected.values, color=colors, edgecolor='black', lw=0.3)
    ax.set_xticks([])
    ax.set_ylabel('Detected genes (log₂-CPM > 1)', fontsize=9)
    ax.set_title('B. Gene detection per sample',
                 fontsize=10, loc='left', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def panel_c_pca(ax, log_cpm, cohort):
    """Panel C: PCA of log2-CPM values."""
    # Use top 2000 most variable genes
    variance = log_cpm.var(axis=1)
    top_genes = variance.nlargest(2000).index
    X = log_cpm.loc[top_genes].T.values  # samples × genes
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_ * 100
    
    for stg, color in zip(STAGE_ORDER, STAGE_COLORS):
        smps = cohort[cohort['group'] == stg]['column_name'].tolist()
        idxs = [list(log_cpm.columns).index(s) for s in smps
                if s in log_cpm.columns]
        if not idxs:
            continue
        ax.scatter(coords[idxs, 0], coords[idxs, 1],
                    color=color, s=70, edgecolors='black', lw=0.5,
                    alpha=0.85,
                    label=STAGE_LABELS[STAGE_ORDER.index(stg)])
    
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)', fontsize=9)
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)', fontsize=9)
    ax.set_title('C. PCA on top 2000 variable genes',
                 fontsize=10, loc='left', fontweight='bold')
    ax.legend(fontsize=7, loc='best', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0, color='grey', lw=0.3, alpha=0.5)
    ax.axvline(0, color='grey', lw=0.3, alpha=0.5)


def panel_d_sample_distance(ax, log_cpm, cohort):
    """Panel D: sample-sample distance heatmap."""
    # Use top 2000 most variable genes
    variance = log_cpm.var(axis=1)
    top_genes = variance.nlargest(2000).index
    X = log_cpm.loc[top_genes].T.values  # samples × genes
    
    # Order samples by stage
    order = []
    order_colors = []
    for stg in STAGE_ORDER:
        smps = cohort[cohort['group'] == stg]['column_name'].tolist()
        order.extend(smps)
        order_colors.extend([STAGE_COLORS[STAGE_ORDER.index(stg)]] * len(smps))
    
    idx_order = [list(log_cpm.columns).index(s) for s in order]
    X_ord = X[idx_order]
    
    # Euclidean distance
    D = squareform(pdist(X_ord, metric='euclidean'))
    
    im = ax.imshow(D, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Color bar on the side showing stage
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Stage boundaries
    stage_boundaries = [0]
    for stg in STAGE_ORDER:
        count = (cohort['group'] == stg).sum()
        stage_boundaries.append(stage_boundaries[-1] + count)
    for b in stage_boundaries[1:-1]:
        ax.axhline(b - 0.5, color='white', lw=0.8)
        ax.axvline(b - 0.5, color='white', lw=0.8)
    
    # Stage label centers
    centers = [(stage_boundaries[i] + stage_boundaries[i+1]) / 2
               for i in range(len(stage_boundaries) - 1)]
    ax.set_xticks(centers)
    ax.set_xticklabels(STAGE_LABELS, fontsize=7, rotation=45, ha='right')
    ax.set_yticks(centers)
    ax.set_yticklabels(STAGE_LABELS, fontsize=7)
    
    ax.set_title('D. Sample-sample Euclidean distance',
                 fontsize=10, loc='left', fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label('Euclidean distance', fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def main():
    print("Loading inputs...")
    counts, log_cpm, cohort = load_inputs()
    print(f"  counts: {counts.shape}")
    print(f"  log_cpm: {log_cpm.shape}")
    print(f"  cohort: {len(cohort)} samples")
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig,
                   hspace=0.38, wspace=0.28,
                   left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    
    panel_a_library_sizes(ax_A, counts, cohort)
    panel_b_genes_detected(ax_B, log_cpm, cohort)
    panel_c_pca(ax_C, log_cpm, cohort)
    panel_d_sample_distance(ax_D, log_cpm, cohort)
    
    # Shared legend
    legend_elements = [
        Patch(facecolor=c, edgecolor='black', label=lbl)
        for c, lbl in zip(STAGE_COLORS, STAGE_LABELS)
    ]
    fig.legend(handles=legend_elements, loc='lower center',
                ncol=6, fontsize=8, frameon=False,
                bbox_to_anchor=(0.5, 0.01))
    
    plt.suptitle('Supplementary Figure S1. Quality control of the '
                 'curated cohort (n=40, GSE246221)',
                 fontsize=12, fontweight='bold', y=0.97)
    
    out_png = FIG_DIR / 'figureS1_QC.png'
    out_pdf = FIG_DIR / 'figureS1_QC.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    main()
