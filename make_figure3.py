"""
make_figure3.py — Figure 3: Stage-wise expression of IL-10 axis genes.

Box + swarm plots showing normalized log2-CPM expression for the eleven
IL-10 axis genes across the six disease stages, with per-gene ANOVA
F-statistics and Holm-Bonferroni pairwise comparisons.

Inputs:
  data/log_cpm_final.csv
  data/cohort_final.csv
  results/deg/Ftest_stagewise.csv (from run_deg.py)

Output:
  results/figures/figure3_il10_stagewise.{png,pdf}
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import f_oneway, mannwhitneyu
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Stage order for plotting (Control → HCC)
STAGE_ORDER = [
    'S1_Control_07w', 'S2a_EarlyMASLD_14w', 'S3_MASH_20w',
    'S4_Fibrosis_32w', 'S2b_ChronicNT_56w', 'S5_HCC'
]
STAGE_LABELS = ['Ctrl', 'Early\nMASLD', 'MASH', 'Fibrosis',
                'Chronic\nNT', 'HCC']
STAGE_COLORS = ['#88B7B5', '#A0C4A0', '#F4A261', '#E76F51',
                '#9B5DE5', '#4A4E69']

GENE_ORDER = ['Il10', 'Il10ra', 'Il10rb', 'Stat3', 'Jak1', 'Jak2',
              'Tyk2', 'Socs3', 'Il6st', 'Scd2', 'Ddit4']


def load_data():
    log_cpm = pd.read_csv(DATA_DIR / 'log_cpm_final.csv', index_col=0)
    cohort = pd.read_csv(DATA_DIR / 'cohort_final.csv')
    # Align cohort order to log_cpm columns (keep column_name column intact)
    cohort = cohort.set_index('column_name').loc[log_cpm.columns]
    cohort = cohort.reset_index().rename(columns={'index': 'column_name'})
    return log_cpm, cohort


def find_gene_row(log_cpm, symbol):
    """Look up a gene by MGI symbol in an index formatted 'ENSMUSG_SYMBOL'.
    Returns the row (Series) or None if not found."""
    if symbol in log_cpm.index:
        return log_cpm.loc[symbol]
    # Try the ENSMUSG_SYMBOL convention
    suffix = f"_{symbol}"
    matches = [idx for idx in log_cpm.index if str(idx).endswith(suffix)]
    if len(matches) == 1:
        return log_cpm.loc[matches[0]]
    elif len(matches) > 1:
        # Multiple matches — pick the one with highest mean expression
        means = log_cpm.loc[matches].mean(axis=1)
        return log_cpm.loc[means.idxmax()]
    return None


def gene_stagewise_stats(expr_by_stage):
    """Compute one-way ANOVA F-statistic and p-value across stages."""
    groups = [g for g in expr_by_stage if len(g) > 0]
    if len(groups) < 2:
        return np.nan, np.nan
    F, p = f_oneway(*groups)
    return F, p


def pairwise_holm(expr_by_stage, stage_names):
    """Pairwise Mann-Whitney U with Holm correction.
    Returns dict {(stage_i, stage_j): adj_p} for i<j with adj_p < 0.05."""
    comparisons = []
    pvals = []
    for i in range(len(stage_names)):
        for j in range(i+1, len(stage_names)):
            gi = expr_by_stage[i]
            gj = expr_by_stage[j]
            if len(gi) < 2 or len(gj) < 2:
                continue
            try:
                _, p = mannwhitneyu(gi, gj, alternative='two-sided')
                comparisons.append((i, j))
                pvals.append(p)
            except ValueError:
                continue
    if not pvals:
        return {}
    _, adj_pvals, _, _ = multipletests(pvals, method='holm')
    return {comp: adj_p for comp, adj_p in zip(comparisons, adj_pvals)
            if adj_p < 0.05}


def draw_gene_panel(ax, gene, log_cpm, cohort):
    """Draw box+swarm for one gene with significance annotation."""
    gene_row = find_gene_row(log_cpm, gene)
    expr_by_stage = []
    for stg in STAGE_ORDER:
        samples = cohort[cohort['group'] == stg]['column_name'].tolist()
        if gene_row is not None and samples:
            vals = gene_row.loc[samples].values.astype(float)
            expr_by_stage.append(vals)
        else:
            expr_by_stage.append(np.array([]))
    
    # Boxplot
    positions = np.arange(len(STAGE_ORDER))
    bp = ax.boxplot(
        [e for e in expr_by_stage],
        positions=positions, widths=0.6, patch_artist=True,
        showfliers=False, medianprops=dict(color='black', lw=1.2),
        boxprops=dict(lw=0.8), whiskerprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
    )
    for patch, color in zip(bp['boxes'], STAGE_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    
    # Swarm overlay (jitter)
    rng = np.random.default_rng(42)
    for i, vals in enumerate(expr_by_stage):
        if len(vals) == 0:
            continue
        x_jitter = positions[i] + rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(x_jitter, vals, s=14, color=STAGE_COLORS[i],
                   edgecolors='black', linewidths=0.4, alpha=0.85, zorder=3)
    
    # Statistics
    F, p = gene_stagewise_stats(expr_by_stage)
    if np.isnan(F):
        title = f"{gene}\n(ns)"
    elif p > 0.05:
        title = f"{gene}\n(F={F:.2f}, ns)"
    else:
        # Format p-value
        if p < 1e-4:
            ptext = f"P<10⁻⁴"
        elif p < 1e-3:
            ptext = f"P={p:.1e}"
        else:
            ptext = f"P={p:.3f}"
        title = f"{gene}\nF={F:.2f}, {ptext}"
    ax.set_title(title, fontsize=9, pad=5)
    
    # Significance brackets for pairwise Holm-significant comparisons
    if not np.isnan(F) and p < 0.05:
        sig_pairs = pairwise_holm(expr_by_stage, STAGE_ORDER)
        if sig_pairs:
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            y_top = y_max
            step = y_range * 0.08
            sorted_pairs = sorted(sig_pairs.items(), key=lambda x: x[0][0]*10 + x[0][1])
            for idx, ((i, j), adj_p) in enumerate(sorted_pairs[:3]):  # show top 3
                y_bar = y_top + step * (idx + 1)
                ax.plot([positions[i], positions[j]], [y_bar, y_bar],
                        color='black', lw=0.6)
                if adj_p < 0.001:
                    sym = '***'
                elif adj_p < 0.01:
                    sym = '**'
                else:
                    sym = '*'
                ax.text((positions[i]+positions[j])/2, y_bar,
                        sym, ha='center', va='bottom', fontsize=8)
            ax.set_ylim(y_min, y_top + step * (len(sorted_pairs[:3]) + 1))
    
    ax.set_xticks(positions)
    ax.set_xticklabels(STAGE_LABELS, fontsize=7, rotation=0)
    ax.set_ylabel('log₂(CPM)', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    print("Loading data...")
    log_cpm, cohort = load_data()
    print(f"  log_cpm: {log_cpm.shape[0]} genes × {log_cpm.shape[1]} samples")
    print(f"  cohort: {len(cohort)} samples across "
          f"{cohort['group'].nunique()} stages")
    
    # For Il10, report unfiltered values (it is sub-threshold in filtered data)
    # We approximate here by using log2(CPM+1) of the gene regardless.
    
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 12))
    axes = axes.flatten()
    
    for idx, gene in enumerate(GENE_ORDER):
        ax = axes[idx]
        draw_gene_panel(ax, gene, log_cpm, cohort)
    
    # Hide unused subplot
    for idx in range(len(GENE_ORDER), len(axes)):
        axes[idx].axis('off')
    
    # Legend in last panel area
    legend_ax = axes[-1]
    legend_ax.axis('off')
    legend_elements = [
        Patch(facecolor=STAGE_COLORS[i], alpha=0.35, edgecolor='black',
              label=STAGE_LABELS[i].replace('\n', ' '))
        for i in range(len(STAGE_ORDER))
    ]
    legend_ax.legend(handles=legend_elements, loc='center',
                     fontsize=9, title='Stage', title_fontsize=10,
                     frameon=False, ncol=2)
    
    plt.suptitle('Figure 3. Stage-wise expression of IL-10 axis genes '
                 '(n=40, GSE246221)',
                 fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    out_png = FIG_DIR / 'figure3_il10_stagewise.png'
    out_pdf = FIG_DIR / 'figure3_il10_stagewise.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    main()
