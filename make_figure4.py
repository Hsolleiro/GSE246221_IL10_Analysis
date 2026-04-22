"""
make_figure4.py — Figure 4: Cell-type signature remodeling across
MASLD → HCC and association with the IL-10 axis.

Four panels:
  A) Gene-level heatmap of signature markers (z-scored log2-CPM)
  B) Stage-resolved trajectories (z-scored per signature) for 4 cell types
  C) Per-signature distribution across stages with ANOVA F-stats
  D) Correlation heatmap: signatures × IL-10 axis genes

Inputs:
  data/log_cpm_final.csv
  data/cohort_final.csv
  data/signature_panel.csv
  results/celltypes/scores_median.csv
  results/celltypes/scores_zscore.csv
  results/celltypes/anova_per_signature.csv
  results/celltypes/il10_correlations_r.csv
  results/celltypes/il10_correlations_fdr.csv

Output:
  results/figures/figure4_celltype_signatures.{png,pdf}
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import f_oneway, mannwhitneyu
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CELLTYPES_DIR = RESULTS_DIR / "celltypes"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STAGE_ORDER = [
    'S1_Control_07w', 'S2a_EarlyMASLD_14w', 'S3_MASH_20w',
    'S4_Fibrosis_32w', 'S2b_ChronicNT_56w', 'S5_HCC'
]
STAGE_LABELS = ['Ctrl', 'Early\nMASLD', 'MASH', 'Fibrosis',
                'Chronic\nNT', 'HCC']
STAGE_COLORS = ['#88B7B5', '#A0C4A0', '#F4A261', '#E76F51',
                '#9B5DE5', '#4A4E69']

CELLTYPE_ORDER = ['Hepatocytes', 'Macrophages_Monocytes',
                  'NK_cells', 'HSC_Fibrosis']
CELLTYPE_LABELS = ['Hepatocytes', 'Macrophages /\nMonocytes',
                   'NK cells', 'HSC / Fibrosis']
CELLTYPE_COLORS = {'Hepatocytes': '#2E86AB',
                   'Macrophages_Monocytes': '#E63946',
                   'NK_cells': '#F4A261',
                   'HSC_Fibrosis': '#6A4C93'}

GENE_ORDER_IL10 = ['Il10ra', 'Il10rb', 'Stat3', 'Jak1', 'Jak2',
                   'Socs3', 'Il6st', 'Scd2', 'Ddit4']


def find_gene_row(log_cpm, symbol):
    if symbol in log_cpm.index:
        return log_cpm.loc[symbol]
    suffix = f"_{symbol}"
    matches = [idx for idx in log_cpm.index if str(idx).endswith(suffix)]
    if len(matches) == 1:
        return log_cpm.loc[matches[0]]
    elif len(matches) > 1:
        means = log_cpm.loc[matches].mean(axis=1)
        return log_cpm.loc[means.idxmax()]
    return None


def load_inputs():
    log_cpm = pd.read_csv(DATA_DIR / 'log_cpm_final.csv', index_col=0)
    cohort = pd.read_csv(DATA_DIR / 'cohort_final.csv')
    cohort = cohort.set_index('column_name').loc[log_cpm.columns]
    cohort = cohort.reset_index().rename(columns={'index': 'column_name'})
    sig_panel = pd.read_csv(DATA_DIR / 'signature_panel.csv')
    scores = pd.read_csv(CELLTYPES_DIR / 'scores_median.csv', index_col=0)
    anova = pd.read_csv(CELLTYPES_DIR / 'anova_per_signature.csv')
    corr_r = pd.read_csv(CELLTYPES_DIR / 'il10_correlations_r.csv', index_col=0)
    corr_fdr = pd.read_csv(CELLTYPES_DIR / 'il10_correlations_fdr.csv',
                            index_col=0)
    return log_cpm, cohort, sig_panel, scores, anova, corr_r, corr_fdr


def panel_a_gene_heatmap(ax, log_cpm, cohort, sig_panel):
    """Panel A: gene-level heatmap of signature markers (z-scored)."""
    # Get gene list per signature (use up to 12 per celltype)
    gene_rows = []
    row_celltype = []
    for ct in CELLTYPE_ORDER:
        genes_for_ct = sig_panel[sig_panel['signature'] == ct]['gene'].tolist()
        # Keep top 12 by mean expression
        present = [g for g in genes_for_ct if find_gene_row(log_cpm, g) is not None]
        present = present[:12]
        for g in present:
            gene_rows.append(g)
            row_celltype.append(ct)
    
    # Order samples by stage
    sample_order = []
    stage_boundaries = []
    for stg in STAGE_ORDER:
        smps = cohort[cohort['group'] == stg]['column_name'].tolist()
        sample_order.extend(smps)
        stage_boundaries.append(len(sample_order))
    
    # Build matrix: genes × samples
    mat = np.zeros((len(gene_rows), len(sample_order)))
    for i, g in enumerate(gene_rows):
        row = find_gene_row(log_cpm, g)
        if row is not None:
            mat[i, :] = row.loc[sample_order].values
        else:
            mat[i, :] = np.nan
    
    # Z-score per row
    mat_z = (mat - np.nanmean(mat, axis=1, keepdims=True)) / \
            (np.nanstd(mat, axis=1, keepdims=True) + 1e-9)
    mat_z = np.clip(mat_z, -2.5, 2.5)
    
    im = ax.imshow(mat_z, aspect='auto', cmap='RdBu_r',
                    vmin=-2.5, vmax=2.5, interpolation='nearest')
    
    # Vertical lines separating stages
    for b in stage_boundaries[:-1]:
        ax.axvline(b - 0.5, color='black', lw=0.5)
    
    # Horizontal lines separating cell types
    prev_ct = row_celltype[0]
    for i, ct in enumerate(row_celltype):
        if ct != prev_ct:
            ax.axhline(i - 0.5, color='black', lw=0.8)
            prev_ct = ct
    
    # Stage labels at bottom
    stage_mid = [0] + list(stage_boundaries)
    stage_centers = [(stage_mid[i] + stage_mid[i+1]) / 2 for i in range(len(stage_mid)-1)]
    ax.set_xticks(stage_centers)
    ax.set_xticklabels(STAGE_LABELS, fontsize=7)
    
    # Gene labels on y
    ax.set_yticks(range(len(gene_rows)))
    ax.set_yticklabels(gene_rows, fontsize=5.5)
    
    # Cell-type annotations on right
    prev_ct = None
    start = 0
    for i, ct in enumerate(row_celltype + [None]):
        if ct != prev_ct and prev_ct is not None:
            mid = (start + i - 1) / 2
            ax.text(len(sample_order) + 0.5, mid,
                    CELLTYPE_LABELS[CELLTYPE_ORDER.index(prev_ct)].replace('\n', ' '),
                    fontsize=7, ha='left', va='center',
                    rotation=0, fontweight='bold',
                    color=CELLTYPE_COLORS[prev_ct])
            start = i
        if prev_ct is None:
            start = i
        prev_ct = ct
    
    ax.set_title('A. Gene-level signature markers (z-scored log₂-CPM)',
                 fontsize=10, loc='left', fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.08, aspect=30)
    cbar.set_label('z-score', fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def panel_b_trajectories(ax, scores, cohort):
    """Panel B: stage-resolved trajectories (z-scored per signature)."""
    # z-score each signature column
    scores_z = (scores - scores.mean(axis=0)) / (scores.std(axis=0) + 1e-9)
    
    for ct in CELLTYPE_ORDER:
        means = []
        sems = []
        for stg in STAGE_ORDER:
            smps = cohort[cohort['group'] == stg]['column_name'].tolist()
            vals = scores_z.loc[smps, ct].values
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        means = np.array(means)
        sems = np.array(sems)
        x = np.arange(len(STAGE_ORDER))
        ax.errorbar(x, means, yerr=sems, marker='o', markersize=7,
                     linewidth=1.8, capsize=3,
                     color=CELLTYPE_COLORS[ct],
                     label=CELLTYPE_LABELS[CELLTYPE_ORDER.index(ct)].replace('\n', ' '))
    
    ax.axhline(0, color='grey', linestyle='--', lw=0.6, alpha=0.6)
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_LABELS, fontsize=7.5)
    ax.set_ylabel('Signature score (z-score)', fontsize=9)
    ax.set_title('B. Stage-resolved signature trajectories',
                 fontsize=10, loc='left', fontweight='bold')
    ax.legend(fontsize=7, loc='upper left', frameon=False,
              ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def panel_c_distributions(ax, scores, cohort, anova):
    """Panel C: per-signature box+swarm with ANOVA."""
    # Get positions: 4 signatures × 6 stages = 24 boxes, grouped
    n_sig = len(CELLTYPE_ORDER)
    n_stg = len(STAGE_ORDER)
    group_width = 6.5  # space for each signature
    box_width = 0.7
    
    rng = np.random.default_rng(42)
    
    for si, ct in enumerate(CELLTYPE_ORDER):
        base_x = si * group_width
        all_data = []
        positions = []
        for sj, stg in enumerate(STAGE_ORDER):
            smps = cohort[cohort['group'] == stg]['column_name'].tolist()
            vals = scores.loc[smps, ct].values
            all_data.append(vals)
            positions.append(base_x + sj)
        
        bp = ax.boxplot(
            all_data, positions=positions, widths=box_width,
            patch_artist=True, showfliers=False,
            medianprops=dict(color='black', lw=1.0),
            boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6),
            capprops=dict(lw=0.6),
        )
        for patch, color in zip(bp['boxes'], STAGE_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        
        # Swarm
        for pi, (vals, pos) in enumerate(zip(all_data, positions)):
            if len(vals) == 0:
                continue
            jit = pos + rng.uniform(-0.2, 0.2, size=len(vals))
            ax.scatter(jit, vals, s=10, color=STAGE_COLORS[pi],
                        edgecolors='black', linewidths=0.3,
                        alpha=0.85, zorder=3)
        
        # Group label + F-stat
        anova_row = anova[anova['signature'] == ct].iloc[0]
        F_stat = anova_row['F']
        p_val = anova_row['adj.P.Val']
        if p_val < 1e-4:
            ptext = 'FDR<10⁻⁴'
        else:
            ptext = f'FDR={p_val:.1e}'
        label_text = CELLTYPE_LABELS[si].replace('\n', ' ')
        y_pos = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10
        ax.text(base_x + (n_stg-1)/2, y_pos,
                 f'{label_text}\nF={F_stat:.1f}, {ptext}',
                 ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    
    # X-ticks: show stage labels for each group
    all_positions = []
    all_labels = []
    for si in range(n_sig):
        for sj in range(n_stg):
            all_positions.append(si * group_width + sj)
            all_labels.append(STAGE_LABELS[sj])
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, fontsize=5.5, rotation=0)
    ax.set_ylabel('Signature score (median log₂-CPM)', fontsize=9)
    ax.set_title('C. Signature score distributions per stage',
                 fontsize=10, loc='left', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Make room on top for labels
    cur_ylim = ax.get_ylim()
    ax.set_ylim(cur_ylim[0], cur_ylim[1] + (cur_ylim[1] - cur_ylim[0]) * 0.15)


def panel_d_correlation_heatmap(ax, corr_r, corr_fdr):
    """Panel D: correlation heatmap (signatures × IL-10 axis genes)."""
    # Order columns
    cols_available = [g for g in GENE_ORDER_IL10 if g in corr_r.columns]
    R = corr_r.loc[CELLTYPE_ORDER, cols_available].values
    FDR = corr_fdr.loc[CELLTYPE_ORDER, cols_available].values
    
    im = ax.imshow(R, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Stars for significance
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            fdr = FDR[i, j]
            if fdr < 0.001:
                sym = '***'
            elif fdr < 0.01:
                sym = '**'
            elif fdr < 0.05:
                sym = '*'
            else:
                sym = ''
            if sym:
                ax.text(j, i, sym, ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='black' if abs(R[i, j]) < 0.6 else 'white')
    
    ax.set_xticks(range(len(cols_available)))
    ax.set_xticklabels(cols_available, fontsize=7.5,
                        rotation=45, ha='right')
    ax.set_yticks(range(len(CELLTYPE_ORDER)))
    ax.set_yticklabels([l.replace('\n', ' ') for l in CELLTYPE_LABELS],
                        fontsize=8)
    ax.set_title('D. Signature × IL-10 axis correlations (Pearson r)',
                 fontsize=10, loc='left', fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Pearson r', fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def main():
    print("Loading inputs...")
    log_cpm, cohort, sig_panel, scores, anova, corr_r, corr_fdr = load_inputs()
    print(f"  log_cpm: {log_cpm.shape}")
    print(f"  cohort: {len(cohort)} samples")
    print(f"  signature_panel: {len(sig_panel)} entries")
    
    # Align scores to cohort order
    scores = scores.loc[cohort['column_name']]
    
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(2, 2, figure=fig,
                   height_ratios=[1.4, 1.0],
                   width_ratios=[1.3, 1.0],
                   hspace=0.38, wspace=0.28,
                   left=0.08, right=0.95, top=0.94, bottom=0.06)
    
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    
    panel_a_gene_heatmap(ax_A, log_cpm, cohort, sig_panel)
    panel_b_trajectories(ax_B, scores, cohort)
    panel_c_distributions(ax_C, scores, cohort, anova)
    panel_d_correlation_heatmap(ax_D, corr_r, corr_fdr)
    
    plt.suptitle(
        'Figure 4. Cell-type signature remodeling across MASLD → HCC '
        'and its association with the IL-10 axis\n'
        '(n=40, GSE246221; signature-enrichment approach, NOT '
        'a cell-proportion deconvolution)',
        fontsize=11, fontweight='bold', y=0.99
    )
    
    out_png = FIG_DIR / 'figure4_celltype_signatures.png'
    out_pdf = FIG_DIR / 'figure4_celltype_signatures.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    main()
