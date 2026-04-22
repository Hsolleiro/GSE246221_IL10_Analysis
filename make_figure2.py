"""
Figure 2: Stage-resolved relative dynamics of the IL-10 signaling axis.

Panel A: Forest plot of log2FC (± 95% CI) for 11 IL-10 axis genes
         across six pairwise contrasts.
Panel B: Heatmap of logFC per contrast.
Panel C: F-test ranking of dynamic components.

Inputs:  results/deg/*.csv (from run_deg.py)
Outputs: results/figures/figure2_il10_dynamics.{png,pdf}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEG_DIR = REPO_ROOT / 'results' / 'deg'
FIG_DIR = REPO_ROOT / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Config: the 11 IL-10 axis genes and functional groups
# ----------------------------------------------------------------------------
IL10_AXIS = {
    'Il10':   'Ligand',
    'Il10ra': 'Receptor',
    'Il10rb': 'Receptor',
    'Stat3':  'Transducer',
    'Jak1':   'Transducer',
    'Jak2':   'Transducer',
    'Tyk2':   'Transducer',
    'Socs3':  'Feedback',
    'Il6st':  'Co-receptor',
    'Scd2':   'Effector',
    'Ddit4':  'Effector',
}

CONTRASTS = ['EarlyMASLD_vs_Control', 'MASH_vs_EarlyMASLD', 'Fibrosis_vs_MASH',
             'ChronicNT_vs_Fibrosis', 'HCC_vs_ChronicNT', 'HCC_vs_Control']

# ----------------------------------------------------------------------------
# Load & extract IL-10 axis per contrast
# ----------------------------------------------------------------------------
il10_data = {}
for c in CONTRASTS:
    df = pd.read_csv(DEG_DIR / f'{c}.csv')
    # Gene symbol is the last token after "_" in the "gene" column if IDs have
    # ENSMUSG prefixes; otherwise gene is the name directly.
    def extract_sym(name):
        return name.rsplit('_', 1)[-1] if '_' in name else name
    df['symbol'] = df['gene'].apply(extract_sym)
    df = df[df['symbol'].isin(IL10_AXIS.keys())]
    il10_data[c] = df.set_index('symbol')

# F-test
ftest = pd.read_csv(DEG_DIR / 'F_test_any_stage.csv')
ftest['symbol'] = ftest['gene'].apply(lambda g: g.rsplit('_', 1)[-1] if '_' in g else g)
ftest = ftest[ftest['symbol'].isin(IL10_AXIS.keys())].set_index('symbol')

# ----------------------------------------------------------------------------
# Figure layout
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.1],
              width_ratios=[1.6, 1.0], hspace=0.35, wspace=0.35,
              left=0.08, right=0.97, top=0.93, bottom=0.08)

ax_A = fig.add_subplot(gs[0, :])  # forest plot full width
ax_B = fig.add_subplot(gs[1, 0])  # heatmap
ax_C = fig.add_subplot(gs[1, 1])  # F-test bars

# ---- Panel A: Forest plot ----
gene_order = ['Il10', 'Il10ra', 'Il10rb', 'Stat3', 'Jak1', 'Jak2', 'Tyk2',
              'Socs3', 'Il6st', 'Scd2', 'Ddit4']
contrast_colors = plt.cm.tab10(np.linspace(0, 1, len(CONTRASTS)))
y_positions = np.arange(len(gene_order))
dy = 0.12

for c_idx, c in enumerate(CONTRASTS):
    df = il10_data[c]
    offsets = y_positions + (c_idx - len(CONTRASTS)/2) * dy
    logfcs, ci_lo, ci_hi, fdrs = [], [], [], []
    for g in gene_order:
        if g in df.index:
            row = df.loc[g]
            lfc = row['logFC']
            # 95% CI from t and p — approximate using standard error = logFC / t
            se = abs(lfc / row['t']) if row['t'] != 0 else np.nan
            ci = 1.96 * se
            logfcs.append(lfc)
            ci_lo.append(lfc - ci)
            ci_hi.append(lfc + ci)
            fdrs.append(row['adj.P.Val'])
        else:
            logfcs.append(np.nan); ci_lo.append(np.nan); ci_hi.append(np.nan)
            fdrs.append(1.0)
    logfcs = np.array(logfcs); ci_lo = np.array(ci_lo); ci_hi = np.array(ci_hi)
    # Filled = significant (FDR<0.05), open = not
    sig = np.array(fdrs) < 0.05
    ax_A.errorbar(logfcs[~sig], offsets[~sig],
                   xerr=[logfcs[~sig]-ci_lo[~sig], ci_hi[~sig]-logfcs[~sig]],
                   fmt='o', mfc='white', mec=contrast_colors[c_idx],
                   ecolor=contrast_colors[c_idx], markersize=5, capsize=0,
                   elinewidth=0.8)
    ax_A.errorbar(logfcs[sig], offsets[sig],
                   xerr=[logfcs[sig]-ci_lo[sig], ci_hi[sig]-logfcs[sig]],
                   fmt='o', mfc=contrast_colors[c_idx], mec=contrast_colors[c_idx],
                   ecolor=contrast_colors[c_idx], markersize=6, capsize=0,
                   elinewidth=1.0,
                   label=c.replace('_vs_', ' vs '))

ax_A.axvline(0, color='black', linestyle='--', lw=0.7, alpha=0.5)
ax_A.set_yticks(y_positions)
ax_A.set_yticklabels(gene_order, style='italic')
ax_A.set_xlabel('log₂ fold change (95% CI)')
ax_A.set_title('A — Stage-wise pairwise contrasts for the IL-10 signaling axis',
                loc='left', fontsize=11, fontweight='bold')
ax_A.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8,
             frameon=False)
ax_A.invert_yaxis()
ax_A.grid(axis='x', alpha=0.3)

# ---- Panel B: Heatmap of logFC ----
heatmap_data = pd.DataFrame(
    {c: [il10_data[c].loc[g, 'logFC'] if g in il10_data[c].index else np.nan
         for g in gene_order] for c in CONTRASTS},
    index=gene_order
)
sig_mat = pd.DataFrame(
    {c: [il10_data[c].loc[g, 'adj.P.Val'] if g in il10_data[c].index else 1.0
         for g in gene_order] for c in CONTRASTS},
    index=gene_order
)

vmax = np.nanmax(np.abs(heatmap_data.values))
im = ax_B.imshow(heatmap_data.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  aspect='auto')
ax_B.set_yticks(range(len(gene_order)))
ax_B.set_yticklabels(gene_order, style='italic', fontsize=9)
ax_B.set_xticks(range(len(CONTRASTS)))
ax_B.set_xticklabels([c.replace('_vs_', ' vs ') for c in CONTRASTS],
                     rotation=45, ha='right', fontsize=8)
ax_B.set_title('B — logFC heatmap with significance', loc='left',
                fontsize=11, fontweight='bold')

# Stars for significance
for i in range(len(gene_order)):
    for j in range(len(CONTRASTS)):
        fdr = sig_mat.iloc[i, j]
        if fdr < 0.001: star = '***'
        elif fdr < 0.01: star = '**'
        elif fdr < 0.05: star = '*'
        else: star = ''
        if star:
            ax_B.text(j, i, star, ha='center', va='center',
                       fontsize=8, color='black')

cbar = fig.colorbar(im, ax=ax_B, fraction=0.046, pad=0.04)
cbar.set_label('log₂ FC', fontsize=9)

# ---- Panel C: F-test ranking ----
ftest_sorted = ftest.reindex(gene_order).sort_values('F', ascending=True)
colors_c = ['#E64B35' if row['adj.P.Val'] < 0.05 else 'lightgray'
            for _, row in ftest_sorted.iterrows()]
ax_C.barh(range(len(ftest_sorted)), ftest_sorted['F'], color=colors_c,
           edgecolor='black', linewidth=0.5)
ax_C.set_yticks(range(len(ftest_sorted)))
ax_C.set_yticklabels(ftest_sorted.index, style='italic', fontsize=9)
ax_C.set_xlabel('F-statistic (any-stage difference)')
ax_C.set_title('C — Longitudinal F-test ranking', loc='left',
                fontsize=11, fontweight='bold')
ax_C.grid(axis='x', alpha=0.3)

# Title
fig.suptitle('Figure 2. Stage-resolved dynamics of the IL-10 signaling axis '
             'in a MASLD/MASH-to-HCC mouse model (GSE246221, n=40).',
             fontsize=12, fontweight='bold', y=0.98)

plt.savefig(FIG_DIR / 'figure2_il10_dynamics.png', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'figure2_il10_dynamics.pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR / 'figure2_il10_dynamics'}.{{png,pdf}}")
