"""
Figure 2 — Stage-resolved relative dynamics of the IL-10 signaling axis.

Narrative: Shows HOW each IL-10 pathway component changes across disease
progression in terms of log2 fold-change + confidence intervals + statistical
significance (limma-voom robust eBayes).

Multi-panel:
  A  Forest plot — log2FC ± 95% CI per gene across 6 stage-wise contrasts
  B  Heatmap summary — log2FC across contrasts, with FDR significance stars
  C  Longitudinal F-test — shows which genes vary across stages overall
     (ranked, with FDR-stratified coloring)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

# -------------------- Genes + contrasts --------------------
axis_genes = ['Il10', 'Il10ra', 'Il10rb',                  # ligand + receptors
              'Stat3', 'Jak1', 'Jak2', 'Tyk2',              # transducers
              'Socs3',                                       # feedback
              'Il6st',                                       # gp130 co-receptor
              'Scd2', 'Ddit4']                               # IL-10-responsive effectors

# Visual grouping for panel organization
gene_block = {
    'Ligand':     ['Il10'],
    'Receptors':  ['Il10ra', 'Il10rb'],
    'Transducers': ['Stat3', 'Jak1', 'Jak2', 'Tyk2'],
    'Feedback':   ['Socs3'],
    'Co-receptor': ['Il6st'],
    'Effectors (IL-10-responsive)': ['Scd2', 'Ddit4'],
}
block_colors = {
    'Ligand':                      '#777777',
    'Receptors':                   '#C44E52',
    'Transducers':                 '#4C72B0',
    'Feedback':                    '#F0C419',
    'Co-receptor':                 '#55A868',
    'Effectors (IL-10-responsive)':'#7570b3',
}

# Contrasts (6 stage-wise)
contrast_names = [
    'EarlyMASLD_vs_Control',
    'MASH_vs_EarlyMASLD',
    'Fibrosis_vs_MASH',
    'ChronicNT_vs_Fibrosis',
    'HCC_vs_ChronicNT',
    'HCC_vs_Control',
]
contrast_labels = {
    'EarlyMASLD_vs_Control': 'Early-MASLD\nvs Control',
    'MASH_vs_EarlyMASLD':    'MASH\nvs Early-MASLD',
    'Fibrosis_vs_MASH':      'Fibrosis\nvs MASH',
    'ChronicNT_vs_Fibrosis': 'Chronic-NT\nvs Fibrosis',
    'HCC_vs_ChronicNT':      'HCC\nvs Chronic-NT',
    'HCC_vs_Control':        'HCC\nvs Control',
}

# -------------------- Load results --------------------
# For each contrast, get logFC, t, P, adj.P, and compute SE from logFC/t
contrasts_data = {}
for c in contrast_names:
    df = pd.read_csv(f'/home/claude/deg/{c}.csv')
    df = df.copy()
    # Standard error = |logFC / t|; with moderated t, this is the shrunken SE
    df['SE'] = np.where(df['t'] != 0, np.abs(df['logFC'] / df['t']), np.nan)
    # 95% CI approx using t-distribution df = 40-6 = 34 (residual df)
    # Add prior df (~1500 after eBayes robust) → effectively normal approximation
    df['CI_low']  = df['logFC'] - 1.96 * df['SE']
    df['CI_high'] = df['logFC'] + 1.96 * df['SE']
    contrasts_data[c] = df

# Extract rows for axis genes across all contrasts
def find_gene_row(df, sym):
    hits = df[df['gene'].str.endswith('_' + sym)]
    return hits.iloc[0] if len(hits) > 0 else None

# Build a tidy table: long format (gene, contrast, logFC, SE, CI_low, CI_high, padj)
rows = []
for g in axis_genes:
    for c in contrast_names:
        r = find_gene_row(contrasts_data[c], g)
        if r is None:
            rows.append({'gene': g, 'contrast': c, 'logFC': np.nan, 'SE': np.nan,
                         'CI_low': np.nan, 'CI_high': np.nan, 'padj': np.nan, 'detected': False})
        else:
            rows.append({'gene': g, 'contrast': c,
                         'logFC': r['logFC'], 'SE': r['SE'],
                         'CI_low': r['CI_low'], 'CI_high': r['CI_high'],
                         'padj': r['adj.P.Val'], 'detected': True})
long_df = pd.DataFrame(rows)

# F-test: check whether gene varies across stages overall
ftest = pd.read_csv('/home/claude/deg/F_test_any_stage.csv')
ftest_dict = {}
for g in axis_genes:
    hit = ftest[ftest['gene'].str.endswith('_' + g)]
    if len(hit) > 0:
        ftest_dict[g] = {'F': hit.iloc[0]['F'], 'padj': hit.iloc[0]['adj.P.Val']}
    else:
        ftest_dict[g] = {'F': np.nan, 'padj': np.nan}

# Save long table
long_df.to_csv('/home/claude/signatures/IL10_axis_contrasts_long.csv', index=False)

# =========================================================================
# BUILD FIGURE
# =========================================================================
fig = plt.figure(figsize=(17, 13))
gs = GridSpec(
    nrows=2, ncols=2,
    height_ratios=[2.0, 1.1],
    width_ratios=[1.45, 1.15],
    figure=fig,
    hspace=0.45, wspace=0.30,
    left=0.10, right=0.97, top=0.92, bottom=0.08
)
ax_A = fig.add_subplot(gs[0, 0])   # forest plot
ax_B = fig.add_subplot(gs[0, 1])   # heatmap
ax_C = fig.add_subplot(gs[1, :])   # F-test ranking

# =========================================================================
# PANEL A — Forest plot (logFC ± 95% CI per gene across contrasts)
# =========================================================================
# Stack genes vertically, grouped by block. Within each gene, show 6 contrasts
# as separate rows with color by contrast direction.

# Gene ordering: group block by block
ordered_genes = []
gene_blocks_flat = []
for block, genes in gene_block.items():
    for g in genes:
        ordered_genes.append(g)
        gene_blocks_flat.append(block)

# Y positions: each gene gets 6 slots (one per contrast), plus a separator
y_positions = {}
y_cur = 0
gene_y_ranges = {}  # for block annotations
for g in ordered_genes:
    y_start = y_cur
    for i, c in enumerate(contrast_names):
        y_positions[(g, c)] = y_cur
        y_cur += 1
    gene_y_ranges[g] = (y_start, y_cur - 1)
    y_cur += 0.8   # gap between genes

# Colors per contrast (gradient to imply direction of disease progression)
import matplotlib.cm as cm
contrast_palette = {
    'EarlyMASLD_vs_Control':'#6FAAD7',
    'MASH_vs_EarlyMASLD':   '#88C298',
    'Fibrosis_vs_MASH':     '#E5B34A',
    'ChronicNT_vs_Fibrosis':'#D46A6A',
    'HCC_vs_ChronicNT':     '#8766B0',
    'HCC_vs_Control':       '#2C2C2C',
}

# Plot forest: for each (gene, contrast), draw horizontal error bar
for _, r in long_df.iterrows():
    if not r['detected'] or pd.isna(r['logFC']):
        continue
    y = y_positions[(r['gene'], r['contrast'])]
    color = contrast_palette[r['contrast']]
    # error bar
    ax_A.plot([r['CI_low'], r['CI_high']], [y, y], color=color, lw=1.2, zorder=2)
    # point
    sig_size = 50 if (pd.notna(r['padj']) and r['padj'] < 0.05) else 28
    face = color if (pd.notna(r['padj']) and r['padj'] < 0.05) else 'white'
    edge = color
    lw = 1.3 if (pd.notna(r['padj']) and r['padj'] < 0.05) else 1.0
    ax_A.scatter([r['logFC']], [y], s=sig_size, color=face, edgecolor=edge,
                 linewidths=lw, zorder=3)
    # significance stars next to significant points
    if pd.notna(r['padj']) and r['padj'] < 0.05:
        stars = '***' if r['padj'] < 0.001 else '**' if r['padj'] < 0.01 else '*'
        # place stars on the side opposite to the effect
        x_text = r['CI_high'] + 0.06 if r['logFC'] >= 0 else r['CI_low'] - 0.06
        ha = 'left' if r['logFC'] >= 0 else 'right'
        ax_A.text(x_text, y, stars, ha=ha, va='center', fontsize=7.5,
                  color=color, fontweight='bold')

# Y-tick labels: gene names (at center of each gene's block)
y_ticks = []
y_tick_labels = []
for g in ordered_genes:
    y_s, y_e = gene_y_ranges[g]
    y_ticks.append((y_s + y_e) / 2)
    y_tick_labels.append(g)
ax_A.set_yticks(y_ticks)
ax_A.set_yticklabels([f'$\\mathit{{{g}}}$' for g in y_tick_labels], fontsize=11)
# Color gene labels by block
for t, block in zip(ax_A.get_yticklabels(), gene_blocks_flat):
    t.set_color(block_colors[block])
    t.set_fontweight('bold')

# Vertical reference line at logFC=0
ax_A.axvline(0, color='black', lw=0.6, ls='-', zorder=1)

# Faint horizontal separators between gene blocks
block_boundaries = []
prev_block = None
for g, block in zip(ordered_genes, gene_blocks_flat):
    if prev_block is not None and block != prev_block:
        y_s, _ = gene_y_ranges[g]
        block_boundaries.append(y_s - 0.4)
    prev_block = block
for yb in block_boundaries:
    ax_A.axhline(yb, color='gray', lw=0.6, alpha=0.4, zorder=0)

ax_A.set_xlabel('log$_2$ fold change (95% CI)', fontsize=11)
ax_A.set_title('A — Forest plot: IL-10 axis genes across 6 stage-wise contrasts\n(filled = FDR<0.05; open = ns)',
               fontsize=12, loc='left', pad=8)
ax_A.invert_yaxis()

# Annotate Il10 (not detected / filtered out)
il10_y = (gene_y_ranges['Il10'][0] + gene_y_ranges['Il10'][1]) / 2
ax_A.text(0.02, il10_y,
          'below detection threshold\n(< 10 reads / 35M library across all samples)',
          ha='left', va='center', fontsize=8, fontstyle='italic', color='gray',
          transform=ax_A.get_yaxis_transform())

# Contrast legend at top
legend_elements = []
from matplotlib.lines import Line2D
for c in contrast_names:
    legend_elements.append(Line2D([0], [0], marker='o', color=contrast_palette[c],
                                  markersize=7, lw=0,
                                  label=contrast_labels[c].replace('\n', ' ')))
ax_A.legend(handles=legend_elements, loc='lower right', fontsize=8.5,
            ncol=1, frameon=True, framealpha=0.95, edgecolor='gray',
            bbox_to_anchor=(1.0, 0.01))

# Add block labels on far left
x_block_lab = ax_A.get_xlim()[0] - 0.18 * (ax_A.get_xlim()[1] - ax_A.get_xlim()[0])
seen_blocks = set()
for g, block in zip(ordered_genes, gene_blocks_flat):
    if block in seen_blocks: continue
    seen_blocks.add(block)
    # find y range for this block
    genes_in_block = [gg for gg, bb in zip(ordered_genes, gene_blocks_flat) if bb == block]
    y_starts = [gene_y_ranges[gg][0] for gg in genes_in_block]
    y_ends   = [gene_y_ranges[gg][1] for gg in genes_in_block]
    y_mid = (min(y_starts) + max(y_ends)) / 2
    ax_A.text(x_block_lab, y_mid, block, ha='right', va='center',
              fontsize=9, color=block_colors[block],
              fontweight='bold', rotation=0,
              clip_on=False)

ax_A.set_xlim(left=ax_A.get_xlim()[0])  # keep auto-computed right bound

# =========================================================================
# PANEL B — Heatmap of log2FC across contrasts × gene
# =========================================================================
# Matrix: rows = genes (ordered same as A), cols = contrasts
hm_mat = np.full((len(ordered_genes), len(contrast_names)), np.nan)
hm_padj = np.full((len(ordered_genes), len(contrast_names)), np.nan)
for i, g in enumerate(ordered_genes):
    for j, c in enumerate(contrast_names):
        r = long_df[(long_df['gene'] == g) & (long_df['contrast'] == c)]
        if len(r) > 0 and r.iloc[0]['detected']:
            hm_mat[i, j] = r.iloc[0]['logFC']
            hm_padj[i, j] = r.iloc[0]['padj']

vmax = np.nanmax(np.abs(hm_mat))
im = ax_B.imshow(hm_mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
# Annotations
for i in range(hm_mat.shape[0]):
    for j in range(hm_mat.shape[1]):
        lfc = hm_mat[i, j]; padj = hm_padj[i, j]
        if pd.isna(lfc):
            ax_B.text(j, i, 'NA', ha='center', va='center', fontsize=8, color='gray')
            continue
        stars = ''
        if pd.notna(padj):
            if padj < 0.001: stars = '***'
            elif padj < 0.01: stars = '**'
            elif padj < 0.05: stars = '*'
        color = 'white' if abs(lfc) > vmax * 0.55 else 'black'
        ax_B.text(j, i, f'{lfc:+.2f}{stars}', ha='center', va='center',
                  fontsize=8.5, color=color)
ax_B.set_xticks(range(len(contrast_names)))
ax_B.set_xticklabels([contrast_labels[c] for c in contrast_names], fontsize=8.5)
ax_B.set_yticks(range(len(ordered_genes)))
ax_B.set_yticklabels([f'$\\mathit{{{g}}}$' for g in ordered_genes], fontsize=10)
for t, block in zip(ax_B.get_yticklabels(), gene_blocks_flat):
    t.set_color(block_colors[block]); t.set_fontweight('bold')

# Block separators
cum = 0
prev_block = None
for g, block in zip(ordered_genes, gene_blocks_flat):
    if prev_block is not None and block != prev_block:
        ax_B.axhline(cum - 0.5, color='black', lw=0.7)
    cum += 1
    prev_block = block

cbar = fig.colorbar(im, ax=ax_B, shrink=0.7, pad=0.02)
cbar.set_label('log$_2$ fold change', fontsize=9)
ax_B.set_title('B — Summary heatmap of log$_2$FC\n(*FDR<0.05, **<0.01, ***<0.001)',
               fontsize=12, loc='left', pad=8)

# =========================================================================
# PANEL C — F-test ranking (which genes vary across stages overall)
# =========================================================================
# Horizontal bar: -log10(FDR) from F-test, colored by block
ftest_df = pd.DataFrame([
    {'gene': g, 'F': ftest_dict[g]['F'], 'padj': ftest_dict[g]['padj'],
     'block': b}
    for g, b in zip(ordered_genes, gene_blocks_flat)
])
ftest_df['minus_log10_fdr'] = -np.log10(ftest_df['padj'].clip(lower=1e-20))
ftest_df = ftest_df.sort_values('minus_log10_fdr', ascending=True)

bar_colors = [block_colors[b] for b in ftest_df['block']]
y_bar = np.arange(len(ftest_df))
ax_C.barh(y_bar, ftest_df['minus_log10_fdr'], color=bar_colors,
          edgecolor='black', linewidth=0.6, alpha=0.85)
# Threshold line at FDR=0.05
ax_C.axvline(-np.log10(0.05), color='red', ls='--', lw=1, label='FDR = 0.05')
ax_C.axvline(-np.log10(0.01), color='orange', ls=':', lw=1, label='FDR = 0.01')
ax_C.set_yticks(y_bar)
ax_C.set_yticklabels([f'$\\mathit{{{g}}}$' for g in ftest_df['gene']], fontsize=10.5)
for t, block in zip(ax_C.get_yticklabels(), ftest_df['block']):
    t.set_color(block_colors[block]); t.set_fontweight('bold')
ax_C.set_xlabel('$-$log$_{10}$(FDR) — F-test across all 6 stages', fontsize=10.5)
ax_C.set_title('C — Overall longitudinal significance per IL-10 axis gene '
                '(limma-voom F-test, any-stage difference)',
                fontsize=12, loc='left', pad=8)
ax_C.legend(loc='lower right', fontsize=9, framealpha=0.95)
ax_C.grid(axis='x', alpha=0.3)

# Annotate bars with F-statistic
for i, (_, row) in enumerate(ftest_df.iterrows()):
    if pd.notna(row['F']):
        ax_C.text(row['minus_log10_fdr'] + 0.1, i,
                  f"F={row['F']:.2f}",
                  va='center', ha='left', fontsize=8, color='black')

# Suptitle
fig.suptitle(
    'Figure 2. Stage-resolved relative dynamics of the IL-10 signaling axis '
    'in a MASLD/MASH-to-HCC mouse model (GSE246221, n=40)',
    fontsize=13.5, y=0.985
)

plt.savefig('/home/claude/figure2_il10_dynamics.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/claude/figure2_il10_dynamics.pdf', bbox_inches='tight')
plt.close()
print("Saved: /home/claude/figure2_il10_dynamics.png / .pdf")

print("\n=== F-test summary (sorted) ===")
print(ftest_df[['gene','block','F','padj']].round(4).to_string(index=False))

"""
Figure 3 — Stage-wise expression of IL-10 axis components in MASLD/MASH-to-HCC.

Shows absolute log2-CPM per gene per stage (box + swarm) with ANOVA F-test
overall significance and pairwise Dunn-Holm post-hoc brackets for the most
relevant contrasts.

Layout: grid of per-gene panels, grouped visually by functional class.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.stats import f_oneway, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# -------------------- Genes + organization --------------------
axis_genes = ['Il10', 'Il10ra', 'Il10rb',
              'Stat3', 'Jak1', 'Jak2', 'Tyk2',
              'Socs3', 'Il6st',
              'Scd2', 'Ddit4']

gene_block = {
    'Il10':    ('Ligand',     '#777777', 'ligand (below threshold)'),
    'Il10ra':  ('Receptors',  '#C44E52', 'IL-10 receptor α'),
    'Il10rb':  ('Receptors',  '#C44E52', 'IL-10 receptor β'),
    'Stat3':   ('Transducers','#4C72B0', 'transducer'),
    'Jak1':    ('Transducers','#4C72B0', 'transducer'),
    'Jak2':    ('Transducers','#4C72B0', 'transducer'),
    'Tyk2':    ('Transducers','#4C72B0', 'transducer'),
    'Socs3':   ('Feedback',   '#F0C419', 'negative feedback'),
    'Il6st':   ('Co-receptor','#55A868', 'gp130 co-receptor'),
    'Scd2':    ('Effectors',  '#7570b3', 'IL-10-responsive effector'),
    'Ddit4':   ('Effectors',  '#7570b3', 'IL-10-responsive effector'),
}

# -------------------- Load data --------------------
log_cpm = pd.read_csv('/home/claude/log_cpm_final.csv', index_col=0)
cohort = pd.read_csv('/home/claude/cohort_final.csv')
order_map = {c: i for i, c in enumerate(log_cpm.columns)}
cohort['_o'] = cohort['column_name'].map(order_map)
cohort = cohort.sort_values('_o').reset_index(drop=True).drop(columns='_o')
sample_to_group = dict(zip(cohort['column_name'], cohort['group']))

# Il10 is filtered out; load unfiltered log_cpm to show it in context
try:
    log_cpm_unfilt = pd.read_csv('/home/claude/log_cpm_unfiltered.csv', index_col=0)
except FileNotFoundError:
    # Regenerate from raw counts if not cached
    print("Generating unfiltered log2(CPM+1) for Il10 visualization...")
    counts = pd.read_csv('/home/claude/counts_final.csv', index_col=0)
    lib = counts.sum(axis=0)
    nf = pd.read_csv('/home/claude/tmm_factors.csv', index_col=0).iloc[:, 0]
    eff_lib = lib * nf.values
    log_cpm_unfilt = np.log2(((counts + 0.5).div(eff_lib + 1, axis=1)) * 1e6)
    log_cpm_unfilt.to_csv('/home/claude/log_cpm_unfiltered.csv')

def find_gene(sym, idx):
    hits = [g for g in idx if g.endswith('_' + sym)]
    return hits[0] if hits else None

# Stage setup
group_order = ['S1_Control_07w','S2a_EarlyMASLD_14w','S3_MASH_20w',
               'S4_Fibrosis_32w','S2b_ChronicNT_56w','S5_HCC']
group_xlab = {
    'S1_Control_07w':    'Control\n7w\n(n=5)',
    'S2a_EarlyMASLD_14w':'Early\nMASLD\n14w\n(n=5)',
    'S3_MASH_20w':       'MASH\n20w\n(n=5)',
    'S4_Fibrosis_32w':   'Fibrosis\n32w\n(n=5)',
    'S2b_ChronicNT_56w': 'Chronic\ninflam-NT\n56w\n(n=6)',
    'S5_HCC':            'HCC\n44-56w\n(n=14)',
}
colors_stage = {
    'S1_Control_07w':'#4C72B0','S2a_EarlyMASLD_14w':'#55A868','S3_MASH_20w':'#F0C419',
    'S4_Fibrosis_32w':'#C44E52','S2b_ChronicNT_56w':'#937860','S5_HCC':'#2C2C2C',
}
short_labels = ['Ctrl','Early','MASH','Fib','ChrNT','HCC']

# -------------------- Post-hoc testing function --------------------
def dunn_pairwise(vals_per_group):
    """Pairwise Mann-Whitney U with Holm-Bonferroni correction."""
    pairs = []
    for i, g1 in enumerate(group_order):
        for g2 in group_order[i+1:]:
            v1 = vals_per_group[g1]; v2 = vals_per_group[g2]
            if len(v1) < 2 or len(v2) < 2:
                pairs.append((g1, g2, np.nan, np.nan))
                continue
            try:
                U, p = mannwhitneyu(v1, v2, alternative='two-sided')
            except ValueError:
                p = 1.0
            pairs.append((g1, g2, p, np.mean(v2) - np.mean(v1)))
    pvals = [p for _, _, p, _ in pairs if not np.isnan(p)]
    if len(pvals) > 0:
        _, padj, _, _ = multipletests(pvals, method='holm')
        padj_iter = iter(padj)
        out = [(a, b, next(padj_iter) if not np.isnan(p) else np.nan, d) for a, b, p, d in pairs]
    else:
        out = pairs
    return {(a, b): p for a, b, p, _ in out}

# Pairwise contrasts we want to show significance brackets for
# (the most biologically informative transitions, ordered from early to late)
key_pairs = [
    ('S1_Control_07w',     'S5_HCC'),             # Control → HCC (overall disease)
    ('S2a_EarlyMASLD_14w', 'S5_HCC'),             # Early-MASLD → HCC
    ('S3_MASH_20w',        'S5_HCC'),             # MASH → HCC
    ('S4_Fibrosis_32w',    'S5_HCC'),             # Fibrosis → HCC
    ('S2b_ChronicNT_56w',  'S5_HCC'),             # ChronicNT → HCC
    ('S1_Control_07w',     'S4_Fibrosis_32w'),    # Control → Fibrosis (key transition)
    ('S3_MASH_20w',        'S4_Fibrosis_32w'),    # MASH → Fibrosis
    ('S1_Control_07w',     'S2b_ChronicNT_56w'),  # Control → ChronicNT
]

# =========================================================================
# Build figure — grid of 11 panels
# =========================================================================
# Grid: 3 rows x 4 cols, last slot unused (11 genes)
n_rows, n_cols = 3, 4
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(n_rows, n_cols, figure=fig,
              hspace=0.60, wspace=0.35,
              left=0.06, right=0.97, top=0.91, bottom=0.06)

axes = {}
for i, g in enumerate(axis_genes):
    r, c = divmod(i, n_cols)
    axes[g] = fig.add_subplot(gs[r, c])

# For each gene, plot and annotate
for g in axis_genes:
    ax = axes[g]
    block, color, subtitle = gene_block[g]

    # Data: Il10 uses unfiltered, others use filtered
    if g == 'Il10':
        row_idx = find_gene(g, log_cpm_unfilt.index)
        if row_idx is None:
            ax.axis('off'); continue
        vals = log_cpm_unfilt.loc[row_idx]
        used_unfiltered = True
    else:
        row_idx = find_gene(g, log_cpm.index)
        if row_idx is None:
            ax.axis('off'); continue
        vals = log_cpm.loc[row_idx]
        used_unfiltered = False

    # Group data
    data_per_group = {grp: vals.loc[[s for s in vals.index if sample_to_group[s] == grp]].values
                      for grp in group_order}

    # Box + swarm
    positions = np.arange(len(group_order))
    box_data = [data_per_group[grp] for grp in group_order]
    bp = ax.boxplot(box_data, positions=positions, widths=0.62,
                    patch_artist=True, showfliers=False,
                    medianprops={'color':'black','lw':1.4})
    for patch, grp in zip(bp['boxes'], group_order):
        patch.set_facecolor(colors_stage[grp])
        patch.set_alpha(0.32)
        patch.set_edgecolor(colors_stage[grp])

    # Individual points with jitter
    np.random.seed(0)
    for pos, grp in zip(positions, group_order):
        d = data_per_group[grp]
        xj = pos + np.random.uniform(-0.14, 0.14, size=len(d))
        ax.scatter(xj, d, c=colors_stage[grp], s=26, alpha=0.9,
                   edgecolors='white', linewidths=0.5, zorder=3)

    # ANOVA F-test across stages
    groups_for_anova = [d for d in box_data if len(d) > 1]
    if len(groups_for_anova) >= 2:
        F, p_anova = f_oneway(*groups_for_anova)
    else:
        F, p_anova = np.nan, np.nan

    # Post-hoc pairwise Dunn-Holm
    dunn = dunn_pairwise(data_per_group)

    # Significance brackets
    all_vals = np.concatenate([d for d in box_data if len(d) > 0])
    y_min, y_max = np.min(all_vals), np.max(all_vals)
    span = (y_max - y_min) if y_max > y_min else 1
    y_cur = y_max + 0.10 * span
    for (a, b) in key_pairs:
        p = dunn.get((a, b), np.nan)
        if pd.isna(p) or p >= 0.05: continue
        ia = group_order.index(a); ib = group_order.index(b)
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*'
        ax.plot([ia, ia, ib, ib],
                [y_cur, y_cur + 0.02*span, y_cur + 0.02*span, y_cur],
                lw=0.9, c='black')
        ax.text((ia+ib)/2, y_cur + 0.03*span, stars,
                ha='center', va='bottom', fontsize=8.5)
        y_cur += 0.14 * span
    ax.set_ylim(y_min - 0.05*span, y_cur + 0.08*span)

    # Axis styling
    ax.set_xticks(positions)
    ax.set_xticklabels(short_labels, fontsize=8.5)
    y_label = 'log$_2$(CPM+1)' if used_unfiltered else 'log$_2$-CPM'
    ax.set_ylabel(y_label, fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Title: gene + subtitle + ANOVA stat (italic gene name)
    if pd.isna(p_anova):
        title_p = ''
    else:
        title_p = f'\nANOVA F = {F:.1f}, P = {p_anova:.1e}' if p_anova < 0.05 else f'\nANOVA F = {F:.1f}, ns'
    ax.set_title(f'$\\mathit{{{g}}}$  ({subtitle}){title_p}',
                 fontsize=10.5, color=color, pad=5, fontweight='bold')

    # If Il10 uses unfiltered, annotate in-panel
    if used_unfiltered:
        ax.text(0.02, 0.98, 'Unfiltered\n(sub-threshold)',
                transform=ax.transAxes, va='top', ha='left', fontsize=7.5,
                style='italic', color='gray',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', lw=0.4))

# Turn off any unused axis cells
total_cells = n_rows * n_cols
for k in range(len(axis_genes), total_cells):
    r, c = divmod(k, n_cols)
    ax_empty = fig.add_subplot(gs[r, c])
    ax_empty.axis('off')
    # Add legend in empty cell
    if k == total_cells - 1:
        # Legend of stage colors
        ax_empty.text(0.05, 0.95, 'Stage legend:', ha='left', va='top',
                      fontsize=11, fontweight='bold')
        for i, grp in enumerate(group_order):
            y = 0.87 - i * 0.11
            ax_empty.scatter([0.09], [y], s=100, c=colors_stage[grp],
                             edgecolors='white', linewidths=0.5, transform=ax_empty.transAxes)
            ax_empty.text(0.18, y, group_xlab[grp].replace('\n', ' / '),
                          ha='left', va='center', fontsize=9,
                          color=colors_stage[grp], fontweight='bold',
                          transform=ax_empty.transAxes)

        ax_empty.text(0.05, 0.22,
                      'Pairwise post-hoc:\nMann-Whitney +\nHolm-Bonferroni\n\n* FDR<0.05\n** FDR<0.01\n*** FDR<0.001',
                      ha='left', va='top', fontsize=8.5, transform=ax_empty.transAxes,
                      bbox=dict(boxstyle='round,pad=0.4', fc='#f7f7f7', ec='gray', lw=0.5))

# Suptitle
fig.suptitle(
    'Figure 3. Stage-wise expression of IL-10 axis components in a MASLD/MASH-to-HCC mouse model '
    '(GSE246221, n=40)',
    fontsize=13.5, y=0.96
)

plt.savefig('/home/claude/figure3_il10_expression.png', dpi=300)
plt.savefig('/home/claude/figure3_il10_expression.pdf')
plt.close()
print("Saved: /home/claude/figure3_il10_expression.png / .pdf")

"""
Figure 4 — Four-cell-type compartment dynamics in MASLD→HCC and
association with the IL-10 signaling axis (GSE246221, n=40).

Panels:
  A  Gene-level heatmap (samples × signature genes, z-scored)
  B  Stage trajectories: mean ± SEM line plots for 4 populations
  C  Box+swarm per population × stage (Holm-adjusted Mann-Whitney)
  D  Correlation heatmap: 4 signature scores × IL-10 axis transcripts
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import pickle
import warnings
warnings.filterwarnings('ignore')

# -------------------- Load --------------------
log_cpm = pd.read_csv('/home/claude/log_cpm_final.csv', index_col=0)
cohort = pd.read_csv('/home/claude/cohort_final.csv')
order_map = {c: i for i, c in enumerate(log_cpm.columns)}
cohort['_o'] = cohort['column_name'].map(order_map)
cohort = cohort.sort_values('_o').reset_index(drop=True).drop(columns='_o')
sample_to_group = dict(zip(cohort['column_name'], cohort['group']))

with open('/home/claude/celltypes4/blob.pkl', 'rb') as f:
    blob = pickle.load(f)
signatures = blob['signatures']
coverage   = blob['coverage']
dunn_res   = blob['dunn_results']
stats_df   = blob['stats_df']
corr_r     = blob['corr_r']
corr_fdr   = blob['corr_fdr']
scores_med = pd.read_csv('/home/claude/celltypes4/scores_median.csv', index_col=0)

# -------------------- Style --------------------
group_order = ['S1_Control_07w','S2a_EarlyMASLD_14w','S3_MASH_20w',
               'S4_Fibrosis_32w','S2b_ChronicNT_56w','S5_HCC']
group_short = {
    'S1_Control_07w':'Control 7w','S2a_EarlyMASLD_14w':'Early-MASLD 14w',
    'S3_MASH_20w':'MASH 20w','S4_Fibrosis_32w':'Fibrosis 32w',
    'S2b_ChronicNT_56w':'Chron. inflam-NT 56w','S5_HCC':'HCC 44-56w',
}
group_xlab = {
    'S1_Control_07w':'Control\n(7w, n=5)','S2a_EarlyMASLD_14w':'Early\nMASLD\n(14w, n=5)',
    'S3_MASH_20w':'MASH\n(20w, n=5)','S4_Fibrosis_32w':'Fibrosis\n(32w, n=5)',
    'S2b_ChronicNT_56w':'Chronic\ninflam-NT\n(56w, n=6)','S5_HCC':'HCC\n(44-56w, n=14)',
}
colors_stage = {
    'S1_Control_07w':'#4C72B0','S2a_EarlyMASLD_14w':'#55A868','S3_MASH_20w':'#F0C419',
    'S4_Fibrosis_32w':'#C44E52','S2b_ChronicNT_56w':'#937860','S5_HCC':'#2C2C2C',
}

# Population order: parenchymal → stromal → innate lymphoid → myeloid
pop_order = ['Hepatocytes', 'HSC_Fibrosis', 'NK_cells', 'Macrophages_Monocytes']
pop_labels = {
    'Hepatocytes':            'Hepatocytes\n(parenchymal)',
    'HSC_Fibrosis':           'Hepatic stellate\ncells / Fibrosis',
    'NK_cells':               'NK cells\n(innate lymphoid)',
    'Macrophages_Monocytes':  'Macrophages &\nMonocytes',
}
pop_colors = {
    'Hepatocytes':            '#2a9d8f',  # teal-green (parenchymal)
    'HSC_Fibrosis':           '#e76f51',  # burnt orange (stromal/fibrotic)
    'NK_cells':               '#457b9d',  # blue (innate lymphoid)
    'Macrophages_Monocytes':  '#7570b3',  # purple (myeloid)
}

def find_gene(sym, idx):
    h = [g for g in idx if g.endswith('_' + sym)]
    return h[0] if h else None

# Sample ordering: within each stage, order by PC1 of the 4 scores
from sklearn.decomposition import PCA
sample_order = []
pc1 = pd.Series(PCA(n_components=1).fit_transform(scores_med[pop_order].fillna(0).values)[:,0],
                index=scores_med.index)
for g in group_order:
    ing = [s for s in pc1.index if sample_to_group[s] == g]
    ing.sort(key=lambda s: pc1[s])
    sample_order.extend(ing)

# =========================================================================
# Build the figure
# =========================================================================
fig = plt.figure(figsize=(19, 20))
gs = GridSpec(
    nrows=4, ncols=2,
    height_ratios=[2.6, 0.85, 1.15, 1.00],
    width_ratios=[2.4, 1.0],
    figure=fig,
    hspace=0.55, wspace=0.30,
    left=0.10, right=0.79, top=0.965, bottom=0.04
)
ax_A = fig.add_subplot(gs[0, :])
ax_B = fig.add_subplot(gs[1, :])
ax_C_base = gs[2, :].subgridspec(1, 4, wspace=0.45)
axes_C = [fig.add_subplot(ax_C_base[0, i]) for i in range(4)]
ax_D = fig.add_subplot(gs[3, 0])
ax_D_leg = fig.add_subplot(gs[3, 1])
ax_D_leg.axis('off')

# =========================================================================
# PANEL A — gene-level heatmap
# =========================================================================
gene_rows = []
for pop in pop_order:
    for g in signatures[pop]['genes']:
        full = find_gene(g, log_cpm.index)
        gene_rows.append((pop, g, full))

mat, row_labels, row_pop = [], [], []
for pop, sym, full in gene_rows:
    if full is None:
        mat.append(np.full(len(sample_order), np.nan))
    else:
        vals = log_cpm.loc[full, sample_order].values
        z = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-9)
        mat.append(z)
    row_labels.append(sym); row_pop.append(pop)
mat = np.array(mat)

vmax = np.nanpercentile(np.abs(mat), 98)
im = ax_A.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')

sample_labels_short = [s[7:] if s.startswith('Batch1_') else s for s in sample_order]
ax_A.set_xticks(range(len(sample_order)))
ax_A.set_xticklabels(sample_labels_short, rotation=90, fontsize=6.5)
for t, s in zip(ax_A.get_xticklabels(), sample_order):
    t.set_color(colors_stage[sample_to_group[s]])

ax_A.set_yticks(range(len(row_labels)))
ax_A.set_yticklabels([f'$\\mathit{{{g}}}$' for g in row_labels], fontsize=9)
for t, pop in zip(ax_A.get_yticklabels(), row_pop):
    t.set_color(pop_colors[pop])

# Horizontal separators + right-side labels
cum = 0
for pop in pop_order:
    n = sum(1 for p in row_pop if p == pop)
    if cum > 0:
        ax_A.axhline(cum - 0.5, color='black', linewidth=1.0)
    y_mid = cum + n/2 - 0.5
    ax_A.text(len(sample_order) + 1.2, y_mid, pop_labels[pop],
              ha='left', va='center', fontsize=10, color=pop_colors[pop],
              fontweight='bold', transform=ax_A.transData, clip_on=False)
    cum += n

# Vertical separators between stages
cum = 0
for g in group_order:
    n = sum(1 for s in sample_order if sample_to_group[s] == g)
    if cum > 0:
        ax_A.axvline(cum - 0.5, color='white', linewidth=1.8)
    cum += n

ax_A.set_title('A — Gene-level expression of signature genes across the cohort '
               '(row-wise z-scored log$_2$-CPM)',
               fontsize=12.5, loc='left', pad=14)

# Stage band below
n_rows = len(row_labels)
cum = 0
for g in group_order:
    n = sum(1 for s in sample_order if sample_to_group[s] == g)
    ax_A.plot([cum - 0.4, cum + n - 0.6], [n_rows + 8.0, n_rows + 8.0],
              color=colors_stage[g], lw=5, solid_capstyle='butt', clip_on=False)
    ax_A.annotate(group_short[g], xy=(cum + n/2 - 0.5, n_rows + 7.3),
                  ha='center', va='bottom', fontsize=9,
                  color=colors_stage[g], fontweight='bold', annotation_clip=False)
    cum += n

# Colorbar on the far right
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cax = inset_axes(ax_A, width='1.2%', height='60%', loc='center right',
                 bbox_to_anchor=(1.22, 0, 1, 1), bbox_transform=ax_A.transAxes, borderpad=0)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('z-score per gene', fontsize=9)
cbar.ax.tick_params(labelsize=8)

# =========================================================================
# PANEL B — trajectories
# =========================================================================
x_pos = np.arange(len(group_order))
for pop in pop_order:
    means, sems = [], []
    for g in group_order:
        v = scores_med.loc[[s for s in scores_med.index if sample_to_group[s] == g], pop].dropna().values
        means.append(np.mean(v))
        sems.append(np.std(v, ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0)
    means, sems = np.array(means), np.array(sems)
    # For display: z-score each pop's means to show trajectories on comparable scale
    m_std = (means - means.mean()) / (means.std() if means.std() > 0 else 1)
    s_std = sems / (means.std() if means.std() > 0 else 1)
    ax_B.plot(x_pos, m_std, marker='o', ms=9, lw=2.2, c=pop_colors[pop],
              label=pop_labels[pop].replace('\n', ' '), zorder=3)
    ax_B.fill_between(x_pos, m_std - s_std, m_std + s_std,
                       color=pop_colors[pop], alpha=0.15, zorder=2)
ax_B.set_xticks(x_pos)
ax_B.set_xticklabels([group_xlab[g] for g in group_order], fontsize=9)
ax_B.set_ylabel('Standardized signature score\n(z-score across stages)', fontsize=10)
ax_B.set_title('B — Stage-resolved compartment dynamics (mean ± SEM, per-population z-scored for display)',
                fontsize=12.5, loc='left', pad=8)
ax_B.grid(axis='y', alpha=0.3)
ax_B.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)
ax_B.axhline(0, color='gray', lw=0.5, ls='--')

# =========================================================================
# PANEL C — box+swarm per signature (4 axes)
# =========================================================================
short_xlabs = ['Ctrl','Early','MASH','Fib','ChrNT','HCC']

key_pairs = [
    ('S1_Control_07w', 'S5_HCC'),
    ('S2a_EarlyMASLD_14w', 'S5_HCC'),
    ('S3_MASH_20w', 'S5_HCC'),
    ('S1_Control_07w', 'S4_Fibrosis_32w'),
    ('S3_MASH_20w', 'S4_Fibrosis_32w'),
    ('S4_Fibrosis_32w', 'S5_HCC'),
]

for ax, pop in zip(axes_C, pop_order):
    data = [scores_med.loc[[s for s in scores_med.index if sample_to_group[s] == g], pop].dropna().values
            for g in group_order]
    bp = ax.boxplot(data, positions=np.arange(len(group_order)), widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops={'color':'black','lw':1.4})
    for patch, g in zip(bp['boxes'], group_order):
        patch.set_facecolor(colors_stage[g]); patch.set_alpha(0.32)
        patch.set_edgecolor(colors_stage[g])
    np.random.seed(0)
    for i, (g, d) in enumerate(zip(group_order, data)):
        xj = i + np.random.uniform(-0.14, 0.14, size=len(d))
        ax.scatter(xj, d, c=colors_stage[g], s=24, alpha=0.9,
                   edgecolors='white', linewidths=0.5, zorder=3)
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(short_xlabs, fontsize=8.5)
    ax.set_title(pop_labels[pop], fontsize=10, color=pop_colors[pop], pad=6, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # ANOVA stat in corner
    srow = stats_df[stats_df['signature'] == pop].iloc[0]
    ax.text(0.02, 0.98, f"F = {srow['F']:.1f}\nFDR = {srow['FDR_ANOVA']:.1e}",
            transform=ax.transAxes, ha='left', va='top',
            fontsize=7.5, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', lw=0.4))

    # Significance brackets
    all_vals = np.concatenate([d for d in data if len(d) > 0])
    y_max = np.max(all_vals); y_min = np.min(all_vals)
    span = y_max - y_min if y_max > y_min else 1
    y_cur = y_max + 0.10 * span
    dunn = dunn_res[pop]
    for (a, b) in key_pairs:
        p = dunn.get((a, b), np.nan)
        if pd.isna(p) or p >= 0.05: continue
        ia = group_order.index(a); ib = group_order.index(b)
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*'
        ax.plot([ia, ia, ib, ib],
                [y_cur, y_cur + 0.02*span, y_cur + 0.02*span, y_cur],
                lw=0.9, c='black')
        ax.text((ia+ib)/2, y_cur + 0.03*span, stars, ha='center', va='bottom', fontsize=8.5)
        y_cur += 0.14 * span
    ax.set_ylim(y_min - 0.05*span, y_cur + 0.08*span)

axes_C[0].set_ylabel('Signature score (median log$_2$-CPM)', fontsize=9.5)
axes_C[0].annotate(
    'C — Per-signature distribution across stages  '
    '(Holm-adjusted Mann-Whitney: * FDR<0.05, ** <0.01, *** <0.001)',
    xy=(0, 1.22), xycoords='axes fraction',
    ha='left', va='bottom', fontsize=12.5, annotation_clip=False
)

# =========================================================================
# PANEL D — Correlation heatmap
# =========================================================================
il10_genes = ['Il10','Il10ra','Il10rb','Stat3','Jak1','Jak2','Socs3','Il6st','Scd2','Ddit4']
cm_data = corr_r.loc[pop_order, il10_genes].values.astype(float)
vmax_c = 1.0
im_D = ax_D.imshow(cm_data, cmap='RdBu_r', vmin=-vmax_c, vmax=vmax_c, aspect='auto')
for i in range(cm_data.shape[0]):
    for j in range(cm_data.shape[1]):
        r = cm_data[i, j]
        p = corr_fdr.loc[pop_order[i], il10_genes[j]]
        if pd.isna(r):
            ax_D.text(j, i, 'NA', ha='center', va='center', fontsize=8, color='gray'); continue
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        color = 'white' if abs(r) > 0.6 else 'black'
        ax_D.text(j, i, f'{r:+.2f}{stars}', ha='center', va='center',
                   fontsize=9, color=color)
ax_D.set_xticks(range(len(il10_genes)))
ax_D.set_xticklabels([f'$\\mathit{{{g}}}$' for g in il10_genes], fontsize=10.5)
ax_D.set_yticks(range(len(pop_order)))
ax_D.set_yticklabels([pop_labels[p] for p in pop_order], fontsize=10)
for t, pop in zip(ax_D.get_yticklabels(), pop_order):
    t.set_color(pop_colors[pop])
cbar_D = fig.colorbar(im_D, ax=ax_D, shrink=0.9, aspect=14, pad=0.02)
cbar_D.set_label('Pearson r', fontsize=9)
ax_D.set_title('D — Correlation of cell-type signatures with IL-10 axis transcripts  '
                '(stars: FDR-adjusted; *<0.05, **<0.01, ***<0.001)',
                fontsize=11.5, loc='left', pad=8)

# Interpretation legend
interp = (
    "Interpretation\n"
    "━━━━━━━━━━━━━━━━━━━━\n"
    "HEPATOCYTES\n"
    "decline progressively with disease;\n"
    "NEGATIVELY correlate with Il10ra\n"
    "(−0.51), Il10rb (−0.72), Scd2 (−0.61):\n"
    "IL-10 axis enrichment in bulk reflects\n"
    "loss of parenchymal compartment.\n\n"
    "HSC / FIBROSIS\n"
    "peak in Fibrosis 32w; POSITIVELY\n"
    "correlate with IL-10 receptors —\n"
    "HSC are IL-10 targets (anti-fibrotic).\n\n"
    "NK CELLS\n"
    "activated from Fibrosis onward;\n"
    "POSITIVE correlation with Il10ra/\n"
    "Il10rb (+0.68 each) and Scd2 (+0.49)\n"
    "— IL-10-producing NK compartment.\n\n"
    "MACROPHAGES & MONOCYTES\n"
    "expand progressively; correlate with\n"
    "Il10ra (+0.58), Jak1 (+0.44), Il6st\n"
    "(+0.53); Socs3 is negatively corr.\n"
    "(−0.38) — feedback decoupling.\n\n"
    "━━━━━━━━━━━━━━━━━━━━\n"
    "Signatures from CellMarker 2.0\n"
    "(Hu 2023 NAR) + literature canonical\n"
    "(Halpern 2017, Remmerie 2020,\n"
    "Mederacke 2013, Grant 2008)."
)
ax_D_leg.text(0.0, 1.0, interp, ha='left', va='top', fontsize=8.5, family='sans-serif',
              bbox=dict(boxstyle='round,pad=0.6', fc='#f7f7f7', ec='#cccccc', lw=0.5))

# Suptitle
fig.suptitle(
    'Figure 4. Four-cell-type compartment remodeling across MASLD→HCC and its association '
    'with the IL-10 signaling axis (GSE246221, n=40, mouse liver bulk RNA-seq).',
    fontsize=13.5, y=0.995
)

plt.savefig('/home/claude/figure4_4celltypes.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/claude/figure4_4celltypes.pdf', bbox_inches='tight')
plt.close()
print("Saved: /home/claude/figure4_4celltypes.png / .pdf")

"""
Figure S1 — QC and transcriptomic architecture of the n=40 STZ+HFD cohort.
Supplementary figure showing overall structure of the dataset.

Panels:
  A  PCA PC1 vs PC2 (main axis of disease progression)
  B  PCA PC2 vs PC3 (HCC heterogeneity)
  C  Sample-to-sample Euclidean distance heatmap (Ward-ordered)
  D  Library size distribution per stage
  E  Biological marker validation (Afp/Alb/Col1a1) per stage
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import warnings
warnings.filterwarnings('ignore')

log_cpm = pd.read_csv('/home/claude/log_cpm_final.csv', index_col=0)
counts = pd.read_csv('/home/claude/counts_final.csv', index_col=0)
cohort = pd.read_csv('/home/claude/cohort_final.csv')
order_map = {c: i for i, c in enumerate(log_cpm.columns)}
cohort['_o'] = cohort['column_name'].map(order_map)
cohort = cohort.sort_values('_o').reset_index(drop=True).drop(columns='_o')
sample_to_group = dict(zip(cohort['column_name'], cohort['group']))

group_order = ['S1_Control_07w','S2a_EarlyMASLD_14w','S3_MASH_20w',
               'S4_Fibrosis_32w','S2b_ChronicNT_56w','S5_HCC']
group_labels = ['Control (7w)','Early MASLD (14w)','MASH (20w)',
                'Fibrosis (32w)','Chronic inflam-NT (56w)','HCC (44–56w)']
colors_stage = {
    'S1_Control_07w':'#4C72B0','S2a_EarlyMASLD_14w':'#55A868','S3_MASH_20w':'#F0C419',
    'S4_Fibrosis_32w':'#C44E52','S2b_ChronicNT_56w':'#937860','S5_HCC':'#2C2C2C',
}

# PCA on top 2000 variable genes
gv = log_cpm.var(axis=1)
top = gv.nlargest(2000).index
X = log_cpm.loc[top].T.values
Xc = X - X.mean(axis=0, keepdims=True)
pca = PCA(n_components=6)
scores = pca.fit_transform(Xc)
ve = pca.explained_variance_ratio_ * 100

# Distance matrix + hierarchical clustering
D = squareform(pdist(X, metric='euclidean'))
lnk = linkage(pdist(X, metric='euclidean'), method='ward')
order_idx = leaves_list(lnk)
D_ord = D[np.ix_(order_idx, order_idx)]
labels_ord = [cohort['column_name'].iloc[i].replace('Batch1_','') for i in order_idx]
group_ord = [cohort['group'].iloc[i] for i in order_idx]

# Figure
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 4, figure=fig,
              width_ratios=[1, 1, 1, 1],
              height_ratios=[1, 1.2, 1],
              hspace=0.45, wspace=0.35,
              left=0.07, right=0.97, top=0.93, bottom=0.06)

# Panel A: PC1 vs PC2
ax_A = fig.add_subplot(gs[0, 0:2])
for g, gl in zip(group_order, group_labels):
    idx = np.where(cohort['group'].values == g)[0]
    if len(idx) == 0: continue
    ax_A.scatter(scores[idx, 0], scores[idx, 1], c=colors_stage[g], s=90, alpha=0.9,
                 edgecolors='white', linewidths=1, label=f"{gl} (n={len(idx)})")
    ax_A.scatter(scores[idx, 0].mean(), scores[idx, 1].mean(),
                 c=colors_stage[g], marker='*', s=320, edgecolors='black', linewidths=1.2)
ax_A.set_xlabel(f"PC1 ({ve[0]:.1f}%)", fontsize=11)
ax_A.set_ylabel(f"PC2 ({ve[1]:.1f}%)", fontsize=11)
ax_A.axhline(0, color='gray', lw=0.5); ax_A.axvline(0, color='gray', lw=0.5)
ax_A.grid(alpha=0.3)
ax_A.set_title('A — PCA: principal axis of disease progression',
                fontsize=11.5, loc='left', pad=6)
ax_A.legend(loc='upper left', fontsize=8, framealpha=0.95, edgecolor='gray')

# Panel B: PC2 vs PC3
ax_B = fig.add_subplot(gs[0, 2:4])
for g, gl in zip(group_order, group_labels):
    idx = np.where(cohort['group'].values == g)[0]
    if len(idx) == 0: continue
    ax_B.scatter(scores[idx, 1], scores[idx, 2], c=colors_stage[g], s=90, alpha=0.9,
                 edgecolors='white', linewidths=1)
    ax_B.scatter(scores[idx, 1].mean(), scores[idx, 2].mean(),
                 c=colors_stage[g], marker='*', s=320, edgecolors='black', linewidths=1.2)
ax_B.set_xlabel(f"PC2 ({ve[1]:.1f}%)", fontsize=11)
ax_B.set_ylabel(f"PC3 ({ve[2]:.1f}%)", fontsize=11)
ax_B.axhline(0, color='gray', lw=0.5); ax_B.axvline(0, color='gray', lw=0.5)
ax_B.grid(alpha=0.3)
ax_B.set_title('B — PCA: HCC intra-tumoral heterogeneity (PC3)',
                fontsize=11.5, loc='left', pad=6)

# Panel C: Distance heatmap
ax_C = fig.add_subplot(gs[1, :])
im = ax_C.imshow(D_ord, cmap='viridis_r', aspect='auto')
ax_C.set_xticks(range(len(labels_ord)))
ax_C.set_yticks(range(len(labels_ord)))
ax_C.set_xticklabels(labels_ord, rotation=90, fontsize=6.5)
ax_C.set_yticklabels(labels_ord, fontsize=6.5)
for t, g in zip(ax_C.get_yticklabels(), group_ord):
    t.set_color(colors_stage[g])
for t, g in zip(ax_C.get_xticklabels(), group_ord):
    t.set_color(colors_stage[g])
cbar = fig.colorbar(im, ax=ax_C, fraction=0.014, pad=0.01)
cbar.set_label('Euclidean distance (log$_2$-CPM space)', fontsize=9)
ax_C.set_title('C — Sample-to-sample distance (Ward-ordered, top 2000 variable genes)',
                fontsize=11.5, loc='left', pad=6)

# Panel D: Library size per stage
ax_D = fig.add_subplot(gs[2, 0])
lib_size = counts.sum(axis=0) / 1e6
lib_by_group = [lib_size.loc[[s for s in lib_size.index if sample_to_group[s] == g]].values
                for g in group_order]
bp = ax_D.boxplot(lib_by_group, positions=np.arange(len(group_order)), widths=0.6,
                  patch_artist=True, showfliers=False,
                  medianprops={'color':'black','lw':1.2})
for patch, g in zip(bp['boxes'], group_order):
    patch.set_facecolor(colors_stage[g]); patch.set_alpha(0.35)
np.random.seed(0)
for i, (g, d) in enumerate(zip(group_order, lib_by_group)):
    xj = i + np.random.uniform(-0.12, 0.12, size=len(d))
    ax_D.scatter(xj, d, c=colors_stage[g], s=22, edgecolors='white', lw=0.4)
ax_D.set_xticks(range(len(group_order)))
ax_D.set_xticklabels(['Ctrl','Early','MASH','Fib','ChrNT','HCC'], fontsize=8.5)
ax_D.set_ylabel('Library size (millions)', fontsize=10)
ax_D.set_title('D — Library sizes', fontsize=11.5, loc='left', pad=6)
ax_D.grid(axis='y', alpha=0.3)

# Panel E, F, G: Biological marker validation
def find_gene(sym, idx):
    hits = [g for g in idx if g.endswith('_' + sym)]
    return hits[0] if hits else None

marker_panels = [
    ('Afp',    'Hepatoblast / HCC'),
    ('Alb',    'Hepatocyte function'),
    ('Col1a1', 'Fibrosis'),
]
short_labels = ['Ctrl','Early','MASH','Fib','ChrNT','HCC']
for i, (sym, tag) in enumerate(marker_panels):
    ax = fig.add_subplot(gs[2, i+1])
    row = find_gene(sym, log_cpm.index)
    if row is None: ax.axis('off'); continue
    vals = log_cpm.loc[row]
    d_by_g = [vals.loc[[s for s in vals.index if sample_to_group[s] == g]].values
              for g in group_order]
    bp = ax.boxplot(d_by_g, positions=np.arange(len(group_order)), widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops={'color':'black','lw':1.2})
    for patch, g in zip(bp['boxes'], group_order):
        patch.set_facecolor(colors_stage[g]); patch.set_alpha(0.35)
    np.random.seed(0)
    for j, (g, d) in enumerate(zip(group_order, d_by_g)):
        xj = j + np.random.uniform(-0.12, 0.12, size=len(d))
        ax.scatter(xj, d, c=colors_stage[g], s=22, edgecolors='white', lw=0.4)
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(short_labels, fontsize=8.5)
    ax.set_ylabel('log$_2$-CPM', fontsize=10)
    ax.set_title(f'$\\mathit{{{sym}}}$  —  {tag}', fontsize=10.5, loc='left', pad=6)
    ax.grid(axis='y', alpha=0.3)

# Overall title
fig.suptitle(
    'Figure S1. Quality control and transcriptomic architecture of the STZ+HFD cohort (GSE246221, n=40)',
    fontsize=13, y=0.965
)

plt.savefig('/home/claude/figureS1_QC.png', dpi=180)
plt.savefig('/home/claude/figureS1_QC.pdf')
plt.close()
print("Saved: /home/claude/figureS1_QC.png / .pdf")
print(f"PC1 {ve[0]:.1f}%  PC2 {ve[1]:.1f}%  PC3 {ve[2]:.1f}%")
