"""
Cell-type signature enrichment on the n=40 cohort.

Computes per-sample median log2-CPM scores across four hepatic compartments:
  1. Hepatocytes (parenchymal)
  2. Hepatic Stellate Cells (HSC / fibrosis)
  3. NK cells (innate lymphoid)
  4. Macrophages & Monocytes

Also computes Pearson correlations between each compartment score and the
IL-10 axis transcripts.

Inputs:  data/log_cpm_final.csv, data/cohort_final.csv, data/signature_panel.csv
Outputs: results/celltypes/{scores_median.csv, scores_zscore.csv,
         anova_per_signature.csv, il10_correlations_{r,p,fdr}.csv}
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / 'data'
RESULTS_DIR = REPO_ROOT / 'results'
OUT_DIR = RESULTS_DIR / 'celltypes'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
log_cpm = pd.read_csv(DATA_DIR / 'log_cpm_final.csv', index_col=0)
cohort = pd.read_csv(DATA_DIR / 'cohort_final.csv')
panel = pd.read_csv(DATA_DIR / 'signature_panel.csv')

print(f"log_cpm: {log_cpm.shape}")
print(f"cohort: {len(cohort)} samples")
print(f"signature panel: {len(panel)} marker genes across "
      f"{panel['signature'].nunique()} cell types")

order_map = {c: i for i, c in enumerate(log_cpm.columns)}
cohort = cohort.assign(_ord=cohort['column_name'].map(order_map))
cohort = cohort.sort_values('_ord').reset_index(drop=True).drop(columns='_ord')

# Build gene symbol → full-index mapping (index is ENSMUSG00000000001_Gnai3)
gene_to_idx = {}
for full_id in log_cpm.index:
    if '_' in full_id:
        sym = full_id.rsplit('_', 1)[1]
        gene_to_idx[sym] = full_id
    else:
        gene_to_idx[full_id] = full_id

# ----------------------------------------------------------------------------
# Compartment signature scores (per-sample median log2-CPM of marker genes)
# ----------------------------------------------------------------------------
scores = {}
for signature in panel['signature'].unique():
    markers = panel[panel['signature'] == signature]['gene'].tolist()
    present_idx = [gene_to_idx[g] for g in markers if g in gene_to_idx]
    if len(present_idx) < 3:
        print(f"  WARNING: {signature} has only {len(present_idx)} markers, skipping")
        continue
    sub = log_cpm.loc[present_idx]
    scores[signature] = sub.median(axis=0)
    print(f"  {signature:35s}  {len(present_idx):3d}/{len(markers):3d} markers present")

scores_df = pd.DataFrame(scores)
scores_df.index.name = 'sample'
scores_df.to_csv(OUT_DIR / 'scores_median.csv')

scores_z = (scores_df - scores_df.mean()) / scores_df.std(ddof=1)
scores_z.to_csv(OUT_DIR / 'scores_zscore.csv')

# ----------------------------------------------------------------------------
# ANOVA across stages
# ----------------------------------------------------------------------------
scores_with_group = scores_df.copy()
scores_with_group['group'] = [
    cohort.set_index('column_name').loc[s, 'group'] for s in scores_df.index
]

anova_rows = []
for signature in scores.keys():
    groups_vals = [scores_with_group[scores_with_group['group'] == g][signature].values
                   for g in sorted(scores_with_group['group'].unique())]
    F, p = stats.f_oneway(*groups_vals)
    anova_rows.append({'signature': signature, 'F': F, 'P.Value': p})

anova_df = pd.DataFrame(anova_rows)
_, fdrs, _, _ = multipletests(anova_df['P.Value'], method='fdr_bh')
anova_df['adj.P.Val'] = fdrs
anova_df.to_csv(OUT_DIR / 'anova_per_signature.csv', index=False)
print(f"\nANOVA per compartment:")
for _, row in anova_df.iterrows():
    print(f"  {row['signature']:35s}  F={row['F']:6.2f}  FDR={row['adj.P.Val']:.2e}")

# ----------------------------------------------------------------------------
# Pearson correlations: signature scores vs IL-10 axis
# ----------------------------------------------------------------------------
il10_axis = ['Il10', 'Il10ra', 'Il10rb', 'Stat3', 'Jak1', 'Jak2',
             'Socs3', 'Il6st', 'Scd2', 'Ddit4']
il10_idx_map = {g: gene_to_idx[g] for g in il10_axis if g in gene_to_idx}
il10_present = list(il10_idx_map.keys())

corr_r = pd.DataFrame(index=list(scores.keys()), columns=il10_present, dtype=float)
corr_p = pd.DataFrame(index=list(scores.keys()), columns=il10_present, dtype=float)

for ct in scores.keys():
    for g in il10_present:
        x = scores_df[ct].values
        y = log_cpm.loc[il10_idx_map[g], scores_df.index].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            corr_r.loc[ct, g] = np.nan
            corr_p.loc[ct, g] = np.nan
        else:
            r, p = stats.pearsonr(x[mask], y[mask])
            corr_r.loc[ct, g] = r
            corr_p.loc[ct, g] = p

flat_p = corr_p.values.flatten()
ok = np.isfinite(flat_p)
flat_fdr = np.full_like(flat_p, np.nan)
_, fdrs_ok, _, _ = multipletests(flat_p[ok], method='fdr_bh')
flat_fdr[ok] = fdrs_ok
corr_fdr = pd.DataFrame(flat_fdr.reshape(corr_r.shape),
                        index=corr_r.index, columns=corr_r.columns)

corr_r.to_csv(OUT_DIR / 'il10_correlations_r.csv')
corr_p.to_csv(OUT_DIR / 'il10_correlations_p.csv')
corr_fdr.to_csv(OUT_DIR / 'il10_correlations_fdr.csv')

print(f"\nIL-10 axis correlations (Pearson r):")
print(corr_r.round(2).to_string())

print(f"\nDone. Results in {OUT_DIR}/")
