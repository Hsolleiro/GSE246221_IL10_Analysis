"""
Differential expression analysis of GSE246221 STZ+HFD cohort (n=40).

Uses the custom Python limma-voom implementation (see limma_voom.py).
Computes six pairwise stage-wise contrasts plus a longitudinal F-test.

Outputs: results/deg/*.csv (one file per contrast + F_test_any_stage.csv).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths (portable — resolved relative to this script's location)
# ----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / 'data'
RESULTS_DIR = REPO_ROOT / 'results'
DEG_DIR = RESULTS_DIR / 'deg'
DEG_DIR.mkdir(parents=True, exist_ok=True)

# Import local library
sys.path.insert(0, str(SCRIPT_DIR))
from limma_voom import tmm_norm_factors, voom, fit_contrasts_directly, f_test_contrasts

# ----------------------------------------------------------------------------
# Load inputs
# ----------------------------------------------------------------------------
counts = pd.read_csv(DATA_DIR / 'counts_final.csv', index_col=0)
cohort = pd.read_csv(DATA_DIR / 'cohort_final.csv')
print(f"Counts matrix: {counts.shape[0]} genes x {counts.shape[1]} samples")
print(f"Cohort: {len(cohort)} samples")

# Build group vector aligned with count columns
cohort_map = dict(zip(cohort['column_name'], cohort['group']))
sample_order = list(counts.columns)
groups = np.array([cohort_map[s] for s in sample_order])
print(f"Group counts: {pd.Series(groups).value_counts().to_dict()}")

# ----------------------------------------------------------------------------
# Design matrix: ~0 + group (one-hot per stage)
# ----------------------------------------------------------------------------
group_levels = ['S1_Control_07w', 'S2a_EarlyMASLD_14w', 'S3_MASH_20w',
                'S4_Fibrosis_32w', 'S2b_ChronicNT_56w', 'S5_HCC']
design = np.zeros((len(sample_order), len(group_levels)))
for j, g in enumerate(group_levels):
    design[:, j] = (groups == g).astype(float)
print(f"Design matrix: {design.shape}  col-sums: {design.sum(axis=0).astype(int).tolist()}")

# ----------------------------------------------------------------------------
# TMM normalization + voom precision weights
# ----------------------------------------------------------------------------
nf = tmm_norm_factors(counts.values)
lib_size = counts.sum(axis=0).values
print(f"TMM factors range: {nf.min():.3f} - {nf.max():.3f}")

y, weights, Amean, sigma, df_resid = voom(
    counts.values, design,
    lib_size=lib_size, norm_factors=nf, span=0.5
)
print(f"voom output: y {y.shape}, weights range [{weights.min():.3g}, {weights.max():.3g}]")

# ----------------------------------------------------------------------------
# Contrast definitions
# ----------------------------------------------------------------------------
idx = {g: i for i, g in enumerate(group_levels)}
def c_vec(a, b):
    v = np.zeros(len(group_levels))
    v[idx[a]] = 1
    v[idx[b]] = -1
    return v

contrasts = {
    'EarlyMASLD_vs_Control':  c_vec('S2a_EarlyMASLD_14w', 'S1_Control_07w'),
    'MASH_vs_EarlyMASLD':     c_vec('S3_MASH_20w',        'S2a_EarlyMASLD_14w'),
    'Fibrosis_vs_MASH':       c_vec('S4_Fibrosis_32w',    'S3_MASH_20w'),
    'ChronicNT_vs_Fibrosis':  c_vec('S2b_ChronicNT_56w',  'S4_Fibrosis_32w'),
    'HCC_vs_ChronicNT':       c_vec('S5_HCC',             'S2b_ChronicNT_56w'),
    'HCC_vs_Control':         c_vec('S5_HCC',             'S1_Control_07w'),
}

# ----------------------------------------------------------------------------
# Fit pairwise contrasts (limma-voom + robust empirical Bayes)
# ----------------------------------------------------------------------------
gene_names = counts.index.tolist()
results, meta = fit_contrasts_directly(
    y, weights, design, contrasts, gene_names,
    trend=True, robust=True, span=0.5
)
print(f"\nEmpirical Bayes: df0={meta['df_prior']:.2f}, "
      f"s02_mean={meta['s2_prior']:.3f}")

# Save each contrast
for name, df in results.items():
    df.to_csv(DEG_DIR / f'{name}.csv', index=False)
    n_sig = (df['adj.P.Val'] < 0.05).sum()
    print(f"  {name:30s}  {n_sig:5d} genes FDR<0.05")

# ----------------------------------------------------------------------------
# Longitudinal F-test (any-stage difference)
# ----------------------------------------------------------------------------
# Build contrast matrix for 5 independent pairwise contrasts vs baseline
C = np.column_stack([c_vec('S2a_EarlyMASLD_14w', 'S1_Control_07w'),
                     c_vec('S3_MASH_20w',        'S1_Control_07w'),
                     c_vec('S4_Fibrosis_32w',    'S1_Control_07w'),
                     c_vec('S2b_ChronicNT_56w',  'S1_Control_07w'),
                     c_vec('S5_HCC',             'S1_Control_07w')])

ftest = f_test_contrasts(y, weights, design, C, gene_names,
                          trend=True, robust=True, span=0.5)
ftest.to_csv(DEG_DIR / 'F_test_any_stage.csv', index=False)
n_sig_f = (ftest['adj.P.Val'] < 0.05).sum()
print(f"\nF-test (any-stage difference): {n_sig_f} genes FDR<0.05")

print(f"\nDone. Results in {DEG_DIR}/")
