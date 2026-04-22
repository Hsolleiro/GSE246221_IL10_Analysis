"""
PyDESeq2 sensitivity analysis — cross-validates limma-voom results
using an independent method (Wald test on rounded integer counts).

Inputs:  data/counts_final.csv, data/cohort_final.csv
Outputs: results/deg_deseq2/*.csv (one per contrast + summary.csv)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / 'data'
RESULTS_DIR = REPO_ROOT / 'results'
OUT_DIR = RESULTS_DIR / 'deg_deseq2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference

counts = pd.read_csv(DATA_DIR / 'counts_final.csv', index_col=0)
cohort = pd.read_csv(DATA_DIR / 'cohort_final.csv')
sample_order = list(counts.columns)
cohort_map = dict(zip(cohort['column_name'], cohort['group']))

# PyDESeq2 expects (samples x genes) and metadata indexed by samples
counts_t = counts.T.round().astype(int)  # rounded — RSEM decimals
metadata = pd.DataFrame({'group': [cohort_map[s] for s in sample_order]},
                        index=sample_order)
print(f"Counts for DESeq2: {counts_t.shape}, Metadata: {metadata.shape}")
print(metadata['group'].value_counts())

# ----------------------------------------------------------------------------
# Fit DESeq2 model
# ----------------------------------------------------------------------------
inference = DefaultInference(n_cpus=4)
dds = DeseqDataSet(
    counts=counts_t,
    metadata=metadata,
    design_factors='group',
    inference=inference,
    quiet=True,
)
dds.deseq2()
print("DESeq2 fit complete")

# ----------------------------------------------------------------------------
# Extract each contrast
# ----------------------------------------------------------------------------
contrasts = [
    ('EarlyMASLD_vs_Control',  'S2a_EarlyMASLD_14w', 'S1_Control_07w'),
    ('MASH_vs_EarlyMASLD',     'S3_MASH_20w',        'S2a_EarlyMASLD_14w'),
    ('Fibrosis_vs_MASH',       'S4_Fibrosis_32w',    'S3_MASH_20w'),
    ('ChronicNT_vs_Fibrosis',  'S2b_ChronicNT_56w',  'S4_Fibrosis_32w'),
    ('HCC_vs_ChronicNT',       'S5_HCC',             'S2b_ChronicNT_56w'),
    ('HCC_vs_Control',         'S5_HCC',             'S1_Control_07w'),
]

summary_rows = []
for name, A, B in contrasts:
    ds = DeseqStats(dds, contrast=['group', A, B], inference=inference, quiet=True)
    ds.summary()
    out = ds.results_df.copy()
    out['gene'] = out.index
    # Rename to match limma-voom column naming
    out = out.rename(columns={
        'log2FoldChange': 'logFC',
        'lfcSE': 'lfcSE',
        'stat': 'Wald_stat',
        'pvalue': 'P.Value',
        'padj': 'adj.P.Val',
    })
    out = out[['gene', 'baseMean', 'logFC', 'lfcSE', 'Wald_stat',
               'P.Value', 'adj.P.Val']].reset_index(drop=True)
    out.to_csv(OUT_DIR / f'{name}.csv', index=False)
    n_sig = (out['adj.P.Val'] < 0.05).sum()
    print(f"  {name:30s}  {n_sig:5d} genes padj<0.05")
    summary_rows.append({'contrast': name, 'A': A, 'B': B, 'n_padj_0.05': n_sig})

pd.DataFrame(summary_rows).to_csv(OUT_DIR / 'summary.csv', index=False)
print(f"\nDone. Results in {OUT_DIR}/")
