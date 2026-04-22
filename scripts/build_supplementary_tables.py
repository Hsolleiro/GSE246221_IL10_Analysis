"""
build_supplementary_tables.py — Build Supplementary Tables A1-A4

Creates Supplementary_Tables_A1-A4.xlsx with multiple sheets:
  A1   — Cohort composition (n=40 samples × stage metadata)
  A1b  — Histological grading distributions per stage
  A2a  — limma-voom DE: all 6 pairwise contrasts (top rows)
  A2b  — limma-voom DE: F-test longitudinal (all genes)
  A2c  — PyDESeq2 DE: HCC vs Control for cross-validation
  A3   — Cell-type signature panel (4 cell types × marker genes)
  A4a  — Per-sample signature scores (n=40 × 4 signatures)
  A4b  — Per-signature ANOVA (F-statistics + FDR)

Inputs:
  data/cohort_final.csv
  results/deg/*.csv
  results/deg_deseq2/*.csv
  data/signature_panel.csv
  results/celltypes/*.csv

Output:
  results/supplementary/Supplementary_Tables_A1-A4.xlsx
"""

from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SUPP_DIR = RESULTS_DIR / "supplementary"
SUPP_DIR.mkdir(parents=True, exist_ok=True)


def sheet_A1_cohort(cohort):
    """Sheet A1: cohort composition."""
    cols_keep = ['column_name', 'GSM', 'batch', 'age', 'diet',
                  'STZ', 'sex', 'genotype', 'strain', 'group']
    cols_keep = [c for c in cols_keep if c in cohort.columns]
    out = cohort[cols_keep].copy()
    out.columns = [c.replace('_', ' ').title() for c in out.columns]
    return out


def sheet_A1b_histology(cohort):
    """Sheet A1b: histological grading per stage.
    We use age+STZ+diet as a proxy for histological stage if grading
    isn't directly available. This is documented honestly."""
    if 'steatosis' in cohort.columns:
        # Actual histology
        pivot = cohort.groupby(['group', 'steatosis']).size().unstack(fill_value=0)
        return pivot
    else:
        # Derived from metadata
        summary = cohort.groupby('group').agg(
            n_samples=('column_name', 'count'),
            age=('age', 'first'),
            diet=('diet', 'first'),
            STZ=('STZ', 'first'),
        ).reset_index()
        summary.columns = ['Stage', 'n', 'Age (weeks)', 'Diet', 'STZ treatment']
        # Add histological annotation note
        summary['Expected histology'] = [
            'Normal parenchyma' if 'Control' in s else
            'Microsteatosis' if 'Early' in s else
            'Macrosteatosis + inflammation' if 'MASH' in s else
            'Fibrotic septa' if 'Fibrosis' in s else
            'Chronic non-tumor inflammation' if 'ChronicNT' in s else
            'HCC nodules'
            for s in summary['Stage']
        ]
        return summary


def sheet_A2a_limma_contrasts(contrasts_list):
    """Sheet A2a: top DE genes from each limma-voom contrast."""
    rows = []
    for contrast_name, df in contrasts_list:
        # Top 50 by adj.P.Val
        top = df.nsmallest(50, 'adj.P.Val').copy()
        top['contrast'] = contrast_name
        rows.append(top)
    combined = pd.concat(rows, ignore_index=False)
    # Reorder columns
    col_order = ['contrast', 'gene', 'logFC', 'AveExpr', 't',
                  'P.Value', 'adj.P.Val', 'B']
    col_order = [c for c in col_order if c in combined.columns]
    combined = combined[col_order]
    return combined


def sheet_A2b_ftest(ftest_df):
    """Sheet A2b: longitudinal F-test all genes."""
    # Top 200 by adj.P.Val
    top = ftest_df.nsmallest(200, 'adj.P.Val').copy()
    return top


def sheet_A2c_deseq2(deseq_df):
    """Sheet A2c: PyDESeq2 results for HCC vs Control."""
    top = deseq_df.nsmallest(100, 'padj').copy()
    return top


def sheet_A3_signature_panel(sig_panel):
    """Sheet A3: cell-type signature panel."""
    return sig_panel


def sheet_A4a_scores(scores, cohort):
    """Sheet A4a: per-sample signature scores."""
    scores_out = scores.copy()
    scores_out.index.name = 'sample'
    # Add stage info
    stage_lookup = cohort.set_index('column_name')['group'].to_dict()
    scores_out['stage'] = [stage_lookup.get(s, '?') for s in scores_out.index]
    # Reorder so stage is first
    cols = ['stage'] + [c for c in scores_out.columns if c != 'stage']
    return scores_out[cols].reset_index()


def sheet_A4b_anova(anova_df):
    """Sheet A4b: per-signature ANOVA."""
    out = anova_df.copy()
    out.columns = [c.replace('_', ' ').title() if c != 'signature' else 'Signature'
                    for c in out.columns]
    return out


def load_all_contrasts():
    """Load all limma-voom contrast results (6 pairwise contrasts)."""
    deg_dir = RESULTS_DIR / 'deg'
    contrasts = []
    if not deg_dir.exists():
        print("  WARNING: results/deg/ not found — skipping contrast sheets")
        return contrasts
    for csv in sorted(deg_dir.glob('*_vs_*.csv')):
        name = csv.stem
        df = pd.read_csv(csv)
        contrasts.append((name, df))
    return contrasts


def main():
    print("Loading inputs...")
    cohort = pd.read_csv(DATA_DIR / 'cohort_final.csv')
    sig_panel = pd.read_csv(DATA_DIR / 'signature_panel.csv')
    
    # Try to load DE results
    contrasts = load_all_contrasts()
    print(f"  Found {len(contrasts)} limma-voom contrasts")
    
    ftest_path = RESULTS_DIR / 'deg' / 'Ftest_stagewise.csv'
    ftest_df = None
    if ftest_path.exists():
        ftest_df = pd.read_csv(ftest_path)
        print(f"  F-test: {len(ftest_df)} genes")
    
    deseq_path = RESULTS_DIR / 'deg_deseq2' / 'HCC_vs_Control_deseq2.csv'
    deseq_df = None
    if deseq_path.exists():
        deseq_df = pd.read_csv(deseq_path)
        print(f"  DESeq2: {len(deseq_df)} genes")
    
    scores_path = RESULTS_DIR / 'celltypes' / 'scores_median.csv'
    anova_path = RESULTS_DIR / 'celltypes' / 'anova_per_signature.csv'
    scores = pd.read_csv(scores_path, index_col=0) if scores_path.exists() else None
    anova = pd.read_csv(anova_path) if anova_path.exists() else None
    
    out_path = SUPP_DIR / 'Supplementary_Tables_A1-A4.xlsx'
    print(f"\nWriting {out_path}...")
    
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        # Sheet A1
        A1 = sheet_A1_cohort(cohort)
        A1.to_excel(writer, sheet_name='A1_cohort_composition', index=False)
        print(f"  ✓ A1_cohort_composition ({len(A1)} rows)")
        
        # Sheet A1b
        A1b = sheet_A1b_histology(cohort)
        A1b.to_excel(writer, sheet_name='A1b_histology_stages', index=False)
        print(f"  ✓ A1b_histology_stages")
        
        # Sheet A2a
        if contrasts:
            A2a = sheet_A2a_limma_contrasts(contrasts)
            A2a.to_excel(writer, sheet_name='A2a_limma_contrasts_top50',
                          index=False)
            print(f"  ✓ A2a_limma_contrasts_top50 ({len(A2a)} rows)")
        
        # Sheet A2b
        if ftest_df is not None:
            A2b = sheet_A2b_ftest(ftest_df)
            A2b.to_excel(writer, sheet_name='A2b_Ftest_top200',
                          index=False)
            print(f"  ✓ A2b_Ftest_top200 ({len(A2b)} rows)")
        
        # Sheet A2c
        if deseq_df is not None:
            A2c = sheet_A2c_deseq2(deseq_df)
            A2c.to_excel(writer, sheet_name='A2c_DESeq2_HCCvsCtrl',
                          index=False)
            print(f"  ✓ A2c_DESeq2_HCCvsCtrl ({len(A2c)} rows)")
        
        # Sheet A3
        A3 = sheet_A3_signature_panel(sig_panel)
        A3.to_excel(writer, sheet_name='A3_signature_panel', index=False)
        print(f"  ✓ A3_signature_panel ({len(A3)} rows)")
        
        # Sheet A4a
        if scores is not None:
            A4a = sheet_A4a_scores(scores, cohort)
            A4a.to_excel(writer, sheet_name='A4a_sample_scores',
                          index=False)
            print(f"  ✓ A4a_sample_scores ({len(A4a)} rows)")
        
        # Sheet A4b
        if anova is not None:
            A4b = sheet_A4b_anova(anova)
            A4b.to_excel(writer, sheet_name='A4b_signature_anova',
                          index=False)
            print(f"  ✓ A4b_signature_anova ({len(A4b)} rows)")
    
    print(f"\nDone: {out_path}")


if __name__ == '__main__':
    main()
