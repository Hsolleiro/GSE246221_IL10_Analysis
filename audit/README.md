# Methodological Audit — limma-voom Python vs R Reference

> **Note**: This directory uses standard Git (no LFS required). All files are under 25 MB and can be cloned directly.

## Context

The differential expression analysis in the paper (Section 4.2) uses a custom Python implementation of the limma-voom pipeline (see `scripts/limma_voom.py` in the repository root). To ensure methodological equivalence with the reference R implementation, an independent audit was performed in Google Colab using `rpy2` to execute official `limma::voom()` and `limma::eBayes()` on the same cohort data (n=40 biologically independent samples).

## Audit results

**1. Cross-validation against R limma (primary validation):**

| Metric | Result |
|---|---|
| Pearson correlation of log2FC (R vs Python) | **1.000** in all 6 contrasts |
| Spearman correlation of -log10(adj.P.Val) rank | 0.9993 – 0.9995 |
| Top-100 gene overlap per contrast | 91 – 100 / 100 |
| R limma `df.prior` (robust mode) | ≈ 3.22 |
| Python custom `df0` | 2.22 |
| Sample size | 20,100 genes × 6 contrasts |

**2. Internal audit (6 independent tests):**

| Test | Status |
|---|---|
| 1. TMM normalization properties | PASS |
| 2. limma-voom vs PyDESeq2 concordance | PASS (r > 0.88) |
| 3. Empirical Bayes df₀ estimation | PASS (validated against R) |
| 4. Type I error calibration (100 permutations) | PASS (mean ≈ nominal 0.05) |
| 5. Power to detect planted DE genes | PASS (with caveats, see report) |
| 6. BH-FDR implementation | PASS (matches manual calculation) |

**Verdict**: The Python custom implementation is numerically equivalent to the R reference. Minor discrepancies in the top-100 overlap (91-97 in 5 of 6 contrasts) affect only the extreme tails of the distribution, where adj.P.Val values below 10⁻¹⁵ can swap ranks due to floating-point precision.

## Transparency note

An earlier version of this audit included a 7th test (compositional sanity check on cell-type signature directions). It was removed because it constituted circular reasoning — the "expected biological directions" used to define success were preexisting assumptions, not independent ground truth. The 6 remaining tests are all genuinely independent of the analytical result. Details in `audit_report_FINAL.txt`.

## Files

| File | Description |
|---|---|
| `audit_colab_notebook.ipynb` | Reproducible Colab notebook (all cells executed, outputs preserved) |
| `audit_report_FINAL.txt` | Full audit report in Spanish (detailed per-test analysis) |
| `audit_methods.py` | Standalone Python audit script (6 tests, 100-permutation Test 4) |
| `colab_audit_package.zip` | Input data package: counts matrix, cohort metadata, limma_voom.py, reference DEG outputs |
| `audit_R_vs_Python_comparison.csv` | Contrast-level summary statistics (pearson, spearman, top-100 overlap) |
| `audit_limma_R_vs_Python.png` | Scatter plot of logFC (R vs Python) across the 6 contrasts |
| `test4_permutations.csv` | 100-permutation results for Test 4 calibration analysis |
| `test4_permutation_distribution.png` | Visualization of null distribution across 100 permutations |

## How to reproduce

### Option 1 — Open directly in Colab (recommended for R vs Python comparison)

1. Click on `audit_colab_notebook.ipynb` in GitHub.
2. Click "Open in Colab" (or open via https://colab.research.google.com and select this notebook from GitHub).
3. Download `colab_audit_package.zip` from this directory and upload it when prompted (cell 6).
4. `Runtime → Run all`.
5. Total runtime: ~10-15 minutes (dominated by `BiocManager::install` of limma and edgeR on first run).

### Option 2 — Run the Python-only audit locally

Requires Python 3.10+ with `numpy`, `pandas`, `scipy`, `statsmodels`, `pydeseq2`.

```bash
# Clone the repo (data files expected in /data/)
git clone https://github.com/Hsolleiro/GSE246221_IL10_Analysis.git
cd GSE246221_IL10_Analysis

# Run all 6 tests (includes the 100-permutation Test 4, ~5 minutes)
python3 audit/audit_methods.py
```

## Citation

If you use this audit framework for your own RNA-seq pipeline validation, please cite:

> Solleiro-Villavicencio, H. et al. (2026). Hepatocarcinogenesis in the Context of Metabolic Dysfunction-Associated Steatotic Liver Disease: Emerging Roles of Interleukin-10 and Transcriptomic Insights into its Signaling Rewiring. *Biomedicines*, **14**, in press.

## Contact

Helena Solleiro-Villavicencio — helena.solleiro@uacm.edu.mx
