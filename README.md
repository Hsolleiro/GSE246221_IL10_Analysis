# GSE246221_IL10_Analysis

Reproducible Python pipeline for the stage-resolved transcriptomic analysis of the IL-10 axis in the MASLD/MASH-to-HCC mouse model (GEO accession GSE246221), as presented in Section 4.2 of:

> Solleiro-Villavicencio, H. et al. (2026). Hepatocarcinogenesis in the Context of Metabolic Dysfunction-Associated Steatotic Liver Disease: Emerging Roles of Interleukin-10 and Transcriptomic Insights into its Signaling Rewiring. *Biomedicines*, **14**, in press.

## Repository structure

```
GSE246221_IL10_Analysis/
│
├── README.md                           # this file
│
├── scripts/                             # all analysis scripts (Python 3.10+)
│   ├── limma_voom.py                   # custom limma-voom implementation
│   ├── symbol_normalizer.py            # gene symbol cleanup utilities
│   ├── run_deg.py                      # differential expression (limma-voom)
│   ├── run_deseq2.py                   # differential expression (PyDESeq2, sensitivity)
│   ├── run_celltype_signatures.py      # cell-type signature enrichment
│   ├── make_figure2.py                 # Figure 2 — IL-10 axis dynamics
│   ├── make_figure3.py                 # Figure 3 — stage-wise expression
│   ├── make_figure4.py                 # Figure 4 — cell-type signature remodeling
│   ├── make_figureS1_QC.py             # Supplementary Figure S1 — QC
│   ├── build_supplementary_tables.py   # builds Tables A1-A4 xlsx
│   ├── build_supplementary_methods.py  # builds Supplementary Methods docx
│   └── audit_methods.py                # internal methodological audit
│
├── data/                                # curated input data (n=40 cohort)
│   ├── counts_final.csv                # raw count matrix (20,100 genes × 40 samples)
│   ├── cohort_final.csv                # sample metadata + stage annotation
│   ├── log_cpm_final.csv               # TMM-normalized log2-CPM matrix
│   └── signature_panel.csv             # 4 cell-type marker gene lists
│
└── audit/                              # R↔Python methodological audit (see audit/README.md)
    ├── README.md
    ├── audit_colab_notebook.ipynb
    ├── audit_methods.py
    ├── colab_audit_package.zip
    ├── audit_R_vs_Python_comparison.csv
    ├── audit_limma_R_vs_Python.png
    ├── test4_permutations.csv
    └── test4_permutation_distribution.png
```

## How to reproduce

### Requirements

Python 3.10+ with the following packages:
```
numpy, pandas, scipy, statsmodels, scikit-learn, pydeseq2, 
matplotlib, openpyxl, python-docx
```

Install via:
```bash
pip install numpy pandas scipy statsmodels scikit-learn pydeseq2 matplotlib openpyxl python-docx
```

### Running the pipeline

From the repository root:

```bash
# 1. Differential expression (limma-voom)
python3 scripts/run_deg.py

# 2. Sensitivity analysis with PyDESeq2
python3 scripts/run_deseq2.py

# 3. Cell-type signature enrichment
python3 scripts/run_celltype_signatures.py

# 4. Figures
python3 scripts/make_figure2.py
python3 scripts/make_figure3.py
python3 scripts/make_figure4.py
python3 scripts/make_figureS1_QC.py

# 5. Supplementary materials
python3 scripts/build_supplementary_tables.py
python3 scripts/build_supplementary_methods.py

# 6. Optional: run the 6-test internal methodological audit
python3 scripts/audit_methods.py
```

Results are written to `results/` (created automatically):
- `results/deg/` — limma-voom contrast tables and F-test
- `results/deg_deseq2/` — PyDESeq2 sensitivity results
- `results/celltypes/` — per-sample signature scores, ANOVA, IL-10 correlations
- `results/figures/` — Figures 2, 3, 4, S1 (PNG 300 dpi + PDF)
- `results/supplementary/` — Tables A1-A4 (xlsx) + Supplementary Methods (docx)

## Key methodological notes

- **Cohort curation**: Starting from n=55 samples in GSE246221, we excluded all pharmacological-intervention samples (Tirzepatide + vehicle, n=10) and all HFD-only Batch 2 samples (n=5), yielding a final cohort of **n=40 biologically independent Batch 1 samples** across 6 disease stages.
- **Signature analysis is NOT deconvolution**: Cell-type signature scores are per-sample median log2-CPM of curated marker gene panels. They report relative enrichment across stages, not absolute cellular proportions. This distinction is explicit throughout.
- **Custom Python limma-voom vs R reference**: The custom implementation (`scripts/limma_voom.py`) has been validated against the reference R limma package on the same cohort data. Pearson correlation of log2FoldChange estimates is **1.000** in all six pairwise contrasts. See `audit/` for details.

## Citation

If you use this code or adapt it for your own analyses, please cite the Biomedicines paper above.

## Contact

Helena Solleiro-Villavicencio — helena.solleiro@uacm.edu.mx
