# GSE246221 Transcriptomic Re-analysis
**Targeted Analysis of the IL-10 Signaling Axis in a MASLD to HCC Mouse Model**

This repository contains the custom Python pipeline utilized for the bioinformatic re-analysis of bulk liver RNA-seq data from the NCBI Gene Expression Omnibus ([GSE246221](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE246221)). 

The script was specifically designed to evaluate the stage-resolved dynamics of the IL-10 signaling axis (*Il10*, *Il10ra*, *Il10rb*, *Stat3*, *Socs3*) across the progression from healthy liver, through inflammation, MASH, and fibrosis, to hepatocellular carcinoma (HCC).

## Features
* **Automated Metadata Curation:** Classifies samples into precise disease stages based on age, diet, and tissue annotation, excluding ambiguous samples.
* **Identifier Harmonization:** Resolves hybrid transcript identifiers to official NCBI gene symbols.
* **Robust Normalization:** Applies Log2-Counts Per Million (Log2-CPM) transformation.
* **Integrated Statistical Visualization:** Performs global One-Way ANOVA and Tukey’s Honestly Significant Difference (HSD) post-hoc tests. Automatically computes exact coordinates to draw significance brackets and asterisks directly onto seaborn distributions, preventing *p*-hacking and ensuring transparency.

## Requirements
To run this pipeline, you need Python 3.8+ and the following libraries:
```bash
pip install pandas numpy scipy statsmodels seaborn matplotlib GEOparse
