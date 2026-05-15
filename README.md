# PCA_Gene_Expression

This repository contains the PCA assignment on gene expression data (GSE5325).

## Assignment goals

1. Plot GATA3 vs XBP1 expression for all 105 samples and color by class label:
	 - `1` = ER+ breast cancer
	 - `0` = ER- breast cancer
2. Run PCA on the 2-gene matrix and project samples onto PC1.

## Repository contents

- `pca_assignment.py` : main script for loading data, plotting, PCA, and summary generation.
- `data/class.tsv` : class labels (ER+/ER-).
- `data/filtered.tsv.gz` : filtered gene expression matrix.
- `data/columns.tsv.gz` : gene ID mapping metadata.
- `figure_1a_gata3_xbp1.png` : scatter for GATA3 vs XBP1.
- `figure_1b_pc_axes_raw.png` : PC axes over raw 2D scatter.
- `figure_1c_pc1_projection.png` : PC1 projection panel figure.
- `figure_scree_full_matrix.png` : scree plot for full matrix PCA.
- `figure_biplot_pc1_pc2.png` : biplot with selected loadings.
- `OUTPUT_SUMMARY.txt` : textual summary of results.

## Dependencies

Install Python packages:

```bash
pip install -r requirements.txt
```

## Run

From this repository root:

```bash
python pca_assignment.py
```

The script generates all output figures and updates `OUTPUT_SUMMARY.txt`.

## Notes

- The script expects data files inside the local `data/` directory.
- Gene/probe IDs used for core assignment plots:
	- GATA3: `4359`
	- XBP1: `4404`
- Additional exploratory plots (scree and biplot) are included as supplemental analysis.