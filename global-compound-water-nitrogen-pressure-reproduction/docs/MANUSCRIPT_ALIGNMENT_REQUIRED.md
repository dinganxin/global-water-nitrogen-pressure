# Required manuscript synchronization before deposit

The package's current model product uses effective CLM land area. The earlier
external-matching tables retained NPI values from the pre-correction product.
The observational concentrations, matched basin IDs and sample sizes did not
change; the model NPI column was reconciled to the packaged current product.

| Comparison | Earlier manuscript | Current-product result | Sample |
| --- | ---: | ---: | ---: |
| GRQA v1.4 | 0.43 (`0.42895`) | **0.44** (`0.43605`) | 617 basins |
| African inland waters | 0.16 (`0.16209`) | **0.17** (`0.16710`) | 44 basins |
| IMAGE-GNM | 0.48 (`0.47977`) | **0.48** (`0.47977`) | 2,166 basins |

In `Water_IPI_NPI _261113.docx`, update the two observational coefficients
in the Discussion limitations paragraph and in the caption of Extended Data
Fig. 7. The corrected artwork is
`reference_figures/extended_data/ExtendedData_Fig07_independent_nitrogen_benchmarking.png`.
Do not replace the old values in the manuscript without also replacing the
old Extended Data Fig. 7 image used in Word or Overleaf. A DOI for this archive
does not exist yet; update the Data and Code availability statements only after
deposit.

The GRQA matching itself was not rerun from the 1.2 GB original archive in
this release. Its supplied 617-basin observational table is unchanged except
for the model NPI column. The Borges original workbook and eight IMAGE-GNM
ASCII files were re-read in this correction and verified by SHA-256.
