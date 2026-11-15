# External benchmark provenance

The benchmark statistics use the effective-CLM-land-area NPI product supplied
in this archive. The locked Extended Data Fig. 7 and
`repository_data/Repository_Data_05_external_evaluation_summary.csv` report:

| Comparison | Spearman r_s | Basins |
| --- | ---: | ---: |
| GRQA v1.4 nitrate-N observations | 0.44 | 617 |
| African inland-water nitrate-N observations | 0.17 | 44 |
| IMAGE-GNM agricultural nitrogen transfer | 0.48 | 2,166 |

The supplied 617-basin GRQA observational table was retained from the
existing quality-filtered matching, with its model NPI column recalculated
using the current product. The original GRQA archive was not reprocessed from
raw records in this figure-only archive. The Borges workbook and eight
IMAGE-GNM ASCII files were reread for the other comparisons and verified
against their source SHA-256 checksums. Those original files and their
matched derivatives are not redistributed here; download links and rights
notices are in `THIRD_PARTY_NOTICES.md`.
