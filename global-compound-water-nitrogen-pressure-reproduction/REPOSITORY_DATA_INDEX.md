# Machine-readable result tables

These five CSVs are repository result data, **not** five more Extended Data
display items. Supplementary Table 1 is separate under `supplementary/`.

| File | Rows | Scope |
| --- | ---: | --- |
| `Repository_Data_01_global_pressure_summary.csv` | 3 | Global historical and future pressure classes, non-positive-Q cropland and ensemble summaries |
| `Repository_Data_02_IPCC_AR6_regional_summary.csv` | 129 | Region-period pressure classes, emerging area, pressure indices and member support |
| `Repository_Data_03_top30_HydroBASINS_level5.csv` | 30 | Ranked SSP3-7.0 basin IDs and cumulative compound-area share |
| `Repository_Data_04_robustness_tests_long.csv` | 54 | Threshold, low-positive-Q and basin-scale sensitivity tests; inapplicable fields are blank |
| `Repository_Data_05_external_evaluation_summary.csv` | 3 | Spatial Spearman correlations and sample sizes for IMAGE-GNM, GRQA v1.4 and African inland-water comparisons |

Areas ending in `_million_km2` are in million square kilometres. Q <= 0 is
a separate hydrological state, not a fifth positive-Q pressure class.
External correlations compare spatial ordering; they do not validate the
absolute NPI as a measured river concentration. The regional summary covers
43 regions per period. Its total cropland area is slightly below the global
area (by 0.00304, 0.00319 and 0.01104 million km2 in historical, SSP1-2.6
and SSP3-7.0, respectively). Four basin units have no region code in the
used IPCC AR6 raster mask; their cropland is included globally but omitted
from the 43-region sum. In SSP3-7.0 their area is entirely Low with positive
Q. Compound and non-positive-Q areas reconcile exactly. No basin was
reassigned merely to force closure.
