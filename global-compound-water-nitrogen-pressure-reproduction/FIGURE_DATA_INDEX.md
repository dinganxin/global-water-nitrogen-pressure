# Figure inputs

All paths are relative to this package. These are **plotting inputs**,
not the original CESM2 simulations or raw water-quality archives.

| Figure | Principal input |
| --- | --- |
| Fig. 1 | Authored illustration only: `reference_figures/main/Fig01_conceptual_framework.jpg`; no plotting dataset or drawing script |
| Fig. 2 | `data/figure_inputs/Fig02_native_grid_components_QIRR_2045_2074.npz`, companion member-difference and crop-grid arrays, Natural Earth boundaries |
| Fig. 3 | `data/core/NPI_IPI_QIRR_strict_positiveQ_period_products_01pct.npz`, `tables/main/Fig03_global_class_area.csv`, Natural Earth and IPCC AR6 boundaries |
| Fig. 4 | period products, `tables/main/Fig04_temporal_transition_summary_strict_positiveQ_01pct.csv` |
| Fig. 5 | period products, basin metadata and regional exposure/fingerprint/ensemble tables in `tables/main/` |
| Extended Data Fig. 1 (IPI) | period products and Natural Earth line cache |
| Extended Data Fig. 2 (NPI) | period products and Natural Earth line cache |
| Extended Data Fig. 3 | `tables/threshold_sensitivity_strict_positiveQ.csv` |
| Extended Data Fig. 4 | `tables/low_positive_Q_sensitivity_strict_positiveQ.csv` |
| Extended Data Fig. 5 | `data/extended_data/runoff_denominator_sensitivity_strict_positiveQ_01pct.npz`, map cache and tables in `tables/extended_data/ExtendedData_Fig05/` |
| Extended Data Fig. 6 | `data/extended_data/HydroBASINS_levels3_to_6_sensitivity_effective_land_area.npz` and corresponding CSV in `tables/` |
| Extended Data Fig. 7 | Packaged `data/validation/GRQA_v1.4_nonoutlier_basin_constraint.csv` plus African matched observations and IMAGE-GNM ranks recreated locally by `prepare_external_benchmarks.py` from the original downloads; the latter two inputs are not distributed in this archive |
| Extended Data Fig. 8 | period products and member-agreement CSV in `tables/extended_data/` |
| Extended Data Fig. 9 | annual global and component-substitution CSVs in `tables/extended_data/` |
| Extended Data Fig. 10 | regional time-series and summary CSVs in `tables/extended_data/`, basin metadata |

The duplicated period-product NPZ at `data/` and `data/core/` is intentional:
different plotting entry points reference these exact paths. Likewise, two
identical basin metadata CSVs are kept because separate scripts expect their
original names. Do not remove either copy without updating and retesting the
code.
