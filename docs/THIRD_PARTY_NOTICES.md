# Third-party source notices

This file distinguishes source licensing from the authors' own CC BY 4.0
release. An official download link does not transfer the source provider's
rights to the authors or to archive users.

| Source | Record, attribution and use in this archive |
| --- | --- |
| IPCC AR6 WGI Atlas | [Reference regions v4](https://github.com/IPCC-WG1/Atlas/tree/main/reference-regions), licensed by the Atlas repository under [CC BY 4.0](https://github.com/IPCC-WG1/Atlas/blob/main/LICENSE.md). The bundled 44-region layer is an ArcGIS-projected and filtered derivative of the official 58-feature v4 shapefile (12 ocean regions and polar ARO/ARS omitted). Cite Iturbide et al., *Scientific Data* **9**, 629 (2022), doi:10.1038/s41597-022-01739-y, and the IPCC WGI Atlas. |
| Natural Earth | `ne_110m_admin_0_countries` is the 1:110m Admin-0 countries layer. [Natural Earth terms](https://www.naturalearthdata.com/about/terms-of-use/) identify its vector data as public domain. Credit: Made with Natural Earth. |
| GRQA v1.4 | Virro et al., [versioned data record](https://doi.org/10.5281/zenodo.15335450), `GRQA_data_v1.4.zip` > `GRQA_data/NO3N_GRQA.csv`, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) per Zenodo record metadata. Also cite Virro et al., *Earth System Science Data* **13**, 5483-5507 (2021), doi:10.5194/essd-13-5483-2021. The supplied basin table is a modified, 1985-2014, quality-filtered and aggregated derivative with model NPI values recomputed against the current product. |
| HydroBASINS | [HydroBASINS v1 standard](https://www.hydrosheds.org/products/hydrobasins) files `hybas_{continent}_lev01-12_v1c.zip` underlie the analysis; the full polygons are not bundled. The [HydroSHEDS licence](https://data.hydrosheds.org/file/technical-documentation/HydroSHEDS_TechDoc_v1_4.pdf) imposes attribution and redistribution conditions. Cite Lehner and Grill, *Hydrological Processes* **27**, 2171-2186 (2013), doi:10.1002/hyp.9740. Any bundled basin codes or assignments are analytical derivatives, not a stand-alone HydroBASINS distribution; they are not relicensed as if they were original boundaries. |
| African inland-water observations | Borges et al., *Nature Geoscience* **8**, 637-642 (2015), doi:10.1038/ngeo2486; [ORBi annex `ngeo2486-s2.xlsx`](https://orbi.uliege.be/handle/2268/184379). The local input was byte-identical to this annex (SHA-256 `b2acac3c21f7c0ab9f02de2cf61ac5abab0038c6de502b600249fb1509b81da4`). The [ORBi user licence](https://orbi.uliege.be/page/user-license) does not grant an unrestricted right to publish derived values. Neither the workbook nor matched basin observations are bundled. |
| IMAGE-GNM | Beusen et al., [20th-century DANS data record](https://doi.org/10.17026/dans-zgs-9k9m) and model description in *Geoscientific Model Development* **8**, 4045-4067 (2015), doi:10.5194/gmd-8-4045-2015. The analysis uses `surface_runoff_agri.asc` plus `groundwater_agri.asc` for 1985, 1990, 1995 and 2000. The DANS record does not expose a clear reusable licence through its metadata; original rasters and matched ranks are not bundled. |

To reproduce Figure 7, obtain the Borges workbook and IMAGE-GNM rasters
directly from their original repositories and run `prepare_external_benchmarks.py`
as described in the root README. The script verifies all nine source checksums
before constructing the omitted matched files. Redistribution of those newly
generated files is governed by the respective original source conditions.
