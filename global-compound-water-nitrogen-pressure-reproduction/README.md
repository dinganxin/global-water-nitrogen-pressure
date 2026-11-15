# Water-NPI-IPI figure sources

This is the figure-only companion to `Water_NPI_IPI_analysis_and_figure_sources_20261114_v9`.
It contains the final source inputs and plotting code for five main figures and
ten Extended Data figures. Figure 1 is artwork created by the authors; no
plotting script is claimed for it. The period products and CSVs are processed
figure inputs, not raw CESM2-CLM5 simulation output. Analysis-rebuild and raw
ingestion scripts are intentionally absent from this smaller archive.

## Contents

- `code/figures/`: 14 plotting entry points and one shared plotting helper.
- `data/`: processed arrays, basin metadata, validation inputs and map layers
  needed by the plotting scripts. Original model and third-party archives are
  not bundled.
- `tables/`: final figure-supporting CSVs.
- `repository_data/`: five compact result tables used in the manuscript.
- `supplementary/`: Supplementary Table 1 and its description.
- `reference_figures/`: the 15 approved main and Extended Data images.
- `FIGURE_DATA_INDEX.md` and `figure_manifest.csv`: figure-to-source mapping.
- `source_sha256.csv`: integrity hashes for the distributed source files.

The folder name `main` denotes figures in the main text. `extended_data`
denotes Extended Data figures. `strict_positiveQ` identifies the final
classification, whereas `Q <= 0` is reported as a separate state. The `01pct`
suffix identifies the 1% cropland-domain criterion; it is not a significance
level. Some period-product files occur at two paths because the established
plotting scripts use both locations; their contents are identical.

## Reproduce the figures

Run the plotting scripts from this directory using the figure environment in
`requirements.txt` (tested with Python 3.7.3):

```text
python verify_package.py
python run_all.py --skip-external
python verify_package.py --require-generated --skip-external
```

These commands render and check 13 figures without external downloads. The
main Fig. 1 image is supplied artwork, not a code-generated figure. The
approved images remain in `reference_figures/`; regenerated files are written
to `figures/` and may not be byte-identical across systems.

Extended Data Fig. 7 additionally requires the Borges et al. African
inland-water workbook and eight IMAGE-GNM agricultural nitrogen ASCII files.
The external original files and matched derivatives are deliberately not
redistributed here. Obtain the originals from the links in
`docs/THIRD_PARTY_NOTICES.md`. Using the separate benchmark-preparation
environment in `requirements-benchmark-prep.txt` (tested with Python 3.13.9),
run:

```text
python prepare_external_benchmarks.py --africa-xlsx <ngeo2486-s2.xlsx> --gnm-root <GNM_database/nitrogen>
python verify_external_benchmarks.py
```

The preparation script verifies the original input SHA-256 checksums before
creating local derived files in `data/validation/`. Then use the figure
environment to run `python run_all.py` and
`python verify_package.py --require-generated`. Do not add the locally
generated Borges or IMAGE-GNM matched values to a public copy of this archive
without confirming redistribution rights.

## Scope and release checks

The larger companion archive contains annual member-basin inputs and the
analysis rebuild. This figure-only archive can reproduce plots from packaged
processed inputs, but does not independently recreate the scientific results
from raw simulations. Figure inputs, figure code and locked reference images
were taken from the same version 9 complete archive; no numerical values or
approved images were changed in preparing this smaller archive.

The current land-fraction-adjusted benchmark values are GRQA v1.4
`r_s = 0.44` (617 basins), African inland waters `r_s = 0.17` (44 basins), and
IMAGE-GNM `r_s = 0.48` (2,166 basins). Before public release, align the
manuscript's Discussion, Extended Data Fig. 7 caption and artwork with these
values as described in `docs/MANUSCRIPT_ALIGNMENT_REQUIRED.md`. Record the
repository DOI in the manuscript only after the final archive is deposited.

Original code is licensed under `LICENSE-CODE`; author-owned results and
figures under `LICENSE-DATA-AND-FIGURES.md`. Third-party rights are listed in
`docs/THIRD_PARTY_NOTICES.md` and are not overridden by the authors' licences.
