# Agricultural development reshapes global hotspots of compound water–nitrogen pressure

**Anxin Ding**<sup>1,†</sup>, **Yi Yao**<sup>2,†,*</sup>

<sup>1</sup> School of Geographical Sciences, School of Carbon Neutrality Future Technology, Fujian Normal University, Fuzhou, China  
<sup>2</sup> Institute for Atmospheric and Climate Science, ETH Zurich, Zurich, Switzerland

† These authors contributed equally.  
\* Corresponding author.

## Overview

This repository contains the plotting code and processed inputs used to reproduce the main and Extended Data figures associated with the manuscript. The bundled inputs are sufficient to regenerate 13 figures without external downloads. Extended Data Fig. 7 requires additional benchmark data. Main Fig. 1 is original artwork by the authors and is included as a reference image.

The figure workflow starts from processed inputs. The analysis rebuild is documented in the companion analysis archive.

## Reproducing the figures

The figure environment was tested with Python 3.7.3. From the repository root, enter the `global-compound-water-nitrogen-pressure-reproduction` directory, install the dependencies listed in `requirements.txt`, and run the following commands in that directory.

```bash
python verify_package.py
python run_all.py --skip-external
python verify_package.py --require-generated --skip-external
```

Regenerated figures are written to `figures/`. Reference images are provided in `reference_figures/`.

To reproduce Extended Data Fig. 7, follow the benchmark preparation instructions in the [figure reproduction guide](./global-compound-water-nitrogen-pressure-reproduction/README.md). The guide also describes the software requirements, verification steps, and external data sources.

## Figure sources and licences

- The [figure data index](./global-compound-water-nitrogen-pressure-reproduction/FIGURE_DATA_INDEX.md) maps each figure to its plotting script and input files, where applicable.
- [LICENSE-CODE](./global-compound-water-nitrogen-pressure-reproduction/LICENSE-CODE) covers the original code.
- [LICENSE-DATA-AND-FIGURES.md](./global-compound-water-nitrogen-pressure-reproduction/LICENSE-DATA-AND-FIGURES.md) covers the results and figures owned by the authors.
- [Third-party notices](./global-compound-water-nitrogen-pressure-reproduction/docs/THIRD_PARTY_NOTICES.md) document the sources and applicable terms for third-party materials.

## Contact

For questions about the code or figure reproduction, please contact either of the authors.

- **Anxin Ding** — [danxin@whu.edu.cn](mailto:danxin@whu.edu.cn)
- **Yi Yao** (corresponding author) — [yi.yao@env.ethz.ch](mailto:yi.yao@env.ethz.ch)
