"""Regenerate the 14 programmatically plotted manuscript figures.

Approved reference figures remain untouched in ``reference_figures``. Fig. 1
is an authored illustration without plotting code and is not regenerated.
"""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    *(ROOT / "code/figures/main" / f"Fig{i:02d}_plot.py" for i in range(2, 6)),
    *(ROOT / "code/figures/extended_data" / f"ExtendedData_Fig{i:02d}_plot.py" for i in range(1, 11)),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-external", action="store_true", help="Render 13 figures without the separately downloaded Fig. 7 inputs")
    args = parser.parse_args()
    if not args.skip_external:
        validation = ROOT / "data/validation"
        needed = ["African_inland_water_reference_basin_table.csv", "ED07_benchmark_percentile_ranks.npz"]
        if not all((validation / name).is_file() for name in needed):
            raise SystemExit("Extended Data Fig. 7 requires external inputs. Run prepare_external_benchmarks.py with the original Borges workbook and IMAGE-GNM rasters, or use --skip-external.")
    completed = 0
    for script in SCRIPTS:
        if args.skip_external and script.name == "ExtendedData_Fig07_plot.py":
            continue
        print(f"Running {script.relative_to(ROOT)}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
        completed += 1
    print("{} plotting entry points completed.".format(completed))


if __name__ == "__main__":
    main()
