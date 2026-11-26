"""Hash every distributed source file in the figure-only archive."""

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "source_sha256.csv"
SKIP_ROOTS = {"figures", "audit"}
GENERATED_EXTERNAL = {
    "data/validation/African_inland_water_reference_basin_table.csv",
    "data/validation/ED07_benchmark_percentile_ranks.npz",
    "data/validation/ExtendedData_Fig07_panel_b_observations_GRQA_v1.4.csv",
}


def source_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST or path.suffix == ".pyc":
            continue
        relative = path.relative_to(ROOT)
        if (relative.parts[0] in SKIP_ROOTS
                or relative.as_posix() in GENERATED_EXTERNAL
                or "__pycache__" in relative.parts):
            continue
        yield relative.as_posix(), path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
        writer.writerow(("relative_path", "sha256", "size_bytes"))
        count = 0
        for relative, path in source_files():
            writer.writerow((relative, digest(path), path.stat().st_size))
            count += 1
    print("Hashed {} figure-package source files".format(count))


if __name__ == "__main__":
    main()
