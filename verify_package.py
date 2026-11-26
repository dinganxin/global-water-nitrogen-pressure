"""Check the integrity and basic numerical consistency of this figure package."""

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def rows(path):
    with (ROOT / path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def near(actual, expected, label, tolerance=0.00001):
    if abs(actual - expected) > tolerance:
        raise AssertionError("{}: {} != {}".format(label, actual, expected))


def check_figures(require_generated, skip_external=False):
    manifest = rows("figure_manifest.csv")
    assert len(manifest) == 14, "Expected four plotted main and ten Extended Data figures"
    assert Counter("main" if r["item"].startswith("Main") else "extended" for r in manifest) == {
        "main": 4, "extended": 10
    }
    assert all(r["generated_path"] for r in manifest)
    for row in manifest:
        if require_generated:
            if skip_external and row["item"] == "Extended Data Fig. 7":
                continue
            generated = ROOT / row["generated_path"]
            assert generated.is_file() and generated.stat().st_size > 0, (
                "Missing regenerated figure: {}".format(generated)
            )
    print("Figure output manifest: 14/14 entries valid")
    if require_generated:
        expected = 13 if skip_external else 14
        print("Regenerated figures: {}/{} present".format(expected, expected))
    return manifest


def check_tables():
    names = [
        "Repository_Data_01_global_pressure_summary.csv",
        "Repository_Data_02_IPCC_AR6_regional_summary.csv",
        "Repository_Data_03_top30_HydroBASINS_level5.csv",
        "Repository_Data_04_robustness_tests_long.csv",
        "Repository_Data_05_external_evaluation_summary.csv",
    ]
    tables = [rows("repository_data/" + name) for name in names]
    assert [len(table) for table in tables] == [3, 129, 30, 54, 3]
    global_rows, regional, _, _, external = tables
    by_period = {row["period_key"]: row for row in global_rows}
    assert set(by_period) == {"historical", "ssp126", "ssp370"}
    for period, expected in {
        "historical": 1.0640395132975,
        "ssp126": 1.265151424302,
        "ssp370": 1.706779094364,
    }.items():
        row = by_period[period]
        near(float(row["compound_high_pressure_area_million_km2"]), expected, period)
        near(
            sum(float(row[field]) for field in (
                "low_pressure_area_million_km2",
                "nitrate_pressure_only_area_million_km2",
                "irrigation_pressure_only_area_million_km2",
                "compound_high_pressure_area_million_km2",
            )),
            float(row["positive_Q_computable_area_million_km2"]),
            period + " positive-Q class sum",
        )
        near(
            float(row["positive_Q_computable_area_million_km2"])
            + float(row["nonpositive_Q_limiting_state_area_million_km2"]),
            float(row["total_crop_domain_area_million_km2"]),
            period + " domain sum",
        )
    grouped = defaultdict(list)
    for row in regional:
        grouped[row["period_key"]].append(row)
    assert set(grouped) == set(by_period)
    for period, entries in grouped.items():
        assert len(entries) == 43
        assert len({row["region_code"] for row in entries}) == 43
        for field in (
            "compound_high_pressure_area_million_km2",
            "nonpositive_Q_limiting_state_area_million_km2",
        ):
            near(
                sum(float(row[field]) for row in entries),
                float(by_period[period][field]),
                period + " regional " + field,
            )
    assert len(external) == 3
    assert sorted(int(row["n"]) for row in external) == [44, 617, 2166]
    expected_corr = {44: 0.16710, 617: 0.43605, 2166: 0.47977}
    for row in external:
        n = int(row["n"])
        near(float(row["spearman_r"]), expected_corr[n], "external correlation " + str(n))
    validation = ROOT / "data/validation"
    assert len(rows("data/validation/GRQA_v1.4_nonoutlier_basin_constraint.csv")) == 617
    africa_path = validation / "African_inland_water_reference_basin_table.csv"
    if africa_path.is_file():
        africa = rows("data/validation/African_inland_water_reference_basin_table.csv")
        assert len(africa) == 45
        assert {row["dataset"] for row in africa} == {"African inland waters"}
    supplementary = rows("supplementary/Supplementary_Table_1_cropland_irrigation_status.csv")
    for row in supplementary:
        near(
            float(row["rainfed_million_km2"]) + float(row["irrigated_million_km2"]),
            float(row["total_million_km2"]),
            "Supplementary Table 1 area sum",
        )
    selected = [
        row for row in supplementary
        if row["period"] == "ssp370" and row["classification"] == "central"
        and row["state"] == "Emerging Compound" and row["region_code"] == "GLOBAL"
    ]
    assert len(selected) == 1
    near(float(selected[0]["total_million_km2"]), 0.803512, "emerging Compound")
    near(float(selected[0]["irrigated_million_km2"]), 0.270407, "irrigated emerging")
    print("Five result CSVs and Supplementary Table 1: expected rows and selected values valid")


def check_code():
    scripts = [ROOT / "code/figures/main/Fig{:02d}_plot.py".format(i) for i in range(2, 6)]
    scripts += [ROOT / "code/figures/extended_data/ExtendedData_Fig{:02d}_plot.py".format(i)
                for i in range(1, 11)]
    assert all(path.is_file() for path in scripts), "A plotting entry point is missing"
    assert not (ROOT / "code/figures/main/Fig01_export.py").exists(), (
        "Fig. 1 is authored artwork, not a plotted figure"
    )
    print("Plotting entry points: 14/14 present")


def check_checksums():
    checksum_file = ROOT / "source_sha256.csv"
    assert checksum_file.is_file(), "Source checksum manifest is missing"
    checksums = rows("source_sha256.csv")
    assert len(checksums) >= 80, "Incomplete figure source inventory"
    assert len({row["relative_path"] for row in checksums}) == len(checksums)
    from build_source_manifest import source_files
    assert {row["relative_path"] for row in checksums} == {
        relative for relative, _ in source_files()
    }, "Unlisted or missing source file"
    for row in checksums:
        path = ROOT / row["relative_path"]
        assert path.is_file(), "Missing source file: {}".format(path)
        assert path.stat().st_size == int(row["size_bytes"]), "Size changed: {}".format(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"].lower(), (
            "Checksum changed: {}".format(path)
        )
    print("Source file checksums: {}/{} valid".format(len(checksums), len(checksums)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-generated", action="store_true")
    parser.add_argument("--skip-external", action="store_true")
    args = parser.parse_args()
    check_figures(args.require_generated, args.skip_external)
    check_tables()
    check_code()
    check_checksums()
    print("Package integrity checks passed. Raw-to-figure science was not revalidated.")


if __name__ == "__main__":
    main()
