"""Recompute reported spatial rank correlations from packaged matched values."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
VALIDATION = ROOT / "data/validation"
SUMMARY = ROOT / "repository_data/Repository_Data_05_external_evaluation_summary.csv"


def rank_correlation(left, right):
    x = pd.Series(np.asarray(left, dtype=float)).rank(method="average")
    y = pd.Series(np.asarray(right, dtype=float)).rank(method="average")
    return float(x.corr(y))


def observation_result(filename):
    frame = pd.read_csv(VALIDATION / filename)
    observed = pd.to_numeric(frame["observed_no3_mgN_L"], errors="coerce")
    model = pd.to_numeric(frame["model_npi_mgN_L"], errors="coerce")
    valid = np.isfinite(observed) & np.isfinite(model) & (observed > 0) & (model >= 0)
    return int(valid.sum()), rank_correlation(model[valid], observed[valid])


def main():
    needed = [
        VALIDATION / "GRQA_v1.4_nonoutlier_basin_constraint.csv",
        VALIDATION / "African_inland_water_reference_basin_table.csv",
        VALIDATION / "ED07_benchmark_percentile_ranks.npz",
    ]
    missing = [path.name for path in needed if not path.is_file()]
    if missing:
        raise SystemExit("Prepare external benchmark inputs before this check: {}".format(", ".join(missing)))
    summary = pd.read_csv(SUMMARY).set_index("n")
    cases = [
        observation_result("GRQA_v1.4_nonoutlier_basin_constraint.csv"),
        observation_result("African_inland_water_reference_basin_table.csv"),
    ]
    with np.load(VALIDATION / "ED07_benchmark_percentile_ranks.npz") as ranks:
        x = ranks["model_rank"]
        y = ranks["gnm_rank"]
        valid = np.isfinite(x) & np.isfinite(y)
        cases.append((int(valid.sum()), rank_correlation(x[valid], y[valid])))
    if sorted(n for n, _ in cases) != [44, 617, 2166]:
        raise AssertionError("Unexpected matched sample sizes")
    for n, correlation in cases:
        expected = float(summary.loc[n, "spearman_r"])
        if abs(correlation - expected) > 0.00002:
            raise AssertionError("Correlation mismatch for n={}: {} != {}".format(n, correlation, expected))
        print("n={}: Spearman r_s={:.8f}; reported={:.5f}".format(n, correlation, expected))
    print("Three external benchmark statistics reproduced from matched data")


if __name__ == "__main__":
    main()
