"""Recreate restricted Fig. 7 inputs from user-downloaded original files.

The Borges workbook and IMAGE-GNM rasters are not redistributed. This script
checks their identities, computes basin matches and ranks, and writes derived
files into data/validation/ for the existing Fig. 7 plotting code.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "data/NPI_IPI_QIRR_strict_positiveQ_period_products_01pct.npz"
AFRICA_SHA256 = "b2acac3c21f7c0ab9f02de2cf61ac5abab0038c6de502b600249fb1509b81da4"
GNM_SHA256 = {
    (1985, "surface_runoff_agri.asc"): "7d20a75251a0193db985bc1ae9844afd71ccfcefbd5025bf07be68c213cf1734",
    (1985, "groundwater_agri.asc"): "691a10b645c807d0d943f72601b9289810b0382012be34761bbf77054e952f6a",
    (1990, "surface_runoff_agri.asc"): "80e19813d5ad0a9d9699afe36225ff5294c0501d5ba8aaa84f4fd3c388ebaa6e",
    (1990, "groundwater_agri.asc"): "7af6618e4fd178cead9fd33e76ab3e03bc1288190a17c481f1484b4c1e884fe7",
    (1995, "surface_runoff_agri.asc"): "b9561cff78598574beb7197ada37fe4d5b86d3526133a0344eb9c66522c841f6",
    (1995, "groundwater_agri.asc"): "f8f3c98667abd19f42791170c60bff6c8c583869be89fb05754a8fbe18637a22",
    (2000, "surface_runoff_agri.asc"): "c6dbed8e59aa892614b80b8c095dad78b770a841aba9ff9d40c19e328c34a4f8",
    (2000, "groundwater_agri.asc"): "7237f05ad8a7a463d0a50f4b120ae5d9ccb1cfbac68c1ce3f1f903124b102f2b",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked(path, expected):
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError("Input checksum differs from documented source: {}".format(path))
    return path


def africa_basin_table(workbook, model):
    raw = pd.read_excel(workbook, sheet_name=0)
    date = pd.to_datetime(raw["Timestamp (GMT)"], errors="coerce")
    lat_col = next(c for c in raw if str(c).startswith("Latitude"))
    lon_col = next(c for c in raw if str(c).startswith("Longitude"))
    no3_col = next(c for c in raw if str(c).startswith("NO3-"))
    raw["date"] = date
    raw["Latitude"] = pd.to_numeric(raw[lat_col], errors="coerce")
    raw["Longitude"] = pd.to_numeric(raw[lon_col], errors="coerce")
    raw["no3_umol_L"] = pd.to_numeric(raw[no3_col], errors="coerce")
    raw = raw[
        raw["date"].dt.year.between(1985, 2014)
        & np.isfinite(raw["Latitude"])
        & np.isfinite(raw["Longitude"])
        & np.isfinite(raw["no3_umol_L"])
        & (raw["no3_umol_L"] >= 0)
    ].copy()
    raw["observed_no3_mgN_L"] = raw["no3_umol_L"] * 0.0140067
    raw["lat_key"] = raw["Latitude"].round(5)
    raw["lon_key"] = raw["Longitude"].round(5)
    station = raw.groupby(["lat_key", "lon_key"], as_index=False).agg(
        Latitude=("Latitude", "median"),
        Longitude=("Longitude", "median"),
        observed_no3_mgN_L=("observed_no3_mgN_L", "median"),
    )
    station["dataset"] = "African inland waters"
    station["station_id"] = ["AFR_{:04d}".format(i) for i in range(1, len(station) + 1)]

    lon = np.asarray(model["lon"], dtype=float)
    lat = np.asarray(model["lat"], dtype=float)
    code_grid = np.asarray(model["basin_code_grid"])
    basin_ids = np.asarray(model["basin_ids"], dtype=np.int64)
    crop_area = np.asarray(model["basin_crop_area_historical_m2"], dtype=float)
    npi = np.asarray(model["NPI_IRR_historical_basin"], dtype=float)
    ix = np.abs(lon[None, :] - np.mod(station["Longitude"].to_numpy(), 360.0)[:, None]).argmin(axis=1)
    iy = np.abs(lat[None, :] - station["Latitude"].to_numpy()[:, None]).argmin(axis=1)
    codes = code_grid[iy, ix].astype(np.int64)
    valid_code = (codes >= 0) & (codes < basin_ids.size)
    safe_code = np.clip(codes, 0, basin_ids.size - 1)
    station["basin_code"] = np.where(valid_code, codes, -1)
    station["HYBAS_ID"] = np.where(valid_code, basin_ids[safe_code], -1)
    station["model_npi_mgN_L"] = np.where(valid_code, npi[safe_code], np.nan)
    station["crop_area_m2"] = np.where(valid_code, crop_area[safe_code], np.nan)
    matched = station[
        valid_code
        & np.isfinite(station["model_npi_mgN_L"])
        & np.isfinite(station["crop_area_m2"])
        & (station["crop_area_m2"] > 0)
    ]
    return matched.groupby(["dataset", "basin_code", "HYBAS_ID"], as_index=False).agg(
        observed_no3_mgN_L=("observed_no3_mgN_L", "median"),
        model_npi_mgN_L=("model_npi_mgN_L", "first"),
        station_count=("station_id", "nunique"),
        Latitude=("Latitude", "median"),
        Longitude=("Longitude", "median"),
    )


def read_ascii(path):
    header = {}
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for _ in range(6):
            name, value = stream.readline().split()[:2]
            header[name.lower()] = float(value)
    values = np.loadtxt(path, skiprows=6, dtype=np.float64)
    values[values == header.get("nodata_value", -9999.0)] = np.nan
    return values, header


def percentile_rank(values, valid):
    out = np.full(values.shape, np.nan, dtype=float)
    out[valid] = pd.Series(values[valid]).rank(method="average", pct=True).to_numpy()
    return out


def gnm_basin_ranks(gnm_root, model):
    snapshots = []
    header = None
    for year in (1985, 1990, 1995, 2000):
        surface, h1 = read_ascii(checked(gnm_root / str(year) / "surface_runoff_agri.asc", GNM_SHA256[year, "surface_runoff_agri.asc"]))
        groundwater, h2 = read_ascii(checked(gnm_root / str(year) / "groundwater_agri.asc", GNM_SHA256[year, "groundwater_agri.asc"]))
        if surface.shape != groundwater.shape or h1 != h2:
            raise ValueError("Unaligned IMAGE-GNM rasters for {}".format(year))
        snapshots.append(surface + groundwater)
        header = h1
    stack = np.stack(snapshots, axis=0)
    count = np.sum(np.isfinite(stack), axis=0)
    load = np.divide(
        np.nansum(stack, axis=0), count,
        out=np.full(stack.shape[1:], np.nan, dtype=float), where=count > 0,
    )
    cell = float(header["cellsize"])
    gnm_lon = float(header["xllcorner"]) + (np.arange(load.shape[1]) + 0.5) * cell
    gnm_lat = (float(header["yllcorner"]) + (np.arange(load.shape[0]) + 0.5) * cell)[::-1]

    model_lon = np.asarray(model["lon"], dtype=float)
    model_lat = np.asarray(model["lat"], dtype=float)
    order = np.argsort(((model_lon + 180.0) % 360.0) - 180.0)
    sorted_lon = ((model_lon[order] + 180.0) % 360.0) - 180.0
    code_grid = np.asarray(model["basin_code_grid"])[:, order]
    ix = np.abs(sorted_lon[None, :] - gnm_lon[:, None]).argmin(axis=1)
    iy = np.abs(model_lat[None, :] - gnm_lat[:, None]).argmin(axis=1)
    mapped = code_grid[np.ix_(iy, ix)]
    n_basins = np.asarray(model["basin_ids"]).size
    valid = np.isfinite(load) & np.isfinite(mapped) & (mapped >= 0) & (mapped < n_basins)
    gnm_load = np.bincount(mapped[valid].astype(np.int64), weights=load[valid], minlength=n_basins).astype(float)
    gnm_load[gnm_load <= 0] = np.nan

    area_km2 = np.asarray(model["basin_area_m2"], dtype=float) / 1e6
    model_load = np.asarray(model["Nload_sum_IRR_historical_gN"], dtype=float) / (3 * 30 * 1000.0)
    model_yield = np.divide(model_load, area_km2, out=np.full(n_basins, np.nan), where=area_km2 > 0)
    gnm_yield = np.divide(gnm_load, area_km2, out=np.full(n_basins, np.nan), where=area_km2 > 0)
    common = (
        (np.asarray(model["basin_crop_area_historical_m2"], dtype=float) > 0)
        & np.isfinite(model_yield) & (model_yield > 0)
        & np.isfinite(gnm_yield) & (gnm_yield > 0)
    )
    return percentile_rank(model_yield, common), percentile_rank(gnm_yield, common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--africa-xlsx", type=Path, required=True)
    parser.add_argument("--gnm-root", type=Path, required=True, help="Path to GNM_database/nitrogen")
    args = parser.parse_args()
    checked(args.africa_xlsx, AFRICA_SHA256)
    output = ROOT / "data/validation"
    with np.load(MODEL, allow_pickle=False) as model:
        grqa_path = output / "GRQA_v1.4_nonoutlier_basin_constraint.csv"
        grqa = pd.read_csv(grqa_path)
        codes = grqa["basin_code"].to_numpy(dtype=int)
        npi = np.asarray(model["NPI_IRR_historical_basin"], dtype=float)
        if len(grqa) != 617 or (codes < 0).any() or (codes >= len(npi)).any():
            raise AssertionError("GRQA match does not align with the current model product")
        current_npi = npi[codes]
        update_grqa = not np.allclose(
            grqa["model_npi_mgN_L"].to_numpy(dtype=float), current_npi,
            rtol=0.0, atol=1e-12, equal_nan=True,
        )
        if update_grqa:
            grqa["model_npi_mgN_L"] = current_npi
        africa = africa_basin_table(args.africa_xlsx, model)
        model_rank, gnm_rank = gnm_basin_ranks(args.gnm_root, model)
    if len(africa) != 45 or int(np.isfinite(gnm_rank).sum()) != 2166:
        raise AssertionError("External source matching differs from documented analysis")
    if update_grqa:
        grqa.to_csv(grqa_path, index=False)
    africa.to_csv(output / "African_inland_water_reference_basin_table.csv", index=False)
    np.savez_compressed(output / "ED07_benchmark_percentile_ranks.npz", model_rank=model_rank, gnm_rank=gnm_rank)
    print("Prepared African basin matches and IMAGE-GNM ranks in {}".format(output))


if __name__ == "__main__":
    main()
