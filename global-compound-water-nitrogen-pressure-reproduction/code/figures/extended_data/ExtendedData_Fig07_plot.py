# -*- coding: utf-8 -*-
"""Build Extended Data Fig. 7 using GRQA v1.4 observations."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_helpers as core


STRICT_ROOT = Path(__file__).resolve().parents[3]
core.MODEL_NPZ = (
    STRICT_ROOT
    / "data"
    / "NPI_IPI_QIRR_strict_positiveQ_period_products_01pct.npz"
)
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = STRICT_ROOT
OUT = PACKAGE_ROOT / "figures" / "extended_data"
AUDIT = PACKAGE_ROOT / "audit"
DATA = PACKAGE_ROOT / "data" / "validation"
OUT.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)

STEM = (
    OUT
    / "ExtendedData_Fig07_independent_nitrogen_benchmarking"
)
GRQA_BASIN_CSV = DATA / "GRQA_v1.4_nonoutlier_basin_constraint.csv"
REFERENCE_OBSERVATION_CSV = (
    DATA / "African_inland_water_reference_basin_table.csv"
)

TEXT = "#20282D"
MUTED = "#68737A"
GRID = "#E2E7EA"
COAST = "#6F797F"
BORDER = "#C4C9CC"
SUPPORT = "#ECEFF1"

TEAL_COLORS = ["#F0F6F5", "#D5E8E5", "#A8CFCA", "#72B2AD", "#397F7B"]
WARM_COLORS = ["#FBF7EC", "#F2E4C7", "#DEC16F", "#C38A4B", "#8F584C"]
AFRICA_ACCENT = "#82749D"

RANK_BOUNDS = np.linspace(0.0, 1.0, 6)
TEAL_CMAP = mpl.colors.ListedColormap(TEAL_COLORS)
WARM_CMAP = mpl.colors.ListedColormap(WARM_COLORS)
RANK_NORM = mpl.colors.BoundaryNorm(RANK_BOUNDS, 5)


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.2,
        "axes.titlesize": 7.6,
        "axes.labelsize": 6.7,
        "xtick.labelsize": 6.1,
        "ytick.labelsize": 6.1,
        "legend.fontsize": 4.8,
        "axes.linewidth": 0.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def panel_heading(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        0.012,
        0.985,
        label,
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        color=TEXT,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.8),
        zorder=30,
    )
    ax.text(
        0.0,
        1.012,
        title,
        transform=ax.transAxes,
        fontsize=7.6,
        fontweight="normal",
        color=TEXT,
        ha="left",
        va="bottom",
        zorder=30,
    )


def setup_map(ax: plt.Axes, cache) -> None:
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    if cache is not None:
        core.draw_segments(
            ax,
            cache["border_points"],
            cache["border_ranges"],
            BORDER,
            0.18,
            6,
        )
        core.draw_segments(
            ax,
            cache["coast_points"],
            cache["coast_ranges"],
            COAST,
            0.36,
            7,
        )
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.22)


def draw_basin_rank(
    ax: plt.Axes,
    values: np.ndarray,
    code_grid: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    cmap: mpl.colors.ListedColormap,
) -> mpl.collections.QuadMesh:
    lon180, grid = core.basin_to_grid(values, code_grid, lon)
    return ax.pcolormesh(
        lon180,
        lat,
        np.ma.masked_invalid(grid),
        cmap=cmap,
        norm=RANK_NORM,
        shading="auto",
        rasterized=True,
        zorder=2,
    )


def draw_support_domain(
    ax: plt.Axes,
    support: np.ndarray,
    code_grid: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> None:
    values = np.where(support, 1.0, np.nan)
    lon180, grid = core.basin_to_grid(values, code_grid, lon)
    ax.pcolormesh(
        lon180,
        lat,
        np.ma.masked_invalid(grid),
        cmap=mpl.colors.ListedColormap([SUPPORT]),
        vmin=0,
        vmax=1,
        shading="auto",
        rasterized=True,
        zorder=1,
    )


def safe_spearman(x, y) -> float:
    """Compute Spearman correlation without invoking multithreaded BLAS."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 3:
        return float("nan")
    x_rank = pd.Series(x_arr[valid]).rank(method="average").to_numpy(dtype=float).copy()
    y_rank = pd.Series(y_arr[valid]).rank(method="average").to_numpy(dtype=float).copy()
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = float(np.sqrt(np.dot(x_rank, x_rank) * np.dot(y_rank, y_rank)))
    if denominator == 0:
        return float("nan")
    return float(np.dot(x_rank, y_rank) / denominator)


def paired_ranks(frame):
    valid = (
        np.isfinite(frame["model_npi_mgN_L"])
        & np.isfinite(frame["observed_no3_mgN_L"])
        & (frame["model_npi_mgN_L"] >= 0)
        & (frame["observed_no3_mgN_L"] > 0)
    )
    paired = frame.loc[valid].copy()
    ones = np.ones(len(paired), dtype=bool)
    paired["npi_rank"] = core.percentile_rank(
        paired["model_npi_mgN_L"].to_numpy(dtype=float), ones
    )
    paired["observed_rank"] = core.percentile_rank(
        paired["observed_no3_mgN_L"].to_numpy(dtype=float), ones
    )
    rho = safe_spearman(
        paired["model_npi_mgN_L"], paired["observed_no3_mgN_L"]
    )
    return paired, float(rho)


def draw_rank_relationship(
    ax: plt.Axes,
    frame,
    rho: float,
    label: str,
    letter: str,
    color: str,
    marker: str,
) -> None:
    x = frame["npi_rank"].to_numpy(dtype=float)
    y = frame["observed_rank"].to_numpy(dtype=float)
    ax.scatter(
        x,
        y,
        s=18 if marker == "o" else 22,
        marker=marker,
        facecolor=color,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.48,
        rasterized=True,
        zorder=3,
    )

    bin_edges = np.linspace(0, 1, 11)
    bin_index = np.digitize(x, bin_edges[1:-1], right=False)
    med_x, med_y = [], []
    for index in range(10):
        selected = bin_index == index
        if np.any(selected):
            med_x.append(float(np.median(x[selected])))
            med_y.append(float(np.median(y[selected])))
    ax.plot(
        med_x,
        med_y,
        color=color,
        linewidth=1.55,
        marker="o",
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.8,
        zorder=4,
    )
    ax.plot([0, 1], [0, 1], color="#AEB5BA", linewidth=0.7, linestyle=(0, (3, 2)))
    panel_heading(ax, letter, label)
    ax.text(
        0.03,
        0.95,
        f"$n$ = {len(frame):,}\n$r_s$ = {rho:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color=TEXT,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ticks = np.linspace(0, 1, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Historical NPI percentile rank")
    ax.set_ylabel("Observed nitrate-N percentile rank")
    ax.set_box_aspect(0.56)
    ax.grid(color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#454B50")
    ax.spines["bottom"].set_color("#454B50")


def centered_colorbar_axis(fig: plt.Figure, slot) -> plt.Axes:
    sub = slot.subgridspec(1, 3, width_ratios=[0.19, 0.62, 0.19])
    return fig.add_subplot(sub[0, 1])


def build_figure(
    z,
    grqa_basin,
    africa_basin,
    model_rank,
    gnm_rank,
):
    cache = core.load_line_cache()
    lon = np.asarray(z["lon"], dtype=float)
    lat = np.asarray(z["lat"], dtype=float)
    code_grid = np.asarray(z["basin_code_grid"])
    npi = np.asarray(z["NPI_IRR_historical_basin"], dtype=float)
    crop = np.asarray(z["basin_crop_area_historical_m2"], dtype=float) > 0
    npi[~crop] = np.nan
    npi_rank = core.percentile_rank(npi, crop & np.isfinite(npi))

    grqa_plot = grqa_basin[
        np.isfinite(grqa_basin["observed_no3_mgN_L"])
        & (grqa_basin["observed_no3_mgN_L"] > 0)
        & np.isfinite(grqa_basin["model_npi_mgN_L"])
    ].copy()
    africa_plot = africa_basin[
        np.isfinite(africa_basin["observed_no3_mgN_L"])
        & (africa_basin["observed_no3_mgN_L"] > 0)
        & np.isfinite(africa_basin["model_npi_mgN_L"])
    ].copy()
    observed_values = np.concatenate(
        [
            grqa_plot["observed_no3_mgN_L"].to_numpy(dtype=float),
            africa_plot["observed_no3_mgN_L"].to_numpy(dtype=float),
        ]
    )
    observed_ranks = core.percentile_rank(
        observed_values, np.ones(observed_values.shape, dtype=bool)
    )
    grqa_plot["joint_observed_rank"] = observed_ranks[: len(grqa_plot)]
    africa_plot["joint_observed_rank"] = observed_ranks[len(grqa_plot) :]
    grqa_plot["rank_bin"] = np.digitize(
        grqa_plot["joint_observed_rank"], RANK_BOUNDS[1:-1], right=False
    )
    africa_plot["rank_bin"] = np.digitize(
        africa_plot["joint_observed_rank"], RANK_BOUNDS[1:-1], right=False
    )

    grqa_pairs, grqa_rho = paired_ranks(grqa_basin)
    africa_pairs, africa_rho = paired_ranks(africa_basin)

    panel_b_grqa = grqa_plot.copy()
    panel_b_grqa["figure_source"] = "GRQA v1.4"
    panel_b_africa = africa_plot.copy()
    panel_b_africa["figure_source"] = "African inland waters"
    pd.concat(
        [panel_b_grqa, panel_b_africa],
        ignore_index=True,
        sort=False,
    ).to_csv(DATA / "ExtendedData_Fig07_panel_b_observations_GRQA_v1.4.csv", index=False)

    fig = plt.figure(figsize=(7.20, 4.20), dpi=600, facecolor="white")
    gs = fig.add_gridspec(
        5,
        2,
        height_ratios=[1.0, 0.042, 0.10, 1.0, 0.042],
        left=0.025,
        right=0.985,
        top=0.970,
        bottom=0.060,
        wspace=0.060,
        hspace=0.000,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    cax_ab = centered_colorbar_axis(fig, gs[1, :])
    ax_c = fig.add_subplot(gs[3, 0])
    ax_d = fig.add_subplot(gs[3, 1])
    cax_cd = centered_colorbar_axis(fig, gs[4, :])

    warm_mesh = draw_basin_rank(
        ax_a, npi_rank, code_grid, lon, lat, WARM_CMAP
    )
    setup_map(ax_a, cache)
    panel_heading(ax_a, "a", "Historical NPI percentile rank")

    support = np.zeros(len(npi), dtype=bool)
    support[grqa_plot["basin_code"].to_numpy(dtype=int)] = True
    support[africa_plot["basin_code"].to_numpy(dtype=int)] = True
    draw_support_domain(ax_b, support, code_grid, lon, lat)
    for rank_bin, color in enumerate(WARM_COLORS):
        for frame, marker, size in [
            (grqa_plot, "o", 14.0),
            (africa_plot, "D", 18.0),
        ]:
            selected = frame[frame["rank_bin"] == rank_bin]
            if selected.empty:
                continue
            ax_b.scatter(
                ((selected["Longitude"].to_numpy() + 180.0) % 360.0) - 180.0,
                selected["Latitude"],
                s=size,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.40,
                alpha=0.82,
                rasterized=True,
                zorder=9 + rank_bin,
            )
    setup_map(ax_b, cache)
    panel_heading(ax_b, "b", "Observed nitrate-N percentile rank")
    ax_b.text(
        0.018,
        0.205,
        f"GRQA: $r_s$ = {grqa_rho:.2f}; $n$ = {len(grqa_pairs):,}\n"
        f"Africa: $r_s$ = {africa_rho:.2f}; "
        f"$n$ = {len(africa_pairs):,}",
        transform=ax_b.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.8,
        linespacing=1.25,
        color=TEXT,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.96, pad=1.0),
        zorder=20,
    )
    ax_b.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=5.5,
                markerfacecolor=WARM_COLORS[3],
                markeredgecolor="white",
                markeredgewidth=0.5,
                label="GRQA v1.4",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="",
                markersize=5.2,
                markerfacecolor=WARM_COLORS[3],
                markeredgecolor="white",
                markeredgewidth=0.5,
                label="African inland NO$_3$",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.015, 0.015),
        frameon=False,
        borderaxespad=0,
        handletextpad=0.42,
        labelspacing=0.42,
    )

    teal_mesh = draw_basin_rank(
        ax_c, model_rank, code_grid, lon, lat, TEAL_CMAP
    )
    setup_map(ax_c, cache)
    panel_heading(ax_c, "c", "Modelled nitrate-N loss percentile rank")

    draw_basin_rank(ax_d, gnm_rank, code_grid, lon, lat, TEAL_CMAP)
    setup_map(ax_d, cache)
    panel_heading(
        ax_d,
        "d",
        "IMAGE-GNM agricultural diffuse N-load percentile rank",
    )
    common = np.isfinite(model_rank) & np.isfinite(gnm_rank)
    gnm_rho = safe_spearman(model_rank[common], gnm_rank[common])
    ax_d.text(
        0.018,
        0.065,
        f"$n$ = {int(common.sum()):,}\n$r_s$ = {gnm_rho:.2f}",
        transform=ax_d.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.8,
        linespacing=1.12,
        color=TEXT,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.6),
        zorder=20,
    )

    cb_ab = fig.colorbar(
        warm_mesh,
        cax=cax_ab,
        orientation="horizontal",
        ticks=RANK_BOUNDS,
        boundaries=RANK_BOUNDS,
        spacing="proportional",
    )
    cb_ab.set_label("Percentile rank", labelpad=0.9, fontsize=6.7)
    cb_ab.ax.tick_params(length=1.8, width=0.35, pad=0.8, labelsize=6.1)
    cb_ab.ax.minorticks_off()

    cb_cd = fig.colorbar(
        teal_mesh,
        cax=cax_cd,
        orientation="horizontal",
        ticks=RANK_BOUNDS,
        boundaries=RANK_BOUNDS,
        spacing="proportional",
    )
    cb_cd.set_label("Percentile rank", labelpad=0.9, fontsize=6.7)
    cb_cd.ax.tick_params(length=1.8, width=0.35, pad=0.8, labelsize=6.1)
    cb_cd.ax.minorticks_off()

    # Equal-aspect maps leave internal whitespace inside their GridSpec cells,
    # so hspace does not reliably control the visible map-colorbar gap. Set the
    # gap explicitly from the map bottom to the colorbar top instead.
    fig.canvas.draw()
    colorbar_map_gap = 0.016

    def place_colorbar_below(cax, map_axes):
        cbar_pos = cax.get_position()
        map_bottom = min(ax.get_position().y0 for ax in map_axes)
        cax.set_position(
            [
                cbar_pos.x0,
                map_bottom - colorbar_map_gap - cbar_pos.height,
                cbar_pos.width,
                cbar_pos.height,
            ]
        )

    place_colorbar_below(cax_ab, [ax_a, ax_b])
    place_colorbar_below(cax_cd, [ax_c, ax_d])

    fig.savefig(STEM.with_suffix(".png"), dpi=600)
    fig.savefig(STEM.with_suffix(".pdf"))
    fig.savefig(STEM.with_suffix(".svg"))
    plt.close(fig)

    summary = {
        "framework": "strict positive-Q",
        "panel_order": [
            "Historical NPI percentile rank",
            "Observed nitrate-N percentile rank",
            "Modelled nitrate-N loss percentile rank",
            "IMAGE-GNM agricultural diffuse N-load percentile rank",
        ],
        "image_gnm": {"n": int(common.sum()), "spearman_rs": gnm_rho},
        "grqa_v1_4": {
            "selection": (
                "1985-2014; at least one valid year; official IQR outliers excluded"
            ),
            "n": int(len(grqa_pairs)),
            "spearman_rs": grqa_rho,
        },
        "african_inland_water": {
            "n": int(len(africa_pairs)),
            "spearman_rs": africa_rho,
        },
        "percentile_note": (
            "Percentile ranks are computed separately within each valid data "
            "domain and represent relative spatial ordering, not absolute magnitude."
        ),
    }
    (AUDIT / "EXTENDED_DATA_FIG07_GRQA_v1.4_AUDIT.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "comparison": "Historical NPI vs GRQA v1.4",
                "selection": (
                    "1985-2014; >=1 valid year; official IQR outliers excluded"
                ),
                "n": int(len(grqa_pairs)),
                "spearman_rs": grqa_rho,
            },
            {
                "comparison": "Historical NPI vs African inland waters",
                "selection": "Existing independently processed comparison",
                "n": int(len(africa_pairs)),
                "spearman_rs": africa_rho,
            },
            {
                "comparison": "Modelled nitrate-N loss vs IMAGE-GNM",
                "selection": "Common positive agricultural domain",
                "n": int(common.sum()),
                "spearman_rs": gnm_rho,
            },
        ]
    ).to_csv(DATA / "ExtendedData_Fig07_updated_statistics_GRQA_v1.4.csv", index=False)
    return summary


def main() -> None:
    z = np.load(core.MODEL_NPZ, allow_pickle=False)
    lon = np.asarray(z["lon"], dtype=float)
    lat = np.asarray(z["lat"], dtype=float)
    code_grid = np.asarray(z["basin_code_grid"])
    basin_ids = np.asarray(z["basin_ids"], dtype=np.int64)
    basin_area = np.asarray(z["basin_area_m2"], dtype=float)
    crop_area = np.asarray(z["basin_crop_area_historical_m2"], dtype=float)
    npi = np.asarray(z["NPI_IRR_historical_basin"], dtype=float)

    grqa_basin = pd.read_csv(GRQA_BASIN_CSV)
    reference_observations = pd.read_csv(REFERENCE_OBSERVATION_CSV)
    africa_basin = reference_observations.loc[
        reference_observations["dataset"].eq("African inland waters")
    ].copy()

    model_load_kgN_yr = (
        np.asarray(z["Nload_sum_IRR_historical_gN"], dtype=float)
        / (core.N_MEMBERS * core.N_YEARS * 1000.0)
    )
    basin_area_km2 = basin_area / 1.0e6
    model_yield = np.divide(
        model_load_kgN_yr,
        basin_area_km2,
        out=np.full_like(model_load_kgN_yr, np.nan),
        where=basin_area_km2 > 0,
    )

    benchmark = np.load(DATA / "ED07_benchmark_percentile_ranks.npz", allow_pickle=False)
    model_rank = np.asarray(benchmark["model_rank"], dtype=float)
    gnm_rank = np.asarray(benchmark["gnm_rank"], dtype=float)

    summary = build_figure(
        z, grqa_basin, africa_basin, model_rank, gnm_rank
    )
    print(json.dumps(summary, indent=2))
    print(STEM.with_suffix(".png"))


if __name__ == "__main__":
    main()
