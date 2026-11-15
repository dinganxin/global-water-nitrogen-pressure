#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extended Data Fig. 8: member agreement and cropland area (visual refinement)."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    import shapefile
except ImportError:
    shapefile = None


ROOT = Path(__file__).resolve().parents[3]
PRODUCT = ROOT / "data" / "NPI_IPI_QIRR_strict_positiveQ_period_products_01pct.npz"
FIG_DIR = ROOT / "figures" / "extended_data"
TABLE_DIR = ROOT / "tables" / "extended_data"
AUDIT_DIR = ROOT / "audit"
COUNTRY_ROOT = ROOT / "data" / "geospatial" / "natural_earth_admin0"

PERIODS = ("historical", "ssp126", "ssp370")
PERIOD_LABELS = ("Historical", "SSP1-2.6", "SSP3-7.0")
PERIOD_COLORS = ("#7D8794", "#4C91BF", "#C94F59")
MEMBER_LABELS = ("IRR01", "IRR02", "IRR03")
MEMBER_OFFSETS = np.array([-0.11, 0.0, 0.11])

TEXT = "#22282D"
MUTED = "#66727B"
GRID = "#E7EBED"
COUNTRY = "#AAB2B7"
CONNECTOR = "#E1E5E8"

COMPOUND_COLORS = ("#E4E7E8", "#EED5D1", "#D4938C", "#A65359")
NONPOSITIVE_COLORS = ("#E4E7E8", "#DED7E7", "#AC9DC0", "#6E618E")
COUNT_BOUNDS = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.6,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def find_country_shapefile() -> Path | None:
    if shapefile is None or not COUNTRY_ROOT.exists():
        return None
    candidates = list(COUNTRY_ROOT.rglob("*.shp"))
    if not candidates:
        return None
    preferred = [
        path
        for path in candidates
        if "admin_0_countries" in path.stem.lower()
        or "ne_110m_admin_0" in path.stem.lower()
    ]
    return preferred[0] if preferred else candidates[0]


def load_country_segments() -> list[np.ndarray]:
    shp_path = find_country_shapefile()
    if shp_path is None:
        return []
    reader = shapefile.Reader(str(shp_path))
    segments: list[np.ndarray] = []
    for shape in reader.shapes():
        points = np.asarray(shape.points, dtype=float)
        if points.size == 0:
            continue
        parts = list(shape.parts) + [len(points)]
        for start, end in zip(parts[:-1], parts[1:]):
            segment = points[start:end]
            if segment.shape[0] >= 2:
                segments.append(segment)
    return segments


def add_country_boundaries(ax: plt.Axes, segments: list[np.ndarray]) -> None:
    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors=COUNTRY,
                linewidths=0.24,
                alpha=0.78,
                zorder=3,
            )
        )


def basin_values_to_grid(values: np.ndarray, code_grid: np.ndarray) -> np.ndarray:
    grid = np.full(code_grid.shape, np.nan, dtype=float)
    valid = code_grid >= 0
    grid[valid] = np.asarray(values, dtype=float)[code_grid[valid]]
    return grid


def panel_heading(
    ax: plt.Axes,
    letter: str,
    title: str,
    title_size: float = 8.4,
) -> None:
    ax.text(
        -0.005,
        1.045,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.3,
        fontweight="bold",
        color=TEXT,
    )
    ax.annotate(
        title,
        xy=(-0.005, 1.045),
        xycoords="axes fraction",
        xytext=(16, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=title_size,
        color=TEXT,
    )


def draw_agreement_map(
    ax: plt.Axes,
    member_count: np.ndarray,
    evaluated_mask: np.ndarray,
    code_grid: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    colors: tuple[str, str, str, str],
    boundaries: list[np.ndarray],
) -> mpl.collections.QuadMesh:
    display = np.where(evaluated_mask, member_count, np.nan)
    grid = basin_values_to_grid(display, code_grid)
    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    norm = BoundaryNorm(COUNT_BOUNDS, cmap.N)
    mesh = ax.pcolormesh(
        lon,
        lat,
        grid,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=False,
        edgecolors="none",
        antialiased=False,
        zorder=1,
    )
    add_country_boundaries(ax, boundaries)
    ax.set_xlim(-180, 180)
    ax.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def add_discrete_colorbar(
    fig: plt.Figure,
    cax: plt.Axes,
    mesh: mpl.collections.QuadMesh,
) -> None:
    colorbar = fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
        boundaries=COUNT_BOUNDS,
        ticks=[0, 1, 2, 3],
        spacing="uniform",
    )
    colorbar.set_label("Number of members", labelpad=1.1, fontsize=7.0)
    colorbar.ax.tick_params(length=1.8, width=0.55, pad=1.1, labelsize=6.8)
    colorbar.outline.set_edgecolor("#777F84")
    colorbar.outline.set_linewidth(0.5)
    colorbar.solids.set_edgecolor("white")
    colorbar.solids.set_linewidth(0.35)


def style_member_axis(ax: plt.Axes) -> None:
    ax.set_xlim(-0.36, 2.36)
    ax.set_ylim(0.0, 2.05)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticks(range(3))
    ax.set_xticklabels(PERIOD_LABELS)
    ax.set_ylabel("Cropland area (million km²)")
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#697176")
    ax.spines["bottom"].set_color("#697176")
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="both", width=0.55, length=2.5, color="#697176")


def draw_member_panel(
    ax: plt.Axes,
    values_by_period: list[np.ndarray],
) -> None:
    for member_index, offset in enumerate(MEMBER_OFFSETS):
        member_values = [
            float(values_by_period[period_index][member_index])
            for period_index in range(len(PERIODS))
        ]
        ax.plot(
            np.arange(len(PERIODS), dtype=float) + offset,
            member_values,
            color=CONNECTOR,
            linewidth=0.55,
            alpha=0.85,
            zorder=1,
        )

    for period_index, (values, color) in enumerate(
        zip(values_by_period, PERIOD_COLORS)
    ):
        mean_value = float(np.mean(values))
        ax.plot(
            [period_index - 0.22, period_index + 0.22],
            [mean_value, mean_value],
            color=color,
            linewidth=2.2,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.scatter(
            period_index + MEMBER_OFFSETS,
            values,
            s=20,
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=3,
        )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    # Explicit margins are already set by GridSpec. Avoid bbox_inches="tight",
    # which is unstable in the Windows Matplotlib build used for this project.
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    product = np.load(PRODUCT, allow_pickle=False)
    boundaries = load_country_segments()

    code_grid = np.asarray(product["basin_code_grid"], dtype=int)
    lon = np.asarray(product["lon"], dtype=float)
    lat = np.asarray(product["lat"], dtype=float)
    crop_ssp3 = np.asarray(product["basin_crop_area_ssp370_m2"], dtype=float) > 0

    class_members_ssp3 = np.asarray(
        product["class_member_IRR_ssp370_basin"], dtype=np.int8
    )
    q_members_ssp3 = np.asarray(
        product["QIRR_nonpositive_member_ssp370_basin"], dtype=np.int8
    )
    if class_members_ssp3.shape[0] != 3 or q_members_ssp3.shape[0] != 3:
        raise ValueError("Expected exactly three independently processed members.")

    compound_count = np.sum(class_members_ssp3 == 3, axis=0).astype(np.int8)
    nonpositive_count = np.sum(q_members_ssp3 == 1, axis=0).astype(np.int8)
    if not set(np.unique(compound_count)).issubset({0, 1, 2, 3}):
        raise AssertionError("Strict-compound agreement is not restricted to 0-3.")
    if not set(np.unique(nonpositive_count)).issubset({0, 1, 2, 3}):
        raise AssertionError("Q <= 0 agreement is not restricted to 0-3.")

    # A zero in panel a means all three members were numerically evaluable and
    # none classified the basin as strict compound. Partial/non-evaluable basins
    # remain white and are diagnosed separately in panel c.
    compound_evaluated = crop_ssp3 & np.all(class_members_ssp3 >= 0, axis=0)
    q_state_evaluated = crop_ssp3

    member_rows: list[dict[str, object]] = []
    compound_values: list[np.ndarray] = []
    nonpositive_values: list[np.ndarray] = []
    for period, period_label in zip(PERIODS, PERIOD_LABELS):
        crop_area = np.asarray(product[f"basin_crop_area_{period}_m2"], dtype=float)
        member_class = np.asarray(
            product[f"class_member_IRR_{period}_basin"], dtype=np.int8
        )
        member_q = np.asarray(
            product[f"QIRR_nonpositive_member_{period}_basin"], dtype=np.int8
        )
        compound_period = np.array(
            [crop_area[member_class[i] == 3].sum() / 1.0e12 for i in range(3)]
        )
        nonpositive_period = np.array(
            [crop_area[member_q[i] == 1].sum() / 1.0e12 for i in range(3)]
        )
        compound_values.append(compound_period)
        nonpositive_values.append(nonpositive_period)
        for member_index, member_label in enumerate(MEMBER_LABELS):
            member_rows.append(
                {
                    "period": period,
                    "period_label": period_label,
                    "member": member_label,
                    "strict_compound_million_km2": compound_period[member_index],
                    "nonpositive_Q_million_km2": nonpositive_period[member_index],
                }
            )

    table = pd.DataFrame(member_rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table_path = (
        TABLE_DIR
        / "ExtendedData_Fig08_member_compound_and_nonpositiveQ_strict_positiveQ.csv"
    )
    if table_path.exists():
        archived = pd.read_csv(table_path)
        if not archived[["period", "period_label", "member"]].equals(
            table[["period", "period_label", "member"]]
        ):
            raise AssertionError("Member identities differ from the archived source table.")
        for column in ("strict_compound_million_km2", "nonpositive_Q_million_km2"):
            if not np.allclose(archived[column], table[column], rtol=0, atol=1e-11):
                raise AssertionError(f"Member values differ from the archived source table: {column}")
    else:
        table.to_csv(table_path, index=False, encoding="utf-8-sig")
    for period in PERIODS:
        if table.loc[table["period"] == period, "member"].tolist() != list(MEMBER_LABELS):
            raise AssertionError(f"Member ordering is not one-to-one in {period}.")

    fig = plt.figure(figsize=(7.4, 5.1), dpi=600)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.5, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.040,
        right=0.965,
        bottom=0.055,
        top=0.940,
        wspace=0.17,
        hspace=0.22,
    )
    left_top = grid[0, 0].subgridspec(2, 1, height_ratios=[1.0, 0.09], hspace=0.11)
    left_bottom = grid[1, 0].subgridspec(2, 1, height_ratios=[1.0, 0.09], hspace=0.11)
    ax_a = fig.add_subplot(left_top[0])
    cax_a = fig.add_subplot(left_top[1])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(left_bottom[0])
    cax_c = fig.add_subplot(left_bottom[1])
    ax_d = fig.add_subplot(grid[1, 1])
    for cax in (cax_a, cax_c):
        slot = cax.get_position()
        cax.set_position(
            [slot.x0 + 0.14 * slot.width, slot.y0 + 0.38 * slot.height,
             0.72 * slot.width, 0.52 * slot.height]
        )

    mesh_a = draw_agreement_map(
        ax_a,
        compound_count,
        compound_evaluated,
        code_grid,
        lon,
        lat,
        COMPOUND_COLORS,
        boundaries,
    )
    panel_heading(ax_a, "a", "Ensemble agreement on compound pressure, SSP3-7.0")
    add_discrete_colorbar(fig, cax_a, mesh_a)

    style_member_axis(ax_b)
    draw_member_panel(ax_b, compound_values)
    panel_heading(ax_b, "b", "Member-specific compound cropland")

    mesh_c = draw_agreement_map(
        ax_c,
        nonpositive_count,
        q_state_evaluated,
        code_grid,
        lon,
        lat,
        NONPOSITIVE_COLORS,
        boundaries,
    )
    panel_heading(ax_c, "c", "Ensemble agreement on the Q ≤ 0 state, SSP3-7.0")
    add_discrete_colorbar(fig, cax_c, mesh_c)

    style_member_axis(ax_d)
    draw_member_panel(ax_d, nonpositive_values)
    panel_heading(ax_d, "d", "Member-specific cropland with Q ≤ 0")

    member_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="white",
        markeredgecolor="#5E6870",
        markeredgewidth=1.0,
        markersize=4.8,
        label="Individual member",
    )
    mean_handle = Line2D(
        [0],
        [0],
        color="#5E6870",
        linewidth=2.2,
        label="Ensemble mean",
    )
    ax_b.legend(
        handles=[member_handle, mean_handle],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.97),
        frameon=False,
        ncol=1,
        labelspacing=0.35,
        handlelength=1.45,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    output_stem = (
        FIG_DIR
        / "ExtendedData_Fig08_member_agreement"
    )
    save_figure(fig, output_stem)

    crop_ssp3_area = np.asarray(
        product["basin_crop_area_ssp370_m2"], dtype=float
    )
    unanimous_area = (
        crop_ssp3_area[
            np.asarray(
                product["compound_member_unanimous_IRR_ssp370_basin"],
                dtype=np.int8,
            )
            == 1
        ].sum()
        / 1.0e12
    )
    ensemble_strict_area = (
        crop_ssp3_area[
            np.asarray(product["class_IRR_ssp370_basin"], dtype=np.int8) == 3
        ].sum()
        / 1.0e12
    )
    unanimous_share = 100.0 * unanimous_area / ensemble_strict_area
    if round(unanimous_share, 1) != 87.8:
        raise AssertionError(f"Expected 87.8% unanimous share, got {unanimous_share:.6f}%")
    if not set(np.unique(compound_count[compound_evaluated])).issubset({0, 1, 2, 3}):
        raise AssertionError("Invalid displayed compound member count.")
    if not set(np.unique(nonpositive_count[q_state_evaluated])).issubset({0, 1, 2, 3}):
        raise AssertionError("Invalid displayed Q <= 0 member count.")
    summary = {
        "figure_stem": str(output_stem),
        "member_table": str(table_path),
        "compound_member_count_values": np.unique(compound_count).astype(int).tolist(),
        "nonpositive_Q_member_count_values": np.unique(nonpositive_count)
        .astype(int)
        .tolist(),
        "ssp370_unanimous_strict_compound_million_km2": unanimous_area,
        "ssp370_ensemble_strict_compound_million_km2": ensemble_strict_area,
        "ssp370_unanimous_share_percent": unanimous_share,
        "panel_a_evaluated_cropland_million_km2": crop_ssp3_area[
            compound_evaluated
        ].sum()
        / 1.0e12,
        "member_values": table.to_dict(orient="records"),
        "map_legend": {
            "compound": list(COMPOUND_COLORS),
            "nonpositive_Q": list(NONPOSITIVE_COLORS),
            "white": "outside the evaluated cropland domain",
        },
        "panel_c_evaluated_cropland_million_km2": crop_ssp3_area[
            q_state_evaluated
        ].sum()
        / 1.0e12,
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = (
        AUDIT_DIR
        / "ExtendedData_Fig08_member_agreement_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
