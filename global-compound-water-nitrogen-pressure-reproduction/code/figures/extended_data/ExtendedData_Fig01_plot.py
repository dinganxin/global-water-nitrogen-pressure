# -*- coding: utf-8 -*-
"""Fig. 2-style IPI spatial maps with an explicit non-positive-Q mask.

This script intentionally follows the visual grammar of
``plot_fig2_component_diagnostics_final.py``:

    * 7.20 x 4.20 inch canvas
    * 2 x 2 compact panel layout
    * one aligned colorbar beneath each map panel
    * subdued country boundaries
    * no large in-figure title

It uses the final strict-positive-Q NPI-IPI product. Numerical IPI values are
shown only where Q > 0; crop-containing basins with Q <= 0 are overlaid using
the same restrained one-grid-cell/one-dot texture as the final main-text Fig. 3.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[3]
ROOT = SOURCE_ROOT
DATA = ROOT / "data"
FIG_DIR = ROOT / "figures" / "extended_data"
CODE_DIR = Path(__file__).resolve().parent
PRODUCT = DATA / "NPI_IPI_QIRR_strict_positiveQ_period_products_01pct.npz"
LINE_CACHE = SOURCE_ROOT / "data" / "figure_inputs" / "natural_earth_110m_line_cache_numeric.npz"

for folder in (FIG_DIR, CODE_DIR):
    folder.mkdir(parents=True, exist_ok=True)


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.2,
        "axes.linewidth": 0.35,
        "axes.edgecolor": "#444444",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 600,
        "mathtext.default": "regular",
    }
)


MAP_XLIM = (-180, 180)
MAP_YLIM = (-63, 87)
MAP_FRAME_COLOR = "#555555"
MAP_FRAME_LW = 0.22

PERIODS = [
    ("historical", "Historical", "1985-2014"),
    ("ssp126", "SSP1-2.6", "2045-2074"),
    ("ssp370", "SSP3-7.0", "2045-2074"),
    ("diff", "SSP3-7.0 - SSP1-2.6", "2045-2074"),
]


NPI_BOUNDS = np.array([0, 1, 2, 5, 10, 20, 50, 100], dtype=float)
WSI_BOUNDS = np.array([0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20], dtype=float)
DNPI_BOUNDS = np.array([-50, -20, -10, -5, -2, 0, 2, 5, 10, 20, 50], dtype=float)
DWSI_BOUNDS = np.array([-1.60, -0.80, -0.40, -0.20, -0.05, 0, 0.05, 0.20, 0.40, 0.80, 1.60], dtype=float)


def make_discrete_cmap(name: str, colors: list[str], under: str | None = None, over: str | None = None):
    cmap = ListedColormap(colors, name=name)
    cmap.set_bad((1, 1, 1, 0))
    cmap.set_under(under if under is not None else colors[0])
    cmap.set_over(over if over is not None else colors[-1])
    return cmap


CMAP_NPI = make_discrete_cmap(
    "npi_warm",
    ["#FCFAF2", "#F4E9CD", "#EFE2B7", "#DFC46F", "#C98B43", "#AE5D36", "#8E3E2D"],
    under="#FCFAF2",
    over="#8E3E2D",
)

CMAP_WSI = make_discrete_cmap(
    "wsi_green_blue",
    ["#F7FBF7", "#DDEEDB", "#BAD9B5", "#7DB9AA", "#4694A4", "#2C728A", "#285A7B", "#354D78"],
    under="#FBFEFB",
    over="#354D78",
)

CMAP_DIFF = make_discrete_cmap(
    "npi_wsi_difference_teal_brown",
    ["#3F9EA0", "#5AAEAA", "#79BFB6", "#B8DDD6", "#EEF4F0", "#F7F1E7", "#E5C99A", "#D49D58", "#C18440", "#B97535"],
    under="#157A83",
    over="#8B4A24",
)

LIMITING_STATE_COLOR = "#ECEEEF"
LIMITING_STATE_EDGE = "#858C91"
LIMITING_STATE_DOT = "#9DA3A7"


class QMaskLegendHandle:
    """Legend-only handle for a light limiting-state box with a centre dot."""

    def __init__(self, label: str):
        self._label = label

    def get_label(self) -> str:
        return self._label


class HandlerQMask(HandlerBase):
    """Draw the non-positive-Q legend key used in the main-text Fig. 3."""

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        rectangle = Rectangle(
            (xdescent, ydescent),
            width,
            height,
            facecolor=LIMITING_STATE_COLOR,
            edgecolor=LIMITING_STATE_EDGE,
            linewidth=0.30,
            transform=trans,
        )
        dot = Line2D(
            [xdescent + width / 2.0],
            [ydescent + height / 2.0],
            marker="o",
            markersize=1.05,
            markerfacecolor=LIMITING_STATE_DOT,
            markeredgecolor=LIMITING_STATE_DOT,
            markeredgewidth=0,
            alpha=0.78,
            linestyle="none",
            transform=trans,
        )
        return [rectangle, dot]


def sort_lon(lon: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(lon, dtype=float)
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon180)
    return lon180[order], grid[:, order]


def basin_to_grid(values: np.ndarray, code_grid: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    grid = np.full(code_grid.shape, np.nan, dtype=float)
    ok = np.isfinite(code_grid) & (code_grid >= 0)
    rr, cc = np.where(ok)
    idx = code_grid[rr, cc].astype(np.int64)
    valid = idx < values.size
    grid[rr[valid], cc[valid]] = values[idx[valid]]
    return sort_lon(lon, grid)


def values_for_period(z, metric: str, period: str) -> np.ndarray:
    key = "NPI" if metric == "NPI" else "IPI"
    values = z[f"{key}_IRR_{period}_basin"].astype(float).copy()
    area = z[f"basin_crop_area_{period}_m2"].astype(float)
    values[~np.isfinite(area) | (area <= 0)] = np.nan
    return values


def diff_values(z, metric: str) -> np.ndarray:
    s3 = values_for_period(z, metric, "ssp370")
    s1 = values_for_period(z, metric, "ssp126")
    area_union = (z["basin_crop_area_ssp370_m2"].astype(float) > 0) | (
        z["basin_crop_area_ssp126_m2"].astype(float) > 0
    )
    out = s3 - s1
    out[~area_union] = np.nan
    return out


def load_boundary_segments() -> list[np.ndarray]:
    if not LINE_CACHE.exists():
        return []

    cache = np.load(LINE_CACHE, allow_pickle=False)
    segments: list[np.ndarray] = []

    for ranges_key, points_key in [
        ("coast_ranges", "coast_points"),
        ("border_ranges", "border_points"),
    ]:
        points = cache[points_key]
        for start, end in cache[ranges_key]:
            pts = points[start:end]
            if pts.shape[0] >= 2:
                keep = (pts[:, 1] >= MAP_YLIM[0] - 3) & (pts[:, 1] <= MAP_YLIM[1] + 3)
                pts = pts[keep]
                if pts.shape[0] >= 2:
                    segments.append(pts)
    return segments


def add_country_boundaries(ax, boundary_segments: list[np.ndarray]) -> None:
    if not boundary_segments:
        return
    lc = LineCollection(
        boundary_segments,
        colors="#5F5F5F",
        linewidths=0.15,
        alpha=0.68,
        zorder=1000,
        clip_on=True,
    )
    try:
        lc.set_capstyle("round")
        lc.set_joinstyle("round")
    except Exception:
        pass
    ax.add_collection(lc)


def style_map(ax) -> None:
    ax.set_xlim(*MAP_XLIM)
    ax.set_ylim(*MAP_YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw the axes frame as lines. This is visually identical to a Rectangle
    # patch and avoids a Windows/Matplotlib patch-transform crash.
    frame_kwargs = dict(
        transform=ax.transAxes,
        color=MAP_FRAME_COLOR,
        linewidth=MAP_FRAME_LW,
        zorder=2200,
        clip_on=False,
        antialiased=True,
        solid_capstyle="butt",
    )
    ax.plot([0, 1], [0, 0], **frame_kwargs)
    ax.plot([0, 1], [1, 1], **frame_kwargs)
    ax.plot([0, 0], [0, 1], **frame_kwargs)
    ax.plot([1, 1], [0, 1], **frame_kwargs)


def add_panel_label(ax, label: str, title: str, right_label: str | None = None) -> None:
    ax.text(
        0.012,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        fontweight="bold",
        color="#222222",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.8),
        zorder=2000,
    )

    ax.text(
        0.000,
        1.012,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        color="#111111",
    )

    if right_label is not None:
        ax.text(
            1.000,
            1.012,
            right_label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.9,
            color="#333333",
        )


def _format_tick(value: float) -> str:
    if abs(value) >= 1:
        return f"{value:g}"
    return f"{value:g}"


def _tick_position_from_bounds(bounds: np.ndarray, tick: float) -> float:
    matches = np.where(np.isclose(bounds, tick, rtol=0, atol=1e-10))[0]
    if len(matches):
        return float(matches[0])
    return float(np.interp(tick, bounds, np.arange(len(bounds))))


def add_aligned_discrete_colorbar(fig, ax, cmap, bounds, ticks, label, extend="neither"):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4.2%", pad=0.032)

    colors = list(cmap.colors)
    n = len(colors)
    ext = 0.45 if extend == "both" else 0.0

    for i, color in enumerate(colors):
        cax.add_patch(
            Rectangle(
                (i, 0),
                1,
                1,
                facecolor=color,
                edgecolor="none",
                linewidth=0,
            )
        )

    if extend == "both":
        under_color = cmap(-1.0)
        over_color = cmap(2.0)
        cax.add_patch(
            Polygon(
                [[0, 0], [-ext, 0.5], [0, 1]],
                closed=True,
                facecolor=under_color,
                edgecolor="none",
            )
        )
        cax.add_patch(
            Polygon(
                [[n, 0], [n + ext, 0.5], [n, 1]],
                closed=True,
                facecolor=over_color,
                edgecolor="none",
            )
        )

    cax.add_patch(
        Rectangle(
            (0, 0),
            n,
            1,
            facecolor="none",
            edgecolor="#555555",
            linewidth=0.25,
        )
    )

    cax.set_xlim(-ext, n + ext)
    cax.set_ylim(0, 1)
    cax.set_yticks([])
    cax.set_xticks([_tick_position_from_bounds(bounds, t) for t in ticks])
    cax.set_xticklabels([_format_tick(float(t)) for t in ticks])
    cax.tick_params(axis="x", labelsize=6.1, length=1.8, width=0.35, pad=0.8)
    cax.set_xlabel(label, fontsize=6.7, labelpad=0.9)

    for spine in cax.spines.values():
        spine.set_visible(False)

    return cax


def plot_map_panel(
    fig,
    ax,
    lon,
    lat,
    grid,
    label,
    title,
    cmap,
    bounds,
    cbar_label,
    ticks,
    boundary_segments,
    right_label=None,
    extend="neither",
):
    style_map(ax)
    norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=(extend == "neither"))
    im = ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_invalid(grid),
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
        zorder=3,
    )
    add_country_boundaries(ax, boundary_segments)
    add_panel_label(ax, label, title, right_label=right_label)
    add_aligned_discrete_colorbar(fig, ax, cmap, bounds, ticks, cbar_label, extend=extend)
    return im


def gridded_metric(z, metric: str, period: str) -> tuple[np.ndarray, np.ndarray]:
    values = diff_values(z, metric) if period == "diff" else values_for_period(z, metric, period)
    return basin_to_grid(values, z["basin_code_grid"], z["lon"])


def gridded_nonpositive_q(z, period: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the period-specific Q<=0 mask, or its union for the difference."""
    if period == "diff":
        limiting = (
            np.asarray(
                z["nonpositive_Q_limiting_state_ssp126_basin"], dtype=bool
            )
            | np.asarray(
                z["nonpositive_Q_limiting_state_ssp370_basin"], dtype=bool
            )
        )
    else:
        limiting = np.asarray(
            z[f"nonpositive_Q_limiting_state_{period}_basin"], dtype=bool
        )
    return basin_to_grid(
        limiting.astype(float),
        z["basin_code_grid"],
        z["lon"],
    )


def add_nonpositive_q_overlay(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    limiting_grid: np.ndarray,
    show_legend: bool = False,
) -> None:
    """Overlay one restrained dot per model grid cell where Q<=0."""
    dot_mask = np.isfinite(limiting_grid) & (limiting_grid >= 0.5)
    if not np.any(dot_mask):
        return

    fill_grid = np.ma.masked_where(
        ~dot_mask,
        np.ones_like(limiting_grid, dtype=float),
    )
    ax.pcolormesh(
        lon,
        lat,
        fill_grid,
        cmap=ListedColormap([LIMITING_STATE_COLOR]),
        vmin=0.5,
        vmax=1.5,
        shading="nearest",
        linewidth=0,
        antialiased=False,
        rasterized=True,
        zorder=4,
    )

    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    ax.scatter(
        xx[dot_mask],
        yy[dot_mask],
        s=0.82,
        marker=".",
        color=LIMITING_STATE_DOT,
        alpha=0.70,
        linewidths=0,
        rasterized=True,
        zorder=5,
    )

    if show_legend:
        handle = QMaskLegendHandle(r"$Q\leq0$ (IPI undefined)")
        ax.legend(
            handles=[handle],
            handler_map={QMaskLegendHandle: HandlerQMask()},
            loc="lower left",
            bbox_to_anchor=(0.012, 0.015),
            frameon=True,
            fancybox=False,
            framealpha=0.88,
            facecolor="white",
            edgecolor="none",
            borderpad=0.18,
            handlelength=1.20,
            handletextpad=0.38,
            fontsize=4.8,
        )


def plot_metric(metric: str) -> Path:
    z = np.load(PRODUCT, allow_pickle=False)
    boundary_segments = load_boundary_segments()
    lat = z["lat"].astype(float)

    fig = plt.figure(figsize=(7.20, 4.20), facecolor="white", dpi=600)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.985,
        top=0.925,
        bottom=0.088,
        wspace=0.070,
        hspace=0.075,
    )
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    if metric == "NPI":
        cmap_abs = CMAP_NPI
        bounds_abs = NPI_BOUNDS
        bounds_diff = DNPI_BOUNDS
        ticks_abs = [0, 1, 2, 5, 10, 20, 50, 100]
        ticks_diff = [-50, -10, -2, 0, 2, 10, 50]
        cbar_abs = r"NPI (mg N L$^{-1}$)"
        cbar_diff = r"$\Delta$NPI (mg N L$^{-1}$)"
        outfile = (
            "ExtendedData_NPI_spatial_distribution_"
            "strict_positiveQ_qmask_scatter_dots_light_01pct"
        )
    else:
        cmap_abs = CMAP_WSI
        bounds_abs = WSI_BOUNDS
        bounds_diff = DWSI_BOUNDS
        ticks_abs = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        ticks_diff = [-1.6, -0.4, 0, 0.4, 1.6]
        cbar_abs = "IPI"
        cbar_diff = r"$\Delta$IPI"
        outfile = "ExtendedData_Fig01_IPI_spatial_distribution"

    for idx, (period, scenario_label, years) in enumerate(PERIODS):
        lon, grid = gridded_metric(z, metric, period)
        is_diff = period == "diff"
        panel_title = scenario_label
        plot_map_panel(
            fig,
            axes[idx],
            lon,
            lat,
            grid,
            chr(ord("a") + idx),
            panel_title,
            CMAP_DIFF if is_diff else cmap_abs,
            bounds_diff if is_diff else bounds_abs,
            cbar_diff if is_diff else cbar_abs,
            ticks_diff if is_diff else ticks_abs,
            boundary_segments=boundary_segments,
            right_label=None,
            extend="both" if is_diff else "neither",
        )
        qlon, qgrid = gridded_nonpositive_q(z, period)
        add_nonpositive_q_overlay(
            axes[idx],
            qlon,
            lat,
            qgrid,
            show_legend=(idx == 0),
        )

    png = FIG_DIR / f"{outfile}.png"
    pdf = FIG_DIR / f"{outfile}.pdf"
    svg = FIG_DIR / f"{outfile}.svg"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.018)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.018)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.018)
    plt.close(fig)
    return png


def main() -> None:
    if not PRODUCT.exists():
        raise FileNotFoundError(PRODUCT)
    outputs = [plot_metric("IPI")]
    target = CODE_DIR / Path(__file__).name
    if Path(__file__).resolve() != target.resolve():
        shutil.copy2(Path(__file__), target)
    for out in outputs:
        print(out)
        print(out.with_suffix(".pdf"))
        print(out.with_suffix(".svg"))
    print(CODE_DIR / Path(__file__).name)


if __name__ == "__main__":
    main()
