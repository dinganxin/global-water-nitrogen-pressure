# -*- coding: utf-8 -*-
"""
Global NPI--IPI pressure classes using signed QIRR.

Panels:
    a. Historical
    b. SSP1-2.6
    c. SSP3-7.0
    d. High-pressure cropland expansion

Main refinements relative to the previous version:
    1. Softer, colour-blind-aware palette with identical map/bar colours.
    2. Shorter panel titles to avoid repeated wording.
    3. Lighter country boundaries and thinner IPCC AR6 focus-region outlines.
    4. IPCC labels are shown only in panel a, with a subtle white halo.
    5. Cleaner grouped bar chart with restrained gridlines, aligned labels and more balanced layout.
    6. Vector-friendly PDF/SVG export with embedded TrueType fonts.
"""

from __future__ import annotations

from pathlib import Path
import math
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# Use the project-vendored pure-Python shapefile reader for reproducibility.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))


# =============================================================================
# Paths
# =============================================================================

SOURCE_ROOT = Path(__file__).resolve().parents[3]
OUT = SOURCE_ROOT

FIG_FINAL = OUT / "figures" / "main"
CODE_FINAL = OUT / "code" / "figures" / "main"
FIG_FINAL.mkdir(parents=True, exist_ok=True)
CODE_FINAL.mkdir(parents=True, exist_ok=True)

PRODUCT_PATH = (
    SOURCE_ROOT
    / "data"
    / "core"
    / "NPI_IPI_QIRR_strict_positiveQ_period_products_01pct.npz"
)

SUMMARY_PATH = (
    SOURCE_ROOT
    / "tables"
    / "main"
    / "Fig03_global_class_area.csv"
)

NE_ROOT = SOURCE_ROOT / "data" / "geospatial" / "natural_earth_admin0"
IPCC_BOUNDARY_PATH = SOURCE_ROOT / "data" / "geospatial" / "IPCC_AR6_regions" / "IPCCBoundary.shp"


# =============================================================================
# Optional shapefile dependency
# =============================================================================

try:
    import shapefile  # pyshp
    HAS_SHAPEFILE = True
except Exception:
    HAS_SHAPEFILE = False


# =============================================================================
# Global style: restrained publication figure style
# =============================================================================

mpl.rcParams.update(
    {
        # Keep Arial when available, but add fallbacks to avoid runtime warnings.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.linewidth": 0.35,
        "axes.edgecolor": "#4A4A4A",
        "axes.labelcolor": "#202020",
        "xtick.color": "#202020",
        "ytick.color": "#202020",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
        "figure.dpi": 600,
        "mathtext.default": "regular",
        "legend.frameon": False,
        "hatch.color": "#858C91",
        "hatch.linewidth": 0.22,
    }
)


# =============================================================================
# Colours and labels
# =============================================================================
# Palette logic:
#     Low      : very light neutral grey
#     I-only   : muted blue
#     N-only   : warm ochre/gold
#     Compound : muted brick red
# This preserves the semantic meaning while reducing saturation and red dominance.

CLASS_COLORS = {
    0: "#E7E8EA",  # Low pressure: light neutral grey
    1: "#D6A33D",  # N-only: muted ochre
    2: "#4E8FB8",  # I-only: muted blue
    3: "#C95B55",  # Compound: muted brick red
}

# Use exactly the same colours in maps and bars to avoid visual inconsistency.
BAR_COLORS = {1: CLASS_COLORS[1], 2: CLASS_COLORS[2], 3: CLASS_COLORS[3]}

# The non-positive-Q domain is a denominator-status mask, not a fifth class.
# Keep it visually quieter than the four threshold-defined pressure classes.
LIMITING_STATE_COLOR = "#ECEEEF"
LIMITING_STATE_EDGE = "#858C91"
LIMITING_STATE_DOT = "#9DA3A7"


class QMaskLegendHandle:
    """Legend-only handle for a limiting-state rectangle with one centre dot."""

    def __init__(self, label: str):
        self._label = label

    def get_label(self) -> str:
        return self._label


class HandlerQMask(HandlerBase):
    """Draw a light rectangle with one restrained centred dot."""

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

CLASS_LABELS = {
    0: "Low",
    1: "N-only",
    2: "I-only",
    3: "Compound",
}

BAR_PERIODS = [
    ("historical", "Historical"),
    ("ssp126", "SSP1–2.6"),
    ("ssp370", "SSP3–7.0"),
]

MAP_TITLES = {
    "historical": "Historical",
    "ssp126": "SSP1–2.6",
    "ssp370": "SSP3–7.0",
}

MAP_XLIM = (-180, 180)
MAP_YLIM = (-60, 84)

MAP_FRAME_COLOR = "#6A6A6A"
MAP_FRAME_LW = 0.20

BOUNDARY_COLOR = "#7B7B7B"
BOUNDARY_LW = 0.10
BOUNDARY_ALPHA = 0.38

IPCC_BOUNDARY_COLOR = "#4A4A4A"   # #4A4A4A  4D5961
IPCC_BOUNDARY_LW = 0.50
IPCC_BOUNDARY_ALPHA = 0.85
MERCATOR_RADIUS_M = 6378137.0
IPCC_BOUNDARY_LS = (0, (4.5, 2.5))

# Representative established and emerging hotspot regions used consistently
# with Extended Data Fig. 7. Label coordinates are deliberately placed just
# outside the corresponding dashed boundaries to avoid obscuring map cells.
FOCUS_IPCC_REGIONS = {"SAS", "WCA", "WAF", "MED", "CNA", "ESAF"}
FOCUS_IPCC_LABEL_POS = {
    "CNA": (-78.0, 38.0),
    "MED": (14.5, 51.0),
    "WCA": (60.0, 51.0),
    "WAF": (-13.0, -3),
    "SAS": (63.0, 12.0),
    "ESAF": (49.0, -32.0),
}


# =============================================================================
# Utilities
# =============================================================================

def sort_lon(lon: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort longitude to -180..180 and reorder the corresponding grid columns."""
    lon = np.asarray(lon, dtype=float)
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon180)
    return lon180[order], grid[:, order]


def basin_to_grid(
    values: np.ndarray,
    code_grid: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map basin-level values back to the global grid."""
    values = np.asarray(values, dtype=float)
    grid = np.full(code_grid.shape, np.nan, dtype=float)

    ok = np.isfinite(code_grid) & (code_grid >= 0)
    rr, cc = np.where(ok)
    idx = code_grid[rr, cc].astype(int)
    valid = idx < values.size
    grid[rr[valid], cc[valid]] = values[idx[valid]]

    return sort_lon(lon, grid)


def class_grid(z: np.lib.npyio.NpzFile, exp: str, period: str) -> np.ndarray:
    """Read classes and enforce the period-specific agricultural domain."""
    key = f"class_{exp}_{period}_basin"
    if key not in z.files:
        raise KeyError(f"Missing key in product file: {key}")
    values = z[key].astype(float).copy()
    area_key = f"basin_crop_area_{period}_m2"
    if area_key not in z.files:
        raise KeyError(f"Missing key in product file: {area_key}")
    crop_area = np.asarray(z[area_key], dtype=float)
    values[~np.isfinite(crop_area) | (crop_area <= 0)] = np.nan
    return values


def get_area(summary: pd.DataFrame, period_key: str, column: str) -> float:
    """Read global cropland area from the summary table."""
    row = summary[
        (summary["experiment"] == "IRR")
        & (summary["period_key"] == period_key)
    ]
    if row.empty:
        raise ValueError(f"No summary row found for period_key = {period_key}")
    if column not in row.columns:
        raise KeyError(f"Column not found in summary table: {column}")
    return float(row.iloc[0][column])


def nice_ymax(value: float, base: float = 0.5) -> float:
    """Round y-axis maximum upward to a visually clean value."""
    if not np.isfinite(value) or value <= 0:
        return base
    return math.ceil(value / base) * base


# =============================================================================
# Country boundaries
# =============================================================================

def find_country_shapefile() -> Path | None:
    """Find Natural Earth country shapefile."""
    if not HAS_SHAPEFILE:
        warnings.warn("pyshp is not installed. Country boundaries will not be drawn.")
        return None

    if not NE_ROOT.exists():
        warnings.warn(f"Natural Earth folder not found: {NE_ROOT}")
        return None

    shp_files = list(NE_ROOT.rglob("*.shp"))
    if len(shp_files) == 0:
        warnings.warn(f"No .shp file found under: {NE_ROOT}")
        return None

    preferred = [
        p for p in shp_files
        if "ne_110m_admin_0_countries" in p.name.lower()
        or "admin_0_countries" in p.name.lower()
    ]
    shp_path = preferred[0] if preferred else shp_files[0]

    if not shp_path.with_suffix(".shx").exists() or not shp_path.with_suffix(".dbf").exists():
        warnings.warn(f"Companion .shx/.dbf files missing for: {shp_path}")
        return None

    print(f"[Boundary] Using shapefile: {shp_path}")
    return shp_path


def _split_antimeridian(xs, ys):
    """Split lines crossing the antimeridian to avoid long horizontal artifacts."""
    seg_x, seg_y = [], []
    prev_x = None

    for x, y in zip(xs, ys):
        if prev_x is not None and abs(x - prev_x) > 180:
            if len(seg_x) > 1:
                yield seg_x, seg_y
            seg_x, seg_y = [], []

        seg_x.append(x)
        seg_y.append(y)
        prev_x = x

    if len(seg_x) > 1:
        yield seg_x, seg_y


def read_country_boundary_segments(shp_path: Path | None):
    """Read country boundary line segments."""
    if shp_path is None:
        return None

    reader = shapefile.Reader(str(shp_path))
    segments = []

    for shp in reader.shapes():
        pts = shp.points
        if not pts:
            continue

        parts = list(shp.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            part = pts[parts[i]: parts[i + 1]]
            if len(part) < 2:
                continue

            xs = [((p[0] + 180.0) % 360.0) - 180.0 for p in part]
            ys = [p[1] for p in part]

            for sx, sy in _split_antimeridian(xs, ys):
                if len(sx) >= 2:
                    segments.append(np.column_stack([sx, sy]))

    print(f"[Boundary] Number of boundary segments = {len(segments)}")
    return segments if len(segments) > 0 else None


def add_country_boundaries(ax: plt.Axes, boundary_segments) -> None:
    """Add subtle country boundaries to a map panel."""
    if boundary_segments is None:
        return

    lc = LineCollection(
        boundary_segments,
        colors=BOUNDARY_COLOR,
        linewidths=BOUNDARY_LW,
        alpha=BOUNDARY_ALPHA,
        zorder=900,
        clip_on=True,
    )
    try:
        lc.set_capstyle("round")
        lc.set_joinstyle("round")
    except Exception:
        pass
    ax.add_collection(lc)


# =============================================================================
# IPCC AR6 regional boundaries
# =============================================================================

def mercator_to_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert WGS84 World Mercator metres to longitude/latitude degrees."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    lon = np.degrees(x / MERCATOR_RADIUS_M)
    lat = np.degrees(2.0 * np.arctan(np.exp(y / MERCATOR_RADIUS_M)) - np.pi / 2.0)

    lon = ((lon + 180.0) % 360.0) - 180.0
    lat = np.clip(lat, -89.999, 89.999)

    return lon, lat


def find_ipcc_boundary_shapefile() -> Path | None:
    """Find the IPCC AR6 boundary shapefile."""
    if not HAS_SHAPEFILE:
        warnings.warn("pyshp is not installed. IPCC AR6 boundaries will not be drawn.")
        return None

    if not IPCC_BOUNDARY_PATH.exists():
        warnings.warn(f"IPCC AR6 boundary shapefile not found: {IPCC_BOUNDARY_PATH}")
        return None

    if (
        not IPCC_BOUNDARY_PATH.with_suffix(".shx").exists()
        or not IPCC_BOUNDARY_PATH.with_suffix(".dbf").exists()
    ):
        warnings.warn(f"Companion .shx/.dbf files missing for: {IPCC_BOUNDARY_PATH}")
        return None

    print(f"[IPCC AR6 boundary] Using shapefile: {IPCC_BOUNDARY_PATH}")
    return IPCC_BOUNDARY_PATH


def read_ipcc_boundary_segments(shp_path: Path | None):
    """Read selected IPCC AR6 regional boundary line segments in lon/lat coordinates."""
    if shp_path is None:
        return None

    reader = shapefile.Reader(str(shp_path))
    segments = []

    fields = [f[0] for f in reader.fields[1:]]
    try:
        acronym_idx = fields.index("Acronym")
    except ValueError as exc:
        raise KeyError("Field 'Acronym' was not found in the IPCC boundary shapefile.") from exc

    for shp, rec in zip(reader.shapes(), reader.records()):
        region_code = str(rec[acronym_idx]).strip()
        if region_code not in FOCUS_IPCC_REGIONS:
            continue

        pts = shp.points
        if not pts:
            continue

        parts = list(shp.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            part = pts[parts[i]: parts[i + 1]]
            if len(part) < 2:
                continue

            arr = np.asarray(part, dtype=float)
            xs, ys = mercator_to_lonlat(arr[:, 0], arr[:, 1])

            ok = np.isfinite(xs) & np.isfinite(ys)
            xs = xs[ok]
            ys = ys[ok]
            if xs.size < 2:
                continue

            for sx, sy in _split_antimeridian(xs, ys):
                if len(sx) >= 2:
                    segments.append(np.column_stack([sx, sy]))

    print(f"[IPCC AR6 boundary] Selected regions = {sorted(FOCUS_IPCC_REGIONS)}")
    print(f"[IPCC AR6 boundary] Number of selected boundary segments = {len(segments)}")
    return segments if len(segments) > 0 else None


def add_ipcc_boundaries(ax: plt.Axes, boundary_segments) -> None:
    """Add selected IPCC AR6 regional boundaries."""
    if boundary_segments is None:
        return

    # lc = LineCollection(
    #     boundary_segments,
    #     colors=IPCC_BOUNDARY_COLOR,
    #     linewidths=IPCC_BOUNDARY_LW,
    #     alpha=IPCC_BOUNDARY_ALPHA,
    #     zorder=1200,
    #     clip_on=True,
    # )
    lc = LineCollection(
        boundary_segments,
        colors=IPCC_BOUNDARY_COLOR,
        linewidths=IPCC_BOUNDARY_LW,
        linestyles=IPCC_BOUNDARY_LS,
        alpha=IPCC_BOUNDARY_ALPHA,
        zorder=1200,
        clip_on=True,
    )
    try:
        lc.set_capstyle("round")
        lc.set_joinstyle("round")
    except Exception:
        pass
    ax.add_collection(lc)


def add_focus_ipcc_labels(ax: plt.Axes) -> None:
    """Add compact labels for selected IPCC AR6 regions in panel a only."""
    halo = [pe.withStroke(linewidth=0.65, foreground="white", alpha=0.76)]
    for code, (x, y) in FOCUS_IPCC_LABEL_POS.items():
        ax.text(
            x,
            y,
            code,
            ha="center",
            va="center",
            multialignment="center",
            fontsize=4.85,
            fontweight="bold",
            color=IPCC_BOUNDARY_COLOR,
            zorder=1450,
            clip_on=True,
            path_effects=halo,
        )


# =============================================================================
# Map plotting
# =============================================================================

def make_class_colormap() -> tuple[ListedColormap, BoundaryNorm]:
    """Discrete pressure-class colormap."""
    cmap = ListedColormap([CLASS_COLORS[i] for i in range(4)])
    cmap.set_bad((1, 1, 1, 0))
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    return cmap, norm


def draw_map_frame(ax: plt.Axes) -> None:
    """Draw a thin rectangular frame around map."""
    for spine in ax.spines.values():
        spine.set_visible(False)

    frame = Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        fill=False,
        edgecolor=MAP_FRAME_COLOR,
        linewidth=MAP_FRAME_LW,
        zorder=2200,
        clip_on=False,
    )
    ax.add_patch(frame)


def style_map(ax: plt.Axes) -> None:
    """Style one map panel."""
    ax.set_xlim(*MAP_XLIM)
    ax.set_ylim(*MAP_YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    ax.margins(0)
    draw_map_frame(ax)


def add_panel_label(ax: plt.Axes, label: str, title: str) -> None:
    """Add compact panel label and short title."""
    ax.text(
        0.000,
        1.020,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=8.0,
        color="#111111",
    )
    ax.text(
        0.050,
        1.021,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#111111",
    )


def add_map_legend(ax: plt.Axes) -> None:
    """Add four pressure classes plus the independent Q<=0 mask."""
    # Matplotlib fills multi-column legends column-wise. This ordering yields:
    # row 1: Low | I-only; row 2: N-only | Compound; row 3: Q<=0.
    handles = [
        Patch(facecolor=CLASS_COLORS[0], edgecolor="none", label=CLASS_LABELS[0]),
        Patch(facecolor=CLASS_COLORS[1], edgecolor="none", label=CLASS_LABELS[1]),
        QMaskLegendHandle(r"$Q\leq0$"),
        Patch(facecolor=CLASS_COLORS[2], edgecolor="none", label=CLASS_LABELS[2]),
        Patch(facecolor=CLASS_COLORS[3], edgecolor="none", label=CLASS_LABELS[3]),
    ]

    leg = ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.016, 0.086),
        frameon=True,
        ncol=2,
        fontsize=4.75,
        handlelength=1.45,
        handleheight=0.60,
        handletextpad=0.35,
        columnspacing=0.90,
        labelspacing=0.36,
        borderpad=0.28,
        borderaxespad=0.0,
        framealpha=0.88,
        facecolor="white",
        edgecolor="none",
        handler_map={QMaskLegendHandle: HandlerQMask()},
    )
    leg.set_zorder(2600)


def plot_pressure_class_map(
    ax: plt.Axes,
    z: np.lib.npyio.NpzFile,
    period: str,
    label: str,
    title: str,
    boundary_segments,
    ipcc_boundary_segments,
    show_legend: bool = False,
    show_ipcc_labels: bool = False,
) -> None:
    """Plot one pressure-class map."""
    lon = z["lon"]
    lat = z["lat"]
    code = z["basin_code_grid"]

    cmap, norm = make_class_colormap()
    vals = class_grid(z, "IRR", period)
    plon, grid = basin_to_grid(vals, code, lon)

    ax.pcolormesh(
        plon,
        lat,
        np.ma.masked_invalid(grid),
        cmap=cmap,
        norm=norm,
        shading="nearest",
        linewidth=0,
        antialiased=False,
        rasterized=True,
        zorder=2,
    )

    # Four pressure classes are defined only where Q > 0. The complete Q <= 0
    # domain is shown as a separate denominator-status mask using a dot texture.
    nonpositive_key = f"nonpositive_Q_limiting_state_{period}_basin"
    if nonpositive_key not in z.files:
        raise KeyError(f"Missing key in product file: {nonpositive_key}")
    nonpositive = np.asarray(z[nonpositive_key], dtype=bool)
    crop_area = np.asarray(z[f"basin_crop_area_{period}_m2"], dtype=float)
    nonpositive &= np.isfinite(crop_area) & (crop_area > 0)
    _, nonpositive_grid = basin_to_grid(nonpositive.astype(float), code, lon)
    dot_mask = np.isfinite(nonpositive_grid)
    dot_mask[dot_mask] = nonpositive_grid[dot_mask] >= 0.5
    limiting_grid = np.ma.masked_where(
        ~dot_mask,
        np.ones_like(nonpositive_grid, dtype=float),
    )

    # Retain the established light-grey limiting-state fill, then place one
    # small dot at the centre of every selected model grid cell. Unlike a
    # renderer-dependent hatch, this texture remains aligned to the data grid.
    ax.pcolormesh(
        plon,
        lat,
        limiting_grid,
        cmap=ListedColormap([LIMITING_STATE_COLOR]),
        vmin=0.5,
        vmax=1.5,
        shading="nearest",
        linewidth=0,
        antialiased=False,
        rasterized=True,
        zorder=3,
    )
    yy, xx = np.meshgrid(lat, plon, indexing="ij")
    ax.scatter(
        xx[dot_mask],
        yy[dot_mask],
        s=0.82,
        marker=".",
        color=LIMITING_STATE_DOT,
        alpha=0.70,
        linewidths=0,
        rasterized=True,
        zorder=4,
    )

    add_country_boundaries(ax, boundary_segments)
    add_ipcc_boundaries(ax, ipcc_boundary_segments)
    if show_ipcc_labels:
        add_focus_ipcc_labels(ax)
    style_map(ax)
    add_panel_label(ax, label, title)
    if show_legend:
        add_map_legend(ax)


# =============================================================================
# Bar plotting
# =============================================================================

def style_bar_axis(ax: plt.Axes) -> None:
    """Style the grouped bar axis."""
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.35)
        ax.spines[spine].set_color("#555555")

    ax.grid(axis="y", color="#E6E9ED", linewidth=0.42, zorder=0)
    ax.set_axisbelow(True)


def add_panel_title_fig(
    fig: plt.Figure,
    label: str,
    title: str,
    x_label: float,
    x_title: float,
    y: float,
) -> None:
    """Add panel label and title in figure coordinates."""
    fig.text(
        x_label,
        y,
        label,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=8.0,
        color="#111111",
    )
    fig.text(
        x_title,
        y + 0.001,
        title,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#111111",
    )


def plot_high_pressure_bars(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Plot high-pressure cropland area as grouped bars."""
    area_cols = {
        1: "nitrate_pressure_only_area_million_km2",
        2: "irrigation_pressure_only_area_million_km2",
        3: "compound_high_pressure_area_million_km2",
    }

    bar_order = [2, 1, 3]  # I-only, N-only, Compound
    x = np.array([0.00, 0.62, 1.24])
    x_labels = [label for _, label in BAR_PERIODS]

    values = {cls: [] for cls in bar_order}
    for period_key, _ in BAR_PERIODS:
        for cls in bar_order:
            values[cls].append(get_area(summary, period_key, area_cols[cls]))
    for cls in bar_order:
        values[cls] = np.array(values[cls], dtype=float)

    limiting_state = np.array(
        [
            get_area(
                summary,
                period_key,
                "nonpositive_Q_limiting_state_area_million_km2",
            )
            for period_key, _ in BAR_PERIODS
        ],
        dtype=float,
    )

    compound = values[3]
    hist_compound = compound[0]
    if hist_compound > 0:
        pct_change = (compound - hist_compound) / hist_compound * 100.0
    else:
        pct_change = np.full_like(compound, np.nan)

    bar_width = 0.104
    offsets = {
        2: -bar_width * 1.68,
        1: -bar_width * 0.56,
        3: bar_width * 0.56,
    }
    limiting_offset = bar_width * 1.68

    text_dark = "#2F2F2F"
    compound_text = "#8F3430"

    for cls in bar_order:
        xpos = x + offsets[cls]
        ax.bar(
            xpos,
            values[cls],
            width=bar_width,
            color=BAR_COLORS[cls],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )

        for i, (xi, val) in enumerate(zip(xpos, values[cls])):
            if not np.isfinite(val) or val <= 0.18:
                continue

            if cls == 3:
                # For future scenarios, place percentage and area on two separate lines.
                if i == 0:
                    ax.text(
                        xi,
                        val + 0.070,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=5.0,
                        color=compound_text,
                        fontweight="bold",
                        zorder=4,
                    )
                else:
                    ax.text(
                        xi,
                        val + 0.255,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=5.0,
                        color=compound_text,
                        fontweight="bold",
                        zorder=4,
                    )
                    ax.text(
                        xi,
                        val + 0.070,
                        f"(+{pct_change[i]:.1f}%)",
                        ha="center",
                        va="bottom",
                        fontsize=4.85,
                        color=compound_text,
                        fontweight="bold",
                        zorder=4,
                    )
            else:
                ax.text(
                    xi,
                    val + 0.070,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=4.95,
                    color=text_dark,
                    fontweight="normal",
                    zorder=4,
                )

    limiting_x = x + limiting_offset
    ax.bar(
        limiting_x,
        limiting_state,
        width=bar_width,
        color=LIMITING_STATE_COLOR,
        edgecolor=LIMITING_STATE_EDGE,
        linewidth=0.30,
        hatch="...",
        zorder=3,
    )
    for xi, val in zip(limiting_x, limiting_state):
        if np.isfinite(val) and val > 0.18:
            ax.text(
                xi,
                val + 0.070,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=4.95,
                color="#555C61",
                fontweight="normal",
                zorder=4,
            )

    ax.set_ylabel("Cropland area (million km$^2$)", fontsize=6.35, labelpad=1.2)
    ax.set_xlim(-0.31, 1.55)

    max_value = float(
        max(
            np.nanmax(values[1]),
            np.nanmax(values[2]),
            np.nanmax(values[3]),
            np.nanmax(limiting_state),
        )
    )
    ymax = nice_ymax(max_value * 1.18, base=0.5)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=5.8)
    ax.tick_params(axis="y", labelsize=5.95, length=2.0, width=0.35, pad=2.0)
    ax.tick_params(axis="x", length=0, pad=2.6)

    style_bar_axis(ax)

    # Redraw the baseline and bottom ticks above the bars. This keeps every bar
    # anchored at exactly zero while preventing its face/edge from obscuring
    # the horizontal axis, following the established axis-redraw treatment.
    ax.axhline(
        0.0,
        color="#555555",
        linewidth=0.35,
        zorder=12,
        clip_on=False,
    )
    ax.spines["bottom"].set_zorder(13)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_zorder(14)
        tick.tick2line.set_zorder(14)

    handles = [
        Patch(facecolor=BAR_COLORS[2], edgecolor="none", label="I-only"),
        Patch(facecolor=BAR_COLORS[1], edgecolor="none", label="N-only"),
        Patch(facecolor=BAR_COLORS[3], edgecolor="none", label="Compound"),
        QMaskLegendHandle(r"$Q\leq0$"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.016, 0.932),
        frameon=False,
        fontsize=5.25,
        ncol=4,
        handlelength=1.45,
        handleheight=0.60,
        columnspacing=0.84,
        handletextpad=0.34,
        borderaxespad=0.0,
        handler_map={QMaskLegendHandle: HandlerQMask()},
    )


# =============================================================================
# Layout
# =============================================================================

def create_axes(fig: plt.Figure) -> tuple[dict[str, plt.Axes], dict[str, float]]:
    """
    Create a manual 2 x 2 layout.

    Panels a--c are map panels. Panel d is a quantitative summary and is shifted
    slightly right to create more breathing room for its y-axis label.
    """
    left = 0.042
    right = 0.975
    gap_x = 0.030
    panel_w = (right - left - gap_x) / 2.0

    fig_w, fig_h = fig.get_size_inches()
    panel_h = (panel_w * fig_w / 2.5) / fig_h

    top = 0.920
    gap_y = 0.078
    y_top = top - panel_h
    y_bottom = y_top - gap_y - panel_h

    x_left = left
    x_right = left + panel_w + gap_x

    d_shift = 0.018
    d_left = x_right + d_shift
    d_w = panel_w - d_shift

    axes = {
        "a": fig.add_axes([x_left, y_top, panel_w, panel_h]),
        "b": fig.add_axes([x_right, y_top, panel_w, panel_h]),
        "c": fig.add_axes([x_left, y_bottom, panel_w, panel_h]),
        "d": fig.add_axes([d_left, y_bottom, d_w, panel_h]),
    }

    meta = {
        "x_right": x_right,
        "panel_w": panel_w,
        "y_bottom": y_bottom,
        "panel_h": panel_h,
    }
    return axes, meta


# =============================================================================
# Main plotting function
# =============================================================================

def plot_fig3(z: np.lib.npyio.NpzFile, summary: pd.DataFrame) -> None:
    """Plot the four-panel figure."""
    shp_path = find_country_shapefile()
    boundary_segments = read_country_boundary_segments(shp_path)
    ipcc_shp_path = find_ipcc_boundary_shapefile()
    ipcc_boundary_segments = read_ipcc_boundary_segments(ipcc_shp_path)

    fig = plt.figure(figsize=(7.20, 4.04), dpi=600, facecolor="white")
    axes, meta = create_axes(fig)

    plot_pressure_class_map(
        ax=axes["a"],
        z=z,
        period="historical",
        label="a",
        title=MAP_TITLES["historical"],
        boundary_segments=boundary_segments,
        ipcc_boundary_segments=ipcc_boundary_segments,
        show_legend=True,
        show_ipcc_labels=True,
    )

    plot_pressure_class_map(
        ax=axes["b"],
        z=z,
        period="ssp126",
        label="b",
        title=MAP_TITLES["ssp126"],
        boundary_segments=boundary_segments,
        ipcc_boundary_segments=ipcc_boundary_segments,
        show_legend=False,
        show_ipcc_labels=False,
    )

    plot_pressure_class_map(
        ax=axes["c"],
        z=z,
        period="ssp370",
        label="c",
        title=MAP_TITLES["ssp370"],
        boundary_segments=boundary_segments,
        ipcc_boundary_segments=ipcc_boundary_segments,
        show_legend=False,
        show_ipcc_labels=False,
    )

    add_panel_title_fig(
        fig=fig,
        label="d",
        title="Pressure classes and hydrological limiting state",
        x_label=meta["x_right"],
        x_title=meta["x_right"] + 0.050 * meta["panel_w"],
        y=meta["y_bottom"] + meta["panel_h"] + 0.012,
    )
    plot_high_pressure_bars(axes["d"], summary)

    prefix = "Fig03_global_pressure_classes"
    png = FIG_FINAL / f"{prefix}.png"
    pdf = FIG_FINAL / f"{prefix}.pdf"
    svg = FIG_FINAL / f"{prefix}.svg"

    save_kwargs = dict(bbox_inches="tight", pad_inches=0.018, facecolor="white")
    fig.savefig(png, **save_kwargs)
    fig.savefig(pdf, **save_kwargs)
    fig.savefig(svg, **save_kwargs)
    plt.close(fig)

    print("[Saved]")
    print(png)
    print(pdf)
    print(svg)

    target = CODE_FINAL / Path(__file__).name
    if Path(__file__).resolve() != target.resolve():
        shutil.copy2(Path(__file__), target)


def main() -> None:
    """Run figure generation."""
    if not PRODUCT_PATH.exists():
        raise FileNotFoundError(f"Cannot find product file: {PRODUCT_PATH}")
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Cannot find summary file: {SUMMARY_PATH}")

    z = np.load(PRODUCT_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    plot_fig3(z, summary)


if __name__ == "__main__":
    main()
