# -*- coding: utf-8 -*-
"""
Fig. 2 | Component diagnostics underlying future pressure divergence.

Panels:
    a. Cropland extent and irrigation expansion, SSP3-7.0 - SSP1-2.6
    b. Nitrate-N loss change, SSP3-7.0 - SSP1-2.6
    c. Irrigation water withdrawal change, SSP3-7.0 - SSP1-2.6
    d. Irrigation-conditioned runoff change, SSP3-7.0 - SSP1-2.6

Main settings:
    1. Panel a uses fixed-color cropland/activity background.
    2. The cropland background indicates cropland/activity cells only;
       it does not encode activity_fraction magnitude.
    3. Panel a includes a subtle Cropland legend.
    4. Near-zero irrigation change in panel a is transparent:
           abs(irri_diff) < 0.01 -> NaN
    5. Blank areas inside map frames are white.
    6. Country boundaries are subdued for figure legibility.
    7. Panels a-d all use separate under/over colors for colorbar triangles.
    8. Colorbar labels:
           螖 irrigated-area fraction
           螖N (g N m^-2 yr^-1)
           螖I (mm yr^-1)
           螖Q (mm yr^-1)
"""

from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import (
    ListedColormap,
    BoundaryNorm,
    to_rgba,
)
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))

try:
    import shapefile  # pyshp
    HAS_SHAPEFILE = True
except Exception:
    HAS_SHAPEFILE = False


# =============================================================================
# Paths
# =============================================================================

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
NATIVE_INPUT = (
    PACKAGE_ROOT
    / "data"
    / "figure_inputs"
    / "Fig02_native_grid_components_QIRR_2045_2074.npz"
)

TECH = (
    PACKAGE_ROOT
    / "data"
    / "figure_inputs"
    / "irrigation_technology_area_contribution_01pct.npz"
)

CROP_GRID = (
    PACKAGE_ROOT
    / "data"
    / "figure_inputs"
    / "cropland_union_mask_and_level5_grid_01pct.npz"
)

FIG_MAIN = PACKAGE_ROOT / "figures" / "main"
FIG_MAIN.mkdir(parents=True, exist_ok=True)

NE_ROOT = PACKAGE_ROOT / "data" / "geospatial" / "natural_earth_admin0"


# =============================================================================
# Global style
# =============================================================================

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


# =============================================================================
# Cropland background style
# =============================================================================

# Fixed cropland background color.
# A muted grey-blue is used to make cropland visible while avoiding confusion
# with the teal-brown irrigation-change color scale.
CROP_BG_COLOR = "#9AAAB8"
CROP_BG_ALPHA = 0.68


# =============================================================================
# Utilities
# =============================================================================

def sort_lon(lon, grid):
    """Sort longitude to -180..180."""
    lon = np.asarray(lon, dtype=float)
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon180)
    return lon180[order], grid[:, order]


def grid_edges(x):
    """Return cell edges from cell centers."""
    x = np.asarray(x, dtype=float)

    if x.size < 2:
        raise ValueError("Coordinate vector must contain at least two values.")

    mid = (x[:-1] + x[1:]) / 2.0
    first = x[0] - (mid[0] - x[0])
    last = x[-1] + (x[-1] - mid[-1])

    return np.concatenate([[first], mid, [last]])


def safe_divide(num, den):
    """Safe elementwise division."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)

    out = np.full(num.shape, np.nan, dtype=float)
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    out[ok] = num[ok] / den[ok]

    return out


def infer_basin_area_from_code_grid(code_grid, lon, lat, nbasin):
    """
    Estimate basin total area from regular lon-lat grid and basin_code_grid.
    Used only if basin_area_m2 is not available.
    """
    R = 6371000.0

    lon_e = grid_edges(lon)
    lat_e = grid_edges(lat)

    dlon = np.deg2rad(np.abs(np.diff(lon_e)))
    sinlat = np.sin(np.deg2rad(lat_e))
    d_sinlat = np.abs(np.diff(sinlat))

    cell_area = (R ** 2) * d_sinlat[:, None] * dlon[None, :]

    basin_area = np.zeros(nbasin, dtype=float)

    ok = np.isfinite(code_grid) & (code_grid >= 0)
    rr, cc = np.where(ok)

    idx = code_grid[rr, cc].astype(np.int64)
    valid = idx < nbasin

    np.add.at(basin_area, idx[valid], cell_area[rr[valid], cc[valid]])

    basin_area[basin_area <= 0] = np.nan
    return basin_area


def get_basin_area_m2(annual, z, nbasin):
    """Get basin total area if available; otherwise infer it."""
    candidate_keys = [
        "basin_area_m2",
        "basin_total_area_m2",
        "basin_land_area_m2",
        "basin_valid_area_m2",
        "basin_cell_area_m2",
    ]

    for k in candidate_keys:
        if k in annual.files:
            area = annual[k].astype(float)
            if area.size == nbasin:
                print(f"[Info] Using basin total area from annual product: {k}")
                return area

    print(
        "[Info] No basin-area variable found. "
        "Inferring basin area from basin_code_grid and lon/lat."
    )

    return infer_basin_area_from_code_grid(
        z["basin_code_grid"],
        z["lon"],
        z["lat"],
        nbasin,
    )


def basin_to_grid(values, code_grid, lon):
    """Map basin-level values to every grid cell in each represented basin."""
    values = np.asarray(values, dtype=float)
    grid = np.full(code_grid.shape, np.nan, dtype=float)

    ok = np.isfinite(code_grid) & (code_grid >= 0)
    rr, cc = np.where(ok)

    idx = code_grid[rr, cc].astype(np.int64)
    valid = idx < values.size

    grid[rr[valid], cc[valid]] = values[idx[valid]]

    return sort_lon(lon, grid)


def first_existing(npz, candidate_keys):
    """Return first available array and key from npz."""
    if npz is None:
        return None, None

    for k in candidate_keys:
        if k in npz.files:
            return npz[k].astype(float), k

    return None, None


# =============================================================================
# Load data
# =============================================================================

def load_fig2_products():
    """Load the audited native-grid Fig. 2 input product."""
    if not NATIVE_INPUT.exists():
        raise FileNotFoundError(f"Cannot find: {NATIVE_INPUT}")
    return np.load(NATIVE_INPUT, allow_pickle=False)


def load_crop_background():
    """
    Load gridded activity_fraction as cropland background.

    The original activity_fraction is used only to define cropland/activity cells.
    In the final figure, the background is drawn with a fixed color and does not
    encode activity_fraction magnitude.
    """
    if not CROP_GRID.exists():
        raise FileNotFoundError(f"Cannot find crop grid file: {CROP_GRID}")

    d = np.load(CROP_GRID)

    required = ["lat", "lon", "activity_fraction", "valid_mask", "basin_code_grid"]
    for k in required:
        if k not in d.files:
            raise KeyError(f"Missing key in crop grid file: {k}")

    lat = d["lat"].astype(float)
    lon = d["lon"].astype(float)
    activity_fraction = d["activity_fraction"].astype(float)
    valid_mask = d["valid_mask"].astype(bool)
    basin_code_grid = d["basin_code_grid"]

    # Valid cropland/activity domain.
    # valid_mask is expected to represent the cropland/activity exposure domain
    # used in the previous analysis.
    bg = np.where(
        valid_mask
        & np.isfinite(activity_fraction)
        & (activity_fraction > 0)
        & np.isfinite(basin_code_grid)
        & (basin_code_grid >= 0),
        activity_fraction,
        np.nan,
    )

    print("[Crop background] file =", CROP_GRID)
    print("[Crop background] activity_fraction shape =", activity_fraction.shape)
    print(
        "[Crop background] activity_fraction min/max =",
        float(np.nanmin(activity_fraction)),
        float(np.nanmax(activity_fraction)),
    )
    print("[Crop background] finite bg cells =", int(np.sum(np.isfinite(bg))))
    print(
        "[Crop background] bg percentiles 1/5/50/95/99 =",
        np.nanpercentile(bg, [1, 5, 50, 95, 99]),
    )

    lon_sorted, bg_sorted = sort_lon(lon, bg)
    return lon_sorted, lat, bg_sorted


def load_analysis_domain_mask():
    """Load the exact 1% cropland analysis domain used for map display."""
    if not CROP_GRID.exists():
        raise FileNotFoundError(f"Cannot find crop grid file: {CROP_GRID}")

    d = np.load(CROP_GRID)
    required = ["activity_fraction", "valid_mask", "basin_code_grid"]
    for k in required:
        if k not in d.files:
            raise KeyError(f"Missing key in crop grid file: {k}")

    activity_fraction = d["activity_fraction"].astype(float)
    valid_mask = d["valid_mask"].astype(bool)
    basin_code_grid = d["basin_code_grid"]

    mask = (
        valid_mask
        & np.isfinite(activity_fraction)
        & (activity_fraction > 0)
        & np.isfinite(basin_code_grid)
        & (basin_code_grid >= 0)
    )
    print("[Analysis domain] valid display cells =", int(np.sum(mask)))
    return mask


# =============================================================================
# Native-grid diagnostics
# =============================================================================

def native_component_differences(data):
    """Return the native-grid SSP3-7.0 minus SSP1-2.6 physical fields."""
    return {
        "N": np.asarray(data["delta_nitrate_N_loss_gN_m2_yr"], dtype=float),
        "I": np.asarray(
            data["delta_surface_irrigation_withdrawal_mm_yr"], dtype=float
        ),
        "Q": np.asarray(
            data["delta_irrigation_conditioned_runoff_mm_yr"], dtype=float
        ),
    }


# =============================================================================
# Country boundaries
# =============================================================================

def find_country_shapefile():
    """Find Natural Earth admin-0 country shapefile."""
    if not HAS_SHAPEFILE:
        raise ImportError("pyshp is not installed. Please install it with: pip install pyshp")

    if not NE_ROOT.exists():
        raise FileNotFoundError(f"Natural Earth root folder not found: {NE_ROOT}")

    shp_files = list(NE_ROOT.rglob("*.shp"))

    print(f"[Boundary] NE_ROOT = {NE_ROOT}")
    print(f"[Boundary] Number of .shp files found = {len(shp_files)}")

    for p in shp_files:
        print(f"[Boundary] Found shp: {p}")

    if len(shp_files) == 0:
        raise FileNotFoundError(f"No .shp file found under: {NE_ROOT}")

    preferred = [
        p for p in shp_files
        if "ne_110m_admin_0_countries" in p.name.lower()
        or "admin_0_countries" in p.name.lower()
    ]

    shp_path = preferred[0] if preferred else shp_files[0]

    shx_path = shp_path.with_suffix(".shx")
    dbf_path = shp_path.with_suffix(".dbf")

    if not shx_path.exists() or not dbf_path.exists():
        raise FileNotFoundError(
            "Found .shp file but missing companion files.\n"
            f"shp: {shp_path}\n"
            f"shx: {shx_path} | exists={shx_path.exists()}\n"
            f"dbf: {dbf_path} | exists={dbf_path.exists()}"
        )

    print(f"[Boundary] Using shapefile: {shp_path}")
    return shp_path


def _split_antimeridian(xs, ys):
    """Split line across dateline."""
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


def read_country_boundary_segments(shp_path):
    """Read country boundary line segments."""
    reader = shapefile.Reader(str(shp_path))
    shapes = reader.shapes()

    print(f"[Boundary] Number of country shapes = {len(shapes)}")

    segments = []

    for shp in shapes:
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

    print(f"[Boundary] Number of boundary line segments = {len(segments)}")

    if len(segments) == 0:
        raise RuntimeError("No country boundary line segments generated.")

    return segments


def add_country_boundaries(ax, boundary_segments):
    """Add thin, subdued country boundaries."""
    # Drawing the small Natural Earth segment set directly avoids a Windows
    # Matplotlib LineCollection transform crash observed in the production
    # environment, without changing the boundary geometry or appearance.
    for segment in boundary_segments:
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color="#555555",
            linewidth=0.16,
            alpha=0.72,
            zorder=1000,
            clip_on=True,
            solid_capstyle="round",
            solid_joinstyle="round",
        )


# =============================================================================
# Colormaps
# =============================================================================

def make_discrete_cmap(
    name,
    colors,
    bad=(1, 1, 1, 0),
    alphas=None,
    under=None,
    over=None,
):
    """
    Create discrete colormap with separate under/over colors.

    Normal colors correspond to values within the colorbar bounds.
    Under/over colors are used for extension triangles and out-of-range values.
    """
    if alphas is not None:
        rgba = []
        for c, a in zip(colors, alphas):
            r, g, b, _ = to_rgba(c)
            rgba.append((r, g, b, a))
        colors = rgba

    cmap = ListedColormap(colors, name=name)
    cmap.set_bad(bad)

    if under is None:
        cmap.set_under(colors[0])
    else:
        cmap.set_under(under)

    if over is None:
        cmap.set_over(colors[-1])
    else:
        cmap.set_over(over)

    return cmap


# a. Irrigated-area fraction difference
CMAP_FRAC = make_discrete_cmap(
    "teal_brown_fraction_with_extensions",
    [
        "#0B7F79",
        "#4DAFA5",
        "#A9DCD6",
        "#FFFFFF",
        "#EFE1B5",
        "#D69A36",
        "#9C5D08",
    ],
    bad=(1, 1, 1, 0),
    alphas=[
        0.98,
        0.94,
        0.82,
        0.00,
        0.82,
        0.94,
        0.98,
    ],
    under="#004B49",
    over="#5C2F00",
)

# b. Nitrate-N loss change
CMAP_N = make_discrete_cmap(
    "nloss_teal_brown_with_extensions",
    [
        "#0A7C78",
        "#60BDB4",
        "#C8E9E4",
        "#FFFFFF",
        "#EED79A",
        "#C98A2B",
        "#8C510A",
    ],
    bad=(1, 1, 1, 0),
    under="#004B49",
    over="#5A2F00",
)

# c. Irrigation water withdrawal change
CMAP_I = make_discrete_cmap(
    "iww_green_purple_with_extensions",
    [
        "#1B7F3A",
        "#66B96F",
        "#D7EFD3",
        "#FFFFFF",
        "#E6CFE8",
        "#A56AB2",
        "#5B176D",
    ],
    bad=(1, 1, 1, 0),
    under="#075A22",
    over="#3B004F",
)

# d. Water availability change
CMAP_QCHANGE = make_discrete_cmap(
    "qchange_purple_green_with_extensions",
    [
        "#6A1B78",
        "#A56AB2",
        "#E3CBE8",
        "#FFFFFF",
        "#D7EFD3",
        "#69B96E",
        "#2F8B45",
    ],
    bad=(1, 1, 1, 0),
    under="#3B0050",
    over="#075A22",
)

BOUNDS_FRAC = np.array(
    [-0.20, -0.10, -0.05, -0.01, 0.01, 0.05, 0.10, 0.20],
    dtype=float,
)

BOUNDS_N = np.array(
    [-5.0, -2.0, -1.0, -0.1, 0.1, 1.0, 2.0, 5.0],
    dtype=float,
)

BOUNDS_WATER = np.array(
    [-100, -50, -20, -5, 5, 20, 50, 100],
    dtype=float,
)

BOUNDS_Q = np.array(
    [-300, -100, -50, -25, 25, 50, 100, 300],
    dtype=float,
)


# =============================================================================
# Plot style
# =============================================================================

MAP_XLIM = (-180, 180)
MAP_YLIM = (-63, 87)


def style_map(ax):
    """Style one global map panel."""
    ax.set_xlim(*MAP_XLIM)
    ax.set_ylim(*MAP_YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")

    ax.set_xticks([])
    ax.set_yticks([])

    for name, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.25)
        spine.set_edgecolor("#444444")
        spine.set_zorder(3000)
        spine.set_snap(True)

    ax.set_facecolor("white")


def add_crop_activity_background(ax, crop_bg):
    """
    Add fixed-color cropland/activity background.

    The background only indicates where cropland/activity cells exist.
    It does not encode the magnitude of activity_fraction.
    """
    bg_lon, bg_lat, bg_grid = crop_bg

    crop_mask = np.where(np.isfinite(bg_grid), 1.0, np.nan)
    crop_masked = np.ma.masked_invalid(crop_mask)

    crop_cmap = ListedColormap([CROP_BG_COLOR])
    crop_cmap.set_bad((1, 1, 1, 0))

    ax.pcolormesh(
        bg_lon,
        bg_lat,
        crop_masked,
        cmap=crop_cmap,
        vmin=0.5,
        vmax=1.5,
        shading="nearest",
        rasterized=True,
        alpha=CROP_BG_ALPHA,
        zorder=2,
    )


def add_crop_background_legend(ax):
    """
    Add a subtle legend for the fixed cropland/activity background in panel a.

    The legend color is identical to the cropland background color.
    """
    handle = Patch(
        facecolor=CROP_BG_COLOR,
        edgecolor="none",
        alpha=CROP_BG_ALPHA,
        label="Cropland",
    )

    leg = ax.legend(
        handles=[handle],
        loc="lower left",
        bbox_to_anchor=(0.018, 0.038),
        frameon=True,
        fontsize=5.3,
        handlelength=2.25,
        handleheight=0.52,
        borderpad=0.18,
        labelspacing=0.16,
        handletextpad=0.40,
        borderaxespad=0.0,
    )

    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("none")
    frame.set_alpha(0.70)

    for text in leg.get_texts():
        text.set_color("#5A5A5A")

    leg.set_zorder(2500)


def add_panel_label(ax, label, title, right_label=None):
    """Add panel label and panel title."""
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


def add_aligned_colorbar(fig, ax, im, ticks, label):
    """Add colorbar aligned with map frame."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4.2%", pad=0.032)

    cb = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        extend="both",
        extendfrac=0.075,
        ticks=ticks,
    )

    cb.outline.set_linewidth(0.35)
    cb.ax.tick_params(labelsize=6.1, length=1.8, width=0.35, pad=0.8)
    # DejaVu Sans contains the Unicode superscript-minus glyphs used in the
    # physical units, avoiding the displaced mathtext exponents seen with Arial.
    cb.set_label(
        label,
        fontsize=6.7,
        labelpad=0.9,
        fontfamily="DejaVu Sans",
    )
    return cb


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
    boundary_segments,
    right_label=None,
    crop_bg=None,
    panel_alpha=1.0,
):
    """Plot one map panel."""
    style_map(ax)

    if crop_bg is not None:
        add_crop_activity_background(ax, crop_bg)

    norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=False)

    im = ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_invalid(grid),
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
        alpha=panel_alpha,
        zorder=3,
    )

    add_country_boundaries(ax, boundary_segments)
    add_panel_label(ax, label, title, right_label=right_label)
    add_aligned_colorbar(fig, ax, im, bounds, cbar_label)

    return im


def restrict_to_crop_mask_if_compatible(grid, code_grid):
    """Mask gridded data to crop-containing basin cells when shapes match."""
    if grid.shape == code_grid.shape:
        return np.where(code_grid >= 0, grid, np.nan)

    warnings.warn(
        f"Panel-a TECH grid shape {grid.shape} differs from basin_code_grid shape {code_grid.shape}. "
        "Irrigation difference is not masked to crop-containing basins."
    )
    return grid


# =============================================================================
# Main figure
# =============================================================================

def plot_fig2_component_diagnostics():
    """Draw Fig. 2 from native-grid physical component differences."""
    shp_path = find_country_shapefile()
    boundary_segments = read_country_boundary_segments(shp_path)
    if not NATIVE_INPUT.exists():
        raise FileNotFoundError(
            f"Missing native-grid input: {NATIVE_INPUT}. "
            "Run build_Fig02_native_grid_inputs.py first."
        )
    data = np.load(NATIVE_INPUT, allow_pickle=False)
    print("[Stage] Loaded native-grid input", flush=True)
    lon = np.asarray(data["lon"], dtype=float)
    lat = np.asarray(data["lat"], dtype=float)

    # Panel a defines the agricultural exposure domain used for the main figure.
    irrigation_difference = np.asarray(
        data["irrigation_fraction_difference"], dtype=float
    )
    irrigation_visible = np.isfinite(irrigation_difference)
    irrigation_magnitude = np.zeros_like(irrigation_difference)
    irrigation_magnitude[irrigation_visible] = np.abs(
        irrigation_difference[irrigation_visible]
    )
    irrigation_visible &= irrigation_magnitude >= 0.01
    lon_a, grid_a = sort_lon(
        lon,
        np.where(irrigation_visible, irrigation_difference, np.nan),
    )
    _, crop_grid = sort_lon(lon, np.asarray(data["cropland_background"], dtype=float))
    crop_bg = (lon_a, lat, crop_grid)

    # Component differences remain native-grid fields. Panels b and d are
    # displayed only in the agricultural exposure domain; panel c retains the
    # withdrawal field's own spatial support.
    lon_n, grid_n = sort_lon(
        lon, np.asarray(data["delta_nitrate_N_loss_gN_m2_yr"], dtype=float)
    )
    lon_i, grid_i = sort_lon(
        lon,
        np.asarray(data["delta_surface_irrigation_withdrawal_mm_yr"], dtype=float),
    )
    lon_q, grid_q = sort_lon(
        lon,
        np.asarray(data["delta_irrigation_conditioned_runoff_mm_yr"], dtype=float),
    )
    if not (np.allclose(lon_n, lon_a) and np.allclose(lon_q, lon_a)):
        raise AssertionError("Longitude ordering differs between Fig. 2 panels")
    agricultural_domain = np.isfinite(crop_grid)
    grid_n = np.where(agricultural_domain, grid_n, np.nan)
    grid_q = np.where(agricultural_domain, grid_q, np.nan)
    print("[Stage] Prepared four panel grids", flush=True)

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
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

    # Panel a
    plot_map_panel(
        fig,
        axes[0],
        lon_a,
        lat,
        grid_a,
        "a",
        "Cropland extent and irrigation expansion",
        CMAP_FRAC,
        BOUNDS_FRAC,
        "Δ irrigated-area fraction",
        boundary_segments=boundary_segments,
        right_label="SSP3-7.0 - SSP1-2.6",
        crop_bg=crop_bg,
        panel_alpha=0.88,
    )
    add_crop_background_legend(axes[0])
    print("[Stage] Panel a complete", flush=True)

    # Panel b
    plot_map_panel(
        fig,
        axes[1],
        lon_n,
        lat,
        grid_n,
        "b",
        "Nitrate-N loss change",
        CMAP_N,
        BOUNDS_N,
        "ΔN (g N m⁻² yr⁻¹)",
        boundary_segments=boundary_segments,
        right_label="SSP3-7.0 - SSP1-2.6",
    )
    print("[Stage] Panel b complete", flush=True)

    # Panel c
    plot_map_panel(
        fig,
        axes[2],
        lon_i,
        lat,
        grid_i,
        "c",
        "Irrigation water withdrawal change",
        CMAP_I,
        BOUNDS_WATER,
        "ΔI (mm yr⁻¹)",
        boundary_segments=boundary_segments,
        right_label="SSP3-7.0 - SSP1-2.6",
    )
    print("[Stage] Panel c complete", flush=True)

    # Panel d
    plot_map_panel(
        fig,
        axes[3],
        lon_q,
        lat,
        grid_q,
        "d",
        "Irrigation-conditioned runoff change",
        CMAP_QCHANGE,
        BOUNDS_Q,
        "ΔQ (mm yr⁻¹)",
        boundary_segments=boundary_segments,
        right_label="SSP3-7.0 - SSP1-2.6",
    )
    print("[Stage] Panel d complete", flush=True)

    png = FIG_MAIN / "Fig02_component_diagnostics.png"
    pdf = FIG_MAIN / "Fig02_component_diagnostics.pdf"
    svg = FIG_MAIN / "Fig02_component_diagnostics.svg"

    print("[Stage] Saving PNG", flush=True)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.018)
    print("[Stage] Saving PDF", flush=True)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.018)
    print("[Stage] Saving SVG", flush=True)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.018)

    plt.close(fig)

    print("[Saved]")
    print(png)
    print(pdf)
    print(svg)


if __name__ == "__main__":
    plot_fig2_component_diagnostics()



