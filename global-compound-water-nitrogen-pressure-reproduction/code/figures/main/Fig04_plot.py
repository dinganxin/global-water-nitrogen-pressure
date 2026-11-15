# -*- coding: utf-8 -*-
"""
Final Fig. 4 | Historical-to-future pathways into emerging compound water-nitrogen pressure.

This script uses the version-B NPI-IPI basin products:
    Nload = sum(Nloss_grid * grid-cell area)
    NPI   = Nload / Q_IRR, for Q_IRR > 0
    IPI   = IWW / Q_IRR, for Q_IRR > 0

For Q_IRR <= 0, both ratios are undefined and the exposed cropland is retained
as an independent hydrological limiting state outside the four pressure classes.

Figure logic:
    a,b. Historical-to-future transitions with area conservation:
         historical cropland classes plus expanded cropland -> future classes
         plus contracted cropland.

This two-panel version only keeps:
    a. SSP1-2.6
    b. SSP3-7.0

Notes:
    1. Contracted cropland is shown in grey-brown to distinguish it from
       expanded cropland.
    2. The percentage in the red annotation is:
       expansion-derived Emerging Compound / total Emerging Compound.
    3. Expanded and Contracted denote net changes in cropland exposure at the
       HydroBASINS level-5 scale, not pixel-level land-conversion trajectories.
    4. Non-emerging flows <= 0.10 million km2 are omitted to reduce visual
       clutter; emerging flows > 0.006 million km2 are retained.
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox


# =============================================================================
# Paths
# =============================================================================

OUT = Path(__file__).resolve().parents[3]

FIG_DIR = OUT / "figures" / "main"
TABLE_DIR = OUT / "tables" / "main"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

PRODUCT_PATH = OUT / "data" / "NPI_IPI_QIRR_strict_positiveQ_period_products_01pct.npz"


# =============================================================================
# Global style
# =============================================================================

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.0,
        "axes.linewidth": 0.35,
        "axes.edgecolor": "#333333",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 600,
        "mathtext.default": "regular",
    }
)


# =============================================================================
# Colors and labels
# =============================================================================

CLASS_COLORS = {
    0: "#E7E8EA",  # Low pressure; matched to Fig. 3
    1: "#D6A33D",  # Nitrate-pressure only; matched to Fig. 3
    2: "#4E8FB8",  # Irrigation-pressure only; matched to Fig. 3
    3: "#C95B55",  # Compound; matched to Fig. 3
    4: "#D1D4D6",  # Non-positive-Q hydrological limiting state
    5: "#7B946A",  # Expanded cropland
    6: "#B7B1AA",  # Contracted cropland
}

CLASS_LABELS = {
    0: "Low",
    1: "N-only",
    2: "I-only",
    3: "Compound",
    4: r"$Q\leq0$",
    5: "Expanded",
    6: "Contracted",
}

LIMITING_STATE_DOT = "#9DA3A7"


class QMaskLegendHandle:
    """Legend-only handle for the independent Q<=0 limiting state."""

    def __init__(self, label: str):
        self._label = label

    def get_label(self) -> str:
        return self._label


class HandlerQMask(HandlerBase):
    """Draw a limiting-state rectangle with one restrained centred dot."""

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
            facecolor=CLASS_COLORS[4],
            edgecolor="none",
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


# =============================================================================
# Data utilities
# =============================================================================

def weights_million_km2(z: np.lib.npyio.NpzFile, period: str) -> np.ndarray:
    """Return basin cropland area in million km2."""
    return np.asarray(z[f"basin_crop_area_{period}_m2"], dtype=np.float64) / 1e12


def class_array(z: np.lib.npyio.NpzFile, experiment: str, period: str) -> np.ndarray:
    """Return basin class array for a given experiment and period."""
    return np.asarray(z[f"class_{experiment}_{period}_basin"], dtype=np.int16)


def pathway_state_array(z: np.lib.npyio.NpzFile, period: str) -> np.ndarray:
    """Return four strict classes plus the independent limiting state (code 4)."""
    state = class_array(z, "IRR", period).copy()
    limiting = np.asarray(
        z[f"nonpositive_Q_limiting_state_{period}_basin"],
        dtype=bool,
    )
    state[limiting] = 4
    return state


def temporal_expanded_matrix(z: np.lib.npyio.NpzFile, future_period: str) -> np.ndarray:
    """
    Historical + expanded cropland -> future + contracted cropland.

    Persistent cropland in each basin is represented by:
        min(historical cropland area, future cropland area).

    Future cropland in excess of historical cropland is assigned to the
    Expanded source node. Historical cropland in excess of future cropland is
    assigned to the Contracted target node.
    """
    hist_w = weights_million_km2(z, "historical")
    fut_w = weights_million_km2(z, future_period)

    overlap = np.minimum(hist_w, fut_w)
    expanded = np.maximum(fut_w - hist_w, 0.0)
    contracted = np.maximum(hist_w - fut_w, 0.0)

    hist_cls = pathway_state_array(z, "historical")
    fut_cls = pathway_state_array(z, future_period)

    # Rows: four strict classes, limiting state, Expanded.
    # Columns: four strict classes, limiting state, Contracted.
    mat = np.zeros((6, 6), dtype=float)

    hist_valid = np.isfinite(hist_w) & (hist_w > 0) & (hist_cls >= 0)
    fut_valid = np.isfinite(fut_w) & (fut_w > 0) & (fut_cls >= 0)
    both_valid = hist_valid & fut_valid

    # Persistent cropland: historical class -> future class.
    # Basins without a valid future class are not dropped; their historical
    # evaluated cropland is assigned to the contracted target below.
    for i in range(5):
        for j in range(5):
            mat[i, j] = float(np.sum(overlap[both_valid & (hist_cls == i) & (fut_cls == j)]))

    # Expanded cropland source -> future classes. If a basin enters the
    # evaluated future domain from an unevaluated historical state, all future
    # evaluated cropland is treated as expanded exposure.
    for j in range(5):
        mat[5, j] = float(
            np.sum(expanded[both_valid & (fut_cls == j)])
            + np.sum(fut_w[(~hist_valid) & fut_valid & (fut_cls == j)])
        )

    # Historical classes -> contracted cropland target. If a basin leaves the
    # evaluated future domain, all historical evaluated cropland is assigned to
    # contracted exposure to conserve source-class area.
    for i in range(5):
        mat[i, 5] = float(
            np.sum(contracted[both_valid & (hist_cls == i)])
            + np.sum(hist_w[hist_valid & (~fut_valid) & (hist_cls == i)])
        )

    source_total = float(
        np.sum(hist_w[hist_valid])
        + np.sum(expanded[both_valid])
        + np.sum(fut_w[(~hist_valid) & fut_valid])
    )
    target_total = float(
        np.sum(fut_w[fut_valid])
        + np.sum(contracted[both_valid])
        + np.sum(hist_w[hist_valid & (~fut_valid)])
    )
    if not np.isclose(mat.sum(), source_total, atol=1.0e-9, rtol=0.0):
        raise ValueError("Transition matrix does not conserve source exposure area.")
    if not np.isclose(mat.sum(), target_total, atol=1.0e-9, rtol=0.0):
        raise ValueError("Transition matrix does not conserve target exposure area.")
    return mat


def strict_emerging_metrics(z: np.lib.npyio.NpzFile, future_period: str) -> tuple[float, float]:
    """Return emerging compound area and its expansion-derived component.

    Emerging compound follows the manuscript-wide basin definition: future
    compound cropland in basins that were not compound during 1985-2014.
    For basins outside the historical agricultural domain, all future
    cropland is treated as expanded exposure.
    """
    hist_w = weights_million_km2(z, "historical")
    fut_w = weights_million_km2(z, future_period)
    hist_cls = pathway_state_array(z, "historical")
    fut_cls = pathway_state_array(z, future_period)

    eligible = np.isfinite(fut_w) & (fut_cls == 3) & (hist_cls != 3)
    hist_valid = np.isfinite(hist_w) & (hist_w > 0) & (hist_cls >= 0)
    expanded = np.where(hist_valid, np.maximum(fut_w - hist_w, 0.0), fut_w)
    return float(np.sum(fut_w[eligible])), float(np.sum(expanded[eligible]))


# =============================================================================
# Geometry helpers
# =============================================================================

def stack_segments(
    totals: np.ndarray,
    y0: float,
    y1: float,
    gap: float,
    order: list[int] | None = None,
) -> dict[int, tuple[float, float]]:
    """Convert node totals to stacked vertical node segments."""
    if order is None:
        order = list(range(len(totals)))

    active = [i for i in order if i < len(totals) and totals[i] > 1e-9]

    height_total = y1 - y0 - gap * max(len(active) - 1, 0)
    denom = float(np.sum(totals[active])) if active else 1.0

    segments: dict[int, tuple[float, float]] = {}

    y = y1
    for i in active:
        h = height_total * totals[i] / denom
        segments[i] = (y - h, y)
        y -= h + gap

    return segments


def node_mid_positions(segs: dict[int, tuple[float, float]]) -> dict[int, float]:
    """Place each label at the vertical centre of the corresponding node."""
    return {i: (ya + yb) / 2 for i, (ya, yb) in segs.items()}


def draw_flow(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y0a: float,
    y0b: float,
    y1a: float,
    y1b: float,
    color: str,
    alpha: float,
    zorder: int = 2,
) -> None:
    """Draw one alluvial flow."""
    dx = x1 - x0
    c0 = x0 + dx * 0.42
    c1 = x1 - dx * 0.42

    verts = [
        (x0, y0a),
        (c0, y0a),
        (c1, y1a),
        (x1, y1a),
        (x1, y1b),
        (c1, y1b),
        (c0, y0b),
        (x0, y0b),
        (x0, y0a),
    ]

    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]

    ax.add_patch(
        PathPatch(
            MplPath(verts, codes),
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=zorder,
        )
    )


def draw_node(
    ax: plt.Axes,
    x: float,
    ya: float,
    yb: float,
    color: str,
    is_limiting_state: bool = False,
) -> None:
    """Draw one node rectangle."""
    ax.add_patch(
        Rectangle(
            (x - 0.030, ya),
            0.060,
            yb - ya,
            facecolor=color,
            edgecolor="white",
            linewidth=0.60,
            zorder=5,
        )
    )
    if is_limiting_state:
        ax.plot(
            [x],
            [(ya + yb) / 2.0],
            marker="o",
            markersize=1.25,
            markerfacecolor=LIMITING_STATE_DOT,
            markeredgecolor=LIMITING_STATE_DOT,
            markeredgewidth=0,
            alpha=0.78,
            linestyle="none",
            zorder=6,
        )


def is_key_flow(i: int, j: int) -> bool:
    """Identify key flows to highlight in red."""
    # Historical-to-future: any non-compound source becoming Compound.
    return (j == 3) and (i != 3)


def flow_threshold(value: float, is_key: bool) -> bool:
    """
    Decide whether a flow should be drawn.

    Higher thresholds are used for non-key flows to reduce visual fog.
    Key red flows are retained even when relatively small.
    """
    if is_key:
        return value > 0.006

    return value > 0.10


# =============================================================================
# Alluvial panel
# =============================================================================

def plot_alluvial(
    ax: plt.Axes,
    mat: np.ndarray,
    panel: str,
    title: str,
    left_title: str,
    right_title: str,
    emerging_value: float,
    expansion_value: float,
) -> None:
    """Draw one temporal alluvial panel."""
    n_left, n_right = mat.shape

    left_tot = mat.sum(axis=1)
    right_tot = mat.sum(axis=0)

    y0, y1 = 0.145, 0.820

    left_order = [0, 5, 4, 1, 2, 3]
    right_order = [0, 5, 4, 1, 2, 3]

    left_gap = 0.032
    # Identical gaps keep the area scale, and therefore every flow width,
    # exactly equal at the source and target nodes.
    right_gap = left_gap

    left_seg = stack_segments(left_tot, y0, y1, gap=left_gap, order=left_order)
    right_seg = stack_segments(right_tot, y0, y1, gap=right_gap, order=right_order)

    left_scale = {
        k: (left_seg[k][1] - left_seg[k][0]) / left_tot[k]
        for k in left_seg
        if left_tot[k] > 0
    }

    right_scale = {
        k: (right_seg[k][1] - right_seg[k][0]) / right_tot[k]
        for k in right_seg
        if right_tot[k] > 0
    }

    left_cur = {k: v[0] for k, v in left_seg.items()}
    right_cur = {k: v[0] for k, v in right_seg.items()}

    # Red flow widths must follow the strict emerging-hotspot definition.
    # Expanded -> Compound also contains expansion inside basins that were
    # already Compound historically; that component is not emerging.
    strict_red_total = float(np.sum(mat[[0, 1, 2, 4], 3]) + expansion_value)
    if not np.isclose(strict_red_total, emerging_value, rtol=0.0, atol=1.0e-8):
        raise ValueError(
            "Strict emerging-flow mismatch: "
            f"red flows={strict_red_total:.12f}, metric={emerging_value:.12f}"
        )
    if expansion_value < -1.0e-12 or expansion_value > mat[5, 3] + 1.0e-8:
        raise ValueError(
            "Expansion-derived emerging area must lie within the total "
            "Expanded -> Compound flow."
        )

    # Keep the original area-ranked flow geometry at every source node and at
    # all future nodes except Compound. Only the right-hand Compound endpoints
    # are packed independently, so the incoming pathways remain contiguous
    # without changing the established appearance of the rest of the alluvial.
    strict_expanded = float(np.clip(expansion_value, 0.0, mat[5, 3]))
    established_expanded = float(mat[5, 3] - strict_expanded)
    compound_components = [
        ((0, "regular"), float(mat[0, 3])),
        ((5, "expanded_emerging"), strict_expanded),
        ((1, "regular"), float(mat[1, 3])),
        ((2, "regular"), float(mat[2, 3])),
        ((4, "regular"), float(mat[4, 3])),
        ((5, "expanded_established"), established_expanded),
        ((3, "regular"), float(mat[3, 3])),
    ]
    compound_right_bounds: dict[tuple[int, str], tuple[float, float]] = {}
    compound_cursor = right_seg[3][1]
    for component_key, component_value in compound_components:
        if component_value <= 0:
            continue
        height = component_value * right_scale[3]
        compound_right_bounds[component_key] = (
            compound_cursor - height,
            compound_cursor,
        )
        compound_cursor -= height

    if not np.isclose(compound_cursor, right_seg[3][0], rtol=0.0, atol=1.0e-10):
        raise ValueError("Packed Compound endpoints do not conserve node area.")

    # Draw large flows first, as in the preferred pre-reordering version.
    pairs = [(i, j, float(mat[i, j])) for i in range(n_left) for j in range(n_right)]
    pairs.sort(key=lambda t: t[2], reverse=True)

    for i, j, value in pairs:
        if value <= 0 or i not in left_seg or j not in right_seg:
            continue

        key = is_key_flow(i, j)
        y0a = left_cur[i]
        y0b = y0a + value * left_scale[i]
        left_cur[i] = y0b

        if j != 3:
            y1a = right_cur[j]
            y1b = y1a + value * right_scale[j]
            right_cur[j] = y1b
        elif i != 5:
            y1a, y1b = compound_right_bounds[(i, "regular")]

        # Split Expanded -> Compound into strict emerging and established
        # expansion components. Their source geometry remains exactly where the
        # old area-ranked layout placed the full flow; only their Compound-node
        # endpoints are packed into the compact right-hand order.
        if i == 5 and j == 3:
            y0_split = y0a + strict_expanded * left_scale[i]
            strict_y1a, strict_y1b = compound_right_bounds[(5, "expanded_emerging")]
            established_y1a, established_y1b = compound_right_bounds[
                (5, "expanded_established")
            ]

            if established_expanded > 0.006:
                draw_flow(
                    ax,
                    0.235,
                    0.765,
                    y0_split,
                    y0b,
                    established_y1a,
                    established_y1b,
                    CLASS_COLORS[5],
                    0.075,
                    zorder=2,
                )

            if strict_expanded > 0.006:
                draw_flow(
                    ax,
                    0.235,
                    0.765,
                    y0a,
                    y0_split,
                    strict_y1a,
                    strict_y1b,
                    CLASS_COLORS[3],
                    0.88,
                    zorder=4,
                )
            continue

        if not flow_threshold(value, key):
            continue

        if key:
            color = CLASS_COLORS[3]
            alpha = 0.88
            zorder = 4
        else:
            if j == 5:
                color = CLASS_COLORS[6]  # Contracted cropland
            elif i == 5:
                color = CLASS_COLORS[5]  # Expanded cropland
            else:
                color = CLASS_COLORS.get(j, "#999999")
            alpha = 0.075
            zorder = 2

        draw_flow(
            ax,
            0.235,
            0.765,
            y0a,
            y0b,
            y1a,
            y1b,
            color,
            alpha,
            zorder=zorder,
        )

    # Left/right node headers.
    ax.text(
        0.16,
        0.875,
        left_title,
        ha="center",
        va="bottom",
        fontsize=6.35,
        fontweight="semibold",
        color="#2F2F2F",
        linespacing=0.95,
    )

    ax.text(
        0.84,
        0.875,
        right_title,
        ha="center",
        va="bottom",
        fontsize=6.35,
        fontweight="semibold",
        color="#2F2F2F",
        linespacing=0.95,
    )

    # Draw nodes and labels.
    for side, x, segs, totals in [
        ("left", 0.16, left_seg, left_tot),
        ("right", 0.84, right_seg, right_tot),
    ]:
        pos = node_mid_positions(segs)

        for i, (ya, yb) in segs.items():
            # Index 5 denotes Expanded on the source side and Contracted on
            # the target side; index 4 is the hydrological limiting state.
            if side == "right" and i == 5:
                label_id = 6
            elif side == "left" and i == 5:
                label_id = 5
            else:
                label_id = i

            draw_node(
                ax,
                x,
                ya,
                yb,
                CLASS_COLORS[label_id],
                is_limiting_state=(label_id == 4),
            )

            ha = "right" if side == "left" else "left"

            # Slightly larger horizontal offset gives one-line labels more room.
            tx = x - 0.052 if side == "left" else x + 0.052

            text = f"{totals[i]:.2f}"

            ax.text(
                tx,
                pos[i],
                text,
                ha=ha,
                va="center",
                fontsize=5.25,
                color="#222222",
                linespacing=0.90,
            )

    # Key metric text.
    key_value = emerging_value
    expanded_share = expansion_value / emerging_value * 100.0 if emerging_value > 0 else np.nan

    key_text = (
        f"Emerging compound: {key_value:.2f} million km² "
        f"({expanded_share:.1f}% from expansion)"
    )

    ax.text(
        0.50,
        0.020,
        key_text,
        ha="center",
        va="center",
        fontsize=5.70,
        color=CLASS_COLORS[3],
        fontweight="semibold",
    )

    # Panel label and short title.
    ax.text(
        0.000,
        0.980,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color="#111111",
    )

    ax.text(
        0.052,
        0.980,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.35,
        color="#111111",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# =============================================================================
# Summary table
# =============================================================================

def transition_label(i: int, j: int) -> str:
    """Return display label for temporal transition table."""
    left_label = CLASS_LABELS.get(i, str(i))

    if j == 5:
        right_label = "Contracted"
    else:
        right_label = CLASS_LABELS.get(j, str(j))

    return f"{left_label} -> {right_label}"


def write_summary_table(
    mats: dict[str, np.ndarray],
    strict_metrics: dict[str, tuple[float, float]],
) -> Path:
    """Write a transition summary table for two-panel Fig. 4."""
    path = (
        TABLE_DIR
        / "Fig04_temporal_transition_summary_strict_positiveQ_01pct.csv"
    )

    rows = []

    for panel, mat in mats.items():
        emerging, expanded_value = strict_metrics[panel]
        expanded_share = expanded_value / emerging * 100.0 if emerging > 0 else np.nan

        rows.append(
            {
                "panel": panel,
                "metric": "Emerging Compound",
                "value_million_km2": emerging,
                "share_percent": "",
            }
        )

        rows.append(
            {
                "panel": panel,
                "metric": "Expansion-derived Emerging Compound",
                "value_million_km2": expanded_value,
                "share_percent": expanded_share,
            }
        )

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] > 0.0:
                    rows.append(
                        {
                            "panel": panel,
                            "metric": transition_label(i, j),
                            "value_million_km2": float(mat[i, j]),
                            "share_percent": "",
                        }
                    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "panel",
                "metric",
                "value_million_km2",
                "share_percent",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


# =============================================================================
# Main plotting
# =============================================================================

def plot_fig4ab(z: np.lib.npyio.NpzFile) -> None:
    """Plot Fig. 4a,b only."""
    mats = {
        "a": temporal_expanded_matrix(z, "ssp126"),
        "b": temporal_expanded_matrix(z, "ssp370"),
    }
    strict_metrics = {
        "a": strict_emerging_metrics(z, "ssp126"),
        "b": strict_emerging_metrics(z, "ssp370"),
    }

    table = write_summary_table(mats, strict_metrics)

    fig = plt.figure(figsize=(7.55, 3.05), dpi=600, facecolor="white")

    axes = {
        "a": fig.add_axes([0.065, 0.130, 0.415, 0.720]),
        "b": fig.add_axes([0.460, 0.130, 0.415, 0.720]),
    }

    plot_alluvial(
        axes["a"],
        mats["a"],
        "a",
        "Historical-to-future pathways, SSP1-2.6",
        "Source class",
        "Future class",
        *strict_metrics["a"],
    )

    plot_alluvial(
        axes["b"],
        mats["b"],
        "b",
        "Historical-to-future pathways, SSP3-7.0",
        "Source class",
        "Future class",
        *strict_metrics["b"],
    )

    legend_order = [0, 5, 6, 4, 1, 2, 3]
    legend_handles = []
    for i in legend_order:
        if i == 4:
            legend_handles.append(QMaskLegendHandle(CLASS_LABELS[i]))
        else:
            legend_handles.append(
                Patch(
                    facecolor=CLASS_COLORS[i],
                    edgecolor="none",
                    label=CLASS_LABELS[i],
                )
            )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.407, 0.052),
        ncol=7,
        frameon=False,
        fontsize=4.85,
        handlelength=1.60,
        handleheight=0.62,
        handletextpad=0.34,
        columnspacing=0.95,
        borderaxespad=0.0,
        handler_map={QMaskLegendHandle: HandlerQMask()},
    )

    base = "Fig04_historical_to_future_pathways"

    png = FIG_DIR / f"{base}.png"
    pdf = FIG_DIR / f"{base}.pdf"
    svg = FIG_DIR / f"{base}.svg"

    # Lock the original panel bounds so moving the legend does not recrop the figure.
    export_bbox = Bbox.from_bounds(0.46075, 0.36510, 6.17550, 2.32848)
    fig.savefig(png, bbox_inches=export_bbox)
    fig.savefig(pdf, bbox_inches=export_bbox)
    fig.savefig(svg, bbox_inches=export_bbox)

    plt.close(fig)

    print("[Saved]")
    print(png)
    print(pdf)
    print(svg)
    print(table)


def main() -> None:
    """Run Fig. 4a,b generation."""
    if not PRODUCT_PATH.exists():
        raise FileNotFoundError(PRODUCT_PATH)

    z = np.load(PRODUCT_PATH)

    plot_fig4ab(z)


if __name__ == "__main__":
    main()



