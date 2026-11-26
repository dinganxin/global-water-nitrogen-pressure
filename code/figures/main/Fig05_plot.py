# -*- coding: utf-8 -*-
"""
Fig. 5 with regional Compound and non-positive-Q exposure shown separately.

Outputs:
    1. Revised four-panel Fig. 5:
       a Regional exposure (Compound and non-positive-Q state)
       b Pressure fingerprints
       c Basin-scale concentration
       d Member agreement (strict 3/3 support)

    2. Standalone final Fig. 5d panel:
       Strict three-member agreement only

This script does not overwrite previous Fig. 5 versions or products.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.transforms import blended_transform_factory


OUT = Path(__file__).resolve().parents[3]
FIG_DIR = OUT / "figures" / "main"
TABLE_DIR = OUT / "tables" / "main"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

REGION_TABLE = OUT / "tables" / "IPCC_AR6_region_class_summary_strict_positiveQ.csv"
GLOBAL_TABLE = OUT / "tables" / "global_class_area_summary_strict_positiveQ.csv"
PRODUCT_PATH = OUT / "data" / "NPI_IPI_QIRR_strict_positiveQ_period_products_01pct.npz"
META_PATH = OUT / "data" / "basin_metadata" / "HydroBASINS_level5_metadata_and_IPCC_assignment.csv"

NPI_THRESHOLD = 10.0
IPI_THRESHOLD = 0.4
TOP_N_REGION = 10


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.0,
        "axes.linewidth": 0.45,
        "axes.edgecolor": "#333333",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 600,
        "mathtext.default": "regular",
    }
)

COL_TEXT = "#222222"
COL_MUTED = "#6B7280"
COL_GRID = "#EDF1F4"
COL_SSP1 = "#CBB8B2"
COL_SSP3 = "#B9453F"
COL_BASIN = "#7B6F9B"
COL_TOP_POINT = "#C65A2E"
COL_QSTATE = "#6F8792"

CMAP_NPI = LinearSegmentedColormap.from_list("npi_mauve", ["#F3F0F4", "#B9AEC2", "#75677F"])
CMAP_IPI = LinearSegmentedColormap.from_list("ipi_teal", ["#EEF5F4", "#A4CCCA", "#4E8C90"])


REGION_NAME_REPLACE = {
    "S.Asia": "South Asia",
    "W.C.Asia": "W. Central Asia",
    "Western-Africa": "Western Africa",
    "W.Southern-Africa": "W. Southern Africa",
    "C.North-America": "C. North America",
    "N.Eastern-Africa": "N. Eastern Africa",
    "E.Asia": "East Asia",
    "E.Southern-Africa": "E. Southern Africa",
    "N.Central-America": "N. Central America",
    "Arabian-Peninsula": "Arabian Peninsula",
}


def pretty_region_name(name: str) -> str:
    name = str(name)
    return REGION_NAME_REPLACE.get(name, name.replace("-", " "))


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        -0.025,
        1.055,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        0.055,
        1.055,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#111111",
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=6.1, width=0.45, length=2.4, color="#555555")
    ax.grid(color=COL_GRID, linewidth=0.38, zorder=0)
    ax.set_axisbelow(True)


def set_left_aligned_ylabels(ax: plt.Axes, y: np.ndarray, labels: list[str], x_axes: float = -0.095, fontsize: float = 6.3) -> None:
    """Draw y labels manually so all region codes share a fixed left edge."""
    ax.set_yticks(y)
    ax.set_yticklabels([])
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for yi, label in zip(y, labels):
        ax.text(x_axes, yi, label, transform=transform, ha="left", va="center", fontsize=fontsize, color=COL_TEXT, clip_on=False)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Return a cropland-area-weighted quantile for finite basin values."""
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    x = np.asarray(values[valid], dtype=np.float64)
    w = np.asarray(weights[valid], dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cumulative = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return float(np.interp(quantile, cumulative, x))


def load_meta(basin_ids: np.ndarray) -> pd.DataFrame:
    cols = ["HYBAS_ID", "dominant_ipcc_region_code", "dominant_ipcc_region_name"]
    meta = pd.read_csv(META_PATH, usecols=cols).drop_duplicates("HYBAS_ID").set_index("HYBAS_ID")
    rows = []
    for basin_id in basin_ids.astype(np.int64):
        if int(basin_id) in meta.index:
            row = meta.loc[int(basin_id)].to_dict()
        else:
            row = {"dominant_ipcc_region_code": "NA", "dominant_ipcc_region_name": "not assigned"}
        row["HYBAS_ID"] = int(basin_id)
        rows.append(row)
    return pd.DataFrame(rows)


def build_region_priority(region_table: pd.DataFrame) -> pd.DataFrame:
    irr = region_table[
        region_table["period_key"].isin(["ssp126", "ssp370"])
    ].copy()
    wide = irr.pivot_table(
        index=["region_code", "region_name"],
        columns="period_key",
        values="compound_high_pressure_area_million_km2",
        aggfunc="first",
    ).reset_index()
    wide = wide.rename(columns={"ssp126": "compound_ssp126_million_km2", "ssp370": "compound_ssp370_million_km2"})
    wide["region_name_pretty"] = wide["region_name"].map(pretty_region_name)
    wide["ssp3_minus_ssp1_million_km2"] = wide["compound_ssp370_million_km2"] - wide["compound_ssp126_million_km2"]
    return wide.sort_values("compound_ssp370_million_km2", ascending=False)


def build_region_mechanism(z: np.lib.npyio.NpzFile, top_codes: list[str]) -> pd.DataFrame:
    basin_ids = np.asarray(z["basin_ids"], dtype=np.int64)
    meta = load_meta(basin_ids)
    code = (
        meta["dominant_ipcc_region_code"]
        .fillna("NA")
        .astype(str)
        .to_numpy()
    )
    name = (
        meta["dominant_ipcc_region_name"]
        .fillna("not assigned")
        .astype(str)
        .to_numpy()
    )
    npi = np.asarray(z["NPI_IRR_ssp370_basin"], dtype=np.float64)
    ipi = np.asarray(z["IPI_IRR_ssp370_basin"], dtype=np.float64)
    cls = np.asarray(z["class_IRR_ssp370_basin"], dtype=np.int16)
    agreement = np.asarray(z["compound_member_agreement_IRR_ssp370_basin"], dtype=np.float64)
    # Product-level flag: all three independently processed members classify
    # the basin as Compound. This is strict 3/3 support, not mean agreement.
    unanimous_compound = np.asarray(
        z["compound_member_unanimous_IRR_ssp370_basin"], dtype=bool
    )
    area = np.asarray(z["basin_crop_area_ssp370_m2"], dtype=np.float64) / 1e12
    rows = []
    for rcode in sorted(set(code)):
        if rcode in {"", "NA"}:
            continue
        region_mask = code == rcode
        compound_mask = region_mask & (cls == 3) & np.isfinite(area) & (area > 0)
        compound_area = float(np.sum(area[compound_mask]))
        if compound_area <= 0:
            continue
        unanimous_area = float(np.sum(area[compound_mask & unanimous_compound]))
        sample_idx = np.where(region_mask)[0][0]
        finite_npi = compound_mask & np.isfinite(npi)
        rows.append(
            {
                "region_code": rcode,
                "region_name": pretty_region_name(name[sample_idx]),
                "compound_area_million_km2": compound_area,
                # Area-weighted medians avoid undue leverage from small positive
                # denominators while retaining threshold-relative interpretation.
                "NPI_threshold_multiple": weighted_quantile(
                    npi[finite_npi], area[finite_npi], 0.5
                ) / NPI_THRESHOLD,
                "NPI_threshold_multiple_q25": weighted_quantile(
                    npi[finite_npi], area[finite_npi], 0.25
                ) / NPI_THRESHOLD,
                "NPI_threshold_multiple_q75": weighted_quantile(
                    npi[finite_npi], area[finite_npi], 0.75
                ) / NPI_THRESHOLD,
                "IPI_threshold_multiple": weighted_quantile(
                    ipi[compound_mask], area[compound_mask], 0.5
                ) / IPI_THRESHOLD,
                "IPI_threshold_multiple_q25": weighted_quantile(
                    ipi[compound_mask], area[compound_mask], 0.25
                ) / IPI_THRESHOLD,
                "IPI_threshold_multiple_q75": weighted_quantile(
                    ipi[compound_mask], area[compound_mask], 0.75
                ) / IPI_THRESHOLD,
                "compound_weighted_member_agreement": weighted_mean(agreement[compound_mask], area[compound_mask]),
                "unanimous_compound_area_million_km2": unanimous_area,
                "unanimous_compound_share_percent": unanimous_area / compound_area * 100.0,
                "is_priority_region": rcode in top_codes,
            }
        )
    return pd.DataFrame(rows).sort_values("compound_area_million_km2", ascending=False)


def build_nonpositive_q_exposure(
    z: np.lib.npyio.NpzFile,
    priority: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize period-specific cropland exposure in the non-positive-Q state."""
    basin_ids = np.asarray(z["basin_ids"], dtype=np.int64)
    meta = load_meta(basin_ids)
    region_code = meta["dominant_ipcc_region_code"].fillna("NA").astype(str).to_numpy()

    out = priority.copy()
    for period in ("ssp126", "ssp370"):
        state = np.asarray(z[f"nonpositive_Q_limiting_state_{period}_basin"], dtype=bool)
        area = np.asarray(z[f"basin_crop_area_{period}_m2"], dtype=np.float64) / 1e12
        valid = state & np.isfinite(area) & (area > 0)
        area_by_code = {
            code: float(np.sum(area[valid & (region_code == code)]))
            for code in np.unique(region_code)
        }
        count_by_code = {
            code: int(np.sum(valid & (region_code == code)))
            for code in np.unique(region_code)
        }
        out[f"nonpositive_q_{period}_million_km2"] = out["region_code"].map(area_by_code).fillna(0.0)
        out[f"nonpositive_q_{period}_basin_count"] = out["region_code"].map(count_by_code).fillna(0).astype(int)

    out["nonpositive_q_ssp3_minus_ssp1_million_km2"] = (
        out["nonpositive_q_ssp370_million_km2"]
        - out["nonpositive_q_ssp126_million_km2"]
    )
    return out


def _plot_regional_dumbbells(
    ax: plt.Axes,
    top: pd.DataFrame,
    x1_col: str,
    x3_col: str,
    delta_col: str,
    show_ylabels: bool,
    subtitle: str,
    shared_xlim: tuple[float, float],
    show_legend: bool = False,
) -> None:
    y = np.arange(len(top))
    x1 = top[x1_col].to_numpy(float)
    x3 = top[x3_col].to_numpy(float)
    ax.hlines(y, x1, x3, color="#D7D0CC", linewidth=2.8, zorder=1)
    ax.scatter(x1, y, s=20, color=COL_SSP1, edgecolor="white", linewidth=0.45, zorder=3, label="SSP1-2.6")
    ax.scatter(x3, y, s=25, color=COL_SSP3, edgecolor="white", linewidth=0.45, zorder=4, label="SSP3-7.0")
    for yi, val1, val3, delta in zip(
        y,
        x1,
        x3,
        top[delta_col].to_numpy(float),
    ):
        if abs(delta) < 0.005:
            delta = 0.0
        label_x = min(max(val1, val3) + 0.025, shared_xlim[1] - 0.28)
        ax.text(label_x, yi, f"{val3:.2f} ({delta:+.2f})", va="center", ha="left", fontsize=5.3, color=COL_TEXT)
    if show_ylabels:
        set_left_aligned_ylabels(ax, y, top["region_code"].tolist(), x_axes=-0.205, fontsize=6.3)
    else:
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_xlim(*shared_xlim)
    ax.set_xticks([0.0, 0.4, 0.8])
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.9, color=COL_QSTATE if "non-positive" in subtitle else COL_TEXT)
    if show_legend:
        ax.legend(frameon=False, fontsize=5.7, loc="lower right", handlelength=0.9, labelspacing=0.28, borderaxespad=0.15)
    style_axis(ax)
    ax.grid(axis="y", visible=False)


def plot_panel_a(ax: plt.Axes, priority_with_q: pd.DataFrame) -> list[str]:
    """Plot two exposure states on one axis using colour, shape and fill."""
    top_codes = priority_with_q.head(TOP_N_REGION)["region_code"].tolist()
    top = priority_with_q.head(TOP_N_REGION).iloc[::-1].copy()
    y = np.arange(len(top), dtype=float)
    y_compound = y
    y_qstate = y

    c1 = top["compound_ssp126_million_km2"].to_numpy(float)
    c3 = top["compound_ssp370_million_km2"].to_numpy(float)
    q1 = top["nonpositive_q_ssp126_million_km2"].to_numpy(float)
    q3 = top["nonpositive_q_ssp370_million_km2"].to_numpy(float)

    ax.hlines(y_qstate, q1, q3, color=COL_QSTATE, linewidth=1.35, linestyles=(0, (3, 2)), alpha=0.90, zorder=1)
    ax.hlines(y_compound, c1, c3, color="#D7B9B6", linewidth=2.0, linestyles="-", alpha=0.85, zorder=3)

    # Colour and marker encode state; marker fill encodes scenario.
    ax.scatter(q1, y_qstate, s=19, marker="s", facecolor="white", edgecolor=COL_QSTATE, linewidth=0.85, zorder=2)
    ax.scatter(q3, y_qstate, s=23, marker="s", facecolor=COL_QSTATE, edgecolor="white", linewidth=0.45, zorder=3)
    ax.scatter(c1, y_compound, s=20, marker="o", facecolor="white", edgecolor=COL_SSP3, linewidth=0.85, zorder=4)
    ax.scatter(c3, y_compound, s=24, marker="o", facecolor=COL_SSP3, edgecolor="white", linewidth=0.45, zorder=5)

    set_left_aligned_ylabels(ax, y, top["region_code"].tolist(), x_axes=-0.105, fontsize=6.3)
    ax.set_xlim(0.0, 1.10)
    ax.set_ylim(-0.55, len(top) - 0.45)
    ax.set_xlabel("Cropland area (million km\u00b2)", fontsize=6.2)
    style_axis(ax)
    ax.grid(axis="y", visible=False)

    state_handles = [
        plt.Line2D([], [], marker="o", linestyle="-", linewidth=1.5, color=COL_SSP3, markersize=4.6, markerfacecolor=COL_SSP3, markeredgecolor="white", markeredgewidth=0.45, label="Compound"),
        plt.Line2D([], [], marker="s", linestyle=(0, (3, 2)), linewidth=1.35, color=COL_QSTATE, markersize=4.4, markerfacecolor=COL_QSTATE, markeredgecolor="white", markeredgewidth=0.45, label=r"$Q \leq 0$"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=4.4, markerfacecolor=COL_MUTED, markeredgecolor="white", markeredgewidth=0.4, label="SSP3-7.0"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=4.4, markerfacecolor="white", markeredgecolor=COL_MUTED, markeredgewidth=0.8, label="SSP1-2.6"),
    ]
    ax.legend(
        handles=state_handles,
        ncol=2,
        frameon=False,
        fontsize=5.55,
        loc="lower right",
        handletextpad=0.35,
        columnspacing=0.85,
        labelspacing=0.32,
        borderaxespad=0.18,
    )
    panel_label(ax, "a", "Regional exposure")
    return top_codes


def plot_panel_b(ax: plt.Axes, mechanism: pd.DataFrame, top_codes: list[str]) -> pd.DataFrame:
    top = mechanism.set_index("region_code").loc[top_codes].reset_index()
    top = top.sort_values("compound_area_million_km2", ascending=True)
    rows = top["region_code"].tolist()
    values = top[["NPI_threshold_multiple", "IPI_threshold_multiple"]].to_numpy(float)
    npi_log_values = np.log10(np.clip(values[:, 0], 1.0, 30.0))
    npi_norm = Normalize(vmin=0.0, vmax=np.log10(30.0))
    ipi_min = float(np.nanmin(values[:, 1]))
    ipi_max = float(np.nanmax(values[:, 1]))
    ipi_norm = Normalize(vmin=ipi_min, vmax=ipi_max if ipi_max > ipi_min else ipi_min + 1.0)

    block_h = 1.20
    row_gap = 0.18
    row_step = block_h + row_gap
    row_y = np.arange(len(rows)) * row_step

    ax.set_xlim(0, 2)
    ax.set_ylim(0, row_y[-1] + block_h + 0.25)
    for i, row in enumerate(rows):
        npi_strength = float(npi_norm(npi_log_values[i]))
        ipi_strength = float(ipi_norm(values[i, 1]))
        npi_face = CMAP_NPI(npi_strength)
        wsi_face = CMAP_IPI(ipi_strength)
        y0 = row_y[i]
        yc = y0 + block_h / 2.0
        ax.add_patch(plt.Rectangle((0.02, y0), 0.86, block_h, facecolor=npi_face, edgecolor="white", linewidth=0.75))
        ax.add_patch(plt.Rectangle((1.02, y0), 0.86, block_h, facecolor=wsi_face, edgecolor="white", linewidth=0.75))
        ax.text(-0.22, yc, row, va="center", ha="left", fontsize=6.3, color=COL_TEXT, clip_on=False)
        npi_text = "white" if npi_strength > 0.76 else "#2B241D"
        wsi_text = "white" if ipi_strength > 0.76 else "#142829"
        ax.text(0.45, yc, f"{values[i,0]:.1f}x", va="center", ha="center", fontsize=5.7, color=npi_text)
        ax.text(1.45, yc, f"{values[i,1]:.1f}x", va="center", ha="center", fontsize=5.7, color=wsi_text)
    ax.text(0.230, -0.045, "NPI multiple", transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color="#65586E", clip_on=False)
    ax.text(0.770, -0.045, "IPI multiple", transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color="#326F73", clip_on=False)
    ax.axis("off")
    panel_label(ax, "b", "Pressure fingerprints")
    return top


def plot_panel_c(ax: plt.Axes, z: np.lib.npyio.NpzFile, global_compound_area: float, top_n: int = 30) -> pd.DataFrame:
    cls = np.asarray(z["class_IRR_ssp370_basin"], dtype=np.int16)
    basin_ids = np.asarray(z["basin_ids"], dtype=np.int64)
    area = np.asarray(z["basin_crop_area_ssp370_m2"], dtype=np.float64) / 1e12
    mask = (cls == 3) & np.isfinite(area) & (area > 0)
    compound_basin_count = int(np.sum(mask))
    order = np.argsort(area[mask])[::-1]
    compound_area = area[mask][order][:top_n]
    compound_basin_ids = basin_ids[mask][order][:top_n]
    df = pd.DataFrame(
        {
            "rank": np.arange(1, len(compound_area) + 1, dtype=int),
            "HYBAS_ID": compound_basin_ids.astype(np.int64),
            "compound_area_million_km2": compound_area,
        }
    )
    df["cumulative_area_million_km2"] = df["compound_area_million_km2"].cumsum()
    df["cumulative_share_percent"] = df["cumulative_area_million_km2"] / global_compound_area * 100.0

    rank = df["rank"].to_numpy(int)
    share = df["cumulative_share_percent"].to_numpy(float)
    ax.fill_between(rank, share, color="#EAE6F4", alpha=0.72, zorder=1)
    ax.plot(rank, share, color=COL_BASIN, linewidth=1.8, zorder=3)
    ax.scatter(rank, share, s=14, color=COL_BASIN, edgecolor="white", linewidth=0.35, zorder=4)
    highlight_ranks = [5, 10, 20, 30]
    highlight = df[df["rank"].isin(highlight_ranks)].copy()
    ax.scatter(
        highlight["rank"].to_numpy(int),
        highlight["cumulative_share_percent"].to_numpy(float),
        s=32,
        color=COL_TOP_POINT,
        edgecolor="white",
        linewidth=0.55,
        zorder=5,
    )
    for k in highlight_ranks:
        val = float(df.loc[df["rank"] == k, "cumulative_share_percent"].iloc[0])
        ax.axvline(k, color="#D6D2E4", linewidth=0.6, zorder=0)
        xoff = {5: -0.35, 10: 0.15, 20: -0.15, 30: -0.45}.get(k, 0.0)
        yoff = {5: 5.2, 10: 5.8, 20: 3.2, 30: 3.2}.get(k, 3.0)
        ax.text(
            k + xoff,
            val + yoff,
            f"Top {k}: {val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=5.8,
            color=COL_TEXT,
        )
    ax.set_xlim(1, 31.5)
    y_max = max(50.0, np.ceil((np.nanmax(share) + 5.0) / 10.0) * 10.0)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0.0, y_max + 0.1, 10.0))
    ax.set_xticks([1, 5, 10, 20, 30])
    ax.set_xlabel("HydroBASINS level-5 rank", fontsize=6.2)
    ax.set_ylabel("Cumulative share of SSP3-7.0 compound cropland (%)", fontsize=6.2)
    style_axis(ax)
    panel_label(ax, "c", "Basin-scale accumulation")
    ax.text(
        0.985,
        1.055,
        f"{compound_basin_count} Compound basin units",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=COL_MUTED,
    )
    return df


def plot_panel_d(ax: plt.Axes, mechanism: pd.DataFrame, top_codes: list[str], z: np.lib.npyio.NpzFile, show_label: bool = True) -> pd.DataFrame:
    top = mechanism.set_index("region_code").loc[top_codes].reset_index()
    top = top.sort_values("compound_area_million_km2", ascending=True)
    y = np.arange(len(top))
    share = top["unanimous_compound_share_percent"].to_numpy(float)

    cls = np.asarray(z["class_IRR_ssp370_basin"], dtype=np.int16)
    # Use the verified strict 3/3 flag for the global reference as well as for
    # the regional bars, keeping the numerator definition identical.
    unanimous_compound = np.asarray(
        z["compound_member_unanimous_IRR_ssp370_basin"], dtype=bool
    )
    area = np.asarray(z["basin_crop_area_ssp370_m2"], dtype=np.float64) / 1e12
    compound = (cls == 3) & np.isfinite(area) & (area > 0)
    global_unanimous_share = float(
        np.sum(area[compound & unanimous_compound]) / np.sum(area[compound]) * 100.0
    )

    ax.barh(y, share, color="#D0D6DC", edgecolor="white", linewidth=0.45, height=0.58, zorder=2)
    ax.axvline(global_unanimous_share, color=COL_SSP3, linestyle="-", linewidth=0.80, zorder=3)
    ax.text(
        global_unanimous_share - 1.2,
        len(top) - 0.18,
        f"Global: {global_unanimous_share:.0f}%",
        fontsize=5.5,
        color=COL_SSP3,
        ha="right",
        va="center",
    )
    set_left_aligned_ylabels(ax, y, top["region_code"].tolist(), x_axes=-0.130, fontsize=6.3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Compound cropland supported by all three members (%)", fontsize=6.2)
    style_axis(ax)
    ax.grid(axis="y", visible=False)
    if show_label:
        panel_label(ax, "d", "Member agreement")

    out = top[
        [
            "region_code",
            "region_name",
            "compound_area_million_km2",
            "unanimous_compound_area_million_km2",
            "unanimous_compound_share_percent",
        ]
    ].copy()
    out.loc[len(out)] = [
        "GLOBAL",
        "Global compound cropland",
        float(np.sum(area[compound])),
        float(np.sum(area[compound & unanimous_compound])),
        global_unanimous_share,
    ]
    return out


def main() -> None:
    region = pd.read_csv(REGION_TABLE)
    global_summary = pd.read_csv(GLOBAL_TABLE)
    z = np.load(PRODUCT_PATH)

    priority = build_region_priority(region)
    priority_with_q = build_nonpositive_q_exposure(z, priority)
    priority.to_csv(
        TABLE_DIR / "Fig05_region_priority_summary_strict_positiveQ_01pct.csv",
        index=False,
        encoding="utf-8-sig",
    )
    top_codes = priority.head(TOP_N_REGION)["region_code"].tolist()
    mechanism = build_region_mechanism(z, top_codes)
    global_compound_area = float(
        global_summary[
            (global_summary["experiment"] == "IRR") & (global_summary["period_key"] == "ssp370")
        ]["compound_high_pressure_area_million_km2"].iloc[0]
    )

    fig = plt.figure(figsize=(7.50, 5.75), dpi=600, facecolor="white")
    ax_a = fig.add_axes([0.102, 0.560, 0.420, 0.365])
    ax_b = fig.add_axes([0.600, 0.560, 0.325, 0.365])
    ax_c = fig.add_axes([0.102, 0.115, 0.420, 0.310])
    ax_d = fig.add_axes([0.600, 0.115, 0.325, 0.310])

    top_codes = plot_panel_a(ax_a, priority_with_q)
    pressure_fingerprints = plot_panel_b(ax_b, mechanism, top_codes)
    basin_accumulation = plot_panel_c(ax_c, z, global_compound_area, top_n=30)
    member_agreement_summary = plot_panel_d(ax_d, mechanism, top_codes, z, show_label=True)

    basin_accumulation.to_csv(
        TABLE_DIR / "Fig05c_top30_basin_accumulation_strict_positiveQ_01pct.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pressure_fingerprints.to_csv(
        TABLE_DIR / "Fig05b_regional_pressure_fingerprints_weighted_median_strict_positiveQ_01pct.csv",
        index=False,
        encoding="utf-8-sig",
    )

    priority_with_q.to_csv(
        TABLE_DIR / "Fig05a_regional_compound_and_nonpositiveQ_exposure.csv",
        index=False,
        encoding="utf-8-sig",
    )

    base = "Fig05_regional_priorities_and_robustness"
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"{base}.{ext}", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)

    member_agreement_summary.to_csv(
        TABLE_DIR / "Fig05d_unanimous_3of3_member_agreement_strict_positiveQ.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("[Saved revised Fig. 5 with regional non-positive-Q exposure]")
    print(FIG_DIR / f"{base}.png")
    print(TABLE_DIR / "Fig05d_unanimous_3of3_member_agreement_strict_positiveQ.csv")


if __name__ == "__main__":
    main()



