# -*- coding: utf-8 -*-
"""Regional trajectories of established and emerging compound-pressure hotspots.

Strict-positive-Q version:
- annual NPI-IPI classes are recomputed from member-resolved N, signed
  irrigation-conditioned Q and irrigation withdrawal;
- only basins with Q > 0 are assigned to the four pressure classes;
- emerging compound cropland is referenced to the member-specific historical
  strict-positive-Q period classification;
- the six regions capture established hotspots (SAS, WCA and MED), emerging
  African frontiers (WAF and ESAF), and a temperate North American
  agricultural system (CNA);
- this version keeps the v26 layout unchanged but lightens the SSP3-7.0
  analysis-period shading and grid lines to avoid a grey cast when rendered
  in Word/PDF viewers;
- panel letters are retained but redundant panel titles are removed because
  the y-axis labels already define each quantity;
- panels a-c keep formal y-axis labels;
- panel a only labels the historical and SSP3-7.0 segments;
- tighter four-panel layout with shorter, taller plotting windows;
- panel letters are placed next to each panel's y axis;
- panel d legend is placed near the 24% x-axis position.
- tighter panel spacing and reduced outer margins.
- panel c uses the focused 2015-2074 x-axis because emerging compound
  pressure is defined relative to the historical baseline.
- panel d plot geometry is aligned with panels b and c.
- axes are redrawn after grid lines so horizontal and vertical axes have
  consistent visual weight.
- panel d region labels are moved closer to the y-axis and the style key
  uses transparent neutral swatches.
- time-series end labels are direct labels without slanted leader lines;
- panel d uses tightly paired regional bars with a colour style key.
- regional colours are softened to better match the low-saturation
  Extended Data figure set.
- outer margins are reduced and direct line labels are placed close to
  curve endpoints rather than being forced onto an evenly spaced stack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageChops


# =============================================================================
# Paths
# =============================================================================

SOURCE_ROOT = Path(__file__).resolve().parents[3]
ROOT = SOURCE_ROOT
DATA = ROOT / "data"
TABLE_DIR = ROOT / "tables" / "extended_data"
FIG_DIR = ROOT / "figures" / "extended_data"
CODE_DIR = Path(__file__).resolve().parent

DYNAMIC_ANNUAL = DATA / "annual_inputs" / "nitrate_loss_dynamic_cropland_annual_members_01pct.npz"

HYDRO_ANNUAL = DATA / "annual_inputs" / "hydrology_irrigation_annual_members_01pct.npz"

PERIOD_PRODUCTS = (
    DATA / "NPI_IPI_QIRR_strict_positiveQ_period_products_01pct.npz"
)

META = ROOT / "data" / "basin_metadata" / "HydroBASINS_level5_metadata_and_IPCC_assignment_landfrac_corrected.csv"

OUT_TABLE = (
    TABLE_DIR
    / "ExtendedData_Fig10_regional_hotspot_timeseries_representative6_CNA_strict_positiveQ_01pct.csv"
)

OUT_SUMMARY = (
    TABLE_DIR
    / "ExtendedData_Fig10_regional_hotspot_summary_representative6_CNA_strict_positiveQ_01pct.csv"
)

OUT = (
    FIG_DIR
    / "ExtendedData_Fig10_regional_hotspot_trajectories"
)


# =============================================================================
# Region setup
# =============================================================================

REGIONS = ["SAS", "WCA", "MED", "WAF", "ESAF", "CNA"]
LEGEND_ORDER = ["SAS", "WCA", "MED", "WAF", "CNA", "ESAF"]

# Curves are drawn in this order. Later curves appear above earlier curves.
DRAW_ORDER = ["MED", "WCA", "CNA", "ESAF", "WAF", "SAS"]

REGION_FULL = {
    "SAS": "South Asia",
    "MED": "Mediterranean",
    "WCA": "W. Central Asia",
    "WAF": "Western Africa",
    "ESAF": "E. Southern Africa",
    "CNA": "Central North America",
}

# Softened regional palette. The hues remain distinct, but saturation is
# reduced relative to the high-contrast plotting palette.
COL = {
    "SAS": "#C9652A",
    "MED": "#C87FA3",
    "WCA": "#2E7FB2",
    "WAF": "#209979",
    "ESAF": "#8A78B8",
    "CNA": "#D79A22",
    "axis": "#333333",
    "grid": "#EEF1F3",
    "text": "#172026",
    "muted": "#606B75",
    "shade": "#F9FAFB",
}

NO3_THRESHOLD = 10.0
IPI_THRESHOLD = 0.4
MIN_Q_L = 1.0

TRAJ_LINE_WIDTH = 4


# =============================================================================
# Fonts and drawing helpers
# =============================================================================

def font(size: int, bold: bool = False):
    candidates = [
        r"C:/Windows/Fonts/arialbd.ttf" if bold else r"C:/Windows/Fonts/arial.ttf",
        r"C:/Windows/Fonts/calibrib.ttf" if bold else r"C:/Windows/Fonts/calibri.ttf",
    ]

    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()


F_PANEL = font(34, True)
F_TITLE = font(34, True)
F_AXIS = font(25)
F_TICK = font(21)
F_SMALL = font(20)
F_SMALL_BOLD = font(21, True)
F_LABEL = font(19, True)
F_TINY = font(18)


def rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def lighten(hex_color: str, amount: float = 0.62):
    r, g, b = rgb(hex_color)
    return tuple(int(c + (255 - c) * amount) for c in (r, g, b))


def text(draw, xy, s, fnt, fill=None, anchor=None, **kwargs):
    draw.text(xy, s, font=fnt, fill=fill or COL["text"], anchor=anchor, **kwargs)


def draw_rotated_text(img, xy, s, fnt, fill=None, angle=90):
    """Draw rotated text centered at xy."""
    fill = fill or COL["text"]

    if hasattr(fnt, "getbbox"):
        bbox = fnt.getbbox(s)
    else:
        w0, h0 = fnt.getsize(s)
        bbox = (0, 0, w0, h0)
    w = bbox[2] - bbox[0] + 12
    h = bbox[3] - bbox[1] + 12

    layer = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    d = ImageDraw.Draw(layer)
    d.text((6 - bbox[0], 6 - bbox[1]), s, font=fnt, fill=fill)

    rotated = layer.rotate(angle, expand=True)
    x, y = xy

    img.paste(
        rotated,
        (int(x - rotated.size[0] / 2), int(y - rotated.size[1] / 2)),
        rotated,
    )


def rolling(y, window=5):
    return (
        pd.Series(np.asarray(y, dtype=float))
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


# =============================================================================
# Data processing
# =============================================================================

def load_region_codes(basin_ids):
    meta = (
        pd.read_csv(META, usecols=["HYBAS_ID", "dominant_ipcc_region_code"])
        .drop_duplicates("HYBAS_ID")
    )

    meta = meta.set_index("HYBAS_ID")

    return np.asarray([
        str(meta.loc[int(b), "dominant_ipcc_region_code"])
        if int(b) in meta.index
        else "NA"
        for b in basin_ids.astype(int)
    ])


def compound_bool(nload_gn, q_l, iww_l):
    q = np.where(np.isfinite(q_l) & (q_l > MIN_Q_L), q_l, np.nan)
    npi = (nload_gn.astype(float) / q.astype(float)) * 1000.0
    ipi = iww_l.astype(float) / q.astype(float)

    return (
        np.isfinite(npi)
        & np.isfinite(ipi)
        & (npi >= NO3_THRESHOLD)
        & (ipi >= IPI_THRESHOLD)
    )


def build_timeseries():
    """Recompute annual strict-positive-Q regional trajectories.

    N, signed irrigation-conditioned Q and I are kept member resolved. Annual
    basin classes are assigned only where Q > 1 L, matching the numerical
    tolerance used by the period product. Emerging compound cropland is
    evaluated against each member's strict historical period class.
    """
    dynamic = np.load(DYNAMIC_ANNUAL, allow_pickle=False)
    hydro = np.load(HYDRO_ANNUAL, allow_pickle=False)
    period = np.load(PERIOD_PRODUCTS, allow_pickle=False)

    basin_ids = np.asarray(dynamic["basin_ids"], dtype=np.int64)
    if not np.array_equal(basin_ids, np.asarray(hydro["basin_ids"], dtype=np.int64)):
        raise ValueError("Dynamic-cropland and hydrological basin orders differ.")
    if not np.array_equal(basin_ids, np.asarray(period["basin_ids"], dtype=np.int64)):
        raise ValueError("Annual and strict period-product basin orders differ.")

    region_codes = load_region_codes(basin_ids)
    region_masks = {code: region_codes == code for code in REGIONS}
    hist_member_class = np.asarray(
        period["class_member_IRR_historical_basin"], dtype=np.int8
    )
    if hist_member_class.shape != (3, basin_ids.size):
        raise ValueError(
            f"Unexpected historical member-class shape: {hist_member_class.shape}"
        )

    blocks = [
        (
            "historical",
            np.asarray(dynamic["hist_years"], dtype=int),
            np.asarray(dynamic["nload_irr_hist_gN"], dtype=np.float64),
            np.asarray(hydro["qavail_irr_hist_L"], dtype=np.float64),
            np.asarray(hydro["iww_hist_L"], dtype=np.float64),
            np.asarray(dynamic["basin_crop_area_hist_m2"], dtype=np.float64),
        ),
        (
            "ssp126",
            np.asarray(dynamic["future_years"], dtype=int),
            np.asarray(dynamic["nload_irr_future_gN"], dtype=np.float64)[0],
            np.asarray(hydro["qavail_irr_future_L"], dtype=np.float64)[0],
            np.asarray(hydro["iww_future_L"], dtype=np.float64)[0],
            np.asarray(dynamic["basin_crop_area_future_m2"], dtype=np.float64)[0],
        ),
        (
            "ssp370",
            np.asarray(dynamic["future_years"], dtype=int),
            np.asarray(dynamic["nload_irr_future_gN"], dtype=np.float64)[1],
            np.asarray(hydro["qavail_irr_future_L"], dtype=np.float64)[1],
            np.asarray(hydro["iww_future_L"], dtype=np.float64)[1],
            np.asarray(dynamic["basin_crop_area_future_m2"], dtype=np.float64)[1],
        ),
    ]

    rows = []
    for scenario, years, nload, qirr, iww, crop_area in blocks:
        expected = (3, years.size, basin_ids.size)
        if nload.shape != expected or qirr.shape != expected or iww.shape != expected:
            raise ValueError(
                f"{scenario} N/Q/I shape mismatch: "
                f"{nload.shape}, {qirr.shape}, {iww.shape}; expected {expected}"
            )
        if crop_area.shape != (years.size, basin_ids.size):
            raise ValueError(
                f"{scenario} cropland-area shape mismatch: {crop_area.shape}"
            )

        for member in range(3):
            hist_noncompound = hist_member_class[member] != 3
            for t, year in enumerate(years):
                compound = compound_bool(
                    nload[member, t], qirr[member, t], iww[member, t]
                )
                positive_q = np.isfinite(qirr[member, t]) & (
                    qirr[member, t] > MIN_Q_L
                )
                area = crop_area[t]
                crop = np.isfinite(area) & (area > 0)
                compound &= crop
                emerging = (
                    compound & hist_noncompound
                    if scenario != "historical"
                    else np.zeros_like(compound)
                )

                global_compound = float(np.sum(area[compound]) / 1.0e12)
                global_emerging = (
                    float(np.sum(area[emerging]) / 1.0e12)
                    if scenario != "historical"
                    else np.nan
                )
                global_nonpositive_q = float(
                    np.sum(area[crop & ~positive_q]) / 1.0e12
                )

                for code, region_mask in region_masks.items():
                    regional_compound = compound & region_mask
                    regional_emerging = emerging & region_mask
                    comp_area = float(np.sum(area[regional_compound]) / 1.0e12)
                    emerging_area = (
                        float(np.sum(area[regional_emerging]) / 1.0e12)
                        if scenario != "historical"
                        else np.nan
                    )
                    rows.append(
                        {
                            "region_code": code,
                            "region_name": REGION_FULL[code],
                            "scenario": scenario,
                            "member": member + 1,
                            "year": int(year),
                            "compound_area_million_km2": comp_area,
                            "global_compound_area_million_km2": global_compound,
                            "compound_share_percent": (
                                comp_area / global_compound * 100.0
                                if global_compound > 0
                                else np.nan
                            ),
                            "emerging_area_million_km2": emerging_area,
                            "global_emerging_area_million_km2": global_emerging,
                            "emerging_share_percent": (
                                emerging_area / global_emerging * 100.0
                                if scenario != "historical"
                                and global_emerging > 0
                                else np.nan
                            ),
                            "global_nonpositive_Q_area_million_km2": (
                                global_nonpositive_q
                            ),
                        }
                    )

    member_df = pd.DataFrame(rows)
    ens = member_df.groupby(
        ["region_code", "region_name", "scenario", "year"], as_index=False
    ).agg(
        compound_area_million_km2=("compound_area_million_km2", "mean"),
        global_compound_area_million_km2=(
            "global_compound_area_million_km2",
            "mean",
        ),
        compound_share_percent=("compound_share_percent", "mean"),
        emerging_area_million_km2=("emerging_area_million_km2", "mean"),
        global_emerging_area_million_km2=(
            "global_emerging_area_million_km2",
            "mean",
        ),
        emerging_share_percent=("emerging_share_percent", "mean"),
        global_nonpositive_Q_area_million_km2=(
            "global_nonpositive_Q_area_million_km2",
            "mean",
        ),
    )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ens.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

    future = ens[
        ens["scenario"].isin(["ssp126", "ssp370"])
        & ens["year"].between(2045, 2074)
    ]
    summary = (
        future.groupby(["scenario", "region_code", "region_name"], as_index=False)
        .agg(
            compound_area_million_km2=("compound_area_million_km2", "mean"),
            compound_share_percent=("compound_share_percent", "mean"),
            emerging_area_million_km2=("emerging_area_million_km2", "mean"),
            emerging_share_percent=("emerging_share_percent", "mean"),
        )
    )
    summary["period"] = "2045-2074"
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    if ens.duplicated(["scenario", "year", "region_code"]).any():
        raise RuntimeError("Duplicate ensemble trajectory rows detected.")
    if set(ens["region_code"]) != set(REGIONS):
        raise RuntimeError("One or more selected regions are absent.")
    if not np.isfinite(
        ens.loc[ens.scenario == "ssp370", "compound_area_million_km2"]
    ).all():
        raise RuntimeError("Non-finite SSP3-7.0 compound trajectory values.")

    return ens, summary


# =============================================================================
# Geometry helpers
# =============================================================================

def panel_header(draw, box, letter):
    """Draw panel letter close to the upper-left of its plotting area."""
    x0, y0, _, _ = box
    text(
        draw,
        (x0 + 108, y0 + 6),
        letter,
        F_PANEL,
        fill="#000000",
        anchor="lt",
    )


def draw_region_legend(draw, box):
    """Shared legend for panels a-c."""
    x0, y0, _, _ = box
    y = y0 + 30
    plot_left = box[0] + 108
    plot_right = box[2] - 68
    legend_width = 520
    x = (plot_left + plot_right) / 2 - legend_width / 2

    for code in LEGEND_ORDER:
        color = rgb(COL[code])
        draw.line([(x, y), (x + 28, y)], fill=color, width=TRAJ_LINE_WIDTH)
        text(draw, (x + 36, y), code, F_TINY, fill=COL["text"], anchor="lm")
        x += 88


def map_xy(box, xlim, ylim, x, y):
    x0, y0, x1, y1 = box

    xx = x0 + (np.asarray(x) - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
    yy = y1 - (np.asarray(y) - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)

    return xx, yy


def draw_axes(
    img,
    draw,
    plot,
    xlim,
    ylim,
    yticks,
    ylabel,
    xlabel="Year",
    xticks=(1990, 2010, 2030, 2050, 2070),
):
    x0, y0, x1, y1 = plot

    for yt in yticks:
        _, yy = map_xy(plot, xlim, ylim, [xlim[0]], [yt])
        yy = float(yy[0])

        draw.line([(x0, yy), (x1, yy)], fill=COL["grid"], width=2)
        draw.line([(x0 - 8, yy), (x0, yy)], fill=COL["axis"], width=2)
        text(draw, (x0 - 14, yy), f"{yt:g}", F_TICK, anchor="ra", fill=COL["text"])

    for xt in xticks:
        xx, _ = map_xy(plot, xlim, ylim, [xt], [ylim[0]])
        xx = float(xx[0])

        draw.line([(xx, y1), (xx, y1 + 8)], fill=COL["axis"], width=2)
        text(draw, (xx, y1 + 19), str(xt), F_TICK, anchor="ma", fill=COL["text"])

    if xlim[0] <= 2014.5 <= xlim[1]:
        xb, _ = map_xy(plot, xlim, ylim, [2014.5], [ylim[0]])
        draw.line([(float(xb[0]), y0), (float(xb[0]), y1)], fill="#A8A8A8", width=2)

    draw.line([(x0, y1), (x1, y1)], fill=COL["axis"], width=2)
    draw.line([(x0, y0), (x0, y1)], fill=COL["axis"], width=2)

    draw_rotated_text(
        img,
        (x0 - 84, (y0 + y1) / 2),
        ylabel,
        F_AXIS,
        fill=COL["text"],
        angle=90,
    )

    if xlabel:
        text(draw, ((x0 + x1) / 2, y1 + 58), xlabel, F_AXIS, anchor="ma", fill=COL["text"])


def shade_analysis_period(draw, plot, xlim, ylim):
    start = max(2045, xlim[0])
    end = min(2074, xlim[1])

    if end <= start:
        return

    x0, y0 = map_xy(plot, xlim, ylim, [start], [ylim[1]])
    x1, y1 = map_xy(plot, xlim, ylim, [end], [ylim[0]])

    draw.rectangle(
        [float(x0[0]), float(y0[0]), float(x1[0]), float(y1[0])],
        fill=COL["shade"],
    )


def draw_phase_labels(draw, plot, xlim, ylim):
    """Only used in panel a."""
    if xlim[0] <= 1998 <= xlim[1]:
        xh, yh = map_xy(plot, xlim, ylim, [1998], [ylim[1]])
        text(
            draw,
            (float(xh[0]), float(yh[0]) + 12),
            "Historical",
            F_TINY,
            fill=COL["muted"],
            anchor="ma",
        )


def draw_ssp3_label(draw, plot, xlim, ylim):
    """Label the shaded late-century analysis period."""
    if xlim[0] <= 2057 <= xlim[1]:
        xs, ys = map_xy(plot, xlim, ylim, [2057], [ylim[1]])
        text(
            draw,
            (float(xs[0]), float(ys[0]) + 12),
            "SSP3-7.0",
            F_TINY,
            fill=COL["muted"],
            anchor="ma",
        )


def place_labels(labels, min_gap=24):
    labels = sorted(labels, key=lambda item: item["y"])

    adjusted = []
    last = -10**9

    for item in labels:
        y = max(item["y"], last + min_gap)
        adjusted.append({**item, "label_y": y})
        last = y

    if adjusted:
        overflow = adjusted[-1]["label_y"] - adjusted[-1]["plot_bottom"]

        if overflow > 0:
            for item in adjusted:
                item["label_y"] -= overflow

    return adjusted


# =============================================================================
# Time-series helpers
# =============================================================================

def connected(ens, code, scenario, metric):
    h = ens[(ens.region_code == code) & (ens.scenario == "historical")].sort_values("year")
    f = ens[(ens.region_code == code) & (ens.scenario == scenario)].sort_values("year")

    x = np.r_[h["year"].to_numpy(int), f["year"].to_numpy(int)]
    y = np.r_[
        rolling(h[metric].to_numpy(float)),
        rolling(f[metric].to_numpy(float)),
    ]

    return x, y


def future_series(ens, code, scenario, metric):
    f = ens[(ens.region_code == code) & (ens.scenario == scenario)].sort_values("year")
    return f["year"].to_numpy(int), rolling(f[metric].to_numpy(float))


def draw_time_panel(
    img,
    draw,
    ens,
    box,
    letter,
    metric,
    ylabel,
    ylim,
    yticks,
    future_only=False,
    show_phase=False,
    show_xlabel=True,
    show_ssp3_label=True,
):
    panel_header(draw, box, letter)

    plot = (box[0] + 108, box[1] + 54, box[2] - 68, box[3] - 76)

    xlim = (2015, 2074) if future_only else (1985, 2074)
    xticks = (2020, 2040, 2060) if future_only else (1990, 2010, 2030, 2050, 2070)

    shade_analysis_period(draw, plot, xlim, ylim)

    draw_axes(
        img,
        draw,
        plot,
        xlim,
        ylim,
        yticks,
        ylabel,
        xlabel="Year" if show_xlabel else None,
        xticks=xticks,
    )

    if show_phase:
        draw_phase_labels(draw, plot, xlim, ylim)

    if show_ssp3_label:
        draw_ssp3_label(draw, plot, xlim, ylim)

    end_labels = []

    for code in DRAW_ORDER:
        color = rgb(COL[code])

        if future_only:
            x, y = future_series(ens, code, "ssp370", metric)
        else:
            x, y = connected(ens, code, "ssp370", metric)

        xx, yy = map_xy(plot, xlim, ylim, x, y)
        pts = list(zip(xx.astype(float), yy.astype(float)))

        draw.line(pts, fill=color, width=TRAJ_LINE_WIDTH, joint="curve")

        if len(yy) > 0:
            end_labels.append(
                {
                    "code": code,
                    "y": float(yy[-1]),
                    "color": COL[code],
                    "plot_top": plot[1],
                    "plot_bottom": plot[3],
                }
            )

    # Direct curve-end labels are intentionally omitted in this version.
    # The shared legend above panel a keeps panels a-c clean and avoids
    # label-to-line mismatches where the top regional curves converge.


def trim_uniform_outer_margin(img, pad=28):
    """Trim white outer whitespace and add a uniform white margin."""
    bg = Image.new(img.mode, img.size, "white")
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()

    if bbox is None:
        return img

    cropped = img.crop(bbox)
    out = Image.new(img.mode, (cropped.size[0] + 2 * pad, cropped.size[1] + 2 * pad), "white")
    out.paste(cropped, (pad, pad))
    return out


# =============================================================================
# Hatched bars for panel d
# =============================================================================

def draw_hatched_bar(img, draw, rect, color_hex, radius=5):
    """Draw a hatched bar clipped to a rounded rectangle."""
    x0, y0, x1, y1 = [int(v) for v in rect]

    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=radius,
        fill=lighten(color_hex, 0.64),
        outline=rgb(color_hex),
        width=1,
    )

    layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
    ld = ImageDraw.Draw(layer)

    h = y1 - y0
    step = 22

    for sx in range(x0 - h - 20, x1 + 20, step):
        ld.line(
            [(sx, y1 + 3), (sx + h + 3, y0 - 3)],
            fill=(*rgb(color_hex), 130),
            width=2,
        )

    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)

    md.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=radius,
        fill=255,
    )

    alpha = layer.getchannel("A")
    clipped_alpha = ImageChops.multiply(alpha, mask)
    layer.putalpha(clipped_alpha)

    img.paste(layer, (0, 0), layer)


def draw_transparent_hatched_key(img, draw, rect, outline="#6D7680", radius=4):
    """Draw a transparent hatched legend swatch clipped to a rounded rectangle."""
    x0, y0, x1, y1 = [int(v) for v in rect]

    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=radius,
        fill=None,
        outline=rgb(outline),
        width=2,
    )

    layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
    ld = ImageDraw.Draw(layer)

    h = y1 - y0
    for sx in range(x0 - h - 10, x1 + 10, 14):
        ld.line(
            [(sx, y1 + 2), (sx + h + 2, y0 - 2)],
            fill=(*rgb(outline), 150),
            width=2,
        )

    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=255)

    alpha = layer.getchannel("A")
    clipped_alpha = ImageChops.multiply(alpha, mask)
    layer.putalpha(clipped_alpha)
    img.paste(layer, (0, 0), layer)


def draw_summary_bars(img, draw, summary, box):
    panel_header(draw, box, "d")

    x0, y0, x1, y1 = box
    plot = (x0 + 108, y0 + 54, x1 - 80, y1 - 76)

    px0, py0, px1, py1 = plot

    s3 = summary[summary.scenario == "ssp370"].set_index("region_code")

    ordered = sorted(
        REGIONS,
        key=lambda c: float(s3.loc[c, "compound_share_percent"]),
        reverse=True,
    )

    x_max = 50.0

    for t in [0, 10, 20, 30, 40, 50]:
        xx = px0 + t / x_max * (px1 - px0)

        draw.line([(xx, py0), (xx, py1)], fill=COL["grid"], width=2)
        draw.line([(xx, py1), (xx, py1 + 8)], fill=COL["axis"], width=2)
        text(draw, (xx, py1 + 18), str(t), F_TICK, anchor="ma", fill=COL["text"])

    group_gap = 55
    bar_h = 18
    top = py0 + 35

    for j, code in enumerate(ordered):
        all_val = float(s3.loc[code, "compound_share_percent"])
        em_val = float(s3.loc[code, "emerging_share_percent"])

        yc = top + j * group_gap

        y_all = yc - bar_h / 2
        y_em = yc + bar_h / 2

        x_all = px0 + all_val / x_max * (px1 - px0)
        x_em = px0 + em_val / x_max * (px1 - px0)

        text(
            draw,
            (x0 + 36, yc),
            code,
            F_SMALL,
            fill=COL["text"],
            anchor="lm",
        )
        draw.line([(px0 - 8, yc), (px0, yc)], fill=COL["axis"], width=2)

        draw.rounded_rectangle(
            [px0, y_all - bar_h / 2, x_all, y_all + bar_h / 2],
            radius=4,
            fill=rgb(COL[code]),
        )

        draw_hatched_bar(
            img,
            draw,
            [px0, y_em - bar_h / 2, x_em, y_em + bar_h / 2],
            COL[code],
            radius=4,
        )

        text(
            draw,
            (x_all + 9, y_all),
            f"{all_val:.1f}",
            F_SMALL,
            fill=COL["text"],
            anchor="lm",
        )

        text(
            draw,
            (x_em + 9, y_em),
            f"{em_val:.1f}",
            F_SMALL,
            fill=COL["text"],
            anchor="lm",
        )

    # Redraw axes on top of the bars, following the Fig. 9 panel-d treatment.
    # This keeps the bars numerically anchored at zero but prevents them from
    # visually covering the axis/tick line.
    draw.line([(px0, py1), (px1, py1)], fill=COL["axis"], width=2)
    draw.line([(px0, py0), (px0, py1)], fill=COL["axis"], width=2)

    # Compact transparent style key in the lower-right corner of panel d.
    legend_x = px0 + 24.0 / x_max * (px1 - px0)
    legend_y1 = py1 - 72
    legend_y2 = py1 - 46

    draw.rounded_rectangle(
        [legend_x, legend_y1 - 9, legend_x + 42, legend_y1 + 9],
        radius=4,
        fill=None,
        outline=rgb(COL["muted"]),
        width=2,
    )

    text(
        draw,
        (legend_x + 52, legend_y1),
        "All compound",
        F_TINY,
        fill=COL["muted"],
        anchor="lm",
    )

    draw_transparent_hatched_key(
        img,
        draw,
        [legend_x, legend_y2 - 9, legend_x + 42, legend_y2 + 9],
        radius=4,
    )

    text(
        draw,
        (legend_x + 52, legend_y2),
        "Emerging compound",
        F_TINY,
        fill=COL["muted"],
        anchor="lm",
    )

    text(
        draw,
        ((px0 + px1) / 2, py1 + 58),
        "Regional contribution to global category (%)",
        F_AXIS,
        anchor="ma",
        fill=COL["text"],
    )


# =============================================================================
# Main plotting
# =============================================================================

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)

    ens = pd.read_csv(OUT_TABLE)
    summary = pd.read_csv(OUT_SUMMARY)

    img = Image.new("RGB", (1800, 1125), "white")
    draw = ImageDraw.Draw(img)

    boxes = {
        "a": (48, 38, 865, 520),
        "b": (878, 38, 1748, 520),
        "c": (48, 570, 865, 1052),
        "d": (878, 570, 1748, 1052),
    }

    draw_time_panel(
        img,
        draw,
        ens,
        boxes["a"],
        "a",
        "compound_area_million_km2",
        "Compound cropland (million km²)",
        (0, 1.20),
        [0, 0.3, 0.6, 0.9, 1.2],
        future_only=False,
        show_phase=True,
        show_xlabel=True,
    )
    draw_region_legend(draw, boxes["a"])
    # The upper row keeps year ticks but omits the axis label for a cleaner
    # two-row publication layout.

    draw_time_panel(
        img,
        draw,
        ens,
        boxes["b"],
        "b",
        "compound_share_percent",
        "Share of global compound area (%)",
        (0, 50),
        [0, 10, 20, 30, 40, 50],
        future_only=False,
        show_phase=False,
        show_xlabel=True,
        show_ssp3_label=False,
    )

    draw_time_panel(
        img,
        draw,
        ens,
        boxes["c"],
        "c",
        "emerging_share_percent",
        "Share of global emerging compound area (%)",
        (0, 50),
        [0, 10, 20, 30, 40, 50],
        future_only=True,
        show_phase=False,
        show_ssp3_label=False,
    )

    draw_summary_bars(img, draw, summary, boxes["d"])

    img = trim_uniform_outer_margin(img, pad=28)

    # Export a higher-pixel version to avoid the slightly soft appearance that
    # can occur when this raster figure is inserted into Word/PDF layouts.
    upscale = 2
    img = img.resize((img.width * upscale, img.height * upscale), Image.Resampling.LANCZOS)

    png = OUT.with_suffix(".png")
    pdf = OUT.with_suffix(".pdf")

    img.save(png, dpi=(600, 600))
    img.save(pdf, "PDF", resolution=600)

    target = CODE_DIR / Path(__file__).name

    if Path(__file__).resolve() != target.resolve():
        target.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    print(png)
    print(pdf)
    print(OUT_TABLE)
    print(OUT_SUMMARY)


if __name__ == "__main__":
    main()
