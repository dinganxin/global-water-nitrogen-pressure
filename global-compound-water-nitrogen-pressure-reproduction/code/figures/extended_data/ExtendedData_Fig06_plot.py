# -*- coding: utf-8 -*-
"""Recompute HydroBASINS level 3-6 sensitivity under the strict Q>0 domain."""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data" / "extended_data" / "HydroBASINS_levels3_to_6_sensitivity_effective_land_area.npz"
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures" / "extended_data"
CODE_DIR = Path(__file__).resolve().parent
TABLE = TABLE_DIR / "HydroBASINS_levels3_to_6_sensitivity_strict_positiveQ.csv"
BASE = "ExtendedData_Fig06_HydroBASINS_scale_sensitivity"

LEVELS = [3, 4, 5, 6]
PERIODS = ["historical", "ssp126", "ssp370"]
PERIOD_LABELS = {
    "historical": "Historical",
    "ssp126": "SSP1-2.6",
    "ssp370": "SSP3-7.0",
}
PERIOD_COLORS = {
    "historical": "#7A8492",
    "ssp126": "#4F8FBD",
    "ssp370": "#C84D5A",
}


def rebuild_table() -> pd.DataFrame:
    z = np.load(SOURCE)
    rows: list[dict[str, float | int | str]] = []
    for level in LEVELS:
        for period in PERIODS:
            npi = np.asarray(z[f"NPI_level{level}_{period}_basin"], dtype=float)
            ipi = np.asarray(z[f"IPI_level{level}_{period}_basin"], dtype=float)
            limiting = np.asarray(
                z[f"nonpositive_Q_level{level}_{period}_basin"],
                dtype=bool,
            )
            area = np.asarray(
                z[f"basin_crop_area_m2_level{level}_{period}"],
                dtype=float,
            ) / 1.0e12
            exposed = np.isfinite(area) & (area > 0)
            computable = exposed & (~limiting) & np.isfinite(npi) & np.isfinite(ipi)

            n_high = computable & (npi >= 10.0)
            i_high = computable & (ipi >= 0.4)
            compound = n_high & i_high
            n_only = n_high & (~i_high)
            i_only = (~n_high) & i_high
            low = computable & (~n_high) & (~i_high)
            limiting_exposed = exposed & limiting

            strict_sum = float(
                np.sum(area[low])
                + np.sum(area[n_only])
                + np.sum(area[i_only])
                + np.sum(area[compound])
                + np.sum(area[limiting_exposed])
            )
            domain_sum = float(np.sum(area[exposed]))
            if not np.isclose(strict_sum, domain_sum, rtol=0.0, atol=1.0e-8):
                raise ValueError(
                    f"Level {level}, {period}: strict partition does not "
                    f"conserve area ({strict_sum} != {domain_sum})."
                )

            rows.append(
                {
                    "hydrobasins_level": level,
                    "period_key": period,
                    "period": PERIOD_LABELS[period],
                    "total_crop_domain_area_million_km2": domain_sum,
                    "positive_Q_computable_area_million_km2": float(
                        np.sum(area[computable])
                    ),
                    "nonpositive_Q_limiting_state_area_million_km2": float(
                        np.sum(area[limiting_exposed])
                    ),
                    "low_pressure_area_million_km2": float(np.sum(area[low])),
                    "nitrate_pressure_only_area_million_km2": float(
                        np.sum(area[n_only])
                    ),
                    "irrigation_pressure_only_area_million_km2": float(
                        np.sum(area[i_only])
                    ),
                    "compound_high_pressure_area_million_km2": float(
                        np.sum(area[compound])
                    ),
                    "nitrate_pressure_area_million_km2": float(
                        np.sum(area[n_only | compound])
                    ),
                    "irrigation_pressure_area_million_km2": float(
                        np.sum(area[i_only | compound])
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "figure.dpi": 600,
            "axes.linewidth": 0.6,
        }
    )

    fig = plt.figure(figsize=(6.15, 4.85), dpi=600, facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.980,
        bottom=0.115,
        top=0.945,
        wspace=0.22,
        hspace=0.42,
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    metrics = [
        (
            "compound_high_pressure_area_million_km2",
            "a  Compound pressure",
        ),
        (
            "nitrate_pressure_area_million_km2",
            "b  Nitrate-pressure exceedance",
        ),
        (
            "irrigation_pressure_area_million_km2",
            "c  Irrigation-pressure exceedance",
        ),
        (
            "nonpositive_Q_limiting_state_area_million_km2",
            "d  $Q \\leq 0$ state",
        ),
    ]

    handles = []
    labels = []
    all_max = float(
        df[
            [
                "compound_high_pressure_area_million_km2",
                "nitrate_pressure_area_million_km2",
                "irrigation_pressure_area_million_km2",
                "nonpositive_Q_limiting_state_area_million_km2",
            ]
        ].to_numpy(float).max()
    )
    y_max = np.ceil((all_max * 1.08) / 0.5) * 0.5

    for ax, (metric, title) in zip(axes, metrics):
        for period in PERIODS:
            sub = (
                df[df["period_key"] == period]
                .set_index("hydrobasins_level")
                .loc[LEVELS]
            )
            line = ax.plot(
                LEVELS,
                sub[metric].to_numpy(float),
                marker="o",
                markersize=3.1,
                linewidth=1.30,
                color=PERIOD_COLORS[period],
                label=PERIOD_LABELS[period],
                solid_capstyle="round",
            )[0]
            if ax is axes[0]:
                handles.append(line)
                labels.append(PERIOD_LABELS[period])
        ax.axvline(
            5,
            color="#CBD5DF",
            linewidth=0.7,
            linestyle=(0, (2.2, 2.2)),
            zorder=0,
        )
        ax.set_title(title, loc="left", fontweight="bold", pad=4)
        ax.set_xlim(2.85, 6.15)
        ax.set_xticks(LEVELS)
        ax.set_xlabel("HydroBASINS level")
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.arange(0, y_max + 0.01, 1.0))
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.55)
        ax.tick_params(length=2.6, width=0.55)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#3B3B3B")
            ax.spines[spine].set_linewidth(0.6)
    for ax in axes:
        ax.set_ylabel("Cropland area (million km$^2$)")
    axes[0].legend(
        handles,
        labels,
        frameon=False,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.99),
        handlelength=1.45,
        labelspacing=0.32,
    )

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            FIG_DIR / f"{BASE}.{ext}",
            bbox_inches="tight",
            pad_inches=0.035,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    df = rebuild_table()
    df.to_csv(TABLE, index=False, encoding="utf-8-sig")
    plot(df)
    target = CODE_DIR / Path(__file__).name
    if Path(__file__).resolve() != target.resolve():
        shutil.copy2(Path(__file__), target)
    print(TABLE)
    print(FIG_DIR / f"{BASE}.png")


if __name__ == "__main__":
    main()
