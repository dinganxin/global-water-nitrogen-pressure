# -*- coding: utf-8 -*-
"""Two-panel low-positive-Q sensitivity under the strict positive-Q framework."""

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
TABLE = ROOT / "tables" / "low_positive_Q_sensitivity_strict_positiveQ.csv"
FIG_DIR = ROOT / "figures" / "extended_data"
CODE_DIR = Path(__file__).resolve().parent
BASE = "ExtendedData_Fig04_low_positive_Q_sensitivity"

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
STRATEGIES = [
    "main",
    "floor_low_p01",
    "floor_low_p05",
    "exclude_low_p01",
    "exclude_low_p05",
]
STRATEGY_LABELS = [
    "Main",
    "Floor\np01",
    "Floor\np05",
    "Exclude\np01",
    "Exclude\np05",
]
STRATEGY_LABELS_HORIZONTAL = [
    "Main",
    "Floor p01",
    "Floor p05",
    "Exclude p01",
    "Exclude p05",
]


def style_axis(ax: plt.Axes, grid_axis: str) -> None:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#3B3B3B")
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_zorder(10)
    ax.grid(axis=grid_axis, color="#E5E7EB", linewidth=0.55, zorder=0)
    ax.tick_params(length=2.6, width=0.55)
    ax.set_axisbelow(True)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    target = CODE_DIR / Path(__file__).name
    if Path(__file__).resolve() != target.resolve():
        shutil.copy2(Path(__file__), target)

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "figure.dpi": 600,
            "axes.linewidth": 0.6,
        }
    )

    df = pd.read_csv(TABLE)
    df["strategy"] = pd.Categorical(df["strategy"], STRATEGIES, ordered=True)
    df["period_key"] = pd.Categorical(df["period_key"], PERIODS, ordered=True)
    df = df.sort_values(["strategy", "period_key"])

    fig = plt.figure(figsize=(6.65, 3.35), dpi=600, facecolor="white")
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.43, 1.00],
        left=0.078,
        right=0.980,
        bottom=0.245,
        top=0.900,
        wspace=0.32,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    x = np.arange(len(STRATEGIES))
    width = 0.22
    for k, period in enumerate(PERIODS):
        vals = (
            df[df["period_key"] == period]
            .set_index("strategy")
            .loc[STRATEGIES, "compound_high_pressure_area_million_km2"]
            .to_numpy(float)
        )
        ax_a.bar(
            x + (k - 1) * width,
            vals,
            width=width,
            color=PERIOD_COLORS[period],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
            label=PERIOD_LABELS[period],
        )
    ax_a.set_title("a  Low positive-Q treatment", loc="left", fontweight="bold", pad=4)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(STRATEGY_LABELS, fontsize=6.1, linespacing=0.92)
    ax_a.set_ylabel("Strict compound cropland (million km$^2$)")
    ax_a.set_ylim(0, 2.05)
    ax_a.set_yticks(np.arange(0, 2.01, 0.5))
    ax_a.legend(
        frameon=False,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.985),
        handlelength=1.20,
        columnspacing=0.95,
        fontsize=6.5,
    )
    style_axis(ax_a, "y")

    differences = []
    for strategy in STRATEGIES:
        sub = df[df["strategy"] == strategy].set_index("period_key")
        differences.append(
            float(
                sub.loc["ssp370", "compound_high_pressure_area_million_km2"]
                - sub.loc["ssp126", "compound_high_pressure_area_million_km2"]
            )
        )
    differences = np.asarray(differences)
    y = np.arange(len(STRATEGIES))
    ax_b.barh(
        y,
        differences,
        color="#C84D5A",
        edgecolor="white",
        linewidth=0.45,
        height=0.56,
        zorder=3,
    )
    for yi, value in zip(y, differences):
        ax_b.text(
            value + 0.012,
            yi,
            f"+{value:.2f}",
            ha="left",
            va="center",
            fontsize=6.6,
            color="#17202A",
        )
    ax_b.set_title("b  Scenario difference", loc="left", fontweight="bold", pad=4)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(STRATEGY_LABELS_HORIZONTAL, fontsize=6.25)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 0.62)
    ax_b.set_xticks(np.arange(0, 0.61, 0.2))
    ax_b.set_xlabel("SSP3-7.0 - SSP1-2.6 (million km²)")
    style_axis(ax_b, "x")

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            FIG_DIR / f"{BASE}.{ext}",
            bbox_inches="tight",
            pad_inches=0.035,
            facecolor="white",
        )
    plt.close(fig)
    print(FIG_DIR / f"{BASE}.png")


if __name__ == "__main__":
    main()
