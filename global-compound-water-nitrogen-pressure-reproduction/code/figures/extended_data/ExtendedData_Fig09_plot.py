#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Polished Extended Data figure for trajectories and driver sensitivities."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "figures" / "extended_data"
TABLE_DIR = ROOT / "tables" / "extended_data"
AUDIT_DIR = ROOT / "audit"
PRODUCT = ROOT / "data" / "NPI_IPI_QIRR_strict_positiveQ_period_products_01pct.npz"


def rolling(values: np.ndarray, window: int = 5) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rolling(
        window=window, center=True, min_periods=1
    ).mean().to_numpy()

TEXT = "#22282D"
MUTED = "#69757E"
GRID = "#E5EAEE"
ZERO = "#3B4146"
WINDOW = "#EFF1F3"

SCENARIOS = (
    ("historical", "Historical", "#7D8794"),
    ("ssp126", "SSP1-2.6", "#4C91BF"),
    ("ssp370", "SSP3-7.0", "#C94F59"),
)

DRIVER_LABELS = ("Cropland", "N", "I", "Q")
DRIVER_KEYS = (
    "Cropland exposure",
    "Nitrate-N load",
    "Irrigation withdrawal",
    "Water availability",
)
DRIVER_COLORS = ("#7E9D72", "#CB6D5D", "#4F9999", "#6F82B3")


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.2,
            "axes.titlesize": 9.3,
            "axes.labelsize": 8.7,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def panel_heading(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        -0.025,
        1.045,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.6,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        0.075,
        1.045,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.3,
        color=TEXT,
    )


def style_axis(ax: plt.Axes, grid_axis: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(ZERO)
    ax.spines["bottom"].set_color(ZERO)
    ax.tick_params(axis="both", width=0.7, length=3.0, color=ZERO)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)


def plot_trajectory(
    ax: plt.Axes,
    annual: pd.DataFrame,
    metric: str,
    letter: str,
    title: str,
    show_period_labels: bool = False,
) -> None:
    ax.axvspan(2045, 2074, color=WINDOW, alpha=0.70, zorder=-3)
    for scenario, _, color in SCENARIOS:
        subset = annual.loc[annual["scenario"] == scenario]
        pivot = subset.pivot(
            index="year", columns="member", values=metric
        ).sort_index()
        years = pivot.index.to_numpy(dtype=int)
        member_values = pivot.to_numpy(dtype=float).T
        smooth_members = np.vstack(
            [rolling(member_values[i], 5) for i in range(member_values.shape[0])]
        )
        mean_values = np.mean(smooth_members, axis=0)
        lower = np.min(smooth_members, axis=0)
        upper = np.max(smooth_members, axis=0)
        ax.fill_between(
            years,
            lower,
            upper,
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        ax.plot(years, mean_values, color=color, linewidth=1.8, zorder=3)

    ax.axvline(2014.5, color="#929A9F", linewidth=0.8, zorder=2)
    panel_heading(ax, letter, title)
    ax.set_xlim(1985, 2074)
    ax.set_xticks([1990, 2010, 2030, 2050, 2070])
    ax.set_ylim(0.0, 2.5)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel("Year")
    ax.set_ylabel("Cropland area (million km²)")
    style_axis(ax, grid_axis="y")
    if show_period_labels:
        ax.text(
            2002,
            0.75,
            "Historical",
            ha="center",
            va="center",
            fontsize=7.1,
            color=MUTED,
        )
        ax.text(
            2059.5,
            0.75,
            "2045–2074",
            ha="center",
            va="center",
            fontsize=7.1,
            color=MUTED,
        )


def signed_label(value: float) -> str:
    if value >= 0:
        return f"+{value:.2f}"
    return f"−{abs(value):.2f}"


def plot_driver_panel(
    ax: plt.Axes,
    ablation: pd.DataFrame,
    metric: str,
    letter: str,
    title: str,
    xlim: tuple[float, float],
    xticks: list[float],
) -> None:
    values = np.array(
        [
            float(
                ablation.loc[
                    ablation["configuration"] == key, metric
                ].iloc[0]
            )
            for key in DRIVER_KEYS
        ],
        dtype=float,
    )
    y = np.arange(len(DRIVER_LABELS))
    bars = ax.barh(
        y,
        values,
        color=DRIVER_COLORS,
        height=0.54,
        edgecolor="none",
        zorder=3,
    )
    ax.axvline(0.0, color=ZERO, linewidth=0.9, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(DRIVER_LABELS)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel("Area difference (million km²)")
    panel_heading(ax, letter, title)
    style_axis(ax, grid_axis="x")

    label_offset = 0.012
    for bar, value in zip(bars, values):
        x = value + (label_offset if value >= 0 else -label_offset)
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            signed_label(value),
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=7.4,
            color=TEXT,
        )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    annual = pd.read_csv(
        TABLE_DIR / "ExtendedData_Fig09_global_annual_compound_and_nonpositiveQ.csv"
    )
    ablation = pd.read_csv(
        TABLE_DIR / "ExtendedData_Fig09_component_substitution_strict_positiveQ.csv"
    )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    annual_path = (
        TABLE_DIR
        / "ExtendedData_Fig09_global_annual_compound_and_nonpositiveQ.csv"
    )
    ablation_path = (
        TABLE_DIR
        / "ExtendedData_Fig09_component_substitution_strict_positiveQ.csv"
    )
    if not annual_path.exists():
        annual.to_csv(annual_path, index=False, encoding="utf-8-sig")
    if not ablation_path.exists():
        ablation.to_csv(ablation_path, index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.2, 5.0), dpi=600)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.095,
        right=0.985,
        bottom=0.095,
        top=0.92,
        wspace=0.32,
        hspace=0.42,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_trajectory(
        ax_a,
        annual,
        "strict_compound_million_km2",
        "a",
        "Global compound-pressure cropland",
        show_period_labels=True,
    )
    plot_trajectory(
        ax_b,
        annual,
        "nonpositive_Q_million_km2",
        "b",
        "Global cropland with Q ≤ 0",
    )
    plot_driver_panel(
        ax_c,
        ablation,
        "full_minus_configuration_total_million_km2",
        "c",
        "Total compound-pressure cropland",
        xlim=(-0.6, 0.6),
        xticks=[-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6],
    )
    plot_driver_panel(
        ax_d,
        ablation,
        "full_minus_configuration_emerging_million_km2",
        "d",
        "Emerging compound-pressure cropland",
        xlim=(-0.5, 0.5),
        xticks=[-0.5, -0.25, 0.0, 0.25, 0.5],
    )

    handles = [
        Line2D([0], [0], color=color, linewidth=1.8, label=label)
        for _, label, color in SCENARIOS
    ]
    ax_a.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        ncol=3,
        frameon=False,
        columnspacing=0.85,
        handlelength=1.55,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    output_stem = (
        FIG_DIR
        / "ExtendedData_Fig09_temporal_and_driver_sensitivity"
    )
    save_figure(fig, output_stem)

    summary = {
        "figure_stem": str(output_stem),
        "annual_table": str(annual_path),
        "ablation_table": str(ablation_path),
        "trajectory_axes": {
            "xlim": [1985, 2074],
            "ylim_million_km2": [0.0, 2.5],
        },
        "driver_axes": {
            "panel_c_xlim_million_km2": [-0.6, 0.6],
            "panel_d_xlim_million_km2": [-0.5, 0.5],
            "drivers": list(DRIVER_LABELS),
        },
        "driver_values": ablation.to_dict(orient="records"),
        "interpretation": (
            "One-at-a-time nonlinear sensitivity; neither additive nor causal."
        ),
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = (
        AUDIT_DIR
        / "ExtendedData_Fig09_temporal_and_driver_sensitivity_labels_refined_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
