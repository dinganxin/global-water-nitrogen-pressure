# -*- coding: utf-8 -*-
"""Draw Extended Data Fig. 3 from the threshold-sensitivity table."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "tables"
FIGS = ROOT / "figures" / "extended_data"
THRESHOLD_TABLE = TABLES / "threshold_sensitivity_strict_positiveQ.csv"

BASE = "ExtendedData_Fig03_threshold_robustness"
POLISHED = BASE


def soft_cmap(name: str, colors: list[str]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, colors, N=256)


def text_color(value: float, vmax: float) -> str:
    return "white" if np.isfinite(value) and value / vmax > 0.72 else "#17202A"


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.8,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "axes.linewidth": 0.6,
        }
    )

    df = pd.read_csv(THRESHOLD_TABLE)
    s1 = df[df["period_key"] == "ssp126"].copy()
    s3 = df[df["period_key"] == "ssp370"].copy()
    diff = s3.merge(
        s1,
        on=["nitrate_threshold_mgN_L", "ipi_threshold"],
        suffixes=("_ssp370", "_ssp126"),
    )
    diff["ssp370_minus_ssp126_million_km2"] = (
        diff["compound_high_pressure_area_million_km2_ssp370"]
        - diff["compound_high_pressure_area_million_km2_ssp126"]
    )
    nvals = sorted(df["nitrate_threshold_mgN_L"].unique())
    wvals = sorted(df["ipi_threshold"].unique())

    abs_cmap = soft_cmap("compound_bright_coral", ["#FFF7EC", "#FDE0BD", "#F7A777", "#E76D62", "#B63A68"])
    inc_cmap = soft_cmap("additional_bright_coral", ["#FFF4E6", "#F9C98F", "#EF845F", "#C8466D"])

    fig = plt.figure(figsize=(6.65, 5.35), dpi=600, facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.965,
        bottom=0.085,
        top=0.955,
        wspace=0.18,
        hspace=0.25,
    )
    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ]
    )

    panels = [
        ("historical", "a", "Historical", axes[0, 0]),
        ("ssp126", "b", "SSP1-2.6", axes[0, 1]),
        ("ssp370", "c", "SSP3-7.0", axes[1, 0]),
    ]

    cbar_specs = []
    abs_im = None
    for period, letter, title, ax in panels:
        sub = df[df.period_key == period]
        mat = sub.pivot(
            index="ipi_threshold",
            columns="nitrate_threshold_mgN_L",
            values="compound_high_pressure_area_million_km2",
        ).loc[wvals, nvals]
        abs_im = ax.imshow(mat.values, origin="lower", cmap=abs_cmap, vmin=0, vmax=3.5, aspect=0.80)
        ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=4)
        ax.set_xticks(np.arange(len(nvals)))
        ax.set_xticklabels([f"{v:g}" for v in nvals])
        ax.set_yticks(np.arange(len(wvals)))
        ax.set_yticklabels([f"{v:g}" for v in wvals])
        ax.set_xlabel("NPI threshold (mg N L$^{-1}$)")
        ax.set_ylabel("IPI threshold" if letter in {"a", "c"} else "")
        for i in range(len(wvals)):
            for j in range(len(nvals)):
                val = float(mat.values[i, j])
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.4, color=text_color(val, 3.5))
        # Highlight default threshold cell: NPI = 10 mg N L-1, IPI = 0.4.
        ax.add_patch(Rectangle((0.5, 0.5), 1.0, 1.0, fill=False, edgecolor="#27323A", lw=1.0))
        ax.tick_params(length=2.6, width=0.55)
        for spine in ax.spines.values():
            spine.set_color("#3B3B3B")
            spine.set_linewidth(0.6)
        cbar_specs.append(
            {
                "ax": ax,
                "im": abs_im,
                "ticks": [0, 1, 2, 3],
                "label": "Strict compound cropland (million km$^2$)" if letter == "b" else "",
                "extend": "neither",
            }
        )

    inc_mat = diff.pivot(
        index="ipi_threshold",
        columns="nitrate_threshold_mgN_L",
        values="ssp370_minus_ssp126_million_km2",
    ).loc[wvals, nvals]
    axd = axes[1, 1]
    inc_im = axd.imshow(inc_mat.values, origin="lower", cmap=inc_cmap, vmin=0.2, vmax=0.7, aspect=0.80)
    axd.set_title("d  SSP3-7.0 - SSP1-2.6", loc="left", fontweight="bold", pad=4)
    axd.set_xticks(np.arange(len(nvals)))
    axd.set_xticklabels([f"{v:g}" for v in nvals])
    axd.set_yticks(np.arange(len(wvals)))
    axd.set_yticklabels([f"{v:g}" for v in wvals])
    axd.set_xlabel("NPI threshold (mg N L$^{-1}$)")
    axd.set_ylabel("")
    for i in range(len(wvals)):
        for j in range(len(nvals)):
            val = float(inc_mat.values[i, j])
            axd.text(j, i, f"+{val:.2f}", ha="center", va="center", fontsize=7.4, color="white" if val >= 0.62 else "#17202A")
    axd.add_patch(Rectangle((0.5, 0.5), 1.0, 1.0, fill=False, edgecolor="#27323A", lw=1.0))
    axd.tick_params(length=2.6, width=0.55)
    for spine in axd.spines.values():
        spine.set_color("#3B3B3B")
        spine.set_linewidth(0.6)
    cbar_specs.append(
        {
            "ax": axd,
            "im": inc_im,
            "ticks": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "label": "Additional strict compound cropland (million km$^2$)",
            "extend": "neither",
        }
    )

    # Add colorbars after the heatmap aspect ratio has been resolved, so each
    # colorbar exactly matches its parent heatmap height.
    fig.canvas.draw()
    for spec in cbar_specs:
        bbox = spec["ax"].get_position()
        cax = fig.add_axes([bbox.x1 + 0.007, bbox.y0, 0.0095, bbox.height])
        cb = fig.colorbar(spec["im"], cax=cax, extend=spec["extend"])
        cb.set_ticks(spec["ticks"])
        cb.ax.tick_params(labelsize=6.5, length=2.0, width=0.5)
        if spec["label"]:
            cb.set_label(spec["label"], fontsize=7.2)

    for base in [BASE]:
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(FIGS / f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(FIGS / f"{POLISHED}.png")


if __name__ == "__main__":
    main()
