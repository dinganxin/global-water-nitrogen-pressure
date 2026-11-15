"""Small plotting helpers for the final ED07 maps."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MODEL_NPZ = ROOT / "data/NPI_IPI_QIRR_strict_positiveQ_period_products_01pct.npz"
LINE_CACHE = ROOT / "data/figure_inputs/natural_earth_110m_line_cache_numeric.npz"
N_MEMBERS = 3
N_YEARS = 30


def percentile_rank(values, mask):
    result = np.full(values.shape, np.nan, dtype=float)
    ok = mask & np.isfinite(values)
    if np.any(ok):
        result[ok] = pd.Series(values[ok]).rank(method="average", pct=True).to_numpy()
    return result


def load_line_cache():
    if not LINE_CACHE.exists():
        return None
    z = np.load(LINE_CACHE, allow_pickle=False)
    return {key: np.asarray(z[key]) for key in z.files}


def draw_segments(ax, points, ranges, color, linewidth, zorder):
    for start, end in ranges:
        segment = points[int(start):int(end)]
        if segment.shape[0] >= 2:
            ax.plot(segment[:, 0], segment[:, 1], color=color,
                    linewidth=linewidth, zorder=zorder)


def basin_to_grid(values, code_grid, lon):
    grid = np.full(code_grid.shape, np.nan, dtype=float)
    ok = np.isfinite(code_grid) & (code_grid >= 0)
    rr, cc = np.where(ok)
    idx = code_grid[rr, cc].astype(np.int64)
    good = idx < len(values)
    grid[rr[good], cc[good]] = values[idx[good]]
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon180)
    return lon180[order], grid[:, order]
