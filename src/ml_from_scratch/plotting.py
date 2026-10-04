"""Matplotlib helpers: the dark indigo / violet / cyan "Universe" style.

Requires the optional ``plots`` extra (``pip install ml-from-scratch[plots]``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

BG = "#0a0a18"  # deep space
PANEL = "#12122b"  # axes background (indigo-tinted)
GRID = "#2a2a55"
TEXT = "#eceefb"
MUTED = "#b4b9d6"
VIOLET = "#8b6dff"
CYAN = "#22d3ee"
PINK = "#ec4899"
AMBER = "#fbbf24"
CYCLE = [VIOLET, CYAN, PINK, AMBER, "#34d399", "#60a5fa"]

#: Diverging colormap for decision regions: cyan <-> deep indigo <-> violet-pink.
UNIVERSE_CMAP = LinearSegmentedColormap.from_list("universe", [CYAN, "#1b1750", PINK])
#: Sequential colormap indigo -> violet -> cyan.
UNIVERSE_SEQ = LinearSegmentedColormap.from_list("universe_seq", ["#14123a", VIOLET, CYAN])


def apply_universe_style() -> None:
    """Install the Universe colours in matplotlib's global ``rcParams``."""
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 13,
            "axes.titleweight": "normal",
            "axes.grid": True,
            "axes.prop_cycle": cycler(color=CYCLE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.alpha": 0.45,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "legend.facecolor": BG,
            "legend.edgecolor": GRID,
            "legend.labelcolor": TEXT,
            "lines.linewidth": 2.2,
            "font.size": 11,
            "figure.dpi": 100,
            "savefig.dpi": 160,
        }
    )


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Save ``fig`` to ``path`` (creating parent directories) and close it."""
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out


def decision_grid(
    X: np.ndarray, resolution: int = 300, margin: float = 0.4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mesh covering the 2-D data ``X``: returns ``xx, yy`` and the flattened grid points."""
    x0, x1 = X[:, 0].min() - margin, X[:, 0].max() + margin
    y0, y1 = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x0, x1, resolution), np.linspace(y0, y1, resolution))
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])
