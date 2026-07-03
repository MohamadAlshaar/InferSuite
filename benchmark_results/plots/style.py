"""Shared publication style for thesis figures.

Vector PDF (for LaTeX) + PNG (preview) output, serif fonts, colourblind-safe
palette (Okabe–Ito), clean spines. Import and call set_style() once.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe–Ito colourblind-safe palette.
C = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "grey":   "#999999",
}
CYCLE = [C["blue"], C["orange"], C["green"], C["red"], C["purple"], C["sky"], C["grey"]]


def set_style() -> None:
    mpl.rcParams.update({
        "figure.figsize": (5.4, 3.3),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,           # editable text in the PDF (no type-3)
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.9,
        "lines.markersize": 5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def save(fig, path_stem: str) -> None:
    """Save a figure as both .pdf (thesis) and .png (preview)."""
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png")
    plt.close(fig)
    print(f"  wrote {path_stem}.pdf / .png")
