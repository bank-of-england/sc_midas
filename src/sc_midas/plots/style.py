"""Shared style helpers for forecast combination visualisations."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = []

# ---------------------------------------------------------------------------
# 2022 brand palette (boe_brand_main + boe_brand_secondary + boe_identity)
# ---------------------------------------------------------------------------

# Primary brand colours
_BOE_DARK_BLUE = "#12273F"  # boe_brand_main / dark blue
_BOE_AQUA = "#3CD7D9"  # boe_brand_main / identity
_BOE_STONE = "#C4C9CF"  # boe_brand_main / stone

# Secondary brand colours
_BOE_ORANGE = "#FF7300"  # boe_brand_secondary / identity
_BOE_PURPLE = "#9E71FE"  # boe_brand_secondary / identity
_BOE_GOLD = "#D4AF37"  # boe_brand_secondary / identity

# Additional identity colours
_BOE_GREEN = "#A5D700"
_BOE_PINK = "#FF50C8"
_BOE_AZURE = "#5297FF"
_BOE_YELLOW = "#FFD200"
_BOE_PEACH = "#FF9173"
_BOE_RED = "#FD015B"

# Categorical palette — ordered for visual distinctiveness
_BOE_COLOURS = [
    _BOE_DARK_BLUE,  # 1 — dark blue    (primary)
    _BOE_AQUA,  # 2 — aqua         (primary)
    _BOE_ORANGE,  # 3 — orange       (secondary)
    _BOE_PURPLE,  # 4 — purple       (secondary)
    _BOE_GOLD,  # 5 — gold         (secondary)
    _BOE_AZURE,  # 6 — azure        (identity)
    _BOE_GREEN,  # 7 — green        (identity)
    _BOE_PINK,  # 8 — pink         (identity)
    _BOE_PEACH,  # 9 — peach        (identity)
    _BOE_RED,  # 10— red          (identity)
]

# Sequential colormap for heatmaps — white → aqua (boe_shades_aqua)
_BOE_CMAP_SEQ = mpl.colors.LinearSegmentedColormap.from_list(
    "boe_aqua",
    [
        "#ffffff",  # white base
        "#CEF5F5",  # boe_extra_light_aqua
        "#A7EDEE",  # boe_light_aqua
        "#77E3E4",  # boe_mid_aqua
        "#3CD7D9",  # boe_aqua
        "#34BCC1",  # boe_dark_aqua
    ],
)


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------


def _thin_xticklabels(ax: plt.Axes, max_labels: int = 12) -> None:
    """Hide intermediate x-tick labels when there are too many to display.

    Parameters
    ----------
    ax : plt.Axes
        Axes whose tick labels are modified.
    max_labels : int
        Approximate maximum number of visible labels (default 12).
    """
    ticks = ax.get_xticklabels()
    n = len(ticks)
    if n <= max_labels:
        return
    step = int(np.ceil(n / max_labels))
    for i, label in enumerate(ticks):
        if i % step != 0:
            label.set_visible(False)


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------


def _apply_boe_style(fig: plt.Figure, axes_2d) -> None:
    """Apply the shared white-background style to a figure and its axes."""
    fig.patch.set_facecolor("white")

    axes = axes_2d.flat if hasattr(axes_2d, "flat") else [axes_2d]
    for ax in axes:
        if not ax.get_visible():
            continue
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        ax.tick_params(colors="#444444")
        ax.xaxis.label.set_color("#444444")
        ax.yaxis.label.set_color("#444444")
        ax.title.set_color("#222222")
        ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
