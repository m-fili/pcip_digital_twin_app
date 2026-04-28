"""Shared plotting helpers — paper-style figures, dark theme.

Call `setup_internal_page` (from theme.py) before importing this; that's
where the matplotlib rcParams are set for the dark palette.
"""
import io
import matplotlib.pyplot as plt

# Re-export the brand palette so pages can use COLORS without importing theme
from theme import (  # noqa: F401
    PLOT_PALETTE as COLORS,
    CYAN, VIOLET, AMBER, ROSE, GREEN, CORAL,
    INK, INK_DIM, INK_MUTE, HAIR, HAIR_SOFT,
    NAVY_000, NAVY_100, NAVY_200,
)

plt.rcParams["figure.figsize"] = (8, 5)


def prettify(ax):
    """Open spines, light grid, dark-theme-friendly."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle="--", linewidth=0.4, alpha=0.25)
    ax.grid(which="major", linestyle="-", linewidth=0.6, alpha=0.4)


def add_subplot_label(ax, label, size=16):
    ax.text(
        -0.1, 1.05, label, transform=ax.transAxes,
        fontsize=size, fontweight="bold", va="top", ha="right",
        color=INK,
    )


def fig_to_png_bytes(fig, dpi=200):
    """Save figure to PNG bytes, preserving dark facecolor."""
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi, bbox_inches="tight",
        facecolor=fig.get_facecolor(), edgecolor="none",
    )
    buf.seek(0)
    return buf.getvalue()
